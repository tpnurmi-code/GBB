"""End-to-end smoke test for the public synthetic-data workflow."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from gbb.config import config
from gbb.data.dataset import NiftiLaminarDataset
from gbb.data.files import get_subject_files
from gbb.models.factory import build_model
from gbb.synthetic.config import SyntheticFMRIConfig
from gbb.synthetic.generator import MechanisticSyntheticFMRI


@pytest.mark.integration
def test_synthetic_nifti_to_h1_backward(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate synthetic fMRI, load one batch, and backpropagate through H1."""

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Generate a complete temporary synthetic dataset.
    # ------------------------------------------------------------------
    output_dir = tmp_path / "synthetic_gbb_quick"

    synthetic_config = SyntheticFMRIConfig.quick(output_dir=output_dir)
    synthetic_config.overwrite = True

    result = MechanisticSyntheticFMRI(synthetic_config).generate_dataset()
    data_dir = result.output_dir

    assert data_dir.exists()
    assert (data_dir / "group_roi_mask.nii").exists()
    assert (data_dir / "cortical_columns_7T.nii").exists()
    assert len(result.run_files) == (
        synthetic_config.num_subjects * synthetic_config.num_runs
    )

    # ------------------------------------------------------------------
    # 2. Configure the production loader for the generated dataset.
    #
    # monkeypatch automatically restores these values after the test.
    # ------------------------------------------------------------------
    window_size = 12
    prediction_horizon = 3

    monkeypatch.setattr(config, "TR", synthetic_config.tr)
    monkeypatch.setattr(
        config,
        "RAW_SAMPLING_RATE",
        synthetic_config.raw_stimulus_sampling_rate,
    )
    monkeypatch.setattr(config, "WINDOW_SIZE", window_size)
    monkeypatch.setattr(config, "PREDICTION_HORIZON", prediction_horizon)

    monkeypatch.setattr(config, "STIMULUS_MODE", "DENSE")
    monkeypatch.setattr(config, "DENSE_STIMULUS_EXT", "_stim.mat")
    monkeypatch.setattr(config, "STIMULUS_INPUT_CHANNELS", 1)
    monkeypatch.setattr(config, "CONVOLVE_STIMULUS", False)
    monkeypatch.setattr(
        config,
        "STIMULUS_INJECTION_MODE",
        "COORDS_REGION_INTERSECTION",
    )
    monkeypatch.setattr(config, "SENSORY_REGIONS", ["S1_Postcentral"])

    monkeypatch.setattr(
        config,
        "MASK_FILE",
        str(data_dir / "group_roi_mask.nii"),
    )
    monkeypatch.setattr(
        config,
        "COLUMNAR_MASK_FILE",
        str(data_dir / "cortical_columns_7T.nii"),
    )

    # Keep the CI model small while preserving all major model stages.
    monkeypatch.setattr(config, "CFC_BACKBONE_UNITS", 16)
    monkeypatch.setattr(config, "CNN_LAYERS", 1)
    monkeypatch.setattr(config, "KAN_LAYERS", 1)
    monkeypatch.setattr(config, "KAN_BASIS_FUNCTIONS", 3)
    monkeypatch.setattr(config, "DROPOUT", 0.0)
    monkeypatch.setattr(config, "FEAT_EXT_DROPOUT", 0.0)
    monkeypatch.setattr(config, "USE_HEMODYNAMIC_HEAD", False)

    # ------------------------------------------------------------------
    # 3. Discover the generated NIfTI files through the production
    #    file-discovery function.
    # ------------------------------------------------------------------
    runs = get_subject_files(
        data_dir,
        num_runs=synthetic_config.num_runs,
    )

    assert len(runs) == (
        synthetic_config.num_subjects * synthetic_config.num_runs
    )
    assert all("fmri" in run for run in runs)
    assert all("events" in run for run in runs)

    # ------------------------------------------------------------------
    # 4. Load the files through the production NIfTI dataset.
    # ------------------------------------------------------------------
    dataset = NiftiLaminarDataset(
        data_list=runs,
        mask_img=data_dir / "group_roi_mask.nii",
        window_size=window_size,
        run_type="train",
        sensory_regions=["S1_Postcentral"],
    )

    assert len(dataset) > 0
    assert dataset.num_nodes == synthetic_config.num_nodes
    assert dataset.adjacency.shape == (
        dataset.num_nodes,
        dataset.num_nodes,
    )

    # The synthetic atlas should select a subset of nodes, not silently
    # fall back to stimulating the entire network.
    selected_sensory_nodes = int(dataset.sensory_mask.sum().item())

    assert selected_sensory_nodes > 0
    assert selected_sensory_nodes < dataset.num_nodes

    # ------------------------------------------------------------------
    # 5. Read one CPU batch.
    # ------------------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    fmri_history, stimulus_history, target = next(iter(loader))

    assert fmri_history.shape == (
        2,
        window_size,
        dataset.num_nodes,
    )
    assert stimulus_history.shape == (
        2,
        window_size,
        1,
    )
    assert target.shape == (
        2,
        prediction_horizon,
        dataset.num_nodes,
    )

    assert torch.isfinite(fmri_history).all()
    assert torch.isfinite(stimulus_history).all()
    assert torch.isfinite(target).all()

    # The dense stimulus file should have been loaded rather than replaced
    # by an all-zero stimulus.
    assert torch.count_nonzero(stimulus_history) > 0

    # ------------------------------------------------------------------
    # 6. Construct the production H1 model.
    # ------------------------------------------------------------------
    model = build_model(
        num_nodes=dataset.num_nodes,
        time_points=window_size,
        sensory_mask=dataset.sensory_mask,
        model_type="H1",
        hidden_dim=16,
        use_hemodynamic_head=False,
    )
    model.train()

    # ------------------------------------------------------------------
    # 7. Forward pass.
    # ------------------------------------------------------------------
    prediction, attention, hidden, head_weights = model(
        fmri_history,
        stimulus_history,
        dataset.adjacency,
    )

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    assert hidden.shape[:2] == (
        fmri_history.shape[0],
        dataset.num_nodes,
    )
    assert attention is not None

    # Head weights are only collected when return_head_weights=True.
    assert head_weights is None

    # ------------------------------------------------------------------
    # 8. MSE and backward pass.
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    loss = criterion(prediction, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert any(
        parameter.grad is not None
        for parameter in trainable_parameters
    )

    finite_gradients = [
        parameter.grad
        for parameter in trainable_parameters
        if parameter.grad is not None
    ]

    assert finite_gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in finite_gradients
    )

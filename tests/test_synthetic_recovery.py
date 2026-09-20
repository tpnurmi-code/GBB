import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from gbb.analysis.synthetic_recovery import (
    compute_recovery_metrics,
    evaluate_synthetic_recovery,
)
from gbb.synthetic.export import write_node_scalar_maps


def test_perfect_recovery_metrics() -> None:
    truth = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    metrics = compute_recovery_metrics(
        truth,
        truth.copy(),
    )

    assert metrics.n_nodes == 4
    assert metrics.pearson_r == pytest.approx(1.0)
    assert metrics.spearman_rho == pytest.approx(1.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.regression_slope == pytest.approx(1.0)
    assert metrics.regression_intercept == pytest.approx(0.0)
    assert metrics.lin_ccc == pytest.approx(1.0)


def test_correlation_does_not_imply_absolute_recovery() -> None:
    truth = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    learned = (
        2.0 * truth
        + 1.0
    )

    metrics = compute_recovery_metrics(
        truth,
        learned,
    )

    assert metrics.pearson_r == pytest.approx(1.0)
    assert metrics.spearman_rho == pytest.approx(1.0)

    assert metrics.regression_slope == pytest.approx(2.0)
    assert metrics.regression_intercept == pytest.approx(1.0)

    assert metrics.mae > 0.0
    assert metrics.rmse > 0.0
    assert metrics.lin_ccc < 1.0


def test_write_node_scalar_maps(tmp_path) -> None:
    label_mask = np.zeros(
        (3, 2, 2),
        dtype=np.int16,
    )

    label_mask[0, :, :] = 1
    label_mask[1, :, :] = 2

    affine = np.eye(
        4,
        dtype=np.float64,
    )

    values = np.asarray(
        [1.5, -0.5],
        dtype=np.float64,
    )

    output = tmp_path / "maps"

    write_node_scalar_maps(
        output,
        {
            "GT_Test": values,
        },
        label_mask,
        affine,
    )

    saved_vector = np.load(
        output / "GT_Test.npy"
    )

    assert np.array_equal(
        saved_vector,
        values,
    )

    image = nib.load(
        str(
            output
            / "GT_Test.nii.gz"
        )
    )

    data = image.get_fdata()

    assert np.allclose(
        data[label_mask == 1],
        1.5,
    )

    assert np.allclose(
        data[label_mask == 2],
        -0.5,
    )

    assert np.allclose(
        data[label_mask == 0],
        0.0,
    )


def test_synthetic_recovery_evaluator(tmp_path) -> None:
    dataset = (
        tmp_path
        / "synthetic_dataset"
    )

    ground_truth_directory = (
        dataset
        / "ground_truth"
    )

    ground_truth_directory.mkdir(
        parents=True
    )

    tau = np.asarray(
        [1.8, 2.5, 3.3, 4.1],
        dtype=np.float64,
    )

    drive = np.asarray(
        [0.35, 0.15, -0.15, -0.35],
        dtype=np.float64,
    )

    np.savez_compressed(
        ground_truth_directory
        / "mechanistic_ground_truth.npz",
        tau_seconds=tau,
        intrinsic_drive=drive,
    )

    pd.DataFrame(
        {
            "node_id": [1, 2, 3, 4],
            "label": [
                "S1",
                "M1",
                "S2",
                "Association",
            ],
        }
    ).to_csv(
        ground_truth_directory
        / "node_ground_truth.csv",
        index=False,
    )

    learned_directory = (
        tmp_path
        / "learned"
    )

    learned_directory.mkdir()

    np.save(
        learned_directory
        / "Map_CfC_Tau_s.npy",
        tau,
    )

    np.save(
        learned_directory
        / "Map_CfC_IntrinsicDrive.npy",
        drive,
    )

    output_directory = (
        tmp_path
        / "recovery"
    )

    summary = evaluate_synthetic_recovery(
        dataset,
        learned_directory,
        output_directory,
    )

    assert set(
        summary["target"]
    ) == {
        "tau",
        "intrinsic_drive",
    }

    assert np.allclose(
        summary["pearson_r"],
        1.0,
    )

    assert np.allclose(
        summary["lin_ccc"],
        1.0,
    )

    assert (
        output_directory
        / "recovery_summary.csv"
    ).is_file()

    assert (
        output_directory
        / "recovery_tau_nodes.csv"
    ).is_file()

    assert (
        output_directory
        / "recovery_intrinsic_drive_nodes.csv"
    ).is_file()

    report_path = (
        output_directory
        / "recovery_report.json"
    )

    assert report_path.is_file()

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["schema_version"] == 1

    assert len(
        report["targets"]
    ) == 2


def test_recovery_shape_mismatch_fails() -> None:
    truth = np.asarray(
        [1.0, 2.0, 3.0]
    )

    learned = np.asarray(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        compute_recovery_metrics(
            truth,
            learned,
        )
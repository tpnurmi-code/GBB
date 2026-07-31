"""Export interpretable model parameters and data-dependent interaction maps.

The names in this module deliberately distinguish learned model quantities from
validated biological measurements. For example, graph attention is exported as
``effective interaction`` rather than as causal or neurotransmitter-specific
connectivity.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.maskers import NiftiLabelsMasker

from gbb.config import config


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _dataset_reference(dataloader):
    dataset = dataloader.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _as_node_vector(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


def _make_masker(mask_path, masker=None):
    if masker is not None:
        return masker
    fitted = NiftiLabelsMasker(
        labels_img=str(mask_path),
        standardize=False,
        detrend=False,
    )
    fitted.fit()
    return fitted


def _save_node_vector(
    values,
    *,
    name: str,
    map_directory: Path,
    mask_path,
    masker,
) -> None:
    vector = _as_node_vector(values)
    np.save(map_directory / f"{name}.npy", vector)
    export_scalar_map_to_csv(
        vector,
        mask_path,
        map_directory / f"{name}.csv",
        level="region",
    )
    image = masker.inverse_transform(vector.reshape(1, -1))
    image.to_filename(str(map_directory / f"{name}.nii.gz"))


def _collect_attention_matrices(
    model,
    dataloader,
    *,
    num_batches: int,
) -> np.ndarray:
    """Collect one signed effective-interaction matrix per batch.

    Matrix convention is ``[target, source]`` because that is the convention
    used inside :class:`MultiHeadFastKANLayer`.
    """
    raw_model = _unwrap_model(model)
    device = next(raw_model.parameters()).device
    dataset = _dataset_reference(dataloader)
    adjacency = dataset.adjacency.to(device)
    matrices: list[np.ndarray] = []

    raw_model.eval()
    with torch.no_grad():
        for batch in islice(iter(dataloader), max(1, int(num_batches))):
            fmri, stimulus = batch[0], batch[1]
            fmri = fmri.to(device)
            stimulus = stimulus.to(device)
            _, attention, _, _ = raw_model(fmri, stimulus, adjacency)
            if attention is None:
                continue
            matrices.append(attention.mean(dim=0).detach().cpu().numpy())
    if not matrices:
        raise RuntimeError("No attention matrices were produced by the model")
    return np.stack(matrices, axis=0)


def _fastkan_parameter_maps(model) -> dict[str, np.ndarray]:
    raw_model = _unwrap_model(model)
    if not raw_model.kan_layers:
        return {}

    layers = list(raw_model.kan_layers)
    coefficient_stack = torch.stack(
        [layer.spline_coeffs.detach() for layer in layers], dim=0
    )  # layers,target,source,head,basis
    attenuation_stack = torch.stack(
        [torch.sigmoid(layer.edge_attenuation_logits.detach()) for layer in layers],
        dim=0,
    )

    coefficient_magnitude = coefficient_stack.abs()
    if coefficient_stack.shape[-1] >= 3:
        second_difference = torch.diff(coefficient_stack, n=2, dim=-1).abs()
        curve_complexity = second_difference.mean(dim=(0, 2, 3, 4))
    else:
        curve_complexity = coefficient_magnitude.mean(dim=(0, 2, 3, 4))

    # Narrow basis usage -> larger specificity. Absolute coefficients serve as
    # non-negative importance weights; this is a model-derived diagnostic only.
    basis_grid = layers[0].mu.detach().to(coefficient_stack)
    basis_weights = coefficient_magnitude.mean(dim=(0, 2, 3))  # target,basis
    normalized = basis_weights / basis_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    center = (normalized * basis_grid).sum(dim=-1, keepdim=True)
    variance = (normalized * (basis_grid - center).square()).sum(dim=-1)
    tuning_specificity = 1.0 / (variance.sqrt() + 1e-6)

    edge_attenuation = attenuation_stack.mean(dim=(0, 2, 3))
    basis_usage = coefficient_magnitude.mean(dim=(0, 2, 3, 4))
    return {
        "Map_FastKAN_CurveComplexity": curve_complexity.cpu().numpy(),
        "Map_FastKAN_TuningSpecificity": tuning_specificity.cpu().numpy(),
        "Map_EdgeAttenuation": edge_attenuation.cpu().numpy(),
        "Map_FastKAN_BasisUsage": basis_usage.cpu().numpy(),
    }


def save_mechanistic_atlas(
    model,
    mask_path,
    node_vals=None,
    output_dir=None,
    masker=None,
    dataloader=None,
    num_batches: int = 10,
    datestring: str = "",
):
    """Save static parameter maps and data-dependent interaction summaries.

    ``node_vals`` is retained for compatibility with the original call site and
    is exported as validation connection contribution when supplied.
    """
    output_root = Path(output_dir or config.RESULTS_DIR)
    map_directory = output_root / (f"maps_{datestring}" if datestring else "maps")
    map_directory.mkdir(parents=True, exist_ok=True)
    fitted_masker = _make_masker(mask_path, masker)
    raw_model = _unwrap_model(model)

    static_maps = {
        "Map_CfC_Tau_s": raw_model.cfc.get_tau_values(),
        "Map_CfC_IntrinsicDrive": raw_model.cfc.get_intrinsic_drive(),
        **_fastkan_parameter_maps(raw_model),
    }
    if node_vals is not None:
        static_maps["Map_ValidationConnectionContribution"] = node_vals

    for name, values in static_maps.items():
        vector = _as_node_vector(values)
        if vector.size != raw_model.num_nodes:
            raise ValueError(
                f"{name} contains {vector.size} values; expected {raw_model.num_nodes}"
            )
        _save_node_vector(
            vector,
            name=name,
            map_directory=map_directory,
            mask_path=mask_path,
            masker=fitted_masker,
        )

    if dataloader is not None:
        batch_matrices = _collect_attention_matrices(
            raw_model,
            dataloader,
            num_batches=num_batches,
        )
        average_matrix = batch_matrices.mean(axis=0)
        np.save(map_directory / "EffectiveInteraction_target_by_source.npy", average_matrix)
        np.savetxt(
            map_directory / "EffectiveInteraction_target_by_source.csv",
            average_matrix,
            delimiter=",",
        )

        incoming = average_matrix.sum(axis=1)
        outgoing = average_matrix.sum(axis=0)
        net_flow = outgoing - incoming
        interaction_maps = {
            "Map_EffectiveIncomingStrength": incoming,
            "Map_EffectiveOutgoingStrength": outgoing,
            "Map_EffectiveNetFlow": net_flow,
        }
        for name, values in interaction_maps.items():
            _save_node_vector(
                values,
                name=name,
                map_directory=map_directory,
                mask_path=mask_path,
                masker=fitted_masker,
            )
    return map_directory


def export_map_stability(
    model,
    dataloader,
    output_dir,
    datestring: str = "",
    num_batches: int = 10,
):
    """Export within-loader variability of effective incoming interaction.

    This is a diagnostic across sampled batches, not the cross-seed,
    cross-subject, or cross-dataset stability required for biological
    identifiability claims.
    """
    output_root = Path(output_dir)
    stability_directory = output_root / (
        f"stability_{datestring}" if datestring else "stability"
    )
    stability_directory.mkdir(parents=True, exist_ok=True)
    matrices = _collect_attention_matrices(model, dataloader, num_batches=num_batches)
    incoming_by_batch = matrices.sum(axis=2)  # batch,target
    mean = incoming_by_batch.mean(axis=0)
    standard_deviation = incoming_by_batch.std(axis=0, ddof=0)
    coefficient_of_variation = standard_deviation / (np.abs(mean) + 1e-8)

    np.save(stability_directory / "incoming_mean.npy", mean)
    np.save(stability_directory / "incoming_sd.npy", standard_deviation)
    np.save(stability_directory / "incoming_cv.npy", coefficient_of_variation)
    pd.DataFrame(
        {
            "node_index": np.arange(mean.size),
            "incoming_mean": mean,
            "incoming_sd": standard_deviation,
            "incoming_cv": coefficient_of_variation,
        }
    ).to_csv(stability_directory / "batch_stability.csv", index=False)
    return stability_directory


def export_scalar_map_to_csv(
    fmri_data,
    mask_path,
    output_csv,
    level: str = "region",
):
    """Export a scalar/vector/4-D map with MNI coordinates.

    At ``level='region'`` one row is produced per non-zero integer atlas label.
    A one-dimensional input is interpreted as one value per label. A 3-D or 4-D
    array is sampled inside the labelled mask. At ``level='voxel'`` one row is
    produced for every non-zero voxel.
    """
    if level not in {"region", "voxel"}:
        raise ValueError("level must be 'region' or 'voxel'")

    mask_image = nib.load(str(mask_path))
    mask_data = mask_image.get_fdata()
    affine = mask_image.affine
    region_ids = np.sort(np.unique(mask_data[mask_data != 0]))
    values = np.asarray(fmri_data)
    rows: list[dict[str, float | int]] = []

    if values.ndim == 1:
        if level != "region":
            raise ValueError("A one-dimensional input supports only level='region'")
        if values.size != region_ids.size:
            raise ValueError(
                f"Input contains {values.size} values, but mask has "
                f"{region_ids.size} non-zero labels"
            )
        for region_id, value in zip(region_ids, values, strict=True):
            indices = np.argwhere(mask_data == region_id)
            center = nib.affines.apply_affine(affine, indices.mean(axis=0))
            rows.append(
                {
                    "Region_ID": int(region_id),
                    "MNI_X": float(center[0]),
                    "MNI_Y": float(center[1]),
                    "MNI_Z": float(center[2]),
                    "value": float(value),
                }
            )
    else:
        if values.shape[:3] != mask_data.shape:
            raise ValueError(
                f"Spatial shape mismatch: data={values.shape[:3]}, mask={mask_data.shape}"
            )
        if level == "region":
            for region_id in region_ids:
                region_mask = mask_data == region_id
                indices = np.argwhere(region_mask)
                center = nib.affines.apply_affine(affine, indices.mean(axis=0))
                region_values = values[region_mask]
                row: dict[str, float | int] = {
                    "Region_ID": int(region_id),
                    "MNI_X": float(center[0]),
                    "MNI_Y": float(center[1]),
                    "MNI_Z": float(center[2]),
                }
                if values.ndim == 3:
                    row["value"] = float(np.nanmean(region_values))
                elif values.ndim == 4:
                    mean_series = np.nanmean(region_values, axis=0)
                    row.update(
                        {f"t_{index}": float(value) for index, value in enumerate(mean_series)}
                    )
                else:
                    raise ValueError("Data must be 1-D, 3-D, or 4-D")
                rows.append(row)
        else:
            for voxel_index in np.argwhere(mask_data != 0):
                coordinate = nib.affines.apply_affine(affine, voxel_index)
                voxel_value = values[tuple(voxel_index)]
                row = {
                    "i": int(voxel_index[0]),
                    "j": int(voxel_index[1]),
                    "k": int(voxel_index[2]),
                    "Region_ID": int(mask_data[tuple(voxel_index)]),
                    "MNI_X": float(coordinate[0]),
                    "MNI_Y": float(coordinate[1]),
                    "MNI_Z": float(coordinate[2]),
                }
                if values.ndim == 3:
                    row["value"] = float(voxel_value)
                elif values.ndim == 4:
                    row.update(
                        {
                            f"t_{index}": float(value)
                            for index, value in enumerate(np.asarray(voxel_value))
                        }
                    )
                else:
                    raise ValueError("Data must be 1-D, 3-D, or 4-D")
                rows.append(row)

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame
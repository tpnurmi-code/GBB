"""Disk export utilities matching the existing GBB local-data layout."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from .anatomy import SyntheticAnatomy
from .config import SyntheticFMRIConfig
from .dynamics import NeuralGroundTruth
from .hemodynamics import HemodynamicGroundTruth
from .network import GroundTruthNetwork


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "NIfTI export requires nibabel. Install the GBB runtime dependencies "
            "or run `python -m pip install nibabel`."
        ) from exc
    return nib


def prepare_output_directory(config: SyntheticFMRIConfig) -> Path:
    output = Path(config.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output}. Use overwrite=True or --overwrite."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _synthetic_affine(config: SyntheticFMRIConfig) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = config.voxel_size_mm
    affine[1, 1] = config.voxel_size_mm
    affine[2, 2] = config.voxel_size_mm
    center = (np.asarray(config.volume_shape, dtype=np.float64) - 1.0) / 2.0
    affine[:3, 3] = -center * config.voxel_size_mm
    return affine


def build_label_volumes(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build compact non-overlapping labelled parcels and column masks."""
    shape = config.volume_shape
    label_mask = np.zeros(shape, dtype=np.int16)
    coarse_mask = np.zeros(shape, dtype=np.int16)
    column_mask = np.zeros(shape, dtype=np.int16)
    affine = _synthetic_affine(config)
    voxels_by_node: list[np.ndarray] = []

    radius = config.parcel_radius_voxels
    margin = radius + 2
    spacing = 2 * radius + 3
    columns_x = int(np.ceil(np.sqrt(config.num_columns)))
    columns_y = int(np.ceil(config.num_columns / columns_x))
    layer_count = len(config.cortical_layers)

    required_x = margin * 2 + columns_x * spacing
    required_y = margin * 2 + columns_y * spacing
    required_z = margin * 2 + layer_count * spacing
    if required_x >= shape[0] or required_y >= shape[1] or required_z >= shape[2]:
        raise ValueError(
            "volume_shape is too small for the requested columns/layers. "
            f"Need at least approximately {(required_x + 1, required_y + 1, required_z + 1)}."
        )

    for node in range(anatomy.num_nodes):
        column_zero = int(anatomy.column_ids[node] - 1)
        col_x = column_zero % columns_x
        col_y = column_zero // columns_x
        layer = int(anatomy.layer_index[node])
        center = np.array(
            [
                margin + col_x * spacing,
                margin + col_y * spacing,
                margin + layer * spacing,
            ],
            dtype=int,
        )
        grid = np.indices(shape).reshape(3, -1).T
        squared = np.sum((grid - center[None, :]) ** 2, axis=1)
        selected = grid[squared <= radius**2]
        if selected.size == 0:
            raise RuntimeError(f"No voxels generated for node {node}")
        index_tuple = tuple(selected.T)
        label_mask[index_tuple] = node + 1
        coarse_mask[index_tuple] = int(anatomy.network_ids[node]) + 1
        column_mask[index_tuple] = int(anatomy.column_ids[node])
        voxels_by_node.append(selected)

    return label_mask, coarse_mask, column_mask, affine, voxels_by_node


def write_masks_and_metadata(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    nib = _require_nibabel()
    label_mask, coarse_mask, column_mask, affine, voxels_by_node = build_label_volumes(
        config, anatomy
    )
    nib.save(nib.Nifti1Image(label_mask, affine), str(output_dir / "group_roi_mask.nii"))
    
    nib.save(
        nib.Nifti1Image(coarse_mask, affine),
        str(output_dir / "group_roi_mask_10.nii"),
    )

    # The explicit cortical-column NIfTI belongs to the
    # synthetic 7T-CBV configuration only.
    if str(config.response_kind).lower() == "cbv":
        nib.save(
            nib.Nifti1Image(column_mask, affine),
            str(output_dir / "cortical_columns_7T.nii"),
        )

    metadata: dict[str, dict[str, Any]] = {}
    for node in range(anatomy.num_nodes):
        metadata[str(node + 1)] = {
            "label": anatomy.labels[node],
            "centroid_mni": anatomy.coordinates_mm[node].round(4).tolist(),
            "hierarchy": float(anatomy.hierarchy[node]),
            "spatial_gradient": float(anatomy.spatial_gradient[node]),
            "layer": anatomy.layer_names[int(anatomy.layer_index[node])],
            "column_id": int(anatomy.column_ids[node]),
            "network_id": int(anatomy.network_ids[node]),
            "hemisphere": str(anatomy.hemisphere[node]),
            "synthetic": True,
        }
    (output_dir / "mask_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return label_mask, affine, voxels_by_node


def write_events_tsv(events: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["onset", "duration", "amplitude", "trial_type"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(events)


def write_dense_stimulus_mat(
    config: SyntheticFMRIConfig,
    neural_stimulus: np.ndarray,
    path: Path,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    raw_count = config.n_timepoints * config.samples_per_tr
    raw_time = np.arange(raw_count, dtype=np.float64) / config.raw_stimulus_sampling_rate
    neural_time = np.arange(neural_stimulus.size, dtype=np.float64) * config.neural_dt
    raw = np.interp(raw_time, neural_time, neural_stimulus, left=0.0, right=0.0)
    # Nonnegative accelerometer-magnitude-like signal with small sensor noise.
    raw = np.clip(raw + rng.normal(0.0, 0.025, size=raw_count), 0.0, None)
    savemat(path, {"data": raw.astype(np.float32)[:, None]})


def write_node_timeseries_npz(
    path: Path,
    *,
    observed: np.ndarray,
    noiseless: np.ndarray,
    neural: np.ndarray,
    stimulus: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    noise_components: dict[str, np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {
        "observed_fmri": observed.astype(np.float32),
        "noiseless_fmri": noiseless.astype(np.float32),
        "neural_activity": neural.astype(np.float32),
        "stimulus_neural_rate": stimulus.astype(np.float32),
    }
    payload.update({f"diagnostic_{key}": value for key, value in diagnostics.items()})
    payload.update({f"noise_{key}": value for key, value in noise_components.items()})
    np.savez_compressed(path, **payload)


def write_fmri_nifti(
    config: SyntheticFMRIConfig,
    observed_node_series: np.ndarray,
    label_mask: np.ndarray,
    affine: np.ndarray,
    voxels_by_node: list[np.ndarray],
    path: Path,
    *,
    seed: int,
) -> Path:
    nib = _require_nibabel()
    rng = np.random.default_rng(seed)
    t, n = observed_node_series.shape
    if n != len(voxels_by_node):
        raise ValueError("Node series and mask parcels do not match")

    volume = np.zeros(config.volume_shape + (t,), dtype=np.float32)
    for node, voxels in enumerate(voxels_by_node):
        local_scale = rng.normal(1.0, 0.035, size=voxels.shape[0])
        local_offset = rng.normal(0.0, 0.08, size=voxels.shape[0])
        local_noise = rng.normal(
            0.0,
            config.voxel_noise_sd,
            size=(voxels.shape[0], t),
        )
        parcel = (
            1000.0
            + 12.0 * local_scale[:, None] * observed_node_series[:, node][None, :]
            + 12.0 * local_offset[:, None]
            + 12.0 * local_noise
        )
        volume[tuple(voxels.T)] = parcel.astype(np.float32)

    image = nib.Nifti1Image(volume, affine)
    image.header.set_xyzt_units("mm", "sec")
    zooms = (config.voxel_size_mm,) * 3 + (config.tr,)
    image.header.set_zooms(zooms)
    final_path = path.with_suffix(".nii.gz") if config.compression else path.with_suffix(".nii")
    nib.save(image, str(final_path))
    return final_path


def write_ground_truth(
    output_dir: Path,
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    network: GroundTruthNetwork,
    neural: NeuralGroundTruth,
    hemodynamics: HemodynamicGroundTruth,
) -> None:
    ground_dir = output_dir / "ground_truth"
    ground_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        ground_dir / "mechanistic_ground_truth.npz",
        coordinates_mm=anatomy.coordinates_mm,
        hierarchy=anatomy.hierarchy,
        spatial_gradient=anatomy.spatial_gradient,
        layer_index=anatomy.layer_index,
        column_ids=anatomy.column_ids,
        network_ids=anatomy.network_ids,
        sensory_nodes=anatomy.sensory_nodes,
        tau_seconds=neural.tau_seconds,
        intrinsic_drive=neural.intrinsic_drive,
        stimulus_gain=neural.stimulus_gain,
        edge_source=network.source,
        edge_target=network.target,
        edge_channel=network.channel,
        edge_signed_weight=network.signed_weight,
        edge_delay_steps=network.delay_steps,
        edge_delay_seconds=network.delay_seconds,
        edge_velocity_mm_s=network.velocity_mm_s,
        edge_distance_mm=network.distance_mm,
        rbf_centers=network.rbf_centers,
        rbf_widths=network.rbf_widths,
        rbf_coefficients=network.rbf_coefficients,
        adjacency_by_channel=network.adjacency_by_channel,
        weight_by_channel=network.weight_by_channel,
        hrf_time_to_peak_s=hemodynamics.time_to_peak_s,
        hrf_dispersion_s=hemodynamics.dispersion_s,
        hrf_undershoot_ratio=hemodynamics.undershoot_ratio,
        hrf_amplitude=hemodynamics.amplitude,
        hrf_onset_delay_s=hemodynamics.onset_delay_s,
        hrf_kernels=hemodynamics.kernels,
    )

    node_fields = [
        "node_id",
        "label",
        "x_mm",
        "y_mm",
        "z_mm",
        "hemisphere",
        "network_id",
        "column_id",
        "layer",
        "hierarchy",
        "spatial_gradient",
        "tau_seconds",
        "intrinsic_drive",
        "stimulus_gain",
        "hemodynamic_peak_s",
        "hemodynamic_dispersion_s",
        "hemodynamic_amplitude",
    ]
    with (ground_dir / "node_ground_truth.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=node_fields)
        writer.writeheader()
        for node in range(anatomy.num_nodes):
            writer.writerow(
                {
                    "node_id": node + 1,
                    "label": anatomy.labels[node],
                    "x_mm": anatomy.coordinates_mm[node, 0],
                    "y_mm": anatomy.coordinates_mm[node, 1],
                    "z_mm": anatomy.coordinates_mm[node, 2],
                    "hemisphere": anatomy.hemisphere[node],
                    "network_id": anatomy.network_ids[node],
                    "column_id": anatomy.column_ids[node],
                    "layer": anatomy.layer_names[int(anatomy.layer_index[node])],
                    "hierarchy": anatomy.hierarchy[node],
                    "spatial_gradient": anatomy.spatial_gradient[node],
                    "tau_seconds": neural.tau_seconds[node],
                    "intrinsic_drive": neural.intrinsic_drive[node],
                    "stimulus_gain": neural.stimulus_gain[node],
                    "hemodynamic_peak_s": hemodynamics.time_to_peak_s[node],
                    "hemodynamic_dispersion_s": hemodynamics.dispersion_s[node],
                    "hemodynamic_amplitude": hemodynamics.amplitude[node],
                }
            )

    edge_fields = [
        "edge_id",
        "source_node",
        "target_node",
        "channel",
        "signed_weight",
        "distance_mm",
        "velocity_mm_s",
        "delay_seconds",
        "delay_steps",
        "layer_gain",
    ]
    with (ground_dir / "edge_ground_truth.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=edge_fields)
        writer.writeheader()
        for edge in range(network.num_edges):
            writer.writerow(
                {
                    "edge_id": edge,
                    "source_node": int(network.source[edge]) + 1,
                    "target_node": int(network.target[edge]) + 1,
                    "channel": network.channel_names[int(network.channel[edge])],
                    "signed_weight": network.signed_weight[edge],
                    "distance_mm": network.distance_mm[edge],
                    "velocity_mm_s": network.velocity_mm_s[edge],
                    "delay_seconds": network.delay_seconds[edge],
                    "delay_steps": network.delay_steps[edge],
                    "layer_gain": network.layer_gain[edge],
                }
            )

    manifest = {
        "synthetic": True,
        "contains_participant_data": False,
        "response_kind": config.response_kind,
        "num_nodes": anatomy.num_nodes,
        "num_edges": network.num_edges,
        "edge_density_across_channels": network.density,
        "channel_names": list(network.channel_names),
        "config": config.to_dict(),
        "interpretation_warning": (
            "Driver-like, suppressive-like, and gain-modulatory-like are synthetic "
            "functional channels and not neurotransmitter labels."
        ),
    }
    (ground_dir / "ground_truth_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def write_dataset_description(output_dir: Path, config: SyntheticFMRIConfig) -> None:
    description = {
        "Name": "GBB Mechanistic Synthetic Ground-Truth fMRI",
        "Synthetic": True,
        "ParticipantData": False,
        "Privacy": "Generated entirely from mathematical mechanisms and random seeds.",
        "ResponseKind": config.response_kind.upper(),
        "TRSeconds": config.tr,
        "NeuralIntegrationStepSeconds": config.neural_dt,
        "Subjects": config.num_subjects,
        "RunsPerSubject": config.num_runs,
        "TimepointsPerRun": config.n_timepoints,
        "Columns": config.num_columns,
        "Layers": list(config.cortical_layers),
        "Notes": (
            "Pseudo-MNI metadata and toy labelled volumes are intended for software "
            "and parameter-recovery validation, not anatomical inference."
        ),
    }
    (output_dir / "dataset_description.json").write_text(
        json.dumps(description, indent=2), encoding="utf-8"
    )

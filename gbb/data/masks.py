"""Helpers for labelled NIfTI masks and their metadata."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from nilearn.image import resample_to_img
from scipy.ndimage import center_of_mass


def load_mask(mask_img):
    """Load a mask path or return an already loaded nibabel image."""
    if isinstance(mask_img, (str, os.PathLike, Path)):
        path = Path(mask_img)
        if not path.is_file():
            raise FileNotFoundError(f"Mask file not found: {path}")
        image = nib.load(str(path))
    else:
        path = None
        image = mask_img

    if image is None or not hasattr(image, "get_fdata"):
        raise TypeError("mask_img must be a NIfTI path or a nibabel image")

    data = np.asarray(image.get_fdata())
    if data.ndim != 3:
        raise ValueError(f"Label mask must be 3D, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Label mask contains NaN or infinite values")

    return image, data, np.asarray(image.affine), path


def get_region_ids(mask_data: np.ndarray) -> np.ndarray:
    """Return sorted non-zero integer region identifiers."""
    mask_data = np.asarray(mask_data)
    nonzero = mask_data[mask_data != 0]
    if nonzero.size and not np.allclose(nonzero, np.round(nonzero)):
        raise ValueError("Label mask contains non-integer region identifiers")

    region_ids = np.unique(nonzero.astype(np.int64, copy=False))
    region_ids.sort()
    return region_ids


def load_mask_metadata(mask_path: Path | None) -> dict[int, dict[str, Any]]:
    """Load ``mask_metadata.json`` next to a mask, when present."""
    if mask_path is None:
        return {}

    metadata_path = mask_path.parent / "mask_metadata.json"
    if not metadata_path.is_file():
        return {}

    with metadata_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)

    if not isinstance(raw, dict):
        raise ValueError(f"Mask metadata must be a JSON object: {metadata_path}")

    metadata: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            region_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid region ID {key!r} in {metadata_path}") from exc

        if not isinstance(value, dict):
            raise ValueError(f"Metadata for region {region_id} must be an object")
        metadata[region_id] = value

    return metadata


def compute_centroids(
    mask_data: np.ndarray,
    affine: np.ndarray,
    region_ids: np.ndarray,
    metadata: dict[int, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Return one MNI centroid per region in ``region_ids`` order."""
    metadata = metadata or {}
    coords: list[np.ndarray] = []

    for region_id in region_ids:
        region_id_int = int(region_id)
        info = metadata.get(region_id_int, {})

        if "centroid_mni" in info:
            centroid = np.asarray(info["centroid_mni"], dtype=float)
            if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
                raise ValueError(f"Invalid centroid_mni for region {region_id_int}")
        else:
            voxel_center = center_of_mass(mask_data == region_id)
            if not np.all(np.isfinite(voxel_center)):
                raise ValueError(f"Could not compute centroid for region {region_id_int}")
            centroid = nib.affines.apply_affine(affine, voxel_center)

        coords.append(np.asarray(centroid, dtype=float))

    return np.asarray(coords, dtype=float)


def get_region_labels(
    region_ids: np.ndarray,
    metadata: dict[int, dict[str, Any]] | None = None,
) -> list[str]:
    """Return one label per region in ``region_ids`` order."""
    metadata = metadata or {}
    return [
        str(
            metadata.get(int(region_id), {}).get(
                "label",
                f"Region_{int(region_id)}",
            )
        )
        for region_id in region_ids
    ]


def get_voxel_coords_from_mask(
    mask_path,
    num_expected: int | None = None,
) -> np.ndarray:
    """Return MNI coordinates of every non-zero voxel in reading order."""
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")

    image = nib.load(str(path))
    data = np.asarray(image.get_fdata())
    voxel_indices = np.argwhere(data > 0)
    coords = nib.affines.apply_affine(image.affine, voxel_indices)

    if num_expected is not None and len(coords) != num_expected:
        raise ValueError(
            f"Mask contains {len(coords)} non-zero voxels; expected exactly {num_expected}"
        )

    return np.asarray(coords, dtype=float)


def load_column_ids(
    *,
    columnar_mask_path: Path,
    parcellation_img,
    mask_data: np.ndarray,
    region_ids: np.ndarray,
    node_region_ids: torch.Tensor,
    policy: str = "DISABLED",
) -> torch.Tensor:
    """Return one column ID per node under an explicit column-mask policy."""

    normalized_policy = str(policy).upper()

    valid_policies = {
        "DISABLED",
        "OPTIONAL",
        "ERROR",
        "REGION_FALLBACK",
    }

    if normalized_policy not in valid_policies:
        raise ValueError(
            "policy must be one of "
            "'DISABLED', 'OPTIONAL', 'ERROR', "
            f"or 'REGION_FALLBACK'; got {normalized_policy!r}"
        )

    region_ids = np.asarray(region_ids)

    if node_region_ids.numel() != len(region_ids):
        raise ValueError(
            "node_region_ids must contain exactly one value per region"
        )

    # Columns are intentionally not part of this experiment.
    if normalized_policy == "DISABLED":
        return node_region_ids.detach().clone().to(dtype=torch.long)

    if not columnar_mask_path.is_file():
        message = (
            f"Columnar mask file not found: {columnar_mask_path}"
        )

        # OPTIONAL means absence is expected and not an error.
        if normalized_policy == "OPTIONAL":
            return node_region_ids.detach().clone().to(dtype=torch.long)

        # REGION_FALLBACK explicitly reports degraded behaviour.
        if normalized_policy == "REGION_FALLBACK":
            warnings.warn(
                f"{message}. Using region-group IDs instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return node_region_ids.detach().clone().to(dtype=torch.long)

        raise FileNotFoundError(message)

    columnar_img = nib.load(str(columnar_mask_path))
    columnar_img = resample_to_img(
        source_img=columnar_img,
        target_img=parcellation_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    columnar_data = np.asarray(columnar_img.get_fdata())
    if columnar_data.shape != mask_data.shape:
        raise ValueError(
            f"Resampled columnar mask has shape {columnar_data.shape}; expected {mask_data.shape}"
        )

    column_ids: list[int] = []
    for node_index, region_id in enumerate(region_ids):
        values = columnar_data[mask_data == region_id]
        values = values[np.isfinite(values) & (values > 0)].astype(np.int64)

        if not values.size:
            message = f"Columnar mask contains no positive column ID for region ID {int(region_id)}"
            if normalized_policy == "REGION_FALLBACK":
                warnings.warn(
                    f"{message}. Using the node's region-group ID instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                column_ids.append(int(node_region_ids[node_index].item()))
                continue
            raise ValueError(message)

        column_ids.append(int(np.bincount(values).argmax()))

    return torch.tensor(column_ids, dtype=torch.long)

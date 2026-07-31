"""Helpers for labelled NIfTI masks and their metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass


def load_mask(mask_img):
    """Load a mask path or return an already loaded nibabel image."""
    if isinstance(mask_img, (str, os.PathLike, Path)):
        path = Path(mask_img)
        image = nib.load(str(path))
    else:
        path = None
        image = mask_img

    if image is None or not hasattr(image, "get_fdata"):
        raise TypeError("mask_img must be a NIfTI path or a nibabel image")

    data = np.asarray(image.get_fdata())
    if data.ndim != 3:
        raise ValueError(f"Label mask must be 3D, got shape {data.shape}")

    return image, data, np.asarray(image.affine), path


def get_region_ids(mask_data: np.ndarray) -> np.ndarray:
    """Return sorted non-zero integer region identifiers."""
    region_ids = np.unique(mask_data)
    region_ids = region_ids[region_ids != 0]
    region_ids.sort()
    return region_ids


def load_mask_metadata(mask_path: Path | None) -> dict[int, dict[str, Any]]:
    """Load ``mask_metadata.json`` next to a mask, when present."""
    if mask_path is None:
        return {}

    metadata_path = mask_path.parent / "mask_metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)

    metadata: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            metadata[int(key)] = value
        except (TypeError, ValueError):
            continue
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
        str(metadata.get(int(region_id), {}).get("label", f"Region_{int(region_id)}"))
        for region_id in region_ids
    ]


def get_voxel_coords_from_mask(mask_path, num_expected: int | None = None) -> np.ndarray:
    """Return MNI coordinates of every non-zero voxel in reading order."""
    image = nib.load(str(mask_path))
    data = image.get_fdata()
    voxel_indices = np.argwhere(data > 0)
    coords = nib.affines.apply_affine(image.affine, voxel_indices)

    if num_expected is not None and len(coords) != num_expected:
        if len(coords) < num_expected:
            raise ValueError(
                f"Mask contains {len(coords)} non-zero voxels, fewer than expected {num_expected}."
            )
        coords = coords[:num_expected]
    return np.asarray(coords, dtype=float)
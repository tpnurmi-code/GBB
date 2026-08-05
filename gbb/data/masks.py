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


def _load_column_ids(self, mask_data: np.ndarray) -> torch.Tensor:
        policy = str(
            getattr(config, "COLUMNAR_MASK_POLICY", "ERROR")
        ).upper()

        if policy not in {"ERROR", "REGION_FALLBACK"}:
            raise ValueError(
                "COLUMNAR_MASK_POLICY must be 'ERROR' or "
                f"'REGION_FALLBACK'; got {policy!r}"
            )

        columnar_path = Path(str(config.COLUMNAR_MASK_FILE))

        if not columnar_path.is_file():
            message = f"Columnar mask file not found: {columnar_path}"

            if policy == "REGION_FALLBACK":
                warnings.warn(
                    f"{message}. Using region-group IDs instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.node_region_ids.clone()

            raise FileNotFoundError(message)

        columnar_img = nib.load(str(columnar_path))
        columnar_img = resample_to_img(
            source_img=columnar_img,
            target_img=self.parcellation_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        columnar_data = np.asarray(columnar_img.get_fdata())
        column_ids: list[int] = []

        for node_index, region_id in enumerate(self.region_ids):
            values = columnar_data[mask_data == region_id]
            values = values[
                np.isfinite(values) & (values > 0)
            ].astype(int)

            if not values.size:
                message = (
                    "Columnar mask contains no positive column ID for "
                    f"region ID {int(region_id)}"
                )

                if policy == "REGION_FALLBACK":
                    warnings.warn(
                        f"{message}. Using the node's region-group ID instead.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    column_ids.append(
                        int(self.node_region_ids[node_index].item())
                    )
                    continue

                raise ValueError(message)

            column_ids.append(
                int(np.bincount(values).argmax())
            )

        return torch.tensor(column_ids, dtype=torch.long)
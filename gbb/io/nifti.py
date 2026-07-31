"""Neuroimaging and HDF5 serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import nibabel as nib
import numpy as np


def save_nifti(
    data_array,
    mask_data,
    affine,
    output_path,
) -> None:
    """Save values aligned either to non-zero voxels or labelled regions.

    ``data_array`` may contain one value per non-zero voxel or one value per
    non-zero integer label. Region-level values are broadcast to every voxel in
    their corresponding label.
    """
    values = np.asarray(data_array).reshape(-1)
    mask = np.asarray(mask_data)
    output = np.zeros(mask.shape, dtype=np.float32)
    nonzero = mask != 0
    region_ids = np.sort(np.unique(mask[nonzero]))

    if values.size == int(nonzero.sum()):
        output[nonzero] = values
    elif values.size == region_ids.size:
        for value, region_id in zip(values, region_ids, strict=True):
            output[mask == region_id] = value
    else:
        raise ValueError(
            f"Received {values.size} values, but mask contains "
            f"{int(nonzero.sum())} non-zero voxels and {region_ids.size} labels."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(output, affine), str(path))


def save_cifti(matrix_data, output_path) -> None:
    matrix = np.asarray(matrix_data)
    if matrix.ndim != 2:
        raise ValueError(f"CIFTI matrix must be two-dimensional, got {matrix.shape}")
    rows, columns = matrix.shape
    row_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=rows)
    column_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=columns)
    header = nib.cifti2.Cifti2Header.from_axes((row_axis, column_axis))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.cifti2.Cifti2Image(matrix, header=header), str(path))


def save_hdf5(data_dict: Mapping[str, object], output_path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for key, value in data_dict.items():
            handle.create_dataset(str(key), data=np.asarray(value))
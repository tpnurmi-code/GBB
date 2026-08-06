"""fMRI run loading, parcellation, and validation helpers."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import zscore


def load_fmri_run(
    *,
    fmri_path: Path,
    masker: NiftiLabelsMasker,
    parcellation_img: nib.spatialimages.SpatialImage,
    expected_nodes: int,
) -> np.ndarray:
    """Load one fMRI run and return finite node-wise z-scored time series."""
    if expected_nodes <= 0:
        raise ValueError("expected_nodes must be positive")
    if not fmri_path.is_file():
        raise FileNotFoundError(f"fMRI file not found: {fmri_path}")

    functional_img = nib.load(str(fmri_path))
    resampled = resample_to_img(
        source_img=functional_img,
        target_img=parcellation_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    fmri_data = np.asarray(
        masker.transform(resampled),
        dtype=np.float32,
    )

    expected_shape_text = f"(time, {expected_nodes})"
    if fmri_data.ndim != 2 or fmri_data.shape[1] != expected_nodes:
        raise ValueError(f"Masker returned {fmri_data.shape}; expected {expected_shape_text}")
    if fmri_data.shape[0] == 0:
        raise ValueError(f"No fMRI time points were extracted from {fmri_path}")
    if not np.all(np.isfinite(fmri_data)):
        bad_count = int(fmri_data.size - np.count_nonzero(np.isfinite(fmri_data)))
        raise ValueError(
            f"Extracted fMRI data from {fmri_path} contains {bad_count} non-finite values"
        )

    node_std = np.std(fmri_data, axis=0)
    constant_nodes = np.flatnonzero(node_std <= 1e-8)
    if constant_nodes.size:
        raise ValueError(
            f"Extracted fMRI data from {fmri_path} contains "
            f"{constant_nodes.size} constant or near-constant nodes. "
            "First affected node indices: "
            f"{constant_nodes[:10].tolist()}"
        )

    standardized = np.asarray(
        zscore(fmri_data, axis=0),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(standardized)):
        raise RuntimeError(f"Z-scoring produced non-finite values for {fmri_path}")

    return standardized

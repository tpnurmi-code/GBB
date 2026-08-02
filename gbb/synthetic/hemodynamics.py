"""Region-varying BOLD and CBV observation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gamma

from .anatomy import SyntheticAnatomy
from .config import SyntheticFMRIConfig


@dataclass(slots=True)
class HemodynamicGroundTruth:
    """Node-specific hemodynamic parameters and kernels."""

    response_kind: str
    time_to_peak_s: np.ndarray
    dispersion_s: np.ndarray
    undershoot_ratio: np.ndarray
    amplitude: np.ndarray
    onset_delay_s: np.ndarray
    kernels: np.ndarray


def _bold_kernel(
    time: np.ndarray,
    peak: float,
    dispersion: float,
    undershoot_ratio: float,
    onset_delay: float,
) -> np.ndarray:
    shifted = np.maximum(0.0, time - onset_delay)
    peak_shape = max(2.1, peak / max(0.25, dispersion) + 1.0)
    positive = gamma.pdf(shifted, a=peak_shape, scale=max(0.25, dispersion))
    undershoot_peak = peak + 10.0
    undershoot_dispersion = max(0.8, dispersion * 1.8)
    undershoot_shape = max(2.1, undershoot_peak / undershoot_dispersion + 1.0)
    negative = gamma.pdf(
        shifted,
        a=undershoot_shape,
        scale=undershoot_dispersion,
    )
    kernel = positive - undershoot_ratio * negative
    kernel[time < onset_delay] = 0.0
    norm = np.sum(np.abs(kernel))
    if norm <= 1e-12:
        kernel[0] = 1.0
        norm = 1.0
    return kernel / norm


def _cbv_kernel(
    time: np.ndarray,
    peak: float,
    dispersion: float,
    onset_delay: float,
) -> np.ndarray:
    shifted = np.maximum(0.0, time - onset_delay)
    shape = max(2.1, peak / max(0.25, dispersion) + 1.0)
    kernel = gamma.pdf(shifted, a=shape, scale=max(0.25, dispersion))
    kernel[time < onset_delay] = 0.0
    total = kernel.sum()
    if total <= 1e-12:
        kernel[0] = 1.0
        total = 1.0
    return kernel / total


def build_hemodynamic_ground_truth(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    *,
    seed_offset: int = 0,
    subject_scale: np.ndarray | None = None,
) -> HemodynamicGroundTruth:
    """Construct node-specific kernels with hierarchy and laminar variation."""
    rng = np.random.default_rng(config.seed + 3037 + seed_offset)
    n = anatomy.num_nodes
    layer_fraction = anatomy.layer_index / max(1, len(anatomy.layer_names) - 1)

    if config.response_kind == "bold":
        peak = 4.8 + 1.5 * anatomy.hierarchy + 0.30 * layer_fraction
        dispersion = 0.75 + 0.35 * anatomy.hierarchy + 0.12 * (1.0 - layer_fraction)
        undershoot = 0.08 + 0.12 * anatomy.hierarchy
        amplitude = 0.75 + 0.50 * (1.0 - anatomy.hierarchy) + 0.10 * layer_fraction
        onset = 0.25 + 0.35 * anatomy.hierarchy
    else:
        peak = 3.3 + 0.9 * anatomy.hierarchy + 0.18 * layer_fraction
        dispersion = 0.65 + 0.25 * anatomy.hierarchy
        undershoot = np.zeros(n, dtype=np.float64)
        amplitude = 0.85 + 0.35 * (1.0 - anatomy.hierarchy) + 0.12 * layer_fraction
        onset = 0.15 + 0.22 * anatomy.hierarchy

    peak += rng.normal(0.0, 0.18, size=n)
    dispersion *= rng.lognormal(mean=0.0, sigma=0.05, size=n)
    amplitude *= rng.lognormal(mean=0.0, sigma=0.08, size=n)
    onset += rng.normal(0.0, 0.06, size=n)

    if subject_scale is not None:
        subject_scale = np.asarray(subject_scale, dtype=np.float64)
        peak *= subject_scale
        dispersion *= np.sqrt(subject_scale)
        amplitude /= np.sqrt(subject_scale)

    peak = np.clip(peak, 2.2, 8.5)
    dispersion = np.clip(dispersion, 0.35, 2.0)
    amplitude = np.clip(amplitude, 0.35, 2.0)
    onset = np.clip(onset, 0.0, 1.2)

    kernel_duration = 32.0 if config.response_kind == "bold" else 24.0
    time = np.arange(0.0, kernel_duration, config.neural_dt, dtype=np.float64)
    kernels = np.empty((n, time.size), dtype=np.float64)
    for node in range(n):
        if config.response_kind == "bold":
            kernels[node] = _bold_kernel(
                time,
                peak=float(peak[node]),
                dispersion=float(dispersion[node]),
                undershoot_ratio=float(undershoot[node]),
                onset_delay=float(onset[node]),
            )
        else:
            kernels[node] = _cbv_kernel(
                time,
                peak=float(peak[node]),
                dispersion=float(dispersion[node]),
                onset_delay=float(onset[node]),
            )

    return HemodynamicGroundTruth(
        response_kind=config.response_kind,
        time_to_peak_s=peak.astype(np.float64),
        dispersion_s=dispersion.astype(np.float64),
        undershoot_ratio=undershoot.astype(np.float64),
        amplitude=amplitude.astype(np.float64),
        onset_delay_s=onset.astype(np.float64),
        kernels=kernels.astype(np.float64),
    )


def observe_fmri(
    config: SyntheticFMRIConfig,
    neural_activity: np.ndarray,
    hemodynamics: HemodynamicGroundTruth,
) -> np.ndarray:
    """Convolve latent neural activity and sample at the fMRI TR."""
    neural_activity = np.asarray(neural_activity, dtype=np.float64)
    if neural_activity.ndim != 2:
        raise ValueError("neural_activity must have shape (steps, nodes)")
    steps, nodes = neural_activity.shape
    if hemodynamics.kernels.shape[0] != nodes:
        raise ValueError("Hemodynamic kernels and neural nodes do not match")

    convolved = np.empty_like(neural_activity, dtype=np.float64)
    for node in range(nodes):
        signal = fftconvolve(
            neural_activity[:, node],
            hemodynamics.kernels[node],
            mode="full",
        )[:steps]
        convolved[:, node] = hemodynamics.amplitude[node] * signal

    sample_indices = np.rint(
        np.arange(config.n_timepoints, dtype=np.float64) * config.tr / config.neural_dt
    ).astype(int)
    sample_indices = np.clip(sample_indices, 0, steps - 1)
    sampled = convolved[sample_indices]
    sampled -= sampled.mean(axis=0, keepdims=True)
    scale = sampled.std(axis=0, keepdims=True) + 1e-6
    # Keep region-specific amplitude differences while preventing extreme scales.
    sampled = sampled / scale * hemodynamics.amplitude[None, :]
    return sampled.astype(np.float32)

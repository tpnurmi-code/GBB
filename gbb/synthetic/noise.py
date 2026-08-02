"""Statistically realistic nuisance structure for synthetic fMRI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import cdist

from .anatomy import SyntheticAnatomy
from .config import SyntheticFMRIConfig


@dataclass(slots=True)
class NoiseComponents:
    temporal_ar: np.ndarray
    spatial_measurement: np.ndarray
    global_signal: np.ndarray
    drift: np.ndarray
    physiological: np.ndarray
    motion: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return (
            self.temporal_ar
            + self.spatial_measurement
            + self.global_signal
            + self.drift
            + self.physiological
            + self.motion
        )


def _standardize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return (vector - vector.mean()) / (vector.std() + 1e-8)


def _one_over_f_noise(rng: np.random.Generator, length: int, exponent: float = 1.0) -> np.ndarray:
    frequencies = np.fft.rfftfreq(length)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    amplitude = np.zeros_like(frequencies)
    amplitude[1:] = 1.0 / np.power(frequencies[1:], exponent / 2.0)
    spectrum = amplitude * np.exp(1j * phases)
    signal = np.fft.irfft(spectrum, n=length)
    return _standardize(signal)


def build_measurement_noise(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    *,
    seed: int,
    run_scale: float = 1.0,
) -> NoiseComponents:
    """Generate autocorrelated, spatial, global, drift, physiological and motion noise."""
    rng = np.random.default_rng(seed)
    t = config.n_timepoints
    n = anatomy.num_nodes

    # Node-wise AR(1) noise.
    innovations = rng.normal(0.0, 1.0, size=(t, n))
    temporal_ar = np.zeros((t, n), dtype=np.float64)
    for index in range(1, t):
        temporal_ar[index] = (
            config.temporal_ar * temporal_ar[index - 1]
            + np.sqrt(max(1e-6, 1.0 - config.temporal_ar**2)) * innovations[index]
        )
    temporal_ar *= config.measurement_noise_sd * run_scale

    # Spatial covariance based on pseudo-MNI distance.
    distances = cdist(anatomy.coordinates_mm, anatomy.coordinates_mm)
    covariance = np.exp(-distances / config.spatial_noise_scale_mm)
    covariance += np.eye(n) * 1e-5
    chol = np.linalg.cholesky(covariance)
    spatial_measurement = rng.normal(size=(t, n)) @ chol.T
    spatial_measurement *= 0.45 * config.measurement_noise_sd * run_scale

    # Global low-frequency signal with hierarchy-dependent regional loading.
    global_course = _one_over_f_noise(rng, t, exponent=1.1)
    global_loading = 0.75 + 0.35 * anatomy.hierarchy
    global_signal = (
        config.global_signal_sd * run_scale * global_course[:, None] * global_loading[None, :]
    )

    # Very-low-frequency drift. Low-pass when the run is long enough.
    drift_course = _one_over_f_noise(rng, t, exponent=1.8)
    if t >= 20:
        nyquist = 0.5 / config.tr
        cutoff_hz = min(0.012, 0.8 * nyquist)
        if cutoff_hz > 0:
            sos = butter(2, cutoff_hz / nyquist, btype="lowpass", output="sos")
            drift_course = sosfiltfilt(sos, drift_course)
    drift_course = _standardize(drift_course)
    drift_loading = rng.normal(1.0, 0.15, size=n)
    drift = config.drift_sd * run_scale * drift_course[:, None] * drift_loading[None, :]

    # Cardiac/respiratory-like sinusoids become aliased at common fMRI TRs.
    time_s = np.arange(t, dtype=np.float64) * config.tr
    respiratory_hz = rng.uniform(0.22, 0.34)
    cardiac_hz = rng.uniform(0.85, 1.15)
    phys_course = (
        np.sin(2.0 * np.pi * respiratory_hz * time_s + rng.uniform(0, 2 * np.pi))
        + 0.45 * np.sin(2.0 * np.pi * cardiac_hz * time_s + rng.uniform(0, 2 * np.pi))
    )
    phys_course = _standardize(phys_course)
    phys_loading = rng.normal(0.8, 0.18, size=n) * (
        1.15 - 0.25 * anatomy.hierarchy
    )
    physiological = (
        config.physiological_sd * run_scale * phys_course[:, None] * phys_loading[None, :]
    )

    # Sparse motion-like spikes with a global and spatially patterned component.
    motion = np.zeros((t, n), dtype=np.float64)
    spike_mask = rng.random(t) < config.motion_spike_probability
    for frame in np.where(spike_mask)[0]:
        center = rng.integers(0, n)
        spatial_pattern = np.exp(-distances[center] / 28.0)
        global_pattern = rng.normal(0.55, 0.12)
        amplitude = rng.normal(0.0, config.motion_spike_sd * run_scale)
        motion[frame] += amplitude * (global_pattern + spatial_pattern)
        if frame + 1 < t:
            motion[frame + 1] += 0.35 * motion[frame]

    return NoiseComponents(
        temporal_ar=temporal_ar.astype(np.float32),
        spatial_measurement=spatial_measurement.astype(np.float32),
        global_signal=global_signal.astype(np.float32),
        drift=drift.astype(np.float32),
        physiological=physiological.astype(np.float32),
        motion=motion.astype(np.float32),
    )


def add_measurement_noise(
    signal: np.ndarray,
    components: NoiseComponents,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != components.total.shape:
        raise ValueError(
            f"Signal shape {signal.shape} does not match noise shape {components.total.shape}"
        )
    return (signal + components.total).astype(np.float32)

"""Continuous-time latent neural dynamics for synthetic fMRI generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anatomy import SyntheticAnatomy
from .config import SyntheticFMRIConfig
from .network import GroundTruthNetwork


@dataclass(slots=True)
class NeuralGroundTruth:
    """Node-specific mechanistic parameters known to the generator."""

    tau_seconds: np.ndarray
    intrinsic_drive: np.ndarray
    stimulus_gain: np.ndarray
    hierarchy: np.ndarray
    spatial_gradient: np.ndarray
    layer_index: np.ndarray


def build_neural_ground_truth(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    seed_offset: int = 0,
) -> NeuralGroundTruth:
    """Create known regional time constants, signed drives, and input gains."""
    rng = np.random.default_rng(config.seed + 2027 + seed_offset)

    hierarchy_component = anatomy.hierarchy
    layer_component = anatomy.layer_index / max(1, len(anatomy.layer_names) - 1)
    tau_normalized = 0.72 * hierarchy_component + 0.18 * (1.0 - layer_component)
    tau_normalized += 0.10 * anatomy.spatial_gradient
    tau_normalized += rng.normal(0.0, 0.025, size=anatomy.num_nodes)
    tau_normalized = np.clip(tau_normalized, 0.0, 1.0)
    tau = config.tau_min_s + tau_normalized * (config.tau_max_s - config.tau_min_s)

    # Signed intrinsic drive combines a posterior-anterior gradient, hierarchy,
    # and laminar offsets. It deliberately contains positive and negative nodes.
    layer_offsets = np.choose(
        anatomy.layer_index,
        [-0.10, 0.08, 0.02],
        mode="clip",
    )
    intrinsic = (
        0.34 * (0.5 - anatomy.hierarchy)
        + 0.20 * (anatomy.spatial_gradient - 0.5)
        + layer_offsets
        + rng.normal(0.0, config.intrinsic_drive_sd, size=anatomy.num_nodes)
    )
    intrinsic -= intrinsic.mean()
    intrinsic = np.clip(intrinsic, -0.75, 0.75)

    stimulus_gain = np.zeros(anatomy.num_nodes, dtype=np.float64)
    stimulus_gain[anatomy.sensory_nodes] = 1.0
    # Small spillover to superficial/deep nodes in the same sensory columns.
    sensory_columns = set(anatomy.column_ids[anatomy.sensory_nodes].tolist())
    for node in range(anatomy.num_nodes):
        if anatomy.column_ids[node] in sensory_columns and not anatomy.sensory_nodes[node]:
            stimulus_gain[node] = 0.32 if anatomy.layer_index[node] == 2 else 0.20

    return NeuralGroundTruth(
        tau_seconds=tau.astype(np.float64),
        intrinsic_drive=intrinsic.astype(np.float64),
        stimulus_gain=stimulus_gain,
        hierarchy=anatomy.hierarchy.copy(),
        spatial_gradient=anatomy.spatial_gradient.copy(),
        layer_index=anatomy.layer_index.copy(),
    )


def make_block_stimulus(
    config: SyntheticFMRIConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    """Generate a jittered block stimulus at the neural integration rate."""
    stimulus = np.zeros(config.neural_steps, dtype=np.float64)
    events: list[dict[str, float | str]] = []
    onset = config.first_block_onset_s
    while onset + config.block_duration_s < config.duration_s - config.tr:
        jittered = max(0.0, onset + float(rng.uniform(-config.stimulus_jitter_s, config.stimulus_jitter_s)))
        start = int(round(jittered / config.neural_dt))
        stop = min(
            config.neural_steps,
            start + int(round(config.block_duration_s / config.neural_dt)),
        )
        stimulus[start:stop] = config.stimulus_amplitude
        events.append(
            {
                "onset": float(jittered),
                "duration": float(config.block_duration_s),
                "amplitude": float(config.stimulus_amplitude),
                "trial_type": "hand_movement",
            }
        )
        onset += config.block_duration_s + config.inter_block_interval_s

    # Smooth on/off transitions slightly to avoid a numerically sharp neural input.
    transition_steps = max(1, int(round(0.30 / config.neural_dt)))
    kernel = np.ones(transition_steps, dtype=np.float64) / transition_steps
    stimulus = np.convolve(stimulus, kernel, mode="same")
    return stimulus, events


def _evaluate_edge_nonlinearity(
    values: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    basis = np.exp(-0.5 * ((values[:, None] - centers) / widths) ** 2)
    return np.sum(basis * coefficients, axis=1)


def simulate_neural_dynamics(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    network: GroundTruthNetwork,
    parameters: NeuralGroundTruth,
    stimulus: np.ndarray,
    *,
    seed: int,
    tau_scale: np.ndarray | None = None,
    connectivity_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Simulate delayed nonlinear multi-channel continuous-time activity.

    Returns
    -------
    neural_activity:
        Array with shape ``(neural_steps, nodes)``.
    diagnostics:
        Per-step aggregate channel drives useful for debugging and validation.
    """
    rng = np.random.default_rng(seed)
    n = anatomy.num_nodes
    steps = config.neural_steps
    if stimulus.shape != (steps,):
        raise ValueError(f"stimulus must have shape {(steps,)}, got {stimulus.shape}")

    tau = parameters.tau_seconds.copy()
    if tau_scale is not None:
        tau *= np.asarray(tau_scale, dtype=np.float64)
    tau = np.clip(tau, config.tau_min_s * 0.65, config.tau_max_s * 1.45)

    edge_weight = network.signed_weight.copy()
    if connectivity_scale is not None:
        edge_weight *= np.asarray(connectivity_scale, dtype=np.float64)

    activity = np.zeros((steps, n), dtype=np.float64)
    activity[0] = rng.normal(0.0, 0.03, size=n)
    driver_trace = np.zeros((steps, n), dtype=np.float32)
    suppressive_trace = np.zeros((steps, n), dtype=np.float32)
    modulatory_trace = np.zeros((steps, n), dtype=np.float32)

    src = network.source
    dst = network.target
    ch = network.channel
    delay = network.delay_steps

    # Weakly spatially correlated neural innovations.
    distances = np.linalg.norm(
        anatomy.coordinates_mm[:, None, :] - anatomy.coordinates_mm[None, :, :], axis=-1
    )
    covariance = np.exp(-distances / max(10.0, 0.75 * config.spatial_noise_scale_mm))
    covariance += np.eye(n) * 1e-5
    chol = np.linalg.cholesky(covariance)

    for t in range(1, steps):
        delayed_indices = np.maximum(0, t - delay)
        delayed_values = activity[delayed_indices, src]
        nonlinear = _evaluate_edge_nonlinearity(
            delayed_values,
            network.rbf_centers,
            network.rbf_widths,
            network.rbf_coefficients,
        )
        edge_messages = edge_weight * (0.35 * delayed_values + 0.65 * nonlinear)

        driver = np.zeros(n, dtype=np.float64)
        suppressive = np.zeros(n, dtype=np.float64)
        modulation = np.zeros(n, dtype=np.float64)
        np.add.at(driver, dst[ch == 0], edge_messages[ch == 0])
        np.add.at(suppressive, dst[ch == 1], edge_messages[ch == 1])
        np.add.at(modulation, dst[ch == 2], edge_messages[ch == 2])

        recurrent = (driver + suppressive) * (1.0 + 0.35 * np.tanh(modulation))
        sensory_input = parameters.stimulus_gain * stimulus[t]
        neural_noise = config.neural_noise_sd * (chol @ rng.normal(size=n))
        total_drive = parameters.intrinsic_drive + recurrent + sensory_input + neural_noise
        equilibrium = np.tanh(total_drive)
        activity[t] = activity[t - 1] + (config.neural_dt / tau) * (
            equilibrium - activity[t - 1]
        )
        activity[t] = np.clip(activity[t], -2.5, 2.5)

        driver_trace[t] = driver
        suppressive_trace[t] = suppressive
        modulatory_trace[t] = modulation

    diagnostics = {
        "driver_like_drive": driver_trace,
        "suppressive_like_drive": suppressive_trace,
        "gain_modulatory_like_drive": modulatory_trace,
    }
    return activity.astype(np.float32), diagnostics

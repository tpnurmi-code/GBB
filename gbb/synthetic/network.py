"""Ground-truth sparse directed network with FastKAN-like nonlinear edges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from .anatomy import SyntheticAnatomy
from .config import SyntheticFMRIConfig


@dataclass(slots=True)
class GroundTruthNetwork:
    """Edge-list representation of the mechanistic ground-truth network."""

    source: np.ndarray
    target: np.ndarray
    channel: np.ndarray
    channel_names: tuple[str, ...]
    signed_weight: np.ndarray
    delay_steps: np.ndarray
    delay_seconds: np.ndarray
    velocity_mm_s: np.ndarray
    distance_mm: np.ndarray
    rbf_centers: np.ndarray
    rbf_widths: np.ndarray
    rbf_coefficients: np.ndarray
    layer_gain: np.ndarray
    adjacency_by_channel: np.ndarray
    weight_by_channel: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.source.size)

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency_by_channel.shape[-1])

    @property
    def density(self) -> float:
        possible = self.num_channels * self.num_nodes * max(1, self.num_nodes - 1)
        return float(self.num_edges / possible)


def _layer_pair_gain(source_layer: int, target_layer: int, channel: int) -> float:
    """Encode known laminar input/output preferences.

    Layer order is deep=0, middle=1, superficial=2. The values are hypotheses
    for synthetic recovery, not claims about a specific empirical cortex.
    """
    if channel == 0:  # driver-like
        if source_layer == 2 and target_layer == 1:
            return 1.35  # superficial feed-forward output to middle input
        if source_layer == 1 and target_layer == 2:
            return 1.15
        if source_layer == target_layer:
            return 0.95
        return 0.75
    if channel == 1:  # suppressive-like
        if source_layer == target_layer:
            return 1.25
        if source_layer == 0 and target_layer in {1, 2}:
            return 1.10
        return 0.85
    # gain-modulatory-like: deep-to-superficial and feedback are emphasized.
    if source_layer == 0 and target_layer == 2:
        return 1.35
    if source_layer == 2 and target_layer == 0:
        return 1.10
    return 0.80


def _edge_probability(
    distance: float,
    same_column: bool,
    same_network: bool,
    hierarchy_difference: float,
    channel: int,
    target_density: float,
) -> float:
    local = np.exp(-distance / 35.0)
    network_bonus = 1.6 if same_network else 0.7
    column_bonus = 3.2 if same_column else 1.0
    if channel == 0:
        direction_bonus = 1.5 if hierarchy_difference > 0 else 0.75
    elif channel == 1:
        direction_bonus = 1.25 if abs(hierarchy_difference) < 0.25 else 0.9
    else:
        direction_bonus = 1.45 if hierarchy_difference < 0 else 0.65
    probability = target_density * local * network_bonus * column_bonus * direction_bonus * 4.0
    return float(np.clip(probability, 0.002, 0.70))


def _make_rbf_parameters(
    rng: np.random.Generator,
    num_basis: int,
    channel: int,
    signed_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.linspace(-2.5, 2.5, num_basis, dtype=np.float64)
    centers += rng.normal(0.0, 0.10, size=num_basis)
    widths = rng.uniform(0.55, 1.15, size=num_basis)

    # Distinct nonlinear edge families. Coefficients are edge-specific while
    # retaining a channel-level semantic prior.
    if channel == 0:
        template = np.tanh(centers) + 0.20 * np.sin(1.5 * centers)
    elif channel == 1:
        template = -np.tanh(centers) + 0.15 * np.cos(2.0 * centers)
    else:
        template = 0.65 * np.tanh(0.8 * centers) + 0.25 * np.sin(centers)
    coefficients = np.sign(signed_weight) * template
    coefficients += rng.normal(0.0, 0.08, size=num_basis)
    coefficients *= rng.uniform(0.65, 1.15)
    return centers, widths, coefficients


def build_ground_truth_network(
    config: SyntheticFMRIConfig,
    anatomy: SyntheticAnatomy,
    seed_offset: int = 0,
) -> GroundTruthNetwork:
    """Construct a sparse, directed, delayed, multi-channel nonlinear graph."""
    rng = np.random.default_rng(config.seed + 1009 + seed_offset)
    n = anatomy.num_nodes
    c = len(config.channel_names)
    distance_matrix = cdist(anatomy.coordinates_mm, anatomy.coordinates_mm)

    source: list[int] = []
    target: list[int] = []
    channel: list[int] = []
    weights: list[float] = []
    delays_s: list[float] = []
    velocities: list[float] = []
    distances: list[float] = []
    layer_gains: list[float] = []
    centers_all: list[np.ndarray] = []
    widths_all: list[np.ndarray] = []
    coeffs_all: list[np.ndarray] = []

    adjacency = np.zeros((c, n, n), dtype=np.float64)
    weight_matrix = np.zeros((c, n, n), dtype=np.float64)

    for ch in range(c):
        for src in range(n):
            for dst in range(n):
                if src == dst:
                    continue
                distance = float(distance_matrix[src, dst])
                same_column = anatomy.column_ids[src] == anatomy.column_ids[dst]
                same_network = anatomy.network_ids[src] == anatomy.network_ids[dst]
                hierarchy_difference = float(anatomy.hierarchy[dst] - anatomy.hierarchy[src])
                probability = _edge_probability(
                    distance=distance,
                    same_column=bool(same_column),
                    same_network=bool(same_network),
                    hierarchy_difference=hierarchy_difference,
                    channel=ch,
                    target_density=config.target_edge_density,
                )
                if rng.random() >= probability:
                    continue

                # Signed magnitudes are kept small enough for stable continuous
                # dynamics. The suppressive channel is negative by construction.
                base_magnitude = rng.lognormal(mean=-2.15, sigma=0.35)
                if ch == 0:
                    sign = 1.0
                elif ch == 1:
                    sign = -1.0
                else:
                    sign = 1.0 if rng.random() < 0.75 else -1.0
                hierarchy_scale = 1.0 + 0.25 * abs(hierarchy_difference)
                layer_gain = _layer_pair_gain(
                    int(anatomy.layer_index[src]), int(anatomy.layer_index[dst]), ch
                )
                signed_weight = sign * base_magnitude * hierarchy_scale * layer_gain

                velocity = float(
                    rng.uniform(config.min_velocity_mm_s, config.max_velocity_mm_s)
                )
                delay = min(config.max_delay_s, max(config.neural_dt, distance / velocity))
                delay += float(rng.uniform(0.0, 0.5 * config.neural_dt))
                delay = min(config.max_delay_s, delay)

                centers, widths, coefficients = _make_rbf_parameters(
                    rng, config.num_rbf_basis, ch, signed_weight
                )
                source.append(src)
                target.append(dst)
                channel.append(ch)
                weights.append(signed_weight)
                delays_s.append(delay)
                velocities.append(velocity)
                distances.append(distance)
                layer_gains.append(layer_gain)
                centers_all.append(centers)
                widths_all.append(widths)
                coeffs_all.append(coefficients)
                adjacency[ch, dst, src] = 1.0
                weight_matrix[ch, dst, src] = signed_weight

    # Guarantee that every node participates in at least one driver-like edge.
    driver_in = adjacency[0].sum(axis=1)
    for dst in np.where(driver_in == 0)[0]:
        candidates = np.argsort(distance_matrix[dst])
        src = int(next(index for index in candidates if index != dst))
        signed_weight = float(rng.uniform(0.06, 0.12))
        distance = float(distance_matrix[src, dst])
        velocity = float(rng.uniform(config.min_velocity_mm_s, config.max_velocity_mm_s))
        delay = min(config.max_delay_s, max(config.neural_dt, distance / velocity))
        centers, widths, coefficients = _make_rbf_parameters(
            rng, config.num_rbf_basis, 0, signed_weight
        )
        layer_gain = _layer_pair_gain(
            int(anatomy.layer_index[src]), int(anatomy.layer_index[dst]), 0
        )
        source.append(src)
        target.append(dst)
        channel.append(0)
        weights.append(signed_weight)
        delays_s.append(delay)
        velocities.append(velocity)
        distances.append(distance)
        layer_gains.append(layer_gain)
        centers_all.append(centers)
        widths_all.append(widths)
        coeffs_all.append(coefficients)
        adjacency[0, dst, src] = 1.0
        weight_matrix[0, dst, src] = signed_weight

    # Normalize incoming absolute weights per target to maintain stable dynamics.
    weight_array = np.asarray(weights, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.int64)
    channel_array = np.asarray(channel, dtype=np.int64)
    for dst in range(n):
        indices = np.where((target_array == dst) & (channel_array != 2))[0]
        total = np.sum(np.abs(weight_array[indices]))
        if total > 0.85:
            weight_array[indices] *= 0.85 / total
    weight_matrix.fill(0.0)
    for edge_index, (ch, dst, src) in enumerate(zip(channel_array, target_array, source)):
        weight_matrix[ch, dst, src] = weight_array[edge_index]

    delay_seconds = np.asarray(delays_s, dtype=np.float64)
    delay_steps = np.maximum(1, np.rint(delay_seconds / config.neural_dt).astype(np.int64))

    return GroundTruthNetwork(
        source=np.asarray(source, dtype=np.int64),
        target=target_array,
        channel=channel_array,
        channel_names=tuple(config.channel_names),
        signed_weight=weight_array,
        delay_steps=delay_steps,
        delay_seconds=delay_seconds,
        velocity_mm_s=np.asarray(velocities, dtype=np.float64),
        distance_mm=np.asarray(distances, dtype=np.float64),
        rbf_centers=np.asarray(centers_all, dtype=np.float64),
        rbf_widths=np.asarray(widths_all, dtype=np.float64),
        rbf_coefficients=np.asarray(coeffs_all, dtype=np.float64),
        layer_gain=np.asarray(layer_gains, dtype=np.float64),
        adjacency_by_channel=adjacency,
        weight_by_channel=weight_matrix,
    )

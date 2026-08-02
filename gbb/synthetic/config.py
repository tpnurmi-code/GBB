"""Configuration objects for mechanistic synthetic fMRI generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ResponseKind = Literal["bold", "cbv"]


@dataclass(slots=True)
class SyntheticFMRIConfig:
    """Configuration for a privacy-safe mechanistic fMRI benchmark.

    The defaults are intentionally modest enough for a laptop while preserving
    all mechanistic components. Increase ``num_columns``, ``n_timepoints`` and
    ``num_subjects`` for a larger public benchmark.
    """

    output_dir: Path = Path("synthetic_gbb_data")
    seed: int = 42
    overwrite: bool = False

    # Dataset size and acquisition
    num_subjects: int = 4
    num_runs: int = 2
    num_columns: int = 12
    cortical_layers: tuple[str, ...] = ("deep", "middle", "superficial")
    n_timepoints: int = 160
    tr: float = 2.5
    neural_dt: float = 0.10
    response_kind: ResponseKind = "bold"
    raw_stimulus_sampling_rate: float = 500.0

    # Synthetic image geometry. Coordinates are pseudo-MNI and contain no anatomy.
    volume_shape: tuple[int, int, int] = (64, 64, 56)
    voxel_size_mm: float = 3.0
    parcel_radius_voxels: int = 2
    voxel_noise_sd: float = 0.20

    # Event design
    block_duration_s: float = 12.5
    inter_block_interval_s: float = 25.0
    first_block_onset_s: float = 20.0
    stimulus_amplitude: float = 1.0
    stimulus_jitter_s: float = 2.0

    # Mechanistic parameter ranges
    tau_min_s: float = 0.75
    tau_max_s: float = 4.50
    intrinsic_drive_sd: float = 0.22
    target_edge_density: float = 0.12
    num_rbf_basis: int = 5
    max_delay_s: float = 0.40
    min_velocity_mm_s: float = 250.0
    max_velocity_mm_s: float = 1800.0

    # Neural and measurement noise
    neural_noise_sd: float = 0.08
    temporal_ar: float = 0.55
    spatial_noise_scale_mm: float = 32.0
    measurement_noise_sd: float = 0.45
    global_signal_sd: float = 0.18
    drift_sd: float = 0.20
    physiological_sd: float = 0.08
    motion_spike_probability: float = 0.015
    motion_spike_sd: float = 1.75

    # Subject heterogeneity
    subject_tau_sd_fraction: float = 0.06
    subject_connectivity_sd_fraction: float = 0.05
    subject_hrf_sd_fraction: float = 0.08
    run_noise_scale_sd_fraction: float = 0.10

    # Output controls
    save_dense_stimulus_mat: bool = True
    save_events_tsv: bool = True
    save_node_timeseries_npz: bool = True
    save_neural_ground_truth: bool = True
    compression: bool = True

    channel_names: tuple[str, ...] = (
        "driver_like",
        "suppressive_like",
        "gain_modulatory_like",
    )
    layer_order: dict[str, int] = field(
        default_factory=lambda: {"deep": 0, "middle": 1, "superficial": 2}
    )

    @property
    def num_nodes(self) -> int:
        return self.num_columns * len(self.cortical_layers)

    @property
    def duration_s(self) -> float:
        return self.n_timepoints * self.tr

    @property
    def neural_steps(self) -> int:
        return int(round(self.duration_s / self.neural_dt))

    @property
    def samples_per_tr(self) -> int:
        return int(round(self.raw_stimulus_sampling_rate * self.tr))

    def validate(self) -> None:
        """Raise ``ValueError`` for contradictory or unstable settings."""
        if self.num_subjects < 1 or self.num_runs < 1:
            raise ValueError("num_subjects and num_runs must be at least one")
        if self.num_columns < 4:
            raise ValueError("num_columns must be at least four")
        if len(self.cortical_layers) < 2:
            raise ValueError("At least two cortical layers are required")
        if self.n_timepoints < 40:
            raise ValueError("n_timepoints must be at least 40")
        if self.tr <= 0 or self.neural_dt <= 0:
            raise ValueError("tr and neural_dt must be positive")
        if self.neural_dt > self.tr:
            raise ValueError("neural_dt must not exceed the fMRI TR")
        if self.tau_min_s <= self.neural_dt:
            raise ValueError("tau_min_s must exceed neural_dt for stable Euler integration")
        if self.tau_max_s <= self.tau_min_s:
            raise ValueError("tau_max_s must exceed tau_min_s")
        if not 0.0 < self.target_edge_density < 0.5:
            raise ValueError("target_edge_density must lie between 0 and 0.5")
        if self.num_rbf_basis < 3:
            raise ValueError("num_rbf_basis must be at least three")
        if self.response_kind not in {"bold", "cbv"}:
            raise ValueError("response_kind must be 'bold' or 'cbv'")
        if self.parcel_radius_voxels < 1:
            raise ValueError("parcel_radius_voxels must be positive")
        if self.samples_per_tr < 1:
            raise ValueError("raw_stimulus_sampling_rate * tr must be at least one")
        if set(self.cortical_layers) != set(self.layer_order):
            raise ValueError("layer_order must define every cortical layer exactly once")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload

    @classmethod
    def quick(cls, output_dir: str | Path = "synthetic_gbb_quick") -> "SyntheticFMRIConfig":
        """Return a small configuration suitable for smoke tests and CI."""
        return cls(
            output_dir=Path(output_dir),
            num_subjects=2,
            num_runs=1,
            num_columns=6,
            n_timepoints=64,
            neural_dt=0.125,
            volume_shape=(40, 40, 36),
            parcel_radius_voxels=1,
        )

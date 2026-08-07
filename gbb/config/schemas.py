"""Typed configuration schema for the Glass-Box Brain project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MissingStimulusPolicy = Literal["ERROR", "WARN", "ZEROS"]
StimulusShapePolicy = Literal["ERROR", "WARN", "COERCE"]
SensorySelectionPolicy = Literal["STRICT", "WARN", "FALLBACK"]
ColumnarMaskPolicy = Literal["DISABLED", "OPTIONAL", "ERROR", "REGION_FALLBACK",]

ModelType = Literal["H1", "H2", "H3", "H4", "H5", "H6"]
StimulusMode = Literal["EVENTS", "DENSE", "NONE"]
StimulusInjectionMode = Literal[
    "REGION_NAME",
    "COORDINATES",
    "COORDS_REGION_INTERSECTION",
]
ResponseFunction = Literal["hrf", "cbv", "uniform"]
CfcPhysicsMode = Literal["standard", "biophysical"]
TauBoundMode = Literal["sigmoid", "softplus"]


@dataclass
class ExperimentConfig:
    seed: int = 42
    super_computer_training: bool = False
    demonstration: bool = True
    model_type: ModelType = "H1"


@dataclass
class PathConfig:
    data_dir: Path
    results_dir: Path
    log_dir: Path
    checkpoint_path: Path
    mask_file: Path
    roi_mask_file: Path
    columnar_mask_file: Path


@dataclass
class MaskerConfig:
    detrend: bool = False
    low_pass: float | None = None
    high_pass: float | None = None
    columnar_mask_policy: ColumnarMaskPolicy = "DISABLED"


@dataclass
class StimulusConfig:
    
    mode: StimulusMode = "DENSE"
    injection_mode: StimulusInjectionMode = "COORDS_REGION_INTERSECTION"
    sensory_regions: list[str] = field(
        default_factory=lambda: ["Postcentral", "SomMot", "Somatosensory", "Par", "S1"]
    )
    excluded_regions: list[str] = field(
        default_factory=lambda: ["Precentral", "M1", "Frontal", "Premotor"]
    )
    mni_coords: tuple[float, float, float] = (-42.0, -25.0, 55.0)
    radius_mm: float = 25.0
    response_function: ResponseFunction = "hrf"
    convolve_stimulus: bool = False
    allow_stimulus_preconv_with_hemo: bool = False
    dense_stimulus_ext: str = "_stim.mat"
    input_channels: int = 1
    raw_sampling_rate: float = 500.0
    dense_stimulus_key: str | None = "stimulus"
    required_trial_type: str | None = "hand_movement"

    missing_stimulus_policy: MissingStimulusPolicy = "ERROR"
    stimulus_shape_policy: StimulusShapePolicy = "ERROR"
    sensory_selection_policy: SensorySelectionPolicy = "STRICT"

    require_nonzero_stimulus: bool = True
    allow_all_nodes_stimulus: bool = False

@dataclass
class HemodynamicConfig:
    use_head: bool = True
    kernel_size: int = 5
    init: ResponseFunction = "hrf"


@dataclass
class DataLoaderConfig:
    train_set_size: float = 0.8
    num_runs: int = 2
    train_workers: int = 4
    test_workers: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 20
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_amp: bool = True
    tr: float = 2.5
    accel_axes: int = 1
    window_size: int = 30
    prediction_horizon: int = 5
    num_nodes_expected: int = 305
    adjacency_sigma: float = 40.0
    prune_start_frac: float = 0.50
    prune_every_epochs: int = 5
    min_epochs_before_prune: int = 30
    disable_hard_prune_in_demo: bool = True
    validate_every_epochs: int = 5
    always_validate_final: bool = True
    maskout_timeseries: bool = True
    maskout_length: int = 5
    dropout: float = 0.3


@dataclass
class FeatureExtractorConfig:
    hidden_dim: int = 64
    dropout: float = 0.1
    cnn_stride: int = 1
    cnn_layers: int = 2
    rnn_layers: int = 2
    transformer_layers: int = 2
    mlp_layers: int = 3
    local_attention_radius: int = 4


@dataclass
class GraphConfig:
    kan_basis_functions: int = 5
    kan_layers: int = 1
    kan_heads: int = 3
    spline_coeff: float = 0.05
    adjacency_prior_strength: float = 0.25
    enable_signed_head_messages: bool = True
    head_signs: tuple[float, ...] = (1.0, -1.0, 0.5)
    graph_mix: float = 0.70


@dataclass
class CfcConfig:
    backbone_layers: int = 2
    backbone_units: int = 128
    physics_mode: CfcPhysicsMode = "biophysical"
    use_seq_cfc: bool = True
    fmri_step_proj_dim: int | None = None
    tau_min_phys: float = 1.5
    tau_max_phys: float = 12.0
    tau_bound_mode: TauBoundMode = "sigmoid"
    gate_target: float = 0.50
    gate_min: float = 0.05
    gate_max: float = 0.95
    f_activity_max: float = 3.0
    drive_max_abs: float = 3.0


@dataclass
class TemporalStimulusEncoderConfig:
    use_temporal_stim_encoder: bool = True
    kernel_size: int = 5
    dropout_prob: float = 0.20
    dropout_start_frac: float = 0.30


@dataclass
class LossConfig:
    lambda_accuracy: float = 1.0
    lambda_correlation: float = 10.0
    lambda_correlation_final: float = 3.0
    reg_phase_start_frac: float = 0.60
    lambda_var: float = 1.25
    lambda_derivative: float = 0.5
    lambda_metabolic: float = 1e-5
    lambda_sparsity: float = 1e-6
    lambda_wiring: float = 1e-5
    lambda_group_lasso: float = 2e-4
    lambda_head_group: float = 0.0
    lambda_smoothness: float = 5.0
    lambda_orth_loss: float = 0.05
    lambda_longterm: float = 0.05
    longterm_threshold_moderate: float = 8.0
    longterm_threshold_extreme: float = 20.0
    lambda_tau_var: float = 0.05
    lambda_tau_saturation: float = 0.20
    lambda_tau_diversity: float = 0.25
    lambda_tau_smooth: float = 0.05
    lambda_tau_hier: float = 0.05
    lambda_rank_loss: float = 0.05
    tau_lognormal_median: float = 5.0
    tau_lognormal_sigma: float = 0.5
    lambda_head_sign: float = 0.0
    head_sign_warmup_frac: float = 0.60
    head_mod_gain_penalty: float = 0.25
    lambda_cfc_pop: float = 0.01
    cfc_node_bias_smooth_w: float = 0.10
    cfc_node_bias_l2_w: float = 0.01
    cfc_gate_reg_w: float = 0.10
    cfc_delta_energy_w: float = 0.05
    cfc_f_activity_w: float = 0.05
    cfc_drive_balance_w: float = 0.05
    cfc_drive_bound_w: float = 0.05


@dataclass
class SmoothnessConfig:
    sigma_mm: float = 40.0
    k_neighbors: int = 15


@dataclass
class OptimizationConfig:
    n_trials: int = 50
    trials_per_eval: int = 3
    baseline_mse: float = 0.80
    max_perf_drop: float = 0.05


@dataclass
class Config:
    experiment: ExperimentConfig
    paths: PathConfig
    masker: MaskerConfig
    stimulus: StimulusConfig
    hemodynamics: HemodynamicConfig
    dataloader: DataLoaderConfig
    training: TrainingConfig
    feature_extractor: FeatureExtractorConfig
    graph: GraphConfig
    cfc: CfcConfig
    temporal_stimulus: TemporalStimulusEncoderConfig
    losses: LossConfig
    smoothness: SmoothnessConfig
    optimization: OptimizationConfig

    def finalize(self) -> "Config":
        if self.cfc.fmri_step_proj_dim is None:
            self.cfc.fmri_step_proj_dim = self.cfc.backbone_units
        return self

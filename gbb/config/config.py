"""Flat compatibility API used by the current runnable implementation.

New dataclass-based code may import ``gbb.config.cfg``. Until every call site has
been migrated, runtime modules should consistently use ``from gbb.config import
config`` and refer to the constants below.
"""

from __future__ import annotations

import torch

from gbb.config import cfg

SEED = cfg.experiment.seed
SUPER_COMPUTER_TRAINING = cfg.experiment.super_computer_training
DEMONSTRATION = cfg.experiment.demonstration
MODEL_TYPE = cfg.experiment.model_type

DATA_DIR = str(cfg.paths.data_dir)
RESULTS_DIR = str(cfg.paths.results_dir)
LOG_DIR = str(cfg.paths.log_dir)
CHECKPOINT_PATH = str(cfg.paths.checkpoint_path)
MASK_FILE = str(cfg.paths.mask_file)
ROI_MASK_FILE = str(cfg.paths.roi_mask_file)
COLUMNAR_MASK_FILE = str(cfg.paths.columnar_mask_file)

DETREND = cfg.masker.detrend
LOW_PASS = cfg.masker.low_pass
HIGH_PASS = cfg.masker.high_pass
COLUMNAR_MASK_POLICY = cfg.masker.columnar_mask_policy

REQUIRED_TRIAL_TYPE = cfg.stimulus.required_trial_type

MISSING_STIMULUS_POLICY = cfg.stimulus.missing_stimulus_policy
STIMULUS_SHAPE_POLICY = cfg.stimulus.stimulus_shape_policy
SENSORY_SELECTION_POLICY = cfg.stimulus.sensory_selection_policy

REQUIRE_NONZERO_STIMULUS = cfg.stimulus.require_nonzero_stimulus
ALLOW_ALL_NODES_STIMULUS = cfg.stimulus.allow_all_nodes_stimulus

STIMULUS_MODE = cfg.stimulus.mode
STIMULUS_INJECTION_MODE = cfg.stimulus.injection_mode
SENSORY_REGIONS = list(cfg.stimulus.sensory_regions)
EXCLUDED_REGIONS = list(cfg.stimulus.excluded_regions)
STIMULUS_MNI_COORDS = list(cfg.stimulus.mni_coords)
STIMULUS_RADIUS_MM = cfg.stimulus.radius_mm
RESPONSE_FUNCTION = cfg.stimulus.response_function
CONVOLVE_STIMULUS = cfg.stimulus.convolve_stimulus
ALLOW_STIMULUS_PRECONV_WITH_HEMO = cfg.stimulus.allow_stimulus_preconv_with_hemo
DENSE_STIMULUS_EXT = cfg.stimulus.dense_stimulus_ext
DENSE_STIMULUS_KEY = cfg.stimulus.dense_stimulus_key
STIMULUS_INPUT_CHANNELS = cfg.stimulus.input_channels
RAW_SAMPLING_RATE = cfg.stimulus.raw_sampling_rate

USE_HEMODYNAMIC_HEAD = cfg.hemodynamics.use_head
HEMO_KERNEL_SIZE = cfg.hemodynamics.kernel_size
HEMO_INIT = cfg.hemodynamics.init

TRAIN_SET_SIZE = cfg.dataloader.train_set_size
NUM_RUNS = cfg.dataloader.num_runs
TRAIN_LOADER_WORKERS = cfg.dataloader.train_workers
TEST_LOADER_WORKERS = cfg.dataloader.test_workers

BATCH_SIZE = cfg.training.batch_size
NUM_EPOCHS = cfg.training.num_epochs
PATIENCE = cfg.training.patience
LEARNING_RATE = cfg.training.learning_rate
WEIGHT_DECAY = cfg.training.weight_decay
GRAD_CLIP = cfg.training.grad_clip
USE_AMP = cfg.training.use_amp
TR = cfg.training.tr
ACCEL_AXES = cfg.training.accel_axes
WINDOW_SIZE = cfg.training.window_size
PREDICTION_HORIZON = cfg.training.prediction_horizon
NUM_NODES_EXPECTED = cfg.training.num_nodes_expected
ADJACENCY_SIGMA = cfg.training.adjacency_sigma
PRUNE_START_FRAC = cfg.training.prune_start_frac
PRUNE_EVERY_EPOCHS = cfg.training.prune_every_epochs
MIN_EPOCHS_BEFORE_PRUNE = cfg.training.min_epochs_before_prune
DISABLE_HARD_PRUNE_IN_DEMO = cfg.training.disable_hard_prune_in_demo
VALIDATE_EVERY_EPOCHS = cfg.training.validate_every_epochs
ALWAYS_VALIDATE_FINAL = cfg.training.always_validate_final
MASKOUT_TIMESERIES = cfg.training.maskout_timeseries
MASKOUT_LENGTH = cfg.training.maskout_length
DROPOUT = cfg.training.dropout

FEAT_EXT_HIDDEN = cfg.feature_extractor.hidden_dim
FEAT_EXT_DROPOUT = cfg.feature_extractor.dropout
CNN_STRIDE = cfg.feature_extractor.cnn_stride
CNN_LAYERS = cfg.feature_extractor.cnn_layers
RNN_LAYERS = cfg.feature_extractor.rnn_layers
TRANSFORMER_LAYERS = cfg.feature_extractor.transformer_layers
MLP_LAYERS = cfg.feature_extractor.mlp_layers
LOCAL_ATTENTION_RADIUS = cfg.feature_extractor.local_attention_radius

KAN_BASIS_FUNCTIONS = cfg.graph.kan_basis_functions
KAN_LAYERS = cfg.graph.kan_layers
KAN_HEADS = cfg.graph.kan_heads
SPLINECOEFF = cfg.graph.spline_coeff
ADJ_PRIOR_STRENGTH = cfg.graph.adjacency_prior_strength
ENABLE_SIGNED_HEAD_MESSAGES = cfg.graph.enable_signed_head_messages
GRAPH_HEAD_SIGNS = list(cfg.graph.head_signs)
GRAPH_MIX = cfg.graph.graph_mix

CFC_BACKBONE_LAYERS = cfg.cfc.backbone_layers
CFC_BACKBONE_UNITS = cfg.cfc.backbone_units
CFC_PHYSICS_MODE = cfg.cfc.physics_mode
USE_SEQ_CFC = cfg.cfc.use_seq_cfc
FMRI_STEP_PROJ_DIM = cfg.cfc.fmri_step_proj_dim
TAU_MIN_PHYS = cfg.cfc.tau_min_phys
TAU_MAX_PHYS = cfg.cfc.tau_max_phys
TAU_BOUND_MODE = cfg.cfc.tau_bound_mode
CFC_GATE_TARGET = cfg.cfc.gate_target
CFC_GATE_MIN = cfg.cfc.gate_min
CFC_GATE_MAX = cfg.cfc.gate_max
CFC_F_ACTIVITY_MAX = cfg.cfc.f_activity_max
CFC_DRIVE_MAX_ABS = cfg.cfc.drive_max_abs

USE_TEMPORAL_STIM_ENCODER = cfg.temporal_stimulus.use_temporal_stim_encoder
STIM_ENCODER_KERNEL = cfg.temporal_stimulus.kernel_size
STIM_DROPOUT_PROB = cfg.temporal_stimulus.dropout_prob
STIM_DROPOUT_START_FRAC = cfg.temporal_stimulus.dropout_start_frac

LAMBDA_ACCURACY = cfg.losses.lambda_accuracy
LAMBDA_CORRELATION = cfg.losses.lambda_correlation
LAMBDA_CORRELATION_FINAL = cfg.losses.lambda_correlation_final
REG_PHASE_START_FRAC = cfg.losses.reg_phase_start_frac
LAMBDA_VAR = cfg.losses.lambda_var
LAMBDA_DERIVATIVE = cfg.losses.lambda_derivative
LAMBDA_METABOLIC = cfg.losses.lambda_metabolic
LAMBDA_SPARSITY = cfg.losses.lambda_sparsity
LAMBDA_WIRING = cfg.losses.lambda_wiring
LAMBDA_GROUP_LASSO = cfg.losses.lambda_group_lasso
LAMBDA_HEAD_GROUP = cfg.losses.lambda_head_group
LAMBDA_SMOOTHNESS = cfg.losses.lambda_smoothness
LAMBDA_ORTH_LOSS = cfg.losses.lambda_orth_loss
LAMBDA_LONGTERM = cfg.losses.lambda_longterm
LONGTERM_THRESHOLD_MODERATE = cfg.losses.longterm_threshold_moderate
LONGTERM_THRESHOLD_EXTREME = cfg.losses.longterm_threshold_extreme
LAMBDA_TAU_VAR = cfg.losses.lambda_tau_var
LAMBDA_TAU_SATURATION = cfg.losses.lambda_tau_saturation
LAMBDA_TAU_DIVERSITY = cfg.losses.lambda_tau_diversity
LAMBDA_TAU_SMOOTH = cfg.losses.lambda_tau_smooth
LAMBDA_TAU_HIER = cfg.losses.lambda_tau_hier
LAMBDA_RANK_LOSS = cfg.losses.lambda_rank_loss
TAU_LOGNORMAL_MEDIAN = cfg.losses.tau_lognormal_median
TAU_LOGNORMAL_SIGMA = cfg.losses.tau_lognormal_sigma
LAMBDA_HEAD_SIGN = cfg.losses.lambda_head_sign
HEAD_SIGN_WARMUP_FRAC = cfg.losses.head_sign_warmup_frac
HEAD_MOD_GAIN_PENALTY = cfg.losses.head_mod_gain_penalty
LAMBDA_CFC_POP = cfg.losses.lambda_cfc_pop
CFC_NODE_BIAS_SMOOTH_W = cfg.losses.cfc_node_bias_smooth_w
CFC_NODE_BIAS_L2_W = cfg.losses.cfc_node_bias_l2_w
CFC_GATE_REG_W = cfg.losses.cfc_gate_reg_w
CFC_DELTA_ENERGY_W = cfg.losses.cfc_delta_energy_w
CFC_F_ACTIVITY_W = cfg.losses.cfc_f_activity_w
CFC_DRIVE_BALANCE_W = cfg.losses.cfc_drive_balance_w
CFC_DRIVE_BOUND_W = cfg.losses.cfc_drive_bound_w

SMOOTHNESS_SIGMA_MM = cfg.smoothness.sigma_mm
SMOOTHNESS_K_NEIGHBORS = cfg.smoothness.k_neighbors

N_TRIALS = cfg.optimization.n_trials
TRIALS_PER_EVAL = cfg.optimization.trials_per_eval
BASELINE_MSE = cfg.optimization.baseline_mse
MAX_PERF_DROP = cfg.optimization.max_perf_drop
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import torch

# --- EXPERIMENT SETUP ---
SEED = 42
SUPER_COMPUTER_TRAINING = False 
DEMONSTRATION = True 
# MODEL_TYPE defaults to H3 but is overwritten by optimization.py
# Options: "H1", "H2", "H3", "H4", "H5", "H6"
MODEL_TYPE = "H1"  

# --- PATHS ---
DATA_DIR = "G:\\Projects\\AI\\data"
RESULTS_DIR = os.path.join(DATA_DIR, "results")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "feature_extractor_foundation.pth")
# 1. THE PHYSICS MASK (305 Nodes)
# Used by Dataset, Model, Hand Knob Stimulus, and Training
# 1. THE PHYSICS MASK (305 Nodes) - High Res
MASK_FILE = os.path.join(DATA_DIR, "group_roi_mask.nii") 

# 2. THE VISUALIZATION MASK (10 ROIs) - Low Res
# Used by utils.py to make diagrams readable
ROI_MASK_FILE = os.path.join(DATA_DIR, "group_roi_mask_10.nii")

#columns
COLUMNAR_MASK_FILE = os.path.join(DATA_DIR, "cortical_columns_7T.nii")

#NiftiMasker parameters
DETREND= False
LOW_PASS = None
HIGH_PASS = None

#Stimulus config
STIMULUS_MODE='DENSE'
# --- STIMULUS INJECTION CONFIGURATION ---
# Options: 
#   "REGION_NAME": (Old) Injects into every region containing the string (e.g., all "Postcentral")
#   "COORDINATES": (New) Injects into nodes within a radius of specific MNI coordinates
STIMULUS_INJECTION_MODE = "COORDS_REGION_INTERSECTION"

# MODE A: Region Name Configuration (e.g., for future Visual tasks use ["Occipital"])
SENSORY_REGIONS = ["Postcentral", "SomMot", "Somatosensory", "Par", "S1"]
EXCLUDED_REGIONS =  ["Precentral", "M1", "Frontal", "Premotor"]
# MODE B: Coordinate Configuration (Biologically Specific)
# Default: Left S1 Hand Knob (Omega Sign). Approx MNI: x=-42, y=-25, z=55
# For Right Hand stimulation, flip x to positive.
STIMULUS_MNI_COORDS = [-42, -25, 55] 
STIMULUS_RADIUS_MM = 25.0  # 15mm captures the hand area without bleeding into Face/Trunk

# --- STIMULUS CONFIGURATION ---
# Options: "EVENTS" (Classic .tsv with HRF convolution) or "DENSE" (Continuous .mat/.npy signals)
# --- CURRENT 3T BOLD DEFAULTS ---
RESPONSE_FUNCTION = "hrf"

# Prefer raw/neural drive when the model already has an observation head.
# Set True only for ablation.
CONVOLVE_STIMULUS = False
ALLOW_STIMULUS_PRECONV_WITH_HEMO = False

USE_HEMODYNAMIC_HEAD = True
HEMO_KERNEL_SIZE = 5
HEMO_INIT = "hrf"   # options: "hrf", "cbv", "uniform"

# File extension to look for when using DENSE mode
DENSE_STIMULUS_EXT = "_stim.mat" 

# Input Dimensionality
# 1 = Accelerometer/Audio (Time, 1)
# 2 = Spectrograms (Time, Freq)
# 3 = Video Frames (Time, H, W)
STIMULUS_INPUT_CHANNELS = 1  

# Sampling rate of the raw sensor data (e.g., 500Hz for Accel)
RAW_SAMPLING_RATE = 500.0

TRAIN_SET_SIZE=0.8

TRAIN_LOADER_WORKERS=4
TEST_LOADER_WORKERS=2

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 16
#NUM_EPOCHS = 200
NUM_EPOCHS = 20
PATIENCE = 15
LEARNING_RATE = 1e-3
TR = 2.5
ACCEL_AXES = 1  # Standard trigger is 1D (0 or 1)
WINDOW_SIZE = 30
PREDICTION_HORIZON = 5  # Number of future TRs predicted by the decoder (multi-step forecast)
NUM_NODES_EXPECTED = 305
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-5
ADJACENCY_SIGMA = 40.0 # Sigma controls how "far" a node's influence spreads (e.g., 10-20mm)
USE_AMP = True # --- MIXED PRECISION ---

# --- TRAINING SCHEDULE GUARDS ---
PRUNE_START_FRAC = 0.50
PRUNE_EVERY_EPOCHS = 5
MIN_EPOCHS_BEFORE_PRUNE = 30
DISABLE_HARD_PRUNE_IN_DEMO = True

VALIDATE_EVERY_EPOCHS = 5
ALWAYS_VALIDATE_FINAL = True



#Regularization
DROPOUT=0.3
MASKOUT_TIMESERIES=True
MASKOUT_LENGTH=5


# --- ARCHITECTURAL HYPERPARAMETERS (OPTIMIZABLE) ---
CNN_STRIDE = 1  
FEAT_EXT_HIDDEN = 64      # Hidden dimension size
FEAT_EXT_DROPOUT = 0.1    # Dropout rate

# Component Depths
CNN_LAYERS = 2            
RNN_LAYERS = 2            
TRANSFORMER_LAYERS = 2    
MLP_LAYERS = 3            

# --- GBB Core Properties (FastKAN & CfC) ---
KAN_BASIS_FUNCTIONS = 5   # Number of Gaussian RBFs per curve
KAN_LAYERS = 1            # Depth of graph processing
KAN_HEADS = 3           # Multi-Head FastKAN (Excitation/Inhibition/Modulation)
SPLINECOEFF = 0.05           #Initialization coefficent of the spline-weights
ADJ_PRIOR_STRENGTH = 0.25
CFC_BACKBONE_LAYERS = 2
CFC_BACKBONE_UNITS = 128


# --- TEMPORAL / STIMULUS / HEMODYNAMIC PATCHES ---
USE_TEMPORAL_STIM_ENCODER = True
STIM_ENCODER_KERNEL = 5
STIM_DROPOUT_PROB = 0.20
STIM_DROPOUT_START_FRAC = 0.30

USE_SEQ_CFC = True
FMRI_STEP_PROJ_DIM = CFC_BACKBONE_UNITS   # keep equal to backbone for residual compatibility

#USE_HEMODYNAMIC_HEAD = True
#HEMO_KERNEL_SIZE = 3
#HEMO_INIT = "cbv"   # options: "cbv", "uniform"

# --- OPTIONAL HEAD-WISE SIGN CONSTRAINTS ---
LAMBDA_HEAD_SIGN = 0.0          # keep 0 until temporal patch is stable
HEAD_SIGN_WARMUP_FRAC = 0.60    # only activate after 60% of epochs
HEAD_MOD_GAIN_PENALTY = 0.25    # weaker gain for modulatory head

# --- SIGNED GRAPH MESSAGE PATCH ---
ENABLE_SIGNED_HEAD_MESSAGES = True
GRAPH_HEAD_SIGNS = [1.0, -1.0, 0.5]   # excitatory, inhibitory, modulatory
GRAPH_MIX = 0.70                      # graph contribution after residual bypass

# --- MODEL PHYSICS ---
# "standard" = Original CfC (Unconstrained decay, good for general time-series)
# "biophysical" = Constrained CfC (Positive decay, Tanh drive, best for fMRI)
CFC_PHYSICS_MODE = "biophysical"

# --- LOSS FUNCTION HYPERPARAMETERS ---
LAMBDA_ACCURACY = 1.0
LAMBDA_CORRELATION = 10.0   
LAMBDA_CORRELATION_FINAL = 3.0
REG_PHASE_START_FRAC = 0.60
LAMBDA_VAR = 1.25
LAMBDA_DERIVATIVE = 0.5   
LAMBDA_METABOLIC = 0.00001    # L2 on Node Activations
# Was 0.01 (suited for Mean). 
# Now ~5e-5 (suited for Sum).

#OLD: LAMBDA_SPARSITY = 0.001 (Too strong! caused collapse)
# NEW: Relax it to allow connections to form
LAMBDA_SPARSITY = 1e-6 # # 0.00001
LAMBDA_WIRING = 0.00001    # Keep this low
LAMBDA_GROUP_LASSO=0.0002
LAMBDA_HEAD_GROUP = 0.0    # Group Lasso on Functional Heads

LAMBDA_SMOOTHNESS = 5.0
# [FIX] Define the Sigma used by dataset.py
#ADJACENCY_SIGMA = 25.0  # Increased to 25mm to bridge the 305-node gaps
SMOOTHNESS_SIGMA_MM = 40.0 # Keep this for reference or if used elsewhere  # Spatial Continuity
SMOOTHNESS_K_NEIGHBORS = 15 # Only smooth with top-k functional neighbors (Biologically plausible sparse coding)

LAMBDA_ORTH_LOSS=0.05

LAMBDA_LONGTERM=0.05
LONGTERM_THRESHOLD_MODERATE = 8.0 # Free until here
LONGTERM_THRESHOLD_EXTREME = 20.0  # Linear until here, then Quadratic

LAMBDA_TAU_VAR = 0.05 # prevents collapse
LAMBDA_TAU_SATURATION = 0.20
LAMBDA_TAU_DIVERSITY = 0.25   # reduce from 2.0
LAMBDA_TAU_SMOOTH = 0.05     # spatial coherence
LAMBDA_TAU_HIER = 0.05       # biological hierarchy
LAMBDA_RANK_LOSS=0.05

TAU_LOGNORMAL_MEDIAN = 5.0
TAU_LOGNORMAL_SIGMA = 0.5

# --- CfC neural-population regularization ---
TAU_MIN_PHYS = 1.5
TAU_MAX_PHYS = 12.0
TAU_BOUND_MODE = "sigmoid"   # "sigmoid" or "softplus"

LAMBDA_CFC_POP = 0.01
CFC_GATE_TARGET = 0.50
CFC_GATE_MIN = 0.05
CFC_GATE_MAX = 0.95
CFC_F_ACTIVITY_MAX = 3.0
CFC_DRIVE_MAX_ABS = 3.0

CFC_NODE_BIAS_SMOOTH_W = 0.10
CFC_NODE_BIAS_L2_W = 0.01
CFC_GATE_REG_W = 0.10
CFC_DELTA_ENERGY_W = 0.05
CFC_F_ACTIVITY_W = 0.05
CFC_DRIVE_BALANCE_W = 0.05
CFC_DRIVE_BOUND_W = 0.05

# --- OPTIMIZATION CONSTRAINTS (OPTUNA) ---
# [NEW] Centralized HPO settings
N_TRIALS = 50           # Total number of architecture candidates to test
TRIALS_PER_EVAL = 3     # Epochs per candidate (Partial Training) to guess performance
BASELINE_MSE = 0.80     # The "Must-Beat" Score (1.0 = guessing mean, 0.80 = learned something)
MAX_PERF_DROP = 0.05    # How much accuracy we sacrifice for biology (5%)

# --- GLOBAL DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBAL DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
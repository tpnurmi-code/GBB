# Glass-Box Brain (GBB) Project Structure

This document describes the organization of the refactored Glass-Box Brain project and the responsibility of each directory and file.

## Architectural overview

The package follows the main computational pipeline:

```text
Neuroimaging files and stimulus data
                │
                ▼
        gbb.data.dataset
  loading, parcellation, windows
                │
                ▼
     gbb.models.feature_extractors
      H1–H6 temporal hypotheses
                │
                ▼
          gbb.models.kan
  interregional FastKAN interactions
                │
                ▼
          gbb.models.cfc
 continuous-time population dynamics
                │
                ▼
    gbb.models.hemodynamics
 optional BOLD/CBV observation model
                │
                ▼
       gbb.training.*
 losses, regularization, optimization
                │
                ▼
  gbb.analysis and gbb.visualization
 maps, connectivity, logs, and figures
```

## Intended repository tree

```text
GBB/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── docs/
│   ├── GBB_PROJECT_STRUCTURE.md 
│   ├── PHASE_II.md
│   ├── project_phases.pdf
│   ├── GBB_project_research_plan.pdf
│   │
├── examples/
│   ├── generate_custom_synthetic.py
│   │
├── gbb/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── defaults.py
│   │   └── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── files.py
│   │   ├── masks.py
│   │   ├── hrf.py
│   │   └── augmentation.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feature_extractors.py
│   │   ├── stimulus.py
│   │   ├── kan.py
│   │   ├── cfc.py
│   │   ├── hemodynamics.py
│   │   ├── mesocort_gbb.py
│   │   └── factory.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── loop.py
│   │   ├── validation.py
│   │   ├── losses.py
│   │   ├── regularizers.py
│   │   ├── metrics.py
│   │   ├── schedules.py
│   │   ├── pruning.py
│   │   ├── checkpointing.py
│   │   └── distributed.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── anatomical_aggregation.py
│   │   ├── map_export.py
│   │   ├── model_outputs.py
│   │   └── log_parser.py
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   └── nifti.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── connectome.py
│       ├── timeseries.py
│       └── training_dynamics.py
│
├── scripts/
│   ├── train.py
│   └── optimize.py
│
└── tests/
    ├── test_end_to_end_smoke.py
    ├── test_synthetic_config.py
    ├── test_synthetic_export.py
    ├── test_synthetic_ground_truth.py
    └── test_synthetic_simulation.py
```

---

## Root files

| File | Purpose |
|---|---|
| `README.md` | Main public introduction to the project. Explains installation, basic usage, architecture, and the current scientific scope. |
| `pyproject.toml` | Python package and development-tool configuration. Defines project metadata, supported Python version, package discovery, and settings for tools such as Ruff and pytest. |
| `requirements.txt` | Runtime dependencies needed to load data, train models, export neuroimaging results, and create figures. |
| `requirements-dev.txt` | Development-only dependencies for testing, linting, formatting, and static checks. |
| `LICENSE` |Apache 2.0 Licence |
| `NOTICE` | Required notifications |


## `gbb/` — installable Python package

The `gbb` directory contains reusable project code. Files in this package should be importable without starting training or performing other expensive work as a side effect.

| File | Purpose |
|---|---|
| `gbb/__init__.py` | Marks `gbb` as a Python package and provides the package-level namespace. It should remain lightweight. |

---

## `gbb/config/` — configuration definitions

This directory defines both the typed long-term configuration API and the current flat compatibility API.

| File | Purpose |
|---|---|
| `config/__init__.py` | Exposes configuration construction and loading functions, including `load_config()`. |
| `config/schemas.py` | Defines dataclasses for each configuration domain: paths, experiment settings, stimulus handling, data loaders, feature extractors, graph layers, CfC dynamics, losses, optimization, and other model settings. |
| `config/defaults.py` | Constructs a complete default `Config` instance and resolves default paths and environment-variable overrides. |
| `config/config.py` | Flat compatibility layer exposing constants such as `BATCH_SIZE`, `TR`, and `MODEL_TYPE` for modules not yet migrated to nested dataclass access. |

### Configuration direction

New code should progressively use typed nested configuration:

```python
cfg.training.batch_size
cfg.feature_extractor.hidden_dim
```

The flat compatibility module remains useful while older modules are being migrated:

```python
from gbb.config import config

batch_size = config.BATCH_SIZE
```

---

## `gbb/data/` — neuroimaging data preparation

This directory handles file discovery, NIfTI parcellation, stimulus loading, window construction, mask metadata, response kernels, and training augmentation.

| File | Purpose |
|---|---|
| `data/__init__.py` | Marks the directory as the data subpackage. |
| `data/dataset.py` | Defines `NiftiLaminarDataset`. Loads one or more fMRI runs, extracts parcel time series with `NiftiLabelsMasker`, aligns stimuli, constructs adjacency and distance information, and exposes prediction windows without allowing windows to cross run boundaries. |
| `data/files.py` | Discovers subject and run files under the configured dataset directory and returns standardized run records for later subject-level splitting. |
| `data/masks.py` | Loads labelled masks, extracts sorted region IDs, reads optional `mask_metadata.json`, computes MNI centroids, resolves region labels, and extracts non-zero voxel coordinates. |
| `data/hrf.py` | Generates canonical HRF and CBV response kernels for optional stimulus convolution or hemodynamic initialization. |
| `data/augmentation.py` | Implements temporal masking augmentation, either across the whole graph or independently for each node. |

---

## `gbb/models/` — model architecture

This directory contains the scientific model components and the complete GBB architecture.

| File | Purpose |
|---|---|
| `models/__init__.py` | Exposes the principal model classes and construction helpers through the model package namespace. |
| `models/feature_extractors.py` | Implements the six alternative H1–H6 temporal feature-extraction hypotheses: CNN, MLP, CNN–LSTM, LSTM, local-attention Transformer, and global-attention Transformer. |
| `models/stimulus.py` | Defines `StimulusTemporalEncoder`, which converts multichannel stimulus time series into hidden stimulus representations used by the model. |
| `models/kan.py` | Implements Gaussian radial-basis `FastKANLinear` transformations and the multi-head FastKAN graph interaction layer used to model data-dependent interregional influences. |
| `models/cfc.py` | Implements the node-wise closed-form continuous-time dynamics layer. It provides interpretable neural time constants and signed intrinsic-drive parameters. |
| `models/hemodynamics.py` | Implements an optional causal temporal observation head that maps predicted neural activity toward BOLD- or CBV-like responses. |
| `models/mesocort_gbb.py` | Defines the complete `MesocortGBB` model, connecting feature extraction, stimulus encoding, FastKAN graph interactions, CfC dynamics, and the optional hemodynamic observation head. |
| `models/factory.py` | Centralizes construction of H1–H6 feature extractors and full `MesocortGBB` instances from configuration values. |

### Main model flow

```text
Per-node fMRI history
        │
        ▼
H1–H6 feature extractor
        │
        ├──────── StimulusTemporalEncoder ◄── stimulus sequence
        │
        ▼
MultiHeadFastKANLayer
        │
        ▼
CfCLayer
        │
        ▼
HemodynamicObservationHead (optional)
        │
        ▼
Future fMRI prediction
```

---

## `gbb/training/` — optimization and evaluation

This directory contains the executable training orchestration and the reusable components used by the training process.

| File | Purpose |
|---|---|
| `training/__init__.py` | Marks the directory as the training subpackage. |
| `training/train.py` | Main package-level command-line training program. Performs subject-level splitting, dataset and DataLoader construction, device setup, model creation, optimization, checkpointing, validation, and post-training exports. |
| `training/loop.py` | Implements `train_one_epoch()`, including forward passes, composite loss calculation, automatic mixed precision, gradient updates, pruning schedules, and training diagnostics. |
| `training/validation.py` | Runs validation without gradient updates and calculates aggregate and node-wise predictive correlations and losses. |
| `training/losses.py` | Contains prediction losses and biologically motivated penalties, including correlation, variance, derivative, transition-weighted, tau-distribution, hierarchy, smoothness, wiring, sparsity, group-lasso, temporal-orthogonality, and head-sign losses. |
| `training/regularizers.py` | Contains model-wide regularizers that do not belong to one prediction loss, including KAN sparsity and CfC population regularization. |
| `training/metrics.py` | Calculates diagnostics such as effective connection density, batch correlation, and the hierarchy index. |
| `training/schedules.py` | Defines epoch-dependent schedules for introducing or scaling biological constraints during training. |
| `training/pruning.py` | Applies irreversible threshold-based hard pruning to selected model weights after the configured warm-up period. |
| `training/checkpointing.py` | Serializes model, optimizer, epoch, and metric state to checkpoint files. |
| `training/distributed.py` | Handles random seeding and execution setup for local runs, `torchrun`, and SLURM/DDP environments, plus distributed cleanup. |

---

## `gbb/analysis/` — result extraction and interpretation

This directory converts trained-model state and predictions into interpretable maps, matrices, tables, and training summaries.

| File | Purpose |
|---|---|
| `analysis/__init__.py` | Marks the directory as the analysis subpackage. |
| `analysis/anatomical_aggregation.py` | Aggregates fine node-level results into coarser anatomical groups by averaging values that share an anatomical label. |
| `analysis/map_export.py` | Extracts and saves interpretable model outputs, including tau, intrinsic drive, FastKAN complexity and tuning summaries, effective interaction matrices, mechanistic NIfTI maps, CSV tables, and map-stability estimates. |
| `analysis/model_outputs.py` | High-level post-training orchestration. Gathers model and dataset information and calls the appropriate map, connectivity, BrainNet, and simulation exporters. |
| `analysis/log_parser.py` | Parses plain-text training logs, removes ANSI codes, aggregates batch and epoch statistics, and creates multi-panel training-dynamics figures. |

---

## `gbb/io/` — scientific file serialization

This directory contains low-level output functions that write scientific data formats without deciding what scientific result should be exported.

| File | Purpose |
|---|---|
| `io/__init__.py` | Marks the directory as the input/output subpackage. |
| `io/nifti.py` | Saves node or voxel data as NIfTI, matrices as CIFTI, and structured arrays as HDF5. Handles labelled-region and voxel-mask alignment checks. |

---

## `gbb/visualization/` — figure generation

This directory creates visual representations of predictions, network structure, simulations, and training behavior.

| File | Purpose |
|---|---|
| `visualization/__init__.py` | Marks the directory as the visualization subpackage. |
| `visualization/connectome.py` | Creates glass-brain connectomes, circular connectivity diagrams, tau-sorted interaction matrices, and BrainNet Viewer `.node` and `.edge` files. |
| `visualization/timeseries.py` | Plots prediction histories and future forecasts, identifies best- and worst-predicted nodes, and simulates stimulus-driven spreading activation. |
| `visualization/training_dynamics.py` | Provides the public visualization import for training-dynamics plots implemented by the log parser. |

---

## `scripts/` — user-facing executable wrappers

Scripts should remain thin. Reusable logic belongs in `gbb/`, while these files parse arguments and call package functions.

| File | Purpose |
|---|---|
| `scripts/train.py` | Minimal executable wrapper for starting the package training entry point with `python scripts/train.py`. |
| `scripts/optimize.py` | Runs Optuna hyperparameter optimization through the refactored dataset, model, training, and validation APIs. |

---

## `tests/` — automated regression and smoke tests

These tests verify that key components remain runnable during refactoring.

| File | Purpose |
|---|---|
| `tests/test_model_forward.py` | Instantiates all H1–H6 alternatives and verifies successful forward and backward passes on synthetic data. |
| `tests/test_losses.py` | Tests important loss-function properties, including bounded correlation loss, empty hierarchy comparisons, and configured tau limits. |

## Files and directories that should not be committed

The generated ZIP currently contains a `.pytest_cache/` directory. This is a local pytest artifact and should be removed from the repository and excluded in `.gitignore`:

```gitignore
.pytest_cache/
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
build/
dist/
```

The archive also contains empty `gbb/scripts/` and `gbb/tests/` directories. The active scripts and tests are at repository root under `scripts/` and `tests/`; the empty package-level directories can be deleted to avoid ambiguity.

## Responsibility boundaries

To keep the repository maintainable:

- `data/` should know how to load and represent data, but not how to train a model.
- `models/` should define differentiable architecture components, but not read files or save figures.
- `training/` should optimize and evaluate models, but not contain neuroimaging export implementation.
- `analysis/` should derive scientific outputs from trained models.
- `io/` should serialize arrays and images without assigning scientific meaning to them.
- `visualization/` should render results, while numerical extraction stays in `analysis/`.
- `scripts/` should be thin entry points rather than alternative implementations of project logic.
- `tests/` should use small synthetic inputs wherever possible so that they run without private neuroimaging datasets.

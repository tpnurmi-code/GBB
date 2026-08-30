# Glass-Box Brain

[![Tests](https://github.com/tpnurmi-code/GBB/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/tpnurmi-code/GBB/actions/workflows/tests.yml)

The Glass-Box Brain (GBB) project lies at the intersection of neuroscience/neuroimaging and AI. It is designed to utilize neuroimaging datasets as training data. Its main purpose is to employ 7-Tesla cerebral blood volume functional magnetic resonance imaging (7T-CBV-fMRI) data since this method offers improved laminar and mesoscale spatial resolution potentially at the level of cortical columns to determine different relationships between different brain regions and cortical columns. The aims of the GBB project are three-fold.

### 1. Mechanistic neuroimaging 
GBB aims at learning novel relationships and mechanisms from 7T-CBV-fMRI neuroimaging data not obtainable otherwise. However, these novel relationships and mechanisms serve only as testable hypotheses with separate experiments, not the ground-truth. Thus, from the neuroscience perspective the GBB is a novel hypothesis generation 'device'.

### 2. Biologically inspired machine learning 
Since the brain is an incredibly energy-efficient organ, the GBB project's goal is to develop novel architectural structures and learning algorithms to make future machine learning architectures more energy-efficient and better at learning patterns from few examples instead of thousands.

### 3. Transparent model design 
The whole architecture is designed to be transparent and therefore function as a "laboratory" where different design choices can be tested.  
  
For further description, see the [short research plan](https://github.com/tpnurmi-code/GBB/blob/main/docs/GBB_Research_Plan_short.pdf) for the project.
## Future of the project 
For the longer-term vision for the project, see the [Phase II documentation](https://github.com/tpnurmi-code/GBB/blob/main/docs/PHASE_II.md)

## Architecture

The architecture employs three distinct components. For more specific description of the file/directory structure, see [the project structure document](https://github.com/tpnurmi-code/GBB/blob/main/docs/GBB_PROJECT_STRUCTURE.md)

### 1. Feature extractors 
The feature-extractor component compares six alternative temporal processing architectures. Spatial interactions between neural populations are subsequently modeled by the FastKAN graph layers.

### 2. Continuous-time neural-population dynamics 
The second component corresponds to the internal neural computations and information processing of a neural population and consists of closed-form continuous-time (CfC) neural networks, which model the internal computations of a neural population with interpretable differential equations [[1]](#1).

### 3. Inter-regional connectivity 
The third component corresponds to connectivity between different neural populations and consists of a variant of Kolmogorov-Arnold networks (FastKAN) and model these connections with a mixture of Gaussian basis functions. The FastKAN component can also contain several attention heads that can model different types of connectivity. Multiple FastKAN interaction heads can learn candidate signed or functionally differentiated connectivity channels. Interpretations such as excitatory, inhibitory, or modulatory roles require independent validation [[2]](#2).

## Outputs

After training, the GBB model can export spatial parameter maps projected onto a group-brain template. These maps may include candidate measures of local timescale, connectivity strength, interaction complexity, tuning specificity, or other interpretable model-derived quantities.

The purpose of these maps is not to claim direct access to hidden neural mechanisms, but to generate biologically testable hypotheses from indirect neuroimaging data.

## Validation philosophy

A central principle of the project is that interpretability must be earned
empirically. A model-derived parameter map is treated as mechanistically
informative only if it demonstrates:

- predictive validity on held-out data;
- stability across random seeds, participants, and datasets;
- robustness to ablation and model-comparison tests;
- recovery of known parameters in synthetic-data experiments; and
- biological anchoring against independent sensorimotor, laminar, or multimodal evidence.

## Current status

The current GBB implementation is a research prototype developed in Python and PyTorch. The codebase has been reorganized into modular data, model, training, analysis, and visualization packages. Validation and reproducibility infrastructure remain under active development.

The document below describes the project, its scope, validation strategy, and future development plans: [GBB project research plan](https://github.com/user-attachments/files/29357478/GBB_project_research_plan.pdf) and [GBB project phase graph](https://github.com/tpnurmi-code/GBB/blob/main/docs/project_phases.pdf)
## Installation

### Requirements

- Python 3.10 or newer
- Git
- `pip`
- PyTorch-compatible CPU, NVIDIA CUDA, or AMD ROCm environment

The project has been developed primarily with Python and PyTorch. GPU
acceleration is recommended for full model training but is not required for
basic imports, testing, or small development runs.

### 1. Clone the repository

```bash
git clone https://github.com/tpnurmi-code/GBB.git
cd GBB
```
### 2. Create a virtual environment
#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

#### Linux or macOS BASH

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
### 3. Install PyTorch

A CUDA- or ROCm-capable GPU is strongly recommended for practical training. 
CPU-only execution is intended primarily for tests, debugging, and very small demonstrations.

Install the PyTorch build appropriate for your operating system and hardware
using the official PyTorch installation selector:

- NVIDIA GPU: install a CUDA-enabled build.
- AMD GPU: install a ROCm-enabled build where supported.
- CPU-only: suitable mainly for installation checks, tests, and small smoke runs.

A basic CPU installation is:

```bash
python -m pip install torch
```
#### GPU training (recommended)

Full GBB training is computationally intensive, and an NVIDIA CUDA-capable GPU
is strongly recommended.

GBB automatically uses a CUDA GPU when PyTorch reports that CUDA is available.
If CUDA is not available, training falls back to the CPU.

After installing PyTorch, verify the installation before starting a full
training run:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

For an NVIDIA GPU, the expected output should include something similar to:

```text
CUDA build: 13.0
CUDA available: True
GPU count: 1
GPU: NVIDIA GeForce RTX 3090
```

The exact CUDA version and GPU model will depend on the system.

> [!IMPORTANT]
> If `CUDA available` is `False`, GBB will run on the CPU instead. Do not start
> a full training experiment until the PyTorch/CUDA installation has been
> corrected, unless CPU execution is intentional.

When training begins, GBB also reports the selected device:

```text
Starting H1 training on cuda:0
```

If the corresponding message reports `cpu`, GPU acceleration is not active.

For AMD GPUs, install the appropriate ROCm-enabled PyTorch build. PyTorch uses
the `torch.cuda` interface for many ROCm device checks as well.

#### CPU-only installation

A CPU installation is sufficient for imports, automated tests, and small
development runs:

```bash
python -m pip install torch
```

CPU execution is not recommended for full GBB model training.

### 4. Install GBB

Install GBB and its required dependencies:

```bash
python -m pip install -r requirements.txt
```

Supported dependency versions are defined in `pyproject.toml`.

### 5. Optional dependencies

For visualization and hyperparameter optimization:

```bash
python -m pip install -e ".[visualization,optimization]"
```

For development and testing:

```bash
python -m pip install -r requirements-dev.txt
```

### 6. Generate a privacy-safe synthetic fMRI dataset

Users without access to the original neuroimaging data can generate a
GBB-compatible synthetic dataset with known mechanistic ground truth.

The quick preset creates a small two-subject, one-run dataset suitable for
testing installation, NIfTI loading, model construction, and the training
pipeline.

```bash
gbb-generate-synthetic \
    --quick \
    --output synthetic_gbb_quick \
    --response bold \
    --overwrite
```

The equivalent Python module command is:

```bash
python -m gbb.synthetic.cli \
    --quick \
    --output synthetic_gbb_quick \
    --response bold \
    --overwrite
```

For a CBV-like synthetic dataset:

```bash
gbb-generate-synthetic \
    --quick \
    --output synthetic_gbb_cbv \
    --response cbv \
    --overwrite
```

A larger custom CBV-like dataset can be generated with:

```bash
gbb-generate-synthetic \
    --output synthetic_gbb_cbv \
    --subjects 6 \
    --runs 2 \
    --columns 16 \
    --timepoints 200 \
    --response cbv \
    --overwrite
```

The generator creates synthetic NIfTI runs, region and cortical-column masks,
stimulus files, event tables, and files containing the known mechanistic ground
truth. The generated data contain no participant data.

### 7. Select the generated dataset

GBB reads the dataset location from the `GBB_DATA_DIR` environment variable.
Set it before importing or starting GBB.

#### Windows PowerShell

```powershell
$env:GBB_DATA_DIR = (Resolve-Path ".\synthetic_gbb_quick").Path
```

For the CBV example:

```powershell
$env:GBB_DATA_DIR = (Resolve-Path ".\synthetic_gbb_cbv").Path
```

#### Linux or macOS

```bash
export GBB_DATA_DIR="$(pwd)/synthetic_gbb_quick"
```

For the CBV example:

```bash
export GBB_DATA_DIR="$(pwd)/synthetic_gbb_cbv"
```

### 8. Verify the generated dataset

#### Windows PowerShell

```powershell
Get-ChildItem synthetic_gbb_quick
Get-ChildItem synthetic_gbb_quick\synthetic_subject_001\NifTi
```

#### Linux or macOS

```bash
find synthetic_gbb_quick -maxdepth 3 -type f | head -30
```

The generated directory should contain files such as:

```text
synthetic_gbb_quick/
├── group_roi_mask.nii
├── group_roi_mask_10.nii
├── cortical_columns_7T.nii
├── ground_truth/
└── synthetic_subject_001/
    └── NifTi/
        ├── rfunctional_run1.nii.gz
        ├── rfunctional_run1_events.tsv
        ├── rfunctional_run1_stim.mat
        └── rfunctional_run1_ground_truth.npz
```

### 9. Configure the data directory

GBB reads the dataset location from the GBB_DATA_DIR environment variable.

#### Windows PowerShell
```powershell
$env:GBB_DATA_DIR = "G:\path\to\your\data"
```
#### Linux or macOS
```bash
export GBB_DATA_DIR="/path/to/your/data"
```
### 10. Verify the installation
Check that the package and its main components can be imported:

```bash
python -c "import gbb; print('GBB import successful')"
python -c "from gbb.data.dataset import NiftiLaminarDataset; print('Dataset import successful')"
python -c "from gbb.models.mesocort_gbb import MesocortGBB; print('Model import successful')"
```

Check the installed PyTorch environment:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 11. Start training
```bash
python -m gbb.training.train
```

## References
<a id="1">[1]</a> 
Hasani R., Lechner M.,  Amini A., Liebenwein L,  Ray A, Tschaikowski M., Tesch G. & Rus D. (2022).
 Closed-form continuous-time neural networks
Nature Machine Intelligence, 4, pages 992–1003. https://www.nature.com/articles/s42256-022-00556-7

<a id="1">[2]</a> 
Ziyao L. (2024). 
Kolmogorov-Arnold Networks are Radial Basis Function Networks
arXiv (https://arxiv.org/html/2405.06721v1)

# Glass-Box Brain

The Glass-Box Brain (GBB) project lies at the intersection of neuroscience/neuroimaging and AI. It is designed to utilize neuroimaging datasets as training data. Its main purpose is to employ 7-Tesla cerebral blood volume functional magnetic resonance imaging (7T-CBV-fMRI) data since this method offers improved laminar and mesoscale spatial resolution potentially at the level of cortical columns to determine different relationships between different brain regions and cortical columns. The aims of the GBB project are three-fold.

### 1. Mechanistic neuroimaging 
GBB aims at learning novel relationships and mechanisms from 7T-CBV-fMRI neuroimaging data not obtainable otherwise.

### 2. Biologically inspired machine learning 
Since the brain is incredibly an energy-efficient organ, the GBB project's goal is to develop novel architectural structures and learning algorithms to make future machine learning architectures more energy-efficient and better at learning patterns from few examples instead of thousands.

### 3. Transparent model design 
The the whole architecture is designed to be transparent and therefore function as a "laboratory" where different design choices can be tested.
## Architecture

The architecture employs three distinct components.

### 1. Feature extractors 
The first component is a feature extractor layer that has six alternative neural network architectures. These alternative architectures correspond to input to a neural population (afference) and function as alternative hypotheses to be tested to examine which spatial and temporal scales are most relevant for information processing in the sensorimotor cortices.

### 2. Continuous-time neural-population dynamics 
The second component corresponds to the internal neural computations and information processing of a neural population and consists of closed-form continuous-time (CfC) neural networks, which model the internal computations of a neural population with interpretable differential equations.

### 3. Inter-regional connectivity 
The third component corresponds to connectivity between different neural populations and consists of a variant of Kolmogorov-Arnold networks (FastKAN) and model these connections with a mixture of Gaussian basis functions. The FastKAN component can also contain several attention heads that can model different types of connectivity such as excitatory, inhibitory and neuromodulatory connectivity Both CfC and FastKAN layers are interpretable and can be used to extract different spatial parameter maps projected on a group-brain after training with the fMRI data.

## Outputs

After training, the GBB model can export spatial parameter maps projected onto a group-brain template. These maps may include candidate measures of local timescale, connectivity strength, interaction complexity, tuning specificity, or other interpretable model-derived quantities.

The purpose of these maps is not to claim direct access to hidden neural mechanisms, but to generate biologically testable hypotheses from indirect neuroimaging data.
Validation philosophy

A central principle of the project is that interpretability must be earned empirically. A model-derived parameter map is treated as mechanistically informative only if it demonstrates:

    predictive validity on held-out data;
    stability across random seeds, subjects, and datasets;
    robustness to ablation and model-comparison tests;
    recovery of known parameters in synthetic-data experiments;
    biological anchoring against independent sensorimotor, laminar, or multimodal evidence.

## Current status

The current GBB implementation is a prototype developed in Python and PyTorch. It is intended as an experimental research framework for interpretable neuroimaging and biologically inspired machine learning. Refractoring of the code is in progress as it contains many files that are too long.

The document below describes the project, its scope, validation strategy, and future development plans: [GBB project research plan](https://github.com/user-attachments/files/29357478/GBB_project_research_plan.pdf) and [GBB project phase graph] (https://github.com/tpnurmi-code/GBB/blob/main/project_phases.pdf) 
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
'''python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
'''
#### Linux or macOS BASH
'''python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
'''
### 3. Install PyTorch
PyTorch builds differ between CPU, NVIDIA CUDA, and AMD ROCm systems.
Select the installation command appropriate for your operating system and
hardware from the official PyTorch installation selector.

For a basic CPU installation:

'''python -m pip install torch'''

### 4. Required Python packages
'''python -m pip install \
    numpy \
    scipy \
    nibabel \
    nilearn \
    pandas \
    matplotlib \
    seaborn \
    h5py \
    tqdm \
    tensorboard
'''

### 5. Optional dependencies:
'''python -m pip install mne optuna'''

## Future plans (Phase II)
After an accurate, stable and biologically plausible baseline model, the GBB will proceed to phase II. This phase will explore novel learning and training rules, algorithms and strategies. 

The GBB architecture is intended to be developed from an autoregressive to a purely generative model by gradually reducing and eventually disabling the input from the fMRI-timeseries to the model during the training process. Thus, when training the generative model, the training will begin as a autoregressive model that tries to predict the future values of the fMRI time-series based on previous values of the fMRI-timeseries and stimuli. The strength of the fMRI-timeseries input signal is then gradually degraded and finally removed, leaving only the input signal from the stimuli. The fMRI time-series is then only used to calculate the prediction error. This approach and move from autoregressive to a causal generative model will be challenging but this kind of pressure of trying to predict brain responses from stimulus data might actually reveal precious results as we expect the generative causal GBB to mimic the real brain more genuinely.

The stable baseline model will proceed from the traditional Adaptive Moment Estimation (Adam) optimizer learning strategy to a test bench to more biologically-inspired learning strategies. These learning strategies include: 
### 1. Global Error Backprop 
This is the standard Adam training and forms a baseline model to compare other learning strategies with. 
### 2. Predictive Error Minimization 
This learning strategy uses predictive coding for the FastKAN layers and Hebbian learning strategy for the CfC layers to train the GBB.  
### 3. Bio-Approximated Gradients 
These gradients are more biologically plausible than standard backpropagation ones but approximate similar kind of gradient descent. In the standard backpropagation/gradient descent algorithm, the connection vector of a neuron is W, and in order to calculate the gradients, a transpose of W, W^{T} must be known. This would suggests that the presynaptic neuron would have to know exactly the strength of the postsynaptic neuron’s connection to the next layer, which seems biolgocially implausible. Bio-Approximated Gradients solves this problem by replacing W^{T} with a random connection matrix B where the weights stay the same after the initial intialization. Remarkably, the forward weights, W adjust to the randomness of B and the Bio-Approximated Gradients begins to approximate backprobagation/gradient descent.    
### 4. Neuromodulated Reinforcement 
This learning strategy mimics the dopaminergic system as a mechanism of learning where the node-specific loss gradient for each neuron is replaced with a global error signal with a radius (sphere of influence). No neuron has direct information whether to adjusts its weights. However, the global error signal will correlate with neuron’s performance after thousands of repetitions, albeit slower than backprobagation 
### 5. Competitive Self-Organization 
This learning strategy uses “winner-takes-all” logic where the node/neuron with highest activation sends lateral inhibition to its neighbouring nodes and is only one in its neighbourhood that gets to update its weights. This creates a sparse network that should maximize efficiency. The different learning strategies are listed below in a table along local vs. global and error-driven vs. activity driven dimensions.
These different advanced learning strategies are compared against each other and the baseline backpropagation strategy on multiple criteria. These criteria are fMRI time-series accuracy measures such as correlation, MSE and the first derivate with both autoregressive and causal generative GBB architectures. Antoher criteria will cover biological plausibility and resource-effieciency of these learning strategies. Biological plausibility will be determined by analysing the network topology and other neural network properties produced by each learning strategy and comparing them to biological equivalents. Efficiency criteria cover sparness of the networks, its memory and computational as well as energy consumption requirements and will be directly related to our green AI goal.
Lastly, we will try an advanced “dendritic” architecture where we will encompass the feature extractors by gated linear units (GLU)  and dendritic tree networks (DTNs). The GLU+DTNs will act as a gating mechanism (Output=Signal×σ(Gate)) that allows selective passing or blocking of information from the features extractors. This gating mechanism simulates shunting inhibition of real neurons and makes our model more expressive, context dependent and biologically more realistic.  

| Learning Mode | LOCAL (Bio-Plausible) | GLOBAL (Math-Heavy) |
|---|---|---|
| **ERROR-DRIVEN (Goal-Directed)** | Predictive Error Minimization<br>Bio-Approximated Gradients | Global Error Backprop (Baseline) |
| **ACTIVITY-DRIVEN (Self-Organizing)** | Competitive Self-Organization | Neuromodulated Reinforcement |

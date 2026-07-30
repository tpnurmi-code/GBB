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

The document below describes the project, its scope, validation strategy, and future development plans: [GBB project research plan](https://github.com/user-attachments/files/29357478/GBB_project_research_plan.pdf)

# Future vision (Phase II)
## GBB Phase II Research Roadmap
> **Status:** Phase II describes a longer-term future vision for the GBB project. None of the components are currently under implementation as Phase II execution starts strictly after a stable Phase I baseline model has been succesfully tested, produced validated results and progressed to scientific publication phase.

## Purpose
Phase I focusses on developing a functional baseline model that produced verifiable and testable brain maps from 7-Tesla Cerebral Blood Volume functional magnetic resonance imaging (7T-CBV-fMRI) data. Phase II then leverages this functional transparent baseline model by modifying it in ways described below and examining which changes produce improved energy-efficiency, successful learning of more complex patterns and ideally learning from a few training examples instead of thousands (one-shot learning). Thus, the stable and tested baseline model works as a "test bench" for developing novel machine learning architectures and algorithms that are bio-inspired from patterns learned from real neuroimaging data.     

## Preconditions from Phase I

Phase II will begin only after the following Phase I endpoints have been
evaluated:

1. Predictive performance has been compared with simpler reference models.
2. Core parameters have been tested using synthetic parameter recovery.
3. Model-derived maps have been assessed across random seeds and data splits.
4. The effects of major architectural components have been examined through
   ablation studies.
5. At least one candidate model-derived result has been compared with
   independent biological evidence.
6. The baseline pipeline can be reproduced from documented configuration,
   data, and software versions.


## Workstream 1 : Stimulus-driven data generation
The GBB architecture is intended to be developed from an autoregressive to a more generative model by gradually reducing and eventually disabling the input from the fMRI-timeseries to the model during the training process. Thus, when training the generative model, the training will begin as a autoregressive model that tries to predict the future values of the fMRI time-series based on previous values of the fMRI-timeseries and stimuli. The strength of the fMRI-timeseries input signal is then gradually degraded and finally removed, leaving only the input signal from the stimuli. The fMRI time-series is then only used to calculate the prediction error. This approach and move from autoregressive to a causal generative model will be challenging but this kind of pressure of trying to predict brain responses from stimulus data might actually reveal precious results as we expect the generative causal GBB to mimic the real brain more genuinely.

## Workstream 2 : Alternative learning strategies to backpropagation
The stable baseline model will proceed from the traditional Adaptive Moment Estimation (Adam) optimizer learning strategy to a test bench to more biologically-inspired learning strategies. These learning strategies include: 
### 1. Global Error Backprop 
This is the standard Adam training and forms a baseline model to compare other learning strategies with. 
### 2. Predictive Error Minimization 
This learning strategy uses predictive coding for the FastKAN layers and Hebbian learning strategy for the CfC layers to train the GBB.  
### 3. Bio-Approximated Gradients 
These gradients are more biologically plausible than standard backpropagation ones but approximate similar kind of gradient descent. In the standard backpropagation/gradient descent algorithm, the connection vector of a neuron is W, and in order to calculate the gradients, a transpose of W, W^{T} must be known. This would suggests that the presynaptic neuron would have to know exactly the strength of the postsynaptic neuron’s connection to the next layer, which seems biologically implausible. Bio-Approximated Gradients solves this problem by replacing W^{T} with a random connection matrix B where the weights stay the same after the initial initialization. Remarkably, the forward weights, W adjust to the randomness of B and the Bio-Approximated Gradients begins to approximate backpropagation/gradient descent.    
### 4. Neuromodulated Reinforcement 
This learning strategy mimics the dopaminergic system as a mechanism of learning where the node-specific loss gradient for each neuron is replaced with a global error signal with a radius (sphere of influence). No neuron has direct information whether to adjusts its weights. However, the global error signal will correlate with neuron’s performance after thousands of repetitions, albeit slower than backpropagation 
### 5. Competitive Self-Organization 
This learning strategy uses “winner-takes-all” logic where the node/neuron with highest activation sends lateral inhibition to its neighbouring nodes and is only one in its neighbourhood that gets to update its weights. This creates a sparse network that should maximize efficiency. The different learning strategies are listed below in a table along local vs. global and error-driven vs. activity driven dimensions.
These different advanced learning strategies are compared against each other and the baseline backpropagation strategy on multiple criteria. These criteria are fMRI time-series accuracy measures such as correlation, MSE and the first derivate with both autoregressive and causal generative GBB architectures. Another criteria will cover biological plausibility and resource-efficiency of these learning strategies. Biological plausibility will be determined by analysing the network topology and other neural network properties produced by each learning strategy and comparing them to biological equivalents. Efficiency criteria cover sparseness of the networks, its memory and computational as well as energy consumption requirements and will be directly related to our green AI goal.
Lastly, we will try an advanced “dendritic” architecture where we will encompass the feature extractors by gated linear units (GLU)  and dendritic tree networks (DTNs). The GLU+DTNs will act as a gating mechanism (Output=Signal×σ(Gate)) that allows selective passing or blocking of information from the features extractors. This gating mechanism simulates shunting inhibition of real neurons and makes our model more expressive, context dependent and biologically more realistic.  

| Learning Mode | LOCAL (Bio-Plausible) | GLOBAL (Math-Heavy) |
|---|---|---|
| **ERROR-DRIVEN (Goal-Directed)** | Predictive Error Minimization<br>Bio-Approximated Gradients | Global Error Backprop (Baseline) |
| **ACTIVITY-DRIVEN (Self-Organizing)** | Competitive Self-Organization | Neuromodulated Reinforcement |

## Workstream 3 : Alternative deep learning architectures
Closed-form Continuous-time Neural Models (CfC) and Kolmogorov–Arnold Networks (KANs) are more transparent and interpretable than most current deep learning (that are more black-box) architectures by design. In the case of CfCs this is due to their compact size, having traceable pathways due to smaller network size and having explicit time-depedence. KANs are more transparent and interpretable due to them having connections encoded as functions instead of traditional network weights, which then can be visualized and analyzed. These properties of CfCs and KANs would enable the GBB project collaborators then to make systematical changes to the trained GBB architecture and observe any changes in performance systematically. Thus, the more transparent and interpretable nature of these GBB components would enable more methodical development of novel machine learning architectures with improved learning, energy-efficiency and complexity solving capabilities.   

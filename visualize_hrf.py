# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 15:48:41 2025

@author: 35840
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from nilearn.glm.first_level import glover_hrf
import config
from models import MesocortGBB
import os

def analyze_impulse_response():
    # 1. Load the Model
    print("--- Loading Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize with same params as training
    # We assume 306 nodes (from standard mask) and 30 time points
    model = MesocortGBB(num_nodes=306, time_points=config.WINDOW_SIZE).to(device)
    
    # Load Weights
    checkpoint_path = os.path.join(config.RESULTS_DIR, "checkpoint_latest.pth")
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found! Using untrained weights (expect random noise).")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle DDP prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")

    model.eval()

    # 2. Identify "S1 / Motor" Nodes
    # We look for nodes with the Highest Tau values, as you observed S1 has high Tau (24s)
    tau_vals = model.cfc.get_tau_values().detach().cpu().numpy()
    
    # Get indices of top 3 slowest nodes (Likely the Hand Knob in your block design)
    s1_indices = np.argsort(tau_vals)[-3:] 
    print(f"Inspecting Nodes with max Tau: {s1_indices} (Tau = {tau_vals[s1_indices]} s)")

    # 3. Simulation Setup
    # We will simulate the CfC cell in isolation to see its intrinsic physics
    simulation_steps = 20  # 20 TRs = 50 seconds
    dt = config.TR         # 2.5s
    
    # Create Impulse Input: [0, 1, 0, 0, ...]
    # Shape: (Time, Batch, Input_Dim)
    stimulus_impulse = torch.zeros(simulation_steps, 1, config.CFC_BACKBONE_UNITS).to(device)
    
    # Create the "Spike" at t=1
    # We project a raw "1.0" stimulus through the learned stim_proj layer
    raw_spike = torch.tensor([[1.0]]).to(device) # (1, 1)
    
    # We need to project this raw spike using the model's learned projection
    # to see how it actually drives the hidden state
    with torch.no_grad():
        projected_spike = model.stim_proj(raw_spike) # (1, 128)
        
    #stimulus_impulse[1, 0, :] = projected_spike
    block_duration_trs = 8 
    start_tr = 5
    for t in range(start_tr, start_tr + block_duration_trs):
        stimulus_impulse[t, 0, :] = projected_spike
    # 4. Run Simulation (Recurrent Loop)
    # We effectively run the equation: h_new = CfC(h_old, stimulus)
    h_current = torch.zeros(1, 306, config.CFC_BACKBONE_UNITS).to(device) # (Batch, Nodes, Hidden)
    
    responses = []
    
    print("--- Running Impulse Simulation ---")
    with torch.no_grad():
        for t in range(simulation_steps):
            # Get stimulus for this step
            # We expand it to match nodes: (1, 306, 128)
            stim_t = stimulus_impulse[t].unsqueeze(1).expand(-1, 306, -1)
            
            # Run CfC Cell
            # Note: CfC takes concatenated input.
            # In your model forward: h_combined = x + i_stim
            # Here we assume x (fMRI input) is 0 to isolate the stimulus response
            
            # For the CfC layer specifically:
            # It calculates gating based on (h, input). 
            # In the full model, 'h' is the node state.
            
            h_next = model.cfc(h_current, stim_t)
            
            # Store the mean activity of the hidden state for the S1 nodes
            # Shape: (1, 306, 64) -> Select S1 indices -> Mean over hidden dim
            s1_activity = h_next[0, s1_indices, :].mean(dim=-1).cpu().numpy()
            responses.append(s1_activity)
            
            h_current = h_next

    responses = np.array(responses) # (Time, 3_Nodes)

    # 5. Generate Reference HRF (Glover) for Comparison
    # We oversample to make it smooth, then slice to TRs
    ref_hrf = glover_hrf(tr=config.TR, oversampling=1, time_length=simulation_steps * config.TR, onset=0)
    # Normalize ref to match response peak for visual comparison
    ref_hrf = ref_hrf / ref_hrf.max() * responses.max()

    # 6. Plotting
    time_axis = np.arange(simulation_steps) * config.TR
    
    plt.figure(figsize=(10, 6))
    
    # Plot Standard Biological HRF
    plt.plot(time_axis, ref_hrf, 'k--', linewidth=2, label='Canonical HRF (Glover)', alpha=0.6)
    
    # Plot Learned Responses
    colors = ['r', 'g', 'b']
    for i, idx in enumerate(s1_indices):
        plt.plot(time_axis, responses[:, i], color=colors[i], marker='o', 
                 label=f'Learned Model (Node {idx}, Tau={tau_vals[idx]:.1f}s)')
        print(responses[:, i])
    plt.title(f"Did the Model Learn the HRF?\nImpulse Response of S1 Nodes vs Biology (TR={config.TR}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Hidden State Activity (a.u.)")
    plt.axvline(x=2.5, color='gray', linestyle=':', label='Stimulus Spike (t=2.5s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(config.RESULTS_DIR, "hrf_analysis.png")
    plt.savefig(save_path)
    print(f"✅ Saved HRF Analysis to {save_path}")
    plt.show()
if __name__ == "__main__":
    analyze_impulse_response()
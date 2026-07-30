import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = "G:/Projects/AI/data/results/Neuromodulation_Functions.h5"

with h5py.File(file_path, 'r') as f:
    grid_x = f['function_grid_x'][:]      
    curves_y = f['function_curves_y'][:]  # (Target, Source, Head, Grid)
    weights = f['head_weights_sample'][:] # (Target, Source, Head)
    
    # --- SELECT WHICH CONNECTION TO VIEW ---
    source_node = 2
    target_node = 0  
      
    # ----------------------------------------

    num_heads = curves_y.shape[2]
    labels = ["Excitatory", "Inhibitory", "Modulatory"]

    plt.figure(figsize=(12, 7))
    
    net_function = np.zeros_like(grid_x)
    
    for head_idx in range(num_heads):
        y = curves_y[target_node, source_node, head_idx, :]
        w_raw = weights[target_node, source_node, head_idx]
        w_scalar = float(np.array(w_raw).flatten()[0])
    
        net_function += y * w_scalar
        # Extract the Weight and FORCE it to a scalar for formatting
        # We use .flatten()[0] to get the actual number out of any array wrapper
        w_raw = weights[target_node, source_node, head_idx]
        w_scalar = float(np.array(w_raw).flatten()[0])

        head_label = labels[head_idx] if head_idx < len(labels) else f"Head {head_idx}"

        plt.plot(grid_x, y, label=f"{head_label} (Weight: {w_scalar:.4f})", linewidth=2.5)
    

    plt.figure()
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axvline(0, color='black', lw=1, ls='--')
    plt.title(f"Synaptic Mechanism: Region {source_node} → Region {target_node}", fontsize=14)
    plt.xlabel("Neighbor Activity (Input)", fontsize=12)
    plt.ylabel("Local Response (Output)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(grid_x, net_function, label="TOTAL NET FUNCTION", color='black', linewidth=4, zorder=10)
    plt.show()
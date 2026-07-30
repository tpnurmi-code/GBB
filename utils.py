import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import glob
import h5py 
import config
import scipy.stats
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import nibabel as nib

from nilearn import plotting, image, datasets
from nilearn.image import math_img, new_img_like, coord_transform
from nilearn.input_data import NiftiLabelsMasker

from tqdm import tqdm
import random
import itertools

import re
from pathlib import Path
    
def get_subject_files(data_dir, num_runs=2):
    subject_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    data_list = []
    mask_path = os.path.join(data_dir, "group_roi_mask.nii")
    for sub_dir in subject_dirs:
        if not os.path.isdir(sub_dir): continue
        sub_id = os.path.basename(sub_dir)
        for run_num in range(1, num_runs + 1):
            filename = f"rfunctional_run{run_num}.nii"
            fmri_path = os.path.join(sub_dir, "NifTi", filename)
            events_path = fmri_path.replace(".nii", "_events.tsv")
            if os.path.exists(fmri_path):
                data_list.append({
                    'id': f"{sub_id}_run{run_num}",
                    'subject_id': sub_id,
                    'fmri': fmri_path,
                    'mask': mask_path,
                    'events': events_path
                    })
    # Let train.py handle shuffling/splitting for reproducibility and subject-level split
    return data_list

# In utils.py

def apply_time_masking(x_batch, mask_len=5, mode='independent'):
    """
    Applies Time-Masking.
    
    Args:
        mode (str): 
            'global' - Whole brain goes dark (Trains ODE/Memory).
            'independent' - Nodes blink independently (Trains Connectivity/Graph).
    """
    x_masked = x_batch.clone()
    B, T, N = x_masked.shape
    safe_limit = T - mask_len - 1
    
    if safe_limit <= 0: return x_masked

    if mode == 'global':
        # Current implementation: Whole brain blinks together
        starts = torch.randint(0, safe_limit, (B,), device=x_batch.device)
        for i in range(B):
            st = starts[i]
            x_masked[i, st : st + mask_len, :] = 0.0
            
    elif mode == 'independent':
        # [NEW] Biologically Plausbile Asynchronous Failure
        # Each node picks its own random time to fail.
        # Shape: (Batch, Nodes)
        starts = torch.randint(0, safe_limit, (B, N), device=x_batch.device)
        
        # Vectorized masking is tricky, so we use a mask tensor
        # Create a grid of time indices: (1, Time, 1)
        t_grid = torch.arange(T, device=x_batch.device).view(1, T, 1)
        
        # Broadcast starts to (Batch, 1, Nodes)
        starts_broad = starts.unsqueeze(1)
        
        # Create Boolean Mask: Time is between start and start+len
        # (Batch, Time, Nodes)
        mask = (t_grid >= starts_broad) & (t_grid < starts_broad + mask_len)
        
        # Apply
        x_masked[mask] = 0.0

    return x_masked

def save_nifti(data_array, mask_data, affine, output_path):
    vol = np.zeros(mask_data.shape)
    vol[mask_data] = data_array
    nib.save(nib.Nifti1Image(vol, affine), output_path)

def save_cifti(matrix_data, output_path):
    rows, cols = matrix_data.shape
    row_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=rows)
    col_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=cols)
    header = nib.cifti2.Cifti2Header.from_axes((row_axis, col_axis))
    img = nib.cifti2.Cifti2Image(matrix_data, header)
    nib.save(img, output_path)

def save_hdf5(data_dict, output_path):
    with h5py.File(output_path, 'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)
    print(f"✅ Saved HDF5: {os.path.basename(output_path)}")

def get_voxel_coords_from_mask(mask_path, num_expected=None):
    """
    Extracts MNI coordinates (x,y,z) for every non-zero voxel in the mask.
    This ensures 1-to-1 mapping with the model nodes trained on this mask.
    """
    print(f"--- Extracting Voxel Coordinates from: {mask_path} ---")
    
    img = nib.load(mask_path)
    data = img.get_fdata()
    affine = img.affine
    
    # 1. Get indices of all non-zero voxels (x, y, z)
    # nilearn/nibabel usually load as (X, Y, Z)
    # We use np.where to get indices
    x_idxs, y_idxs, z_idxs = np.where(data > 0)
    
    # 2. Convert to MNI coordinates
    # Apply affine transformation: MNI = Affine @ [x, y, z, 1]
    # We stack them into a (N, 3) matrix
    voxel_indices = np.stack((x_idxs, y_idxs, z_idxs), axis=1)
    mni_coords = nib.affines.apply_affine(affine, voxel_indices)
    
    num_found = mni_coords.shape[0]
    print(f"Found {num_found} voxels in mask.")
    
    # 3. Validation / Truncation
    # If the mask has more voxels than the model (rare, but possible if mask changed),
    # we assume the model uses the first N voxels (standard reading order).
    if num_expected is not None:
        if num_found != num_expected:
            print(f"WARNING: Mask voxels ({num_found}) != Model nodes ({num_expected}).")
            if num_found > num_expected:
                print(f"Truncating coordinates to first {num_expected}...")
                mni_coords = mni_coords[:num_expected]
            else:
                print("CRITICAL: Mask implies fewer nodes than model expects.")
    
    return mni_coords

def gather_visualization_data(model, dataset, mask_path, dataloader=None):
    """
    Extracts Key Components for Visualization.
    - Handles Multi-Head Attention shapes robustly.
    - Ensures Coords align with Node IDs.
    """
    print("--- Gathering Data for BrainNet ---")
    
    if hasattr(model, 'module'):
        m_ref = model.module
    else:
        m_ref = model
    m_ref.eval()
    
    # --- 1. Get Adjacency Matrix (Robust Shape Handling) ---
    device = next(m_ref.parameters()).device
    
    if dataloader is not None:
        try:
            # Get first batch
            batch_data = next(iter(dataloader))
            # Handle unpacking
            if isinstance(batch_data, (list, tuple)):
                x_real, stim_real = batch_data[0], batch_data[1]
                adj_real = dataset.adjacency.to(device)
            else:
                x_real = None
                
            if x_real is not None:
                x_real = x_real.to(device)
                stim_real = stim_real.to(device)
                num_nodes = x_real.shape[2] 
                #adj_real = torch.ones(num_nodes, num_nodes).to(device)
                
                print(f"  -> Using real batch input: {x_real.shape}")
                
                with torch.no_grad():
                    _, avg_attn, _, _ = m_ref(x_real, stim_real, adj_real)
                
                # avg_attn shape could be (Batch, N, N) or (Batch, Heads, N, N)
                # 1. Average over Batch
                adj_matrix = avg_attn.mean(dim=0) # -> (N, N) or (Heads, N, N)
                
                # 2. Average over Heads if they exist
                if adj_matrix.ndim == 3:
                    print(f"  -> Detected Multi-Head Attention {adj_matrix.shape}. Averaging heads...")
                    adj_matrix = adj_matrix.mean(dim=0) # -> (N, N)
                    
                adj_matrix = adj_matrix.cpu().numpy()
                
        except Exception as e:
            print(f"⚠️ Failed to use real batch: {e}")
            adj_matrix = np.eye(dataset.num_nodes)
    else:
        adj_matrix = np.eye(dataset.num_nodes)

    # --- 2. Get Coordinates (ID-Matched) ---
    print(f"Reading mask: {mask_path}")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    affine = mask_img.affine
    
    # We must match the order of nodes in the dataset (Sorted IDs)
    unique_ids = np.unique(mask_data)
    unique_ids = unique_ids[unique_ids != 0]
    unique_ids = np.sort(unique_ids) # Critical: Sort 1..306
    
    print(f"  -> Found {len(unique_ids)} unique regions in mask.")
    
    coords_list = []
    for region_id in unique_ids:
        # Find all voxels for this specific ID
        indices = np.argwhere(mask_data == region_id)
        if len(indices) > 0:
            # Center of mass
            center_index = indices.mean(axis=0)
            center_mni = nib.affines.apply_affine(affine, center_index)
            coords_list.append(center_mni)
        else:
            # Fallback (shouldn't happen)
            coords_list.append([0, 0, 0])
            
    coords = np.array(coords_list)
    
    # --- 3. Get Tau ---
    if hasattr(m_ref.cfc, 'get_tau_values'):
        tau_values = m_ref.cfc.get_tau_values().detach().cpu().numpy()
    else:
        tau_values = torch.sigmoid(m_ref.cfc.tau_system.weight).detach().cpu().numpy()

    # --- 4. Get Labels ---
    labels = dataset.region_labels
    
    return adj_matrix, coords, labels, tau_values

# utils.py

def visualize_prediction_dynamics(model, dataset, device, save_path):
    """
    Plots a 'Triad' of time-series: 
    1. The Biological Driver (Hand Knob / Region 2)
    2. The Best Model Fit (Highest Correlation)
    3. The Worst Model Fit (Lowest Correlation)
    """
    
    # [FIX 1] Handle Save Path (Folder vs File)
    # If the user passes a directory (config.RESULTS_DIR), append a filename.
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, "prediction_dynamics_triad.png")
    
    model.eval()
    
    found_stim = False
    inputs, targets, stims = None, None, None
    
    # Create a temporary loader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # [FIX 2] Robust Unpacking loop
    # We grab 'batch' first, then manually extract the first 3 items.
    # This ignores any extra items (like indices or masks) that might be in the dataset.
    
    adj = dataset.adjacency.to(device)
    
    for batch in loader:
        x = batch[0]          # fmri
        s = batch[1]          # stimulus
        #adj = batch[2]        # adjacency
        y = batch[2]          # target
    
        if s.sum() > 0:
            inputs, targets, stims, adjs = x.to(device), y.to(device), s.to(device), adj.to(device)
            found_stim = True
            break
    
    if not found_stim:
        inputs, targets, stims, adjs = x.to(device), y.to(device), s.to(device), adj.to(device)
    
    with torch.no_grad():
        preds, _, _, _ = model(inputs, stims, adjs)
    
    # 3. Prepare Data for Plotting
    history = inputs[0].cpu().numpy()     # (30, 305)
    future_true = targets[0].cpu().numpy() # (5, 305)
    future_pred = preds[0].cpu().numpy()   # (5, 305)
    
    # 4. Identify The Triad Nodes
    corrs = []
    num_nodes = history.shape[1]
    
    for i in range(num_nodes):
        # Concatenate history + future for correlation check
        # full_true = np.concatenate([history[:, i], future_true[:, i]]) # Optional full context
        
        # Calculate correlation on the FUTURE prediction (The Test)
        if np.std(future_pred[:, i]) < 1e-5: 
            c = 0
        else: 
            c = np.corrcoef(future_true[:, i], future_pred[:, i])[0,1]
        corrs.append(c)
        
    corrs = np.array(corrs)
    
    # Select Indices
    idx_motor = 2 # Fixed Ground Truth (Hand Knob) - Make sure this index exists!
    if idx_motor >= num_nodes: idx_motor = 0 # Safety fallback
        
    idx_best = np.nanargmax(corrs)
    idx_worst = np.nanargmin(corrs)
    
    indices = [idx_motor, idx_best, idx_worst]
    titles = [
        f"Biological Driver (Node {idx_motor})", 
        f"Best Fit (Node {idx_best})", 
        f"Worst Fit (Node {idx_worst})"
    ]
    
    # 5. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Time axis
    t_history = np.arange(0, 30)
    t_future = np.arange(30, 35)
    
    for ax, idx, title in zip(axes, indices, titles):
        # Plot History (Context)
        ax.plot(t_history, history[:, idx], color='gray', alpha=0.5, label="Context (Input)")
        
        # Plot True Future
        ax.plot(t_future, future_true[:, idx], color='black', linewidth=2, label="Actual Future")
        
        # Plot Predicted Future
        ax.plot(t_future, future_pred[:, idx], color='red', linestyle='--', linewidth=2, label="GBB Prediction")
        
        # Add Stimulus Shade
        if stims.sum() > 0:
            stim_trace = stims[0, :, 0].cpu().numpy()
            # Fill where stim > 0
            # We map stim time to t_history since stim aligns with input window
            ax.fill_between(t_history, -2, 2, where=(stim_trace > 0), color='green', alpha=0.1, label="Stimulus")

        ax.set_title(f"{title} | Window Corr: {corrs[idx]:.2f}")
        ax.set_ylabel("BOLD Signal (z-score)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Time Steps (TR)")
    plt.tight_layout()
    
    # Save safely
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Dynamics visualization saved to {save_path}")

def condense_to_anatomy(data_matrix, fine_labels, coarse_coords_file=None):
    """
    Aggregates a 305x305 matrix (or 305 vector) down to the unique anatomical regions found in labels.
    Returns:
        - condensed_data: (10, 10) or (10,)
        - unique_labels: list of 10 strings
        - coarse_coords: (10, 3) coordinates (if file provided)
    """
    if data_matrix.ndim == 2 and data_matrix.shape[0] == data_matrix.shape[1]:
        if data_matrix.shape[0] != len(fine_labels):
            raise ValueError(
                f"condense_to_anatomy: square matrix size {data_matrix.shape[0]} "
                f"does not match number of fine labels {len(fine_labels)}"
            )
    
    # 1. Identify Unique Regions (e.g., 'Precentral_L') and preserve order
    # We use a dict to preserve insertion order usually, or sort
    unique_labels = sorted(list(set(fine_labels)))
    n_coarse = len(unique_labels)
    
    # 2. Build Index Map
    # region_indices['Precentral_L'] = [0, 1, 2, 45...]
    region_indices = {label: [] for label in unique_labels}
    for idx, label in enumerate(fine_labels):
        region_indices[label].append(idx)
        
    # 3. Aggregate Data
    if data_matrix.ndim == 2 and data_matrix.shape[0] == data_matrix.shape[1]:
        # Adjacency Matrix (N, N) -> (10, 10)
        condensed_data = np.zeros((n_coarse, n_coarse))
        for i, r1 in enumerate(unique_labels):
            for j, r2 in enumerate(unique_labels):
                # Get sub-block of connections
                indices_i = region_indices[r1]
                indices_j = region_indices[r2]
                
                # Extract block: rows of r1, cols of r2
                block = data_matrix[np.ix_(indices_i, indices_j)]
                
                # Average strength
                condensed_data[i, j] = np.mean(block)
                
    elif data_matrix.ndim == 1:
        # Tau Vector (N,) -> (10,)
        condensed_data = np.zeros(n_coarse)
        for i, region in enumerate(unique_labels):
            indices = region_indices[region]
            condensed_data[i] = np.mean(data_matrix[indices])
            
    elif data_matrix.ndim == 2:
        # Spreading Activation (Time, Nodes) -> (Time, 10)
        # Assuming shape is (Time, 305) or (305, Time). We need (Time, 305) usually.
        # Let's standardize to (Time, Nodes)
        if data_matrix.shape[1] != len(fine_labels): 
            data_matrix = data_matrix.T
            
        time_steps = data_matrix.shape[0]
        condensed_data = np.zeros((time_steps, n_coarse))
        for i, region in enumerate(unique_labels):
            indices = region_indices[region]
            # Average activity of all nodes in this region
            condensed_data[:, i] = np.mean(data_matrix[:, indices], axis=1)
        
        # Transpose back for plotting (Regions, Time)
        condensed_data = condensed_data.T 

    # 4. Get Coarse Coordinates (Optional)
    coarse_coords = []
    if coarse_coords_file:
        # We use nilearn to find the centroids of the 10-region mask
        # Note: We rely on the sorted order of labels matching the sorted order of IDs in the mask
        # This is usually safe if specific AAL indices are used, but requires care.
        # Safer strategy: Calculate centroid of the Fine Nodes!
        
        for region in unique_labels:
            # We don't have the 3D coordinates here easily passed in.
            # So we will rely on the caller passing the coarse file,
            # OR we just return None and let plotting handle it if passed explicitly.
            pass
            
        # Simpler approach for BrainNet: Use the coarse mask file directly in the main script
        # to get coordinates, assuming the order matches 'unique_labels'.
        # Because we sorted unique_labels, we must sort the mask IDs too.
        try:
            # Extract centroids from the coarse mask file
            # This is robust if labels are alphabetical? No.
            # Let's return None and handle coords in the main flow to be safe.
            pass
        except:
            pass

    return condensed_data, unique_labels

def visualize_results(model, ref_ds, world_size, all_runs, node_vals, test_loader, nowstring):
    print("\n=== Starting Hybrid Visualization Pipeline ===")
    
    # STREAM 1: HIGH-RESOLUTION (306 Nodes) - The "Mechanistic Atlas"
    # ===============================================================
    # We use the dataset's internal masker (ref_ds.masker) which matches the training data (306 nodes).
    # We use the input mask from all_runs[0] (306 nodes).
    print(f"1. Generating High-Res Mechanistic NIfTI Maps (using {config.MASK_FILE})...")
    
    # This generates: Map_Tau.nii, Map_Control_Authority.nii, Map_Complexity.nii
    if hasattr(model, 'module'): m_ref = model.module
    else: m_ref = model
    
    save_mechanistic_atlas(
        model=m_ref,
        mask_path=all_runs[0]['mask'],
        node_vals=node_vals,
        output_dir=config.RESULTS_DIR,
        masker=ref_ds.masker,
        dataloader=test_loader,
        datestring=nowstring
    )
    
    # ---------------------------------------------------------
    # 2. Generating Condensed Network Graphs
    # ---------------------------------------------------------
    print(f"2. Generating Network Graphs...")
    
    # A. Gather 305-Node Data
    adj_matrix_305, coords_305, labels_305, tau_305 = gather_visualization_data(
        model, ref_ds, config.MASK_FILE, dataloader=test_loader
    )
    
    # B. CONDENSE (305 -> 10)
    print("   -> Condensing 305 nodes to Anatomical Regions for readability...")
    adj_10, labels_10 = condense_to_anatomy(adj_matrix_305, labels_305)
    tau_10, _         = condense_to_anatomy(tau_305, labels_305)

    # C. Get Coordinates for the 10 Regions (from the Coarse Mask)
    # This ensures the dots appear in the center of the anatomical lobe
    # [FIX] Dynamic Mask Selection
    # If the model has ~305 nodes, use the high-res mask. 
    # If it has ~10 nodes, use the low-res mask.
    if ref_ds.num_nodes > 50:
        print(f"  -> High-Res Model detected ({ref_ds.num_nodes} nodes). Using Physics Mask for plotting.")
        plot_mask = config.MASK_FILE
    else:
        print(f"  -> Low-Res Model detected ({ref_ds.num_nodes} nodes). Using ROI Mask for plotting.")
        plot_mask = config.ROI_MASK_FILE

    # Get coordinates from the selected mask
    coords_10 = plotting.find_parcellation_cut_coords(plot_mask)
    
    # Safety Check: Ensure coords match data
    if len(coords_10) != len(node_vals):
        print(f"⚠️ Mismatch: Coords={len(coords_10)}, Data={len(node_vals)}. Truncating to match.")
        min_len = min(len(coords_10), len(node_vals))
        coords_10 = coords_10[:min_len]
        node_vals = node_vals[:min_len]
    
    
    #coords_10 = plotting.find_parcellation_cut_coords(config.ROI_MASK_FILE)

    # D. Plot BrainNet (Using 10 Nodes)
    save_path = os.path.join(config.RESULTS_DIR, "network_overview.png")
    plotting.plot_connectome(
        adjacency_matrix=adj_10,
        node_coords=coords_10,
        node_color=tau_10,
        node_size=80,      # Bigger dots for regions
        edge_threshold='80%', # Show only strongest regional links
        edge_cmap='hot',
        display_mode='lzr',
        colorbar=True,
        title="Effective Connectivity (Region Level)",
        output_file=save_path
    )
    
    # E. Chord Diagram (Using 10 Nodes)
    visualize_chord_diagram(adj_matrix_305, coords_305, os.path.join(config.RESULTS_DIR, "chord_diagram_connectivity.png"))


    # F. Tau Matrix (Using 10 Nodes)
    # This will now be a clean 10x10 matrix of connection strengths
    visualize_tau_sorted_matrix(adj_10, tau_10, labels_10, config.RESULTS_DIR)
    
    # D. BrainNet Viewer Files (Node/Edge) - Condensed version
    export_to_brainnet(
        adj_matrix_305, 
        coords_305, 
        labels_305, 
        tau_305, 
        config.RESULTS_DIR
    )
    

    # ---------------------------------------------------------
    # 3. Spreading Activation (Condensed)
    # ---------------------------------------------------------
    if config.SENSORY_REGIONS:
        print("3. Generating Spreading Activation...")
        # Get the 305-node activation map from the model
        diff_map_305, sorted_names, latencies = simulate_spreading_activation(
            model, 
            ref_ds, 
            duration=config.WINDOW_SIZE)
        
        # Condense to 10 Regions
        # Input: (Time, Nodes) or (Nodes, Time) -> Handled by function
        # Returns: (Regions, Time) for heatmap
        diff_map_10, labels_unique = condense_to_anatomy(diff_map_305, labels_305)
        
        save_path = os.path.join(config.RESULTS_DIR, "Spreading_Activation_Cascade.png")
        plot_spreading_activation(diff_map_10, labels_unique, save_path)
    
    # NIfTI Map: Latency (Seconds)
    # Must use the High-Res masker to map the 306 latency values back to the brain.
    print("   -> Saving Latency NIfTI...")
    try:
        latency_img = ref_ds.masker.inverse_transform(latencies.reshape(1, -1))
        latency_img.to_filename(os.path.join(config.RESULTS_DIR, "Map_Signal_Latency.nii"))
    except Exception as e:
        print(f"⚠️ Could not save Latency NIfTI: {e}")

    print("=== Visualization Complete ===")

def visualize_chord_diagram(adj_matrix, node_coords, save_path):
    """
    Condenses the 305-node High-Res matrix into a 10-ROI Low-Res Chord Diagram.
    """
    
    print("  -> Generating Chord Diagram (Aggregating 305 nodes to 10 ROIs)...")
    
    # 1. Load the Low-Res Visualization Mask (10 ROIs)
    try:
        plot_mask_img = nib.load(getattr(config, "PLOT_MASK_FILE", config.ROI_MASK_FILE))
    except:
        print("Warning: PLOT_MASK_FILE not found. Skipping Chord Diagram.")
        return

    # 2. Map every High-Res Node (305) to a Low-Res ROI (10)
    # We sample the 10-ROI mask at the coordinates of the 305 nodes
    # resampling_coords is shape (305, 3)
    node_roi_labels = []
    
    # Get data from the plotting mask
    plot_data = plot_mask_img.get_fdata()
    plot_affine = plot_mask_img.affine
    
    # Convert MNI coords (mm) to Voxel indices in the Plot Mask
    
    unique_rois = set()
    node_to_roi_map = [] # Stores which ROI index (0-9) each node belongs to

    for i in range(len(node_coords)):
        # x,y,z in mm
        x, y, z = node_coords[i]
        # x,y,z in voxel space
        vx, vy, vz = image.coord_transform(x, y, z, np.linalg.inv(plot_affine))
        
        # Read the value at that voxel (rounded to nearest int)
        try:
            val = int(plot_data[int(round(vx)), int(round(vy)), int(round(vz))])
        except:
            val = 0 # Out of bounds
            
        node_roi_labels.append(val)
        if val > 0: unique_rois.add(val)
        node_to_roi_map.append(val)

    # 3. Create the Condensed 10x10 Matrix
    sorted_rois = sorted(list(unique_rois)) # e.g., [1, 2, ... 10]
    num_plot_rois = len(sorted_rois)
    
    condensed_adj = np.zeros((num_plot_rois, num_plot_rois))
    
    # Map ROI Value -> Matrix Index (e.g., ROI Label 25 -> Index 0)
    val_to_idx = {val: i for i, val in enumerate(sorted_rois)}
    
    # Sum the weights
    # If Node A (in ROI 1) connects to Node B (in ROI 2), add weight to Matrix[1,2]
    rows, cols = adj_matrix.shape
    for r in range(rows):
        for c in range(cols):
            weight = adj_matrix[r, c]
            if weight > 0:
                roi_r = node_to_roi_map[r]
                roi_c = node_to_roi_map[c]
                
                # Ignore background (0)
                if roi_r in val_to_idx and roi_c in val_to_idx:
                    idx_r = val_to_idx[roi_r]
                    idx_c = val_to_idx[roi_c]
                    condensed_adj[idx_r, idx_c] += weight

    # 4. Normalize (Optional: Average instead of Sum to prevent huge numbers)
    # condensed_adj /= rows  # simple scaling

    # 5. Plot the Condensed Matrix
    # Now we have a clean 10x10 matrix!
    import matplotlib.pyplot as plt
    from mne.viz import plot_connectivity_circle
    
    # Define names for the 10 ROIs (You can customize these based on your _10.nii labels)
    # Placeholder names:
    roi_names = [f"ROI_{i}" for i in sorted_rois]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plot_connectivity_circle(condensed_adj, roi_names, ax=ax, show=False)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"     Saved aggregated chord diagram to {save_path}")

def visualize_tau_sorted_matrix(adj_matrix, tau_values, labels, save_dir):
    """
    Plots the Adjacency Matrix sorted by Tau.
    Accepts pre-calculated matrix/tau to avoid model attribute errors.
    """
    print("--- Generating Tau-Sorted Adjacency Matrix ---")
    
    
    # 1. Handle Dimension Mismatch (306 Nodes vs 10 Labels)
    n_matrix = adj_matrix.shape[0]
    n_labels = len(labels)
    
    final_adj = adj_matrix
    final_tau = tau_values
    final_labels = labels

    if n_matrix != n_labels:
        print(f"  -> Adjustment: Matrix({n_matrix}) vs Labels({n_labels})")
        if n_matrix > n_labels:
            # Slice matrix/tau to match the mask labels
            final_adj = adj_matrix[:n_labels, :n_labels]
            final_tau = tau_values[:n_labels]
        elif n_labels > n_matrix:
            # Slice labels to match matrix
            final_labels = labels[:n_matrix]
            
    # 2. Sort Indices by Tau (Fastest to Slowest)
    # Fast (Sensory) -> Top-Left
    # Slow (Association) -> Bottom-Right
    sorted_indices = np.argsort(final_tau)
    
    # Reorder Matrix and Labels
    adj_sorted = final_adj[sorted_indices][:, sorted_indices]
    labels_sorted = [final_labels[i] for i in sorted_indices]
    tau_sorted = np.array(final_tau)[sorted_indices]
    
    # 3. Plot
    plt.figure(figsize=(15, 12))
    
    # Log scale helps see weak connections
    sns.heatmap(
        np.log1p(adj_sorted), 
        cmap='viridis',
        xticklabels=labels_sorted, 
        yticklabels=labels_sorted
    )
    
    plt.title("Effective Connectivity Sorted by Tau (Time Scale)\nLeft=Fast(Sensory) -> Right=Slow(Deep)")
    plt.xlabel("Target Region (Sorted by Tau)")
    plt.ylabel("Source Region (Sorted by Tau)")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Save
    save_path = os.path.join(save_dir, "Tau_Sorted_Connectivity.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved Tau-Sorted Matrix: {save_path}")

def visualize_network(model, mask_img_path, save_path=None):
    """
    Visualizes the learned effective connectivity on a glass brain.
    """
    print("--- Generating Tempo-Structural Connectome (Voxel-wise) ---")
    import matplotlib.pyplot as plt
    from nilearn import plotting
    
    # Handle DDP wrapping
    if hasattr(model, 'module'):
        m_ref = model.module
    else:
        m_ref = model

    # 1. Get Number of Nodes
    if hasattr(m_ref.cfc, 'num_nodes'):
        num_nodes_model = m_ref.cfc.num_nodes
    else:
        num_nodes_model = m_ref.cfc.tau_system.weight.shape[0]

    print(f"Model has {num_nodes_model} nodes.")

    # 2. Get Coordinates
    try:
        print(f"--- Extracting Voxel Coordinates from: {mask_img_path} ---")
        mask_img = nib.load(mask_img_path)
        mask_data = mask_img.get_fdata()
        # Find coordinates of all non-zero voxels (assuming mask is ROI labelled or binary)
        # If mask is binary, this gives all voxels. If ROI map, it gives all.
        # We assume the model nodes map 1-to-1 with these coordinates in order.
        coords = np.array(np.where(mask_data != 0)).T
        
        # We need to transform indices to affine coordinates
        affine = mask_img.affine
        coords_mni = nib.affines.apply_affine(affine, coords)
        
        # Downsample or match? 
        # Crucial: The number of coords found must match num_nodes_model.
        # If mask has 5000 voxels but model has 306 nodes, we have a problem unless
        # the mask is an ATLAS (values 1..306).
        
        # Assuming the mask is an ATLAS (values 1..N):
        unique_labels = np.unique(mask_data)
        unique_labels = unique_labels[unique_labels != 0] # Remove background
        
        if len(unique_labels) == num_nodes_model:
            # Calculate Center of Mass for each ROI
            roi_coords = []
            for label in sorted(unique_labels):
                roi_indices = np.argwhere(mask_data == label)
                center = roi_indices.mean(axis=0)
                center_mni = nib.affines.apply_affine(affine, center)
                roi_coords.append(center_mni)
            coords_plot = np.array(roi_coords)
            print(f"Found {len(coords_plot)} ROIs matching model nodes.")
        else:
            # Fallback: Just take the first N voxels (Warning: this might be wrong if not voxel-wise)
            print(f"⚠️ Mask voxels ({len(coords)}) != Model Nodes ({num_nodes_model}). Using first {num_nodes_model} voxels.")
            coords_plot = coords_mni[:num_nodes_model]

    except Exception as e:
        print(f"⚠️ Coordinate extraction failed: {e}")
        return

    # 3. Get Tau Values (Node Colors) - FIXED
    try:
        # Use the getter method!
        raw_tau = m_ref.cfc.get_tau_values().detach().cpu().numpy()
        
        # Normalize for color mapping (e.g. 0 to 10s)
        # S1 is fast (low tau), Frontal is slow (high tau)
        node_colors = raw_tau
    except Exception as e:
        print(f"⚠️ Tau extraction failed: {e}")
        node_colors = np.ones(num_nodes_model)

    # 4. Get Adjacency (Edge Strength)
    try:
        # We need a dummy forward pass to get the attention weights OR load saved
        # Using saved is safer if training just finished
        adj_path = os.path.join(os.path.dirname(save_path), "adjacency_matrix.npy")
        if os.path.exists(adj_path):
            adj_matrix = np.load(adj_path)
        else:
            # Dummy forward pass logic (omitted for brevity, use saved if possible)
            print("⚠️ No saved adjacency found. Skipping edges.")
            adj_matrix = np.zeros((num_nodes_model, num_nodes_model))
            
        # Threshold for plotting (Top 1%)
        threshold = np.percentile(adj_matrix, 99.5)
        adj_plot = adj_matrix.copy()
        adj_plot[adj_plot < threshold] = 0
        
    except Exception as e:
        print(f"⚠️ Adjacency extraction failed: {e}")
        adj_plot = np.zeros((num_nodes_model, num_nodes_model))

    # 5. Plot
    try:
        print(f"Plotting Connectome to {save_path}...")
        fig = plt.figure(figsize=(12, 6))
        plotting.plot_connectome(
            adjacency_matrix=adj_plot,
            node_coords=coords_plot,
            node_color=node_colors,
            node_size=20,
            edge_cmap='hot',
            display_mode='lzr',
            colorbar=True,
            title=f"Learned Tempo-Structural Connectome (Tau-Colored)",
            output_file=save_path
        )
        print("✅ Network visualization saved.")
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

def export_to_brainnet(adj_matrix, coords, labels, tau_values, save_dir):
    """
    Exports .node and .edge files for BrainNet Viewer (or SurfPlot).
    """
    print("--- Exporting for 3D Visualization ---")
    flat_adj = np.sort(adj_matrix.flatten())
    
    # Let's say we want at least the top 100 edges for a nice plot
    top_k = 30 
    if len(flat_adj) > top_k:
        threshold = flat_adj[-top_k] # The 100th strongest edge
    else:
        threshold = 0.0001
        
    # Safety: If threshold is 0, make it tiny positive
    if threshold == 0: 
        threshold = 1e-9
    
    # 1. Save .node file
    # Format: X Y Z Color Size Label
    node_file = os.path.join(save_dir, "network.node")
    with open(node_file, "w") as f:
        for i in range(len(labels)):
            x, y, z = coords[i]
            color = tau_values[i] # Color by Tau
            size = np.sum(adj_matrix[:, i]) # Size by In-Degree (Hubness)
            label = labels[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {color:.4f} {size:.4f} {label}\n")
            
    # 2. Save .edge file
    # Format: NxN matrix text
    edge_file = os.path.join(save_dir, "network.edge")
    
    # Threshold for 3D view (Keep it sparse!)
    #thresh = np.percentile(adj_matrix, 99.8) # Top 0.2%
    adj_sparse = adj_matrix.copy()
    adj_sparse[adj_sparse < threshold] = 0
    
    np.savetxt(edge_file, adj_sparse, fmt='%.6f', delimiter='\t')
    
    print(f"✅ Exported {node_file} and {edge_file}")
    print("   -> Open these in BrainNet Viewer or use 'netplotbrain' in Python.")

def export_scalar_map_to_csv(fmri_data, mask_path, output_csv, level='region'):
    """
    Converts NIfTI data to a CSV with MNI coordinates.
    
    Args:
        fmri_path: Path to the 4D functional NIfTI file.
        mask_path: Path to the 3D ROI mask/atlas file (activations or labels).
        output_csv: Where to save the result.
        level: 'region' (averages signal per ROI) or 'voxel' (dumps every single voxel).
    """
    print(f"--- Saving csv ---")
    print(f"Mask: {mask_path}")

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    affine = mask_img.affine

    unique_ids = np.unique(mask_data)
    unique_ids = unique_ids[unique_ids != 0]
    unique_ids.sort()

    # --- NEW: handle nodewise vectors directly ---
    arr = np.asarray(fmri_data)

    # Accept (N,), (1,N), or (N,1)
    if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
        arr = arr.reshape(-1)
        if len(arr) != len(unique_ids):
            raise ValueError(
                f"Region-vector length mismatch! Vector: {len(arr)}, Regions in mask: {len(unique_ids)}"
            )

        results = []
        for i, region_id in enumerate(unique_ids):
            region_mask = (mask_data == region_id)
            indices = np.argwhere(region_mask)
            center_index = indices.mean(axis=0)
            mni_coord = nib.affines.apply_affine(affine, center_index)

            row = {
                'Region_ID': int(region_id),
                'MNI_X': round(mni_coord[0], 2),
                'MNI_Y': round(mni_coord[1], 2),
                'MNI_Z': round(mni_coord[2], 2),
                't_0': float(arr[i]),
            }
            results.append(row)

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ Success! Saved to {output_csv}")
        print(f"Preview:\n{df.iloc[:3, :5]}")
        return

    # --- existing volume path below ---
    if arr.shape[:3] != mask_data.shape:
        raise ValueError(f"Shape mismatch! fMRI: {arr.shape}, Mask: {mask_data.shape}")

    fmri_data = arr
    ...
    if fmri_data.shape[:3] != mask_data.shape:
        raise ValueError(f"Shape mismatch! fMRI: {fmri_data.shape}, Mask: {mask_data.shape}")

    results = []
    
    # 2. Identify Regions (Non-zero values in mask)
    # We assume mask contains integer labels (1, 2, 3...) for regions
    unique_ids = np.unique(mask_data)
    unique_ids = unique_ids[unique_ids != 0] # Exclude background
    unique_ids.sort()
    
    print(f"Found {len(unique_ids)} regions/nodes to extract.")

    for region_id in unique_ids:
        # Get boolean mask for this specific region
        region_mask = (mask_data == region_id)
        
        # Get Grid Indices (i, j, k) of all voxels in this region
        # indices shape: (N_voxels, 3)
        indices = np.argwhere(region_mask)
        
        # ---------------------------------------------------------
        # OPTION A: REGION LEVEL (Centroid) - Recommended for AI
        # ---------------------------------------------------------
        if level == 'region':
            # Calculate Center of Mass (average index)
            center_index = indices.mean(axis=0) # [i_avg, j_avg, k_avg]
            
            # Apply Affine: Convert Grid Index -> MNI Coordinate (mm)
            # Matrix multiplication: [x,y,z] = Affine @ [i,j,k,1]
            mni_coord = nib.affines.apply_affine(affine, center_index)
            
            # Extract Time Series (Average over all voxels in region)
            # fmri_data[region_mask] returns (N_voxels, Time) -> mean -> (Time,)
            time_series = fmri_data[region_mask].mean(axis=0)
            
            # Create Row
            row = {
                'Region_ID': int(region_id),
                'MNI_X': round(mni_coord[0], 2),
                'MNI_Y': round(mni_coord[1], 2),
                'MNI_Z': round(mni_coord[2], 2),
            }
            # Add Timepoints (t0, t1, t2...)
            for t, val in enumerate(time_series):
                row[f't_{t}'] = val
                
            results.append(row)

        # ---------------------------------------------------------
        # OPTION B: VOXEL LEVEL (All Voxels) - Warning: Huge File
        # ---------------------------------------------------------
        elif level == 'voxel':
            # Convert ALL indices to MNI at once
            mni_coords = nib.affines.apply_affine(affine, indices)
            
            # Extract activations for these voxels
            voxel_activations = fmri_data[region_mask] # (N_voxels, Time)
            
            for v_idx in range(len(indices)):
                row = {
                    'Region_ID': int(region_id),
                    'MNI_X': round(mni_coords[v_idx][0], 2),
                    'MNI_Y': round(mni_coords[v_idx][1], 2),
                    'MNI_Z': round(mni_coords[v_idx][2], 2),
                }
                # Add timepoints
                for t, val in enumerate(voxel_activations[v_idx]):
                    row[f't_{t}'] = val
                results.append(row)

    # 3. Save to Pandas CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Saved to {output_csv}")
    print(f"Preview:\n{df.iloc[:3, :6]}") # Show first 3 rows, first 6 cols

    
def save_mechanistic_atlas(model, mask_path, node_vals, output_dir=config.RESULTS_DIR,
                           masker=None, dataloader=None, num_batches=10, datestring=''):
    """
    Extracts learnable parameters (Tau, CfC intrinsic drive, connectivity, complexity,
    basis specificity) and saves them as NIfTI/text/NumPy files for visualization.

    Key fixes:
    - Uses real dataloader batches + real adjacency for exported connectivity
    - Separates CfC intrinsic excitability from graph-derived net-flow
    - Computes complexity as a scalar per node
    - Computes tuning specificity over FastKAN basis functions
    """
    print(f"Saving Mechanistic Atlas to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 0. Resolve wrapped model + device
    # -------------------------------
    if hasattr(model, 'module'):
        real_model = model.module
    else:
        real_model = model

    device = next(real_model.parameters()).device
    real_model.eval()

    # -------------------------------
    # 1. Make sure we have a fitted masker
    # -------------------------------
    if masker is None:
        try:
            masker = NiftiLabelsMasker(
                labels_img=mask_path,
                standardize=False,
                detrend=False
            )
            masker.fit()
        except Exception as e:
            print(f"⚠️ Could not initialize/fix masker automatically: {e}")
            masker = None

    # Helper to robustly convert vectors
    def _to_numpy_1d(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().reshape(-1)
        return np.asarray(x).reshape(-1)

    mapdir='maps_'+datestring
    mapdirpath=os.path.join(output_dir,mapdir)
    if not os.path.exists(mapdirpath):
        os.makedirs(mapdirpath)

    # -------------------------------
    # 2. Tau map + intrinsic CfC drive
    # -------------------------------
    try:
        tau_vector = real_model.cfc.get_tau_values().detach().cpu().numpy().reshape(-1)
        np.savetxt(os.path.join(output_dir, "tau_values.txt"), tau_vector)

        drive_logit = real_model.cfc.get_causal_drive().detach().cpu().numpy().reshape(-1)
        np.savetxt(os.path.join(output_dir, "cfc_intrinsic_excitability_logit.txt"), drive_logit)
  
        if masker is not None:
            try: 
                img_tau = masker.inverse_transform(tau_vector.reshape(1, -1))
                img_tau.to_filename(os.path.join(output_dir,mapdir ,"Map_CfC_Tau_s.nii"))
                # legacy alias for older downstream scripts
                img_tau.to_filename(os.path.join(output_dir,mapdir ,"Map_Tau.nii.gz"))
            
                img_drive = masker.inverse_transform(drive_logit.reshape(1, -1))
                img_drive.to_filename(os.path.join(output_dir, mapdir, "Map_CfC_IntrinsicExcitabilityLogit.nii"))
                print("  -> Saved Map_CfC_Tau_s.nii")
            except Exception as e:
                print(f"⚠️ Could not save Tau / CfC intrinsic drive NifTi maps: {e}")
            try:
                export_scalar_map_to_csv(tau_vector, mask_path, os.path.join(output_dir,mapdir ,"Map_CfC_Tau_s.csv"), level='region')
                export_scalar_map_to_csv(drive_logit, mask_path, os.path.join(output_dir,mapdir ,"Map_CfC_IntrinsicExcitabilityLogit.csv"), level='region')
                print("  -> Saved Map_CfC_IntrinsicExcitabilityLogit.nii")
            except Exception as e:
                print(f"⚠️ Could not save Tau / CfC intrinsic drive CSV maps: {e}")
        else:
            print("⚠️ Masker unavailable, skipped NIfTI export for Tau / CfC intrinsic drive.")

    except Exception as e:
        print(f"⚠️ Could not save Tau / CfC intrinsic drive Nifti/CSV maps: {e}")

    # -------------------------------
    # 3. Effective connectivity export from REAL data
    # -------------------------------
    try:
        adj_matrix = None
        node_in_usage = None
        node_out_usage = None
        node_basis_usage = None

        with torch.no_grad():
            if dataloader is not None:
                attn_acc = None
                n = 0
                
                ds_ref = dataloader.dataset
                adj = ds_ref.adjacency.to(device)
                
                for batch in itertools.islice(iter(dataloader), num_batches):
                    #expected dataset order: x, stim, y
                    x, stim, y = batch
                    x = x.to(device)
                    stim = stim.to(device)
                    #adj = adj.to(device)

                    _, avg_attn, _, _ = real_model(x, stim, adj=adj)

                    # average over batch -> (N, N)
                    A = avg_attn.mean(dim=0)
                    attn_acc = A if attn_acc is None else (attn_acc + A)
                    n += 1
                    
                    # Usage summaries from real forward traffic
                    if node_in_usage is None:
                        N_nodes = A.shape[0]
                        node_in_usage = torch.zeros(N_nodes, device=device)
                        node_out_usage = torch.zeros(N_nodes, device=device)
                        node_basis_usage = torch.zeros(N_nodes, config.KAN_BASIS_FUNCTIONS, device=device)
                    
                    node_in_usage += A.sum(dim=1)
                    node_out_usage += A.sum(dim=0)
                    
                    # Node-wise basis usage from real KAN pairwise RBF activations
                    if hasattr(real_model, "last_kan_input") and len(real_model.kan_layers) > 0:
                        layer0 = real_model.kan_layers[0]
                        h = real_model.last_kan_input  # (B, N, C)
                        if h is not None and h.ndim == 3:
                            B, N, C = h.shape
                            w = layer0.pair_proj.weight.squeeze(0)
                            w_i = w[:C].unsqueeze(0)
                            w_j = w[C:].unsqueeze(0)
                            b = layer0.pair_proj.bias.view(1, 1, 1)
                            
                            score_i = F.linear(h, w_i)  # (B,N,1)
                            score_j = F.linear(h, w_j)  # (B,N,1)
                            x_pair = score_i.unsqueeze(2) + score_j.unsqueeze(1) + b  # (B,N,N,1)
                            
                            rbf = torch.exp(
                                -((x_pair - layer0.mu.view(1, 1, 1, -1)) ** 2) /
                                (2 * (layer0.sigma.view(1, 1, 1, -1) ** 2))
                            )  # (B,N,N,G)
                            
                            # node_basis_usage[target_node, basis]
                            usage_A = torch.abs(avg_attn)
                            node_basis_usage += torch.einsum('bnmg,bnm->ng', rbf, usage_A)

                if attn_acc is None:
                    raise RuntimeError("No batches available from dataloader for atlas export.")

                adj_matrix = (attn_acc / max(n, 1)).detach().cpu().numpy()
                node_in_usage = (node_in_usage / max(n, 1)).detach().cpu().numpy()
                node_out_usage = (node_out_usage / max(n, 1)).detach().cpu().numpy()
                node_basis_usage = (node_basis_usage / max(n, 1)).detach().cpu().numpy()

            else:
                # fallback only if no dataloader is available
                num_nodes = len(tau_vector)
                dummy_x = torch.zeros(1, config.WINDOW_SIZE, num_nodes, device=device)
                dummy_stim = torch.zeros(1, config.WINDOW_SIZE, 1, device=device)
                dummy_adj = torch.eye(num_nodes, device=device)

                _, avg_attn, _, _ = real_model(dummy_x, dummy_stim, adj=dummy_adj)
                adj_matrix = avg_attn[0].detach().cpu().numpy()
                print("⚠️ Dataloader not provided; connectivity exported from fallback dummy batch.")
                
                node_in_usage = np.sum(adj_matrix, axis=1)
                node_out_usage = np.sum(adj_matrix, axis=0)
                node_basis_usage = None
        
        if node_in_usage is not None:
            node_in_usage = _to_numpy_1d(node_in_usage)
    
        if node_out_usage is not None:
            node_out_usage = _to_numpy_1d(node_out_usage)
    
        if node_basis_usage is not None:
            if isinstance(node_basis_usage, torch.Tensor):
                node_basis_usage = node_basis_usage.detach().cpu().numpy()
            else:
                node_basis_usage = np.asarray(node_basis_usage)

        np.save(os.path.join(output_dir, "adjacency_matrix.npy"), adj_matrix)
        np.savetxt(os.path.join(output_dir, "effective_connectivity.txt"), adj_matrix)
        print("  -> Saved adjacency_matrix.npy")
        print("  -> Saved effective_connectivity.txt")

        # Save conduction delays if they exist
        if hasattr(real_model.kan_layers[0], 'conduction_delays'):
            delays = torch.abs(real_model.kan_layers[0].conduction_delays).detach().cpu().numpy()
            np.save(os.path.join(output_dir, "edge_attenuation_gates.npy"), delays)
            print("  -> Saved edge_attenuation_gates.npy")
        else:
            print("⚠️ Could not save conduction_delays.npy (not found on first KAN layer).")

    except Exception as e:
        print(f"⚠️ Could not save Connectivity Matrix: {e}")

    # -------------------------------
    # 4. Connectivity gain / contribution map
    # -------------------------------
    try:
        contrib_vector = _to_numpy_1d(node_vals)
    
        if masker is not None:
            try:
                contrib_img = masker.inverse_transform(contrib_vector.reshape(1, -1))
                contrib_img.to_filename(os.path.join(output_dir, mapdir ,"Map_ConnectivityGain_Index.nii"))
            
                # legacy alias
                contrib_img.to_filename(os.path.join(output_dir, mapdir ,"Connectome_Contribution_Map.nii"))
            except Exception as e:
                print(f"⚠️ Could not save Connectivity Gain Nifti map: {e}")    
            
            try:
                export_scalar_map_to_csv(contrib_vector, mask_path, os.path.join(output_dir, mapdir ,"Map_ConnectivityGain_Index.csv"))
                print("  -> Saved Map_ConnectivityGain_Index.nii")
            except Exception as e:
                print(f"⚠️ Could not save Connectivity Gain CSV map: {e}") 
        else:
            print("⚠️ Masker unavailable, skipped connectivity gain NIfTI export.")

    except Exception as e:
        print(f"⚠️ Could not save Connectivity Gain Nifti/CSV map: {e}")

    # -------------------------------
    # 5. Tau-weighted connectivity gain (legacy STII replacement)
    # -------------------------------
    try:
        contrib_vector = _to_numpy_1d(node_vals)
        tau_vector = real_model.cfc.get_tau_values().detach().cpu().numpy().reshape(-1)

        tau_weighted_gain = tau_vector * contrib_vector

        if masker is not None:
            try:
                stii_img = masker.inverse_transform(tau_weighted_gain.reshape(1, -1))
                stii_img.to_filename(os.path.join(output_dir,mapdir ,"Map_TauWeighted_ConnectivityGain.nii"))
                # legacy alias
                stii_img.to_filename(os.path.join(output_dir,mapdir,"Spatiotemporal_Integration_Map.nii"))
                print("  -> Saved Map_TauWeighted_ConnectivityGain.nii")
            except Exception as e:    
                  print(f"⚠️ Could not save Tau-Weighted Connectivity Gain Nifti map: {e}")
            try:
                export_scalar_map_to_csv(tau_weighted_gain, mask_path, os.path.join(output_dir, mapdir,"Map_TauWeighted_ConnectivityGain.csv"))
                print("  -> Saved Map_TauWeighted_ConnectivityGain.csv")
            except Exception as e:    
                print(f"⚠️ Could not save Tau-Weighted Connectivity Gain CSV map: {e}")
        else:
            print("⚠️ Masker unavailable, skipped tau-weighted connectivity gain NIfTI export.")

    except Exception as e:
        print(f"⚠️ Could not save Tau-Weighted Connectivity Gain Nifti/CSV map: {e}")


    # -------------------------------
    # 6. Graph-derived directed net-flow / source-sink map
    # -------------------------------
    try:
        adj_path = os.path.join(output_dir, "adjacency_matrix.npy")
        if not os.path.exists(adj_path):
            raise FileNotFoundError("adjacency_matrix.npy not found. Cannot calculate Attention Net-Flow Index.")

        W_eff = np.load(adj_path)

        # collapse heads if needed
        if W_eff.ndim == 3:
            W_eff = np.mean(W_eff, axis=0)

        W_signed = W_eff
        W_abs = np.abs(W_signed)
        
        signed_in = W_signed.sum(axis=1)
        signed_out = W_signed.sum(axis=0)
        
        mass_in = W_abs.sum(axis=1)
        mass_out = W_abs.sum(axis=0)
        
        signed_net = signed_out - signed_in
        netflow_stable = signed_net / (mass_in + mass_out + 1e-6)


        ## Convention in your code/comments:
        ## W_eff[i, j] = attention from j -> i
        #in_degree = np.sum(W_eff, axis=1)   # row sum: inputs to i
        #out_degree = np.sum(W_eff, axis=0)  # col sum: outputs from j

        #denom = out_degree + in_degree + 1e-9
        #netflow = (out_degree - in_degree) / denom

        if masker is not None:
            try:
                img = masker.inverse_transform(signed_in.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_Attention_SignedInFlow.nii"))
                print("  -> Saved Map_Attention_SignedInFlow.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Map_Attention_SignedInFlow map: {e}")
            try:                
                export_scalar_map_to_csv(signed_in, mask_path, os.path.join(output_dir, mapdir ,"Map_Attention_SignedInFlow.csv"))
                print("  -> Saved Map_Attention_SignedInFlow.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Map_Attention_SignedInFlow map: {e}")
             
                
            try:
                img = masker.inverse_transform(signed_out.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_Attention_SignedOutFlow.nii"))
                print("  -> Saved Map_Attention_SignedOutFlow.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Map_Attention_SignedOutFlow map: {e}")
            try:                
                export_scalar_map_to_csv(signed_out, mask_path, os.path.join(output_dir, mapdir ,"Map_Attention_SignedOutFlow.csv"))
                print("  -> Saved Map_Attention_SignedOutFlow.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Map_Attention_SignedOutFlow map: {e}")  


            try:
                img = masker.inverse_transform(netflow_stable.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_Attention_SignedNetFlow_Stable.nii"))
                print("  -> Saved Map_Attention_SignedNetFlow_Stable.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Map_Attention_SignedNetFlow_Stable map: {e}")
            try:                
                export_scalar_map_to_csv(netflow_stable, mask_path, os.path.join(output_dir, mapdir ,"Map_Attention_SignedNetFlow_Stable.csv"))
                print("  -> Saved Map_Attention_SignedNetFlow_Stable.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Map_Attention_SignedNetFlow_Stable map: {e}")  


            try:
                img = masker.inverse_transform(mass_in.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_Attention_InMass.nii"))
                print("  -> Saved Map_Attention_InMass.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Map_Attention_InMass map: {e}")
            try:                
                export_scalar_map_to_csv(mass_in, mask_path, os.path.join(output_dir, mapdir ,"Map_Attention_InMass.csv"))
                print("  -> Saved Map_Attention_InMass.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Map_Attention_InMass map: {e}")       
                
            try:
                img = masker.inverse_transform(mass_out.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_Attention_OutMass.nii"))
                print("  -> Saved Map_Attention_OutMass.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Map_Attention_OutMass map: {e}")
            try:                
                export_scalar_map_to_csv(mass_out, mask_path, os.path.join(output_dir, mapdir ,"Map_Attention_OutMass.csv"))
                print("  -> Saved Map_Attention_InMass.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Map_Attention_OutMass map: {e}")   
 

        else:
            print("⚠️ Masker unavailable, skipped attention net-flow NIfTI export.")

    except Exception as e:
        print(f"⚠️ Could not save Nifti/CSV Attention Net-Flow map: {e}")
    # -------------------------------
    # 7. FastKAN complexity map (L2 magnitude per node)
    # -------------------------------
    try:
        layer = real_model.kan_layers[0]

        # spline_coeffs expected shape: (Out, In, Head, Basis)
        
        complexity = layer.spline_coeffs.norm(p=2, dim=(1, 2, 3)).detach().cpu().numpy().reshape(-1)
        usage_strength = 0.5 * (mass_in + mass_out)
        usage_strength = usage_strength / (usage_strength.mean() + 1e-8)
        complexity_usage = complexity * usage_strength

        if masker is not None:
            try:
                img = masker.inverse_transform(complexity.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_FastKAN_Complexity_L2.nii"))
                # legacy alias
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_Computational_Complexity.nii"))
                print("  -> Saved Map_FastKAN_Complexity_L2.nii")
            except Exception as e:
                print(f"⚠️ Could not save L2 Complexity Nifti Map: {e}")
            try:    
                export_scalar_map_to_csv(complexity, mask_path, os.path.join(output_dir, mapdir,"Map_FastKAN_Complexity_L2.csv"))
                print("  -> Saved Map_FastKAN_Complexity_L2.csv")
            except Exception as e:
                print(f"⚠️ Could not save L2 Complexity CSV Map: {e}")
            try: 
                img_u = masker.inverse_transform(complexity_usage.reshape(1, -1))
                img_u.to_filename(os.path.join(output_dir, mapdir, "Map_FastKAN_Complexity_UsageWeighted.nii"))
                print("  -> Saved Map_FastKAN_Complexity_UsageWeighted.nii")
            except Exception as e:
                print(f"⚠️ Could not save Complexity (Usage-weighted) Nifti Map: {e}")
            try:     
                export_scalar_map_to_csv(complexity_usage, mask_path, os.path.join(output_dir, mapdir, "Map_FastKAN_Complexity_UsageWeighted.csv"))
                print("  -> Saved Map_FastKAN_Complexity_UsageWeighted.csv")
            except Exception as e:
                print(f"⚠️ Could not save Complexity (Usage-weighted) CSV Map: {e}")    
        else:
            print("⚠️ Masker unavailable, skipped FastKAN complexity map export.")

    except Exception as e:
        print(f"⚠️ Could not save Complexity Nifti/CSV Map: {e}")



    # -------------------------------
    # 8. FastKAN basis specificity map
    # -------------------------------
    try:
        layer = real_model.kan_layers[0]

        # (Out, In, Head, Basis) -> average over In + Head -> (Out, Basis)
        w = torch.abs(layer.spline_coeffs).mean(dim=(1, 2))

        # normalize over Basis
        p = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        entropy = -(p * torch.log(p + 1e-9)).sum(dim=1)

        # specificity = 1 - normalized entropy over basis count
        tuning_specificity = 1.0 - (entropy / np.log(config.KAN_BASIS_FUNCTIONS))
        tuning_specificity = tuning_specificity.detach().cpu().numpy().reshape(-1)
        
        # usage-weighted specificity from real batch basis activations
        if node_basis_usage is not None:
            p_usage = node_basis_usage / (node_basis_usage.sum(axis=1, keepdims=True) + 1e-9)
            ent_usage = -np.sum(p_usage * np.log(p_usage + 1e-9), axis=1)
            tuning_specificity_usage = 1.0 - (ent_usage / np.log(config.KAN_BASIS_FUNCTIONS))
        else:
            tuning_specificity_usage = tuning_specificity

        if masker is not None:
            try:
                img = masker.inverse_transform(tuning_specificity.reshape(1, -1))
                img.to_filename(os.path.join(output_dir,mapdir ,"Map_FastKAN_BasisSpecificity.nii"))
                # legacy alias
                img.to_filename(os.path.join(output_dir, mapdir ,"Map_Tuning_Specificity.nii"))
                print("  -> Saved Map_FastKAN_BasisSpecificity.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Tuning Specificity map: {e}")
            try:            
                export_scalar_map_to_csv(tuning_specificity, mask_path, os.path.join(output_dir, mapdir, "Map_FastKAN_BasisSpecificity.csv"))
                print("  -> Saved Map_FastKAN_BasisSpecificity.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Tuning Specificity map: {e}")
            try:
                img_u = masker.inverse_transform(tuning_specificity_usage.reshape(1, -1))
                img_u.to_filename(os.path.join(output_dir, mapdir, "Map_FastKAN_BasisSpecificity_UsageWeighted.nii"))
                print("  -> Saved Map_FastKAN_BasisSpecificity_UsageWeighted.nii")
            except Exception as e:
                print(f"⚠️ Could not save Nifti Tuning Specificity (Usage-weighted) map: {e}")
            try:
                export_scalar_map_to_csv(tuning_specificity_usage, mask_path, os.path.join(output_dir, mapdir, "Map_FastKAN_BasisSpecificity_UsageWeighted.csv"))
                print("  -> Saved Map_FastKAN_BasisSpecificity_UsageWeighted.csv")
            except Exception as e:
                print(f"⚠️ Could not save CSV Tuning Specificity (Usage-weighted) map: {e}")
        else:
            print("⚠️ Masker unavailable, skipped basis specificity map export.")

    except Exception as e:
        print(f"⚠️ Could not save Nifti/CSV Tuning Specificity map: {e}")

    print("Atlas generation complete.")

def export_map_stability(model, dataloader, output_dir, datestring="", num_batches=10):
    """
    Export pairwise map-stability correlations across available checkpoints.
    Compares: current in-memory, checkpoint_best, checkpoint_latest.
    """
    if hasattr(model, "module"):
        real_model = model.module
    else:
        real_model = model
    device = next(real_model.parameters()).device
    
    def _collect_maps():
        maps = {}
        maps["tau"] = real_model.cfc.get_tau_secs().detach().cpu().numpy().reshape(-1)
        maps["intrinsic_drive"] = real_model.cfc.get_causal_drive().detach().cpu().numpy().reshape(-1)
        
        if dataloader is not None:
            with torch.no_grad():
                ds_ref = dataloader.dataset
                adj = ds_ref.adjacency.to(device, non_blocking=True)
                attn_acc = None
                n = 0
                for batch in itertools.islice(iter(dataloader), num_batches):
                    x, stim, _ = batch
                    x = x.to(device)
                    stim = stim.to(device)
                    _, avg_attn, _, _ = real_model(x, stim, adj)
                    A = avg_attn.mean(dim=0)
                    attn_acc = A if attn_acc is None else (attn_acc + A)
                    n += 1
                if attn_acc is not None:
                    W = (attn_acc / max(n, 1)).detach().cpu().numpy()
                    in_degree = np.sum(W, axis=1)
                    out_degree = np.sum(W, axis=0)
                    netflow = (out_degree - in_degree) / (out_degree + in_degree + 1e-9)
                    maps["attention_netflow"] = netflow.reshape(-1)
        return maps
    
    original_state = {k: v.detach().cpu().clone() for k, v in real_model.state_dict().items()}
    variants = {"current": None}
    best_path = os.path.join(output_dir, "checkpoint_best.pth")
    latest_path = os.path.join(output_dir, "checkpoint_latest.pth")
    if os.path.exists(best_path):
        variants["best"] = best_path
    if os.path.exists(latest_path):
        variants["latest"] = latest_path
    
    maps_by_variant = {}
    variant_epochs = {"current": None}
    for tag, ckpt_path in variants.items():
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            real_model.load_state_dict(state, strict=False)
            variant_epochs[tag] = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
        real_model.eval()
        maps_by_variant[tag] = _collect_maps()
    
    real_model.load_state_dict(original_state, strict=False)
    real_model.eval()
    
    rows = []
    tags = list(maps_by_variant.keys())
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a_tag, b_tag = tags[i], tags[j]
            shared_maps = sorted(set(maps_by_variant[a_tag].keys()) & set(maps_by_variant[b_tag].keys()))
            for map_name in shared_maps:
                va = maps_by_variant[a_tag][map_name]
                vb = maps_by_variant[b_tag][map_name]
                if va.shape != vb.shape:
                    continue
                pear = np.corrcoef(va, vb)[0, 1]
                spear = scipy.stats.spearmanr(va, vb).correlation
                rows.append({
                    "variant_a": a_tag,
                    "epoch_a": variant_epochs.get(a_tag, None),
                    "variant_b": b_tag,
                    "epoch_b": variant_epochs.get(b_tag, None),
                    "map_name": map_name,
                    "pearson_r": float(pear),
                    "spearman_rho": float(spear),
                    "n_nodes": int(va.shape[0]),
                })
    
    if rows:
        out_name = f"map_stability_summary_{datestring}.csv" if datestring else "map_stability_summary.csv"
        out_path = os.path.join(output_dir, out_name)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  -> Saved map stability summary: {out_path}")
    
def simulate_spreading_activation(model, dataset, duration=None, pulse_time=5):
    """
    Performs 'In Silico' stimulation of the Sensory Cortex (S1) 
    and tracks the signal propagation.
    """
    print("--- Running Virtual Cortical Stimulation Experiment ---")
    
    # 1. Setup
    device = next(model.parameters()).device
    model.eval()
    
    # FIX: Use the model's native window size (usually 30)
    # If we feed 50, the linear layer crashes.
    if duration is None:
        import config
        duration = config.WINDOW_SIZE 
        print(f"  -> Using duration={duration} (from config.WINDOW_SIZE)")

    # Get node count
    if hasattr(model, 'module'):
        m_ref = model.module
    else:
        m_ref = model
    
    # Correct node count
    if hasattr(dataset, 'num_nodes'):
        num_nodes = dataset.num_nodes
    else:
        num_nodes = int(m_ref.sensory_mask.numel())
    
    # 2. Create Baseline (The Resting Brain)
    # Shape: (Batch=1, Window, Nodes)
    x_rest = torch.zeros(1, duration, num_nodes, device=device) 
    
    # Stimulus channel
    stim_null = torch.zeros(1, duration, 1).to(device)
    
    # Adjacency (Identity)
    adj = dataset.adjacency.to(device)

    # 3. Create Stimulation (The Poked Brain)
    stim_pulse = torch.zeros(1, duration, 1).to(device)
    # Create a pulse at the specified time
    # Ensure we don't go out of bounds
    t_start = min(pulse_time, duration - 2)
    stim_pulse[:, t_start:t_start+2, :] = 5.0 
    
    # 4. Run Simulation
    with torch.no_grad():
        # The model output is likely the *Prediction* or the *Latent State* sequence
        # We assume the model returns (Batch, Time, Nodes) or (Batch, Nodes)
        out_base, _, _, _ = model(x_rest, stim_null, adj)
        out_stim, _, _, _ = model(x_rest, stim_pulse, adj)
        
    # 5. Calculate "Causal Effect"
    # If output is just a prediction vector (e.g. 5 steps future), we visualize that.
    # If output is the full sequence, we visualize that.
    
    # Check shape to handle (Batch, Nodes, Time) vs (Batch, Time, Nodes)
    diff = (out_stim - out_base).detach().cpu().numpy()
    
    # Squeeze batch
    diff = diff[0] # -> (Time, Nodes) or (Nodes, Time) or (Horizon, Nodes)
    
    # Heuristic to ensure (Nodes, Time) for plotting
    if diff.shape[0] != num_nodes and diff.shape[1] == num_nodes:
        # It's (Time, Nodes) -> Transpose
        diff = diff.T
        
    # 6. Sort Regions by Latency
    # We find the time index of max activation
    time_to_peak = np.argmax(np.abs(diff), axis=1)
    sorted_indices = np.argsort(time_to_peak)
    
    # Reorder
    diff_sorted = diff[sorted_indices, :]
    labels_sorted = [dataset.region_labels[i] for i in sorted_indices]
    
    return diff_sorted, labels_sorted, time_to_peak

def plot_spreading_activation(activation_map, labels, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Plot Heatmap
    # X-axis = Time (TRs)
    # Y-axis = Regions (Sorted by Latency)
    sns.heatmap(
        activation_map, 
        cmap="RdBu_r", # Red = Excitation, Blue = Inhibition
        center=0,
        yticklabels=labels,
        xticklabels=5,
        cbar_kws={'label': 'Signal Change (Delta)'}
    )
    
    plt.title("Propagation of Synthetic Stimulus (S1 Injection)\nTop=Early Responders -> Bottom=Late Responders")
    plt.xlabel("Time (TRs) after Injection")
    plt.ylabel("Cortical Regions (Sorted by Reaction Time)")
    
    # Draw a line at the injection time
    plt.axvline(x=5, color='green', linestyle='--', label='Stimulus Onset')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved Spreading Activation Movie to {save_path}")
    




ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

BATCH_RE = re.compile(
    r"Mean Output Activity:\s*(?P<mean_act>[-+0-9.eE]+)\s*\|"
    r"Max Output Activity:\s*(?P<max_act>[-+0-9.eE]+)\|"
    r"\s*Corr:\s*(?P<corr>[-+0-9.eE]+)\s*\|"
    r"\s*PredStd:\s*(?P<pred_std>[-+0-9.eE]+)\|"
    r"\s*MSE:\s*(?P<mse>[-+0-9.eE]+)\|"
    r"\s*Corr_loss:\s*(?P<corr_loss>[-+0-9.eE]+)\|"
    r"\s*Var_loss:\s*(?P<var_loss>[-+0-9.eE]+)\|"
    r"\s*Derivate loss:\s*(?P<derivative_loss>[-+0-9.eE]+)\s*\|"
    r"\s*Metabolic:\s*(?P<metabolic>[-+0-9.eE]+)\s*\|"
    r"\s*Wiring loss:\s*(?P<wiring_loss>[-+0-9.eE]+)\s*\|"
    r"\s*Sparseness:\s*(?P<sparseness>[-+0-9.eE]+)\s*\|"
    r"\s*Smoothness:\s*(?P<smoothness>[-+0-9.eE]+)\|"
    r"\s*l_group:\s*(?P<l_group>[-+0-9.eE]+)\s*\|"
    r"\s*Long term memory \(tau\) loss:\s*(?P<longterm_tau>[-+0-9.eE]+)\s*\|"
    r"\s*tau diversity loss:\s*(?P<tau_diversity>[-+0-9.eE]+)\s*\|"
    r"\s*temporal orthogonality loss:\s*(?P<temporal_orthogonality>[-+0-9.eE]+)"
)

DENSITY_RE = re.compile(
    r"Density \(>0\.01\):\s*(?P<density1>[-+0-9.eE]+)%\s*\|\s*Density \(>0\.001\):\s*(?P<density2>[-+0-9.eE]+)%"
)

EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*Train:\s*(?P<train>[-+0-9.eE]+)\s*\|\s*Test:\s*(?P<test>[-+0-9.eE]+)\s*\|\s*Corr:\s*(?P<val_corr>[-+0-9.eE]+)"
)


LOSS_KEYS = [
    "mse",
    "corr_loss",
    "var_loss",
    "derivative_loss",
    "metabolic",
    "wiring_loss",
    "sparseness",
    "smoothness",
    "l_group",
    "longterm_tau",
    "tau_diversity",
    "temporal_orthogonality",
]


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s).strip()


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def parse_log(log_path: Path):
    lines = [strip_ansi(line) for line in log_path.read_text(errors="ignore").splitlines()]
    lines = [line for line in lines if line]

    epochs = []
    current_batches = []
    current_epoch_idx = 0

    def finalize_epoch(density1=None, density2=None):
        nonlocal current_batches, current_epoch_idx, epochs
        if not current_batches:
            # still create empty epoch slot if density exists
            if density1 is None and density2 is None:
                return
            epochs.append({
                "epoch": current_epoch_idx,
                "mean_act": np.nan,
                "max_act": np.nan,
                "corr": np.nan,
                "pred_std": np.nan,
                "density1": density1,
                "density2": density2,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "val_corr": np.nan,
                "loss_components": {k: np.nan for k in LOSS_KEYS},
            })
            current_epoch_idx += 1
            return

        epoch_record = {
            "epoch": current_epoch_idx,
            "mean_act": mean_or_nan([b["mean_act"] for b in current_batches]),
            "max_act": mean_or_nan([b["max_act"] for b in current_batches]),
            "corr": mean_or_nan([b["corr"] for b in current_batches]),
            "pred_std": mean_or_nan([b["pred_std"] for b in current_batches]),
            "density1": density1,
            "density2": density2,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_corr": np.nan,
            "loss_components": {k: mean_or_nan([b[k] for b in current_batches]) for k in LOSS_KEYS},
        }
        epochs.append(epoch_record)
        current_batches = []
        current_epoch_idx += 1

    for line in lines:
        m = BATCH_RE.search(line)
        if m:
            batch = {k: float(v) for k, v in m.groupdict().items()}
            current_batches.append(batch)
            continue

        m = DENSITY_RE.search(line)
        if m:
            density1 = float(m.group("density1"))
            density2 = float(m.group("density2"))
            finalize_epoch(density1, density2)
            continue

        m = EPOCH_RE.search(line)
        if m:
            e = int(m.group("epoch"))
            # Attach validation summary to explicit epoch if it exists,
            # otherwise to the most recent finalized epoch.
            target_idx = e if e < len(epochs) else max(len(epochs) - 1, 0)
            if len(epochs) == 0:
                epochs.append({
                    "epoch": e,
                    "mean_act": np.nan,
                    "max_act": np.nan,
                    "corr": np.nan,
                    "pred_std": np.nan,
                    "density1": np.nan,
                    "density2": np.nan,
                    "train_loss": float(m.group("train")),
                    "val_loss": float(m.group("test")),
                    "val_corr": float(m.group("val_corr")),
                    "loss_components": {k: np.nan for k in LOSS_KEYS},
                })
            else:
                epochs[target_idx]["train_loss"] = float(m.group("train"))
                epochs[target_idx]["val_loss"] = float(m.group("test"))
                epochs[target_idx]["val_corr"] = float(m.group("val_corr"))
            continue

    # finalize any trailing partial epoch
    if current_batches:
        finalize_epoch(np.nan, np.nan)

    return epochs


def nearest_epoch_index(epochs, target_epoch):
    epoch_nums = np.array([ep["epoch"] for ep in epochs], dtype=float)
    return int(np.argmin(np.abs(epoch_nums - target_epoch)))


def build_bottom_contribution_data(epochs, total_epochs):
    checkpoint_specs = [0.25, 0.50, 0.75, 1.00]
    selections = []

    for frac in checkpoint_specs:
        target_epoch = max(0, int(round(frac * total_epochs)) - 1)
        idx = nearest_epoch_index(epochs, target_epoch)
        ep = epochs[idx]
        contrib = {k: max(0.0, float(ep["loss_components"].get(k, np.nan))) for k in LOSS_KEYS}
        total = sum(v for v in contrib.values() if np.isfinite(v))
        if total <= 0:
            shares = {k: 0.0 for k in LOSS_KEYS}
        else:
            shares = {k: (v / total) for k, v in contrib.items()}
        selections.append({
            "fraction": frac,
            "target_epoch": target_epoch,
            "actual_epoch": ep["epoch"],
            "shares": shares,
        })

    return selections


def make_figure(epochs, total_epochs, output_path: Path, title: str ):
    if not epochs:
        raise RuntimeError("No epochs could be parsed from the log.")

    epoch_nums = np.array([ep["epoch"] for ep in epochs], dtype=float)
    if total_epochs <= 1:
        x_pct = epoch_nums
        x_label = "Epoch"
    else:
        x_pct = 100.0 * epoch_nums / (total_epochs - 1)
        x_label = "Training progress (% of configured epochs)"

    corr_train = np.array([ep["corr"] for ep in epochs], dtype=float)
    corr_val = np.array([ep["val_corr"] for ep in epochs], dtype=float)

    mean_act = np.array([ep["mean_act"] for ep in epochs], dtype=float)
    max_act = np.array([ep["max_act"] for ep in epochs], dtype=float)

    density1 = np.array([ep["density1"] for ep in epochs], dtype=float)
    density2 = np.array([ep["density2"] for ep in epochs], dtype=float)

    checkpoints = build_bottom_contribution_data(epochs, total_epochs)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.25], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0])

    # 1) Correlation
    ax1.plot(x_pct, corr_train, label="Train batch correlation")
    ax1.plot(x_pct, corr_val, label="Validation correlation")
    ax1.set_ylabel("Correlation")
    ax1.set_title("Correlation over training")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Activity
    ax2.plot(x_pct, mean_act, label="Mean output activity")
    ax2.plot(x_pct, max_act, label="Max output activity")
    ax2.set_ylabel("Activity")
    ax2.set_title("Output activity over training")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3) Densities
    ax3.plot(x_pct, density1, label="Density (>0.01)")
    ax3.plot(x_pct, density2, label="Density (>0.001)")
    ax3.set_ylabel("Density (%)")
    ax3.set_xlabel(x_label)
    ax3.set_title("Effective graph density over training")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4) Loss contribution bars
    x = np.arange(len(checkpoints))
    bottom = np.zeros(len(checkpoints), dtype=float)

    for key in LOSS_KEYS:
        vals = np.array([cp["shares"][key] * 100.0 for cp in checkpoints], dtype=float)
        ax4.bar(x, vals, bottom=bottom, label=key)
        bottom += vals

    labels = [
        f"{int(cp['fraction']*100)}%\n(target E{cp['target_epoch']}, used E{cp['actual_epoch']})"
        for cp in checkpoints
    ]
    ax4.set_xticks(x, labels)
    ax4.set_ylabel("Contribution to logged loss sum (%)")
    ax4.set_title("Relative loss-component contributions at 25%, 50%, 75%, and 100% of training")
    ax4.grid(True, axis="y", alpha=0.3)

    # Compact legend outside plot
    ax4.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.suptitle(title or f"Training dynamics summary: {output_path.stem}", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def visualize_training_dynamics(log_path, output_path, total_epochs, title=None):
    epochs = parse_log(Path(log_path))
    make_figure(epochs, total_epochs, Path(output_path), title)

#def main():
#    parser = argparse.ArgumentParser(description="Visualize GBB training log dynamics.")
#    parser.add_argument("--log", required=True, help="Path to training log txt file.")
#    parser.add_argument("--output", required=True, help="Path to output PNG/PDF.")
#    parser.add_argument("--total-epochs", type=int, default=200,
#                        help="Configured total epochs used to map x-axis to 0-100%%. Default: 200.")
#    parser.add_argument("--title", default=None, help="Optional figure title.")
#    args = parser.parse_args()

#    log_path = Path(args.log)
#    output_path = Path(args.output)

#    epochs = parse_log(log_path)
#    make_figure(epochs, args.total_epochs, output_path, args.title)
#    print(f"Saved visualization to {output_path}")


#if __name__ == "__main__":
#    main()
    

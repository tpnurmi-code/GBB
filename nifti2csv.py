# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 16:42:28 2026

@author: 35840
"""

import nibabel as nib
import numpy as np
import pandas as pd
import os

def export_nifti_to_csv(fmri_path, mask_path, output_csv, level='region'):
    """
    Converts NIfTI data to a CSV with MNI coordinates.
    
    Args:
        fmri_path: Path to the 4D functional NIfTI file.
        mask_path: Path to the 3D ROI mask/atlas file (activations or labels).
        output_csv: Where to save the result.
        level: 'region' (averages signal per ROI) or 'voxel' (dumps every single voxel).
    """
    print(f"--- Loading Data ---")
    print(f"fMRI: {fmri_path}")
    print(f"Mask: {mask_path}")
    
    # 1. Load NIfTI files
    fmri_img = nib.load(fmri_path)
    mask_img = nib.load(mask_path)
    
    fmri_data = fmri_img.get_fdata() # Shape: (X, Y, Z, Time)
    mask_data = mask_img.get_fdata() # Shape: (X, Y, Z)
    affine = mask_img.affine       # The Magic Matrix (Index -> MNI)
    
    # Ensure shapes match (ignoring time dimension)
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

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Update these paths to your actual files
    filenames=["Map_Attention_NetFlow_Index", 'Map_FastKAN_BasisSpecificity', 'Map_Tuning_Specificity', 'Map_Computational_Complexity', 'Map_FastKAN_Complexity_L2', 'Connectome_Contribution_Map', 'Map_ConnectivityGain_Index', 'Map_TauWeighted_ConnectivityGain', 'Spatiotemporal_Integration_Map', 'Map_CfC_IntrinsicExcitabilityLogit']
    #filenames=["Causal_Drive", 'Map_Computational_Complexity', 'Map_Tuning_Specificity', 'Spatiotemporal_Integration_Map']

    for filename in filenames:
        base_dir = r"G:\Projects\AI\data\results\TAU_AND_CORR_CONTR_MAP_4_1_2026_BEST_SO_FAR_BEST"
        base_dir_mask = r"G:\Projects\AI\data"
        base_dir_save=r"G:\Projects\AI\data\results\TAU_AND_CORR_CONTR_MAP_4_1_2026_BEST_SO_FAR_BEST" 
        
        # Example: Converting the first run
        fmri_file = os.path.join(base_dir, (filename + ".nii"))
        mask_file = os.path.join(base_dir_mask, "group_roi_mask.nii") # Your 305-region atlas
        
        save_file = os.path.join(base_dir_save, (filename+".csv"))
        
        # Run extraction (Region level is best for analysis)
        # If you want every single voxel, change level='voxel'
        try:
            export_nifti_to_csv(fmri_file, mask_file, save_file, level='region')
        except Exception as e:
            print(e)
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 21:58:19 2026

@author: 35840
"""

import nibabel as nib
import numpy as np
import pandas as pd
import os

def export_nifti_to_csv(fmri_path, mask_path, output_csv, level='region'):
    print(f"--- Loading Data ---")
    print(f"Input: {fmri_path}")
    print(f"Mask:  {mask_path}")
    
    # 1. Load NIfTI files
    fmri_img = nib.load(fmri_path)
    mask_img = nib.load(mask_path)
    
    fmri_data = fmri_img.get_fdata() 
    mask_data = mask_img.get_fdata() 
    affine = mask_img.affine       

    # --- FIX: Handle 3D (Static) vs 4D (Time-Series) ---
    is_4d = (len(fmri_data.shape) == 4)
    if not is_4d:
        print("  -> Input is 3D (Static Map). extracting single scalar per region.")
    else:
        print(f"  -> Input is 4D (Time Series). Frames: {fmri_data.shape[-1]}")

    results = []
    
    # 2. Identify Regions
    unique_ids = np.unique(mask_data)
    unique_ids = unique_ids[unique_ids != 0]
    unique_ids.sort()
    
    print(f"Found {len(unique_ids)} regions/nodes to extract.")

    for region_id in unique_ids:
        region_mask = (mask_data == region_id)
        indices = np.argwhere(region_mask)
        
        # Calculate Centroid (MNI)
        center_index = indices.mean(axis=0)
        mni_coord = nib.affines.apply_affine(affine, center_index)
        
        row = {
            'Region_ID': int(region_id),
            'MNI_X': round(mni_coord[0], 2),
            'MNI_Y': round(mni_coord[1], 2),
            'MNI_Z': round(mni_coord[2], 2),
        }

        # --- FIX: Extraction Logic ---
        if level == 'region':
            if is_4d:
                # Average over voxels, keep time
                time_series = fmri_data[region_mask].mean(axis=0)
                for t, val in enumerate(time_series):
                    row[f't_{t}'] = val
            else:
                # Average over voxels, single scalar
                val = fmri_data[region_mask].mean()
                row['value'] = val
                
            results.append(row)

    # 3. Save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Saved to {output_csv}")
    print(df.iloc[:3])

# --- USAGE ---
if __name__ == "__main__":
    # Example usage based on your error
    base_dir = r"G:\Projects\AI\data"
    
    # Use the FILE you want to inspect as 'fmri_path'
    # Even if it is a mask file, this new script will handle it.
    file_to_inspect = os.path.join(base_dir, "group_roi_mask_10.nii") 
    mask_file = os.path.join(base_dir, "group_roi_mask.nii") 
    
    save_file = os.path.join(base_dir, "debug_mask_values.csv")
    
    export_nifti_to_csv(file_to_inspect, mask_file, save_file)
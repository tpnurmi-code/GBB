# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 04:00:51 2025

@author: 35840
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
from nilearn import plotting, datasets
# Load the file back
output_dir="G:\\Projects\\AI\\data\\results"

# 1. Load your data
img = nib.load(os.path.join(output_dir, "Connectome_ConductionVelocity.dconn.nii"))
matrix = img.get_fdata()

# 2. Get Coordinates for your 306 nodes
# (Assuming you are using the Schaefer/AAL atlas as discussed previously)
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400)
coords = plotting.find_parcellation_cut_coords(atlas.maps)
# Truncate to match your 306 nodes
coords = coords[:matrix.shape[0]]

# 3. Plot
# We threshold to show only the strongest/fastest connections to avoid a messy blob
plotting.plot_connectome(
    adjacency_matrix=matrix,
    node_coords=coords,
    edge_cmap="turbo",
    edge_threshold="95%", # Only show top 5% strongest connections
    title="Conduction Velocity Network",
    colorbar=True
)
plt.show()





plt.figure()


img = nib.load(os.path.join(output_dir, "Connectome_ConductionVelocity.dconn.nii"))
data = img.get_fdata()

print(f"Shape: {data.shape}") # Should be (306, 306) or similar
print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")

# Plot it quickly
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='turbo', origin='lower')
plt.colorbar(label='Conduction Velocity')
plt.title("Conduction Velocity Matrix")
plt.show()
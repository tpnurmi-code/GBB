# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 16:28:55 2025

@author: 35840
"""

import torch
import os
import numpy as np
import config
from models import MesocortGBB
from dataset import NiftiLaminarDataset
from utils import save_mechanistic_atlas, visualize_network

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.CFC_PHYSICS_MODE = "biophysical"

# Initialize
print("Loading Data...")
ds = NiftiLaminarDataset(
    data_dir=config.DATA_DIR,
    subject_id="S14", 
    window_size=config.WINDOW_SIZE
)

print("Loading Model...")
model = MesocortGBB(num_nodes=ds.num_nodes, time_points=config.WINDOW_SIZE).to(device)
checkpoint = torch.load(os.path.join(config.RESULTS_DIR, "checkpoint_latest.pth"), map_location=device)

# Load Weights
state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Run
print("Generating Atlas...")
save_mechanistic_atlas(model, ds, config.MASK_FILE, config.RESULTS_DIR)

print("Visualizing Network...")
visualize_network(model, config.MASK_FILE, save_path=os.path.join(config.RESULTS_DIR, 'final_connectome.png'))
import os
import random
import numpy as np
import config
from dataset import NiftiLaminarDataset
from models import MesocortGBB
from utils import get_subject_files, save_mechanistic_atlas, visualize_network, visualize_chord_diagram, visualize_tau_sorted_matrix , export_to_brainnet, apply_time_masking
import utils
import math
from datetime import datetime
import time

import nibabel as nib  # type: ignore[import-not-found]

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter 
import torch.nn.functional as F

from contextlib import nullcontext

from tqdm import tqdm
from datetime import datetime

BOLD_GREEN = "\033[1;32m"
BOLD_WHITE = "\033[1;37m"
COLOR_OFF= "\033[0,30m"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_distributed():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        if 'MASTER_ADDR' not in os.environ: os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ: os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

class PearsonCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        # (B,T,N) -> (B*T, N)
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        vx = pred - pred.mean(dim=0, keepdim=True)
        vy = target - target.mean(dim=0, keepdim=True)

        corr = (vx * vy).sum(dim=0) / (
            torch.sqrt((vx**2).sum(dim=0)) *
            torch.sqrt((vy**2).sum(dim=0)) + 1e-8
        )

        # Fisher z-transform (stabilizes gradients)
        corr = torch.clamp(corr, -0.999, 0.999)
        z = 0.5 * torch.log((1 + corr) / (1 - corr))

        return 1.0 - z.mean()

class VarianceLoss(nn.Module):
    def __init__(self, target_variance=1.0):
        super().__init__()
        self.target_variance = target_variance
    def forward(self, x):
        var = x.var(dim=1).mean() 
        return F.mse_loss(var, torch.tensor(self.target_variance, device=x.device))

def derivative_loss(pred, target):
    diff_pred = pred[:, 1:] - pred[:, :-1]
    diff_target = target[:, 1:] - target[:, :-1]
    return F.mse_loss(diff_pred, diff_target)

def calculate_group_lasso(model, column_ids, alpha=0.5):
    """
    Robust Elastic Net Group Lasso.
    Regularizes the SIGNED Causal Drive to enforce 'Inhibition' (Negative) vs 'Excitation' (Positive).
    """
    if hasattr(model, "module"): raw_model = model.module
    else: raw_model = model

    # 1. Check Groups
    unique_cols = torch.unique(column_ids)
    if len(unique_cols) <= 1:
        device = raw_model.cfc.tau_system.weight.device
        return torch.tensor(0.0, device=device)

    # 2. GET SIGNED CAUSAL DRIVE (The vector to regularize)
    # We prefer 'get_causal_drive' (Signed) over 'get_tau_values' (Positive)
    # because Elastic Net needs signed values to create the "Negative Sea".
    if hasattr(raw_model.cfc, 'get_causal_drive'):
        vals = raw_model.cfc.get_causal_drive()
    elif hasattr(raw_model.cfc, 'get_tau_values'):
        # Fallback (Strictly positive, less ideal for inhibition maps)
        vals = raw_model.cfc.get_tau_values()
    else:
        return torch.tensor(0.0, device=raw_model.cfc.tau_system.weight.device)

    # 3. HANDLE DIMENSIONS (The Bug Fix)
    if vals.dim() == 3: # (Batch, Nodes, Time)
        spatial_map = vals.mean(dim=[0, 2])
    elif vals.dim() == 2: # (Batch, Nodes)
        spatial_map = vals.mean(dim=0)
    elif vals.dim() == 1: 
        # [FIX] It is already (Nodes,). Do NOT mean() or reshape(-1) blindly.
        spatial_map = vals 
    else:
        spatial_map = vals.reshape(-1)

    # Safety Check
    if spatial_map.shape[0] != column_ids.shape[0]:
         # print(f"DEBUG: Shape Mismatch. Map: {spatial_map.shape}, Cols: {column_ids.shape}")
         return torch.tensor(0.0, device=vals.device)

    # 4. APPLY ELASTIC NET
    total_loss = 0.0
    for col_id in unique_cols:
        mask = (column_ids == col_id)
        group_drive = spatial_map[mask]
        
        # L1 (Sparsity) - Forces noise to 0
        l1_term = torch.sum(torch.abs(group_drive))
        
        # L2 (Grouping) - Forces neighbors to share values
        l2_term = torch.sqrt(torch.sum(group_drive ** 2) + 1e-8)
        
        total_loss += (alpha * l1_term) + ((1 - alpha) * l2_term)
        
    return total_loss / len(unique_cols)

def calculate_temporal_orthogonality_loss(pred_tensor, num_nodes=None):
    """
    Penalize extreme redundancy across node predictions.

    Accepts:
    - (B, T, N)
    - (B, N, T)
    """
    if num_nodes is None:
        num_nodes = getattr(config, "NUM_NODES_EXPECTED", None)

    if num_nodes is not None:
        if pred_tensor.shape[-1] == num_nodes:
            pred_tensor = pred_tensor.permute(0, 2, 1)  # (B,N,T)
        elif pred_tensor.shape[1] == num_nodes:
            pass  # already (B,N,T)
        else:
            raise ValueError(
                f"Cannot infer node dimension in tensor shape {pred_tensor.shape}; expected num_nodes={num_nodes}"
            )
    else:
        # fallback: assume smaller dimension is time
        if pred_tensor.shape[1] < pred_tensor.shape[2]:
            pass
        else:
            pred_tensor = pred_tensor.permute(0, 2, 1)

    B, N, T = pred_tensor.shape

    node_profiles = pred_tensor.permute(1, 0, 2).reshape(N, -1)
    node_profiles = node_profiles - node_profiles.mean(dim=1, keepdim=True)
    node_profiles_norm = node_profiles / (node_profiles.norm(dim=1, keepdim=True) + 1e-8)

    corr_mat = torch.mm(node_profiles_norm, node_profiles_norm.t())
    eye = torch.eye(N, device=corr_mat.device, dtype=corr_mat.dtype)
    off_diag = corr_mat * (1.0 - eye)

    return torch.mean(torch.relu(torch.abs(off_diag) - 0.8))

def get_effective_density(model, threshold=0.01):
    if hasattr(model, "module"): raw_model = model.module
    else: raw_model = model
    coeffs = raw_model.kan_layers[0].spline_coeffs.detach()
    strength = coeffs.abs().mean(dim=(2, 3)) 
    num_total = strength.numel()
    num_active = (strength > threshold).sum().item()
    return (num_active / num_total) * 100.0

def batch_correlation_debug(pred, target):
    """Debug utility to see correlation per batch item"""
    # pred, target: (Batch, Time, Nodes)
    # Flatten to (Batch, Time*Nodes) for global correlation per subject in batch
    p_flat = pred.reshape(pred.shape[0], -1)
    t_flat = target.reshape(target.shape[0], -1)
    
    vx = p_flat - p_flat.mean(dim=1, keepdim=True)
    vy = t_flat - t_flat.mean(dim=1, keepdim=True)
    
    corr = (vx * vy).sum(dim=1) / (torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy**2).sum(dim=1)) + 1e-8)
    return corr.mean(), "OK"
# src/train.py -> inside calculate_hebbian_losses or similar

def calculate_sparsity_loss(model):
    l_sparse = 0.0
    
    for name, param in model.named_parameters():
        # --- FIX: ADD "spline_coeffs" TO THE CHECK ---
        if ("spline_weights" in name) or ("base_linear.weight" in name) or ("spline_coeffs" in name):
            l_sparse += torch.abs(param).sum()
            
    return l_sparse

# --- NEW HEBBIAN LOSS LOGIC ---
def calculate_hebbian_losses(model, layer_weights, layer_inputs, distance_matrix):
    """
    STRICT L1 MODE.
    Reverts to the robust 'Lasso' style regularization that successfully 
    reduced density in previous runs.
    """
    # 1. Sparsity Loss (Pure L1)
    # Constant pressure on all weights to shrink.
    # No 'epsilon' protection. No 'log' instability.
    l_sparse = calculate_sparsity_loss(model)
    
    # 2. Wiring Loss (Distance-Weighted L1)
    # We REMOVE the 'tax_relief' (Hebbian Shield).
    # Long connections must pay the full price. 
    # Only the most critical long-range links will survive this.
    l_wire = (layer_weights.abs() * distance_matrix).mean()
    
    return l_wire, l_sparse

def neural_development(epoch, total_epochs, model):
    """
    Developmental Schedule for Regularization.
    """
    progress = epoch / total_epochs
    
    # PHASE 1: Synaptogenesis (0 - 30%)
    # Let connections grow. Minimal penalty.
    if progress < 0.30:
        sparsity_mult = 0.1 
        wiring_mult = 0.1
        
    # PHASE 2: Refinement (30% - 70%)
    # The "Critical Period". Linear ramp.
    elif progress < 0.70:
        phase_progress = (progress - 0.30) / 0.40 
        sparsity_mult = 0.1 + (0.9 * phase_progress)
        wiring_mult = 0.1 + (0.9 * phase_progress)
        
    # PHASE 3: Pruning (70% - 100%)
    # Full constraint.
    else:
        sparsity_mult = 1.0
        wiring_mult = 1.0
        
    return sparsity_mult, wiring_mult

def get_dynamic_bioconst_lambda(epoch, target_val=5e-6):
    """
    Bio-Plausible Developmental Schedule (The "Critical Period" Ramp).
    Mimics Synaptic Blooming (0-30%) -> Pruning/Refinement (30-70%) -> Stability (70%).
    """
    # PHASE 1: "BLOOMING" / CRITICAL PERIOD (0% -> 30%)
    # Allow maximum plasticity. No spatial constraints. 
    # Let the model learn raw features from the data.
    if epoch < config.NUM_EPOCHS * 0.30:
        return 0.0
    
    # PHASE 2: "PRUNING" / REFINEMENT (30% -> 70%)
    # Gradually introduce lateral inhibition (Smoothness).
    # This sharpens the boundaries between functional regions.
    elif epoch < config.NUM_EPOCHS * 0.70:
        # Normalize progress from 0.0 to 1.0 within this window
        start_epoch = config.NUM_EPOCHS * 0.30
        end_epoch = config.NUM_EPOCHS * 0.70
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return target_val * progress

    # PHASE 3: "MATURATION" / STABILITY (70% -> 100%)
    # The map is formed. Maintain the topology.
    else:
        return target_val

def transition_weighted_loss(pred, target):
    """
    Calculates purely the velocity-weighted loss.
    """
    # 1. Calculate velocity (temporal derivative)
    # Shape: (Batch, Time-1)
    velocity = torch.abs(target[:, 1:] - target[:, :-1]).mean(dim=-1)
    
    # 2. Create weights: High weight where velocity is high
    # We detach velocity so we don't try to minimize the target's velocity itself
    weights = 1.0 + 5.0 * velocity.detach() 
    
    # 3. Calculate Raw MSE on the transition points
    loss = F.mse_loss(pred[:, 1:], target[:, 1:], reduction='none') # (B, T-1, N)
    
    # 4. Average over nodes first -> (B, T-1)
    loss_per_time = loss.mean(dim=2)
    
    # 5. Apply weights ONLY to this transition loss
    weighted_loss = (loss_per_time * weights).mean()
    
    return weighted_loss

def differentiable_hierarchy_rank_loss(tau, hierarchy_index, margin=0.0, temperature=5.0):
    tau = (tau - tau.mean()) / (tau.std() + 1e-8)
    h = (hierarchy_index - hierarchy_index.mean()) / (hierarchy_index.std() + 1e-8)

    dtau = tau.unsqueeze(0) - tau.unsqueeze(1)
    dh = h.unsqueeze(0) - h.unsqueeze(1)

    sign_h = torch.sign(dh)
    valid = torch.abs(dh) > 1e-3

    # penalize pairs where tau ordering contradicts hierarchy ordering
    loss = F.softplus(-temperature * sign_h * dtau + margin)

    return loss[valid].mean()

def tau_loss(
    tau_values,
    dist_type="uniform",
    adj=None,                 # (N, N) adjacency for spatial smoothness
    hierarchy_index=None,     # (N,) optional cortical hierarchy proxy
):
    """
    Tau distribution + spatial + biological regularizer.

    Components:
    1. Distribution matching (uniform / lognormal)
    2. Boundary penalties (fast vs slow tails)
    3. Variance maximization (prevents collapse)
    4. Spatial smoothness (neighboring regions similar)
    5. Hierarchy alignment (tau follows cortical hierarchy)

    Returns scalar loss.
    """

    # ----------------------------
    # Setup
    # ----------------------------
    tau = tau_values.flatten()
    tau_sorted, _ = torch.sort(tau)

    num_nodes = tau_sorted.size(0)
    device = tau_values.device
    dtype = tau_values.dtype

    tau_min_phys = 1.5
    tau_max_phys = 12.0

    # ----------------------------
    # 1. Boundary penalties
    # ----------------------------
    k = max(1, int(num_nodes * 0.1))
    fastest_10pct = tau_sorted[:k]
    slowest_10pct = tau_sorted[-k:]

    loss_boundary_fast = torch.mean(torch.relu(fastest_10pct - 3.0))
    loss_boundary_slow = torch.mean(torch.relu(9.0 - slowest_10pct))

    # ----------------------------
    # 2. Distribution matching
    # ----------------------------
    tau_clamped = tau_sorted.clamp(min=tau_min_phys, max=tau_max_phys)
    tau_sorted_norm = (tau_clamped - tau_min_phys) / (tau_max_phys - tau_min_phys)

    if dist_type == "uniform":
        target_dist = torch.linspace(0.0, 1.0, num_nodes, device=device, dtype=dtype)
        loss_dist = F.mse_loss(tau_sorted_norm, target_dist)

    elif dist_type == "lognormal":
        mu = math.log(config.TAU_LOGNORMAL_MEDIAN)
        sigma = config.TAU_LOGNORMAL_SIGMA

        eps = 1e-3
        q = torch.linspace(eps, 1.0 - eps, num_nodes, device=device, dtype=dtype)

        z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
        target_tau = torch.exp(mu + sigma * z)

        target_tau = target_tau.clamp(min=tau_min_phys, max=tau_max_phys)
        target_dist = (target_tau - tau_min_phys) / (tau_max_phys - tau_min_phys)

        loss_dist = F.mse_loss(tau_sorted_norm, target_dist)

    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    # ----------------------------
    # 3. Variance maximization (Level 1)
    # ----------------------------
    # Prevents collapse
    target_std = target_tau.std() if dist_type == "lognormal" else (
    (tau_max_phys - tau_min_phys) / math.sqrt(12.0)
    )

    loss_var = (tau.std() - target_std).pow(2)
    tau_norm_unsorted = (tau - tau_min_phys) / (tau_max_phys - tau_min_phys)

    edge_margin = 0.03
    loss_saturation = (
    torch.relu(edge_margin - tau_norm_unsorted).pow(2).mean()
    + torch.relu(tau_norm_unsorted - (1.0 - edge_margin)).pow(2).mean()
    )
    
    # ----------------------------
    # 4. Spatial smoothness (Level 2)
    # ----------------------------
    loss_smooth = 0.0
    if adj is not None:
        # Ensure proper shape
        if adj.ndim == 2:
            tau_i = tau.unsqueeze(0)  # (1, N)
            tau_j = tau.unsqueeze(1)  # (N, 1)
            loss_smooth = torch.mean(adj * (tau_i - tau_j) ** 2)
        else:
            raise ValueError(f"Unexpected adj shape: {adj.shape}")

    # ----------------------------
    # 5. Hierarchy constraint (Level 3)
    # ----------------------------
    loss_hierarchy = 0.0
    if hierarchy_index is not None:
        h = hierarchy_index.to(device=device, dtype=dtype).flatten()

        # Normalize both
        tau_norm = (tau - tau.mean()) / (tau.std() + 1e-8)
        h_norm = (h - h.mean()) / (h.std() + 1e-8)

        # Want positive correlation
        corr = torch.mean(tau_norm * h_norm)
        loss_hierarchy = -corr  # maximize correlation
    #####################
    # Rank calculations
    #####################
    # ----------------------------
    # Combine (Level 4)
    # ----------------------------
    # You should tune these weights!
    # Encourage rank-order consistency between tau and hierarchy
    loss_rank=differentiable_hierarchy_rank_loss(tau, hierarchy_index, margin=0.0, temperature=5.0)
    
    loss = (
        loss_dist
        + loss_boundary_fast
        + loss_boundary_slow
        + config.LAMBDA_TAU_VAR * loss_var
        + config.LAMBDA_TAU_SMOOTH * loss_smooth
        + config.LAMBDA_TAU_HIER * loss_hierarchy
        + config.LAMBDA_TAU_SATURATION * loss_saturation
    )

    return loss, loss_rank

def calculate_smoothness_loss(model, distance_matrix, column_ids, batch_fmri_data):
    """
    Implements "State-Velocity" Clustering.
    A neighbor is defined by:
    1. Anatomy (Distance)
    2. State Similarity (Correlation of Signal)
    3. Velocity Similarity (Correlation of Derivative) - [NEW]
    4. Topology (7T Shield)
    """
    # 1. Get Model Parameter (Tau)
    if hasattr(model, "module"): raw_model = model.module
    else: raw_model = model
    
    if hasattr(raw_model.cfc, 'get_tau_values'):
        vals = raw_model.cfc.get_tau_values().flatten()
    else:
        vals = torch.sigmoid(raw_model.cfc.tau_system.weight).flatten()
    
    device = vals.device

    # -----------------------------------------------------------
    # A. ANATOMICAL WEIGHTS (Static)
    # -----------------------------------------------------------
    sigma = config.SMOOTHNESS_SIGMA_MM 
    dist_sq = distance_matrix.to(device, non_blocking=True) ** 2
    w_anatomy = torch.exp(- dist_sq / (2 * sigma**2))
    w_anatomy[w_anatomy < 0.01] = 0 

    # -----------------------------------------------------------
    # B. FUNCTIONAL WEIGHTS (State + Velocity) [NEW]
    # -----------------------------------------------------------
    # Normalize inputs (Batch, Nodes, Time)
    x = batch_fmri_data.permute(0, 2, 1) # (B, N, T)
    
    # 1. State Correlation (Standard)
    x_norm = x / (x.norm(dim=2, keepdim=True) + 1e-8)
    sim_state = torch.bmm(x_norm, x_norm.transpose(1, 2)).mean(dim=0)
    
    # 2. Velocity Correlation (Derivative)
    # Calculate temporal derivative: dX = X[t+1] - X[t]
    x_vel = x[:, :, 1:] - x[:, :, :-1]
    # Pad with 0 to match size
    x_vel = torch.cat([torch.zeros((x.shape[0], x.shape[1], 1), device=device), x_vel], dim=2)
    
    x_vel_norm = x_vel / (x_vel.norm(dim=2, keepdim=True) + 1e-8)
    sim_vel = torch.bmm(x_vel_norm, x_vel_norm.transpose(1, 2)).mean(dim=0)
    
    # 3. Composite Similarity
    # We require regions to match in BOTH position and direction.
    # We average them, then Clamp negative values.
    w_functional = (sim_state + sim_vel) / 2
    w_functional = torch.clamp(w_functional, min=0) 

    # -----------------------------------------------------------
    # C. TOPOLOGICAL SHIELD (7T Constraint)
    # -----------------------------------------------------------
    c_i = column_ids.unsqueeze(1).to(device, non_blocking=True)
    c_j = column_ids.unsqueeze(0).to(device, non_blocking=True)
    w_shield = (c_i == c_j).float()

    # -----------------------------------------------------------
    # D. COMBINE & SPARSIFY
    # -----------------------------------------------------------
    W = w_anatomy * w_functional * w_shield
    
    # k-NN Sparsification
    k = config.SMOOTHNESS_K_NEIGHBORS
    topk_vals, _ = torch.topk(W, k, dim=1)
    thresholds = topk_vals[:, -1].unsqueeze(1)
    
    mask = (W >= thresholds).float()
    W = W * mask
    W.fill_diagonal_(0)

    # -----------------------------------------------------------
    # E. LOSS
    # -----------------------------------------------------------
    if W.sum() == 0: return torch.tensor(0.0, device=device)
    
    diff = torch.abs(vals.unsqueeze(0) - vals.unsqueeze(1))
    loss = (W * diff).sum() / (W.sum() + 1e-8)
    
    return loss


def calculate_head_sign_loss(model):
    """
    Soft head-wise biological sign constraints for MultiHeadFastKANLayer:
    head 0 -> excitatory (prefer >= 0)
    head 1 -> inhibitory (prefer <= 0)
    head 2 -> modulatory (either sign, but smaller gain)
    """
    if hasattr(model, "module"):
        raw_model = model.module
    else:
        raw_model = model

    total = 0.0
    n_layers = 0

    for layer in raw_model.kan_layers:
        W = layer.spline_coeffs   # (N, N, H, G)

        if W.shape[2] < 3:
            continue

        # wrong-sign mass
        loss_exc = F.relu(-W[:, :, 0, :]).mean()           # head 0 should be >= 0
        loss_inh = F.relu(W[:, :, 1, :]).mean()            # head 1 should be <= 0

        # modulatory head: encourage smaller magnitude, not sign
        loss_mod = torch.abs(W[:, :, 2, :]).mean()

        total = total + loss_exc + loss_inh + config.HEAD_MOD_GAIN_PENALTY * loss_mod
        n_layers += 1

    if n_layers == 0:
        device = next(raw_model.parameters()).device
        return torch.tensor(0.0, device=device)

    return total / n_layers


def hard_prune(model, optimizer):
    if hasattr(model, "module"): raw_model = model.module
    else: raw_model = model
    # Use the fixed threshold from config, or a hardcoded safe value
    threshold = 1e-4  # Lowered from 0.01
    
    for name, param in raw_model.named_parameters():
        if ("spline_weights" in name) or ("spline_coeffs" in name) or ("base_linear.weight" in name):
            
            with torch.no_grad():
                # 1. Create Mask (Keep weights > threshold)
                mask = (torch.abs(param) > threshold).float()
                
                # 2. Hard Kill Weights
                param.mul_(mask)
                
                # 3. CRITICAL: Kill Optimizer State (The Zombie Fix)
                # If we don't do this, Adam "remembers" the velocity and 
                # pushes the weight back up immediately.
                if param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg' in state: 
                        state['exp_avg'].mul_(mask)
                    if 'exp_avg_sq' in state: 
                        state['exp_avg_sq'].mul_(mask)

            # 4. Kill Gradients (The Hook)
            if param.requires_grad:
                # Clear old handles to prevent memory leak
                if hasattr(param, "_pruning_hook_handle"):
                    param._pruning_hook_handle.remove()
                
                def get_pruning_hook(mask_tensor):
                    def hook(grad):
                        return grad * mask_tensor
                    return hook
                    
                handle = param.register_hook(get_pruning_hook(mask))
                param._pruning_hook_handle = handle

def compute_hierarchy_index(adj_matrix, method="degree"):
    """
    Compute hierarchy index from graph structure.

    adj_matrix: (N, N) torch tensor
    returns: (N,) hierarchy index normalized to [0,1]
    """

    A = adj_matrix

    if method == "degree":
        # In-degree (integration)
        h = A.sum(dim=0)  # (N,)

    elif method == "out_degree":
        h = A.sum(dim=1)

    elif method == "symmetric_degree":
        h = A.sum(dim=0) + A.sum(dim=1)

    elif method == "eigenvector":
        # Power iteration
        x = torch.ones(A.size(0), device=A.device)
        for _ in range(10):
            x = A @ x
            x = x / (x.norm() + 1e-8)
        h = x

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to [0,1]
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)

    return h

def calculate_cfc_population_regularization(model, adj=None):
    """
    Regularizes the CfC layer toward a stable neural-population interpretation.

    The goal is not to impose a full neural-mass model, but to bias the CfC
    toward fMRI-plausible local population dynamics:

    - node identity varies smoothly across nearby regions
    - gates do not saturate completely open/closed
    - latent state changes remain energy-limited
    - candidate neural activity remains bounded
    - signed intrinsic drive is balanced and not extreme
    """
    if hasattr(model, "module"):
        raw_model = model.module
    else:
        raw_model = model

    cfc = raw_model.cfc
    device = next(cfc.parameters()).device

    total = torch.tensor(0.0, device=device)

    # -------------------------------
    # 1. Smooth node identity / resting bias
    # -------------------------------
    # node_bias is the learned identity of each region. Nearby or strongly
    # connected nodes should not have completely arbitrary unrelated biases.
    if adj is not None and hasattr(cfc, "node_bias"):
        A = adj.to(device=device, dtype=cfc.node_bias.dtype)

        if A.ndim == 2 and A.shape[0] == cfc.node_bias.shape[0]:
            # Remove self-loops for smoothness term
            A = A.clone()
            A.fill_diagonal_(0.0)

            denom = A.sum() + 1e-8

            bias_i = cfc.node_bias.unsqueeze(1)  # (N,1,H)
            bias_j = cfc.node_bias.unsqueeze(0)  # (1,N,H)

            bias_diff2 = ((bias_i - bias_j) ** 2).mean(dim=-1)  # (N,N)
            loss_bias_smooth = (A * bias_diff2).sum() / denom

            total = total + config.CFC_NODE_BIAS_SMOOTH_W * loss_bias_smooth

    # Keep node bias from becoming an arbitrary hidden lookup table.
    if hasattr(cfc, "node_bias"):
        loss_bias_l2 = cfc.node_bias.pow(2).mean()
        total = total + config.CFC_NODE_BIAS_L2_W * loss_bias_l2

    # -------------------------------
    # 2. Gate regularization
    # -------------------------------
    # g_gate controls whether the state updates or freezes. Fully saturated
    # gates make the model less interpretable as a smooth population process.
    if hasattr(cfc, "last_gate"):
        g = cfc.last_gate

        gate_min = getattr(config, "CFC_GATE_MIN", 0.05)
        gate_max = getattr(config, "CFC_GATE_MAX", 0.95)
        gate_target = getattr(config, "CFC_GATE_TARGET", 0.50)

        loss_gate_bounds = (
            torch.relu(gate_min - g).mean()
            + torch.relu(g - gate_max).mean()
        )

        loss_gate_mean = (g.mean() - gate_target).pow(2)

        total = total + config.CFC_GATE_REG_W * (loss_gate_bounds + loss_gate_mean)

    # -------------------------------
    # 3. Latent energy / smooth temporal update
    # -------------------------------
    # fMRI-visible neural population dynamics should not need violent
    # step-to-step hidden-state jumps.
    if hasattr(cfc, "last_h_delta"):
        loss_delta_energy = cfc.last_h_delta.pow(2).mean()
        total = total + config.CFC_DELTA_ENERGY_W * loss_delta_energy

    # -------------------------------
    # 4. Candidate neural activity bound
    # -------------------------------
    # f_val is the candidate neural population state. Keep it moderate.
    if hasattr(cfc, "last_f_val"):
        f_val = cfc.last_f_val
        f_max = getattr(config, "CFC_F_ACTIVITY_MAX", 3.0)

        loss_f_bound = torch.relu(torch.abs(f_val) - f_max).pow(2).mean()
        total = total + config.CFC_F_ACTIVITY_W * loss_f_bound

    # -------------------------------
    # 5. Signed intrinsic-drive regularization
    # -------------------------------
    # The drive map should be signed but not globally biased all-positive or all-negative.
    if hasattr(cfc, "get_causal_drive"):
        drive = cfc.get_causal_drive()

        loss_drive_balance = drive.mean().pow(2)

        drive_max = getattr(config, "CFC_DRIVE_MAX_ABS", 3.0)
        loss_drive_bound = torch.relu(torch.abs(drive) - drive_max).pow(2).mean()

        total = total + config.CFC_DRIVE_BALANCE_W * loss_drive_balance
        total = total + config.CFC_DRIVE_BOUND_W * loss_drive_bound

    return total

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer, rank, f, train_loader, is_master, loop, scaler, use_amp):
    model.train()
    total_loss = 0
    ds_ref = loader.dataset.datasets[0] if isinstance(loader.dataset, ConcatDataset) else loader.dataset
    if hasattr(loader.dataset, 'dataset'): ds_ref = loader.dataset.dataset 
    static_adj = ds_ref.adjacency.to(device, non_blocking=True)
    dist_mat = ds_ref.distance_matrix.to(device, non_blocking=True) if hasattr(ds_ref, 'distance_matrix') else None

    #ds_ref.
    # 1. Initialize Running Metrics (Accumulators)
    running_loss = 0.0
    running_mse = 0.0
    running_corr = 0.0
    running_var = 0.0
    running_der = 0.0
    running_meta = 0.0
    running_sparsity = 0.0
    running_smoothness = 0.0
    running_wiring = 0.0
    running_longterm = 0.0
    running_tau = 0.0
    
    running_mean_activity=0.0
    running_max_activity=0.0
    running_density1 = 0.0
    running_density2 = 0.0
    
    # --- DYNAMIC HARD PRUNING ---
    # (Usually around 30% of epochs), every 5th epoch
    can_hard_prune = True

    if getattr(config, "DEMONSTRATION", False) and getattr(config, "DISABLE_HARD_PRUNE_IN_DEMO", True):
        can_hard_prune = False
    min_prune_epoch = max(int(config.NUM_EPOCHS * getattr(config, "PRUNE_START_FRAC", 0.50)),int(getattr(config, "MIN_EPOCHS_BEFORE_PRUNE", 30)))   
    if (can_hard_prune and epoch >= min_prune_epoch and epoch % getattr(config, "PRUNE_EVERY_EPOCHS", 5) == 0):
        hard_prune(model, optimizer)
   
    
    
    criterion_corr = PearsonCorrelationLoss()
    criterion_var = VarianceLoss()
    
    for i, (x_fmri, x_stim, y_target) in enumerate(loader):
        x_fmri = x_fmri.to(device, non_blocking=True)
        x_stim = x_stim.to(device, non_blocking=True)
        y_target = y_target.to(device, non_blocking=True)
        
        # --- DATA AUGMENTATION (Training Only) ---
        x_input = x_fmri
        if config.MASKOUT_TIMESERIES:
             # Randomly choose strategy per batch
             strategy = 'global' if np.random.rand() > 0.5 else 'independent'
             x_input = apply_time_masking(x_fmri, mode=strategy)
        
        if epoch >= int(config.STIM_DROPOUT_START_FRAC * config.NUM_EPOCHS):
            if torch.rand(1, device=device).item() < config.STIM_DROPOUT_PROB:
                x_stim = torch.zeros_like(x_stim)

        optimizer.zero_grad(set_to_none=True)
        #pred, avg_attn, h_final, stacked_heads = model(x_input, x_stim, adj, return_head_weights=False)
        
        # --- LOW-RISK MIXED PRECISION: forward only ---
        # AMP policy:
        # - forward pass in autocast (low-risk memory/computation savings)
        # - losses/metrics/regularizers in float32 for numerical stability
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred, avg_attn, h_final, stacked_heads = model(
                x_input, x_stim, static_adj, return_head_weights=False
            )
        
        l_cfc_pop = calculate_cfc_population_regularization(model, adj=static_adj)
        
        # --- SENSITIVE LOSSES IN FLOAT32 ---
        pred = pred.float()
        y_target = y_target.float()
        h_final = h_final.float()
        
        # --- LOSS CALCULATION ---
        
        
        loss_corr = criterion_corr(pred, y_target)
        
        
        #loss_var = criterion_var(pred)
        loss_var = F.mse_loss(pred.std(dim=1)+ 1e-6, y_target.std(dim=1)+ 1e-6)
        loss_der = derivative_loss(pred, y_target)
        
        # --- TAU PHYSICS LOSSES ---
        cfc_layer = model.module.cfc if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cfc
        tau_secs = cfc_layer.get_tau_secs()

        # Long Term Memory Penalty (Soft boundaries)
        excess_moderate = torch.relu(tau_secs - config.LONGTERM_THRESHOLD_MODERATE)
        excess_extreme = torch.relu(tau_secs - config.LONGTERM_THRESHOLD_EXTREME)
        loss_longterm = torch.mean(excess_moderate + (excess_extreme ** 2))
        
        mse_anchor = criterion(pred, y_target)

        if epoch >= int(config.NUM_EPOCHS * 0.75):
            # Calculate the Dynamic Focus
            trans_loss = transition_weighted_loss(pred, y_target)
        
            # BLEND: 70% Stability (Anchor) + 30% Dynamics (Events)
            mse_loss = 0.7 * mse_anchor + 0.3 * trans_loss
        else:
            # Before 75%, just learn the basics
            mse_loss = mse_anchor
        
        # Tau Distribution Strategy
        hierarchy_index = compute_hierarchy_index(static_adj, method="degree")
        l_tau = 0
        if epoch >= int(config.NUM_EPOCHS * 0.90):
            l_tau, rank_loss = tau_loss(tau_secs, dist_type="lognormal", adj=static_adj, hierarchy_index=hierarchy_index)
        else:
            l_tau, rank_loss = tau_loss(tau_secs, dist_type="uniform", adj=static_adj, hierarchy_index=hierarchy_index)
        
        # --- CONNECTIVITY LOSSES ---
        #l_meta = torch.norm(h_final, p=2)
        l_meta = torch.mean(h_final ** 2)
        l_group = calculate_group_lasso(model, ds_ref.column_ids.to(device, non_blocking=True))

        # 2. [FIX] Explicitly Calculate Smoothness
        #l_smooth = torch.tensor(0.0, device=device)
        #if config.LAMBDA_SMOOTHNESS > 0:
            # We must call the function using the distance matrix we just saved
        
        l_smooth = torch.tensor(0.0, device=device)
        if config.LAMBDA_SMOOTHNESS > 0 and dist_mat is not None:
            l_smooth = calculate_smoothness_loss(
                model, 
                ds_ref.distance_matrix.to(device, non_blocking=True), 
                ds_ref.column_ids.to(device, non_blocking=True),
                x_fmri.float()# <--- CHANGED THIS
            )
            
        
        # Get raw weights for Hebbian calc
        if hasattr(model, "module"): raw_model = model.module
        else: raw_model = model
        raw_coeffs = raw_model.kan_layers[0].spline_coeffs
        
        # Collapse to (N, N) edge strength
        connectivity_strength = raw_coeffs.abs().mean(dim=(2, 3))
        
        # Calculate Hebbian & Sparsity Losses
        l_wire, l_sparse = calculate_hebbian_losses(
            model,
            connectivity_strength, 
            raw_model.last_kan_input,  
            dist_mat
        )
        
        l_orth=calculate_temporal_orthogonality_loss(pred)
        
        if config.LAMBDA_HEAD_SIGN > 0 and epoch >= int(config.HEAD_SIGN_WARMUP_FRAC * config.NUM_EPOCHS):
            l_head_sign = calculate_head_sign_loss(model)
        else:
            l_head_sign = torch.tensor(0.0, device=device)
            
        # Developmental Multipliers
        s_mult, w_mult = neural_development(epoch, config.NUM_EPOCHS, model)
        
        current_lambda_wiring =  config.LAMBDA_WIRING * w_mult
        current_lambda_sparsity = config.LAMBDA_SPARSITY * s_mult
        current_lambda_meta = get_dynamic_bioconst_lambda(epoch, config.LAMBDA_METABOLIC)
        current_lambda_group = get_dynamic_bioconst_lambda(epoch, config.LAMBDA_GROUP_LASSO)
        current_lambda_orth = get_dynamic_bioconst_lambda(epoch, config.LAMBDA_ORTH_LOSS)
        current_lambda_smoothness = get_dynamic_bioconst_lambda(epoch, config.LAMBDA_SMOOTHNESS)
        
        reg_start_epoch = int(config.REG_PHASE_START_FRAC * config.NUM_EPOCHS)
        if epoch < reg_start_epoch:
            lambda_corr = config.LAMBDA_CORRELATION
        else:
            corr_phase_len = max(1, int((1.0 - config.REG_PHASE_START_FRAC) * config.NUM_EPOCHS))
            frac = min(1.0, (epoch - reg_start_epoch) / corr_phase_len)
            lambda_corr = config.LAMBDA_CORRELATION + frac * (
                config.LAMBDA_CORRELATION_FINAL - config.LAMBDA_CORRELATION
            )
       
        # TOTAL LOSS
        loss = (config.LAMBDA_ACCURACY * mse_loss) + \
               (lambda_corr * loss_corr) + \
               (config.LAMBDA_VAR   * loss_var) + \
               (config.LAMBDA_DERIVATIVE   * loss_der) + \
               (current_lambda_meta   * l_meta) + \
               (current_lambda_sparsity   * l_sparse) + \
               (current_lambda_group  * l_group) + \
               (current_lambda_orth  * l_orth) + \
               (current_lambda_smoothness   * l_smooth) + \
               (current_lambda_wiring  * l_wire) + \
               (config.LAMBDA_LONGTERM  * loss_longterm) + \
               (config.LAMBDA_TAU_DIVERSITY * l_tau)    + \
               (config.LAMBDA_HEAD_SIGN * l_head_sign)  + \
               (config.LAMBDA_RANK_LOSS * rank_loss) + \
               (config.LAMBDA_CFC_POP * l_cfc_pop)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
       
        pred_std = pred.std(dim=2).mean().item()
        corr, debug_msg = batch_correlation_debug(pred, y_target)
        
        
        if i == 0 and epoch == 0:
            print(f"DEBUG: Adjacency Max: {static_adj.max()}, Min: {static_adj.min()}, Sum: {static_adj.sum()}")
        if i == 0 and epoch == config.NUM_EPOCHS-1:
            print(f"DEBUG: Adjacency Max: {static_adj.max()}, Min: {static_adj.min()}, Sum: {static_adj.sum()}")
        if i == 0:
            running_loss = loss.item()
            running_mse = config.LAMBDA_ACCURACY * mse_loss.item()
            running_corr = corr.item()
            running_var =config.LAMBDA_VAR * loss_var
            running_der = config.LAMBDA_DERIVATIVE   * loss_der
            running_meta = current_lambda_meta*l_meta.item()
            running_sparsity = current_lambda_sparsity   * l_sparse
            running_smoothness = current_lambda_smoothness   * l_smooth
            running_wiring = current_lambda_wiring  * l_wire
            running_longterm = config.LAMBDA_LONGTERM  * loss_longterm
            running_tau = config.LAMBDA_TAU_DIVERSITY * l_tau
            
            running_mean_activity=pred.mean().item()
            running_max_activity=pred.max().item()
            running_density1 = density_1pct = get_effective_density(model, threshold=0.01)
            running_density2 = get_effective_density(model, threshold=1e-5)
        else:
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            running_mse = 0.9 * running_mse + 0.1 * config.LAMBDA_ACCURACY * mse_loss.item()
            running_corr = 0.9 * running_corr + 0.1 * corr.item()
            running_var = 0.9 * running_var + 0.1 * config.LAMBDA_VAR * loss_var
            running_der = 0.9 * running_der + 0.1 * config.LAMBDA_DERIVATIVE * loss_der
            running_meta = 0.9 * running_meta + 0.1 * current_lambda_meta*l_meta.item()
            running_sparsity = 0.9 * running_sparsity + 0.1 * current_lambda_sparsity   * l_sparse
            running_smoothness = 0.9 * running_smoothness + 0.1 * current_lambda_smoothness   * l_smooth
            running_wiring = 0.9 * running_wiring + 0.1 *  current_lambda_wiring  * l_wire
            running_longterm = 0.9 * running_longterm + 0.1 * config.LAMBDA_LONGTERM  * loss_longterm
            running_tau = 0.9 * running_tau + 0.1 * config.LAMBDA_TAU_DIVERSITY * l_tau
            running_mean_activity = 0.9 * running_mean_activity + 0.1 * pred.mean().item()
            running_max_activity = 0.9 * running_max_activity + 0.1 * pred.max().item()
            running_density1 = 0.9 * running_density1 + 0.1 * get_effective_density(model, threshold=0.01)
            running_density2 = 0.9 * running_density2 + 0.1 * get_effective_density(model, threshold=0.001)

        
        
        # Logging
        if i == len(train_loader) - 1: # Last batch
            density_1pct = get_effective_density(model, threshold=0.01)
            density_low = get_effective_density(model, threshold=0.001)
            if is_master:
                #print(f"Density (>0.01): {density_1pct:.2f}% | Density (>0.001): {density_low:.2f}%")
                print(f"Density (>0.01): {density_1pct:.2f}% | Density (>0.001): {density_low:.2f}%", file=f)
            
        if rank == 0 and i % 10 == 0 and writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar("Loss/MSE", mse_loss.item(), step)
            writer.add_scalar("Bio/GroupLasso", l_group.item(), step)
            
            print(f"{BOLD_WHITE} Mean Output Activity: {pred.mean().item():.4f} |Max Output Activity: {pred.max().item():.4f}| Corr: {corr.item():.4f} | PredStd: {pred_std:.6f}| MSE: {config.LAMBDA_ACCURACY*mse_loss.item()}| Corr_loss: {lambda_corr * loss_corr}| Var_loss: {config.LAMBDA_VAR   * loss_var}| Derivate loss: {config.LAMBDA_DERIVATIVE   * loss_der} |Metabolic: {current_lambda_meta*l_meta.item()} | Wiring loss: {current_lambda_wiring*l_wire.item()} |Sparseness:{current_lambda_sparsity*l_sparse.item()} | Smoothness: {current_lambda_smoothness*l_smooth.item()}|l_group: {current_lambda_group *l_group.item()} | Long term memory (tau) loss: {config.LAMBDA_LONGTERM  * loss_longterm} |tau diversity loss: {config.LAMBDA_TAU_DIVERSITY * l_tau} | temporal orthogonality loss: {current_lambda_orth  * l_orth} | {config.LAMBDA_CFC_POP * l_cfc_pop} ", file=f)
            
            postfix_str = (
                f"L:{running_loss:.3f} | "
                f"MSE:{running_mse:.3f} | "
                f"Corr:{running_corr:.3f} | "
                f"Var:{running_var:.3f} | "
                f"Derivate:{running_der:.3f} | "
                f"Meta:{running_meta:.3f} "
                f"Sparsity:{running_sparsity:.3f} "
                f"Smoothness:{running_smoothness:.3f} "
                f"Wiring:{running_wiring:.3f} "
                f"Longterm:{running_longterm:.3f} "
                f"Tau:{running_tau:.3f} " 
                f"Mean act:{running_mean_activity:.3f} " 
                f"Max act:{running_max_activity:.3f} " 
                f"Density_1:{running_density1:.2f}% " 
                f"Density_2:{running_density2:.2f}% " 
            )
            
            
            loop.set_postfix_str(postfix_str)
            #print(f"Mean Output Activity: {pred.mean().item():.4f} |Max Output Activity: {pred.max().item():.4f}| Corr: {corr.item():.4f} | PredStd: {pred_std:.6f}| MSE: {config.LAMBDA_ACCURACY*mse_loss.item()}| Corr_loss: {config.LAMBDA_CORRELATION   * loss_corr}| Var_loss: {config.LAMBDA_VAR   * loss_var} | Derivate loss: {config.LAMBDA_DERIVATIVE   * loss_der} | Metabolic: {current_lambda_meta*l_meta.item()} | Wiring loss: {current_lambda_wiring*l_wire.item()} |Sparseness:{current_lambda_sparsity*l_sparse.item()} | Smoothness: {current_lambda_smoothness*l_smooth.item()}|l_group: {current_lambda_group *l_group.item()} |Long term memory (tau) loss: {config.LAMBDA_LONGTERM  * loss_longterm}| tau diversity loss: {config.LAMBDA_TAU_DIVERSITY * l_tau}")

    return total_loss / len(loader)


def validate(model, test_loader, criterion, device, ds_ref, use_amp):
    model.eval()
    total_loss = 0
    total_corr = 0
    total_corr_isolated = 0
    
    # Accumulator for per-node contribution
    # We will sum up the difference (Corr_Full - Corr_Iso) for each batch
    # and average at the end.
    node_contrib_accumulator = None
    
    with torch.no_grad():
        adj_full = ds_ref.adjacency.to(device, non_blocking=True)              # same regime as training
        for batch_idx, batch_data in enumerate(test_loader):
            fmri  = batch_data[0]
            stim =  batch_data[1]
            target = batch_data[2]
            
            fmri, stim, target = fmri.to(device, non_blocking=True), stim.to(device, non_blocking=True), target.to(device, non_blocking=True)
            batch_size, num_nodes = fmri.shape[0], fmri.shape[2]
            
            # Initialize accumulator on first batch
            if node_contrib_accumulator is None:
                node_contrib_accumulator = torch.zeros(num_nodes).to(device)

            
            #--- 1. Ajency full and isolated models ---
            
            adj_iso  = torch.eye(num_nodes, device=device)      # true self-only isolation
            
            #--- 2. Prediction full and isolated models ---
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_full, _, _, _ = model(fmri, stim, adj_full)
                pred_iso,  _, _, _ = model(fmri, stim, adj_iso)
            
            pred_full = pred_full.float()
            pred_iso = pred_iso.float()
            target = target.float()
            
            loss = criterion(pred_full, target)
            
            # --- 3. METRICS ---
            total_loss += loss.item()

            # Helper to calculate Nodewise Correlation (over Batch dimension)
            # Input shape: [Batch, Nodes] (or Batch*Time, Nodes)
            def calc_node_corr(preds, targs):
                # Flatten batch and time, keep Nodes distinct
                # Shape becomes (N_samples, N_nodes)
                p = preds.reshape(-1, num_nodes) 
                t = targs.reshape(-1, num_nodes)
                
                # Center variables (over dim 0 = Batch/Time)
                p_mean = p - p.mean(dim=0, keepdim=True)
                t_mean = t - t.mean(dim=0, keepdim=True)
                
                # Pearson Correlation per Node
                # sum(xy) / (sqrt(sum(x^2)) * sqrt(sum(y^2)))
                numerator = (p_mean * t_mean).sum(dim=0)
                denominator = torch.sqrt((p_mean**2).sum(dim=0)) * torch.sqrt((t_mean**2).sum(dim=0))
                
                # Add epsilon to prevent div by zero for silent nodes
                return numerator / (denominator + 1e-8)

            # Calculate nodewise correlations
            corr_full_nodes = calc_node_corr(pred_full, target) # Shape: [Nodes]
            corr_iso_nodes = calc_node_corr(pred_iso, target)   # Shape: [Nodes]
            
            # Global Average for printing
            total_corr += corr_full_nodes.mean().item()
            total_corr_isolated += corr_iso_nodes.mean().item()
            
            # Accumulate the DIFFERENCE per node
            # (How much did the neighbors help THIS specific node?)
            node_contrib_accumulator += (corr_full_nodes - corr_iso_nodes)

    # Averages
    avg_loss = total_loss / len(test_loader)
    avg_corr_full = total_corr / len(test_loader)
    avg_corr_iso = total_corr_isolated / len(test_loader)
    
    # Global Contribution
    conn_contribution = avg_corr_full - avg_corr_iso
    
    # Nodewise Contribution (Vector of size N)
    avg_node_contribution = node_contrib_accumulator / len(test_loader)
    
    return avg_loss, avg_corr_full, avg_corr_iso, conn_contribution, avg_node_contribution

def save_checkpoint(model, optimizer, epoch, loss, filename, config_dict=None):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    save_dict = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if config_dict:
        save_dict['config'] = config_dict
        
    torch.save(save_dict, filename)
    


def main():
    rank, world_size, local_rank = setup_distributed()
    is_master = (rank == 0)
    set_seed(config.SEED + rank)
    
    device = torch.device(f"cuda:{local_rank}")
    
    # 1. Prepare Data
    all_runs = get_subject_files(config.DATA_DIR, num_runs=2)
    # Shuffle to ensure mixed batches
    random.shuffle(all_runs)
    
        # Split Train/Test (e.g., 80/20)
    
    
    # group runs by subject
    groups = {}
    for r in all_runs:
        sid = r.get('subject_id', r['id'].split('_run')[0])
        groups.setdefault(sid, []).append(r)
    
    subject_ids = list(groups.keys())
    random.shuffle(subject_ids)
    
    split_idx = int(len(subject_ids) * config.TRAIN_SET_SIZE)
    train_subs = set(subject_ids[:split_idx])
    test_subs  = set(subject_ids[split_idx:])
    
    train_runs = [r for sid in train_subs for r in groups[sid]]
    test_runs  = [r for sid in test_subs  for r in groups[sid]]
    
    print(f"Training on {len(train_runs)} runs, Testing on {len(test_runs)} runs.")
    
    # Initialize Datasets
    # IMPORTANT: We now pass the LIST of runs, and the mask file path.
    # The dataset class handles the loop internally now.
    
    train_ds = NiftiLaminarDataset(
        data_list=train_runs, 
        mask_img=config.MASK_FILE, 
        window_size=config.WINDOW_SIZE,
        run_type='train',
        sensory_regions=config.SENSORY_REGIONS
    )
    
    test_ds = NiftiLaminarDataset(
        data_list=test_runs, 
        mask_img=config.MASK_FILE, 
        window_size=config.WINDOW_SIZE,
        run_type='test',
        sensory_regions=config.SENSORY_REGIONS
    )
    
    
    
    # Create Loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True, num_workers=config.TRAIN_LOADER_WORKERS, pin_memory=True, persistent_workers=True,  prefetch_factor=2)
    
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.TEST_LOADER_WORKERS, pin_memory=True, persistent_workers=True,  prefetch_factor=2)

      
    # 2. Model
    ref_ds = train_ds
    model = MesocortGBB(train_ds.num_nodes, config.WINDOW_SIZE, sensory_mask=train_ds.sensory_mask).to(device, non_blocking=True)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) # find_unused for DDP safety with flexible graphs
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        
        # 1. CRITICAL: Physics parameters (Tau, Biases) must NOT decay.
        if "node_tau_bias" in name or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        
        # 2. CRITICAL: Connection weights (Splines) MUST decay to help sparsity.
        # We removed "spline_weights" from the no_decay list.
        else:
            decay_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 1e-3}, # Gravity enabled for connections
        {'params': no_decay_params, 'weight_decay': 0.0} 
    ], lr=config.LEARNING_RATE)
    
    criterion = nn.MSELoss()
    
    #Use automated mixed precission? Depedends on config.USE_AMP (True/False)
    use_amp = (device.type == "cuda") and getattr(config, "USE_AMP", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # 4. Training Loop
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = f"training_{timestamp}.txt"
    log_file_path=os.path.join(config.LOG_DIR, log_file)
    f = open(log_file_path, "w") if is_master else None
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, timestamp)) if is_master else None
    
    if is_master:
        print(f"Starting training... Logs: {log_file}")
        
    best_corr = -1.0
    start_time = time.time()
    
    #epoch=0
    
    loop = tqdm(range(config.NUM_EPOCHS), desc="Epochs: ", disable=not is_master, leave=False)
    
    for epoch in loop:
        if train_sampler: train_sampler.set_epoch(epoch)
        
        #avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, rank, f, train_loader, is_master, loop)
        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            writer, rank, f, train_loader, is_master, loop, scaler, use_amp
            )
        
        
        # Validation
        if epoch % 5 == 0:
            #test_loss, test_corr, test_corr_iso, conn_contrib, node_vals = validate(model, test_loader, criterion, device)
            test_loss, test_corr, test_corr_iso, conn_contrib, node_vals = validate(
                model, test_loader, criterion, device, ref_ds, use_amp
                )
            
            if is_master:
                #print(f"Epoch {epoch} | Train: {avg_loss:.4f} | Test: {test_loss:.4f} | Corr: {test_corr:.4f}")
                print(f"Epoch {epoch} | Train: {avg_loss:.4f} | Test: {test_loss:.4f} | Corr: {test_corr:.4f}", file=f)
                
                # Checkpoint
                if test_corr > best_corr:
                    best_corr = test_corr
                    save_path = os.path.join(config.RESULTS_DIR, "checkpoint_best.pth")
                    save_checkpoint(model, optimizer, epoch, test_loss, save_path)
                    print("  -> Saved Best Model.", file=f)
                    
        # Save Latest
        if is_master and epoch % 10 == 0:
            save_path = os.path.join(config.RESULTS_DIR, "checkpoint_latest.pth")
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)

    total_time = time.time() - start_time
    test_loss, test_corr, test_corr_iso, conn_contrib, node_vals = validate(model, test_loader, criterion, device, ref_ds, use_amp)
    
    if is_master:
        final_path = os.path.join(config.RESULTS_DIR, "checkpoint_final.pth")
        latest_path = os.path.join(config.RESULTS_DIR, "checkpoint_latest.pth")
    
        save_checkpoint(model, optimizer, config.NUM_EPOCHS - 1, test_loss, final_path)
        save_checkpoint(model, optimizer, config.NUM_EPOCHS - 1, test_loss, latest_path)
    
        print(f"  -> Saved Final Model: {final_path}", file=f)
        print(f"  -> Updated Latest Model: {latest_path}", file=f)
        
        
        print(f"--- Training Finished in {total_time:.2f}s ---")
        now = datetime.now()
        nowstring=now.strftime("%d_%m_%Y_%H_%M")
        
        print(f"{BOLD_GREEN} Test loss: {test_loss}, Test corr: {test_corr}")
        print(f"{BOLD_GREEN} Test loss: {test_loss} ", file=f)
        
        utils.visualize_results(model, ref_ds, world_size, all_runs, node_vals, test_loader, nowstring)
        utils.visualize_prediction_dynamics(model, test_ds, device, config.RESULTS_DIR)
        utils.visualize_training_dynamics(log_file_path, os.path.join(config.RESULTS_DIR, "training_dynamics_"+ nowstring +".png"),config.NUM_EPOCHS, title="GBB training dynamics")
        utils.export_map_stability(model, test_loader, config.RESULTS_DIR, datestring=nowstring)
        
        f.close()
        writer.close()
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
"""Prediction losses and interpretable-model regularizers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gbb.config import config
from gbb.training.regularizers import calculate_sparsity_loss


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class PearsonCorrelationLoss(nn.Module):
    """Bounded ``1 - r`` loss, averaged over nodes."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target differ: {pred.shape} vs {target.shape}")
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=0, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=0)
        denominator = pred_centered.square().sum(dim=0).sqrt() * target_centered.square().sum(dim=0).sqrt()
        correlation = numerator / denominator.clamp_min(1e-8)
        correlation = correlation.clamp(-1.0, 1.0)
        return 1.0 - correlation.mean()


class VarianceLoss(nn.Module):
    def __init__(self, target_variance: float = 1.0) -> None:
        super().__init__()
        self.target_variance = float(target_variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=1, unbiased=False).mean()
        target = variance.new_tensor(self.target_variance)
        return F.mse_loss(variance, target)


def derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    return F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])


def calculate_group_lasso(model, column_ids: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    raw_model = _unwrap_model(model)
    device = next(raw_model.parameters()).device
    column_ids = column_ids.to(device=device)
    unique_groups = torch.unique(column_ids)
    if unique_groups.numel() <= 1:
        return torch.zeros((), device=device)

    if hasattr(raw_model.cfc, "get_intrinsic_drive"):
        spatial_map = raw_model.cfc.get_intrinsic_drive().flatten()
    else:
        spatial_map = raw_model.cfc.get_causal_drive().flatten()
    if spatial_map.numel() != column_ids.numel():
        raise ValueError(
            f"Intrinsic-drive map has {spatial_map.numel()} nodes, "
            f"column_ids has {column_ids.numel()}"
        )

    losses: list[torch.Tensor] = []
    for group_id in unique_groups:
        values = spatial_map[column_ids == group_id]
        if values.numel() == 0:
            continue
        l1 = values.abs().sum()
        l2 = values.square().sum().add(1e-8).sqrt()
        losses.append(alpha * l1 + (1.0 - alpha) * l2)
    return torch.stack(losses).mean() if losses else torch.zeros((), device=device)


def calculate_temporal_orthogonality_loss(
    pred_tensor: torch.Tensor,
    num_nodes: int | None = None,
    allowed_abs_correlation: float = 0.8,
) -> torch.Tensor:
    if pred_tensor.ndim != 3:
        raise ValueError(f"Expected a 3D prediction tensor, got {pred_tensor.shape}")
    num_nodes = num_nodes or pred_tensor.shape[-1]
    if pred_tensor.shape[-1] == num_nodes:
        node_first = pred_tensor.permute(0, 2, 1)
    elif pred_tensor.shape[1] == num_nodes:
        node_first = pred_tensor
    else:
        raise ValueError(f"Cannot identify node axis in {pred_tensor.shape}")

    _, nodes, _ = node_first.shape
    profiles = node_first.permute(1, 0, 2).reshape(nodes, -1)
    profiles = profiles - profiles.mean(dim=1, keepdim=True)
    profiles = profiles / profiles.norm(dim=1, keepdim=True).clamp_min(1e-8)
    correlation = profiles @ profiles.T
    off_diagonal = ~torch.eye(nodes, device=correlation.device, dtype=torch.bool)
    if off_diagonal.sum() == 0:
        return correlation.new_zeros(())
    return F.relu(correlation[off_diagonal].abs() - allowed_abs_correlation).mean()


def calculate_hebbian_losses(
    model,
    layer_weights: torch.Tensor,
    layer_inputs: torch.Tensor | None,
    distance_matrix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distance-weighted wiring cost and global KAN sparsity cost."""
    del layer_inputs
    sparsity = calculate_sparsity_loss(model)
    if distance_matrix is None:
        wiring = layer_weights.new_zeros(())
    else:
        distance_matrix = distance_matrix.to(
            device=layer_weights.device,
            dtype=layer_weights.dtype,
        )
        if distance_matrix.shape != layer_weights.shape:
            raise ValueError(
                f"distance_matrix {distance_matrix.shape} does not match edge weights {layer_weights.shape}"
            )
        wiring = (layer_weights.abs() * distance_matrix).mean()
    return wiring, sparsity


def transition_weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return F.mse_loss(pred, target)
    velocity = (target[:, 1:] - target[:, :-1]).abs().mean(dim=-1).detach()
    weights = 1.0 + 5.0 * velocity
    pointwise = F.mse_loss(pred[:, 1:], target[:, 1:], reduction="none").mean(dim=-1)
    return (pointwise * weights).mean()


def differentiable_hierarchy_rank_loss(
    tau: torch.Tensor,
    hierarchy_index: torch.Tensor | None,
    margin: float = 0.0,
    temperature: float = 5.0,
) -> torch.Tensor:
    if hierarchy_index is None:
        return tau.new_zeros(())
    tau = tau.flatten()
    hierarchy = hierarchy_index.to(device=tau.device, dtype=tau.dtype).flatten()
    if tau.numel() != hierarchy.numel():
        raise ValueError("tau and hierarchy_index must have the same number of nodes")

    tau_z = (tau - tau.mean()) / tau.std(unbiased=False).clamp_min(1e-8)
    hierarchy_z = (hierarchy - hierarchy.mean()) / hierarchy.std(unbiased=False).clamp_min(1e-8)
    delta_tau = tau_z[:, None] - tau_z[None, :]
    delta_hierarchy = hierarchy_z[:, None] - hierarchy_z[None, :]
    valid_pairs = delta_hierarchy.abs() > 1e-3
    valid_pairs.fill_diagonal_(False)
    if valid_pairs.sum() == 0:
        return tau.new_zeros(())

    pair_loss = F.softplus(
        -temperature * torch.sign(delta_hierarchy) * delta_tau + margin
    )
    return pair_loss[valid_pairs].mean()


def tau_loss(
    tau_values: torch.Tensor,
    dist_type: str = "uniform",
    adj: torch.Tensor | None = None,
    hierarchy_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau = tau_values.flatten()
    if tau.numel() == 0:
        return tau.new_zeros(()), tau.new_zeros(())

    tau_min = float(config.TAU_MIN_PHYS)
    tau_max = float(config.TAU_MAX_PHYS)
    if tau_max <= tau_min:
        raise ValueError("TAU_MAX_PHYS must be larger than TAU_MIN_PHYS")

    sorted_tau = tau.sort().values
    normalized = (sorted_tau.clamp(tau_min, tau_max) - tau_min) / (tau_max - tau_min)
    count = tau.numel()
    if dist_type == "uniform":
        target_tau = torch.linspace(tau_min, tau_max, count, device=tau.device, dtype=tau.dtype)
    elif dist_type == "lognormal":
        quantiles = torch.linspace(1e-3, 1.0 - 1e-3, count, device=tau.device, dtype=tau.dtype)
        z_score = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
        target_tau = torch.exp(
            math.log(float(config.TAU_LOGNORMAL_MEDIAN))
            + float(config.TAU_LOGNORMAL_SIGMA) * z_score
        ).clamp(tau_min, tau_max)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    target_normalized = (target_tau - tau_min) / (tau_max - tau_min)
    distribution_loss = F.mse_loss(normalized, target_normalized)
    tail_count = max(1, int(round(count * 0.1)))
    fast_target = tau_min + 0.15 * (tau_max - tau_min)
    slow_target = tau_min + 0.70 * (tau_max - tau_min)
    boundary_loss = F.relu(sorted_tau[:tail_count] - fast_target).mean()
    boundary_loss = boundary_loss + F.relu(slow_target - sorted_tau[-tail_count:]).mean()

    target_std = target_tau.std(unbiased=False)
    variance_loss = (tau.std(unbiased=False) - target_std).square()
    tau_normalized = (tau - tau_min) / (tau_max - tau_min)
    edge_margin = 0.03
    saturation_loss = F.relu(edge_margin - tau_normalized).square().mean()
    saturation_loss = saturation_loss + F.relu(
        tau_normalized - (1.0 - edge_margin)
    ).square().mean()

    smoothness_loss = tau.new_zeros(())
    if adj is not None:
        adjacency = adj.to(device=tau.device, dtype=tau.dtype)
        if adjacency.shape != (count, count):
            raise ValueError(f"Expected adjacency {(count, count)}, got {adjacency.shape}")
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        difference = tau[:, None] - tau[None, :]
        smoothness_loss = (adjacency * difference.square()).sum() / adjacency.sum().clamp_min(1e-8)

    hierarchy_loss = tau.new_zeros(())
    if hierarchy_index is not None:
        hierarchy = hierarchy_index.to(device=tau.device, dtype=tau.dtype).flatten()
        tau_z = (tau - tau.mean()) / tau.std(unbiased=False).clamp_min(1e-8)
        hierarchy_z = (hierarchy - hierarchy.mean()) / hierarchy.std(unbiased=False).clamp_min(1e-8)
        hierarchy_loss = -(tau_z * hierarchy_z).mean()

    total = (
        distribution_loss
        + boundary_loss
        + float(config.LAMBDA_TAU_VAR) * variance_loss
        + float(config.LAMBDA_TAU_SATURATION) * saturation_loss
        + float(config.LAMBDA_TAU_SMOOTH) * smoothness_loss
        + float(config.LAMBDA_TAU_HIER) * hierarchy_loss
    )
    rank_loss = differentiable_hierarchy_rank_loss(tau, hierarchy_index)
    return total, rank_loss


def calculate_smoothness_loss(
    model,
    distance_matrix: torch.Tensor,
    column_ids: torch.Tensor,
    batch_fmri_data: torch.Tensor,
) -> torch.Tensor:
    raw_model = _unwrap_model(model)
    values = raw_model.cfc.get_tau_values().flatten()
    device = values.device
    distance = distance_matrix.to(device=device, dtype=values.dtype)
    columns = column_ids.to(device=device)

    sigma = float(config.SMOOTHNESS_SIGMA_MM)
    anatomy = torch.exp(-distance.square() / (2.0 * sigma**2))
    anatomy = anatomy.masked_fill(anatomy < 0.01, 0.0)

    if batch_fmri_data.ndim != 3 or batch_fmri_data.shape[-1] != values.numel():
        raise ValueError(
            f"Expected fMRI (batch,time,{values.numel()}), got {batch_fmri_data.shape}"
        )
    node_series = batch_fmri_data.to(device=device, dtype=values.dtype).permute(0, 2, 1)
    state_normalized = node_series / node_series.norm(dim=2, keepdim=True).clamp_min(1e-8)
    state_similarity = torch.bmm(state_normalized, state_normalized.transpose(1, 2)).mean(dim=0)

    velocity = torch.diff(node_series, dim=2, prepend=node_series[:, :, :1])
    velocity_normalized = velocity / velocity.norm(dim=2, keepdim=True).clamp_min(1e-8)
    velocity_similarity = torch.bmm(
        velocity_normalized, velocity_normalized.transpose(1, 2)
    ).mean(dim=0)
    functional = ((state_similarity + velocity_similarity) / 2.0).clamp_min(0.0)
    shield = columns[:, None] == columns[None, :]
    weights = anatomy * functional * shield.to(values.dtype)
    weights.fill_diagonal_(0.0)

    k = min(int(config.SMOOTHNESS_K_NEIGHBORS), weights.shape[1])
    if k <= 0 or weights.sum() <= 0:
        return values.new_zeros(())
    thresholds = torch.topk(weights, k=k, dim=1).values[:, -1:]
    weights = weights * (weights >= thresholds)
    weights.fill_diagonal_(0.0)
    if weights.sum() <= 0:
        return values.new_zeros(())

    difference = (values[:, None] - values[None, :]).abs()
    return (weights * difference).sum() / weights.sum().clamp_min(1e-8)


def calculate_head_sign_loss(model) -> torch.Tensor:
    """Soft sign priors for positive-like, negative-like, and low-gain heads."""
    raw_model = _unwrap_model(model)
    losses: list[torch.Tensor] = []
    for layer in raw_model.kan_layers:
        weights = layer.spline_coeffs
        if weights.shape[2] < 3:
            continue
        positive_like = F.relu(-weights[:, :, 0, :]).mean()
        negative_like = F.relu(weights[:, :, 1, :]).mean()
        low_gain = weights[:, :, 2, :].abs().mean()
        losses.append(
            positive_like
            + negative_like
            + float(config.HEAD_MOD_GAIN_PENALTY) * low_gain
        )
    if not losses:
        return next(raw_model.parameters()).new_zeros(())
    return torch.stack(losses).mean()
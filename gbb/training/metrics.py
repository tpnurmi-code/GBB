"""Training and validation metrics."""

from __future__ import annotations

import torch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_effective_density(model, threshold: float = 0.01) -> float:
    raw_model = _unwrap_model(model)
    if not raw_model.kan_layers:
        return 0.0
    coefficients = raw_model.kan_layers[0].spline_coeffs.detach()
    strength = coefficients.abs().mean(dim=(2, 3))
    return 100.0 * float((strength > threshold).float().mean().item())


def batch_correlation_debug(pred: torch.Tensor, target: torch.Tensor):
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
    correlation = numerator / denominator.clamp_min(1e-8)
    return correlation.mean(), "OK"


def compute_hierarchy_index(adj_matrix: torch.Tensor, method: str = "degree") -> torch.Tensor:
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj_matrix must be square")
    adjacency = adj_matrix
    if method == "degree":
        hierarchy = adjacency.sum(dim=0)
    elif method == "out_degree":
        hierarchy = adjacency.sum(dim=1)
    elif method == "symmetric_degree":
        hierarchy = adjacency.sum(dim=0) + adjacency.sum(dim=1)
    elif method == "eigenvector":
        hierarchy = torch.ones(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
        for _ in range(20):
            hierarchy = adjacency @ hierarchy
            hierarchy = hierarchy / hierarchy.norm().clamp_min(1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    return (hierarchy - hierarchy.min()) / (hierarchy.max() - hierarchy.min()).clamp_min(1e-8)
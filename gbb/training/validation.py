"""Validation routines."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def _nodewise_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    num_nodes = prediction.shape[-1]
    pred = prediction.reshape(-1, num_nodes)
    truth = target.reshape(-1, num_nodes)
    pred = pred - pred.mean(dim=0, keepdim=True)
    truth = truth - truth.mean(dim=0, keepdim=True)
    numerator = (pred * truth).sum(dim=0)
    denominator = pred.square().sum(dim=0).sqrt() * truth.square().sum(dim=0).sqrt()
    return numerator / denominator.clamp_min(1e-8)


def validate(
    model,
    test_loader,
    criterion,
    device: torch.device,
    ds_ref=None,
    use_amp: bool = False,
):
    if len(test_loader) == 0:
        raise ValueError("test_loader is empty")
    model.eval()
    ds_ref = ds_ref or test_loader.dataset
    if hasattr(ds_ref, "dataset"):
        ds_ref = ds_ref.dataset
    adjacency_full = ds_ref.adjacency.to(device, non_blocking=True)

    total_loss = 0.0
    total_full = 0.0
    total_isolated = 0.0
    node_contribution = None

    with torch.no_grad():
        for fmri, stimulus, target in test_loader:
            fmri = fmri.to(device, non_blocking=True)
            stimulus = stimulus.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            isolated = torch.eye(fmri.shape[-1], device=device, dtype=adjacency_full.dtype)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=True)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                pred_full, _, _, _ = model(fmri, stimulus, adjacency_full)
                pred_isolated, _, _, _ = model(fmri, stimulus, isolated)

            pred_full = pred_full.float()
            pred_isolated = pred_isolated.float()
            target = target.float()
            total_loss += float(criterion(pred_full, target).item())
            corr_full = _nodewise_correlation(pred_full, target)
            corr_isolated = _nodewise_correlation(pred_isolated, target)
            total_full += float(corr_full.mean().item())
            total_isolated += float(corr_isolated.mean().item())
            difference = corr_full - corr_isolated
            node_contribution = difference if node_contribution is None else node_contribution + difference

    batches = len(test_loader)
    average_loss = total_loss / batches
    average_full = total_full / batches
    average_isolated = total_isolated / batches
    average_node_contribution = node_contribution / batches
    return (
        average_loss,
        average_full,
        average_isolated,
        average_full - average_isolated,
        average_node_contribution,
    )
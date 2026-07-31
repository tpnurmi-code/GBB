"""Training-time data augmentation."""

from __future__ import annotations

import torch


def apply_time_masking(x_batch: torch.Tensor, mask_len: int = 5, mode: str = "independent") -> torch.Tensor:
    """Mask contiguous time ranges globally or independently per node."""
    if x_batch.ndim != 3:
        raise ValueError(f"Expected (batch, time, nodes), got {tuple(x_batch.shape)}")
    if mask_len <= 0:
        return x_batch.clone()

    masked = x_batch.clone()
    batch_size, time_points, num_nodes = masked.shape
    max_start = time_points - mask_len
    if max_start < 0:
        return masked

    if mode == "global":
        starts = torch.randint(0, max_start + 1, (batch_size,), device=x_batch.device)
        for batch_index, start in enumerate(starts.tolist()):
            masked[batch_index, start : start + mask_len, :] = 0.0
    elif mode == "independent":
        starts = torch.randint(
            0,
            max_start + 1,
            (batch_size, num_nodes),
            device=x_batch.device,
        )
        time_grid = torch.arange(time_points, device=x_batch.device).view(1, -1, 1)
        mask = (time_grid >= starts.unsqueeze(1)) & (
            time_grid < starts.unsqueeze(1) + mask_len
        )
        masked[mask] = 0.0
    else:
        raise ValueError("mode must be 'global' or 'independent'")
    return masked
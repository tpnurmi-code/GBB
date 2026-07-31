"""Checkpoint serialization."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(model, optimizer, epoch: int, loss: float, filename) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(loss),
        },
        path,
    )
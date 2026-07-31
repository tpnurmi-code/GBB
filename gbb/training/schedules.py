"""Developmental schedules for biological constraints."""

from __future__ import annotations

from gbb.config import config


def neural_development(epoch: int, total_epochs: int, model=None) -> tuple[float, float]:
    del model
    progress = epoch / max(1, total_epochs)
    if progress < 0.30:
        return 0.1, 0.1
    if progress < 0.70:
        phase = (progress - 0.30) / 0.40
        multiplier = 0.1 + 0.9 * phase
        return multiplier, multiplier
    return 1.0, 1.0


def get_dynamic_bioconst_lambda(
    epoch: int,
    target_val: float = 5e-6,
    total_epochs: int | None = None,
) -> float:
    total_epochs = int(config.NUM_EPOCHS if total_epochs is None else total_epochs)
    progress = epoch / max(1, total_epochs)
    if progress < 0.30:
        return 0.0
    if progress < 0.70:
        return float(target_val) * ((progress - 0.30) / 0.40)
    return float(target_val)
"""Hard pruning utilities."""

from __future__ import annotations

import torch


def hard_prune(model, optimizer, threshold: float = 1e-4) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    for name, parameter in raw_model.named_parameters():
        if not any(
            token in name
            for token in ("spline_weights", "spline_coeffs", "base_linear.weight")
        ):
            continue

        with torch.no_grad():
            mask = (parameter.abs() > threshold).to(parameter.dtype)
            parameter.mul_(mask)
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key].mul_(mask)

        old_handle = getattr(parameter, "_pruning_hook_handle", None)
        if old_handle is not None:
            old_handle.remove()
        parameter._pruning_hook_handle = parameter.register_hook(
            lambda gradient, fixed_mask=mask: gradient * fixed_mask
        )
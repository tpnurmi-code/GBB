"""General model regularizers."""

from __future__ import annotations

import torch

from gbb.config import config


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def calculate_sparsity_loss(model) -> torch.Tensor:
    raw_model = _unwrap_model(model)
    selected = [
        parameter.abs().sum()
        for name, parameter in raw_model.named_parameters()
        if "spline_weights" in name
        or "spline_coeffs" in name
        or "base_linear.weight" in name
    ]
    return torch.stack(selected).sum() if selected else next(raw_model.parameters()).new_zeros(())


def calculate_cfc_population_regularization(model, adj: torch.Tensor | None = None) -> torch.Tensor:
    raw_model = _unwrap_model(model)
    cfc = raw_model.cfc
    total = next(cfc.parameters()).new_zeros(())

    if adj is not None:
        adjacency = adj.to(device=cfc.node_bias.device, dtype=cfc.node_bias.dtype).clone()
        if adjacency.shape == (cfc.num_nodes, cfc.num_nodes):
            adjacency.fill_diagonal_(0.0)
            bias_difference = (
                cfc.node_bias[:, None, :] - cfc.node_bias[None, :, :]
            ).square().mean(dim=-1)
            total = total + float(config.CFC_NODE_BIAS_SMOOTH_W) * (
                adjacency * bias_difference
            ).sum() / adjacency.sum().clamp_min(1e-8)

    total = total + float(config.CFC_NODE_BIAS_L2_W) * cfc.node_bias.square().mean()
    if hasattr(cfc, "last_gate"):
        gate = cfc.last_gate
        gate_bounds = torch.relu(float(config.CFC_GATE_MIN) - gate).mean()
        gate_bounds = gate_bounds + torch.relu(gate - float(config.CFC_GATE_MAX)).mean()
        gate_mean = (gate.mean() - float(config.CFC_GATE_TARGET)).square()
        total = total + float(config.CFC_GATE_REG_W) * (gate_bounds + gate_mean)
    if hasattr(cfc, "last_h_delta"):
        total = total + float(config.CFC_DELTA_ENERGY_W) * cfc.last_h_delta.square().mean()
    if hasattr(cfc, "last_f_val"):
        bound = torch.relu(cfc.last_f_val.abs() - float(config.CFC_F_ACTIVITY_MAX)).square().mean()
        total = total + float(config.CFC_F_ACTIVITY_W) * bound

    drive = cfc.get_intrinsic_drive()
    total = total + float(config.CFC_DRIVE_BALANCE_W) * drive.mean().square()
    drive_bound = torch.relu(drive.abs() - float(config.CFC_DRIVE_MAX_ABS)).square().mean()
    total = total + float(config.CFC_DRIVE_BOUND_W) * drive_bound
    return total
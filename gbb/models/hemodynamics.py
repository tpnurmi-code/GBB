"""Lightweight hemodynamic observation model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HemodynamicObservationHead(nn.Module):
    """Apply a learned causal temporal kernel to future neural predictions."""

    def __init__(self, kernel_size: int = 5, init_mode: str = "hrf", num_nodes: int | None = None) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)
        self.num_nodes = num_nodes

        templates = {
            ("hrf", 3): [0.10, 0.35, 0.55],
            ("hrf", 5): [0.05, 0.15, 0.35, 0.30, 0.15],
            ("hrf", 7): [0.03, 0.07, 0.15, 0.30, 0.25, 0.14, 0.06],
            ("cbv", 3): [0.15, 0.55, 0.30],
            ("cbv", 5): [0.05, 0.15, 0.40, 0.25, 0.15],
        }
        initial = torch.tensor(
            templates.get((init_mode.lower(), kernel_size), [1.0 / kernel_size] * kernel_size),
            dtype=torch.float32,
        )
        initial /= initial.sum().clamp_min(1e-8)

        self.kernel_logits = nn.Parameter(torch.log(initial + 1e-6))
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        if num_nodes is None:
            self.node_log_gain = None
            self.node_bias = None
        else:
            self.node_log_gain = nn.Parameter(torch.zeros(1, 1, num_nodes))
            self.node_bias = nn.Parameter(torch.zeros(1, 1, num_nodes))

    def forward(self, neural_pred: torch.Tensor) -> torch.Tensor:
        if neural_pred.ndim != 3:
            raise ValueError(f"Expected (batch, horizon, nodes), got {tuple(neural_pred.shape)}")
        batch_size, horizon, num_nodes = neural_pred.shape
        if self.num_nodes is not None and num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

        x = neural_pred.permute(0, 2, 1).reshape(batch_size * num_nodes, 1, horizon)
        kernel = torch.softmax(self.kernel_logits, dim=0).view(1, 1, -1)
        y = F.conv1d(F.pad(x, (self.kernel_size - 1, 0)), kernel)[..., :horizon]
        y = y.reshape(batch_size, num_nodes, horizon).permute(0, 2, 1)
        output = torch.exp(self.log_gain) * y + self.bias
        if self.node_log_gain is not None:
            output = torch.exp(self.node_log_gain) * output + self.node_bias
        return output
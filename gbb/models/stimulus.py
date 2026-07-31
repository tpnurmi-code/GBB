"""Stimulus encoders."""

from __future__ import annotations

import torch
from torch import nn


class StimulusTemporalEncoder(nn.Module):
    """Map ``(batch, time, channels)`` stimulus data to a hidden sequence."""

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_dim <= 0 or kernel_size <= 0:
            raise ValueError("in_channels, hidden_dim, and kernel_size must be positive")
        middle_dim = max(hidden_dim // 2, 8)
        self.in_channels = int(in_channels)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                middle_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv1d(middle_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, stimulus: torch.Tensor) -> torch.Tensor:
        if stimulus.ndim != 3 or stimulus.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected stimulus (batch, time, {self.in_channels}), got {tuple(stimulus.shape)}"
            )
        return self.net(stimulus.transpose(1, 2)).transpose(1, 2)
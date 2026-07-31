"""Closed-form continuous-time population dynamics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gbb.config import config


class CfCLayer(nn.Module):
    """Node-wise continuous-time dynamics with interpretable time constants."""

    def __init__(self, input_size: int, hidden_size: int, num_nodes: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_nodes <= 0:
            raise ValueError("input_size, hidden_size, and num_nodes must be positive")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_nodes = int(num_nodes)
        self.input_projection: nn.Module = (
            nn.Identity()
            if input_size == hidden_size
            else nn.Linear(input_size, hidden_size)
        )
        combined_size = hidden_size * 2
        self.f_neural = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.g_gate = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.Sigmoid(),
        )
        self.tau_system = nn.Linear(hidden_size, 1)
        self.drive_head = nn.Linear(hidden_size, 1)
        self.node_bias = nn.Parameter(torch.randn(num_nodes, hidden_size) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        timespan: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or h_prev.ndim != 3:
            raise ValueError("x and h_prev must both have shape (batch, nodes, features)")
        if x.shape[:2] != h_prev.shape[:2]:
            raise ValueError(f"x and h_prev disagree: {tuple(x.shape)} vs {tuple(h_prev.shape)}")
        if x.shape[1] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.shape[1]}")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input feature size {self.input_size}, got {x.shape[-1]}")
        if h_prev.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden feature size {self.hidden_size}, got {h_prev.shape[-1]}")

        x_hidden = self.input_projection(x)
        x_biased = x_hidden + self.node_bias.unsqueeze(0)
        combined = torch.cat([x_biased, h_prev], dim=-1)
        candidate_base = self.f_neural(combined)
        intrinsic_drive = torch.tanh(self.drive_head(candidate_base))
        # The signed scalar drive is broadcast over hidden features. Unlike the
        # old code, the exported drive now participates in the state update and
        # therefore receives task-loss gradients.
        candidate = candidate_base + intrinsic_drive
        gate = self.g_gate(combined)
        tau = self._tau_from_raw(self.tau_system(candidate))
        dt = self._broadcast_timespan(timespan, x)
        decay = torch.exp(-dt / tau.clamp_min(1e-4))
        relaxed = decay * h_prev + (1.0 - decay) * candidate
        h_final = gate * relaxed + (1.0 - gate) * h_prev

        self.last_tau = tau
        self.last_gate = gate
        self.last_f_val = candidate
        self.last_intrinsic_drive = intrinsic_drive
        self.last_h_delta = h_final - h_prev
        return h_final, tau

    @staticmethod
    def _broadcast_timespan(timespan: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timespan):
            return torch.full(
                (reference.shape[0], 1, 1),
                float(timespan),
                device=reference.device,
                dtype=reference.dtype,
            )
        dt = timespan.to(device=reference.device, dtype=reference.dtype)
        if dt.ndim == 0:
            dt = dt.view(1, 1, 1).expand(reference.shape[0], -1, -1)
        elif dt.ndim == 1:
            dt = dt.view(-1, 1, 1)
        elif dt.ndim == 2:
            dt = dt.unsqueeze(-1)
        if dt.shape[0] not in (1, reference.shape[0]):
            raise ValueError(f"Could not broadcast timespan shape {tuple(dt.shape)}")
        return dt

    def _resting_candidate_base(self) -> torch.Tensor:
        hidden_zero = torch.zeros_like(self.node_bias)
        return self.f_neural(torch.cat([self.node_bias, hidden_zero], dim=-1))

    def _resting_candidate(self) -> torch.Tensor:
        candidate_base = self._resting_candidate_base()
        return candidate_base + torch.tanh(self.drive_head(candidate_base))

    def get_tau_values(self) -> torch.Tensor:
        """Return one effective time constant in seconds per node."""
        return self._tau_from_raw(self.tau_system(self._resting_candidate())).squeeze(-1)

    def get_tau_secs(self) -> torch.Tensor:
        return self.get_tau_values()

    def get_tau_values_avg(self) -> torch.Tensor:
        """Backward-compatible repaired alias; returns one value per node."""
        return self.get_tau_values()

    def get_intrinsic_drive(self) -> torch.Tensor:
        """Return a signed resting-state intrinsic-drive score per node."""
        return torch.tanh(self.drive_head(self._resting_candidate_base())).squeeze(-1)

    def get_causal_drive(self) -> torch.Tensor:
        """Deprecated compatibility alias for :meth:`get_intrinsic_drive`."""
        return self.get_intrinsic_drive()

    @staticmethod
    def _tau_from_raw(raw_tau: torch.Tensor) -> torch.Tensor:
        if getattr(config, "TAU_BOUND_MODE", "sigmoid") == "sigmoid":
            tau_min = float(config.TAU_MIN_PHYS)
            tau_max = float(config.TAU_MAX_PHYS)
            if tau_max <= tau_min:
                raise ValueError("TAU_MAX_PHYS must be larger than TAU_MIN_PHYS")
            return tau_min + (tau_max - tau_min) * torch.sigmoid(raw_tau)
        return F.softplus(raw_tau) + float(config.TAU_MIN_PHYS)
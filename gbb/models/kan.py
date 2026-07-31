"""FastKAN-style graph interaction layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gbb.config import config


class FastKANLinear(nn.Module):
    """Linear residual plus fixed Gaussian radial-basis expansion."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 5,
        grid_range: tuple[float, float] = (-4.0, 4.0),
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or num_grids < 2:
            raise ValueError("in_features/out_features must be positive and num_grids >= 2")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_grids = int(num_grids)
        minimum, maximum = grid_range
        mu = torch.linspace(minimum, maximum, num_grids)
        sigma = torch.full((num_grids,), (maximum - minimum) / (num_grids - 1))
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.spline_weights = nn.Parameter(
            torch.randn(out_features, in_features, num_grids) * config.SPLINECOEFF
        )
        self.base_linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dimension {self.in_features}, got {x.shape[-1]}")
        leading_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        basis = torch.exp(
            -((flat.unsqueeze(-1) - self.mu) ** 2) / (2.0 * self.sigma.square())
        )
        spline = torch.einsum("big,oig->bo", basis, self.spline_weights)
        output = self.base_linear(flat) + spline
        return output.reshape(*leading_shape, self.out_features)


class MultiHeadFastKANLayer(nn.Module):
    """Multi-head graph layer with FastKAN edge-score functions."""

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        out_dim: int,
        num_heads: int = 3,
        num_basis: int = 5,
    ) -> None:
        super().__init__()
        if min(num_nodes, in_dim, out_dim, num_heads, num_basis) <= 0:
            raise ValueError("All dimensions must be positive")
        self.num_nodes = int(num_nodes)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.num_basis = int(num_basis)

        self.spline_coeffs = nn.Parameter(
            torch.randn(num_nodes, num_nodes, num_heads, num_basis)
            * config.SPLINECOEFF
        )
        self.edge_attenuation_logits = nn.Parameter(
            torch.zeros(num_nodes, num_nodes, num_heads)
        )
        self.register_buffer("mu", torch.linspace(-4.0, 4.0, num_basis))
        self.register_buffer("sigma", torch.full((num_basis,), 8.0 / max(1, num_basis)))
        self.pair_proj = nn.Linear(in_dim * 2, 1)
        self.value_proj = nn.Linear(in_dim, out_dim, bias=False)

        signs = torch.ones(num_heads)
        configured = list(getattr(config, "GRAPH_HEAD_SIGNS", [1.0, -1.0, 0.5]))
        for index in range(min(num_heads, len(configured))):
            signs[index] = float(configured[index])
        self.register_buffer("head_message_signs", signs)

    @property
    def conduction_delays(self) -> torch.Tensor:
        """Deprecated checkpoint/API alias; values are attenuation logits, not delays."""
        return self.edge_attenuation_logits

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old_key = prefix + "conduction_delays"
        new_key = prefix + "edge_attenuation_logits"
        if old_key in state_dict and new_key not in state_dict:
            old_value = state_dict.pop(old_key)
            # Old values were non-negative and used through exp(-abs(x)). Convert
            # approximately to logits of the same attenuation probability.
            gate = torch.exp(-old_value.abs()).clamp(1e-4, 1.0 - 1e-4)
            state_dict[new_key] = torch.logit(gate)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h.ndim != 3:
            raise ValueError(f"Expected h=(batch,nodes,features), got {tuple(h.shape)}")
        batch_size, num_nodes, channels = h.shape
        if num_nodes != self.num_nodes or channels != self.in_dim:
            raise ValueError(
                f"Expected nodes/features ({self.num_nodes}, {self.in_dim}), "
                f"got ({num_nodes}, {channels})"
            )

        pair_weight = self.pair_proj.weight.squeeze(0)
        source_weight = pair_weight[:channels].unsqueeze(0)
        target_weight = pair_weight[channels:].unsqueeze(0)
        source_score = F.linear(h, source_weight)
        target_score = F.linear(h, target_weight)
        pair_scalar = target_score.unsqueeze(2) + source_score.unsqueeze(1)
        pair_scalar = pair_scalar + self.pair_proj.bias.view(1, 1, 1, 1)

        basis = torch.exp(
            -(
                (pair_scalar.unsqueeze(-1) - self.mu.view(1, 1, 1, 1, -1))
                ** 2
            )
            / (2.0 * self.sigma.view(1, 1, 1, 1, -1).square())
        )
        head_scores = torch.einsum(
            "bnmzg,nmhg->bnmh", basis, self.spline_coeffs
        )
        attenuation = torch.sigmoid(self.edge_attenuation_logits).clamp_min(1e-6)
        head_scores = head_scores + torch.log(attenuation).unsqueeze(0)

        if adj is not None:
            if adj.ndim == 2:
                if adj.shape != (num_nodes, num_nodes):
                    raise ValueError(f"Unexpected adjacency shape {tuple(adj.shape)}")
                adj_batch = adj.unsqueeze(0)
            elif adj.ndim == 3:
                if adj.shape[-2:] != (num_nodes, num_nodes):
                    raise ValueError(f"Unexpected adjacency shape {tuple(adj.shape)}")
                adj_batch = adj
            else:
                raise ValueError(f"Unexpected adjacency shape {tuple(adj.shape)}")

            adj_batch = adj_batch.to(device=h.device, dtype=h.dtype)
            identity = torch.eye(num_nodes, device=h.device, dtype=torch.bool).unsqueeze(0)
            support = (adj_batch > 0) | identity
            prior = torch.log(adj_batch.clamp_min(1e-6))
            head_scores = head_scores + float(config.ADJ_PRIOR_STRENGTH) * prior.unsqueeze(-1)
            head_scores = head_scores.masked_fill(~support.unsqueeze(-1), -torch.inf)

        attention = torch.softmax(head_scores, dim=2)
        values = self.value_proj(h)
        messages = torch.einsum("bnmh,bmo->bnho", attention, values)

        if bool(config.ENABLE_SIGNED_HEAD_MESSAGES):
            signs = self.head_message_signs.to(device=h.device, dtype=h.dtype).view(
                1, 1, self.num_heads, 1
            )
            messages = messages * signs
            signed_attention = attention * signs.squeeze(-1).unsqueeze(2)
            attention_map = signed_attention.mean(dim=-1)
        else:
            attention_map = attention.mean(dim=-1)

        return messages.mean(dim=2), attention_map, attention
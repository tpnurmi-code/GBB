"""Full Glass-Box Brain model."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from gbb.config import config
from gbb.models.cfc import CfCLayer
from gbb.models.feature_extractors import (
    H1_CNN_Extractor,
    H2_MLP_Extractor,
    H3_CNNLSTM_Extractor,
    H4_LSTM_Extractor,
    H5_LocalTransformer_Extractor,
    H6_GlobalTransformer_Extractor,
)
from gbb.models.hemodynamics import HemodynamicObservationHead
from gbb.models.kan import MultiHeadFastKANLayer
from gbb.models.stimulus import StimulusTemporalEncoder


class MesocortGBB(nn.Module):
    """Feature extractor -> FastKAN graph -> CfC dynamics -> observation head."""

    def __init__(
        self,
        num_nodes: int,
        time_points: int,
        freeze_extractor: bool = False,
        sensory_mask: torch.Tensor | None = None,
        model_type: str | None = None,
        hidden_dim: int | None = None,
        use_hemodynamic_head: bool | None = None,
        allow_all_nodes: bool = False,
    ) -> None:

        super().__init__()
        if num_nodes <= 0 or time_points <= 0:
            raise ValueError("num_nodes and time_points must be positive")

        self.num_nodes = int(num_nodes)
        self.time_points = int(time_points)
        self.model_type = str(model_type or config.MODEL_TYPE).upper()
        self.freeze_extractor = bool(freeze_extractor)
                self.stimulus_mode = str(config.STIMULUS_MODE).upper()

        if self.stimulus_mode not in {"EVENTS", "DENSE", "NONE"}:
            raise ValueError(
                f"Unknown STIMULUS_MODE: {self.stimulus_mode}"
            )

        self.stimulus_enabled = self.stimulus_mode != "NONE"
        self.allow_all_nodes = bool(allow_all_nodes)

	self.use_hemodynamic_head = (
            bool(config.USE_HEMODYNAMIC_HEAD)
            if use_hemodynamic_head is None
            else bool(use_hemodynamic_head)
        )

        feature_dim = int(hidden_dim or config.FEAT_EXT_HIDDEN)
        target_dim = int(config.CFC_BACKBONE_UNITS)
        self.extractor = self._build_extractor(feature_dim)
        self.adapter: nn.Module = (
            nn.Identity() if feature_dim == target_dim else nn.Linear(feature_dim, target_dim)
        )

        if self.freeze_extractor:
            for parameter in self.extractor.parameters():
                parameter.requires_grad_(False)
            self.extractor.eval()

        self.kan_layers = nn.ModuleList(
            [
                MultiHeadFastKANLayer(
                    num_nodes=self.num_nodes,
                    in_dim=target_dim,
                    out_dim=target_dim,
                    num_heads=int(config.KAN_HEADS),
                    num_basis=int(config.KAN_BASIS_FUNCTIONS),
                )
                for _ in range(int(config.KAN_LAYERS))
            ]
        )
        self.cfc = CfCLayer(target_dim, target_dim, self.num_nodes)
        self.fmri_step_proj = nn.Linear(1, target_dim)
        self.stim_encoder = StimulusTemporalEncoder(
            in_channels=int(config.STIMULUS_INPUT_CHANNELS),
            hidden_dim=target_dim,
            kernel_size=int(config.STIM_ENCODER_KERNEL),
        )
        # Retained for the explicit static-stimulus ablation path.
        self.stim_proj = nn.Linear(int(config.STIMULUS_INPUT_CHANNELS), target_dim)
        self.dropout = nn.Dropout(float(config.DROPOUT))
        self.norm_decoder = nn.LayerNorm(target_dim)
        self.decoder = nn.Linear(target_dim, int(config.PREDICTION_HORIZON))
        self.hemo_head = HemodynamicObservationHead(
            kernel_size=int(config.HEMO_KERNEL_SIZE),
            init_mode=str(config.HEMO_INIT),
            num_nodes=self.num_nodes,
        )

        if sensory_mask is None:
            if not self.stimulus_enabled:
                sensory_mask = torch.zeros(
                    self.num_nodes,
                    1,
                    dtype=torch.float32,
                )
            elif self.allow_all_nodes:
                sensory_mask = torch.ones(
                    self.num_nodes,
                    1,
                    dtype=torch.float32,
                )
            else:
                raise ValueError(
                    "sensory_mask is required when stimulus input is enabled. "
                    "Set allow_all_nodes=True only for an intentional "
                    "whole-network ablation."
                )

        sensory_mask = torch.as_tensor(
            sensory_mask,
            dtype=torch.float32,
        ).reshape(-1)

        if sensory_mask.numel() != self.num_nodes:
            raise ValueError(
                f"sensory_mask contains {sensory_mask.numel()} values; "
                f"expected {self.num_nodes}"
            )

        if not torch.isfinite(sensory_mask).all():
            raise ValueError(
                "sensory_mask contains non-finite values"
            )

        if not torch.all(
            (sensory_mask == 0) | (sensory_mask == 1)
        ):
            raise ValueError(
                "sensory_mask must be binary (0 or 1)"
            )

        selected_nodes = int(sensory_mask.sum().item())

        if self.stimulus_enabled and selected_nodes == 0:
            raise ValueError(
                "sensory_mask selects no nodes while stimulus input is enabled"
            )

        if not self.stimulus_enabled and selected_nodes != 0:
            raise ValueError(
                "STIMULUS_MODE='NONE' requires an empty sensory_mask"
            )

        if (
            selected_nodes == self.num_nodes
            and not self.allow_all_nodes
        ):
            raise ValueError(
                "sensory_mask selects every node. Set allow_all_nodes=True "
                "only for an intentional whole-network ablation."
            )

        self.register_buffer(
            "sensory_mask",
            sensory_mask.reshape(self.num_nodes, 1),
        )
        self.last_kan_input: torch.Tensor | None = None

    def _build_extractor(self, hidden_dim: int) -> nn.Module:
        common = dict(input_length=self.time_points, hidden_dim=hidden_dim)
        dropout = float(config.FEAT_EXT_DROPOUT)
        if self.model_type == "H1":
            return H1_CNN_Extractor(
                **common,
                layers=int(config.CNN_LAYERS),
                dropout=dropout,
                stride=int(config.CNN_STRIDE),
            )
        if self.model_type == "H2":
            return H2_MLP_Extractor(
                **common,
                layers=int(config.MLP_LAYERS),
                dropout=dropout,
            )
        if self.model_type == "H3":
            return H3_CNNLSTM_Extractor(
                **common,
                dropout=dropout,
                cnn_layers=int(config.CNN_LAYERS),
                rnn_layers=int(config.RNN_LAYERS),
                stride=int(config.CNN_STRIDE),
            )
        if self.model_type == "H4":
            return H4_LSTM_Extractor(
                **common,
                dropout=dropout,
                rnn_layers=int(config.RNN_LAYERS),
            )
        if self.model_type == "H5":
            return H5_LocalTransformer_Extractor(
                **common,
                layers=int(config.TRANSFORMER_LAYERS),
                dropout=dropout,
                attention_radius=int(config.LOCAL_ATTENTION_RADIUS),
            )
        if self.model_type == "H6":
            return H6_GlobalTransformer_Extractor(
                **common,
                layers=int(config.TRANSFORMER_LAYERS),
                dropout=dropout,
            )
        raise ValueError(f"Unknown MODEL_TYPE: {self.model_type}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_extractor:
            self.extractor.eval()
        return self

    def forward(
        self,
        fmri_window: torch.Tensor,
        stim_window: torch.Tensor | None,
        adj: torch.Tensor | None,
        return_head_weights: bool = False,
    ):
        if fmri_window.ndim != 3:
            raise ValueError(
                f"Expected fmri_window=(batch,time,nodes), got {tuple(fmri_window.shape)}"
            )
        batch_size, time_points, num_nodes = fmri_window.shape
        if time_points != self.time_points or num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected time/nodes ({self.time_points}, {self.num_nodes}), "
                f"got ({time_points}, {num_nodes})"
            )
        expected_stimulus_shape = (
            batch_size,
            time_points,
            int(config.STIMULUS_INPUT_CHANNELS),
        )

        if self.stimulus_enabled:
            if stim_window is None:
                raise ValueError(
                    "stim_window is required when "
                    f"STIMULUS_MODE={self.stimulus_mode!r}"
                )

            if tuple(stim_window.shape) != expected_stimulus_shape:
                raise ValueError(
                    f"Expected stim_window={expected_stimulus_shape}, "
                    f"got {tuple(stim_window.shape)}"
                )

            if not torch.isfinite(stim_window).all():
                raise ValueError(
                    "stim_window contains non-finite values"
                )

        else:
            if self.stimulus_enabled:
                if tuple(stim_window.shape) != expected_stimulus_shape:
                    raise ValueError(
                        "Expected a zero placeholder stimulus with shape "
                        f"{expected_stimulus_shape}; got "
                        f"{tuple(stim_window.shape)}"
                    )

                if torch.count_nonzero(stim_window).item() != 0:
                    raise ValueError(
                        "A non-zero stim_window was supplied while "
                        "STIMULUS_MODE='NONE'"
                    )

            stim_window = None

        per_node_series = fmri_window.permute(0, 2, 1).reshape(
            batch_size * num_nodes, time_points, 1
        )
        context_manager = torch.no_grad() if self.freeze_extractor else nullcontext()
        with context_manager:
            extracted = self.extractor(per_node_series)
        context = self.adapter(extracted).reshape(batch_size, num_nodes, -1)

        drives = self.fmri_step_proj(fmri_window.unsqueeze(-1)) + context.unsqueeze(1)
        if stim_window is not None:
            mask = self.sensory_mask.view(1, 1, num_nodes, 1)
            if bool(config.USE_TEMPORAL_STIM_ENCODER):
                stimulus_sequence = self.stim_encoder(stim_window)
                drives = drives + stimulus_sequence.unsqueeze(2) * mask
            else:
                stimulus_static = self.stim_proj(stim_window.mean(dim=1))
                drives = drives + stimulus_static[:, None, None, :] * mask

        self.last_kan_input = drives[:, -1]
        hidden = context
        total_attention: torch.Tensor | None = None
        all_head_weights: list[torch.Tensor] | None = [] if return_head_weights else None
        timespan = torch.full(
            (batch_size, 1),
            float(config.TR),
            device=fmri_window.device,
            dtype=fmri_window.dtype,
        )

        for time_index in range(time_points):
            local_drive = drives[:, time_index]
            graph_state = local_drive
            for layer in self.kan_layers:
                neighbor_message, attention_map, head_weights = layer(graph_state, adj)
                graph_state = graph_state + neighbor_message
                total_attention = (
                    attention_map
                    if total_attention is None
                    else total_attention + attention_map
                )
                if all_head_weights is not None:
                    all_head_weights.append(head_weights)

            graph_mix = float(config.GRAPH_MIX)
            graph_state = graph_mix * graph_state + (1.0 - graph_mix) * local_drive
            hidden, _ = self.cfc(graph_state, hidden, timespan)

        if total_attention is None:
            average_attention = None
        else:
            average_attention = total_attention / (
                time_points * max(1, len(self.kan_layers))
            )

        hidden = self.norm_decoder(self.dropout(hidden))
        neural_delta = F.hardtanh(self.decoder(hidden), min_val=-4.0, max_val=4.0)
        neural_prediction = fmri_window[:, -1, :, None] + neural_delta
        neural_prediction = neural_prediction.permute(0, 2, 1).contiguous()
        prediction = (
            self.hemo_head(neural_prediction)
            if self.use_hemodynamic_head
            else neural_prediction
        )
        return prediction, average_attention, hidden, all_head_weights

    @property
    def tau_values(self) -> torch.Tensor:
        return self.cfc.get_tau_values()

    @property
    def intrinsic_drive(self) -> torch.Tensor:
        return self.cfc.get_intrinsic_drive()

    @property
    def causal_drive(self) -> torch.Tensor:
        """Deprecated compatibility alias for ``intrinsic_drive``."""
        return self.intrinsic_drive
"""Construction helpers for model components."""

from __future__ import annotations

import torch
from torch import nn

from gbb.config import config
from gbb.models.feature_extractors import (
    H1_CNN_Extractor,
    H2_MLP_Extractor,
    H3_CNNLSTM_Extractor,
    H4_LSTM_Extractor,
    H5_LocalTransformer_Extractor,
    H6_GlobalTransformer_Extractor,
)
from gbb.models.mesocort_gbb import MesocortGBB


def build_feature_extractor(
    model_type: str,
    input_length: int,
    hidden_dim: int | None = None,
) -> nn.Module:
    """Build H1-H6 using the flat compatibility configuration."""
    hidden_dim = int(hidden_dim or config.FEAT_EXT_HIDDEN)
    model_type = model_type.upper()
    common = dict(input_length=input_length, hidden_dim=hidden_dim)
    if model_type == "H1":
        return H1_CNN_Extractor(
            **common,
            layers=config.CNN_LAYERS,
            dropout=config.FEAT_EXT_DROPOUT,
            stride=config.CNN_STRIDE,
        )
    if model_type == "H2":
        return H2_MLP_Extractor(
            **common,
            layers=config.MLP_LAYERS,
            dropout=config.FEAT_EXT_DROPOUT,
        )
    if model_type == "H3":
        return H3_CNNLSTM_Extractor(
            **common,
            dropout=config.FEAT_EXT_DROPOUT,
            cnn_layers=config.CNN_LAYERS,
            rnn_layers=config.RNN_LAYERS,
            stride=config.CNN_STRIDE,
        )
    if model_type == "H4":
        return H4_LSTM_Extractor(
            **common,
            dropout=config.FEAT_EXT_DROPOUT,
            rnn_layers=config.RNN_LAYERS,
        )
    if model_type == "H5":
        return H5_LocalTransformer_Extractor(
            **common,
            layers=config.TRANSFORMER_LAYERS,
            dropout=config.FEAT_EXT_DROPOUT,
            attention_radius=config.LOCAL_ATTENTION_RADIUS,
        )
    if model_type == "H6":
        return H6_GlobalTransformer_Extractor(
            **common,
            layers=config.TRANSFORMER_LAYERS,
            dropout=config.FEAT_EXT_DROPOUT,
        )
    raise ValueError(f"Unknown model_type={model_type}")


def build_model(
    num_nodes: int,
    time_points: int,
    sensory_mask: torch.Tensor | None = None,
    *,
    model_type: str | None = None,
    hidden_dim: int | None = None,
    use_hemodynamic_head: bool | None = None,
    freeze_extractor: bool = False,
    allow_all_nodes: bool = False,
) -> MesocortGBB:
    """Build a full model with a signature matching :class:`MesocortGBB`."""
    return MesocortGBB(
        num_nodes=num_nodes,
        time_points=time_points,
        sensory_mask=sensory_mask,
        model_type=model_type,
        hidden_dim=hidden_dim,
        use_hemodynamic_head=use_hemodynamic_head,
        freeze_extractor=freeze_extractor,
        allow_all_nodes=allow_all_nodes,
    )
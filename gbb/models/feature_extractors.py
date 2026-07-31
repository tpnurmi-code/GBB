"""Alternative H1-H6 temporal feature extractors."""

from __future__ import annotations

import torch
from torch import nn


def _as_time_channel(x: torch.Tensor) -> torch.Tensor:
    """Normalize input to ``(batch, time, 1)``."""
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got {tuple(x.shape)}")
    if x.shape[-1] == 1:
        return x
    if x.shape[1] == 1:
        return x.transpose(1, 2)
    raise ValueError(f"Expected a singleton channel dimension, got {tuple(x.shape)}")


class H1_CNN_Extractor(nn.Module):
    """Local temporal convolutional extractor."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int,
        layers: int = 2,
        dropout: float = 0.5,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if input_length <= 0 or stride <= 0:
            raise ValueError("input_length and stride must be positive")

        depth = max(1, int(layers))
        blocks: list[nn.Module] = [
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ]
        in_channels = 32
        for layer_index in range(1, depth):
            out_channels = 64
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=5,
                        padding=2,
                        stride=stride if layer_index == 1 else 1,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            flattened_dim = int(
                self.features(torch.zeros(1, 1, input_length)).flatten(1).shape[1]
            )
        self.features.train(was_training)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_time_channel(x).transpose(1, 2)
        return self.projection(self.features(x))


class H2_MLP_Extractor(nn.Module):
    """Fully mixed temporal MLP extractor."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int,
        layers: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        depth = max(2, int(layers))
        modules: list[nn.Module] = [nn.Flatten(), nn.Linear(input_length, 256), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 2):
            modules.extend([nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])
        modules.extend([nn.Linear(256, hidden_dim), nn.ReLU()])
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(_as_time_channel(x))


class H3_CNNLSTM_Extractor(nn.Module):
    """Local convolution followed by sequential recurrent integration."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int,
        dropout: float = 0.5,
        cnn_layers: int = 2,
        rnn_layers: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        del input_length
        conv_layers: list[nn.Module] = []
        in_channels = 1
        for layer_index in range(max(1, int(cnn_layers))):
            out_channels = 32
            conv_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=5,
                        padding=2,
                        stride=stride if layer_index == 0 else 1,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        conv_layers.append(nn.MaxPool1d(2))
        self.cnn = nn.Sequential(*conv_layers)
        rnn_layers = max(1, int(rnn_layers))
        self.lstm = nn.LSTM(
            in_channels,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_time_channel(x).transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.dropout(hidden[-1])


class H4_LSTM_Extractor(nn.Module):
    """Global recurrent temporal extractor."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int,
        dropout: float = 0.5,
        rnn_layers: int = 2,
    ) -> None:
        super().__init__()
        del input_length
        rnn_layers = max(1, int(rnn_layers))
        self.lstm = nn.LSTM(
            1,
            hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(_as_time_channel(x))
        return self.projection(hidden[-1])


class H5_LocalTransformer_Extractor(nn.Module):
    """Transformer with a sliding temporal attention window.

    The current Siamese model processes each parcel separately, so this mask is
    temporally local. It does not by itself implement local *spatial* attention.
    """

    def __init__(
        self,
        input_length: int,
        hidden_dim: int,
        layers: int = 2,
        dropout: float = 0.1,
        attention_radius: int = 4,
    ) -> None:
        super().__init__()
        model_dim = 64
        self.input_length = int(input_length)
        self.embedding = nn.Linear(1, model_dim)
        self.position = nn.Parameter(torch.randn(1, input_length, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.projection = nn.Linear(model_dim, hidden_dim)

        indexes = torch.arange(input_length)
        local = (indexes[:, None] - indexes[None, :]).abs() <= max(0, int(attention_radius))
        mask = torch.zeros(input_length, input_length)
        mask.masked_fill_(~local, float("-inf"))
        self.register_buffer("attention_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_time_channel(x)
        if x.shape[1] != self.input_length:
            raise ValueError(f"Expected {self.input_length} time points, got {x.shape[1]}")
        x = self.embedding(x) + self.position
        x = self.transformer(x, mask=self.attention_mask)
        return self.projection(x.mean(dim=1))


class H6_GlobalTransformer_Extractor(nn.Module):
    """Transformer with unrestricted temporal self-attention."""

    def __init__(
        self,
        input_length: int,
        hidden_dim: int,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        model_dim = 64
        self.input_length = int(input_length)
        self.embedding = nn.Linear(1, model_dim)
        self.position = nn.Parameter(torch.randn(1, input_length, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(layers)))
        self.projection = nn.Linear(model_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_time_channel(x)
        if x.shape[1] != self.input_length:
            raise ValueError(f"Expected {self.input_length} time points, got {x.shape[1]}")
        x = self.embedding(x) + self.position
        x = self.transformer(x)
        return self.projection(x.mean(dim=1))


# Compatibility aliases for old imports. New code should use the names above.
H5_Transformer_Extractor = H5_LocalTransformer_Extractor
H6_LocalTransformer_Extractor = H6_GlobalTransformer_Extractor
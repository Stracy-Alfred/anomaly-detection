"""
lstm_ae.py — LSTM Autoencoder for multivariate time-series reconstruction.

Architecture:
    Encoder LSTM  →  latent vector (last hidden state projected)
    Latent vector →  Decoder LSTM  →  linear output layer
    Loss = MSE between input window and reconstruction.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Encodes (batch, seq_len, input_dim) → (batch, latent_dim)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, D)
        _, (h_n, _) = self.lstm(x)       # h_n: (num_layers, B, hidden)
        latent = self.fc_latent(h_n[-1])  # use last layer's hidden → (B, latent)
        return latent


class LSTMDecoder(nn.Module):
    """Decodes (batch, latent_dim) → (batch, seq_len, input_dim)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.fc_expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim) → repeat across seq_len
        h = self.fc_expand(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, W, hidden)
        out, _ = self.lstm(h)              # (B, W, hidden)
        return self.fc_out(out)            # (B, W, output_dim)


class LSTMAutoencoder(nn.Module):
    """Full encoder–decoder with convenience helpers."""

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        seq_len: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window mean reconstruction error (scalar per sample)."""
        recon = self.forward(x)
        # Mean over (seq_len, features) → (B,)
        return ((x - recon) ** 2).mean(dim=(1, 2))

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ChannelSpatialAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Conv1d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=2)
        maximum = x.amax(dim=2)
        channel_mask = torch.sigmoid(self.channel(avg) + self.channel(maximum)).unsqueeze(2)
        x = x * channel_mask
        spatial_descriptor = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(spatial_descriptor))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels: int, dropout: float, use_cbam: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.attention = ChannelSpatialAttention1D(channels) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.attention(self.body(x)))


class NovaEncoder(nn.Module):
    """CPU-scaled implementation of the first-paper NOVA-XIDS feature path."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_cbam: bool = True,
        use_lstm: bool = True,
        use_self_attention: bool = True,
    ):
        super().__init__()
        channels = max(16, hidden_dim // 4)
        recurrent_dim = max(16, hidden_dim // 4)
        self.use_lstm = use_lstm
        self.use_self_attention = use_self_attention
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            ResidualAttentionBlock(channels, dropout, use_cbam),
            ResidualAttentionBlock(channels, dropout, use_cbam),
        )
        if use_lstm:
            self.temporal = nn.LSTM(channels, recurrent_dim, batch_first=True, bidirectional=True)
            output_dim = recurrent_dim * 2
        else:
            self.temporal = None
            output_dim = channels
        self.pool = nn.Sequential(nn.Linear(output_dim, output_dim // 2), nn.Tanh(), nn.Linear(output_dim // 2, 1))
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.project(x).unsqueeze(1)
        sequence = self.residual(self.stem(latent)).transpose(1, 2)
        if self.temporal is not None:
            sequence, _ = self.temporal(sequence)
        if self.use_self_attention:
            weights = torch.softmax(self.pool(sequence).squeeze(-1), dim=1)
            representation = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        else:
            representation = sequence.mean(dim=1)
        return representation


class NovaXIDSNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2, **encoder_flags):
        super().__init__()
        self.encoder = NovaEncoder(input_dim, latent_dim, hidden_dim, dropout, **encoder_flags)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.output_dim, num_classes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


class RareGuardXIDSNet(nn.Module):
    """NOVA-compatible backbone with a rare-class prototype memory head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_prototypes: bool = True,
        use_direct_path: bool = True,
        prototype_initial_scale: float = 1.0,
        prototype_gate_bias: float = -2.0,
        **encoder_flags,
    ):
        super().__init__()
        self.encoder = NovaEncoder(input_dim, latent_dim, hidden_dim, dropout, **encoder_flags)
        representation_dim = self.encoder.output_dim
        self.embedding = nn.Sequential(nn.Linear(representation_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.direct_path = (
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            if use_direct_path
            else None
        )
        self.fusion = (
            nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.GELU(),
                nn.LayerNorm(latent_dim),
            )
            if use_direct_path
            else None
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.use_prototypes = use_prototypes
        self.prototype_layer = nn.Linear(latent_dim, num_classes, bias=False) if use_prototypes else None
        self.prototype_log_scale = (
            nn.Parameter(
                torch.tensor(math.log(max(float(prototype_initial_scale), 1e-6)), dtype=torch.float32)
            )
            if use_prototypes
            else None
        )
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        if use_prototypes:
            nn.init.constant_(self.gate[0].bias, float(prototype_gate_bias))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(self.encoder(x))
        if self.direct_path is not None:
            embedding = self.fusion(torch.cat([embedding, self.direct_path(x)], dim=1))
        return embedding

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        if self.prototype_layer is not None:
            prototype_logits = F.linear(F.normalize(z, dim=1), F.normalize(self.prototype_layer.weight, dim=1))
            gate = self.gate(z)
            scale = self.prototype_log_scale.exp().clamp(1.0, 100.0)
            logits = logits + gate * scale * prototype_logits
        return logits, z


def prototype_compactness_loss(z: torch.Tensor, y: torch.Tensor, prototype_weight: torch.Tensor) -> torch.Tensor:
    prototypes = F.normalize(prototype_weight, dim=1)
    return (1.0 - (F.normalize(z, dim=1) * prototypes[y]).sum(dim=1)).mean()

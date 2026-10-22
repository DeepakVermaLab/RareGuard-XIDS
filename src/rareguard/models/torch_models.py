from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probability = F.log_softmax(logits, dim=1)
        log_pt = log_probability.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class FalconIDNet(nn.Module):
    """FALCON-ID style tabular sequence proxy: projection + CNN + BiLSTM + attention."""

    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim))
        self.conv = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(32, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        seq = z.unsqueeze(1)
        conv = self.conv(seq).transpose(1, 2)
        h, _ = self.lstm(conv)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


class RareGuardNet(nn.Module):
    """Rare-aware embedding head compatible with FALCON-ID latent size."""

    def __init__(self, input_dim: int, num_classes: int, latent_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.prototype_layer = nn.Linear(latent_dim, num_classes, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z) + self.prototype_layer(z)
        return logits, z


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T) / temperature
    labels = y.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=z.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    return -mean_log_prob_pos.mean()

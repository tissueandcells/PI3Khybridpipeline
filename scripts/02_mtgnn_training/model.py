"""
model.py — MT-GNN architecture for multi-task PI3K isoform pIC50 prediction.

Architecture:
  3 × GCNConv (Kipf & Welling, 2017) with BN + ReLU + Dropout(0.2)
  Global mean pool
  Shared bottleneck: 256 → 128 (ReLU, Dropout)
  Four isoform-specific linear heads → scalar pIC50

Two variants:
  MTGNN(in_dim=22) — v1 baseline (22-d atom features)
  MTGNN(in_dim=32) — v2 selectivity-aware (32-d features, used with
                     inverse-frequency-weighted MSE + pairwise ranking loss)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MTGNN(nn.Module):
    """Multi-task GCN for PI3K α/β/δ/γ pIC50 prediction."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Four isoform-specific heads
        self.head_alpha = nn.Linear(hidden_dim // 2, 1)
        self.head_beta  = nn.Linear(hidden_dim // 2, 1)
        self.head_delta = nn.Linear(hidden_dim // 2, 1)
        self.head_gamma = nn.Linear(hidden_dim // 2, 1)

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))

        h = global_mean_pool(h, batch)
        h = self.bottleneck(h)

        out = torch.stack(
            [
                self.head_alpha(h).squeeze(-1),
                self.head_beta(h).squeeze(-1),
                self.head_delta(h).squeeze(-1),
                self.head_gamma(h).squeeze(-1),
            ],
            dim=-1,
        )  # shape: [batch, 4]
        return out


def masked_mse_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE with optional per-isoform weights (for v2)."""
    err2 = (preds - labels) ** 2
    err2 = err2 * mask.float()
    if weights is not None:
        err2 = err2 * weights.view(1, -1)
    n = mask.sum().clamp(min=1)
    return err2.sum() / n


def pairwise_ranking_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pairwise ranking loss (v2 auxiliary) — encourages correct inter-isoform ordering."""
    loss = preds.new_zeros(())
    count = 0
    for i in range(preds.size(0)):
        avail = mask[i].nonzero(as_tuple=True)[0]
        if len(avail) < 2:
            continue
        for a_idx in range(len(avail)):
            for b_idx in range(a_idx + 1, len(avail)):
                a, b = avail[a_idx], avail[b_idx]
                # Target rank direction
                true_diff = labels[i, a] - labels[i, b]
                pred_diff = preds[i, a] - preds[i, b]
                # Margin ranking
                if true_diff.abs() < 1e-6:
                    continue
                margin = -torch.sign(true_diff) * pred_diff
                loss = loss + F.relu(margin + 0.1)
                count += 1
    return loss / max(count, 1)

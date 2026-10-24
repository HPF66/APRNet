from __future__ import annotations
import torch
from torch import nn

class ResidueContextBackbone(nn.Module):
    """Residue-context and chain/protein pooling backbone for the full model."""

    def __init__(self, hidden=128, heads=4, layers=1, dropout=0.1, local_decay=0.04) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError('hidden must be divisible by heads')
        self.hidden = hidden
        self.esm_proj = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, hidden))
        self.hapr_proj = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, hidden))
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_gate = nn.Sequential(nn.Linear(hidden * 3 + 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden))
        self.hapr_direct = nn.Linear(hidden, hidden)
        self.local_decay = nn.Parameter(torch.tensor(float(local_decay)))
        context_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.residue_context = nn.TransformerEncoder(context_layer, num_layers=layers)
        self.context_norm = nn.LayerNorm(hidden)
        self.residue_attention = nn.Linear(hidden, 1)
        self.apr_score = nn.Sequential(nn.Linear(hidden + 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.pool_projection = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Dropout(dropout))
        self.chain_attention = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def _relative_bias(self, length: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(length, device=device)
        distance = (idx[:, None] - idx[None, :]).abs()
        bias = -self.local_decay.abs() * distance.float()
        return bias

    def pool_chain(self, h: torch.Tensor, mask: torch.Tensor, papr: torch.Tensor | None=None, rapr: torch.Tensor | None=None) -> torch.Tensor:
        scores = self.residue_attention(h).squeeze(-1).masked_fill(~mask, -10000.0)
        attention = (scores.softmax(dim=-1).unsqueeze(-1) * h).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        mean = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        maximum = h.masked_fill(~mask.unsqueeze(-1), -10000.0).max(dim=1).values
        if papr is None or rapr is None:
            region = attention
        else:
            features = torch.cat([h, papr.unsqueeze(-1), rapr.unsqueeze(-1)], dim=-1)
            region_scores = self.apr_score(features).squeeze(-1).masked_fill(~mask, -10000.0)
            region = (region_scores.softmax(dim=-1).unsqueeze(-1) * h).sum(dim=1)
        return self.pool_projection(torch.cat([attention, mean, maximum, region], dim=-1))

    def pool_sample(self, chain_h: torch.Tensor, chain_to_sample: torch.Tensor, n_samples: int) -> torch.Tensor:
        pooled = []
        for sample_idx in range(n_samples):
            selected = chain_h[chain_to_sample == sample_idx]
            weights = self.chain_attention(selected).squeeze(-1).softmax(dim=0)
            pooled.append((weights.unsqueeze(-1) * selected).sum(dim=0))
        return torch.stack(pooled, dim=0)

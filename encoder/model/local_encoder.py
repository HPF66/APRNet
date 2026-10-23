from __future__ import annotations
from encoder.model.window_attention import WindowAttention
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch.nn import functional as F
import encoder.my_loader.window_data as trainer
import encoder.my_loader.labels as peptide_adapter
from encoder.utils.contrastive import hapr_supcon_loss
HIDDEN_SIZE = 192
WINDOW_SCALES = (6, 8, 10, 12, 16, 20)
EVIDENCE_FEATURES = len(WINDOW_SCALES) * 4

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

class IntegratedContext(nn.Module):
    """C2-family context blocks used inside the APR encoder."""

    def __init__(self, hidden: int, dropout: float, heads: int, radius: int, variant: str, scale: float) -> None:
        super().__init__()
        require(hidden % heads == 0, 'hidden size must be divisible by attention heads')
        self.variant = variant
        self.radius = radius
        self.scale = scale
        self.input_projection = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden), nn.Conv1d(hidden, hidden, 1), nn.SiLU()) for kernel in (3, 7, 15, 31)])
        self.mix = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout))
        self.scale_token_projection = nn.Sequential(nn.Linear(4, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.scale_embedding = nn.Parameter(torch.zeros(len(WINDOW_SCALES), hidden))
        self.scale_query = nn.Linear(hidden, hidden)
        self.scale_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.scale_norm = nn.LayerNorm(hidden)
        self.residue_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.residue_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.context_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        if variant == 'GATED_CONTEXT':
            self.external_gate = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1), nn.Sigmoid())
        else:
            self.external_gate = None

    def forward(self, local_h: Tensor, evidence: Tensor, mask: Tensor) -> Tensor:
        summary = peptide_summary(evidence)
        x = self.input_projection(torch.cat([local_h, summary], dim=-1)) * mask.unsqueeze(-1)
        conv = torch.cat([branch(x.transpose(1, 2)).transpose(1, 2) for branch in self.branches], dim=-1)
        conv = self.mix(conv) * mask.unsqueeze(-1)
        scale_values = evidence.view(evidence.shape[0], evidence.shape[1], len(WINDOW_SCALES), 4)
        tokens = self.scale_token_projection(scale_values) + self.scale_embedding.view(1, 1, len(WINDOW_SCALES), -1)
        query = self.scale_query(x).reshape(-1, 1, x.shape[-1])
        key_value = tokens.reshape(-1, len(WINDOW_SCALES), x.shape[-1])
        scale_context, _ = self.scale_attention(query, key_value, key_value, need_weights=False)
        scale_context = scale_context.reshape(x.shape) * mask.unsqueeze(-1)
        scale_context = self.scale_norm(x + scale_context) * mask.unsqueeze(-1)
        positions = torch.arange(x.shape[1], device=x.device)
        distance = torch.abs(positions[:, None] - positions[None, :])
        attn_mask = distance > self.radius
        residue_context, _ = self.residue_attention(scale_context, scale_context, scale_context, key_padding_mask=~mask, attn_mask=attn_mask, need_weights=False)
        residue_context = torch.nan_to_num(residue_context, nan=0.0, posinf=0.0, neginf=0.0)
        residue_context = self.residue_norm(residue_context + self.ff(residue_context)) * mask.unsqueeze(-1)
        if self.variant == 'HIER_CONTEXT':
            context = scale_context + residue_context
        else:
            context = conv + scale_context + residue_context
        gate = self.context_gate(torch.cat([local_h, context], dim=-1))
        if self.external_gate is not None:
            gate = gate * self.external_gate(torch.cat([local_h, context], dim=-1))
        residual = torch.tanh(self.residual_logit) * self.scale * gate * context
        return (local_h + residual) * mask.unsqueeze(-1)

class IntegratedAPRModel(nn.Module):
    """Frozen Peptide Teacher -> scale-aware local APR -> contextual HAPR."""

    def __init__(self, hidden: int, dropout: float, heads: int, radius: int, variant: str, context_scale: float) -> None:
        super().__init__()
        self.variant = variant
        self.peptide_projection = nn.Sequential(nn.Linear(trainer.WINDOW_WIDTH, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.window_scale_embedding = nn.Parameter(torch.zeros(len(WINDOW_SCALES), hidden))
        self.window_length_projection = nn.Sequential(nn.Linear(1, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.peptide_query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.peptide_query, std=0.02)
        self.window_attention = WindowAttention(hidden, heads, dropout)
        self.scale_window_attention = nn.ModuleList([WindowAttention(hidden, heads, dropout) for _ in WINDOW_SCALES])
        self.e2r_query = nn.Linear(hidden, hidden)
        self.e2r_key = nn.Linear(hidden, hidden)
        self.e2r_value = nn.Linear(hidden, hidden)
        self.e2r_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.e2r_norm = nn.LayerNorm(hidden)
        self.evidence_projection = nn.Sequential(nn.Linear(EVIDENCE_FEATURES, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.local_encoder = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.local_head = nn.Linear(hidden, 1)
        self.context = IntegratedContext(hidden, dropout, heads, radius, variant, context_scale)
        self.apr_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reliability_head = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.window_hidden_decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, trainer.WINDOW_WIDTH))

    def forward(self, base: Tensor, tokens: Tensor, starts: Tensor, lengths: Tensor, token_mask: Tensor, residue_mask: Tensor, token_probability: Tensor | None=None, token_local_probability: Tensor | None=None, token_scale: Tensor | None=None, residue_evidence: Tensor | None=None, profile_hidden: Tensor | None=None, profile_local_probability: Tensor | None=None, return_aux: bool=False, return_encoder: bool=False):
        del base, token_local_probability, profile_hidden, profile_local_probability
        require(residue_evidence is not None, 'residue_evidence is required')
        require(token_scale is not None, 'token_scale is required for C2_ROBUST')
        scale_ids = token_scale.round().long().clamp(0, len(WINDOW_SCALES) - 1)
        scale_h = self.window_scale_embedding[scale_ids]
        length_h = self.window_length_projection((lengths / float(max(WINDOW_SCALES))).unsqueeze(-1))
        peptide_tokens = self.peptide_projection(tokens) + scale_h + length_h
        peptide_tokens = peptide_tokens * token_mask.unsqueeze(-1)
        query = self.peptide_query.expand(tokens.shape[0], residue_mask.shape[1], -1)
        evidence_h = self.evidence_projection(residue_evidence)
        if self.variant == 'C2_CS_E2R':
            scale_views = []
            scale_presence = []
            for scale_index, scale_attention in enumerate(self.scale_window_attention):
                scale_mask = token_mask & (scale_ids == scale_index)
                scale_presence.append(scale_mask.any(dim=1))
                max_scale_windows = int(scale_mask.sum(dim=1).max().item())
                if max_scale_windows == 0:
                    scale_views.append(torch.zeros_like(query))
                    continue
                compact_tokens = peptide_tokens.new_zeros((peptide_tokens.shape[0], max_scale_windows, peptide_tokens.shape[-1]))
                compact_starts = starts.new_zeros((starts.shape[0], max_scale_windows))
                compact_lengths = lengths.new_ones((lengths.shape[0], max_scale_windows))
                compact_mask = torch.zeros((token_mask.shape[0], max_scale_windows), dtype=torch.bool, device=token_mask.device)
                for row in range(scale_mask.shape[0]):
                    selected = torch.nonzero(scale_mask[row], as_tuple=False).flatten()
                    count = min(int(selected.numel()), max_scale_windows)
                    if count:
                        compact_tokens[row, :count] = peptide_tokens[row, selected[:count]]
                        compact_starts[row, :count] = starts[row, selected[:count]]
                        compact_lengths[row, :count] = lengths[row, selected[:count]]
                        compact_mask[row, :count] = True
                scale_views.append(scale_attention(query, compact_tokens, compact_starts, compact_lengths, compact_mask, residue_mask))
            scale_stack = torch.stack(scale_views, dim=2)
            present = torch.stack(scale_presence, dim=1)
            require(bool(present.any(dim=1).all()), 'Every sequence needs at least one peptide scale')
            bsz, tlen, nscale, width = scale_stack.shape
            q = self.e2r_query(evidence_h)
            q = q.reshape(bsz * tlen, 1, width)
            kv = scale_stack.reshape(bsz * tlen, nscale, width)
            missing_scale_mask = (~present[:, None, :]).expand(bsz, tlen, nscale).reshape(bsz * tlen, nscale)
            fused, _ = self.e2r_attention(q, self.e2r_key(kv), self.e2r_value(kv), key_padding_mask=missing_scale_mask, need_weights=False)
            local_peptide = self.e2r_norm(fused.squeeze(1).reshape(bsz, tlen, width) + query)
            local_peptide = local_peptide * residue_mask.unsqueeze(-1)
        else:
            local_peptide = self.window_attention(query, peptide_tokens, starts, lengths, token_mask, residue_mask)
        local_h = self.local_encoder(torch.cat([local_peptide, evidence_h], dim=-1)) * residue_mask.unsqueeze(-1)
        local_logits = self.local_head(local_h).squeeze(-1) * residue_mask
        hapr = self.context(local_h, residue_evidence, residue_mask)
        logits = self.apr_head(hapr).squeeze(-1) * residue_mask
        reliability = torch.sigmoid(self.reliability_head(torch.cat([local_h, hapr], dim=-1)).squeeze(-1)) * residue_mask
        window_hidden_prediction = self.window_hidden_decoder(hapr)
        if return_encoder:
            return {'residue_embedding': hapr, 'hapr_contextual': hapr, 'hapr_peptide': local_h, 'local_embedding': local_h, 'context_embedding': hapr - local_h, 'apr_logits': logits, 'apr_probability': torch.sigmoid(logits) * residue_mask, 'local_logits': local_logits, 'local_probability': torch.sigmoid(local_logits) * residue_mask, 'reliability': reliability, 'window_hidden_prediction': window_hidden_prediction, 'apr_statistics': torch.stack([torch.sigmoid(logits).mul(residue_mask).sum(dim=1) / residue_mask.float().sum(dim=1).clamp_min(1.0), torch.sigmoid(logits).mul(residue_mask).max(dim=1).values, reliability.sum(dim=1) / residue_mask.float().sum(dim=1).clamp_min(1.0), residue_mask.float().mean(dim=1)], dim=-1)}
        return (logits, local_logits) if return_aux else logits

def peptide_summary(evidence: Tensor) -> Tensor:
    soft_or = evidence[..., 0::4]
    maximum = evidence[..., 2::4]
    maximum_local = evidence[..., 3::4]
    return torch.stack([soft_or.max(dim=-1).values, maximum.max(dim=-1).values, maximum_local.max(dim=-1).values, (maximum >= 0.5).float().mean(dim=-1), (maximum_local >= 0.5).float().mean(dim=-1)], dim=-1)

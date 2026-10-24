"""Boundary-aware Context V3 built on the six-scale concat/hidden-skip path.

The boundary signal is part of the encoder trunk: adjacent local/evidence
differences produce an edge feature, the edge prediction gates context
attention, and a residual branch preserves information at predicted edges.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor, nn

import encoder.utils.support as baseline
import encoder.model.scale_fusion as concat
from encoder.model.context import ContextEncoderConfig

VARIANT = "C2_CS_E2R_BOUNDARY_AWARE_CONTEXT_V3"
CHECKPOINT_FORMAT = "hapr_boundary_aware_context_v3"
WINDOW_SCALES = baseline.WINDOW_SCALES
require = baseline.require


class BoundaryRelativeLocalAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, radius: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.radius = radius
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.relative_bias = nn.Embedding(2 * radius + 1, heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.crossing_penalty = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: Tensor, mask: Tensor, edge_strength: Tensor) -> Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        positions = torch.arange(length, device=x.device)
        relative = positions[None, :] - positions[:, None]
        bias_ids = relative.clamp(-self.radius, self.radius) + self.radius
        bias = self.relative_bias(bias_ids).permute(2, 0, 1).to(q.dtype)
        scores = (q @ k.transpose(-1, -2)) * ((hidden // self.heads) ** -0.5)
        scores = scores + bias.unsqueeze(0)

        # Mean edge strength between query and key is a differentiable soft
        # barrier. It limits cross-boundary mixing without hard segmentation.
        edge = edge_strength.clamp(0.0, 1.0)
        prefix = torch.cat([edge.new_zeros(batch, 1), edge], dim=1).cumsum(dim=1)
        lo = torch.minimum(positions[:, None], positions[None, :])
        hi = torch.maximum(positions[:, None], positions[None, :])
        distance = (hi - lo).to(edge.dtype)
        crossing = (prefix[:, hi] - prefix[:, lo]) / distance.clamp_min(1.0)
        scores = scores - self.crossing_penalty.clamp_min(0.0) * crossing[:, None]

        allowed = ((relative.abs() <= self.radius)[None, None, :, :]
                   & mask[:, None, None, :] & mask[:, None, :, None])
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = scores.float().softmax(dim=-1).masked_fill(~allowed, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = self.attention_dropout(weights.to(v.dtype)) @ v
        values = values.transpose(1, 2).reshape(batch, length, hidden)
        return self.output(values).masked_fill(~mask.unsqueeze(-1), 0.0)


class BoundaryContextBlock(nn.Module):
    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = BoundaryRelativeLocalAttention(
            hidden, config.attention_heads, config.context_radius, config.dropout)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * config.context_ffn_multiplier),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(hidden * config.context_ffn_multiplier, hidden),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor, edge_strength: Tensor) -> Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x), mask, edge_strength))
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class BoundaryAwareContextEncoder(nn.Module):
    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.config = config
        self.evidence_projection = nn.Sequential(
            nn.Linear(baseline.EVIDENCE_FEATURES, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([BoundaryContextBlock(config) for _ in range(config.context_layers)])
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, local_h: Tensor, evidence: Tensor, mask: Tensor,
                edge_strength: Tensor, boundary_residual: Tensor) -> Tensor:
        invalid = ~mask.unsqueeze(-1)
        evidence_h = self.evidence_projection(evidence)
        x = self.input_norm(local_h.masked_fill(invalid, 0.0) + evidence_h.masked_fill(invalid, 0.0))
        x = x.masked_fill(invalid, 0.0)
        for block in self.blocks:
            x = block(x, mask, edge_strength)
        return self.output_norm(x + boundary_residual).masked_fill(invalid, 0.0)


class ContextualAPRModel(concat.ContextualAPRModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.model_variant = VARIANT
        hidden = self.model_config.hidden_size
        # Boundary candidates use both representation and residue-evidence jumps.
        self.boundary_candidate = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden // 2),
            nn.GELU(), nn.Dropout(self.model_config.dropout), nn.Linear(hidden // 2, 1))
        self.boundary_candidate[-1].bias.data.fill_(-2.0)
        self.boundary_residual = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden), nn.GELU(),
            nn.Dropout(self.model_config.dropout), nn.Linear(hidden, hidden))
        self.boundary_residual_scale = nn.Parameter(torch.tensor(0.25))
        self.context = BoundaryAwareContextEncoder(self.model_config)

    def _boundary_features(self, local_h: Tensor, evidence_h: Tensor, mask: Tensor):
        local_delta = (local_h[:, 1:] - local_h[:, :-1]).abs()
        evidence_delta = (evidence_h[:, 1:] - evidence_h[:, :-1]).abs()
        pair = torch.cat([local_h[:, :-1], local_h[:, 1:], local_delta, evidence_delta], dim=-1)
        logits = self.boundary_candidate(pair).squeeze(-1)
        strength = torch.sigmoid(logits)
        pair_mask = mask[:, :-1] & mask[:, 1:]
        strength = strength * pair_mask
        # Map each edge score to both incident residues without in-place
        # slice updates; the latter would invalidate autograd's saved views.
        left_strength = torch.cat([strength, local_h.new_zeros(local_h.shape[0], 1)], dim=1)
        right_strength = torch.cat([local_h.new_zeros(local_h.shape[0], 1), strength], dim=1)
        residue_strength = torch.maximum(left_strength, right_strength)
        right_local = torch.cat([local_delta, local_h.new_zeros(local_h.shape[0], 1, local_h.shape[-1])], dim=1)
        right_evidence = torch.cat([evidence_delta, evidence_h.new_zeros(evidence_h.shape[0], 1, evidence_h.shape[-1])], dim=1)
        left_local = torch.cat([local_h.new_zeros(local_h.shape[0], 1, local_h.shape[-1]), local_delta], dim=1)
        left_evidence = torch.cat([evidence_h.new_zeros(evidence_h.shape[0], 1, evidence_h.shape[-1]), evidence_delta], dim=1)
        residue_feature = torch.cat([local_h, evidence_h, torch.maximum(left_local, right_local),
                                     torch.maximum(left_evidence, right_evidence)], dim=-1)
        residual = self.boundary_residual(residue_feature) * residue_strength.unsqueeze(-1)
        return logits * pair_mask, strength, residual

    def forward(self, base, tokens, starts, lengths, token_mask, residue_mask,
                token_probability=None, token_local_probability=None, token_scale=None,
                residue_evidence=None, profile_hidden=None, profile_local_probability=None,
                return_aux=False, return_encoder=False):
        # Reuse the exact six-scale concat/hidden-skip front end from V2.
        del base, token_local_probability, profile_hidden, profile_local_probability
        require(residue_evidence is not None and token_scale is not None, "V3 inputs are incomplete")
        scale_ids = token_scale.round().long().clamp(0, len(WINDOW_SCALES) - 1)
        scale_h = self.window_scale_embedding[scale_ids]
        length_h = self.window_length_projection((lengths / float(max(WINDOW_SCALES))).unsqueeze(-1))
        peptide_tokens = (self.peptide_projection(tokens) + scale_h + length_h) * token_mask.unsqueeze(-1)
        query = self.peptide_query.expand(tokens.shape[0], residue_mask.shape[1], -1)
        evidence_h = self.evidence_projection(residue_evidence)
        scale_views, scale_presence = [], []
        for scale_index, scale_attention in enumerate(self.scale_window_attention):
            scale_mask = token_mask & (scale_ids == scale_index)
            scale_presence.append(scale_mask.any(dim=1))
            max_windows = int(scale_mask.sum(dim=1).max().item())
            if max_windows == 0:
                scale_views.append(torch.zeros_like(query)); continue
            compact_tokens = peptide_tokens.new_zeros((tokens.shape[0], max_windows, peptide_tokens.shape[-1]))
            compact_starts = starts.new_zeros((starts.shape[0], max_windows))
            compact_lengths = lengths.new_ones((lengths.shape[0], max_windows))
            compact_mask = torch.zeros((tokens.shape[0], max_windows), dtype=torch.bool, device=tokens.device)
            for row in range(scale_mask.shape[0]):
                selected = torch.nonzero(scale_mask[row], as_tuple=False).flatten()
                count = min(int(selected.numel()), max_windows)
                if count:
                    compact_tokens[row, :count] = peptide_tokens[row, selected[:count]]
                    compact_starts[row, :count] = starts[row, selected[:count]]
                    compact_lengths[row, :count] = lengths[row, selected[:count]]
                    compact_mask[row, :count] = True
            scale_views.append(scale_attention(query, compact_tokens, compact_starts,
                                               compact_lengths, compact_mask, residue_mask))
        scale_stack = torch.stack(scale_views, dim=2)
        present = torch.stack(scale_presence, dim=1)
        local_peptide = self.fuse_scales(scale_stack, present) * residue_mask.unsqueeze(-1)
        local_h = self.local_encoder(torch.cat([local_peptide, evidence_h], dim=-1)) * residue_mask.unsqueeze(-1)
        local_logits = self.local_head(local_h).squeeze(-1) * residue_mask
        boundary_logits, boundary_strength, boundary_residual = self._boundary_features(local_h, evidence_h, residue_mask)
        gated_local = local_h + boundary_residual
        hapr = self.context(gated_local, residue_evidence, residue_mask, boundary_strength, boundary_residual)
        logits = self.apr_head(hapr).squeeze(-1) * residue_mask
        reliability = torch.sigmoid(self.reliability_head(torch.cat([local_h, hapr], dim=-1)).squeeze(-1)) * residue_mask
        window_hidden_prediction = self.window_hidden_decoder(hapr)
        if return_encoder:
            return {"residue_embedding": hapr, "hapr_contextual": hapr, "hapr_peptide": local_h,
                    "local_embedding": local_h, "context_embedding": hapr - local_h,
                    "apr_logits": logits, "apr_probability": torch.sigmoid(logits) * residue_mask,
                    "local_logits": local_logits, "local_probability": torch.sigmoid(local_logits) * residue_mask,
                    "reliability": reliability, "window_hidden_prediction": window_hidden_prediction,
                    "boundary_logits": boundary_logits,
                    "boundary_strength": boundary_strength,
                    "boundary_residual": boundary_residual,
                    "apr_statistics": torch.stack([torch.sigmoid(logits).mul(residue_mask).sum(dim=1) /
                                                    residue_mask.float().sum(dim=1).clamp_min(1.0),
                                                    torch.sigmoid(logits).max(dim=1).values,
                                                    reliability.sum(dim=1) / residue_mask.float().sum(dim=1).clamp_min(1.0),
                                                    residue_mask.float().mean(dim=1)], dim=-1)}
        return (logits, local_logits) if return_aux else logits


def model_metadata(model):
    info = concat.model_metadata(model)
    info.update(format=CHECKPOINT_FORMAT, variant=VARIANT,
                architecture="six-scale concat/hidden-skip -> boundary feature/gate -> boundary-aware context -> semantic HAPR + boundary residual",
                boundary_main_trunk=True,
                boundary_feature="abs adjacent local_h and evidence_h differences",
                boundary_gate="soft attention penalty from mean edge strength between query/key",
                boundary_residual="residue-local residual injected into context output",
                transfer_representation="boundary-aware HAPR before optional readouts")
    return info


def load_encoder_checkpoint(path: str | Path, device="cpu", *, frozen=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("format") == CHECKPOINT_FORMAT and payload.get("variant") == VARIANT,
            "Expected a Boundary-Aware Context V3 checkpoint")
    model = ContextualAPRModel(ContextEncoderConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(not frozen)
    return model, payload

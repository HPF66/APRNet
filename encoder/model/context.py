"""E2R local encoder followed by a full contextual HAPR encoder.

The frozen teacher and E2R local path are inherited without modification.
Only the context module is replaced. Existing C2 checkpoints are deliberately
not accepted by the versioned checkpoint loader below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

import encoder.utils.support as baseline


VARIANT = "C2_CS_E2R_CONTEXT_V2"
CHECKPOINT_FORMAT = "hapr_context_encoder_v2"


@dataclass(frozen=True)
class ContextEncoderConfig:
    hidden_size: int = 192
    dropout: float = 0.15
    attention_heads: int = 4
    context_radius: int = 64
    context_layers: int = 2
    context_ffn_multiplier: int = 4
    boundary_head: bool = False
    supcon_projection_dim: int = 0

    def __post_init__(self) -> None:
        baseline.require(self.hidden_size >= 4, "hidden_size must be at least 4")
        baseline.require(self.attention_heads > 0 and self.hidden_size % self.attention_heads == 0,
                         "hidden_size must be divisible by positive attention_heads")
        baseline.require(0 <= self.dropout < 1, "dropout must be in [0, 1)")
        baseline.require(self.context_radius >= 0, "context_radius must be nonnegative")
        baseline.require(self.context_layers >= 1, "context_layers must be positive")
        baseline.require(self.context_ffn_multiplier >= 1, "context_ffn_multiplier must be positive")
        baseline.require(self.supcon_projection_dim == 0 or self.supcon_projection_dim >= 2,
                         "supcon_projection_dim must be zero (disabled) or at least 2")


class RelativeLocalAttention(nn.Module):
    """Local self-attention with a learned signed-distance bias per head.

    Finite masked softmax also supports fully padded query rows, including
    holes in the mask, without NaNs in either the forward or backward pass.
    Attention is dense within a chunk; the radius restricts visibility, not
    allocation size. With the default 512-residue chunks this stays bounded.
    """

    def __init__(self, hidden: int, heads: int, radius: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.radius = radius
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.output = nn.Linear(hidden, hidden)
        self.relative_bias = nn.Embedding(2 * radius + 1, heads)
        nn.init.normal_(self.relative_bias.weight, std=0.02)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        positions = torch.arange(length, device=x.device)
        relative = positions[None, :] - positions[:, None]  # key - query
        bias_ids = relative.clamp(-self.radius, self.radius) + self.radius
        bias = self.relative_bias(bias_ids).permute(2, 0, 1).to(q.dtype)
        scores = (q @ k.transpose(-1, -2)) * ((hidden // self.heads) ** -0.5)
        scores = scores + bias.unsqueeze(0)
        allowed = ((relative.abs() <= self.radius)[None, None, :, :]
                   & mask[:, None, None, :] & mask[:, None, :, None])
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = scores.float().softmax(dim=-1).masked_fill(~allowed, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        values = self.attention_dropout(weights.to(v.dtype)) @ v
        values = values.transpose(1, 2).reshape(batch, length, hidden)
        return self.output(values).masked_fill(~mask.unsqueeze(-1), 0.0)


class ContextEncoderBlock(nn.Module):
    """Pre-norm attention + FFN, each with an ordinary residual connection."""

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = RelativeLocalAttention(
            hidden, config.attention_heads, config.context_radius, config.dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * config.context_ffn_multiplier),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(hidden * config.context_ffn_multiplier, hidden),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x), mask))
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x.masked_fill(~mask.unsqueeze(-1), 0.0)


class HAPRContextEncoder(nn.Module):
    """Local features + all 24 evidence values -> context blocks -> HAPR.

    There is no final small/gated correction added to local_h. Local detail is
    retained through the ordinary residual connections inside the main trunk.
    """

    def __init__(self, config: ContextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.evidence_projection = nn.Sequential(
            nn.Linear(baseline.EVIDENCE_FEATURES, config.hidden_size),
            nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([ContextEncoderBlock(config) for _ in range(config.context_layers)])
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, local_h: Tensor, evidence: Tensor, mask: Tensor) -> Tensor:
        baseline.require(local_h.ndim == 3 and local_h.shape[-1] == self.config.hidden_size,
                         "local_h must have shape [batch, length, hidden_size]")
        baseline.require(local_h.shape[1] > 0, "context requires a nonempty sequence axis")
        baseline.require(mask.dtype == torch.bool and mask.shape == local_h.shape[:2],
                         "mask must be boolean [batch, length]")
        baseline.require(evidence.shape == (*local_h.shape[:2], baseline.EVIDENCE_FEATURES),
                         "evidence must have shape [batch, length, 24]")
        invalid = ~mask.unsqueeze(-1)
        local_h = local_h.masked_fill(invalid, 0.0)
        evidence = evidence.masked_fill(invalid, 0.0)
        x = self.input_norm(local_h + self.evidence_projection(evidence)).masked_fill(invalid, 0.0)
        for block in self.blocks:
            x = block(x, mask)
        return self.output_norm(x).masked_fill(invalid, 0.0)


class ContextualAPRModel(baseline.IntegratedAPRModel):
    """Drop-in forward interface for APR training and frozen Amyloid transfer.

    `variant` remains C2_CS_E2R internally to select the inherited local path;
    `model_variant` and checkpoint metadata identify the new architecture.
    `context_embedding` in the inherited output remains HAPR - local_h for
    compatibility. It is a diagnostic delta, not an isolated neighbor vector.
    Use `residue_embedding` / `hapr_contextual` for the actual contextual HAPR.
    """

    def __init__(self, config: ContextEncoderConfig | None = None) -> None:
        config = config or ContextEncoderConfig()
        super().__init__(config.hidden_size, config.dropout, config.attention_heads,
                         config.context_radius, "C2_CS_E2R", 0.10)
        self.model_config = config
        self.model_variant = VARIANT
        self.context = HAPRContextEncoder(config)
        # These optional readouts are training-only. HAPR itself retains its
        # original width and remains the representation exported to Amyloid.
        if config.boundary_head:
            self.boundary_head = nn.Sequential(
                nn.LayerNorm(config.hidden_size * 3),
                nn.Linear(config.hidden_size * 3, config.hidden_size // 2),
                nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_size // 2, 1),
            )
        if config.supcon_projection_dim:
            self.supcon_projection = nn.Sequential(
                nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(), nn.Linear(config.hidden_size, config.supcon_projection_dim),
            )

    def representation_readouts(self, hapr: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """Explicit training readouts; ordinary encoder forward skips these.

        Boundary logits live on adjacent pairs (L-1), not residues (L).
        Calling this separately avoids perturbing the E2R view/dropout RNG when
        these objectives are disabled or during frozen downstream inference.
        """
        result = {}
        if self.model_config.boundary_head:
            left, right = hapr[:, :-1], hapr[:, 1:]
            pair_mask = mask[:, :-1] & mask[:, 1:]
            pair = torch.cat([left, right, (right - left).abs()], dim=-1)
            result["boundary_logits"] = self.boundary_head(pair).squeeze(-1).masked_fill(~pair_mask, 0.)
        if self.model_config.supcon_projection_dim:
            result["contrastive_embedding"] = self.supcon_projection(hapr).masked_fill(~mask.unsqueeze(-1), 0.)
        return result


def load_encoder_checkpoint(path: str | Path, device: str | torch.device = "cpu",
                            *, frozen: bool = True) -> tuple[ContextualAPRModel, dict]:
    """Load a trusted V2 checkpoint for inference/probes/downstream adapters."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    baseline.require(payload.get("format") == CHECKPOINT_FORMAT and payload.get("variant") == VARIANT,
                     "Expected a C2_CS_E2R_CONTEXT_V2 checkpoint; legacy weights are incompatible")
    config = ContextEncoderConfig(**payload["model_config"])
    model = ContextualAPRModel(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(not frozen)
    return model, payload


def model_metadata(model: ContextualAPRModel) -> dict:
    return {
        "format": CHECKPOINT_FORMAT, "variant": VARIANT,
        "model_config": asdict(model.model_config),
        "architecture": "E2R local features -> full evidence projection -> pre-norm relative local attention/FFN blocks -> normalized HAPR",
        "context_role": "main representation stage; ordinary residuals, no bounded context gate",
        "missing_scale_policy": "cross-scale attention masks window scales unavailable at the sequence length",
        "context_embedding_definition": "HAPR - local_h; diagnostic delta only",
        "hapr_width": model.model_config.hidden_size,
        "boundary_head": "adjacent HAPR pair -> annotation transition logit" if model.model_config.boundary_head else "disabled",
        "supcon_projection_dim": model.model_config.supcon_projection_dim,
        "transfer_representation": "HAPR before optional boundary/contrastive readouts",
    }

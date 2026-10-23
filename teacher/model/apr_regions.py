"""Multi-scale APR region head and masked region losses.

The region head is deliberately residual: its residue correction is zero at
initialisation, so an experiment can be compared to the existing token APR
head under identical settings.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleAPRRegionHead(nn.Module):
    """Predict overlapping APR candidates and return a residue-level residual."""

    def __init__(
        self,
        hidden_size: int,
        lengths: Sequence[int] = (6, 8, 10, 12, 16, 20),
        dropout: float = 0.1,
        max_residue_delta: float = 0.5,
        propensity_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        normalized_lengths = tuple(sorted({int(length) for length in lengths}))
        if not normalized_lengths or any(length < 2 for length in normalized_lengths):
            raise ValueError("APR region lengths must contain integers >= 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("APR region dropout must be in [0, 1)")
        if max_residue_delta <= 0:
            raise ValueError("APR region max residue delta must be positive")
        self.lengths = normalized_lengths
        self.hidden_size = int(hidden_size)
        self.max_residue_delta = float(max_residue_delta)
        self.attention_score = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(hidden_size // 4, 8)),
            nn.GELU(),
            nn.Linear(max(hidden_size // 4, 8), 1),
        )
        self.length_embedding = nn.Embedding(len(normalized_lengths), hidden_size)
        self.region_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(hidden_size // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 16), 1),
        )
        self.propensity_projection = nn.Linear(1, hidden_size, bias=False)
        self.propensity_gate_logit = nn.Parameter(torch.tensor(float(propensity_gate_init)))
        residue_input_size = hidden_size * 2 + 1
        self.residue_delta_head = nn.Sequential(
            nn.LayerNorm(residue_input_size),
            nn.Linear(residue_input_size, max(hidden_size // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 16), 1),
        )
        # Start exactly at the existing residue head. Region learning can only
        # change the baseline after its own gradients provide evidence.
        nn.init.zeros_(self.residue_delta_head[-1].weight)
        nn.init.zeros_(self.residue_delta_head[-1].bias)

    @staticmethod
    def _range_pool(values: torch.Tensor, weights: torch.Tensor, length: int) -> torch.Tensor:
        channels = values.transpose(1, 2)
        pooled_values = F.avg_pool1d(
            channels * weights.unsqueeze(-1).transpose(1, 2),
            kernel_size=length,
            stride=1,
        ) * length
        pooled_weights = F.avg_pool1d(
            weights.unsqueeze(1), kernel_size=length, stride=1
        ) * length
        return (pooled_values / pooled_weights.clamp_min(1e-6)).transpose(1, 2)

    @staticmethod
    def _scatter_window_sum(
        candidate_values: torch.Tensor,
        candidate_weights: torch.Tensor,
        length: int,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A grouped transposed convolution sums every candidate back over its
        # covered residues without constructing a [B, L, L] incidence matrix.
        weighted = (candidate_values * candidate_weights.unsqueeze(-1)).transpose(1, 2)
        value_kernel = torch.ones(
            candidate_values.shape[-1], 1, length,
            dtype=candidate_values.dtype,
            device=candidate_values.device,
        )
        weight_kernel = torch.ones(
            1, 1, length,
            dtype=candidate_values.dtype,
            device=candidate_values.device,
        )
        value_sum = F.conv_transpose1d(
            weighted, value_kernel, groups=candidate_values.shape[-1]
        )
        weight_sum = F.conv_transpose1d(
            candidate_weights.unsqueeze(1), weight_kernel
        )
        if value_sum.shape[-1] != output_length:
            value_sum = value_sum[..., :output_length]
            weight_sum = weight_sum[..., :output_length]
        return value_sum.transpose(1, 2), weight_sum.squeeze(1)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        propensity_logits: torch.Tensor | None = None,
    ) -> dict[str, object]:
        if hidden.ndim != 3 or mask.ndim != 2:
            raise ValueError("hidden must be [batch, length, hidden] and mask [batch, length]")
        if hidden.shape[:2] != mask.shape:
            raise ValueError("hidden and mask shapes do not match")
        batch_size, sequence_length, _ = hidden.shape
        residue_mask = mask.to(dtype=hidden.dtype)
        if propensity_logits is None:
            propensity = torch.zeros(
                batch_size, sequence_length, 1, dtype=hidden.dtype, device=hidden.device
            )
        else:
            if propensity_logits.shape != mask.shape:
                raise ValueError("propensity logits must match the residue mask")
            propensity = torch.sigmoid(propensity_logits.detach()).unsqueeze(-1)
        source = hidden + torch.sigmoid(self.propensity_gate_logit) * self.propensity_projection(propensity)
        attention_weights = torch.sigmoid(self.attention_score(source).squeeze(-1)) * residue_mask

        region_logits: list[torch.Tensor] = []
        region_valid: list[torch.Tensor] = []
        region_lengths: list[int] = []
        residue_numerator = torch.zeros_like(hidden)
        residue_denominator = torch.zeros(
            batch_size, sequence_length, dtype=hidden.dtype, device=hidden.device
        )
        for scale_index, length in enumerate(self.lengths):
            if sequence_length < length:
                region_logits.append(hidden.new_empty(batch_size, 0))
                region_valid.append(torch.zeros(batch_size, 0, dtype=torch.bool, device=hidden.device))
                continue
            pooled = self._range_pool(source, attention_weights, length)
            pooled_mask = F.avg_pool1d(
                residue_mask.unsqueeze(1), kernel_size=length, stride=1
            ).squeeze(1) >= (1.0 - 1e-6)
            pooled = pooled + self.length_embedding.weight[scale_index].view(1, 1, -1)
            logits = self.region_head(pooled).squeeze(-1)
            logits = logits.masked_fill(~pooled_mask, 0.0)
            probabilities = torch.sigmoid(logits) * pooled_mask.to(hidden.dtype)
            value_sum, weight_sum = self._scatter_window_sum(
                pooled, probabilities, length, sequence_length
            )
            residue_numerator = residue_numerator + value_sum
            residue_denominator = residue_denominator + weight_sum
            region_logits.append(logits)
            region_valid.append(pooled_mask)
            region_lengths.append(length)

        region_context = residue_numerator / residue_denominator.unsqueeze(-1).clamp_min(1e-6)
        region_context = region_context * residue_mask.unsqueeze(-1)
        residue_input = torch.cat([hidden, region_context, propensity], dim=-1)
        residue_delta = self.max_residue_delta * torch.tanh(
            self.residue_delta_head(residue_input).squeeze(-1)
        )
        residue_delta = residue_delta * residue_mask
        return {
            "region_logits": region_logits,
            "region_valid": region_valid,
            "region_lengths": tuple(self.lengths),
            "region_context": region_context,
            "residue_delta": residue_delta,
            "propensity": propensity.squeeze(-1),
        }


def _evenly_spaced_indices(indices: torch.Tensor, limit: int) -> torch.Tensor:
    if indices.numel() <= limit:
        return indices
    positions = torch.linspace(
        0, indices.numel() - 1, steps=limit, device=indices.device
    ).round().long()
    return indices[positions]


def apr_region_losses(
    region_logits: Sequence[torch.Tensor],
    region_valid: Sequence[torch.Tensor],
    region_lengths: Sequence[int],
    residue_logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
    training_only: torch.Tensor | None = None,
    positive_fraction: float = 0.8,
    max_candidates_per_class: int = 64,
    rank_margin: float = 0.2,
    rank_temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute balanced region BCE, residue consistency, and within-protein ranking."""
    if not 0.5 <= positive_fraction <= 1.0:
        raise ValueError("positive_fraction must be in [0.5, 1]")
    if max_candidates_per_class < 1:
        raise ValueError("max_candidates_per_class must be positive")
    if rank_margin < 0:
        raise ValueError("rank_margin must be non-negative")
    if rank_temperature <= 0:
        raise ValueError("rank_temperature must be positive")
    if training_only is None:
        training_only = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
    trusted_negative = (targets == 0) & (label_weights > 0) & ~training_only.unsqueeze(1)
    residue_prob = torch.sigmoid(residue_logits)
    region_terms: list[torch.Tensor] = []
    consistency_terms: list[torch.Tensor] = []
    rank_terms: list[torch.Tensor] = []
    for logits, valid, length in zip(region_logits, region_valid, region_lengths):
        if logits.shape[1] == 0:
            continue
        target_windows = targets.unfold(1, length, 1)
        weight_windows = label_weights.unfold(1, length, 1)
        negative_windows = trusted_negative.unfold(1, length, 1).all(dim=-1)
        positive_count = ((target_windows == 1) & (weight_windows > 0)).sum(dim=-1)
        center_positive = (
            (target_windows[..., (length - 1) // 2] == 1)
            & (target_windows[..., length // 2] == 1)
        )
        positive_windows = (
            valid
            & center_positive
            & (positive_count.to(logits.dtype) / float(length) >= positive_fraction)
        )
        residue_windows = residue_prob.unfold(1, length, 1).mean(dim=-1)
        for row in range(logits.shape[0]):
            positive = torch.nonzero(positive_windows[row], as_tuple=False).flatten()
            negative = torch.nonzero(negative_windows[row], as_tuple=False).flatten()
            if positive.numel() == 0 and negative.numel() == 0:
                continue
            positive = _evenly_spaced_indices(positive, max_candidates_per_class)
            if negative.numel() > max_candidates_per_class:
                negative = negative[torch.topk(
                    logits[row, negative].detach(), max_candidates_per_class
                ).indices]
            selected = torch.cat([positive, negative])
            labels = torch.cat([
                torch.ones(positive.numel(), device=logits.device),
                torch.zeros(negative.numel(), device=logits.device),
            ])
            selected_logits = logits[row, selected]
            selected_residue = residue_windows[row, selected]
            region_bce = F.binary_cross_entropy_with_logits(
                selected_logits, labels, reduction="none"
            )
            if positive.numel() and negative.numel():
                region_terms.append(
                    0.5 * region_bce[: positive.numel()].mean()
                    + 0.5 * region_bce[positive.numel() :].mean()
                )
            else:
                region_terms.append(region_bce.mean())
            consistency_terms.append(
                (torch.sigmoid(selected_logits) - selected_residue).pow(2).mean()
            )
            if positive.numel() and negative.numel():
                differences = (
                    logits[row, positive].unsqueeze(1)
                    - logits[row, negative].unsqueeze(0)
                    - rank_margin
                ) / rank_temperature
                rank_terms.append(F.softplus(-differences).mean())

    zero = residue_logits.sum() * 0.0
    return {
        "region": torch.stack(region_terms).mean() if region_terms else zero,
        "consistency": torch.stack(consistency_terms).mean() if consistency_terms else zero,
        "rank": torch.stack(rank_terms).mean() if rank_terms else zero,
    }

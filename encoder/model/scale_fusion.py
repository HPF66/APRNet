"""Only replace Context V2 cross-scale fusion; all other computations are inherited.

Per residue: LN(Linear(concat[m_s h_s]) + sum_s(m_s h_s) / sum_s(m_s)).
Six within-scale attentions remain independent and unchanged. The skip is
residue-dependent, not the shared learned query used by the old scale fusion.
Unavailable scales on short sequences are excluded from both fusion paths.
"""
from pathlib import Path
import torch
from torch import Tensor, nn
import encoder.utils.support as baseline
import encoder.model.context as contextual
from encoder.model.context import ContextEncoderConfig

VARIANT = "C2_CS_E2R_CONTEXT_V2_CONCAT_SKIP"
CHECKPOINT_FORMAT = "hapr_context_encoder_v2_concat_skip_v1"
WINDOW_SCALES = baseline.WINDOW_SCALES
require = baseline.require


class ContextualAPRModel(contextual.ContextualAPRModel):
    def __init__(self, config=None):
        # Complete the original initialization first: every shared parameter
        # has identical initialization to Context V2 with the same fold seed.
        super().__init__(config)
        self.model_variant = VARIANT
        self.scale_concat_projection = nn.Linear(len(WINDOW_SCALES) * self.model_config.hidden_size,
                                                 self.model_config.hidden_size)
        # Removed layers were already initialized; removing them does not shift
        # initialization of shared weights. e2r_norm is retained unchanged.
        for name in ('e2r_query', 'e2r_key', 'e2r_value', 'e2r_attention'):
            delattr(self, name)

    def fuse_scales(self, scale_stack, scale_presence=None):
        require(scale_stack.shape[-2] == len(WINDOW_SCALES), "Expected six ordered scales")
        if scale_presence is None:
            scale_presence = torch.ones(
                (scale_stack.shape[0], len(WINDOW_SCALES)),
                dtype=torch.bool, device=scale_stack.device,
            )
        require(scale_presence.dtype == torch.bool,
                "scale_presence must be a boolean [batch, scales] mask")
        require(scale_presence.shape == (scale_stack.shape[0], len(WINDOW_SCALES)),
                "scale_presence must match batch and scale dimensions")
        require(bool(scale_presence.any(dim=1).all()),
                "Every sequence needs at least one peptide scale")
        available = scale_presence[:, None, :, None]
        masked_stack = scale_stack.masked_fill(~available, 0.0)
        projected = self.scale_concat_projection(masked_stack.flatten(-2))
        denominator = scale_presence.sum(dim=1).clamp_min(1).to(scale_stack.dtype)[:, None, None]
        hidden_skip = masked_stack.sum(dim=-2) / denominator
        return self.e2r_norm(projected + hidden_skip)

    def forward(
        self,
        base: Tensor,
        tokens: Tensor,
        starts: Tensor,
        lengths: Tensor,
        token_mask: Tensor,
        residue_mask: Tensor,
        token_probability: Tensor | None = None,
        token_local_probability: Tensor | None = None,
        token_scale: Tensor | None = None,
        residue_evidence: Tensor | None = None,
        profile_hidden: Tensor | None = None,
        profile_local_probability: Tensor | None = None,
        return_aux: bool = False,
        return_encoder: bool = False,
    ):
        del base, token_local_probability, profile_hidden, profile_local_probability
        require(residue_evidence is not None, "residue_evidence is required")
        require(token_scale is not None, "token_scale is required for C2_ROBUST")
        scale_ids = token_scale.round().long().clamp(0, len(WINDOW_SCALES) - 1)
        scale_h = self.window_scale_embedding[scale_ids]
        length_h = self.window_length_projection((lengths / float(max(WINDOW_SCALES))).unsqueeze(-1))
        peptide_tokens = self.peptide_projection(tokens) + scale_h + length_h
        peptide_tokens = peptide_tokens * token_mask.unsqueeze(-1)
        query = self.peptide_query.expand(tokens.shape[0], residue_mask.shape[1], -1)
        evidence_h = self.evidence_projection(residue_evidence)
        if self.variant == "C2_CS_E2R":
            # First perform attention within each scale, retaining six hidden
            # views per residue.  Only then fuse scales with a residue-specific
            # query derived from the peptide evidence.
            scale_views = []
            scale_presence = []
            for scale_index, scale_attention in enumerate(self.scale_window_attention):
                scale_mask = token_mask & (scale_ids == scale_index)
                scale_presence.append(scale_mask.any(dim=1))
                # Do not pass the full six-scale window table to every branch:
                # WindowAttention materializes [B, heads, residues, windows].
                # Compact the selected scale first, reducing peak memory by
                # roughly the number of scales on long proteins.
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
                scale_views.append(
                    scale_attention(query, compact_tokens, compact_starts, compact_lengths, compact_mask, residue_mask)
                )
            scale_stack = torch.stack(scale_views, dim=2)  # [B, T, S, H]
            present = torch.stack(scale_presence, dim=1)  # [B, S]
            local_peptide = self.fuse_scales(scale_stack, present)
            local_peptide = local_peptide * residue_mask.unsqueeze(-1)
        else:
            local_peptide = self.window_attention(query, peptide_tokens, starts, lengths, token_mask, residue_mask)
        local_h = self.local_encoder(torch.cat([local_peptide, evidence_h], dim=-1)) * residue_mask.unsqueeze(-1)
        local_logits = self.local_head(local_h).squeeze(-1) * residue_mask
        hapr = self.context(local_h, residue_evidence, residue_mask)
        logits = self.apr_head(hapr).squeeze(-1) * residue_mask
        reliability = torch.sigmoid(self.reliability_head(torch.cat([local_h, hapr], dim=-1)).squeeze(-1)) * residue_mask
        # Each overlapping peptide window receives a pooled HAPR representation.
        # The loss is computed outside the model so the same code can mask padded
        # windows and weight the frozen teacher targets explicitly.
        window_hidden_prediction = self.window_hidden_decoder(hapr)
        if return_encoder:
            return {
                "residue_embedding": hapr,
                "hapr_contextual": hapr,
                "hapr_peptide": local_h,
                "local_embedding": local_h,
                "context_embedding": hapr - local_h,
                "apr_logits": logits,
                "apr_probability": torch.sigmoid(logits) * residue_mask,
                "local_logits": local_logits,
                "local_probability": torch.sigmoid(local_logits) * residue_mask,
                "reliability": reliability,
                "window_hidden_prediction": window_hidden_prediction,
                "apr_statistics": torch.stack(
                    [
                        torch.sigmoid(logits).mul(residue_mask).sum(dim=1)
                        / residue_mask.float().sum(dim=1).clamp_min(1.0),
                        torch.sigmoid(logits).mul(residue_mask).max(dim=1).values,
                        reliability.sum(dim=1) / residue_mask.float().sum(dim=1).clamp_min(1.0),
                        residue_mask.float().mean(dim=1),
                    ],
                    dim=-1,
                ),
            }
        return (logits, local_logits) if return_aux else logits


def model_metadata(model):
    info = contextual.model_metadata(model)
    info.update(format=CHECKPOINT_FORMAT, variant=VARIANT,
                architecture="E2R six independent scale views -> masked concat projection plus valid-scale hidden skip -> unchanged Context V2 -> HAPR",
                fusion={"kind": "masked_concat_linear_plus_valid_scale_mean_skip", "scale_order": list(WINDOW_SCALES),
                        "formula": "LayerNorm(Linear(concat(m_s*h_s)) + sum(m_s*h_s)/sum(m_s))",
                        "missing_scale_policy": "exclude unavailable scales from projection input and hidden-skip denominator",
                        "within_scale_parameters": "independent, same as reference",
                        "shared_parameter_initialization": "identical at same fold seed"})
    return info


def load_encoder_checkpoint(path, device="cpu", *, frozen=True):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(payload.get("format") == CHECKPOINT_FORMAT and payload.get("variant") == VARIANT,
            "Expected a Context V2 CONCAT_SKIP checkpoint")
    model = ContextualAPRModel(ContextEncoderConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(not frozen)
    return model, payload

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


PEPTIDE_TRANSFER_PREFIXES = (
    "sequence.encoder.",
    "sequence.peptide_adapter.",
    "sequence.peptide_short_cnn.",
    "sequence.peptide_short_cnn_projection.",
    "sequence.peptide_short_cnn_gate_logit",
    "sequence.peptide_length_gate.",
    "sequence.peptide_head.",
    "sequence.peptide_residue_feature_projection.",
    "sequence.peptide_residue_feature_gate_logit",
    "sequence.aggregation_propensity_head.",
)


def is_peptide_transfer_parameter(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in PEPTIDE_TRANSFER_PREFIXES)


def load_peptide_transfer_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only Peptide-learned representation modules into an APR model."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=map_location, weights_only=False
    )
    source_state = checkpoint.get("model", checkpoint)
    target_state = model.state_dict()
    selected = {
        name: tensor
        for name, tensor in source_state.items()
        if is_peptide_transfer_parameter(name)
    }
    unexpected = sorted(set(selected) - set(target_state))
    shape_mismatches = sorted(
        name
        for name in set(selected) & set(target_state)
        if tuple(selected[name].shape) != tuple(target_state[name].shape)
    )
    loadable = {
        name: tensor
        for name, tensor in selected.items()
        if name in target_state and name not in shape_mismatches
    }
    expected = {
        name for name in target_state if is_peptide_transfer_parameter(name)
    }
    trainable_parameters = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_required = sorted(
        name
        for name in expected - set(loadable)
        if name in trainable_parameters
    )
    missing_frozen = sorted(
        name
        for name in expected - set(loadable)
        if name not in trainable_parameters
    )
    if unexpected or shape_mismatches or missing_required:
        raise ValueError(
            "Peptide transfer checkpoint is incompatible: "
            f"unexpected={unexpected}, shape_mismatches={shape_mismatches}, "
            f"missing_required={missing_required}"
        )
    model.load_state_dict(loadable, strict=False)
    excluded_source = sorted(set(source_state) - set(selected))
    if any(name.startswith("sequence.apr_head.") for name in loadable):
        raise AssertionError("APR head must not be loaded from Peptide pretraining")
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_parameter_n": len(loadable),
        "loaded_parameters": sorted(loadable),
        "missing_frozen_parameters": missing_frozen,
        "excluded_source_parameter_n": len(excluded_source),
        "apr_head_loaded": False,
        "structure_loaded": False,
    }

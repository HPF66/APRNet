from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


GRADIENT_CLIPPING_MODES = ("task", "separate", "global", "none")


def trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def _parameters_with_grad(module: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]


def _gradient_norm(parameters: list[nn.Parameter], max_norm: float) -> float:
    if not parameters:
        return 0.0
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
    return float(norm.detach().cpu())


def clip_model_gradients(
    model: nn.Module,
    mode: str,
    max_norm: float,
) -> dict[str, float]:
    """Clip disconnected task branches without cross-scaling them."""
    if mode not in GRADIENT_CLIPPING_MODES:
        raise ValueError(f"Unknown gradient clipping mode: {mode}")
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    if mode == "none":
        parameters = _parameters_with_grad(model)
        return {"global_grad_norm": _gradient_norm(parameters, float("inf"))}

    if mode == "global":
        parameters = _parameters_with_grad(model)
        return {"global_grad_norm": _gradient_norm(parameters, max_norm)}

    if mode == "task":
        norms: dict[str, float] = {}
        sequence = getattr(model, "sequence", None)
        if sequence is not None:
            peptide_prefixes = (
                "peptide_adapter.",
                "peptide_head.",
                "aggregation_propensity_head.",
                "peptide_apr_feature_residual_head.",
                "peptide_residue_feature_projection.",
                "peptide_residue_feature_gate_logit",
            )
            apr_parameters: list[nn.Parameter] = []
            peptide_parameters: list[nn.Parameter] = []
            for name, parameter in sequence.named_parameters():
                if not parameter.requires_grad or parameter.grad is None:
                    continue
                target = (
                    peptide_parameters
                    if name.startswith(peptide_prefixes)
                    else apr_parameters
                )
                target.append(parameter)
            if apr_parameters:
                norms["apr_grad_norm"] = _gradient_norm(apr_parameters, max_norm)
            if peptide_parameters:
                norms["peptide_grad_norm"] = _gradient_norm(
                    peptide_parameters, max_norm
                )
        structure = getattr(model, "structure", None)
        if structure is not None:
            structure_parameters = _parameters_with_grad(structure)
            if structure_parameters:
                norms["structure_grad_norm"] = _gradient_norm(
                    structure_parameters, max_norm
                )
        return norms

    norms: dict[str, float] = {}
    for name in ("sequence", "structure"):
        module = getattr(model, name, None)
        if module is None:
            continue
        parameters = _parameters_with_grad(module)
        if parameters:
            norms[f"{name}_grad_norm"] = _gradient_norm(parameters, max_norm)
    return norms

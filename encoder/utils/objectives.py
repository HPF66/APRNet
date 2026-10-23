"""Optional boundary supervision and regional SupCon for contextual HAPR.

Boundary targets refer to transitions in the annotation, not experimentally
verified physical endpoints. Background retains the dataset's proxy-negative
semantics. Neither target labels nor the training heads enter encoder inputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import encoder.utils.support as baseline
from encoder.utils.contrastive import hapr_supcon_loss


def boundary_detection_loss(logits: Tensor, labels: Tensor, mask: Tensor, *,
                            sample_ids, starts, diagnostics: dict | None = None) -> Tensor:
    """Protein-balanced BCE for y[i] != y[i+1], on valid adjacent pairs.

    Within each protein, positive/negative classes receive equal total weight
    when both exist. Overlapping chunks cannot count one physical pair twice.
    No boundary is manufactured at a chunk or sequence endpoint.
    """
    baseline.require(labels.shape == mask.shape and logits.shape == (labels.shape[0], max(labels.shape[1] - 1, 0)),
                     "Boundary logits must have shape [batch, length-1]")
    baseline.require(len(sample_ids) == labels.shape[0] and len(starts) == labels.shape[0],
                     "Boundary chunk metadata does not match batch")
    if diagnostics is not None:
        diagnostics.update(pairs=0, annotated_boundaries=0, proteins=0)
    if labels.shape[1] < 2:
        return logits.sum() * 0.
    valid_pairs = (mask[:, :-1] & mask[:, 1:]).detach().cpu()
    targets = (labels[:, :-1] != labels[:, 1:]).to(logits.dtype)
    seen, by_protein = set(), {}
    for row, sid in enumerate(sample_ids):
        for col in torch.nonzero(valid_pairs[row], as_tuple=False).flatten().tolist():
            key = (sid, int(starts[row]) + col)
            if key in seen:
                continue
            seen.add(key)
            by_protein.setdefault(sid, []).append(row * logits.shape[1] + col)
    if not by_protein:
        return logits.sum() * 0.
    protein_losses, positives = [], 0
    for indices in by_protein.values():
        selection = torch.tensor(indices, dtype=torch.long, device=logits.device)
        selected_logits = logits.flatten()[selection]
        selected_targets = targets.flatten()[selection]
        losses = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
        positive = selected_targets.bool()
        classes = [losses[valid].mean() for valid in (positive, ~positive) if bool(valid.any())]
        protein_losses.append(torch.stack(classes).mean())
        positives += int(positive.sum().item())
    if diagnostics is not None:
        diagnostics.update(pairs=len(seen), annotated_boundaries=positives, proteins=len(by_protein))
    return torch.stack(protein_losses).mean()


def representation_losses(model, output, batch, args, generator) -> tuple[Tensor, dict]:
    """Add optional supervision while exporting the unprojected HAPR."""
    zero = output["residue_embedding"].sum() * 0.
    boundary, contrast = zero, zero
    stats, boundary_stats, contrast_stats = {}, {}, {}
    # Construct readouts only when an objective actually uses them. This makes
    # the zero-weight configuration identical to the original context baseline.
    if args.boundary_detection_weight > 0 or (args.supcon_weight > 0 and args.supcon_space == "projection"):
        auxiliary = model.representation_readouts(output["residue_embedding"], batch["mask"])
        if "boundary_logits" in output:
            auxiliary["boundary_logits"] = output["boundary_logits"]
    else:
        auxiliary = {}
    if args.boundary_detection_weight > 0:
        baseline.require("boundary_logits" in auxiliary, "Boundary loss requires boundary_head in model_config")
        boundary = boundary_detection_loss(
            auxiliary["boundary_logits"], batch["target"], batch["mask"],
            sample_ids=batch["sample_ids"], starts=batch["starts"], diagnostics=boundary_stats,
        )
    if args.supcon_weight > 0:
        features = output["residue_embedding"]
        if args.supcon_space == "projection":
            baseline.require("contrastive_embedding" in auxiliary, "Projection SupCon requires supcon_projection_dim")
            features = auxiliary["contrastive_embedding"]
        contrast = hapr_supcon_loss(
            features, batch["target"], batch["mask"], args.supcon_temperature,
            args.supcon_max_items, sample_ids=batch["sample_ids"], starts=batch["starts"],
            generator=generator, boundary_radius=args.supcon_boundary_radius,
            cross_protein_weight=args.supcon_cross_protein_weight, diagnostics=contrast_stats,
        )
        # The existing helper calls this HAPR std; with a projection head it
        # measures projected features, so do not misreport it as encoder rank.
        if "normalized_hapr_std" in contrast_stats:
            contrast_stats["normalized_feature_std"] = contrast_stats.pop("normalized_hapr_std")
        contrast_stats["feature_width"] = features.shape[-1]
    stats.update(boundary_detection=float(boundary.detach()), supcon=float(contrast.detach()))
    stats.update({"boundary_" + key: value for key, value in boundary_stats.items()})
    stats.update({"supcon_" + key: value for key, value in contrast_stats.items()})
    return args.boundary_detection_weight * boundary + args.supcon_weight * contrast, stats


def _forward(model, batch):
    return model(batch["base"], batch["tokens"], batch["token_starts"], batch["token_lengths"],
                 batch["token_mask"], batch["mask"], batch["token_probability"],
                 batch["token_local_probability"], batch["token_scale"], batch["residue_evidence"],
                 profile_hidden=batch["profile_hidden"], profile_local_probability=batch["profile_local_probability"],
                 return_encoder=True)


def e2r_base_loss(model, output, batch, pos_weight, args, chunk_counts):
    """Same E2R losses/weights as the baseline, exposed for additive objectives.

    This names the historical `boundary` term accurately as within-label
    similarity. The zero-extra-weight epoch takes the original implementation
    directly; parity is also checked for this factored loss and its gradients.
    """
    labels, mask = batch["target"], batch["mask"]
    row_weights = torch.tensor(
        [(1. / max(chunk_counts[str(sid)], 1))
         * (args.cpad_weight if str(sid).startswith("CPADPEP-") else 1.0)
         for sid in batch["sample_ids"]],
                               dtype=torch.float32, device=labels.device)
    mean = lambda values: baseline._row_weighted_mean(values, mask, row_weights)
    bce = mean(F.binary_cross_entropy_with_logits(output["apr_logits"], labels, pos_weight=pos_weight, reduction="none"))
    local_bce = mean(F.binary_cross_entropy_with_logits(output["local_logits"], labels, pos_weight=pos_weight, reduction="none"))
    adjacent = mask[:, 1:] & mask[:, :-1] & (labels[:, 1:] == labels[:, :-1])
    groups = [adjacent & (labels[:, 1:] == value) for value in (1, 0)]
    probability = output["apr_logits"].sigmoid()
    probability_delta = (probability[:, 1:] - probability[:, :-1]).abs()
    normalized = F.normalize(output["residue_embedding"], dim=-1)
    embedding_delta = 1. - (normalized[:, 1:] * normalized[:, :-1]).sum(dim=-1)
    zero = output["apr_logits"].sum() * 0.

    def adjacent_mean(values):
        parts = [values[valid].mean() for valid in groups if bool(valid.any())]
        return torch.stack(parts).mean() if parts else zero

    smoothness, within_label = adjacent_mean(probability_delta), adjacent_mean(embedding_delta)
    view_a = baseline.perturb_view(batch, args.view_token_noise, args.view_scale_dropout)
    view_b = baseline.perturb_view(batch, args.view_token_noise, args.view_scale_dropout)
    out_a, out_b = _forward(model, view_a), _forward(model, view_b)
    consistency = mean(1. - (F.normalize(out_a["residue_embedding"], dim=-1)
                            * F.normalize(out_b["residue_embedding"], dim=-1)).sum(dim=-1))
    overlap = baseline.overlap_consistency(output["residue_embedding"], batch)
    stability = torch.exp(-((out_a["residue_embedding"] - out_b["residue_embedding"]).abs().mean(dim=-1) / .5)).detach()
    reliability = mean(F.binary_cross_entropy(output["reliability"], stability, reduction="none"))
    measurement, hidden, reconstructed = baseline.window_reconstruction_losses(output, batch, mask)
    loss = (bce + args.local_prior_loss_weight * local_bce + args.smoothness_weight * smoothness
            + args.view_consistency_weight * consistency + args.overlap_consistency_weight * overlap
            + args.boundary_weight * within_label + args.reliability_consistency_weight * reliability
            + args.window_measurement_weight * measurement + args.window_hidden_reconstruction_weight * hidden)
    stats = {name: float(value.detach()) for name, value in {
        "bce": bce, "local_bce": local_bce, "view_consistency": consistency,
        "window_measurement": measurement, "hidden_reconstruction": hidden,
        "within_label_similarity": within_label, "probability_smoothness": smoothness,
        "overlap_consistency": overlap, "reliability_stability": reliability,
    }.items()}
    stats["reconstructed_windows"] = reconstructed
    return loss, stats


def run_epoch(model, loader, device, optimizer, pos_weight, args, chunk_counts,
              diagnostics=None, supcon_generator=None):
    if args.boundary_detection_weight == 0 and args.supcon_weight == 0:
        return baseline.robust_run_epoch(model, loader, device, optimizer, pos_weight, args,
                                         chunk_counts, "C2_CS_E2R", diagnostics=diagnostics,
                                         supcon_generator=supcon_generator)
    model.train(optimizer is not None)
    generator = supcon_generator if supcon_generator is not None else torch.Generator().manual_seed(args.seed + 9100)
    accumulation_steps = max(int(args.gradient_accumulation_steps), 1)
    pending, count, total, component_totals = 0, 0, 0., {}
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for raw in loader:
        batch = baseline.trainer.move(raw, device)
        output = _forward(model, batch)
        base_loss, stats = e2r_base_loss(model, output, batch, pos_weight, args, chunk_counts)
        extra_loss, extra_stats = representation_losses(model, output, batch, args, generator)
        loss = base_loss + extra_loss
        baseline.require(bool(torch.isfinite(loss)), "Non-finite contextual representation training loss")
        if optimizer is not None:
            (loss / accumulation_steps).backward()
            pending += 1
            if pending == accumulation_steps:
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        for name, value in {**stats, **extra_stats}.items():
            component_totals[name] = component_totals.get(name, 0.) + value
        total += float(loss.detach())
        count += 1
    if optimizer is not None and pending:
        # Preserve the baseline accumulation convention for this comparison.
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if diagnostics is not None:
        diagnostics.update({name: value / max(count, 1) for name, value in component_totals.items()})
        diagnostics["batches"] = count
    return total / max(count, 1)

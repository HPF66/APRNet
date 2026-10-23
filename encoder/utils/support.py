from __future__ import annotations
from encoder.model.local_encoder import IntegratedAPRModel, IntegratedContext, peptide_summary
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

def zero_esm(store: dict[str, dict[str, np.ndarray]]) -> None:
    for values in store.values():
        values['esm'] = np.zeros_like(values['esm'], dtype=np.float32)

def metrics_with_threshold(proteins: list[trainer.Protein], probabilities: dict[str, np.ndarray], threshold: float) -> dict[str, float | int]:
    result = trainer.metrics(proteins, probabilities, threshold)
    labels = np.concatenate([p.target.astype(np.int64) for p in proteins])
    values = np.concatenate([probabilities[p.sample_id] for p in proteins])
    prediction = values >= threshold
    result['f1'] = float(f1_score(labels, prediction, zero_division=0))
    result['accuracy'] = float(accuracy_score(labels, prediction))
    return result

def _row_weighted_mean(values: Tensor, mask: Tensor, row_weights: Tensor) -> Tensor:
    per_row = (values * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
    return (per_row * row_weights).sum() / row_weights.sum().clamp_min(1e-06)

def perturb_view(batch: dict[str, object], noise_scale: float, scale_dropout: float) -> dict[str, object]:
    """Create a peptide-only view by dropping whole window scales and adding noise."""
    view = dict(batch)
    tokens = batch['tokens'].clone()
    token_mask = batch['token_mask'].clone()
    evidence = batch['residue_evidence'].clone()
    batch_size = tokens.shape[0]
    keep_scales = torch.ones((batch_size, len(WINDOW_SCALES)), dtype=torch.bool, device=tokens.device)
    for row in range(batch_size):
        present = torch.unique(batch['token_scale'][row][batch['token_mask'][row]].round().long())
        if present.numel() == 0:
            continue
        dropped = torch.rand(present.shape, device=tokens.device) < scale_dropout
        if bool(dropped.all()):
            dropped[0] = False
        keep_scales[row, present[dropped]] = False
        token_mask[row] = token_mask[row] & keep_scales[row, batch['token_scale'][row].round().long()]
        for scale_index in range(len(WINDOW_SCALES)):
            if not bool(keep_scales[row, scale_index]):
                left = scale_index * 4
                evidence[row, :, left:left + 4] = 0.0
    tokens = (tokens + noise_scale * torch.randn_like(tokens)) * token_mask.unsqueeze(-1)
    view['tokens'] = tokens
    view['token_mask'] = token_mask
    view['residue_evidence'] = evidence
    return view

def overlap_consistency(hapr: Tensor, batch: dict[str, object]) -> Tensor:
    """Compare HAPR on residues shared by two chunks of the same protein."""
    sample_ids = batch['sample_ids']
    starts = batch['starts']
    mask = batch['mask']
    losses: list[Tensor] = []
    for left in range(hapr.shape[0]):
        for right in range(left + 1, hapr.shape[0]):
            if sample_ids[left] != sample_ids[right]:
                continue
            start = max(int(starts[left]), int(starts[right]))
            stop = min(int(starts[left]) + int(mask[left].sum()), int(starts[right]) + int(mask[right].sum()))
            if stop <= start:
                continue
            a = hapr[left, start - int(starts[left]):stop - int(starts[left])]
            b = hapr[right, start - int(starts[right]):stop - int(starts[right])]
            losses.append(F.mse_loss(F.normalize(a, dim=-1), F.normalize(b, dim=-1)))
    if not losses:
        return hapr.sum() * 0.0
    return torch.stack(losses).mean()

def window_reconstruction_losses(output: dict[str, Tensor], batch: dict[str, object], residue_mask: Tensor) -> tuple[Tensor, Tensor, int]:
    """Reconstruct frozen peptide observations from the residue representation.

    Only windows fully contained in the current chunk are used.  This avoids
    comparing a clipped chunk occupancy against a teacher probability computed
    from the complete peptide window at chunk boundaries.
    """
    starts = batch['token_starts'].round().long()
    lengths = batch['token_lengths'].round().long().clamp_min(1)
    token_mask = batch['token_mask']
    teacher_probability = batch['token_probability'].detach().clamp(0.0, 1.0)
    teacher_hidden = batch['tokens'].detach()
    hapr = output['residue_embedding']
    decoded = output['window_hidden_prediction']
    bsz, tmax, _ = hapr.shape
    wmax = starts.shape[1]
    fully_inside = token_mask & (starts >= 0) & (starts + lengths <= residue_mask.sum(dim=1, dtype=torch.long).unsqueeze(1))
    pooled_hapr = hapr.new_zeros((bsz, wmax, hapr.shape[-1]))
    pooled_decoded = decoded.new_zeros((bsz, wmax, decoded.shape[-1]))
    counts = hapr.new_zeros((bsz, wmax))
    positions = torch.arange(tmax, device=hapr.device).view(1, tmax)
    residue_valid = residue_mask.bool()
    for window_index in range(wmax):
        left = starts[:, window_index].unsqueeze(1)
        right = (starts[:, window_index] + lengths[:, window_index]).unsqueeze(1)
        valid_window = (positions >= left) & (positions < right) & residue_valid & fully_inside[:, window_index].unsqueeze(1)
        count = valid_window.sum(dim=1).to(hapr.dtype)
        counts[:, window_index] = count
        pooled_hapr[:, window_index] = (hapr * valid_window.unsqueeze(-1).to(hapr.dtype)).sum(dim=1) / count.clamp_min(1).unsqueeze(-1)
        pooled_decoded[:, window_index] = (decoded * valid_window.unsqueeze(-1).to(decoded.dtype)).sum(dim=1) / count.clamp_min(1).to(decoded.dtype).unsqueeze(-1)
    valid_windows = counts > 0
    if not bool(valid_windows.any()):
        zero = hapr.sum() * 0.0
        return (zero, zero, 0)
    hidden_target = teacher_hidden
    hidden_cosine = 1.0 - F.cosine_similarity(pooled_decoded, hidden_target, dim=-1)
    hidden_loss = hidden_cosine[valid_windows].mean()
    pooled_probability = pooled_hapr.new_zeros((bsz, wmax))
    probability = torch.sigmoid(output['apr_logits'])
    for window_index in range(wmax):
        left = starts[:, window_index].unsqueeze(1)
        right = (starts[:, window_index] + lengths[:, window_index]).unsqueeze(1)
        valid_window = (positions >= left) & (positions < right) & residue_valid & fully_inside[:, window_index].unsqueeze(1)
        pooled_probability[:, window_index] = (probability * valid_window.to(probability.dtype)).sum(dim=1) / counts[:, window_index].clamp_min(1).to(probability.dtype)
    measurement_loss = F.smooth_l1_loss(pooled_probability[valid_windows], teacher_probability[valid_windows], reduction='mean')
    return (measurement_loss, hidden_loss, int(valid_windows.sum().item()))

def robust_run_epoch(model: IntegratedAPRModel, loader, device: torch.device, optimizer: torch.optim.Optimizer | None, pos_weight: Tensor, args: argparse.Namespace, chunk_counts: dict[str, int], variant: str, diagnostics: dict | None=None, supcon_generator: torch.Generator | None=None) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    accumulation_steps = max(int(args.gradient_accumulation_steps), 1)
    pending = 0
    component_totals = {}
    if supcon_generator is None:
        supcon_generator = torch.Generator().manual_seed(args.seed + 9100)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for raw in loader:
        batch = trainer.move(raw, device)
        output = model(batch['base'], batch['tokens'], batch['token_starts'], batch['token_lengths'], batch['token_mask'], batch['mask'], batch['token_probability'], batch['token_local_probability'], batch['token_scale'], batch['residue_evidence'], profile_hidden=batch['profile_hidden'], profile_local_probability=batch['profile_local_probability'], return_encoder=True)
        labels = batch['target']
        mask = batch['mask']
        row_weights = torch.tensor([1.0 / max(chunk_counts[str(sample_id)], 1) for sample_id in batch['sample_ids']], dtype=torch.float32, device=device)
        loss_map = F.binary_cross_entropy_with_logits(output['apr_logits'], labels, pos_weight=pos_weight, reduction='none')
        bce = _row_weighted_mean(loss_map, mask, row_weights)
        local_bce = output['apr_logits'].sum() * 0.0
        smoothness = output['apr_logits'].sum() * 0.0
        boundary = output['apr_logits'].sum() * 0.0
        consistency = output['apr_logits'].sum() * 0.0
        overlap = output['apr_logits'].sum() * 0.0
        reliability_loss = output['apr_logits'].sum() * 0.0
        supcon = output['apr_logits'].sum() * 0.0
        contrast_stats = {}
        if variant not in {'C2_CLEAN', 'C2_CLEAN_RECON'}:
            local_map = F.binary_cross_entropy_with_logits(output['local_logits'], labels, pos_weight=pos_weight, reduction='none')
            local_bce = _row_weighted_mean(local_map, mask, row_weights)
            probability = torch.sigmoid(output['apr_logits'])
            adjacent = mask[:, 1:] & mask[:, :-1] & (labels[:, 1:] == labels[:, :-1])
            positive_adjacent = adjacent & (labels[:, 1:] == 1)
            negative_adjacent = adjacent & (labels[:, 1:] == 0)
            probability_delta = torch.abs(probability[:, 1:] - probability[:, :-1])
            smooth_parts = [probability_delta[selected].mean() for selected in (positive_adjacent, negative_adjacent) if bool(torch.any(selected))]
            smoothness = torch.stack(smooth_parts).mean() if smooth_parts else output['apr_logits'].sum() * 0.0
            normalized = F.normalize(output['residue_embedding'], dim=-1)
            embedding_delta = 1.0 - (normalized[:, 1:] * normalized[:, :-1]).sum(dim=-1)
            boundary_parts = [embedding_delta[selected].mean() for selected in (positive_adjacent, negative_adjacent) if bool(torch.any(selected))]
            boundary = torch.stack(boundary_parts).mean() if boundary_parts else output['apr_logits'].sum() * 0.0
            view_a = perturb_view(batch, args.view_token_noise, args.view_scale_dropout)
            view_b = perturb_view(batch, args.view_token_noise, args.view_scale_dropout)
            out_a = model(view_a['base'], view_a['tokens'], view_a['token_starts'], view_a['token_lengths'], view_a['token_mask'], view_a['mask'], view_a['token_probability'], view_a['token_local_probability'], view_a['token_scale'], view_a['residue_evidence'], profile_hidden=view_a['profile_hidden'], profile_local_probability=view_a['profile_local_probability'], return_encoder=True)
            out_b = model(view_b['base'], view_b['tokens'], view_b['token_starts'], view_b['token_lengths'], view_b['token_mask'], view_b['mask'], view_b['token_probability'], view_b['token_local_probability'], view_b['token_scale'], view_b['residue_evidence'], profile_hidden=view_b['profile_hidden'], profile_local_probability=view_b['profile_local_probability'], return_encoder=True)
            consistency_map = 1.0 - (F.normalize(out_a['residue_embedding'], dim=-1) * F.normalize(out_b['residue_embedding'], dim=-1)).sum(dim=-1)
            consistency = _row_weighted_mean(consistency_map, mask, row_weights)
            overlap = overlap_consistency(output['residue_embedding'], batch)
            stability = torch.exp(-((out_a['residue_embedding'] - out_b['residue_embedding']).abs().mean(dim=-1) / 0.5)).detach()
            reliability_loss = _row_weighted_mean(F.binary_cross_entropy(output['reliability'], stability, reduction='none'), mask, row_weights)
            if variant == 'C2_CS_SUPCON':
                supcon = hapr_supcon_loss(output['residue_embedding'], labels, mask, args.supcon_temperature, args.supcon_max_items, sample_ids=batch['sample_ids'], starts=batch['starts'], generator=supcon_generator, boundary_radius=args.supcon_boundary_radius, cross_protein_weight=args.supcon_cross_protein_weight, diagnostics=contrast_stats)
        window_measurement, window_hidden, reconstructed_windows = window_reconstruction_losses(output, batch, mask)
        if variant == 'C2_CLEAN':
            loss = bce
        elif variant == 'C2_CLEAN_RECON':
            loss = bce + args.window_measurement_weight * window_measurement + args.window_hidden_reconstruction_weight * window_hidden
        else:
            loss = bce + args.local_prior_loss_weight * local_bce + args.smoothness_weight * smoothness + args.view_consistency_weight * consistency + args.overlap_consistency_weight * overlap + args.boundary_weight * boundary + args.reliability_consistency_weight * reliability_loss + args.window_measurement_weight * window_measurement + args.window_hidden_reconstruction_weight * window_hidden
            if variant == 'C2_CS_SUPCON':
                loss = loss + args.supcon_weight * supcon
        require(bool(torch.isfinite(loss)), 'non-finite training loss')
        if diagnostics is not None:
            components = {'bce': float(bce.detach()), 'local_bce': float(local_bce.detach()), 'view_consistency': float(consistency.detach()), 'window_measurement': float(window_measurement.detach()), 'hidden_reconstruction': float(window_hidden.detach()), 'supcon': float(supcon.detach()), 'reconstructed_windows': reconstructed_windows, **{f'supcon_{key}': value for key, value in contrast_stats.items()}}
            for key, value in components.items():
                component_totals[key] = component_totals.get(key, 0.0) + value
        if optimizer is not None:
            (loss / float(accumulation_steps)).backward()
            pending += 1
            if pending >= accumulation_steps:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        total += float(loss.detach())
        count += 1
    if optimizer is not None and pending:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if diagnostics is not None:
        diagnostics.update({key: value / max(count, 1) for key, value in component_totals.items()})
        diagnostics['batches'] = count
    return total / max(count, 1)

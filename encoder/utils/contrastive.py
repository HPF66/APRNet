"""Training-only, region-aware supervised contrast on HAPR (no APR logits)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def hapr_supcon_loss(
    hapr: Tensor, labels: Tensor, mask: Tensor, temperature: float,
    max_items: int = 256, *, sample_ids=None, starts=None,
    generator: torch.Generator, boundary_radius: int = 3,
    cross_protein_weight: float = 0.25, diagnostics: dict | None = None,
) -> Tensor:
    """APR anchors attract same-segment and weak cross-protein APR positives.

    Background residues are negatives only, never a single positive cluster.
    Half of the negative budget preferentially samples boundary neighborhoods.
    Overlapping chunks cannot supply duplicate physical residues as positives.
    Sampling uses a dedicated CPU RNG, leaving model/dropout RNG unchanged.
    """
    if temperature <= 0 or max_items < 4 or boundary_radius < 0:
        raise ValueError("invalid supervised contrastive sampling settings")
    if not 0 <= cross_protein_weight <= 1:
        raise ValueError("cross-protein weight must be between zero and one")
    zero = hapr.sum() * 0.0
    if diagnostics is not None:
        diagnostics.update(active=0, anchors=0, sampled_positive=0, sampled_negative=0)
    valid = mask.detach().bool().cpu()
    target = labels.detach().cpu()
    bsz, width = valid.shape
    sample_ids = list(range(bsz)) if sample_ids is None else list(sample_ids)
    starts = [0] * bsz if starts is None else [int(x) for x in starts]
    if len(sample_ids) != bsz or len(starts) != bsz:
        raise ValueError("chunk metadata does not match batch")
    protein_numbers = {key: number for number, key in enumerate(dict.fromkeys(sample_ids))}
    seen = set()
    candidates = []
    for row in range(bsz):
        positive = (target[row] == 1) & valid[row]
        near_positive = F.max_pool1d(
            positive.float()[None, None], 2 * boundary_radius + 1,
            stride=1, padding=boundary_radius,
        )[0, 0].bool()
        segment = 0
        for col in torch.nonzero(valid[row], as_tuple=False).flatten().tolist():
            is_positive = bool(positive[col])
            if is_positive and (col == 0 or not bool(positive[col - 1])):
                segment += 1
            origin = (sample_ids[row], starts[row] + col)
            if origin in seen:
                continue
            seen.add(origin)
            candidates.append((row * width + col, int(is_positive),
                               protein_numbers[sample_ids[row]], row, segment,
                               int(not is_positive and near_positive[col])))
    if not candidates:
        return zero
    data = torch.tensor(candidates, dtype=torch.long)

    def take(indices: Tensor, limit: int) -> Tensor:
        if len(indices) <= limit:
            return indices
        return indices[torch.randperm(len(indices), generator=generator)[:limit]]

    def balanced_take(indices: Tensor, limit: int, groups: Tensor) -> Tensor:
        # Round-robin across proteins/segments so a long APR does not dominate.
        if len(indices) <= limit:
            return indices
        queues = []
        for group in torch.unique(groups[indices]):
            items = indices[groups[indices] == group]
            queues.append(items[torch.randperm(len(items), generator=generator)].tolist())
        order = torch.randperm(len(queues), generator=generator).tolist()
        chosen = []
        while len(chosen) < limit and any(queues):
            for index in order:
                if queues[index]:
                    chosen.append(queues[index].pop())
                    if len(chosen) == limit:
                        break
        return torch.tensor(chosen, dtype=torch.long)

    pos = torch.nonzero(data[:, 1] == 1).flatten()
    neg = torch.nonzero(data[:, 1] == 0).flatten()
    if len(pos) < 2 or len(neg) == 0:
        return zero
    budget = max_items // 2
    groups = data[:, 3] * (width + 1) + data[:, 4]
    pos = balanced_take(pos, budget, groups)
    hard = balanced_take(neg[data[neg, 5] == 1], budget // 2, data[:, 2])
    remaining = neg[~torch.isin(neg, hard)]
    neg = torch.cat([hard, balanced_take(remaining, budget - len(hard), data[:, 2])])
    selected = torch.cat([pos, neg])
    info = data[selected].to(hapr.device)
    features = F.normalize(hapr.flatten(0, 1)[info[:, 0]].float(), dim=-1)
    cosine = features @ features.T
    is_apr = info[:, 1].bool()
    not_self = ~torch.eye(len(selected), dtype=torch.bool, device=hapr.device)
    apr_pair = is_apr[:, None] & is_apr[None, :] & not_self
    same_segment = apr_pair & (info[:, 3, None] == info[None, :, 3]) & (info[:, 4, None] == info[None, :, 4])
    cross_protein = apr_pair & (info[:, 2, None] != info[None, :, 2])
    positives = same_segment | (cross_protein & (cross_protein_weight > 0))
    negatives = is_apr[:, None] & ~is_apr[None, :]
    active = is_apr & positives.any(dim=1) & negatives.any(dim=1)
    if not bool(active.any()):
        return zero
    # Other APR segments in the same protein are ignored, not false negatives.
    candidate_mask = (positives | negatives)[active]
    logits = cosine[active] / temperature
    log_probability = logits - torch.logsumexp(
        logits.masked_fill(~candidate_mask, -torch.inf), dim=1, keepdim=True,
    )
    local_mask, cross_mask = same_segment[active], cross_protein[active]
    local_count, cross_count = local_mask.sum(1), cross_mask.sum(1)
    local_loss = -(log_probability * local_mask).sum(1) / local_count.clamp_min(1)
    cross_loss = -(log_probability * cross_mask).sum(1) / cross_count.clamp_min(1)
    local_weight = (local_count > 0).float()
    cross_weight = (cross_count > 0).float() * cross_protein_weight
    losses = (local_loss * local_weight + cross_loss * cross_weight) / (local_weight + cross_weight).clamp_min(1e-8)
    anchor_proteins = info[active, 2]
    loss = torch.stack([losses[anchor_proteins == p].mean() for p in anchor_proteins.unique()]).mean()
    if diagnostics is not None:
        with torch.no_grad():
            diagnostics.update(
                active=1, anchors=int(active.sum()), sampled_positive=len(pos),
                sampled_negative=len(neg), boundary_negatives=int(info[~is_apr, 5].sum()),
                cross_protein_pairs=int(cross_protein.sum()),
                positive_cosine=float(cosine[positives].mean()),
                negative_cosine=float(cosine[negatives].mean()),
                normalized_hapr_std=float(features.std(dim=0, unbiased=False).mean()),
            )
    return loss

#!/usr/bin/env python3
"""Build nested 1:1--1:5 protein cohorts from primary data and candidates.

The primary labels and chains are copied byte-for-byte at the row level.  New
background records are selected without model scores using joint length,
species and deposited/full-parent strata plus amino-acid-composition distance.
The generated candidates are *source-screened assumed background negatives*;
this script does not claim experimental non-aggregation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


AA = "ACDEFGHIKLMNPQRSTVWY"
RATIOS = (1, 2, 3, 4, 5)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha_sequence(sequence: str) -> str:
    return "SEQ-" + hashlib.sha256(sequence.encode("ascii")).hexdigest()[:12].upper()


def length_bin(length: int) -> str:
    if length < 50:
        return "<50"
    for upper, name in ((100, "50-99"), (200, "100-199"), (400, "200-399"),
                        (800, "400-799"), (1600, "800-1599")):
        if length < upper:
            return name
    return "1600+"


def normalized_scope(value: str) -> str:
    value = (value or "").strip()
    if value in {"full_uniprot_parent_with_partial_structure", "full_parent"}:
        return "full_parent"
    if value in {"deposited_pdb_polymer_construct", "original_target_chain_sequences",
                 "deposited_construct", "clinical_sequence"}:
        return "deposited_construct"
    return value or "deposited_construct"


def sequence_from_ids(row: dict[str, str], sequences: dict[str, str]) -> list[str]:
    return [sequences[item] for item in row.get("sequence_ids", "").split(";") if item in sequences]


def aac(sequence: str) -> tuple[float, ...]:
    length = max(1, len(sequence))
    return tuple(sequence.count(letter) / length for letter in AA)


def mean_vectors(values: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    values = list(values)
    if not values:
        return tuple(0.0 for _ in AA)
    return tuple(sum(value[i] for value in values) / len(values) for i in range(len(AA)))


def distance(vector: tuple[float, ...], reference: tuple[float, ...], scale: tuple[float, ...]) -> float:
    return sum(abs(a - b) / max(s, 0.01) for a, b, s in zip(vector, reference, scale)) / len(AA)


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    if not weights or total <= 0:
        return {key: 0 for key in weights}
    weight_sum = sum(max(0.0, value) for value in weights.values()) or 1.0
    raw = {key: total * max(0.0, value) / weight_sum for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def candidate_record(row: dict[str, str], index: int, reference: tuple[float, ...], scale: tuple[float, ...]) -> dict[str, object]:
    sequence = "".join(row.get("sequence", "").split()).upper()
    pdb_id = row.get("pdb_id", "").upper()
    uniprot = row.get("uniprot_id", "")
    scope = row.get("scope", "deposited_construct") or "deposited_construct"
    species = row.get("species_group", "other") or "other"
    return {
        "row": row,
        "sample_id": row.get("sample_id") or f"BGNC_{pdb_id}_{row.get('entity_id', index)}",
        "pdb_id": pdb_id,
        "uniprot_id": uniprot,
        "sequence": sequence,
        "sequence_id": sha_sequence(sequence),
        "length_bin": length_bin(len(sequence)),
        "species_group": species,
        "scope": scope,
        "cell": "|".join((length_bin(len(sequence)), species, scope)),
        "aac": aac(sequence),
        "score": distance(aac(sequence), reference, scale),
    }


def build_candidate_pool(primary_labels: list[dict[str, str]], primary_sequences: dict[str, str], candidates: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    primary_pdb = {row.get("pdb_id", "").upper() for row in primary_labels if row.get("pdb_id")}
    primary_uniprot = {row.get("uniprot_id", "") for row in primary_labels if row.get("uniprot_id")}
    positive_vectors: list[tuple[float, ...]] = []
    negative_vectors: list[tuple[float, ...]] = []
    for row in primary_labels:
        vectors = [aac(sequence) for sequence in sequence_from_ids(row, primary_sequences) if len(sequence) >= 50]
        target = positive_vectors if row.get("label") == "1" else negative_vectors
        target.extend(vectors)
    # The reference is deliberately independent of model outputs: equal weight
    # to positive and original-negative composition means neither class alone
    # defines the background pool.
    pos_ref = mean_vectors(positive_vectors)
    neg_ref = mean_vectors(negative_vectors)
    reference = tuple((a + b) / 2 for a, b in zip(pos_ref, neg_ref))
    all_vectors = positive_vectors + negative_vectors
    scale = tuple(max(0.02, math.sqrt(sum((v[i] - reference[i]) ** 2 for v in all_vectors) / max(1, len(all_vectors)))) for i in range(len(AA)))
    unique: dict[tuple[str, str], dict[str, object]] = {}
    excluded = Counter()
    for index, row in enumerate(candidates, start=1):
        sequence = "".join(row.get("sequence", "").split()).upper()
        if len(sequence) < 50 or not sequence or set(sequence) - set(AA):
            excluded["invalid_or_short_sequence"] += 1
            continue
        pdb_id = row.get("pdb_id", "").upper()
        uniprot = row.get("uniprot_id", "")
        if not pdb_id or not uniprot:
            excluded["missing_pdb_or_uniprot"] += 1
            continue
        if pdb_id in primary_pdb or uniprot in primary_uniprot:
            excluded["primary_pdb_or_uniprot"] += 1
            continue
        item = candidate_record(row, index, reference, scale)
        key = (str(item["pdb_id"]), str(item["uniprot_id"]))
        existing = unique.get(key)
        if existing is None or (float(item["score"]), str(item["sample_id"])) < (float(existing["score"]), str(existing["sample_id"])):
            unique[key] = item
        else:
            excluded["duplicate_pdb_uniprot"] += 1
    pool = sorted(unique.values(), key=lambda item: (str(item["cell"]), float(item["score"]), str(item["pdb_id"])))
    return pool, {"reference_aac": dict(zip(AA, reference)), "aac_scale": dict(zip(AA, scale)), "excluded": dict(excluded)}


def labels_for_sample(row: dict[str, str], *, sample_id: str, sequence_id: str,
                      source: str, structure_path: str = "", sha256: str = "") -> dict[str, str]:
    result = dict(row)
    result.update({"sample_id": sample_id, "label": "0", "label_name": "Non-amyloid",
                   "sequence_ids": sequence_id,
                   "target_chain_ids": row.get("chain_ids", "A"),
                   "target_chain_count": str(len([value for value in row.get("chain_ids", "A").split(";") if value]) or 1),
                   "label_evidence": "reviewed source-screened background negative; no direct nonaggregation experiment asserted",
                   "evidence_tier": "screened_background_negative", "source": source})
    if structure_path:
        result["structure_path"] = structure_path
    if sha256:
        result["structure_sha256"] = sha256
    return result


def materialize(primary: Path, selected: list[dict[str, object]], output: Path, ratio: int,
                primary_tables: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    labels = [dict(row) for row in primary_tables["labels"]]
    chains = [dict(row) for row in primary_tables["chains"]]
    provenance = [dict(row) for row in primary_tables.get("provenance", [])]
    sequences = {row["sequence_id"]: row["sequence"] for row in primary_tables["sequences"]}
    source = "joint_aac_matched_background_v1"
    for item in selected:
        row = dict(item["row"])
        sample_id = str(item["sample_id"])
        seq_id = str(item["sequence_id"])
        pdb_id = str(item["pdb_id"])
        labels.append(labels_for_sample(row, sample_id=sample_id, sequence_id=seq_id, source=source))
        sequences[seq_id] = str(item["sequence"])
        chain_ids = [value for value in row.get("chain_ids", "A").split(";") if value] or ["A"]
        for chain_index, chain_id in enumerate(chain_ids, start=1):
            chains.append({"sample_id": sample_id, "chain_id": chain_id, "chain_index": str(chain_index),
                           "chain_sample_id": f"{sample_id}_C{chain_index:03d}", "sequence_id": seq_id,
                           "coordinate_length": str(len(str(item["sequence"]))),
                           "coordinate_sequence": str(item["sequence"]),
                           "coordinate_residue_keys": ",".join(str(i) for i in range(1, len(str(item["sequence"])) + 1)),
                           "coordinate_seqres_positions": ",".join(str(i) for i in range(1, len(str(item["sequence"])) + 1)),
                           "coordinate_mask": "1" * len(str(item["sequence"])), "terminal_modifications": "",
                           "nonstandard_residue_positions": ""})
        provenance.append({"sample_id": sample_id, "label": "0", "pdb_id": pdb_id,
                           "uniprot_id": str(item["uniprot_id"]), "sequence_scope": str(item["scope"]),
                           "species_group": str(item["species_group"]), "length_bin": str(item["length_bin"]),
                           "source": source, "label_evidence": "source-screened assumed background negative"})
    out = output / f"ratio{ratio}"
    out.mkdir(parents=True, exist_ok=True)
    write_tsv(out / "labels.tsv", labels)
    write_tsv(out / "chains.tsv", chains)
    write_tsv(out / "sequences.tsv", [{"sequence_id": key, "sequence": value, "sequence_length": str(len(value))}
                                      for key, value in sorted(sequences.items())])
    fasta = "".join(f">{key}\n{value}\n" for key, value in sorted(sequences.items()))
    (out / "sequences.fasta").write_text(fasta, encoding="utf-8")
    write_tsv(out / "provenance.tsv", provenance)
    summary = {"cohort": f"1:{ratio}", "positive": sum(row.get("label") == "1" for row in labels),
               "negative": sum(row.get("label") == "0" for row in labels), "added_background": len(selected),
               "primary_preserved": True, "label_status": "source-screened assumed background negatives"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True, help="Directory with primary labels/chains/sequences TSVs")
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate TSV from download_background_candidates.py")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for ratio1..ratio5")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--allow-shortfall", action="store_true", help="Write available cohorts when quotas cannot be met")
    args = parser.parse_args()
    random.seed(args.seed)
    primary_tables = {name: read_tsv(args.primary / f"{name}.tsv") for name in ("labels", "chains", "sequences")}
    provenance_path = args.primary / "provenance.tsv"
    primary_tables["provenance"] = read_tsv(provenance_path) if provenance_path.exists() else []
    primary_labels = primary_tables["labels"]
    if sum(row.get("label") == "1" for row in primary_labels) != 1218 or sum(row.get("label") == "0" for row in primary_labels) != 199:
        raise SystemExit("Expected primary labels to contain 1,218 positives and 199 original negatives")
    sequences = {row["sequence_id"]: row["sequence"] for row in primary_tables["sequences"]}
    pool, audit = build_candidate_pool(primary_labels, sequences, read_tsv(args.candidates))
    positives = [row for row in primary_labels if row.get("label") == "1"]
    weights = Counter()
    for row in positives:
        values = sequence_from_ids(row, sequences)
        sequence = max(values, key=len, default="")
        if len(sequence) >= 50:
            # Positive samples define the target strata; candidate strata are
            # never changed to force an unavailable species or scope.
            length = length_bin(len(sequence))
            species = "human" if row.get("organism_id") == "9606" or "9606" in row.get("organism", "") else "other"
            scope = normalized_scope(row.get("sequence_scope", "deposited_construct"))
            weights[f"{length}|{species}|{scope}"] += 1
    if not weights:
        raise SystemExit("No >=50 aa positive sequences available for matching")
    existing_counts = Counter()
    for row in primary_labels:
        if row.get("label") == "0":
            values = sequence_from_ids(row, sequences)
            sequence = max(values, key=len, default="")
            if len(sequence) >= 50:
                length = length_bin(len(sequence))
                species = "human" if row.get("organism_id") == "9606" or "9606" in row.get("organism", "") else "other"
                scope = normalized_scope(row.get("sequence_scope", "deposited_construct"))
                existing_counts[f"{length}|{species}|{scope}"] += 1
    by_cell: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in pool:
        by_cell[str(item["cell"])].append(item)
    selected_so_far: list[dict[str, object]] = []
    selection_ids: set[str] = set()
    summaries = []
    for ratio in RATIOS:
        target_extra = 1218 * ratio - 199
        desired = allocate(target_extra, {cell: float(count) for cell, count in weights.items()})
        cell_quota = {cell: max(0, desired.get(cell, 0) - existing_counts.get(cell, 0)) for cell in weights}
        # A quota may be smaller after rounding than the previous ratio. Keep
        # the previous prefix to guarantee nested cohorts.
        previous_by_cell = Counter(str(item["cell"]) for item in selected_so_far)
        for cell, count in previous_by_cell.items():
            cell_quota[cell] = max(cell_quota.get(cell, 0), count)
        chosen = list(selected_so_far)
        for cell in sorted(cell_quota):
            needed = cell_quota[cell] - previous_by_cell.get(cell, 0)
            for item in by_cell.get(cell, []):
                if needed <= 0:
                    break
                if str(item["sample_id"]) not in selection_ids:
                    chosen.append(item)
                    selection_ids.add(str(item["sample_id"]))
                    needed -= 1
        # Fill any remaining quota from the best unused candidate, preserving
        # deterministic ordering and the nested prefix.
        if len(chosen) < target_extra:
            remaining = [item for item in pool if str(item["sample_id"]) not in selection_ids]
            remaining.sort(key=lambda item: (float(item["score"]), str(item["pdb_id"])))
            for item in remaining[:target_extra - len(chosen)]:
                chosen.append(item)
                selection_ids.add(str(item["sample_id"]))
        shortfall = max(0, target_extra - len(chosen))
        if shortfall and not args.allow_shortfall:
            raise SystemExit(f"ratio 1:{ratio} is short by {shortfall} candidates; rerun with --allow-shortfall to audit it")
        selected_so_far = chosen[:target_extra]
        summary = materialize(args.primary, selected_so_far, args.output, ratio, primary_tables)
        summary.update({"target_added_background": target_extra, "shortfall": shortfall,
                        "nested_parent": f"ratio{ratio - 1}" if ratio > 1 else "primary"})
        (args.output / f"ratio{ratio}" / "selection_audit.json").write_text(
            json.dumps({"protocol": "joint_length_species_scope_aac_v1", "seed": args.seed,
                        "selection_uses_model_scores": False, "pool_audit": audit,
                        "selected_sample_ids": [str(item["sample_id"]) for item in selected_so_far],
                        "summary": summary}, indent=2), encoding="utf-8")
        summaries.append(summary)
    (args.output / "construction_manifest.json").write_text(json.dumps({
        "schema": "APRnet.protein_ratio_cohorts.v1", "seed": args.seed,
        "primary_positive": 1218, "primary_negative": 199,
        "ratios": summaries, "nested": True,
        "background_label_status": "source-screened assumed background negatives; no direct nonaggregation experiment asserted",
        "protocol": {"length_bins": ["50-99", "100-199", "200-399", "400-799", "800-1599", "1600+"],
                     "species_scope_joint_matching": True, "aac_distance": "scaled mean absolute fraction distance",
                     "model_scores_used": False, "outcome_based_retuning": False}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

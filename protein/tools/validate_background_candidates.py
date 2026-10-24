#!/usr/bin/env python3
"""Validate primary preservation, ratio counts, nesting and sequence identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sequence_hash(sequence: str) -> str:
    return "SEQ-" + hashlib.sha256(sequence.encode("ascii")).hexdigest()[:12].upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True, help="Directory containing ratio1..ratio5")
    parser.add_argument("--json", type=Path, help="Optional validation report path")
    args = parser.parse_args()
    primary_labels = rows(args.primary / "labels.tsv")
    primary_chains = rows(args.primary / "chains.tsv")
    primary_sequences = rows(args.primary / "sequences.tsv")
    primary_ids = {row["sample_id"] for row in primary_labels}
    primary_label_snapshot = {row["sample_id"]: row.get("label") for row in primary_labels}
    primary_seq_snapshot = {row["sequence_id"]: row.get("sequence") for row in primary_sequences}
    errors: list[str] = []
    for row in primary_sequences:
        sequence = row.get("sequence", "")
        if row.get("sequence_id", "").startswith("SEQ-") and sequence_hash(sequence) != row.get("sequence_id"):
            errors.append(f"primary sequence hash mismatch: {row.get('sequence_id')}")
    if sum(row.get("label") == "1" for row in primary_labels) != 1218:
        errors.append("primary positive count is not 1,218")
    if sum(row.get("label") == "0" for row in primary_labels) != 199:
        errors.append("primary negative count is not 199")
    previous_ids = set(primary_ids)
    report = {"primary": {"samples": len(primary_ids), "positive": 1218, "negative": 199}, "ratios": {}, "errors": errors}
    for ratio in range(1, 6):
        directory = args.cohorts / f"ratio{ratio}"
        label_rows = rows(directory / "labels.tsv")
        chain_rows = rows(directory / "chains.tsv")
        seq_rows = rows(directory / "sequences.tsv")
        sample_ids = {row["sample_id"] for row in label_rows}
        labels = {row["sample_id"]: row.get("label") for row in label_rows}
        if not primary_ids <= sample_ids:
            errors.append(f"ratio{ratio} does not contain all primary samples")
        if any(labels.get(key) != value for key, value in primary_label_snapshot.items()):
            errors.append(f"ratio{ratio} changed a primary label")
        if any(row.get("sequence") != primary_seq_snapshot.get(row.get("sequence_id"))
               for row in seq_rows if row.get("sequence_id") in primary_seq_snapshot):
            errors.append(f"ratio{ratio} changed a primary sequence")
        for row in seq_rows:
            sequence = row.get("sequence", "")
            if row.get("sequence_id", "").startswith("SEQ-") and sequence_hash(sequence) != row.get("sequence_id"):
                errors.append(f"ratio{ratio} sequence hash mismatch: {row.get('sequence_id')}")
        if not previous_ids <= sample_ids:
            errors.append(f"ratio{ratio} is not nested relative to the previous cohort")
        backgrounds = [row for row in label_rows if row["sample_id"] not in primary_ids]
        if any(row.get("label") != "0" for row in backgrounds):
            errors.append(f"ratio{ratio} contains a nonzero background label")
        if len({row.get("pdb_id") for row in backgrounds if row.get("pdb_id")}) != len([row for row in backgrounds if row.get("pdb_id")]):
            errors.append(f"ratio{ratio} repeats a background PDB ID")
        if len({row.get("uniprot_id") for row in backgrounds if row.get("uniprot_id")}) != len([row for row in backgrounds if row.get("uniprot_id")]):
            errors.append(f"ratio{ratio} repeats a background UniProt accession")
        # There are 1,218 positives and ratio*1,218 total negatives.
        expected = 1218 * ratio
        actual_negative = sum(row.get("label") == "0" for row in label_rows)
        if actual_negative != expected:
            errors.append(f"ratio{ratio} has {actual_negative} negatives; expected {expected}")
        report["ratios"][f"1:{ratio}"] = {"samples": len(sample_ids), "positive": sum(row.get("label") == "1" for row in label_rows),
                                             "negative": actual_negative, "added_background": len(backgrounds),
                                             "chains": len(chain_rows), "sequences": len(seq_rows)}
        previous_ids = sample_ids
    report["valid"] = not errors
    target = args.json or (args.cohorts / "validation.json")
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

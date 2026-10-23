from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")

    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    allowed = {"NNK1", "NNK2", "NNK3"}
    unexpected = sorted({row["dataset"] for row in rows} - allowed)
    if unexpected:
        raise ValueError(f"unexpected datasets: {unexpected}")

    rng = random.Random(args.seed)
    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(row["dataset"], int(float(row["seed_bh"])))].append(index)
    test_indices = set()
    for key, indices in sorted(strata.items()):
        rng.shuffle(indices)
        holdout_n = max(1, round(len(indices) * args.fraction))
        test_indices.update(indices[:holdout_n])

    columns = ["aa_seq", "seed_bh", "dataset", "train_test"]
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            {
                "aa_seq": row["aa_seq"],
                "seed_bh": row["seed_bh"],
                "dataset": row["dataset"],
                "train_test": "test" if index in test_indices else "train",
            }
        )
    output_rows.sort(key=lambda row: (row["dataset"], row["train_test"], row["aa_seq"], row["seed_bh"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)

    labels_by_split_sequence = defaultdict(set)
    for row in output_rows:
        if row["aa_seq"]:
            labels_by_split_sequence[(row["train_test"], row["aa_seq"])].add(
                int(float(row["seed_bh"]))
            )
    train_sequences = {
        sequence for split, sequence in labels_by_split_sequence if split == "train"
    }
    test_sequences = {
        sequence for split, sequence in labels_by_split_sequence if split == "test"
    }
    overlap = train_sequences & test_sequences
    conflicts = sum(
        labels_by_split_sequence[("train", sequence)]
        != labels_by_split_sequence[("test", sequence)]
        for sequence in overlap
    )
    summary = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "seed": args.seed,
        "test_fraction_target": args.fraction,
        "row_n": len(output_rows),
        "by_dataset_split_label": {
            "|".join(map(str, key)): value
            for key, value in sorted(
                Counter(
                    (row["dataset"], row["train_test"], int(float(row["seed_bh"])))
                    for row in output_rows
                ).items()
            )
        },
        "train_test_exact_sequence_overlap_n": len(overlap),
        "train_test_overlap_label_conflict_n": conflicts,
        "empty_sequence_n": sum(not row["aa_seq"].strip() for row in output_rows),
    }
    summary_path = args.output.with_suffix(".split.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

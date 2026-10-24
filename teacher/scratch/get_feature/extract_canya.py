from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
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
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from teacher.scratch.get_feature.clean_canya import iter_sheet, load_shared_strings, load_sheet_targets

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.input) as workbook:
        shared = load_shared_strings(workbook)
        targets = load_sheet_targets(workbook)
        rows = list(iter_sheet(workbook, targets["SD_2"], shared))

    columns = ["aa_seq", "seed_bh", "dataset", "train_test"]
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)

    summary = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "by_dataset_split": {
            "|".join(key): value
            for key, value in sorted(
                Counter((row.get("dataset", ""), row.get("train_test", "")) for row in rows).items()
            )
        },
    }
    summary_path = args.output.with_suffix(".extraction.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

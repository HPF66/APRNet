from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
COL_RE = re.compile(r"[A-Z]+")
STOP_RE = re.compile(r"\*")
NULLS = {"", "NA", "N/A", "NaN", "nan", "null", "None"}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def col_number(ref: str) -> int:
    letters = COL_RE.match(ref).group(0)
    result = 0
    for char in letters:
        result = result * 26 + ord(char) - 64
    return result


def text_content(element: ET.Element) -> str:
    return "".join(element.itertext())


def parse_float(value: str):
    if value is None or value.strip() in NULLS:
        return None
    try:
        parsed = float(value.replace(",", ""))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def load_shared_strings(book: zipfile.ZipFile) -> list[str]:
    try:
        with book.open("xl/sharedStrings.xml") as stream:
            root = ET.parse(stream).getroot()
    except KeyError:
        return []
    return [text_content(node) for node in root if local_name(node.tag) == "si"]


def load_sheet_targets(book: zipfile.ZipFile) -> dict[str, str]:
    with book.open("xl/workbook.xml") as stream:
        workbook = ET.parse(stream).getroot()
    with book.open("xl/_rels/workbook.xml.rels") as stream:
        relationships = ET.parse(stream).getroot()
    targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
    result = {}
    for node in workbook.iter():
        if local_name(node.tag) != "sheet":
            continue
        rel_id = node.attrib[f"{{{NS_REL}}}id"]
        target = targets[rel_id].replace("\\", "/")
        if not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        result[node.attrib["name"]] = target
    return result


def decode_cell(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = next((child for child in cell if local_name(child.tag) == "v"), None)
    value = "" if value_node is None else (value_node.text or "")
    if cell_type == "s" and value.isdigit():
        index = int(value)
        return shared[index] if index < len(shared) else value
    if cell_type == "inlineStr":
        inline = next((child for child in cell if local_name(child.tag) == "is"), None)
        return "" if inline is None else text_content(inline)
    return value


def iter_sheet(book: zipfile.ZipFile, target: str, shared: list[str]):
    with book.open(target) as stream:
        header = None
        for event, element in ET.iterparse(stream, events=("end",)):
            if local_name(element.tag) != "row":
                continue
            cells = {}
            for cell in element:
                if local_name(cell.tag) != "c":
                    continue
                ref = cell.attrib.get("r", "")
                if ref:
                    cells[col_number(ref)] = decode_cell(cell, shared)
            width = max(cells, default=0)
            values = [cells.get(index, "") for index in range(1, width + 1)]
            if header is None:
                header = values
                element.clear()
                continue
            row = {header[index]: values[index] if index < len(values) else "" for index in range(len(header))}
            yield row
            element.clear()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_label(row: dict, fitness_missing: bool) -> int:
    seed = parse_float(row.get("seed_bh", ""))
    if seed is not None:
        return 1 if seed > 0 else 0
    if fitness_missing:
        return 0
    # SD_1 includes seed_bh; this fallback is only for malformed rows.
    return 1 if (parse_float(row.get("fitness", "")) or 0) > 0 else 0


def main(input_path: str, output_dir: str):
    input_file = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_file) as book:
        shared = load_shared_strings(book)
        targets = load_sheet_targets(book)

        source_rows = list(iter_sheet(book, targets["SD_1"], shared))
        count_maps = {}
        duplicate_count_keys = {}
        for dataset in ("NNK1", "NNK2", "NNK3", "NNK4"):
            sheet_name = f"SD_7_{dataset}"
            counts = {}
            duplicates = 0
            for row in iter_sheet(book, targets[sheet_name], shared):
                key = row.get("aa_seq", "")
                if key in counts:
                    duplicates += 1
                counts[key] = row
            count_maps[dataset] = counts
            duplicate_count_keys[dataset] = duplicates

    stats = Counter()
    dataset_counts = Counter()
    missing_count_datasets = Counter()
    input_count_values = []
    excluded_tie_datasets = Counter()
    joined = []
    for row in source_rows:
        stats["source_rows"] += 1
        sequence = row.get("aa_seq", "")
        dataset = row.get("dataset", "")
        dataset_counts[dataset] += 1
        counts = count_maps.get(dataset, {}).get(sequence)
        input_count = parse_float((counts or {}).get("incount", ""))
        out_median = parse_float((counts or {}).get("outcount_median", ""))
        out_mean = parse_float((counts or {}).get("outcount_mean", ""))
        fitness_missing = parse_float(row.get("fitness", "")) is None

        if input_count is not None:
            input_count_values.append(input_count)
        elif dataset in count_maps:
            missing_count_datasets[dataset] += 1

        if not sequence:
            stats["excluded_blank_sequence"] += 1
            continue
        if dataset in count_maps and input_count is not None and input_count < 100:
            stats["excluded_input_reads_lt_100"] += 1
            continue
        if dataset in count_maps and input_count is None:
            stats["excluded_missing_input_reads"] += 1
            continue

        sequence_clean = sequence.split("*", 1)[0] if "*" in sequence else sequence
        has_stop = "*" in sequence
        if has_stop:
            stats["stop_codon_sequences"] += 1
        label = choose_label(row, fitness_missing)
        no_output = fitness_missing or (out_median == 0 and out_mean == 0)
        if no_output:
            label = 0
            stats["no_output_reads_labeled_nonaggregator"] += 1

        joined.append({
            "aa_seq": sequence,
            "aa_seq_clean": sequence_clean,
            "dataset": dataset,
            "seed_bh": row.get("seed_bh", ""),
            "fitness": row.get("fitness", ""),
            "sigma": row.get("sigma", ""),
            "input_count": "" if input_count is None else str(input_count),
            "outcount_median": "" if out_median is None else str(out_median),
            "outcount_mean": "" if out_mean is None else str(out_mean),
            "aggregator_label": str(label),
            "label_text": "aggregator" if label else "non-aggregator",
            "has_stop_codon": "1" if has_stop else "0",
            "fitness_missing": "1" if fitness_missing else "0",
        })

    # Only stop-codon truncation can create the duplicate sequences described in the paper.
    grouped = defaultdict(list)
    for record in joined:
        if record["has_stop_codon"] == "1":
            grouped[(record["dataset"], record["aa_seq_clean"])].append(record)

    by_identity = {(record["dataset"], record["aa_seq_clean"]): record for record in joined if record["has_stop_codon"] == "0"}
    cleaned = []
    consumed_stop = set()
    for record in joined:
        key = (record["dataset"], record["aa_seq_clean"])
        if record["has_stop_codon"] == "0":
            record["cleaning_status"] = "kept"
            record["source_rows_merged"] = "1"
            cleaned.append(record)
            continue
        if key in consumed_stop:
            continue
        consumed_stop.add(key)
        group = grouped[key]
        labels = [int(item["aggregator_label"]) for item in group]
        if len(group) > 1:
            stats["truncated_duplicate_groups"] += 1
            if sum(labels) * 2 == len(labels):
                stats["excluded_truncated_label_ties"] += len(group)
                excluded_tie_datasets[group[0]["dataset"]] += len(group)
                continue
            record = dict(group[0])
            numeric_fitness = [parse_float(item["fitness"]) for item in group]
            numeric_fitness = [value for value in numeric_fitness if value is not None]
            if numeric_fitness:
                record["fitness"] = str(sum(numeric_fitness) / len(numeric_fitness))
            record["aggregator_label"] = str(1 if sum(labels) > len(labels) / 2 else 0)
            record["label_text"] = "aggregator" if record["aggregator_label"] == "1" else "non-aggregator"
            record["cleaning_status"] = "stop_truncated_and_merged"
            record["source_rows_merged"] = str(len(group))
            stats["rows_merged_after_stop_truncation"] += len(group)
            cleaned.append(record)
        else:
            record = dict(group[0])
            record["cleaning_status"] = "stop_truncated"
            record["source_rows_merged"] = "1"
            cleaned.append(record)

    cleaned_dataset_counts = Counter()
    aggregator_dataset_counts = Counter()
    for record in cleaned:
        stats["cleaned_rows"] += 1
        stats["aggregator_rows"] += int(record["aggregator_label"])
        cleaned_dataset_counts[record["dataset"]] += 1
        aggregator_dataset_counts[record["dataset"]] += int(record["aggregator_label"])

    fieldnames = [
        "aa_seq", "aa_seq_clean", "dataset", "seed_bh", "fitness", "sigma",
        "input_count", "outcount_median", "outcount_mean", "aggregator_label",
        "label_text", "has_stop_codon", "fitness_missing", "cleaning_status",
        "source_rows_merged",
    ]
    tsv_path = out_dir / "adt5111_cleaned_SD1.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)

    log = {
        "source": str(input_file),
        "source_sha256": sha256(input_file),
        "rules": {
            "remove_input_reads_lt": 100,
            "missing_output_reads_label": "non-aggregator",
            "stop_codon_handling": "keep_prefix_before_first_stop",
            "stop_truncated_duplicates": "mean_fitness_and_majority_label",
            "stop_truncated_label_ties": "discard",
        },
        "source_sheet": "SD_1",
        "count_sheets": ["SD_7_NNK1", "SD_7_NNK2", "SD_7_NNK3", "SD_7_NNK4"],
        "stats": dict(stats),
        "count_sheet_duplicate_rows": duplicate_count_keys,
        "source_dataset_counts": dict(dataset_counts),
        "cleaned_dataset_counts": dict(cleaned_dataset_counts),
        "aggregator_dataset_counts": dict(aggregator_dataset_counts),
        "missing_count_datasets": dict(missing_count_datasets),
        "excluded_tie_datasets": dict(excluded_tie_datasets),
        "input_count_min": min(input_count_values) if input_count_values else None,
        "input_count_max": max(input_count_values) if input_count_values else None,
        "input_count_lt_100_observed": sum(1 for value in input_count_values if value < 100),
        "output": tsv_path.name,
        "output_sha256": sha256(tsv_path),
    }
    with (out_dir / "cleaning_log.json").open("w", encoding="utf-8") as stream:
        json.dump(log, stream, ensure_ascii=False, indent=2)

    with (out_dir / "cleaning_summary.txt").open("w", encoding="utf-8") as stream:
        stream.write("CANYA dataset cleaning summary\n")
        stream.write("================================\n")
        stream.write(f"Source sheet: SD_1\nSource rows: {stats['source_rows']}\n")
        stream.write(f"Cleaned rows: {stats['cleaned_rows']}\n")
        for key, value in sorted(stats.items()):
            if key not in {"source_rows", "cleaned_rows"}:
                stream.write(f"{key}: {value}\n")
        stream.write("\nThe original workbook is preserved separately and was not modified.\n")

    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: clean_canya_dataset.py INPUT_XLSX OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2])

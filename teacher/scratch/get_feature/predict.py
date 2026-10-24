from __future__ import annotations

import argparse
import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from teacher.utils.runtime import build_inference_model as build_model


VALID_SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]+")


def normalize_sequence(value: str) -> str:
    sequence = re.sub(r"\s+", "", value).upper()
    if not sequence or VALID_SEQUENCE.fullmatch(sequence) is None:
        raise ValueError(
            f"Invalid peptide sequence {value!r}; use one-letter amino-acid codes or X"
        )
    return sequence


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    record_id: str | None = None
    parts: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if record_id is not None:
                records.append((record_id, normalize_sequence("".join(parts))))
            record_id = line[1:].strip() or f"record_{len(records) + 1}"
            parts = []
        elif record_id is None:
            raise ValueError(f"FASTA sequence before header at {path}:{line_number}")
        else:
            parts.append(line)
    if record_id is not None:
        records.append((record_id, normalize_sequence("".join(parts))))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def checkpoint_args(values: dict[str, Any]) -> Namespace:
    path_names = {
        "data_root",
        "output_root",
        "split_manifest",
        "sequence_model_path",
        "peptide_transfer_checkpoint",
        "embedding_cache_dir",
        "residue_feature_table",
        "apr_weak_negatives",
        "peptide_auxiliary_workbook",
        "apr_positive_contrastive_workbook",
        "structure_peptide_evidence",
    }
    normalized = dict(values)
    for name in path_names:
        value = normalized.get(name)
        normalized[name] = Path(value) if value else None
    return Namespace(**normalized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trained Peptide classifier and its residue-level propensity head"
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[2] / "pretrained/teacher_canya_cpad.pt")
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="One peptide sequence; repeat this option for multiple peptides",
    )
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sequence-model-path",
        type=Path,
        help="Override the ESM2 directory stored in the checkpoint",
    )
    parser.add_argument(
        "--residue-feature-table",
        type=Path,
        help="Override the residue descriptor table stored in the checkpoint",
    )
    args = parser.parse_args()
    if not args.sequence and args.fasta is None:
        parser.error("provide --sequence or --fasta")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    records = [
        (f"sequence_{index}", normalize_sequence(sequence))
        for index, sequence in enumerate(args.sequence, start=1)
    ]
    if args.fasta is not None:
        records.extend(read_fasta(args.fasta))

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload or "args" not in payload:
        raise ValueError(
            "Expected a full-data Peptide checkpoint containing model and args"
        )
    training_args = checkpoint_args(payload["args"])
    if args.sequence_model_path is not None:
        training_args.sequence_model_path = args.sequence_model_path
    if args.residue_feature_table is not None:
        training_args.residue_feature_table = args.residue_feature_table

    model = build_model(training_args, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            sequences = [sequence for _, sequence in batch]
            output = model.forward_sequence(sequences)
            probabilities = torch.sigmoid(output["peptide_logits"]).cpu()
            main_probabilities = torch.sigmoid(output["peptide_main_logits"]).cpu()
            propensity_probabilities = torch.sigmoid(
                output["propensity_peptide_logits"]
            ).cpu()
            residue_probabilities = torch.sigmoid(
                output["aggregation_propensity_logits"]
            ).cpu()
            for row, ((record_id, sequence), probability) in enumerate(
                zip(batch, probabilities)
            ):
                predictions.append(
                    {
                        "id": record_id,
                        "sequence": sequence,
                        "amyloid_aggregation_probability": float(probability),
                        "predicted_label_at_0_5": int(probability >= 0.5),
                        "main_head_probability": float(main_probabilities[row]),
                        "propensity_bag_probability": float(
                            propensity_probabilities[row]
                        ),
                        "residue_propensity": [
                            {
                                "position": position,
                                "residue": residue,
                                "probability": float(residue_probabilities[row, position - 1]),
                            }
                            for position, residue in enumerate(sequence, start=1)
                        ],
                    }
                )

    result = {
        "checkpoint": str(args.checkpoint),
        "decision_threshold": 0.5,
        "prediction_n": len(predictions),
        "predictions": predictions,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from teacher.train.cross_validate import build_model, make_loader, parse_args as parse_cross_validate_args, set_seed, train_one_fold
from teacher.my_loader.data import AggregatingPeptideDataset, collate_peptides


PATH_ARGUMENTS = {
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


def load_training_args(config_path: Path) -> Namespace:
    stored_values = json.loads(config_path.read_text(encoding="utf-8"))
    original_argv = sys.argv
    try:
        sys.argv = ["cross_validate.py"]
        values = vars(parse_cross_validate_args())
    finally:
        sys.argv = original_argv
    values.update(stored_values)
    for name in PATH_ARGUMENTS:
        value = values.get(name)
        values[name] = Path(value) if value else None
    values["peptide_mutation_pair_lengths_tuple"] = tuple(
        int(item.strip())
        for item in values.get("peptide_mutation_pair_lengths", "6").split(",")
        if item.strip()
    )
    return Namespace(**values)


def selected_epoch_median(fold_results_path: Path) -> tuple[int, list[int]]:
    with fold_results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["selected_epoch"]) for row in rows]
    if not epochs:
        raise ValueError("Cross-validation results contain no selected_epoch values")
    return int(round(float(np.median(epochs)))), epochs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one final Peptide transfer checkpoint on all labeled peptides"
    )
    parser.add_argument("--cv-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data/model construction without optimizer updates",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=None,
        help="Validation-only override when fold_results.csv is not complete",
    )
    args = parser.parse_args()

    training_args = load_training_args(args.cv_run / "config.json")
    fold_results = args.cv_run / "fold_results.csv"
    if fold_results.is_file():
        epochs, fold_epochs = selected_epoch_median(fold_results)
    elif args.dry_run and args.epochs_override is not None:
        epochs, fold_epochs = args.epochs_override, []
    else:
        raise FileNotFoundError(f"Cross-validation results not found: {fold_results}")
    training_args.early_stopping = False
    training_args.device = args.device
    device = torch.device(args.device)
    set_seed(training_args.seed)

    dataset = AggregatingPeptideDataset(
        training_args.data_root / "aggregating_peptides_model_database.xlsx"
    )
    indices = np.arange(len(dataset), dtype=np.int64)
    loader = make_loader(
        dataset,
        indices,
        training_args.batch_size,
        collate_peptides,
        True,
        training_args.num_workers,
        seed=training_args.seed + 10000,
    )
    model = build_model(training_args, device)
    if args.dry_run:
        result = {
            "status": "dry_run_pass",
            "training_record_n": len(dataset),
            "final_epochs": epochs,
            "configured_schedule_horizon_epochs": training_args.epochs,
            "learning_rate": training_args.learning_rate,
            "sequence_backbone_learning_rate": (
                training_args.sequence_backbone_learning_rate
            ),
            "learning_rate_schedule": training_args.learning_rate_schedule,
            "peptide_local_propensity": training_args.peptide_local_propensity,
            "model_parameter_n": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
        print(json.dumps(result, indent=2), flush=True)
        return
    statistics, _, _ = train_one_fold(
        model,
        {"peptide": loader},
        training_args,
        device,
        structure_pos_weight=None,
        epochs=epochs,
        phase="final_peptide_full_data",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(training_args),
        "source_cv_run": str(args.cv_run),
        "selected_epoch_rule": "rounded_median_of_outer_fold_inner_validation_epochs",
        "fold_selected_epochs": fold_epochs,
        "final_epochs": epochs,
        "training_record_n": len(dataset),
        "training_statistics": statistics,
        "purpose": "single Peptide checkpoint for transfer initialization",
    }
    torch.save(payload, args.output)
    summary = {key: value for key, value in payload.items() if key not in {"model", "args"}}
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()

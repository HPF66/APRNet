from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Record:
    sequence: str
    label: int
    dataset: str
    split: str = ""


class SequenceDataset(Dataset):
    def __init__(self, records: list[Record]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class DatasetResidualHeads(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                dataset: nn.Sequential(
                    nn.LayerNorm(input_size),
                    nn.Linear(input_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                for dataset in ("NNK1", "NNK2", "NNK3")
            }
        )
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self, embedding, datasets, base_logits, return_residual_penalty=False
    ):
        logits = base_logits.clone()
        residual_square_sum = base_logits.new_zeros(())
        residual_count = 0
        for dataset, head in self.heads.items():
            indices = [index for index, value in enumerate(datasets) if value == dataset]
            if indices:
                tensor_indices = torch.tensor(indices, device=embedding.device)
                residual = head(embedding.index_select(0, tensor_indices)).squeeze(-1)
                residual = residual.to(base_logits.dtype)
                logits[tensor_indices] = logits[tensor_indices] + residual
                residual_square_sum = residual_square_sum + residual.square().sum()
                residual_count += residual.numel()
        if return_residual_penalty:
            return logits, residual_square_sum / max(1, residual_count)
        return logits


def collate(records: list[Record]):
    return {
        "sequences": [record.sequence for record in records],
        "labels": torch.tensor([record.label for record in records], dtype=torch.float32),
        "datasets": [record.dataset for record in records],
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path, official_split: bool = False):
    rows = []
    skipped_empty = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            sequence_column = "aa_seq" if official_split else "aa_seq_clean"
            label_column = "seed_bh" if official_split else "aggregator_label"
            sequence = row[sequence_column].strip().upper()
            if not sequence:
                skipped_empty += 1
                continue
            rows.append(
                Record(
                    sequence,
                    int(float(row[label_column])),
                    row["dataset"],
                    row.get("train_test", ""),
                )
            )
    return rows, skipped_empty


def consensus_records(rows: list[Record], datasets: set[str], excluded_sequences: set[str]):
    labels = defaultdict(list)
    for row in rows:
        if row.dataset in datasets and row.sequence not in excluded_sequences:
            labels[row.sequence].append(row.label)
    records = []
    conflict_n = 0
    duplicate_observation_n = 0
    for sequence, values in labels.items():
        duplicate_observation_n += max(0, len(values) - 1)
        if min(values) != max(values):
            conflict_n += 1
            continue
        records.append(Record(sequence, values[0], "+".join(sorted(datasets))))
    records.sort(key=lambda record: record.sequence)
    return records, {
        "unique_sequence_n": len(labels),
        "consensus_record_n": len(records),
        "conflicting_sequence_n": conflict_n,
        "duplicate_observation_n": duplicate_observation_n,
    }


def single_dataset_records(rows: list[Record], dataset: str, excluded_sequences: set[str]):
    selected = [
        row for row in rows if row.dataset == dataset and row.sequence not in excluded_sequences
    ]
    selected.sort(key=lambda record: record.sequence)
    return selected


def stratified_holdout(
    records: list[Record], fraction: float, seed: int, by_dataset: bool = False
):
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = random.Random(seed)
    train = []
    validation = []
    strata = defaultdict(list)
    for record in records:
        key = (record.dataset, record.label) if by_dataset else record.label
        strata[key].append(record)
    for values in strata.values():
        rng.shuffle(values)
        holdout_n = max(1, round(len(values) * fraction))
        validation.extend(values[:holdout_n])
        train.extend(values[holdout_n:])
    train.sort(key=lambda record: record.sequence)
    validation.sort(key=lambda record: record.sequence)
    return train, validation


def mcc_from_counts(tp: int, fp: int, tn: int, fn: int):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float):
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    tn = int(np.sum(~predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": int(labels.size),
        "threshold": float(threshold),
        "mcc": mcc_from_counts(tp, fp, tn, fn),
        "accuracy": (tp + tn) / labels.size,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0,
        "positive_n": int(np.sum(positives)),
        "predicted_positive_n": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray):
    from sklearn.metrics import average_precision_score, roc_auc_score

    if np.unique(labels).size < 2:
        return {"auroc": None, "aupr": None}
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
    }


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray):
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    total_positive = int(np.sum(labels == 1))
    total_negative = int(labels.size - total_positive)
    tp = fp = 0
    best_mcc = 0.0
    best_threshold = 0.5
    index = 0
    while index < labels.size:
        probability = sorted_probabilities[index]
        end = index
        while end < labels.size and sorted_probabilities[end] == probability:
            if sorted_labels[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        fn = total_positive - tp
        tn = total_negative - fp
        value = mcc_from_counts(tp, fp, tn, fn)
        if value > best_mcc:
            best_mcc = value
            best_threshold = float(probability)
        index = end
    return best_threshold, best_mcc


@torch.no_grad()
def predict(model, loader, device, dataset_heads=None):
    model.eval()
    if dataset_heads is not None:
        dataset_heads.eval()
    labels = []
    shared = []
    dataset_auxiliary = []
    main = []
    propensity = []
    sequences = []
    datasets = []
    for batch in loader:
        output = model.forward_sequence(batch["sequences"])
        labels.extend(batch["labels"].numpy().astype(np.int64).tolist())
        sequences.extend(batch["sequences"])
        datasets.extend(batch["datasets"])
        shared_logits = output["peptide_logits"]
        auxiliary_logits = shared_logits
        if dataset_heads is not None:
            auxiliary_logits = dataset_heads(
                output["peptide_embedding"], batch["datasets"], shared_logits
            )
        shared.extend(torch.sigmoid(shared_logits).cpu().tolist())
        dataset_auxiliary.extend(torch.sigmoid(auxiliary_logits).cpu().tolist())
        main.extend(torch.sigmoid(output["peptide_main_logits"]).cpu().tolist())
        propensity.extend(torch.sigmoid(output["propensity_peptide_logits"]).cpu().tolist())
    return {
        "sequences": sequences,
        "datasets": datasets,
        "labels": np.asarray(labels, dtype=np.int64),
        # The formal prediction is the shared output that can be transferred to
        # unseen datasets. Keep the fused alias for compatibility with auditors.
        "shared": np.asarray(shared, dtype=np.float64),
        "fused": np.asarray(shared, dtype=np.float64),
        "dataset_auxiliary": np.asarray(dataset_auxiliary, dtype=np.float64),
        "main": np.asarray(main, dtype=np.float64),
        "propensity": np.asarray(propensity, dtype=np.float64),
    }


def checkpoint_args(values: dict, project_root: Path):
    from argparse import Namespace

    path_names = {
        "data_root", "output_root", "split_manifest", "sequence_model_path",
        "peptide_transfer_checkpoint", "embedding_cache_dir", "residue_feature_table",
        "apr_weak_negatives", "peptide_auxiliary_workbook",
        "apr_positive_contrastive_workbook", "structure_peptide_evidence",
    }
    normalized = dict(values)
    for name in path_names:
        value = normalized.get(name)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = project_root / path
            normalized[name] = path
        else:
            normalized[name] = None
    return Namespace(**normalized)


def build_from_checkpoint(args, device):
    from teacher.train.cross_validate import build_model

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    values = json.loads(args.compat_config.read_text(encoding="utf-8"))
    values.update(payload["args"])
    training_args = checkpoint_args(values, args.project_root)
    model = build_model(training_args, device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    allowed_missing = {
        "sequence.encoder.backbone.backbone.base_model.model.embeddings.position_embeddings.weight"
    }
    missing = set(incompatible.missing_keys)
    if missing not in (set(), allowed_missing) or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, training_args, incompatible


def configure_trainable(model, train_mode: str):
    from teacher.utils.transfer import is_peptide_transfer_parameter

    trainable_names = []
    for name, parameter in model.named_parameters():
        originally_trainable = parameter.requires_grad
        allowed = is_peptide_transfer_parameter(name)
        if train_mode == "head":
            allowed = allowed and "encoder.backbone" not in name
        parameter.requires_grad = bool(
            allowed if train_mode == "full" else originally_trainable and allowed
        )
        if parameter.requires_grad:
            trainable_names.append(name)
    if not trainable_names:
        raise RuntimeError("no trainable parameters")
    return trainable_names


def make_optimizer(
    model, head_lr: float, lora_lr: float, backbone_lr: float, weight_decay: float
):
    head = []
    lora = []
    backbone = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name.lower():
            lora.append(parameter)
        elif "encoder.backbone" in name:
            backbone.append(parameter)
        else:
            head.append(parameter)
    groups = []
    if head:
        groups.append({"params": head, "lr": head_lr, "name": "head"})
    if lora:
        groups.append({"params": lora, "lr": lora_lr, "name": "lora"})
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    return (
        torch.optim.AdamW(groups, weight_decay=weight_decay),
        len(head),
        len(lora),
        len(backbone),
    )


def write_predictions(path: Path, prediction, threshold: float):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["sequence", "dataset", "label", "fused_probability", "prediction"]
        )
        for sequence, dataset, label, probability in zip(
            prediction["sequences"], prediction["datasets"],
            prediction["labels"], prediction["fused"]
        ):
            writer.writerow(
                [sequence, dataset, int(label), float(probability),
                 int(probability >= threshold)]
            )


def per_dataset_metrics(prediction, threshold: float):
    datasets = np.asarray(prediction["datasets"])
    result = {}
    for dataset in sorted(set(prediction["datasets"])):
        mask = datasets == dataset
        result[dataset] = metrics(
            prediction["labels"][mask], prediction["fused"][mask], threshold
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compat-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-datasets", default="NNK1,NNK2")
    parser.add_argument("--official-split", action="store_true")
    parser.add_argument("--validation-dataset", default="NNK3")
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--test-dataset", default="NNK4")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--scheduler-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--dataset-heads", action="store_true")
    parser.add_argument("--dataset-head-lr", type=float, default=1e-3)
    parser.add_argument("--dataset-head-hidden-size", type=int, default=256)
    parser.add_argument("--dataset-head-dropout", type=float, default=0.1)
    parser.add_argument("--shared-loss-weight", type=float, default=1.0)
    parser.add_argument("--dataset-aux-weight", type=float, default=0.3)
    parser.add_argument("--dataset-residual-penalty", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--propensity-weight", type=float, default=0.25)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--train-mode", choices=("head", "lora", "full"), default="lora"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args()

    if (
        not args.official_split
        and args.validation_dataset
        and args.validation_fraction
    ):
        parser.error("use either --validation-dataset or --validation-fraction, not both")

    project_root = str(args.project_root.resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    rows, skipped_empty = read_rows(args.input, args.official_split)
    if args.official_split:
        official_train = [row for row in rows if row.split == "train"]
        test_records = [row for row in rows if row.split == "test"]
        unknown_splits = Counter(row.split for row in rows) - Counter(
            {"train": len(official_train), "test": len(test_records)}
        )
        if unknown_splits:
            raise ValueError(f"unexpected official split labels: {unknown_splits}")
        train_sequences = {row.sequence for row in official_train}
        test_sequences = {row.sequence for row in test_records}
        overlap_sequences = train_sequences & test_sequences
        train_labels = defaultdict(set)
        test_labels = defaultdict(set)
        for row in official_train:
            train_labels[row.sequence].add(row.label)
        for row in test_records:
            test_labels[row.sequence].add(row.label)
        overlap_conflicts = sum(
            train_labels[sequence] != test_labels[sequence]
            for sequence in overlap_sequences
        )
        if args.validation_fraction:
            train_records, validation_records = stratified_holdout(
                official_train, args.validation_fraction, args.seed, by_dataset=True
            )
            validation_sequences = {row.sequence for row in validation_records}
        else:
            train_records = official_train
            validation_records = []
            validation_sequences = set()
        train_audit = {
            "official_train_n": len(official_train),
            "official_test_n": len(test_records),
            "train_test_exact_sequence_overlap_n": len(overlap_sequences),
            "train_test_overlap_label_conflict_n": overlap_conflicts,
            "official_train_by_dataset": dict(Counter(row.dataset for row in official_train)),
            "official_test_by_dataset": dict(Counter(row.dataset for row in test_records)),
        }
        test_sequences_removed_n = 0
    else:
        test_sequences = {row.sequence for row in rows if row.dataset == args.test_dataset}
        validation_sequences = (
            {row.sequence for row in rows if row.dataset == args.validation_dataset}
            if args.validation_dataset
            else set()
        )
        train_datasets = {
            item.strip() for item in args.train_datasets.split(",") if item.strip()
        }
        train_records, train_audit = consensus_records(
            rows, train_datasets, test_sequences | validation_sequences
        )
        validation_records = (
            single_dataset_records(rows, args.validation_dataset, test_sequences)
            if args.validation_dataset
            else []
        )
        if args.validation_fraction:
            train_records, validation_records = stratified_holdout(
                train_records, args.validation_fraction, args.seed
            )
        test_records = single_dataset_records(rows, args.test_dataset, set())
        test_sequences_removed_n = len(test_sequences)

    sampling = random.Random(args.seed)
    if args.train_limit and args.train_limit < len(train_records):
        train_records = sampling.sample(train_records, args.train_limit)
    if args.validation_limit and args.validation_limit < len(validation_records):
        validation_records = sampling.sample(validation_records, args.validation_limit)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SequenceDataset(train_records), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True, generator=generator,
    )
    validation_loader = (
        DataLoader(
            SequenceDataset(validation_records), batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        )
        if validation_records else None
    )
    test_loader = DataLoader(
        SequenceDataset(test_records), batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
    )

    device = torch.device(args.device)
    model, training_args, incompatible = build_from_checkpoint(args, device)
    trainable_names = configure_trainable(model, args.train_mode)
    (
        optimizer,
        head_parameter_tensors,
        lora_parameter_tensors,
        backbone_parameter_tensors,
    ) = make_optimizer(
        model, args.head_lr, args.lora_lr, args.backbone_lr, args.weight_decay
    )
    dataset_heads = None
    dataset_head_parameter_n = 0
    if args.dataset_heads:
        model.eval()
        with torch.no_grad():
            sample_output = model.forward_sequence([train_records[0].sequence])
        dataset_heads = DatasetResidualHeads(
            input_size=int(sample_output["peptide_embedding"].shape[-1]),
            hidden_size=args.dataset_head_hidden_size,
            dropout=args.dataset_head_dropout,
        ).to(device)
        optimizer.add_param_group(
            {
                "params": list(dataset_heads.parameters()),
                "lr": args.dataset_head_lr,
                "name": "dataset_heads",
            }
        )
        dataset_head_parameter_n = sum(
            parameter.numel() for parameter in dataset_heads.parameters()
        )
    scheduler_epochs = args.scheduler_epochs or args.epochs
    total_steps = max(1, scheduler_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    positive_weight = torch.tensor(args.pos_weight, device=device)

    config = {
        **vars(args),
        "project_root": str(args.project_root),
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "compat_config": str(args.compat_config),
        "output_dir": str(args.output_dir),
        "skipped_empty_n": skipped_empty,
        "train_n": len(train_records),
        "validation_n": len(validation_records),
        "test_n": len(test_records),
        "train_audit": train_audit,
        "trainable_parameter_tensor_n": len(trainable_names),
        "trainable_parameter_n": (
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            + dataset_head_parameter_n
        ),
        "dataset_head_parameter_n": dataset_head_parameter_n,
        "head_parameter_tensor_n": head_parameter_tensors,
        "lora_parameter_tensor_n": lora_parameter_tensors,
        "backbone_parameter_tensor_n": backbone_parameter_tensors,
        "test_sequences_removed_from_training": test_sequences_removed_n,
        "validation_sequences_removed_from_training": len(validation_sequences),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(config, indent=2, default=str), flush=True)

    history = []
    best_val_mcc = -1.0
    best_epoch = args.epochs
    best_threshold = args.fixed_threshold if args.fixed_threshold is not None else 0.5
    patience_left = args.patience
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if dataset_heads is not None:
            dataset_heads.train()
        loss_sum = shared_sum = dataset_aux_sum = propensity_sum = residual_sum = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                output = model.forward_sequence(batch["sequences"])
                shared_logits = output["peptide_logits"]
                dataset_logits = shared_logits
                residual_penalty = shared_logits.new_zeros(())
                if dataset_heads is not None:
                    dataset_logits, residual_penalty = dataset_heads(
                        output["peptide_embedding"], batch["datasets"], shared_logits,
                        return_residual_penalty=True,
                    )
                shared_loss = F.binary_cross_entropy_with_logits(
                    shared_logits, labels, pos_weight=positive_weight
                )
                dataset_aux_loss = F.binary_cross_entropy_with_logits(
                    dataset_logits, labels, pos_weight=positive_weight
                )
                propensity_loss = F.binary_cross_entropy_with_logits(
                    output["propensity_peptide_logits"], labels, pos_weight=positive_weight
                )
                loss = (
                    args.shared_loss_weight * shared_loss
                    + args.dataset_aux_weight * dataset_aux_loss
                    + args.propensity_weight * propensity_loss
                    + args.dataset_residual_penalty * residual_penalty
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            parameters_to_clip = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if dataset_heads is not None:
                parameters_to_clip.extend(dataset_heads.parameters())
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            batch_n = labels.numel()
            seen += batch_n
            loss_sum += float(loss.detach()) * batch_n
            shared_sum += float(shared_loss.detach()) * batch_n
            dataset_aux_sum += float(dataset_aux_loss.detach()) * batch_n
            propensity_sum += float(propensity_loss.detach()) * batch_n
            residual_sum += float(residual_penalty.detach()) * batch_n

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_shared_loss": shared_sum / seen,
            "train_dataset_aux_loss": dataset_aux_sum / seen,
            "train_propensity_loss": propensity_sum / seen,
            "train_residual_penalty": residual_sum / seen,
            "elapsed_seconds": time.time() - started,
        }
        if validation_loader is not None:
            validation = predict(model, validation_loader, device, dataset_heads)
            threshold, selected_mcc = select_mcc_threshold(
                validation["labels"], validation["fused"]
            )
            row["validation_at_0_5"] = metrics(
                validation["labels"], validation["fused"], 0.5
            )
            row["validation_selected"] = metrics(
                validation["labels"], validation["fused"], threshold
            )
            row["validation_ranking"] = ranking_metrics(
                validation["labels"], validation["fused"]
            )
            row["validation_by_dataset"] = per_dataset_metrics(
                validation, threshold
            )
            row["validation_dataset_auxiliary"] = metrics(
                validation["labels"], validation["dataset_auxiliary"], threshold
            )
            row["validation_dataset_auxiliary_ranking"] = ranking_metrics(
                validation["labels"], validation["dataset_auxiliary"]
            )
            if selected_mcc > best_val_mcc + 1e-6:
                best_val_mcc = selected_mcc
                best_epoch = epoch
                best_threshold = threshold
                patience_left = args.patience
                torch.save(
                    {
                        "model": model.state_dict(),
                        "dataset_heads": (
                            dataset_heads.state_dict()
                            if dataset_heads is not None else None
                        ),
                        "args": vars(training_args),
                        "canya_config": config,
                        "selected_epoch": best_epoch,
                        "decision_threshold": best_threshold,
                        "validation_metrics": row["validation_selected"],
                        "purpose": "shared-supervised transferable CANYA pretraining",
                    },
                    args.output_dir / "best_transfer.pt",
                )
            else:
                patience_left -= 1
        else:
            torch.save(
                {
                    "model": model.state_dict(),
                    "dataset_heads": (
                        dataset_heads.state_dict()
                        if dataset_heads is not None else None
                    ),
                    "args": vars(training_args),
                    "canya_config": config,
                    "selected_epoch": epoch,
                    "decision_threshold": best_threshold,
                    "purpose": "shared-supervised transferable CANYA pretraining",
                },
                args.output_dir / "best_transfer.pt",
            )
        history.append(row)
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
        if validation_loader is not None and patience_left <= 0:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(
        args.output_dir / "best_transfer.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    if dataset_heads is not None:
        dataset_heads.load_state_dict(checkpoint["dataset_heads"], strict=True)
        dataset_heads.to(device)
    summary = {
        "config": config,
        "best_epoch": checkpoint["selected_epoch"],
        "decision_threshold": checkpoint["decision_threshold"],
        "best_validation_mcc": best_val_mcc if validation_loader is not None else None,
        "history": history,
        "test_metrics": None,
    }
    if args.evaluate_test:
        test_prediction = predict(model, test_loader, device, dataset_heads)
        threshold = float(checkpoint["decision_threshold"])
        summary["test_metrics"] = {
            "fused": metrics(test_prediction["labels"], test_prediction["fused"], threshold),
            "fused_ranking": ranking_metrics(
                test_prediction["labels"], test_prediction["fused"]
            ),
            "fused_at_0_5": metrics(test_prediction["labels"], test_prediction["fused"], 0.5),
            "fused_by_dataset": per_dataset_metrics(test_prediction, threshold),
            "fused_ranking_by_dataset": {
                dataset: ranking_metrics(
                    test_prediction["labels"][np.asarray(test_prediction["datasets"]) == dataset],
                    test_prediction["fused"][np.asarray(test_prediction["datasets"]) == dataset],
                )
                for dataset in sorted(set(test_prediction["datasets"]))
            },
            "dataset_auxiliary": metrics(
                test_prediction["labels"],
                test_prediction["dataset_auxiliary"],
                threshold,
            ),
            "dataset_auxiliary_ranking": ranking_metrics(
                test_prediction["labels"], test_prediction["dataset_auxiliary"]
            ),
            "main_head": metrics(test_prediction["labels"], test_prediction["main"], threshold),
            "main_ranking": ranking_metrics(
                test_prediction["labels"], test_prediction["main"]
            ),
            "propensity_head": metrics(
                test_prediction["labels"], test_prediction["propensity"], threshold
            ),
            "propensity_ranking": ranking_metrics(
                test_prediction["labels"], test_prediction["propensity"]
            ),
        }
        write_predictions(
            args.output_dir / "test_predictions.tsv", test_prediction, threshold
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

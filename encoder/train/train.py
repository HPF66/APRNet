#!/usr/bin/env python3
"""Train Boundary-Aware Context V3 on CPAD data and fold manifests.

The default uses the C2_CS_E2R loss. Optional boundary detection and regional
SupCon can be enabled independently without changing the exported HAPR.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import encoder.utils.support as baseline
from encoder.model.residue_encoder import VARIANT, ContextEncoderConfig, ContextualAPRModel, load_encoder_checkpoint, model_metadata
import encoder.utils.objectives as representation_training
from encoder.utils.lineage import window_contract

trainer = baseline.trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("apr-workbook", "fold-csv", "peptide-window-cache", "esm-cache", "output-dir"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--variants", choices=(VARIANT,), default=VARIANT)
    p.add_argument("--folds", default=None)
    p.add_argument("--resume", action="store_true", help="Reuse completed folds with identical run signatures; interrupted folds restart.")
    for flag, default in (
        ("epochs", 50), ("patience", 10), ("minimum-epochs", 20), ("selection-start-epoch", 10),
        ("batch-size", 4), ("gradient-accumulation-steps", 1), ("max-length", 512), ("overlap", 64),
        ("hidden-size", 192), ("attention-heads", 4), ("context-radius", 64),
        ("context-layers", 2), ("context-ffn-multiplier", 4), ("checkpoint-interval", 0),
        ("seed", 2026), ("num-workers", 0),
        ("supcon-max-items", 256), ("supcon-boundary-radius", 3), ("supcon-projection-dim", 128),
    ):
        p.add_argument("--" + flag, type=int, default=default)
    for flag, default in (
        ("dropout", .15), ("learning-rate", 5e-5), ("weight-decay", 0.), ("pos-weight-power", .35),
        ("smoothness-weight", .005), ("local-prior-loss-weight", .25), ("threshold", .5),
        ("view-consistency-weight", .10), ("overlap-consistency-weight", .10),
        ("reliability-consistency-weight", .05),
        ("window-measurement-weight", .10), ("window-hidden-reconstruction-weight", .10),
        ("view-token-noise", .02), ("view-scale-dropout", .15),
        ("boundary-detection-weight", 0.), ("supcon-weight", 0.),
        ("supcon-temperature", .10), ("supcon-cross-protein-weight", .25),
    ):
        p.add_argument("--" + flag, type=float, default=default)
    p.add_argument("--within-label-similarity-weight", "--boundary-weight", dest="boundary_weight",
                   type=float, default=.03,
                   help="Legacy same-label HAPR attraction; NOT boundary detection. Set to 0 when replacing it.")
    p.add_argument("--cpad-weight", type=float, default=1.0,
                   help="Row-loss multiplier for CPADPEP samples; applied to the complete CPAD source.")
    p.add_argument("--supcon-space", choices=("projection", "hapr"), default="projection",
                   help="Train-only projection (default) or direct HAPR contrast for ablation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    config_from_args(args)  # Validate architecture before loading any caches.
    baseline.require(1 <= args.selection_start_epoch <= args.epochs,
                     "selection-start-epoch must be in [1, epochs]")
    baseline.require(1 <= args.minimum_epochs <= args.epochs, "minimum-epochs must be in [1, epochs]")
    baseline.require(args.patience >= 1 and args.batch_size >= 1 and args.gradient_accumulation_steps >= 1,
                     "patience, batch-size and gradient-accumulation-steps must be positive")
    baseline.require(0 <= args.overlap < args.max_length, "overlap must be in [0, max-length)")
    baseline.require(0 < args.threshold < 1 and 0 <= args.view_scale_dropout < 1,
                     "threshold must be in (0,1), view-scale-dropout in [0,1)")
    baseline.require(args.learning_rate > 0 and args.num_workers >= 0 and args.checkpoint_interval >= 0,
                     "learning-rate must be positive; workers/interval must be nonnegative")
    for key, value in vars(args).items():
        if key.endswith("_weight") or key in {"weight_decay", "view_token_noise", "pos_weight_power"}:
            baseline.require(value >= 0, f"{key} must be nonnegative")
    baseline.require(args.cpad_weight > 0, "cpad-weight must be positive")
    baseline.require(args.supcon_temperature > 0 and args.supcon_max_items >= 4,
                     "SupCon temperature must be positive and max-items at least 4")
    baseline.require(args.supcon_boundary_radius >= 0 and args.supcon_projection_dim >= 2
                     and 0 <= args.supcon_cross_protein_weight <= 1,
                     "Invalid SupCon radius, projection dimension or cross-protein weight")
    return args


def config_from_args(args: argparse.Namespace) -> ContextEncoderConfig:
    ordinary = {name: getattr(args, name) for name in ContextEncoderConfig.__dataclass_fields__
                if name not in {"boundary_head", "supcon_projection_dim"}}
    return ContextEncoderConfig(
        **ordinary, boundary_head=args.boundary_detection_weight > 0,
        supcon_projection_dim=args.supcon_projection_dim if args.supcon_weight > 0 and args.supcon_space == "projection" else 0,
    )


def objective_metadata(args: argparse.Namespace) -> dict:
    descriptions = ["protein-balanced APR BCE"]
    for weight, name in (
        ("local_prior_loss_weight", "local BCE"),
        ("smoothness_weight", "same-label probability smoothness"),
        ("view_consistency_weight", "view HAPR consistency"),
        ("overlap_consistency_weight", "overlap HAPR consistency"),
        ("boundary_weight", "same-label adjacent HAPR similarity"),
        ("reliability_consistency_weight", "reliability stability"),
        ("window_measurement_weight", "window occupancy reconstruction"),
        ("window_hidden_reconstruction_weight", "window hidden reconstruction"),
    ):
        if getattr(args, weight) > 0:
            descriptions.append(name)
    if args.boundary_detection_weight:
        descriptions.append("annotation-transition boundary BCE on adjacent HAPR pairs")
    if args.supcon_weight:
        descriptions.append(f"regional supervised contrast in {args.supcon_space} space")
    return {
        "description": " + ".join(descriptions),
        "pos_weight_power": args.pos_weight_power,
        "pos_weight_scope": "training chunks including overlaps; APR and local BCE",
        "within_label_similarity_weight": args.boundary_weight,
        "boundary_detection_weight": args.boundary_detection_weight,
        "boundary_target": "adjacent annotation labels differ; no synthetic chunk endpoints",
        "supcon_weight": args.supcon_weight,
        "supcon_space": args.supcon_space if args.supcon_weight else "disabled",
        "supcon_temperature": args.supcon_temperature,
        "supcon_cross_protein_weight": args.supcon_cross_protein_weight,
        "supcon_boundary_radius": args.supcon_boundary_radius,
        "cpad_weight": args.cpad_weight,
        "cpad_weight_scope": "all CPADPEP samples and all enabled row-weighted objectives",
    }


def run_signature(args: argparse.Namespace) -> str:
    """Prevent resume from silently mixing architectures, objectives or splits."""
    ignored = {"resume", "folds", "device", "num_workers", "output_dir"}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ignored}
    config["input_hashes"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.apr_workbook, args.fold_csv,
                     args.peptide_window_cache / "manifest.json", args.esm_cache / "manifest.json")
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def train_one_fold(train_raw, validation_raw, args, fold, device, checkpoint_path):
    signature = run_signature(args)
    train_ids = sorted(p.sample_id for p in train_raw)
    validation_ids = sorted(p.sample_id for p in validation_raw)
    baseline.require(train_ids and validation_ids and set(train_ids).isdisjoint(validation_ids),
                     "Training and validation must be nonempty and disjoint")
    if checkpoint_path.exists():
        baseline.require(args.resume, f"Checkpoint exists; use --resume or a new output directory: {checkpoint_path}")
        model, checkpoint = load_encoder_checkpoint(checkpoint_path, device)
        baseline.require(checkpoint.get("run_signature") == signature
                         and checkpoint.get("train_sample_ids") == train_ids
                         and checkpoint.get("validation_sample_ids") == validation_ids
                         and checkpoint.get("fold") == fold
                         and "selected_epoch" in checkpoint,
                         "Checkpoint configuration/split differs from this run")
        return model, checkpoint

    # Same initialization seed as E2R. Shared inherited weights initialize
    # before the replacement context, keeping the starting local encoder equal.
    trainer.set_seed(args.seed + fold + 4300)
    model = ContextualAPRModel(config_from_args(args)).to(device)
    proteins = {p.sample_id: p for p in train_raw}
    validation_proteins = {p.sample_id: p for p in validation_raw}
    store = trainer.load_store(train_raw, args.peptide_window_cache, args.esm_cache)
    validation_store = trainer.load_store(validation_raw, args.peptide_window_cache, args.esm_cache)
    baseline.zero_esm(store)
    baseline.zero_esm(validation_store)
    chunks = trainer.make_chunks(train_raw, args.max_length, args.overlap)
    validation_chunks = trainer.make_chunks(validation_raw, args.max_length, args.overlap)
    loader = trainer.make_loader(chunks, proteins, store, args.batch_size, True, args.seed + fold, args.num_workers)
    chunk_counts = {}
    for chunk in chunks:
        chunk_counts[chunk.sample_id] = chunk_counts.get(chunk.sample_id, 0) + 1
    pos_weight = trainer.calculate_pos_weight(chunks, proteins, device, args.pos_weight_power)
    resolved_pos_weight = float(pos_weight.detach().cpu())
    print(json.dumps({"event": "boundary_v3_class_balance", "fold": fold,
                      "pos_weight_power": args.pos_weight_power, "pos_weight": resolved_pos_weight,
                      "scope": "training_chunks_including_overlaps"}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_mcc, best_epoch, best_state, stale = -float("inf"), None, None, 0
    losses, loss_components = [], []
    metadata = {
        **model_metadata(model), "fold": fold, "run_signature": signature,
        "input_contract": window_contract(json.loads((args.peptide_window_cache / 'manifest.json').read_text(encoding='utf-8'))),
        "train_sample_ids": train_ids, "validation_sample_ids": validation_ids,
        "training_objectives": objective_metadata(args), "config": vars(args),
        "esm2_input": "independent residue ESM cache zeroed; frozen peptide teacher retained",
        "label_policy": "APR=1, all non-1 characters=0",
        "pos_weight": resolved_pos_weight,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
    }
    supcon_generator = torch.Generator().manual_seed(args.seed + fold + 9100)
    for epoch in range(1, args.epochs + 1):
        components = {}
        loss = representation_training.run_epoch(model, loader, device, optimizer, pos_weight, args,
                                                  chunk_counts, diagnostics=components,
                                                  supcon_generator=supcon_generator)
        losses.append(float(loss))
        loss_components.append(components)
        validation_prob = trainer.predict(model, validation_chunks, validation_proteins, validation_store, args, device)
        validation_metrics = baseline.metrics_with_threshold(validation_raw, validation_prob, args.threshold)
        score = float(validation_metrics["mcc"])
        baseline.require(np.isfinite(score), "Validation MCC is not finite")
        print(json.dumps({"event": "boundary_v3_epoch", "variant": VARIANT, "fold": fold, "epoch": epoch,
                          "train_loss": loss, "train_components": components,
                          "validation": validation_metrics}), flush=True)
        if args.checkpoint_interval and epoch % args.checkpoint_interval == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({**metadata, "epoch": epoch, "validation": validation_metrics,
                        "purpose": "unselected representation candidate; no outer-test selection",
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       checkpoint_path.with_name(f"fold_{fold}.epoch_{epoch}.pt"))
        if epoch >= args.selection_start_epoch:
            if score > best_mcc + 1e-4:
                best_mcc, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if epoch >= args.minimum_epochs and stale >= args.patience:
                    break
    baseline.require(best_state is not None, "No checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    model.eval().requires_grad_(False)
    checkpoint = {**metadata, "selected_epoch": best_epoch, "validation_mcc": best_mcc,
                  "losses": losses, "loss_components": loss_components, "state_dict": best_state}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return model, checkpoint


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    baseline.require(device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_available()),
                     "Select an available CPU or CUDA device")
    signature = run_signature(args)
    raw = [trainer.Protein(sid, seq, target.astype(np.int8), fold)
           for sid, seq, target, fold in baseline.peptide_adapter.read_proteins(args.apr_workbook, args.fold_csv)]
    all_folds = sorted({p.fold for p in raw})
    baseline.require(len(all_folds) >= 3, "Need distinct train, validation and test folds")
    baseline.require(len({p.sample_id for p in raw}) == len(raw), "Duplicate protein IDs in fold inputs")
    selected = all_folds if args.folds is None else [int(x) for x in args.folds.split(",") if x.strip()]
    baseline.require(selected and len(selected) == len(set(selected)) and set(selected).issubset(all_folds),
                     "Requested folds must be nonempty, unique and present in the manifest")
    output = args.output_dir / VARIANT
    results = []
    for fold in selected:
        result_path = output / "folds" / f"fold_{fold}.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            baseline.require(result.get("run_signature") == signature and result.get("variant") == VARIANT
                             and result.get("fold") == fold and Path(result["checkpoint"]).is_file(),
                             f"Cannot reuse incompatible/incomplete result: {result_path}")
            results.append(result)
            print(json.dumps({"event": "boundary_v3_fold_reused", "fold": fold}), flush=True)
            continue
        baseline.require(args.resume or not result_path.exists(), f"Result already exists: {result_path}")
        validation_fold = all_folds[(all_folds.index(fold) + 1) % len(all_folds)]
        train_raw = [p for p in raw if p.fold not in {fold, validation_fold}]
        validation_raw = [p for p in raw if p.fold == validation_fold]
        test_raw = [p for p in raw if p.fold == fold]
        checkpoint_path = output / "checkpoints" / f"fold_{fold}.pt"
        model, checkpoint = train_one_fold(train_raw, validation_raw, args, fold, device, checkpoint_path)
        test_store = trainer.load_store(test_raw, args.peptide_window_cache, args.esm_cache)
        baseline.zero_esm(test_store)
        predictions = trainer.predict(model, trainer.make_chunks(test_raw, args.max_length, args.overlap),
                                      {p.sample_id: p for p in test_raw}, test_store, args, device)
        result = {
            "variant": VARIANT, "fold": fold, "validation_fold": validation_fold, "run_signature": signature,
            "train_proteins": len(train_raw), "validation_proteins": len(validation_raw),
            "test_proteins": len(test_raw), "selected_epoch": checkpoint["selected_epoch"],
            "checkpoint": str(checkpoint_path),
            "metrics": baseline.metrics_with_threshold(test_raw, predictions, args.threshold),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(json.dumps({"event": "boundary_v3_fold_complete", **result}), flush=True)
        del model, checkpoint, test_store, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "format": "hapr_boundary_aware_context_v3_summary", "variant": VARIANT, "run_signature": signature,
        "model_config": asdict(config_from_args(args)), "training_objectives": objective_metadata(args),
        "outer_policy": "one encoder per outer fold; next fold validates; outer test never selects weights",
        "threshold": args.threshold, "folds": results, "config": vars(args),
    }
    for name in ("mcc", "auroc", "auprc", "f1", "accuracy", "sensitivity", "specificity",
                 "balanced_accuracy", "predicted_positive_rate"):
        values = np.asarray([r["metrics"][name] for r in results], dtype=np.float64)
        summary[name + "_mean"] = float(values.mean())
        summary[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"event": "boundary_v3_complete", "variant": VARIANT,
                      "mcc_mean": summary["mcc_mean"], "auprc_mean": summary["auprc_mean"]}), flush=True)


if __name__ == "__main__":
    main()

"""Full-data refit for Boundary-Aware Context V3.

The epoch budget is sealed from completed CV validation curves before this
script is started. The model is initialized from scratch and never uses a
fold checkpoint. The final artifact is a frozen model that can encode new
sequences on demand.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import encoder.utils.support as baseline
import encoder.train.train as train_v3
import encoder.utils.objectives as objectives
from encoder.utils.lineage import CANONICAL_PEPTIDE_CHECKPOINT, CANONICAL_PEPTIDE_SHA256, verify_window_manifest, window_contract, sha256
from encoder.model.residue_encoder import VARIANT, ContextualAPRModel, load_encoder_checkpoint, model_metadata
trainer = train_v3.trainer

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def rng_state(loader, supcon_generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'loader': loader.generator.get_state(), 'supcon': supcon_generator.get_state()}

def restore_rng(state, loader, supcon_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    loader.generator.set_state(state['loader'])
    supcon_generator.set_state(state['supcon'])

def build_args(root: Path, output: Path, epochs: int, device: str, *, window_cache: Path, base_cache: Path, workbook: Path | None = None, folds: Path | None = None) -> argparse.Namespace:
    return train_v3.parse_args(['--apr-workbook', str(workbook or root / 'data/cpad_full_v1/apr_mixed_unified_v1.xlsx'), '--fold-csv', str(folds or root / 'data/homology10/apr_folds.csv'), '--peptide-window-cache', str(window_cache), '--esm-cache', str(base_cache), '--output-dir', str(output), '--epochs', str(epochs), '--batch-size', '4', '--hidden-size', '192', '--boundary-detection-weight', '0.10', '--cpad-weight', '0.15', '--pos-weight-power', '0.35', '--supcon-weight', '0', '--device', device])

def load_full_data(args):
    tuples = baseline.peptide_adapter.read_proteins(args.apr_workbook, args.fold_csv)
    proteins = [trainer.Protein(sid, seq, target.astype(np.int8), fold) for sid, seq, target, fold in tuples]
    require(len(proteins) == 3031, f'Expected 3031 sequences, found {len(proteins)}')
    require(len({p.sample_id for p in proteins}) == len(proteins), 'Duplicate sample IDs in refit input')
    return proteins

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workbook', type=Path)
    parser.add_argument('--fold-csv', type=Path)
    parser.add_argument('--window-cache', type=Path, required=True)
    parser.add_argument('--base-cache', type=Path, required=True)
    parser.add_argument('--teacher-checkpoint', type=Path, default=CANONICAL_PEPTIDE_CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args_cli = parser.parse_args()
    require(args_cli.epochs >= 1, 'epochs must be positive')
    root = args_cli.root.resolve()
    output = args_cli.output.resolve()
    args = build_args(root, output, args_cli.epochs, args_cli.device,
                      window_cache=args_cli.window_cache.resolve(), base_cache=args_cli.base_cache.resolve(),
                      workbook=args_cli.workbook, folds=args_cli.fold_csv)
    manifest = verify_window_manifest(args.peptide_window_cache, args_cli.teacher_checkpoint)
    input_contract = window_contract(manifest)
    output.mkdir(parents=True, exist_ok=True)
    plan = {'format': 'hapr_boundary_aware_context_v3_full_data_refit_plan_v1', 'variant': VARIANT, 'selection_rule': 'homology10 mean validation MCC peak over common observed epoch horizon', 'selection_split': 'homology10', 'selection_metric': 'validation_mcc', 'e_best': args_cli.epochs, 'random10_reference_peak_epoch': 19, 'data': 'cpad_full_v1 unified APR workbook; all 3031 sequences', 'peptide_teacher_lineage': 'CAnYA shared-supervised checkpoint -> CPAD fine-tuning -> APR transfer', 'peptide_teacher_checkpoint': str(args_cli.teacher_checkpoint), 'peptide_teacher_sha256': CANONICAL_PEPTIDE_SHA256, 'peptide_window_cache': str(args.peptide_window_cache), 'input_contract': input_contract, 'window_manifest_sha256': sha256(args.peptide_window_cache / 'manifest.json'), 'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'initialization': 'fresh ContextualAPRModel; no fold state_dict loaded', 'training': 'fixed full-data epochs; no refit validation or early stopping', 'output_contract': 'frozen model computes HAPR [L,192] for each new sequence at inference time'}
    plan['plan_signature'] = digest(plan)
    plan_path = output / 'refit_plan.json'
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding='utf-8'))
        require(existing == plan, 'Existing refit plan differs; refusing to mix protocols')
    else:
        write_json(plan_path, plan)
    proteins = load_full_data(args)
    by_id = {p.sample_id: p for p in proteins}
    chunks = trainer.make_chunks(proteins, args.max_length, args.overlap)
    chunk_counts = {}
    for chunk in chunks:
        chunk_counts[chunk.sample_id] = chunk_counts.get(chunk.sample_id, 0) + 1
    store = trainer.load_store(proteins, args.peptide_window_cache, args.esm_cache)
    baseline.zero_esm(store)
    trainer.set_seed(args.seed + 4300)
    model = ContextualAPRModel(train_v3.config_from_args(args)).to(args.device)
    loader = trainer.make_loader(chunks, by_id, store, args.batch_size, True, args.seed, args.num_workers)
    supcon_generator = torch.Generator().manual_seed(args.seed + 9100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = trainer.calculate_pos_weight(chunks, by_id, torch.device(args.device), args.pos_weight_power)
    latest_path = output / 'refit_latest.pt'
    completed_epoch, history = (0, [])
    if latest_path.exists():
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        require(saved['plan_signature'] == plan['plan_signature'], 'Resume checkpoint belongs to another plan')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        completed_epoch = int(saved['completed_epoch'])
        history = saved['history']
        restore_rng(saved['rng'], loader, supcon_generator)
        print(json.dumps({'event': 'refit_resumed', 'completed_epoch': completed_epoch}), flush=True)
    print(json.dumps({'event': 'refit_start', 'samples': len(proteins), 'chunks': len(chunks), 'epochs': args_cli.epochs, 'pos_weight': float(pos_weight.detach().cpu()), 'plan_signature': plan['plan_signature']}), flush=True)
    for epoch in range(completed_epoch + 1, args_cli.epochs + 1):
        components = {}
        started = time.monotonic()
        loss = objectives.run_epoch(model, loader, torch.device(args.device), optimizer, pos_weight, args, chunk_counts, diagnostics=components, supcon_generator=supcon_generator)
        row = {'epoch': epoch, 'train_loss': float(loss), 'train_components': components, 'elapsed_seconds': time.monotonic() - started}
        history.append(row)
        atomic_torch_save(latest_path, {'plan_signature': plan['plan_signature'], 'completed_epoch': epoch, 'history': history, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'optimizer': optimizer.state_dict(), 'rng': rng_state(loader, supcon_generator)})
        write_json(output / 'training_progress.json', row)
        print(json.dumps({'event': 'refit_epoch_committed', **row}, default=str), flush=True)
    model.eval().requires_grad_(False)
    metadata = {**model_metadata(model), 'input_contract': input_contract, 'purpose': 'full-data fixed-epoch refit; frozen encoder for new-sequence inference', 'plan_signature': plan['plan_signature'], 'refit_epoch': args_cli.epochs, 'train_sample_ids': sorted((p.sample_id for p in proteins)), 'validation_sample_ids': [], 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'training_objectives': train_v3.objective_metadata(args), 'inference_contract': {'input': 'new amino-acid sequence with frozen peptide teacher window features', 'residue_embedding': 'HAPR [L,192] per sequence; batched output uses padding and mask', 'auxiliary_outputs': 'pAPR and rAPR are optional per-residue readouts, not stored training embeddings'}, 'history': history, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    final_path = output / f'encoder_epoch{args_cli.epochs}.pt'
    atomic_torch_save(final_path, metadata)
    restored, _ = load_encoder_checkpoint(final_path, args.device, frozen=True)
    raw = next(iter(loader))
    batch = trainer.move(raw, torch.device(args.device))
    with torch.inference_mode():
        before = objectives._forward(model, batch)
        after = objectives._forward(restored, batch)
        checks = {}
        for key in ('residue_embedding', 'apr_probability', 'reliability'):
            torch.testing.assert_close(before[key], after[key], atol=1e-06, rtol=1e-05)
            checks[key] = {'shape': list(after[key].shape), 'max_abs_diff': float((before[key] - after[key]).abs().max())}
    write_json(output / 'export_manifest.json', {'format': 'hapr_boundary_aware_context_v3_frozen_export_v1', 'variant': VARIANT, 'plan_signature': plan['plan_signature'], 'checkpoint': str(final_path), 'checkpoint_sha256': hashlib.sha256(final_path.read_bytes()).hexdigest(), 'encoder_frozen': True, 'serialization_checks': checks, 'training_data_is_not_embedding_output': True})
    print(json.dumps({'event': 'refit_complete', 'checkpoint': str(final_path), 'epochs': args_cli.epochs, 'serialization_checks': checks}), flush=True)
if __name__ == '__main__':
    main()

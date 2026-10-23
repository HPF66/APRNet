"""Train one supplied fold of the ESM-2 650M + HAPR + PDB model."""
import argparse
import csv
import json
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--membership', type=Path, required=True)
    p.add_argument('--fold-dir', type=Path, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--sources', type=Path, nargs='+', required=True)
    p.add_argument('--graph-roots', type=Path, nargs='+', required=True)
    p.add_argument('--config', type=Path, default=STAGE / 'utils/config.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='cuda')
    cli = p.parse_args()
    if cli.fold < 0 or cli.threads < 1:
        p.error('--fold must be nonnegative and --threads positive')
    return cli


def main():
    cli = arguments()
    import fcntl
    import numpy as np
    import torch
    from protein.my_loader.data import read_membership, read_split
    from protein.my_loader.graphs import GraphStore
    from protein.feature.store import FeatureStore
    from protein.utils.checkpoint import atomic, load_weights, sha
    from protein.utils.model import make_model
    from protein.utils.training import train_candidate, evaluate
    from protein.utils.evaluation import evaluation_condition

    torch.set_num_threads(cli.threads)
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    saved = json.loads(cli.config.read_text(encoding='utf-8'))
    if saved['esm'] != '650m' or saved['hapr'] != 'epoch20':
        raise ValueError('This release contains only ESM-2 650M + epoch20 HAPR + PDB')
    config, structure = saved['sequence'], saved['structure']
    if config.get('family', 'v2') != 'v2' or config.get('scope', 'global') != 'global':
        raise ValueError('Only the retained V2 global Cross-Attention model is included')
    membership = read_membership(cli.membership)
    condition = evaluation_condition(cli.membership, cli.fold_dir, cli.fold)
    paths = {role: cli.fold_dir / f'{role}.tsv' for role in ('train', 'val', 'test')}
    records = {role: read_split(path, membership) for role, path in paths.items()}
    sets = [set(row.sample_id for row in records[role]) for role in ('train', 'val', 'test')]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Sample overlap between train/val/test')
    if set.union(*sets) != set(membership):
        raise ValueError('The supplied fold must cover the membership table exactly')
    if any(not 0 < sum(row.label for row in group) < len(group) for group in records.values()):
        raise ValueError('Each train/validation/test split must contain both labels')
    needed = {sid for group in records.values() for row in group for sid in row.sequence_ids}
    store = FeatureStore(cli.sources, '650m', 'epoch20', needed)
    graphs = GraphStore(cli.graph_roots)
    out = cli.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protected = [cli.membership, *paths.values(), cli.config]
    for source in cli.sources:
        protected.extend(source / name for name in ('sequences.tsv', 'embeddings/esm2/esm2_650m/manifest.json',
                                                     'embeddings/hapr/epoch20/manifest.json'))
    protected.extend(root / 'manifest.json' for root in cli.graph_roots)
    signature = json.loads(json.dumps({**vars(cli), 'model_config': saved, 'evaluation_condition': condition,
        'input_sha256': {str(path.resolve()): sha(path) for path in protected}}, default=str))
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'run_config.json').exists():
            if json.loads((out / 'run_config.json').read_text()) != signature:
                raise ValueError('Output already belongs to a different configuration or input')
        else:
            atomic(out / 'run_config.json', signature)
        validation = train_candidate(config, structure, records, store, graphs, device, cli.fold, out)
        state = load_weights(validation['checkpoint'])
        model = make_model(config, structure, store).to(device)
        model.load_state_dict(state['state_dict'], strict=True)
        metrics, probability = evaluate(model, records['test'], store, graphs, config, structure, 'epoch20', device)
        ids = np.asarray([r.sample_id for r in records['test']])
        labels = np.asarray([r.label for r in records['test']])
        np.savez_compressed(out / 'test_predictions.npz', sample_ids=ids, labels=labels, probability=probability)
        with (out / 'scores.tsv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['sample_id', 'label', 'probability', 'prediction_at_0_5'])
            writer.writerows(zip(ids.tolist(), labels.tolist(), probability.tolist(), (probability >= .5).astype(int).tolist()))
        result = dict(model='ESM2-650M+HAPR+PDB', esm='650m', arm='epoch20', fold=cli.fold,
            evaluation_condition=condition,
            config=config, structure=structure, validation=validation, test=metrics,
            checkpoint=validation['checkpoint'], threshold=.5,
            checkpoint_selection='validation AUROC; unweighted BCE tie-break at tolerance 1e-5')
        atomic(out / 'result.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

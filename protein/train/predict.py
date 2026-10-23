"""Predict with a saved ESM-2 650M + HAPR + PDB classifier checkpoint."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--membership', type=Path, required=True)
    p.add_argument('--sources', type=Path, nargs='+', required=True)
    p.add_argument('--graph-roots', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--microbatch-size', type=int, default=2)
    cli = p.parse_args()
    if cli.microbatch_size < 1:
        p.error('--microbatch-size must be positive')
    if cli.output.exists():
        raise FileExistsError(cli.output)
    import torch
    from protein.my_loader.data import read_membership
    from protein.my_loader.graphs import GraphStore, loader, batches
    from protein.feature.store import FeatureStore
    from protein.utils.checkpoint import load_weights
    from protein.utils.model import make_model
    from protein.utils.training import logits

    state = load_weights(cli.checkpoint)
    if state.get('esm') != '650m' or state.get('arm') != 'epoch20' or 'structure' not in state:
        raise ValueError('A complete ESM-2 650M + HAPR + PDB checkpoint is required')
    config, structure = state['config'], state['structure']
    records = list(read_membership(cli.membership).values())
    needed = {sid for row in records for sid in row.sequence_ids}
    store = FeatureStore(cli.sources, '650m', 'epoch20', needed)
    device = torch.device(cli.device)
    model = make_model(config, structure, store).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    rows = []
    data = loader(records, store, GraphStore(cli.graph_roots), True, config['batch_size'], False, None)
    with torch.inference_mode():
        for full in data:
            for batch in batches(full, cli.microbatch_size, structure):
                probabilities = logits(model, batch, device).sigmoid().cpu()
                if not torch.isfinite(probabilities).all():
                    raise ValueError('Nonfinite protein probabilities')
                rows.extend((sid, float(prob), int(prob >= .5)) for sid, prob in zip(batch['sample_ids'], probabilities))
    if [r[0] for r in rows] != [r.sample_id for r in records]:
        raise ValueError('Inference sample order changed')
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['sample_id', 'probability', 'prediction_at_0_5'])
        writer.writerows(rows)
    print(f'Wrote {len(rows)} protein predictions to {cli.output}')


if __name__ == '__main__':
    main()

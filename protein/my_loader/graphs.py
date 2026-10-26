import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from protein.my_loader.data import SampleDataset, collate_samples, microbatches
from protein.utils.checkpoint import NODE_DIM, EDGE_DIM


class GraphStore:
    def __init__(self, roots):
        self.cache, self.records, self.files = {}, {}, {}
        if isinstance(roots, (str, Path)):
            roots = [roots]
        for root in map(Path, roots):
            manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('validation') != 'passed':
                raise ValueError(f'Unvalidated structure graph manifest: {root}')
            seen = set()
            for record in manifest['records']:
                sid = record['sample_id']
                if sid in seen:
                    raise ValueError(f'Duplicate graph manifest ID: {sid}')
                seen.add(sid)
                if sid in self.records:
                    if self.records[sid] != record:
                        raise ValueError(f'Conflicting graph: {sid}')
                    continue
                self.records[sid] = record
                self.files[sid] = root / record['file']

    def get(self, sample_id, sequence_ids, sequence_lengths=None):
        if self.records[sample_id]['sequence_ids'] != list(sequence_ids):
            raise ValueError(f'Graph chain mapping mismatch: {sample_id}')
        if sample_id not in self.cache:
            with np.load(self.files[sample_id]) as archive:
                fields = ('node_seq', 'positions', 'instance', 'instance_to_seq', 'node', 'edge_index', 'edge')
                if any(key not in archive for key in fields):
                    raise ValueError(f'Incomplete structure graph: {sample_id}')
                arrays = {key: np.array(archive[key]) for key in fields}
            for key, value in arrays.items():
                if key in ('node', 'edge'):
                    if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                        raise ValueError(f'Invalid graph {key}: {sample_id}')
                elif value.dtype.kind not in 'iu':
                    # Casting fractional coordinates/indices to long would
                    # silently change their identity before model indexing.
                    raise ValueError(f'Noninteger graph {key}: {sample_id}')
            if arrays['positions'].ndim != 1 or arrays['instance_to_seq'].ndim != 1:
                raise ValueError(f'Invalid graph index dimensions: {sample_id}')
            nodes, instances = len(arrays['positions']), len(arrays['instance_to_seq'])
            if (arrays['node'].shape != (nodes, NODE_DIM)
                    or arrays['node_seq'].shape != (nodes,)
                    or arrays['instance'].shape != (nodes,)
                    or arrays['edge_index'].ndim != 2 or arrays['edge_index'].shape[0] != 2
                    or arrays['edge'].shape != (arrays['edge_index'].shape[1], EDGE_DIM)):
                raise ValueError(f'Invalid graph shapes: {sample_id}')
            for key, limit in (('node_seq', len(sequence_ids)), ('instance_to_seq', len(sequence_ids)),
                               ('instance', instances), ('edge_index', nodes)):
                if np.any(arrays[key] < 0) or np.any(arrays[key] >= limit):
                    raise ValueError(f'Graph {key} out of bounds: {sample_id}')
            if np.any(arrays['positions'] < 0):
                raise ValueError(f'Negative graph residue position: {sample_id}')
            if not np.array_equal(arrays['node_seq'], arrays['instance_to_seq'][arrays['instance']]):
                raise ValueError(f'Graph node/instance chain mismatch: {sample_id}')
            tensors = {
                key: torch.as_tensor(value, dtype=torch.float32 if key in ('node', 'edge') else torch.long)
                for key, value in arrays.items()
            }
            if any(not torch.isfinite(tensors[key]).all() for key in ('node', 'edge')):
                raise ValueError(f'Graph features overflow float32: {sample_id}')
            self.cache[sample_id] = tensors
        graph = self.cache[sample_id]
        if sequence_lengths is not None:
            lengths = torch.as_tensor(sequence_lengths, dtype=torch.long)
            if lengths.shape != (len(sequence_ids),) or (lengths <= 0).any():
                raise ValueError(f'Invalid graph sequence lengths: {sample_id}')
            if (graph['positions'] >= lengths[graph['node_seq']]).any():
                raise ValueError(f'Graph residue position out of sequence bounds: {sample_id}')
        return graph


def loader(records, store, graphs, include_hapr, batch_size, shuffle, generator):
    def collate(rows):
        batch = collate_samples(rows, store, include_hapr)
        batch['graphs'] = [graphs.get(r.sample_id, r.sequence_ids,
                                     [len(store.sequences[sid]) for sid in r.sequence_ids]) for r in rows]
        return batch
    return DataLoader(SampleDataset(records), batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate, generator=generator if shuffle else None, pin_memory=True)


def batches(full, size, structure):
    for number, batch in enumerate(microbatches(full, size)):
        values = {key: [] for key in ('chain_index', 'positions', 'node', 'edge_index', 'edge', 'instance_index', 'instance_to_chain')}
        chain_offset = node_offset = instance_offset = 0
        selected_graphs = full['graphs'][number * size:number * size + len(batch['labels'])]
        for sample_index, graph in enumerate(selected_graphs):
            values['chain_index'].append(graph['node_seq'] + chain_offset)
            values['positions'].append(graph['positions'])
            values['node'].append(graph['node'])
            values['edge_index'].append(graph['edge_index'] + node_offset)
            values['edge'].append(graph['edge'])
            values['instance_index'].append(graph['instance'] + instance_offset)
            values['instance_to_chain'].append(graph['instance_to_seq'] + chain_offset)
            chain_offset += int((batch['chain_to_sample'] == sample_index).sum())
            node_offset += len(graph['positions'])
            instance_offset += len(graph['instance_to_seq'])
        batch['graph'] = {key: torch.cat(parts, dim=1 if key == 'edge_index' else 0) for key, parts in values.items()}
        assert chain_offset == len(batch['esm'])
        yield batch

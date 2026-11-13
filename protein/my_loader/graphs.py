import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from protein.my_loader.data import SampleDataset, collate_samples, microbatches


class GraphStore:
    def __init__(self, roots):
        self.cache, self.records, self.files = {}, {}, {}
        if isinstance(roots, (str, Path)):
            roots = [roots]
        for root in map(Path, roots):
            manifest = json.loads((root / 'manifest.json').read_text())
            assert manifest['validation'] == 'passed'
            for record in manifest['records']:
                sid = record['sample_id']
                if sid in self.records:
                    assert self.records[sid] == record, f'Conflicting graph: {sid}'
                    continue
                self.records[sid] = record
                self.files[sid] = root / record['file']

    def get(self, sample_id, sequence_ids):
        assert self.records[sample_id]['sequence_ids'] == list(sequence_ids)
        if sample_id not in self.cache:
            with np.load(self.files[sample_id]) as archive:
                self.cache[sample_id] = {
                    key: torch.as_tensor(np.array(archive[key]), dtype=torch.float32 if key in ('node', 'edge') else torch.long)
                    for key in archive.files
                }
        return self.cache[sample_id]


def loader(records, store, graphs, include_hapr, batch_size, shuffle, generator):
    def collate(rows):
        batch = collate_samples(rows, store, include_hapr)
        batch['graphs'] = [graphs.get(r.sample_id, r.sequence_ids) for r in rows]
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

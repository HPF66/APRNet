from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def table(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    label: int
    sequence_ids: tuple[str, ...]


def read_membership(path):
    result = {}
    for row in table(path):
        sid = row['sample_id']
        sequence_ids = tuple(dict.fromkeys(row['sequence_ids'].split(';')))
        if sid in result or not sequence_ids or any(not value for value in sequence_ids):
            raise ValueError(f'Invalid protein membership: {sid}')
        result[sid] = SampleRecord(sid, int(row['label']), sequence_ids)
    return result


def read_split(path, mapping):
    result = []
    seen = set()
    for row in table(path):
        record = mapping[row['sample_id']]
        if record.sample_id in seen or int(row['label']) != record.label:
            raise ValueError(f'Duplicate sample or altered label: {record.sample_id}')
        if set(row['sequence_ids'].split(';')) != set(record.sequence_ids):
            raise ValueError(f'Altered chain mapping: {record.sample_id}')
        seen.add(record.sample_id)
        result.append(record)
    if not result or {r.label for r in result} != {0, 1}:
        raise ValueError(f'Split must contain both protein classes: {path}')
    return result


class SampleDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_samples(batch, store, include_hapr):
    flat, chain_to_sample, sequence_ids = [], [], []
    for sample_index, record in enumerate(batch):
        for sequence_id in record.sequence_ids:
            flat.append(store.get(sequence_id, include_hapr))
            chain_to_sample.append(sample_index)
            sequence_ids.append(sequence_id)
    max_length = max(len(esm) for esm, _ in flat)
    esm_batch = torch.zeros(len(flat), max_length, store.esm_dim)
    mask = torch.zeros(len(flat), max_length, dtype=torch.bool)
    hapr_batch = torch.zeros(len(flat), max_length, 192) if include_hapr else None
    papr_batch = torch.zeros(len(flat), max_length) if include_hapr else None
    rapr_batch = torch.zeros(len(flat), max_length) if include_hapr else None
    for index, (esm, hapr) in enumerate(flat):
        length = len(esm)
        esm_batch[index, :length] = esm
        mask[index, :length] = True
        if include_hapr:
            hapr_batch[index, :length] = hapr['hapr']
            papr_batch[index, :length] = hapr['pAPR']
            rapr_batch[index, :length] = hapr['rAPR']
    return {'esm': esm_batch, 'mask': mask, 'hapr': hapr_batch,
            'pAPR': papr_batch, 'rAPR': rapr_batch,
            'chain_to_sample': torch.tensor(chain_to_sample, dtype=torch.long),
            'labels': torch.tensor([r.label for r in batch], dtype=torch.float32),
            'sample_ids': [r.sample_id for r in batch], 'sequence_ids': sequence_ids}


def make_loader(records, store, include_hapr, batch_size, shuffle, device):
    return DataLoader(SampleDataset(records), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=device.type == 'cuda',
                      collate_fn=lambda batch: collate_samples(batch, store, include_hapr))


def microbatches(batch, size):
    for start in range(0, len(batch['labels']), size):
        stop = min(len(batch['labels']), start + size)
        selected = (batch['chain_to_sample'] >= start) & (batch['chain_to_sample'] < stop)
        length = int(batch['mask'][selected].sum(1).max())
        small = {key: None if batch[key] is None else batch[key][selected, :length]
                 for key in ('esm', 'mask', 'hapr', 'pAPR', 'rAPR')}
        small['chain_to_sample'] = batch['chain_to_sample'][selected] - start
        small['labels'] = batch['labels'][start:stop]
        small['sample_ids'] = batch['sample_ids'][start:stop]
        small['sequence_ids'] = [sid for sid, keep in zip(batch['sequence_ids'], selected.tolist()) if keep]
        yield small

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from protein.my_loader.data import table


DIMS = {'650m': 1280}


def load_object(path):
    if path.suffix == '.npz':
        with np.load(path) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except (RuntimeError, pickle.UnpicklingError, EOFError, ValueError):
        with path.open('rb') as stream:
            return pickle.load(stream)


def verify_identity(value, sid, sequence, path):
    """Check identity metadata when present; legacy raw tensors have none."""
    if not isinstance(value, dict):
        return

    def scalar(key):
        item = value[key]
        if isinstance(item, (np.ndarray, torch.Tensor)):
            if item.ndim != 0:
                raise ValueError(f'Invalid {key} metadata: {path}')
            item = item.item()
        if isinstance(item, bytes):
            item = item.decode('utf-8')
        return item

    expected = {
        'sequence_id': sid,
        'sequence': sequence,
        'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
        'length': len(sequence),
    }
    for key, wanted in expected.items():
        if key in value and scalar(key) != wanted:
            raise ValueError(f'Feature {key} mismatch for {sid}: {path}')


def feature_index(folder, needed, sequences):
    result = {}
    manifest = folder / 'manifest.json'
    if manifest.is_file():
        seen = set()
        for record in json.loads(manifest.read_text(encoding='utf-8'))['records']:
            sid = record['sequence_id']
            if sid in seen:
                raise ValueError(f'Duplicate feature manifest ID {sid}: {manifest}')
            seen.add(sid)
            if sid not in needed:
                continue
            verify_identity(record, sid, sequences[sid], manifest)
            path = folder / record['file']
            if not path.is_file():
                path = folder / 'sequences' / record['file']
            if not path.is_file():
                raise FileNotFoundError(path)
            result[sid] = path
    for sid in needed - result.keys():
        candidates = [folder / f'{sid}{ext}' for ext in ('.pkl', '.pt', '.npz')]
        candidates += [folder / 'sequences' / f'{sid}{ext}' for ext in ('.pt', '.npz')]
        paths = [p for p in candidates if p.is_file()]
        if paths:
            result[sid] = paths[0]
    return result


class FeatureStore:
    """Resolve existing source tensors without copying data or creating links."""

    def __init__(self, sources, esm, hapr, sequence_ids):
        if esm != "650m" or hapr != "epoch20":
            raise ValueError("This release requires ESM-2 650M and epoch20 HAPR, pAPR, rAPR")
        self.esm_dim = DIMS[esm]
        self.sequences, self.esm_paths, self.hapr_paths = {}, {}, {}
        self.esm_cache, self.hapr_cache = {}, {}
        needed = set(sequence_ids)
        for source in map(Path, sources):
            current = {}
            for row in table(source / 'sequences.tsv'):
                sid, sequence = row['sequence_id'], row['sequence']
                if sid not in needed:
                    continue
                if sid in self.sequences and self.sequences[sid] != sequence:
                    raise ValueError(f'Sequence identifier collision: {sid}')
                expected = 'SEQ-' + hashlib.sha256(sequence.encode()).hexdigest()[:12].upper()
                if expected != sid:
                    raise ValueError(f'Sequence hash mismatch: {sid}')
                current[sid] = sequence
                self.sequences[sid] = sequence
            for sid, path in feature_index(source / 'embeddings/esm2' / f'esm2_{esm}', set(current), current).items():
                self.esm_paths.setdefault(sid, path)
            if hapr:
                for sid, path in feature_index(source / 'embeddings/hapr' / hapr, set(current), current).items():
                    self.hapr_paths.setdefault(sid, path)
        if set(self.sequences) != needed or set(self.esm_paths) != needed:
            raise FileNotFoundError(f'Missing ESM/source sequences: {sorted(needed - self.esm_paths.keys())}')
        if hapr and set(self.hapr_paths) != needed:
            raise FileNotFoundError(f'Missing HAPR sequences: {sorted(needed - self.hapr_paths.keys())}')

    def load_esm(self, sid):
        value = load_object(self.esm_paths[sid])
        verify_identity(value, sid, self.sequences[sid], self.esm_paths[sid])
        if isinstance(value, dict):
            value = next(value[k] for k in ('representations', 'embedding', 'embeddings', 'tensor') if k in value)
            if isinstance(value, dict):
                value = value[max(value)]
        tensor = torch.as_tensor(value, dtype=torch.float32, device='cpu').contiguous()
        if tuple(tensor.shape) != (len(self.sequences[sid]), self.esm_dim) or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid ESM tensor: {sid}, {tuple(tensor.shape)}')
        return tensor

    def load_hapr(self, sid):
        value = load_object(self.hapr_paths[sid])
        verify_identity(value, sid, self.sequences[sid], self.hapr_paths[sid])
        result = {k: torch.as_tensor(value[k], dtype=torch.float32, device='cpu').contiguous()
                  for k in ('hapr', 'pAPR', 'rAPR')}
        length = len(self.sequences[sid])
        if tuple(result['hapr'].shape) != (length, 192):
            raise ValueError(f'Invalid HAPR shape: {sid}')
        for key in ('pAPR', 'rAPR'):
            if tuple(result[key].shape) != (length,):
                raise ValueError(f'Invalid {key} shape: {sid}')
            if ((result[key] < 0) | (result[key] > 1)).any():
                raise ValueError(f'{key} outside [0, 1]: {sid}')
        if any(not torch.isfinite(tensor).all() for tensor in result.values()):
            raise ValueError(f'Nonfinite HAPR: {sid}')
        return result

    def get(self, sid, include_hapr=False):
        if sid not in self.esm_cache:
            self.esm_cache[sid] = self.load_esm(sid)
        if include_hapr and sid not in self.hapr_cache:
            self.hapr_cache[sid] = self.load_hapr(sid)
        return self.esm_cache[sid], self.hapr_cache[sid] if include_hapr else None

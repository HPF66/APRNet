"""Checkpoint I/O and aligned geometry feature dimensions."""
import gzip
import hashlib
import json
from pathlib import Path
import torch

NODE_DIM = EDGE_DIM = 22

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_weights(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with gzip.open(temp, 'wb', compresslevel=1) as stream:
        torch.save(state, stream)
    temp.replace(path)

def load_weights(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=False)
    return torch.load(path, map_location='cpu', weights_only=False)

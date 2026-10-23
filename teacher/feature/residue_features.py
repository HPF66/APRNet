from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from teacher.my_loader.data import AMINO_ACIDS


def load_residue_feature_lookup(
    table_path: str | Path,
    selected_sets: Sequence[str],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    selected = tuple(str(name).strip() for name in selected_sets if str(name).strip())
    if not selected:
        raise ValueError("At least one residue feature set must be selected")
    path = Path(table_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = payload.get("feature_sets", {})
    unknown = [name for name in selected if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown residue feature sets {unknown}; available={sorted(available)}"
        )

    rows: list[list[float]] = []
    for amino_acid in AMINO_ACIDS:
        values: list[float] = []
        for name in selected:
            feature_values = available[name].get(amino_acid)
            if feature_values is None:
                raise ValueError(f"{name} is missing amino acid {amino_acid}")
            values.extend(float(value) for value in feature_values)
        rows.append(values)
    feature_dim = len(rows[0])
    if feature_dim == 0 or any(len(row) != feature_dim for row in rows):
        raise ValueError(f"Invalid residue descriptor dimensions in {path}")
    rows.append([0.0] * feature_dim)
    return torch.tensor(rows, dtype=torch.float32), selected

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from openpyxl import load_workbook
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
WINDOW_WIDTH = 320
ESM_WIDTH = 480
WINDOW_SCALES = (6, 8, 10, 12, 16, 20)
EVIDENCE_FEATURES = len(WINDOW_SCALES) * 4

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class WindowAttention(nn.Module):

    def __init__(self, hidden_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        require(hidden_size % heads == 0, 'hidden size must be divisible by attention heads')
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.relation = nn.Sequential(nn.Linear(2, hidden_size // 4), nn.SiLU(), nn.Linear(hidden_size // 4, 1))
        self.dropout = nn.Dropout(dropout)
        self.scale = (hidden_size // heads) ** (-0.5)
        self.heads = heads

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, token_starts: torch.Tensor, token_lengths: torch.Tensor, token_mask: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = query.shape
        w = tokens.shape[1]
        q = self.query(query).view(b, t, self.heads, d // self.heads).transpose(1, 2)
        k = self.key(tokens).view(b, w, self.heads, d // self.heads).transpose(1, 2)
        v = self.value(tokens).view(b, w, self.heads, d // self.heads).transpose(1, 2)
        positions = torch.arange(t, device=query.device, dtype=query.dtype).view(1, 1, t, 1)
        centers = (token_starts + 0.5 * token_lengths).unsqueeze(1).unsqueeze(1)
        rel = (positions - centers) / token_lengths.clamp_min(1.0).unsqueeze(1).unsqueeze(1)
        length_feature = (token_lengths / 20.0).unsqueeze(1).unsqueeze(1).expand(-1, 1, t, -1)
        relation = torch.cat([rel, length_feature], dim=1).permute(0, 2, 3, 1)
        relation = self.relation(relation).squeeze(-1).unsqueeze(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + relation
        covered = (positions >= token_starts.unsqueeze(1).unsqueeze(1)) & (positions < (token_starts + token_lengths).unsqueeze(1).unsqueeze(1))
        allowed = covered & token_mask.unsqueeze(1).unsqueeze(1) & residue_mask.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(~allowed, -10000.0)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * allowed.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        out = torch.matmul(self.dropout(weights), v).transpose(1, 2).contiguous().view(b, t, d)
        return self.output(out) * residue_mask.unsqueeze(-1)

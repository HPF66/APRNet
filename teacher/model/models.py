from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from teacher.my_loader.data import AMINO_ACIDS, AMINO_TO_INDEX, STRUCTURE_GEOMETRIC_FEATURE_DIM, UNKNOWN_INDEX
from teacher.model.apr_regions import MultiScaleAPRRegionHead
from teacher.feature.residue_features import load_residue_feature_lookup


class TinySequenceBackbone(nn.Module):
    """A lightweight fallback that keeps the package usable without a HF download."""

    def __init__(self, hidden_size: int = 256, layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(UNKNOWN_INDEX + 1, hidden_size)
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size),
                    nn.GELU(),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                    nn.LayerNorm(hidden_size),
                ]
            )
        self.blocks = nn.ModuleList(blocks)
        self.hidden_size = hidden_size

    @staticmethod
    def tokenize(sequence: str) -> list[int]:
        return [AMINO_TO_INDEX.get(char, UNKNOWN_INDEX) for char in sequence]

    def forward(self, sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.embedding.weight.device
        max_length = max(len(sequence) for sequence in sequences)
        token_ids = torch.zeros(len(sequences), max_length, dtype=torch.long, device=device)
        mask = torch.zeros(len(sequences), max_length, dtype=torch.bool, device=device)
        for row, sequence in enumerate(sequences):
            ids = self.tokenize(sequence)
            if ids:
                token_ids[row, : len(ids)] = torch.tensor(ids, device=device)
                mask[row, : len(ids)] = True
        hidden = self.embedding(token_ids).transpose(1, 2)
        for block in self.blocks:
            if isinstance(block, nn.LayerNorm):
                hidden = block(hidden.transpose(1, 2)).transpose(1, 2)
            else:
                hidden = block(hidden)
        return hidden.transpose(1, 2), mask


class ESM2SequenceBackbone(nn.Module):
    """ESM-2 residue embeddings with chunking for proteins longer than the model limit."""

    MODEL_NAMES = {
        "esm2_t6_8m": "facebook/esm2_t6_8M_UR50D",
        "esm2_t12_35m": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150m": "facebook/esm2_t30_150M_UR50D",
        "protbert": "Rostlab/prot_bert",
    }

    def __init__(
        self,
        model_name: str = "esm2_t6_8m",
        freeze: bool = True,
        max_residues: int = 1022,
        overlap: int = 64,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        memory_cache_size: int = 4096,
        lora_rank: int = 0,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 2,
        lora_target_projections: Sequence[str] = ("query", "value"),
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised on server
            raise RuntimeError("transformers is required for the ESM-2 backbone") from exc
        if model_path is not None:
            model_source = str(model_path)
        else:
            model_source = self.MODEL_NAMES.get(model_name, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.backbone = AutoModel.from_pretrained(model_source)
        self.residue_spaced_tokenizer = (
            model_name == "protbert"
            or getattr(self.backbone.config, "model_type", "") == "bert"
        )
        if lora_rank < 0:
            raise ValueError("lora_rank must be non-negative")
        if lora_rank > 0 and cache_dir is not None:
            raise ValueError("LoRA training cannot use a frozen embedding cache")
        self.lora_enabled = lora_rank > 0
        self.freeze_backbone = freeze and not self.lora_enabled
        self.max_residues = min(max_residues, 510) if self.residue_spaced_tokenizer else max_residues
        self.overlap = min(overlap, self.max_residues - 1)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.memory_cache_size = max(int(memory_cache_size), 0)
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        if self.cache_dir is not None:
            if not freeze:
                raise ValueError("ESM embedding caching requires a frozen backbone")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if freeze or self.lora_enabled:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
        if self.lora_enabled:
            layer_count = len(self.backbone.encoder.layer)
            if not 1 <= lora_last_n_layers <= layer_count:
                raise ValueError(
                    f"lora_last_n_layers must be between 1 and {layer_count}"
                )
            projections = tuple(str(item).strip() for item in lora_target_projections)
            if not projections or any(not item for item in projections):
                raise ValueError("At least one LoRA target projection is required")
            first_layer = layer_count - lora_last_n_layers
            target_modules = [
                f"encoder.layer.{layer_index}.attention.self.{projection}"
                for layer_index in range(first_layer, layer_count)
                for projection in projections
            ]
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:  # pragma: no cover - server-only option
                raise RuntimeError(
                    "LoRA training requires PEFT; install peft in the server environment"
                ) from exc
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                ),
            )
            trainable_lora = [
                name
                for name, parameter in self.backbone.named_parameters()
                if parameter.requires_grad and "lora_" in name
            ]
            if not trainable_lora:
                raise RuntimeError("PEFT did not create trainable LoRA parameters")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode_chunk(self, sequence: str, device: torch.device) -> torch.Tensor:
        tokenizer_sequence = (
            " ".join(sequence) if self.residue_spaced_tokenizer else sequence
        )
        encoded = self.tokenizer(
            tokenizer_sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            hidden = self.backbone(**encoded).last_hidden_state[0]
        # ESM places BOS at position 0. The following L positions are residues.
        return hidden[1 : 1 + len(sequence)]

    @staticmethod
    def _sequence_key(sequence: str) -> str:
        return hashlib.sha256(sequence.encode("ascii")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / key[:2] / f"{key}.pt"

    def _remember(self, key: str, embedding: torch.Tensor) -> None:
        if self.memory_cache_size == 0:
            return
        self._memory_cache[key] = embedding.detach().to(device="cpu", dtype=torch.float32)
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)

    def _load_cached(self, key: str, sequence_length: int) -> torch.Tensor | None:
        embedding = self._memory_cache.get(key)
        if embedding is not None:
            self._memory_cache.move_to_end(key)
            return embedding
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(embedding, torch.Tensor):
            raise ValueError(f"Invalid ESM cache entry: {path}")
        if tuple(embedding.shape) != (sequence_length, self.hidden_size):
            raise ValueError(
                f"ESM cache shape mismatch for {path}: got {tuple(embedding.shape)}, "
                f"expected {(sequence_length, self.hidden_size)}"
            )
        embedding = embedding.to(dtype=torch.float32)
        self._remember(key, embedding)
        return embedding

    def _save_cached(self, key: str, embedding: torch.Tensor) -> None:
        cpu_embedding = embedding.detach().to(device="cpu", dtype=torch.float32)
        self._remember(key, cpu_embedding)
        path = self._cache_path(key)
        if path is None or path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            torch.save(cpu_embedding, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _encode_one(self, sequence: str, device: torch.device) -> torch.Tensor:
        key = self._sequence_key(sequence)
        if self.freeze_backbone:
            cached = self._load_cached(key, len(sequence))
            if cached is not None:
                return cached.to(device=device)
        if len(sequence) <= self.max_residues:
            output = self._encode_chunk(sequence, device)
        else:
            step = self.max_residues - self.overlap
            output = torch.zeros(len(sequence), self.hidden_size, device=device)
            counts = torch.zeros(len(sequence), 1, device=device)
            start = 0
            while start < len(sequence):
                end = min(start + self.max_residues, len(sequence))
                chunk = self._encode_chunk(sequence[start:end], device)
                output[start:end] += chunk
                counts[start:end] += 1.0
                if end == len(sequence):
                    break
                start += step
            output = output / counts.clamp_min(1.0)
        if self.freeze_backbone:
            self._save_cached(key, output)
        return output

    def forward(self, sequences: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.backbone.parameters()).device
        unique_sequences = list(dict.fromkeys(sequences))
        if all(len(sequence) <= self.max_residues for sequence in unique_sequences):
            unique_embeddings: dict[str, torch.Tensor] = {}
            missing_sequences: list[str] = []
            for sequence in unique_sequences:
                cached = (
                    self._load_cached(self._sequence_key(sequence), len(sequence))
                    if self.freeze_backbone
                    else None
                )
                if cached is None:
                    missing_sequences.append(sequence)
                else:
                    unique_embeddings[sequence] = cached.to(device=device)
            tokenizer_sequences = [
                " ".join(sequence) if self.residue_spaced_tokenizer else sequence
                for sequence in missing_sequences
            ]
            if tokenizer_sequences:
                encoded = self.tokenizer(
                    tokenizer_sequences,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                    padding=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
                with context:
                    hidden = self.backbone(**encoded).last_hidden_state
                for index, sequence in enumerate(missing_sequences):
                    embedding = hidden[index, 1 : 1 + len(sequence)]
                    unique_embeddings[sequence] = embedding
                    if self.freeze_backbone:
                        self._save_cached(self._sequence_key(sequence), embedding)
        else:
            unique_embeddings = {
                sequence: self._encode_one(sequence, device)
                for sequence in unique_sequences
            }
        embeddings = [unique_embeddings[sequence] for sequence in sequences]
        max_length = max(item.shape[0] for item in embeddings)
        output = torch.zeros(len(embeddings), max_length, self.hidden_size, device=device)
        mask = torch.zeros(len(embeddings), max_length, dtype=torch.bool, device=device)
        for row, embedding in enumerate(embeddings):
            output[row, : embedding.shape[0]] = embedding
            mask[row, : embedding.shape[0]] = True
        return output, mask


class ResidualSequenceProjection(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.delta = nn.Linear(size, size)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.delta(hidden)


class SharedSequenceEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "esm2_t6_8m",
        hidden_size: int = 320,
        freeze_backbone: bool = True,
        chunk_overlap: int = 64,
        extra_feature_dim: int = 0,
        model_path: str | Path | None = None,
        embedding_cache_dir: str | Path | None = None,
        lora_rank: int = 0,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 2,
        lora_target_projections: Sequence[str] = ("query", "value"),
        residue_feature_table: str | Path | None = None,
        residue_feature_sets: Sequence[str] = (),
        residue_feature_gate_init: float = -2.0,
        residue_feature_dropout: float = 0.0,
        sequence_channel_dropout: float = 0.0,
        projection_mode: str = "mlp",
    ):
        super().__init__()
        if backbone == "tiny":
            self.backbone = TinySequenceBackbone(hidden_size=hidden_size)
            backbone_size = hidden_size
        else:
            self.backbone = ESM2SequenceBackbone(
                model_name=backbone,
                freeze=freeze_backbone,
                overlap=chunk_overlap,
                model_path=model_path,
                cache_dir=embedding_cache_dir,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_last_n_layers=lora_last_n_layers,
                lora_target_projections=lora_target_projections,
            )
            backbone_size = self.backbone.hidden_size
        self.backbone_size = backbone_size
        if projection_mode == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(backbone_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
        elif projection_mode == "identity":
            if backbone_size != hidden_size:
                raise ValueError(
                    "identity sequence projection requires backbone_size == hidden_size"
                )
            self.projection = nn.Identity()
        elif projection_mode == "residual":
            if backbone_size != hidden_size:
                raise ValueError(
                    "residual sequence projection requires backbone_size == hidden_size"
                )
            self.projection = ResidualSequenceProjection(hidden_size)
        else:
            raise ValueError("projection_mode must be mlp, identity, or residual")
        self.projection_mode = projection_mode
        self.extra_projection = (
            nn.Sequential(nn.Linear(extra_feature_dim, hidden_size), nn.LayerNorm(hidden_size))
            if extra_feature_dim > 0
            else None
        )
        selected_residue_features = tuple(
            str(name).strip() for name in residue_feature_sets if str(name).strip()
        )
        if selected_residue_features and residue_feature_table is None:
            raise ValueError("residue_feature_sets require residue_feature_table")
        if not 0.0 <= residue_feature_dropout < 1.0:
            raise ValueError("residue_feature_dropout must be in [0, 1)")
        if not 0.0 <= sequence_channel_dropout < 1.0:
            raise ValueError("sequence_channel_dropout must be in [0, 1)")
        if selected_residue_features:
            lookup, selected_residue_features = load_residue_feature_lookup(
                residue_feature_table, selected_residue_features
            )
            self.register_buffer("residue_feature_lookup", lookup)
            with torch.random.fork_rng(devices=[]):
                self.residue_feature_projection = nn.Sequential(
                    nn.Linear(lookup.shape[1], hidden_size),
                    nn.LayerNorm(hidden_size),
                )
            self.residue_feature_gate_logit = nn.Parameter(
                torch.tensor(float(residue_feature_gate_init))
            )
            self.residue_feature_dropout = nn.Dropout(residue_feature_dropout)
        else:
            self.register_buffer("residue_feature_lookup", None)
            self.residue_feature_projection = None
            self.register_parameter("residue_feature_gate_logit", None)
            self.residue_feature_dropout = nn.Identity()
        self.residue_feature_sets = selected_residue_features
        self.sequence_channel_dropout = nn.Dropout1d(sequence_channel_dropout)
        self.hidden_size = hidden_size

    def forward_with_backbone(
        self,
        sequences: Sequence[str],
        extra_features: torch.Tensor | None = None,
        include_residue_features: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_hidden, mask = self.backbone(sequences)
        hidden = self.projection(backbone_hidden)
        if self.residue_feature_projection is not None and include_residue_features:
            token_ids = torch.full(
                hidden.shape[:2],
                UNKNOWN_INDEX,
                dtype=torch.long,
                device=hidden.device,
            )
            for row, sequence in enumerate(sequences):
                indices = [AMINO_TO_INDEX.get(amino_acid, UNKNOWN_INDEX) for amino_acid in sequence]
                if indices:
                    token_ids[row, : len(indices)] = torch.tensor(
                        indices, dtype=torch.long, device=hidden.device
                    )
            residue_features = self.residue_feature_lookup[token_ids]
            residue_hidden = self.residue_feature_projection(residue_features)
            feature_gate = torch.sigmoid(self.residue_feature_gate_logit)
            hidden = hidden + feature_gate * self.residue_feature_dropout(residue_hidden)
        if extra_features is not None:
            if self.extra_projection is None:
                raise ValueError("extra_features were provided but extra_feature_dim is zero")
            if extra_features.shape[:2] != hidden.shape[:2]:
                raise ValueError(
                    f"extra feature shape {tuple(extra_features.shape)} does not match "
                    f"sequence representation {tuple(hidden.shape[:2])}"
                )
            hidden = hidden + self.extra_projection(extra_features)
        hidden = self.sequence_channel_dropout(hidden.transpose(1, 2)).transpose(1, 2)
        return hidden, mask, backbone_hidden

    def forward(
        self,
        sequences: Sequence[str],
        extra_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mask, _ = self.forward_with_backbone(
            sequences, extra_features=extra_features
        )
        return hidden, mask


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = hidden.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min)
    return values.max(dim=1).values


def masked_std(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    mean = masked_mean(hidden, mask).unsqueeze(1)
    variance = ((hidden - mean).pow(2) * weights).sum(dim=1)
    variance = variance / weights.sum(dim=1).clamp_min(1.0)
    return torch.sqrt(variance.clamp_min(1e-8))


def masked_logmeanexp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    lengths = mask.sum(dim=1).clamp_min(1).to(logits.dtype)
    return torch.logsumexp(values, dim=1) - torch.log(lengths)


class ShortPeptideCNNHead(nn.Module):
    """Multi-scale amino-acid motif encoder for short peptide supervision.

    This branch is deliberately independent of the ESM tokenizer.  It preserves
    local 3/5/7-mer evidence that can be diluted by global mean/max pooling on
    the many six-residue records in the peptide workbook.
    """

    def __init__(
        self,
        hidden_size: int = 96,
        kernels: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in kernels)
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Short peptide CNN kernels must be positive odd integers")
        if hidden_size <= 0:
            raise ValueError("Short peptide CNN hidden size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Short peptide CNN dropout must be in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.kernels = kernels
        with torch.random.fork_rng(devices=[]):
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            len(AMINO_ACIDS),
                            hidden_size,
                            kernel_size=kernel,
                            padding=kernel // 2,
                        ),
                        nn.GELU(),
                        nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                        nn.GELU(),
                    )
                    for kernel in kernels
                ]
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(hidden_size * len(kernels), hidden_size, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def one_hot_sequences(
        sequences: Sequence[str], max_length: int, device: torch.device
    ) -> torch.Tensor:
        token_ids = torch.full(
            (len(sequences), max_length),
            UNKNOWN_INDEX,
            dtype=torch.long,
            device=device,
        )
        for row, sequence in enumerate(sequences):
            indices = [AMINO_TO_INDEX.get(aa, UNKNOWN_INDEX) for aa in sequence]
            if indices:
                token_ids[row, : len(indices)] = torch.tensor(
                    indices, dtype=torch.long, device=device
                )
        return F.one_hot(token_ids, num_classes=UNKNOWN_INDEX + 1)[..., : len(AMINO_ACIDS)].to(
            dtype=torch.float32
        )

    def forward(
        self, sequences: Sequence[str], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.one_hot_sequences(
            sequences, max_length=mask.shape[1], device=mask.device
        ).transpose(1, 2)
        hidden = self.fusion(torch.cat([branch(features) for branch in self.branches], dim=1))
        hidden = self.norm(hidden.transpose(1, 2))
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        pooled = torch.cat([masked_mean(hidden, mask), masked_max(hidden, mask)], dim=-1)
        return hidden, pooled


class PeptideLocalPropensityHead(nn.Module):
    """Multi-scale residue evidence trained from peptide-level aggregation labels."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        kernels: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in kernels)
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("Local propensity dimensions must be positive")
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Local propensity kernels must be positive odd integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Local propensity dropout must be in [0, 1)")
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Sequential(
                nn.LayerNorm(input_size),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
            )
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size=kernel,
                            padding=kernel // 2,
                        ),
                        nn.GELU(),
                    )
                    for kernel in kernels
                ]
            )
            self.output = nn.Sequential(
                nn.Conv1d(hidden_size * len(kernels), hidden_size, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, 1, kernel_size=1),
            )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(hidden)
        projected = projected * mask.unsqueeze(-1).to(projected.dtype)
        channels = projected.transpose(1, 2)
        logits = self.output(
            torch.cat([branch(channels) for branch in self.branches], dim=1)
        ).squeeze(1)
        return logits.masked_fill(~mask, 0.0)


class LengthConditionedPeptideGate(nn.Module):
    """Learn length-dependent multipliers without changing the initial model."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("Peptide length-gate hidden size must be positive")
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 2),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, sequences: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.float32,
            device=device,
        )
        features = torch.stack(
            [
                lengths.clamp(max=100.0) / 100.0,
                torch.log1p(lengths) / 5.0,
                lengths.clamp_min(1.0).reciprocal(),
            ],
            dim=-1,
        )
        # A zero-initialized last layer gives exactly one for both multipliers.
        return 2.0 * torch.sigmoid(self.network(features))


class LengthConditionedOutputFusion(nn.Module):
    """Predict a bounded main-head weight from sequence length only."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("Peptide output-fusion hidden size must be positive")
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, sequences: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.float32,
            device=device,
        )
        features = torch.stack(
            [
                lengths.clamp(max=100.0) / 100.0,
                torch.log1p(lengths) / 5.0,
                lengths.clamp_min(1.0).reciprocal(),
            ],
            dim=-1,
        )
        # Start at an equal average and keep either head from being suppressed.
        return 0.2 + 0.6 * torch.sigmoid(self.network(features).squeeze(-1))


class PeptideAttentionPoolingResidual(nn.Module):
    """Small no-op-initialized correction from residue-level attention pooling."""

    def __init__(
        self,
        input_size: int,
        attention_size: int = 32,
        heads: int = 2,
        dropout: float = 0.2,
        max_delta: float = 0.5,
    ):
        super().__init__()
        if input_size <= 0 or attention_size <= 0 or heads <= 0:
            raise ValueError("Peptide attention-pooling sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Peptide attention-pooling dropout must be in [0, 1)")
        if max_delta <= 0:
            raise ValueError("Peptide attention-pooling max delta must be positive")
        self.max_delta = float(max_delta)
        with torch.random.fork_rng(devices=[]):
            self.attention = nn.Sequential(
                nn.Linear(input_size, attention_size),
                nn.Tanh(),
                nn.Linear(attention_size, heads),
            )
            nn.init.zeros_(self.attention[-1].weight)
            nn.init.zeros_(self.attention[-1].bias)
            self.residual = nn.Sequential(
                nn.LayerNorm(input_size * heads),
                nn.Linear(input_size * heads, attention_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(attention_size, 1),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(hidden).masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min
        )
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("blh,bld->bhd", weights, hidden).flatten(start_dim=1)
        return self.max_delta * torch.tanh(self.residual(pooled).squeeze(-1))


class GlobalPeptidePhyschemResidual(nn.Module):
    """Bounded neural correction from target-independent peptide composition."""

    GROUPS = (
        frozenset("AVILMFWY"),
        frozenset("STNQ"),
        frozenset("DE"),
        frozenset("KRH"),
        frozenset("FWY"),
    )
    RESIDUE_MASS = {
        "A": 71.0788,
        "C": 103.1388,
        "D": 115.0886,
        "E": 129.1155,
        "F": 147.1766,
        "G": 57.0519,
        "H": 137.1411,
        "I": 113.1594,
        "K": 128.1741,
        "L": 113.1594,
        "M": 131.1926,
        "N": 114.1038,
        "P": 97.1167,
        "Q": 128.1307,
        "R": 156.1875,
        "S": 87.0782,
        "T": 101.1051,
        "V": 99.1326,
        "W": 186.2132,
        "Y": 163.1760,
    }
    FEATURE_DIM = 432

    def __init__(
        self,
        anchor_size: int,
        hidden_size: int = 128,
        dropout: float = 0.2,
        max_delta: float = 1.0,
    ):
        super().__init__()
        if anchor_size <= 0 or hidden_size <= 0:
            raise ValueError("Peptide physchem branch sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Peptide physchem dropout must be in [0, 1)")
        if max_delta <= 0:
            raise ValueError("Peptide physchem max delta must be positive")
        self.max_delta = float(max_delta)
        with torch.random.fork_rng(devices=[]):
            self.feature_encoder = nn.Sequential(
                nn.LayerNorm(self.FEATURE_DIM),
                nn.Linear(self.FEATURE_DIM, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.anchor_projection = nn.Sequential(
                nn.LayerNorm(anchor_size),
                nn.Linear(anchor_size, hidden_size),
                nn.GELU(),
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(hidden_size * 3),
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gate = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
            self.residual = nn.Sequential(
                nn.Linear(hidden_size, max(hidden_size // 2, 16)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden_size // 2, 16), 1),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    @classmethod
    def sequence_features(cls, sequence: str) -> list[float]:
        length = len(sequence)
        denominator = float(max(length, 1))
        counts = [0] * len(AMINO_ACIDS)
        for amino_acid in sequence:
            index = AMINO_TO_INDEX.get(amino_acid)
            if index is not None:
                counts[index] += 1
        amino_acid_fractions = [count / denominator for count in counts]
        dipeptide_counts = [0] * (len(AMINO_ACIDS) ** 2)
        for first, second in zip(sequence, sequence[1:]):
            first_index = AMINO_TO_INDEX.get(first)
            second_index = AMINO_TO_INDEX.get(second)
            if first_index is not None and second_index is not None:
                dipeptide_counts[first_index * len(AMINO_ACIDS) + second_index] += 1
        dipeptide_denominator = float(max(length - 1, 1))
        dipeptide_fractions = [
            count / dipeptide_denominator for count in dipeptide_counts
        ]
        group_fractions = [
            sum(counts[AMINO_TO_INDEX[aa]] for aa in group) / denominator
            for group in cls.GROUPS
        ]
        explicit_fractions = [
            counts[AMINO_TO_INDEX[aa]] / denominator for aa in "GPC"
        ]
        rough_charge = (
            counts[AMINO_TO_INDEX["K"]]
            + counts[AMINO_TO_INDEX["R"]]
            + 0.1 * counts[AMINO_TO_INDEX["H"]]
            - counts[AMINO_TO_INDEX["D"]]
            - counts[AMINO_TO_INDEX["E"]]
        ) / denominator
        approximate_mass = (
            18.0153 + sum(cls.RESIDUE_MASS.get(aa, 110.0) for aa in sequence)
            if length
            else 0.0
        )
        features = [
            length / 100.0,
            math.log1p(length) / 5.0,
            approximate_mass / 10000.0,
            *amino_acid_fractions,
            *dipeptide_fractions,
            *group_fractions,
            *explicit_fractions,
            rough_charge,
        ]
        if len(features) != cls.FEATURE_DIM:
            raise RuntimeError(f"Expected {cls.FEATURE_DIM} peptide features")
        return features

    def forward(
        self, sequences: Sequence[str], anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(
            [self.sequence_features(sequence) for sequence in sequences],
            dtype=torch.float32,
            device=anchor.device,
        )
        feature_hidden = self.feature_encoder(features)
        anchor_hidden = self.anchor_projection(anchor.detach())
        fused = self.fusion(
            torch.cat(
                [feature_hidden, anchor_hidden, feature_hidden * anchor_hidden],
                dim=-1,
            )
        )
        gate = self.gate(fused).squeeze(-1)
        delta = gate * self.max_delta * torch.tanh(self.residual(fused).squeeze(-1))
        return delta, gate


class FixedLengthPeptideResidual(nn.Module):
    """Position-specific bounded residual for the dominant fixed-length peptides."""

    def __init__(
        self,
        target_length: int = 6,
        hidden_size: int = 64,
        dropout: float = 0.2,
        max_delta: float = 1.0,
    ):
        super().__init__()
        if target_length <= 0 or hidden_size <= 0:
            raise ValueError("Fixed-length peptide residual sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Fixed-length peptide dropout must be in [0, 1)")
        if max_delta <= 0:
            raise ValueError("Fixed-length peptide max delta must be positive")
        self.target_length = int(target_length)
        self.max_delta = float(max_delta)
        input_size = self.target_length * len(AMINO_ACIDS)
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(
                nn.LayerNorm(input_size),
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, max(hidden_size // 2, 16)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden_size // 2, 16), 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, sequences: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        token_ids = torch.full(
            (len(sequences), self.target_length),
            UNKNOWN_INDEX,
            dtype=torch.long,
            device=device,
        )
        active = torch.zeros(len(sequences), dtype=torch.bool, device=device)
        for row, sequence in enumerate(sequences):
            if len(sequence) != self.target_length:
                continue
            active[row] = True
            token_ids[row] = torch.tensor(
                [AMINO_TO_INDEX.get(aa, UNKNOWN_INDEX) for aa in sequence],
                dtype=torch.long,
                device=device,
            )
        one_hot = F.one_hot(
            token_ids, num_classes=UNKNOWN_INDEX + 1
        )[..., : len(AMINO_ACIDS)].to(dtype=torch.float32)
        raw_delta = self.network(one_hot.flatten(start_dim=1)).squeeze(-1)
        return (
            self.max_delta
            * torch.tanh(raw_delta)
            * active.to(dtype=raw_delta.dtype)
        )


class APRLocalCompositionResidual(nn.Module):
    """Bounded local sequence correction that starts as an exact no-op."""

    def __init__(
        self,
        kernels: Sequence[int] = (3, 7, 15),
        hidden_size: int = 32,
        dropout: float = 0.1,
        max_delta: float = 0.5,
    ):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in kernels)
        if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("APR local residual kernels must be positive odd integers")
        if hidden_size <= 0:
            raise ValueError("APR local residual hidden size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("APR local residual dropout must be in [0, 1)")
        if max_delta <= 0:
            raise ValueError("APR local residual max delta must be positive")
        self.kernels = kernels
        self.max_delta = float(max_delta)
        # Keep the anchor model's global RNG stream unchanged for paired comparisons.
        with torch.random.fork_rng(devices=[]):
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(
                            len(AMINO_ACIDS),
                            hidden_size,
                            kernel_size=kernel,
                            padding=kernel // 2,
                        ),
                        nn.GELU(),
                        nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                        nn.GELU(),
                    )
                    for kernel in kernels
                ]
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(hidden_size * len(kernels), hidden_size, kernel_size=1),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Conv1d(hidden_size, 1, kernel_size=1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def one_hot_sequences(
        sequences: Sequence[str], max_length: int, device: torch.device
    ) -> torch.Tensor:
        token_ids = torch.full(
            (len(sequences), max_length),
            UNKNOWN_INDEX,
            dtype=torch.long,
            device=device,
        )
        for row, sequence in enumerate(sequences):
            indices = [AMINO_TO_INDEX.get(amino_acid, UNKNOWN_INDEX) for amino_acid in sequence]
            if indices:
                token_ids[row, : len(indices)] = torch.tensor(
                    indices, dtype=torch.long, device=device
                )
        one_hot = F.one_hot(token_ids, num_classes=UNKNOWN_INDEX + 1)
        return one_hot[..., : len(AMINO_ACIDS)].to(dtype=torch.float32)

    def forward(
        self, sequences: Sequence[str], mask: torch.Tensor
    ) -> torch.Tensor:
        features = self.one_hot_sequences(
            sequences, max_length=mask.shape[1], device=mask.device
        ).transpose(1, 2)
        hidden = torch.cat([branch(features) for branch in self.branches], dim=1)
        raw_delta = self.head(self.fusion(hidden)).squeeze(1)
        delta = self.max_delta * torch.tanh(raw_delta)
        return delta * mask.to(delta.dtype)


class SequenceMultiTaskModel(nn.Module):
    def __init__(
        self,
        apr_context_kernel: int = 0,
        apr_head_mode: str = "mlp",
        peptide_apr_fusion: str = "none",
        peptide_apr_beta_init: float = 0.1,
        peptide_apr_beta_max: float = 1.0,
        peptide_apr_window_size: int = 7,
        peptide_apr_feature_residual_hidden_size: int = 32,
        apr_local_residual: bool = False,
        apr_local_residual_kernels: Sequence[int] = (3, 7, 15),
        apr_local_residual_hidden_size: int = 32,
        apr_local_residual_dropout: float = 0.1,
        apr_local_residual_max_delta: float = 0.5,
        apr_region_head: bool = False,
        apr_region_lengths: Sequence[int] = (6, 8, 10, 12, 16, 20),
        apr_region_dropout: float = 0.1,
        apr_region_max_residue_delta: float = 0.5,
        peptide_residue_features: bool = False,
        peptide_residue_feature_gate_init: float = -2.0,
        peptide_residue_feature_dropout: float = 0.0,
        peptide_backbone_gradients: bool = False,
        peptide_short_cnn: bool = False,
        peptide_short_cnn_hidden_size: int = 96,
        peptide_short_cnn_kernels: Sequence[int] = (3, 5, 7),
        peptide_short_cnn_dropout: float = 0.1,
        peptide_short_cnn_gate_init: float = -1.5,
        peptide_length_conditioned_gate: bool = False,
        peptide_length_gate_hidden_size: int = 16,
        peptide_output_fusion: str = "main",
        peptide_output_main_weight: float = 0.5,
        peptide_output_detach_propensity: bool = False,
        peptide_output_fusion_hidden_size: int = 8,
        peptide_attention_residual: bool = False,
        peptide_attention_residual_hidden_size: int = 32,
        peptide_attention_residual_heads: int = 2,
        peptide_attention_residual_dropout: float = 0.2,
        peptide_attention_residual_max_delta: float = 0.5,
        peptide_global_physchem: bool = False,
        peptide_global_physchem_hidden_size: int = 128,
        peptide_global_physchem_dropout: float = 0.2,
        peptide_global_physchem_max_delta: float = 1.0,
        peptide_length6_residual: bool = False,
        peptide_length6_hidden_size: int = 64,
        peptide_length6_dropout: float = 0.2,
        peptide_length6_max_delta: float = 1.0,
        peptide_local_propensity: bool = False,
        peptide_local_propensity_hidden_size: int = 64,
        peptide_local_propensity_kernels: Sequence[int] = (3, 5, 7),
        peptide_local_propensity_dropout: float = 0.1,
        **encoder_kwargs: Any,
    ):
        super().__init__()
        self.encoder = SharedSequenceEncoder(**encoder_kwargs)
        hidden_size = self.encoder.hidden_size
        if peptide_residue_features and self.encoder.residue_feature_lookup is None:
            raise ValueError(
                "Peptide residue features require a residue feature table and sets"
            )
        if not 0.0 <= peptide_residue_feature_dropout < 1.0:
            raise ValueError("Peptide residue feature dropout must be in [0, 1)")
        self.peptide_residue_features = bool(peptide_residue_features)
        self.peptide_backbone_gradients = bool(peptide_backbone_gradients)
        if apr_context_kernel < 0 or (
            apr_context_kernel > 0 and apr_context_kernel % 2 == 0
        ):
            raise ValueError("apr_context_kernel must be zero or a positive odd number")
        self.apr_context_kernel = int(apr_context_kernel)
        if self.apr_context_kernel:
            with torch.random.fork_rng(devices=[]):
                self.apr_context_depthwise = nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=self.apr_context_kernel,
                    padding=self.apr_context_kernel // 2,
                    groups=hidden_size,
                )
                self.apr_context_pointwise = nn.Conv1d(hidden_size, hidden_size, 1)
                self.apr_context_norm = nn.LayerNorm(hidden_size)
                self.apr_context_dropout = nn.Dropout(0.1)
                nn.init.zeros_(self.apr_context_pointwise.weight)
                nn.init.zeros_(self.apr_context_pointwise.bias)
        self.peptide_short_cnn = (
            ShortPeptideCNNHead(
                hidden_size=peptide_short_cnn_hidden_size,
                kernels=peptide_short_cnn_kernels,
                dropout=peptide_short_cnn_dropout,
            )
            if peptide_short_cnn
            else None
        )
        self.peptide_short_cnn_projection = (
            nn.Sequential(
                nn.Linear(peptide_short_cnn_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            if peptide_short_cnn
            else None
        )
        self.peptide_short_cnn_gate_logit = (
            nn.Parameter(torch.tensor(float(peptide_short_cnn_gate_init)))
            if peptide_short_cnn
            else None
        )
        if peptide_length_conditioned_gate and not (
            peptide_short_cnn or peptide_residue_features
        ):
            raise ValueError(
                "Peptide length-conditioned gating requires the short CNN or "
                "residue features"
            )
        self.peptide_length_gate = (
            LengthConditionedPeptideGate(peptide_length_gate_hidden_size)
            if peptide_length_conditioned_gate
            else None
        )
        if peptide_output_fusion not in {"main", "fixed_average", "length_gate"}:
            raise ValueError(
                "Peptide output fusion must be main, fixed_average, or length_gate"
            )
        if not 0.0 <= peptide_output_main_weight <= 1.0:
            raise ValueError("Peptide output main weight must be in [0, 1]")
        if (
            peptide_output_fusion != "main"
            and peptide_apr_fusion == "none"
            and not peptide_local_propensity
        ):
            raise ValueError("Peptide output fusion requires a propensity head")
        self.peptide_output_fusion = peptide_output_fusion
        self.peptide_output_main_weight = float(peptide_output_main_weight)
        self.peptide_output_detach_propensity = bool(
            peptide_output_detach_propensity
        )
        self.peptide_output_fusion_gate = (
            LengthConditionedOutputFusion(peptide_output_fusion_hidden_size)
            if peptide_output_fusion == "length_gate"
            else None
        )
        self.peptide_attention_residual = (
            PeptideAttentionPoolingResidual(
                input_size=hidden_size,
                attention_size=peptide_attention_residual_hidden_size,
                heads=peptide_attention_residual_heads,
                dropout=peptide_attention_residual_dropout,
                max_delta=peptide_attention_residual_max_delta,
            )
            if peptide_attention_residual
            else None
        )
        peptide_head_input_size = hidden_size * 2 + (
            peptide_short_cnn_hidden_size * 2 if peptide_short_cnn else 0
        )
        self.peptide_head = nn.Sequential(
            nn.LayerNorm(peptide_head_input_size),
            nn.Linear(peptide_head_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )
        self.peptide_global_physchem = (
            GlobalPeptidePhyschemResidual(
                anchor_size=peptide_head_input_size,
                hidden_size=peptide_global_physchem_hidden_size,
                dropout=peptide_global_physchem_dropout,
                max_delta=peptide_global_physchem_max_delta,
            )
            if peptide_global_physchem
            else None
        )
        self.peptide_length6_residual = (
            FixedLengthPeptideResidual(
                target_length=6,
                hidden_size=peptide_length6_hidden_size,
                dropout=peptide_length6_dropout,
                max_delta=peptide_length6_max_delta,
            )
            if peptide_length6_residual
            else None
        )
        self.peptide_local_propensity = bool(peptide_local_propensity)
        if apr_head_mode == "mlp":
            self.apr_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1),
            )
        elif apr_head_mode == "linear":
            self.apr_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )
        else:
            raise ValueError("apr_head_mode must be mlp or linear")
        self.apr_head_mode = apr_head_mode
        if peptide_apr_fusion not in {
            "none",
            "detached",
            "joint",
            "bounded_detached",
            "bounded_joint",
            "positive_detached",
            "positive_joint",
            "window_bounded_detached",
            "window_bounded_joint",
            "window_positive_detached",
            "window_positive_joint",
            "feature_bounded_detached",
            "feature_uncertainty_bounded_detached",
        }:
            raise ValueError(
                "peptide_apr_fusion must be none, detached, joint, "
                "bounded/positive, or window_bounded/window_positive variants"
            )
        if peptide_apr_window_size <= 0 or peptide_apr_window_size % 2 == 0:
            raise ValueError("peptide_apr_window_size must be a positive odd integer")
        if peptide_apr_feature_residual_hidden_size <= 0:
            raise ValueError(
                "peptide_apr_feature_residual_hidden_size must be positive"
            )
        if peptide_apr_beta_max <= 0:
            raise ValueError("peptide_apr_beta_max must be positive")
        if not 0 < peptide_apr_beta_init < peptide_apr_beta_max:
            raise ValueError("peptide_apr_beta_init must be in (0, peptide_apr_beta_max)")
        self.peptide_apr_fusion = peptide_apr_fusion
        self.peptide_apr_beta_max = float(peptide_apr_beta_max)
        self.peptide_apr_window_size = int(peptide_apr_window_size)
        self.peptide_apr_fusion_scale = 1.0
        needs_peptide_transfer_branch = peptide_apr_fusion != "none" or (
            apr_region_head and peptide_local_propensity
        )
        if needs_peptide_transfer_branch:
            with torch.random.fork_rng(devices=[]):
                self.peptide_adapter = nn.Sequential(
                    nn.LayerNorm(self.encoder.backbone_size),
                    nn.Linear(self.encoder.backbone_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                self.aggregation_propensity_head = (
                    None
                    if peptide_apr_fusion.startswith("window_") or peptide_local_propensity
                    else nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size // 2, 1),
                    )
                )
                if peptide_apr_fusion in {
                    "feature_bounded_detached",
                    "feature_uncertainty_bounded_detached",
                }:
                    self.peptide_apr_feature_residual_head = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Linear(
                            hidden_size,
                            peptide_apr_feature_residual_hidden_size,
                        ),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(peptide_apr_feature_residual_hidden_size, 1),
                    )
                    nn.init.zeros_(
                        self.peptide_apr_feature_residual_head[-1].weight
                    )
                    nn.init.zeros_(
                        self.peptide_apr_feature_residual_head[-1].bias
                    )
                else:
                    self.peptide_apr_feature_residual_head = None
                if self.peptide_residue_features:
                    self.peptide_residue_feature_projection = nn.Sequential(
                        nn.Linear(
                            self.encoder.residue_feature_lookup.shape[1], hidden_size
                        ),
                        nn.LayerNorm(hidden_size),
                    )
                    self.peptide_residue_feature_gate_logit = nn.Parameter(
                        torch.tensor(float(peptide_residue_feature_gate_init))
                    )
                    self.peptide_residue_feature_dropout = nn.Dropout(
                        peptide_residue_feature_dropout
                    )
                else:
                    self.peptide_residue_feature_projection = None
                    self.register_parameter(
                        "peptide_residue_feature_gate_logit", None
                    )
                    self.peptide_residue_feature_dropout = nn.Identity()
            if peptide_apr_fusion != "none":
                beta_fraction = peptide_apr_beta_init / peptide_apr_beta_max
                self.peptide_apr_beta_logit = nn.Parameter(
                    torch.tensor(float(torch.logit(torch.tensor(beta_fraction))))
                )
            else:
                self.register_parameter("peptide_apr_beta_logit", None)
        else:
            self.peptide_adapter = None
            self.aggregation_propensity_head = None
            self.peptide_apr_feature_residual_head = None
            self.peptide_residue_feature_projection = None
            self.register_parameter("peptide_residue_feature_gate_logit", None)
            self.peptide_residue_feature_dropout = nn.Identity()
            self.register_parameter("peptide_apr_beta_logit", None)
        if peptide_local_propensity:
            self.aggregation_propensity_head = PeptideLocalPropensityHead(
                input_size=hidden_size,
                hidden_size=peptide_local_propensity_hidden_size,
                kernels=peptide_local_propensity_kernels,
                dropout=peptide_local_propensity_dropout,
            )
        self.apr_local_residual = (
            APRLocalCompositionResidual(
                kernels=apr_local_residual_kernels,
                hidden_size=apr_local_residual_hidden_size,
                dropout=apr_local_residual_dropout,
                max_delta=apr_local_residual_max_delta,
            )
            if apr_local_residual
            else None
        )
        self.apr_region_head = (
            MultiScaleAPRRegionHead(
                hidden_size=hidden_size,
                lengths=apr_region_lengths,
                dropout=apr_region_dropout,
                max_residue_delta=apr_region_max_residue_delta,
            )
            if apr_region_head
            else None
        )

    def windowed_peptide_logits(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        kernel = self.peptide_apr_window_size
        padding = kernel // 2
        channels = hidden.transpose(1, 2)
        weights = mask.unsqueeze(1).to(hidden.dtype)
        local_sum = F.avg_pool1d(
            channels * weights,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        ) * kernel
        local_count = F.avg_pool1d(
            weights,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        ) * kernel
        local_mean = (local_sum / local_count.clamp_min(1.0)).transpose(1, 2)

        minimum = torch.finfo(hidden.dtype).min
        masked_channels = channels.masked_fill(~mask.unsqueeze(1), minimum)
        local_max = F.max_pool1d(
            masked_channels,
            kernel_size=kernel,
            stride=1,
            padding=padding,
        ).transpose(1, 2)
        local_pooled = torch.cat([local_mean, local_max], dim=-1)
        return self.peptide_head(local_pooled).squeeze(-1)

    def forward_sequence(
        self,
        sequences: Sequence[str],
        extra_features: torch.Tensor | None = None,
        isolate_peptide_rng: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden, mask, backbone_hidden = self.encoder.forward_with_backbone(
            sequences,
            extra_features=extra_features,
            include_residue_features=not self.peptide_residue_features,
        )
        peptide_rng_devices = (
            [hidden.device.index if hidden.device.index is not None else torch.cuda.current_device()]
            if hidden.device.type == "cuda"
            else []
        )
        peptide_rng_context = (
            torch.random.fork_rng(devices=peptide_rng_devices)
            if isolate_peptide_rng
            else nullcontext()
        )
        with peptide_rng_context:
            peptide_hidden = hidden
            length_multipliers = (
                self.peptide_length_gate(sequences, hidden.device)
                if self.peptide_length_gate is not None
                else None
            )
            cnn_length_multiplier = (
                length_multipliers[:, 0]
                if length_multipliers is not None
                else None
            )
            feature_length_multiplier = (
                length_multipliers[:, 1]
                if length_multipliers is not None
                else None
            )
            if self.peptide_adapter is not None:
                # Peptide supervision has its own adapter so it cannot distort the
                # APR projection. The frozen ESM representation is the shared input.
                peptide_backbone = (
                    backbone_hidden
                    if self.peptide_backbone_gradients
                    else backbone_hidden.detach()
                )
                peptide_hidden = self.peptide_adapter(peptide_backbone)
            if self.peptide_residue_feature_projection is not None:
                token_ids = torch.full(
                    peptide_hidden.shape[:2],
                    UNKNOWN_INDEX,
                    dtype=torch.long,
                    device=peptide_hidden.device,
                )
                for row, sequence in enumerate(sequences):
                    indices = [
                        AMINO_TO_INDEX.get(amino_acid, UNKNOWN_INDEX)
                        for amino_acid in sequence
                    ]
                    if indices:
                        token_ids[row, : len(indices)] = torch.tensor(
                            indices,
                            dtype=torch.long,
                            device=peptide_hidden.device,
                        )
                residue_features = self.encoder.residue_feature_lookup[token_ids]
                feature_hidden = self.peptide_residue_feature_projection(
                    residue_features
                )
                feature_gate = torch.sigmoid(
                    self.peptide_residue_feature_gate_logit
                )
                if feature_length_multiplier is not None:
                    feature_gate = feature_gate * feature_length_multiplier
                peptide_hidden = peptide_hidden + feature_gate.reshape(-1, 1, 1) * (
                    self.peptide_residue_feature_dropout(feature_hidden)
                )
            short_cnn_hidden = None
            short_cnn_pooled = None
            if self.peptide_short_cnn is not None:
                short_cnn_hidden, short_cnn_pooled = self.peptide_short_cnn(
                    sequences, mask
                )
                short_cnn_gate = torch.sigmoid(self.peptide_short_cnn_gate_logit)
                if cnn_length_multiplier is not None:
                    short_cnn_gate = short_cnn_gate * cnn_length_multiplier
                    short_cnn_pooled = short_cnn_pooled * (
                        cnn_length_multiplier.unsqueeze(-1)
                    )
                peptide_hidden = peptide_hidden + short_cnn_gate.reshape(-1, 1, 1) * (
                    self.peptide_short_cnn_projection(short_cnn_hidden)
                )
            pooled_parts = [
                masked_mean(peptide_hidden, mask),
                masked_max(peptide_hidden, mask),
            ]
            if short_cnn_pooled is not None:
                pooled_parts.append(short_cnn_pooled)
            pooled = torch.cat(pooled_parts, dim=-1)
            peptide_base_logits = self.peptide_head(pooled).squeeze(-1)
            peptide_physchem_delta = None
            peptide_physchem_gate = None
            if self.peptide_global_physchem is not None:
                peptide_physchem_delta, peptide_physchem_gate = (
                    self.peptide_global_physchem(sequences, pooled)
                )
            peptide_logits = peptide_base_logits + (
                peptide_physchem_delta
                if peptide_physchem_delta is not None
                else 0.0
            )
            peptide_length6_delta = None
            if self.peptide_length6_residual is not None:
                peptide_length6_delta = self.peptide_length6_residual(
                    sequences, peptide_logits.device
                )
                peptide_logits = peptide_logits + peptide_length6_delta
            peptide_attention_delta = None
            if self.peptide_attention_residual is not None:
                peptide_attention_delta = self.peptide_attention_residual(
                    peptide_hidden, mask
                )
                peptide_logits = peptide_logits + peptide_attention_delta
            propensity_logits = None
            propensity_peptide_logits = None
            propensity_correction = None
            if self.peptide_local_propensity:
                propensity_logits = self.aggregation_propensity_head(
                    peptide_hidden, mask
                )
                propensity_peptide_logits = masked_logmeanexp(
                    propensity_logits, mask
                )
            elif self.peptide_apr_fusion.startswith("window_"):
                propensity_logits = self.windowed_peptide_logits(
                    peptide_hidden, mask
                )
                propensity_peptide_logits = masked_logmeanexp(
                    propensity_logits, mask
                )
            elif self.aggregation_propensity_head is not None:
                propensity_logits = self.aggregation_propensity_head(
                    peptide_hidden
                ).squeeze(-1)
                propensity_peptide_logits = masked_logmeanexp(
                    propensity_logits, mask
                )
            if propensity_logits is not None:
                if "bounded" in self.peptide_apr_fusion:
                    propensity_correction = torch.tanh(propensity_logits)
                elif "positive" in self.peptide_apr_fusion:
                    propensity_correction = torch.sigmoid(propensity_logits)
                else:
                    propensity_correction = propensity_logits
            peptide_main_logits = peptide_logits
            peptide_output_main_weight = None
            if self.peptide_output_fusion != "main":
                if propensity_peptide_logits is None:
                    raise RuntimeError(
                        "Peptide output fusion requires propensity Peptide logits"
                    )
                if self.peptide_output_fusion == "fixed_average":
                    peptide_output_main_weight = torch.full_like(
                        peptide_main_logits, self.peptide_output_main_weight
                    )
                else:
                    peptide_output_main_weight = self.peptide_output_fusion_gate(
                        sequences, peptide_main_logits.device
                    )
                output_propensity_logits = (
                    propensity_peptide_logits.detach()
                    if self.peptide_output_detach_propensity
                    else propensity_peptide_logits
                )
                fused_probability = (
                    peptide_output_main_weight * torch.sigmoid(peptide_main_logits)
                    + (1.0 - peptide_output_main_weight)
                    * torch.sigmoid(output_propensity_logits)
                )
                peptide_logits = torch.logit(
                    fused_probability.clamp(min=1e-6, max=1.0 - 1e-6)
                )
            feature_residual_delta = None
            if self.peptide_apr_feature_residual_head is not None:
                feature_residual_delta = torch.tanh(
                    self.peptide_apr_feature_residual_head(
                        peptide_hidden.detach()
                    ).squeeze(-1)
                )
        apr_hidden = hidden
        if self.apr_context_kernel:
            context = self.apr_context_depthwise(hidden.transpose(1, 2))
            context = F.gelu(self.apr_context_pointwise(context)).transpose(1, 2)
            context = self.apr_context_dropout(self.apr_context_norm(context))
            apr_hidden = hidden + context
        apr_logits = self.apr_head(apr_hidden).squeeze(-1)
        apr_local_residual_delta = None
        if self.apr_local_residual is not None:
            apr_local_residual_delta = self.apr_local_residual(sequences, mask)
            apr_logits = apr_logits + apr_local_residual_delta
        peptide_apr_beta = None
        apr_correction = (
            feature_residual_delta
            if feature_residual_delta is not None
            else propensity_correction
        )
        peptide_apr_uncertainty_gate = None
        if apr_correction is not None and self.peptide_apr_fusion != "none":
            peptide_apr_beta = (
                self.peptide_apr_fusion_scale
                * self.peptide_apr_beta_max
                * torch.sigmoid(self.peptide_apr_beta_logit)
            )
            if feature_residual_delta is not None:
                correction_for_apr = feature_residual_delta
                if self.peptide_apr_fusion == "feature_uncertainty_bounded_detached":
                    base_probability = torch.sigmoid(apr_logits.detach())
                    peptide_apr_uncertainty_gate = (
                        4.0 * base_probability * (1.0 - base_probability)
                    )
                    correction_for_apr = (
                        correction_for_apr * peptide_apr_uncertainty_gate
                    )
            else:
                correction_for_apr = (
                    apr_correction.detach()
                    if self.peptide_apr_fusion.endswith("detached")
                    else apr_correction
                )
            apr_logits = apr_logits + peptide_apr_beta * correction_for_apr
        apr_region_output = None
        if self.apr_region_head is not None:
            apr_region_output = self.apr_region_head(
                apr_hidden,
                mask,
                propensity_logits=propensity_logits,
            )
            apr_logits = apr_logits + apr_region_output["residue_delta"]
        return {
            "hidden": hidden,
            "peptide_hidden": peptide_hidden,
            "residue_mask": mask,
            "peptide_logits": peptide_logits,
            "peptide_main_logits": peptide_main_logits,
            "peptide_output_main_weight": peptide_output_main_weight,
            "peptide_pooled": pooled,
            "peptide_embedding": pooled,
            "peptide_base_logits": peptide_base_logits,
            "peptide_physchem_delta": peptide_physchem_delta,
            "peptide_physchem_gate": peptide_physchem_gate,
            "peptide_length6_delta": peptide_length6_delta,
            "peptide_attention_delta": peptide_attention_delta,
            "peptide_length_cnn_multiplier": cnn_length_multiplier,
            "peptide_length_feature_multiplier": feature_length_multiplier,
            "apr_logits": apr_logits,
            "aggregation_propensity_logits": propensity_logits,
            "propensity_peptide_logits": propensity_peptide_logits,
            "peptide_apr_beta": peptide_apr_beta,
            "peptide_apr_feature_residual_delta": feature_residual_delta,
            "peptide_apr_uncertainty_gate": peptide_apr_uncertainty_gate,
            "apr_local_residual_delta": apr_local_residual_delta,
            "apr_region_logits": (
                apr_region_output["region_logits"]
                if apr_region_output is not None
                else None
            ),
            "apr_region_valid": (
                apr_region_output["region_valid"]
                if apr_region_output is not None
                else None
            ),
            "apr_region_lengths": (
                apr_region_output["region_lengths"]
                if apr_region_output is not None
                else None
            ),
            "apr_region_context": (
                apr_region_output["region_context"]
                if apr_region_output is not None
                else None
            ),
            "apr_region_residue_delta": (
                apr_region_output["residue_delta"]
                if apr_region_output is not None
                else None
            ),
        }

    def set_peptide_apr_fusion_scale(self, scale: float) -> None:
        if not 0.0 <= scale <= 1.0:
            raise ValueError("Peptide-to-APR fusion scale must be in [0, 1]")
        self.peptide_apr_fusion_scale = float(scale)

    def set_peptide_transfer_eval(self) -> None:
        """Keep frozen Peptide evidence deterministic during APR-only training."""
        modules = (
            self.encoder,
            self.peptide_adapter,
            self.aggregation_propensity_head,
            self.peptide_residue_feature_projection,
            self.peptide_short_cnn,
            self.peptide_short_cnn_projection,
        )
        for module in modules:
            if module is not None:
                module.eval()


class GraphMessagePassing(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        residual: bool = False,
    ):
        super().__init__()
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.neighbor_projection = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = adjacency * node_mask.unsqueeze(1).to(adjacency.dtype)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor = torch.bmm(adjacency, hidden) / degree
        message = F.gelu(
            self.self_projection(hidden) + self.neighbor_projection(neighbor)
        )
        if self.residual:
            updated = self.norm(hidden + self.dropout(message))
        else:
            updated = self.dropout(self.norm(message))
        return updated * node_mask.unsqueeze(-1).to(updated.dtype)


class StructureEncoder(nn.Module):
    POOLING_MODES = {"meanmax", "meanmaxstd", "meanmaxstdattn"}

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 192,
        layers: int = 4,
        message_dropout: float = 0.1,
        residual: bool = False,
        layer_mix: bool = False,
        input_skip: bool = False,
        pooling: str = "meanmax",
        peptide_attention: bool = False,
        peptide_attention_alpha_init: float = 0.1,
        peptide_attention_alpha_max: float = 1.0,
    ):
        super().__init__()
        if pooling not in self.POOLING_MODES:
            raise ValueError(f"Unknown structure pooling mode: {pooling}")
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                GraphMessagePassing(
                    hidden_size,
                    dropout=message_dropout,
                    residual=residual,
                )
                for _ in range(layers)
            ]
        )
        self.layer_mix_logits = (
            nn.Parameter(torch.zeros(layers + 1)) if layer_mix else None
        )
        self.attention_score = (
            nn.Linear(hidden_size, 1) if pooling == "meanmaxstdattn" else None
        )
        if peptide_attention and self.attention_score is None:
            raise ValueError(
                "Peptide attention requires structure pooling meanmaxstdattn"
            )
        if peptide_attention_alpha_max <= 0:
            raise ValueError("Peptide attention alpha max must be positive")
        if not 0 < peptide_attention_alpha_init < peptide_attention_alpha_max:
            raise ValueError("Peptide attention alpha init must be in (0, alpha max)")
        self.peptide_attention = bool(peptide_attention)
        self.peptide_attention_alpha_max = float(peptide_attention_alpha_max)
        if self.peptide_attention:
            alpha_fraction = peptide_attention_alpha_init / peptide_attention_alpha_max
            self.peptide_attention_alpha_logit = nn.Parameter(
                torch.logit(torch.tensor(float(alpha_fraction)))
            )
        else:
            self.register_parameter("peptide_attention_alpha_logit", None)
        self.pooling = pooling
        self.input_skip = input_skip
        self.hidden_size = hidden_size
        pool_width = {
            "meanmax": 2,
            "meanmaxstd": 3,
            "meanmaxstdattn": 4,
        }[pooling]
        self.output_size = hidden_size * (pool_width + (2 if input_skip else 0))

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        peptide_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = F.gelu(self.input_projection(node_features))
        hidden = hidden * node_mask.unsqueeze(-1).to(hidden.dtype)
        input_hidden = hidden
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden, adjacency, node_mask)
            states.append(hidden)
        if self.layer_mix_logits is not None:
            weights = torch.softmax(self.layer_mix_logits, dim=0)
            hidden = sum(weight * state for weight, state in zip(weights, states))
        pooled = [masked_mean(hidden, node_mask), masked_max(hidden, node_mask)]
        if self.pooling in {"meanmaxstd", "meanmaxstdattn"}:
            pooled.append(masked_std(hidden, node_mask))
        if self.attention_score is not None:
            scores = self.attention_score(hidden).squeeze(-1)
            if self.peptide_attention:
                if peptide_evidence is None:
                    raise ValueError("Peptide attention requires residue evidence")
                if peptide_evidence.shape != node_mask.shape:
                    raise ValueError(
                        "Peptide evidence shape must match the structure node mask"
                    )
                prior_logits = torch.logit(
                    peptide_evidence.detach().clamp(1e-4, 1.0 - 1e-4)
                )
                prior_logits = prior_logits.masked_fill(~node_mask, 0.0)
                lengths = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
                prior_mean = prior_logits.sum(dim=1, keepdim=True) / lengths
                centered_prior = (prior_logits - prior_mean) * node_mask
                prior_variance = centered_prior.pow(2).sum(
                    dim=1, keepdim=True
                ) / lengths
                standardized_prior = centered_prior / torch.sqrt(
                    prior_variance + 1e-6
                )
                alpha = self.peptide_attention_alpha_max * torch.sigmoid(
                    self.peptide_attention_alpha_logit
                )
                scores = scores + alpha * standardized_prior.clamp(-3.0, 3.0)
            elif peptide_evidence is not None:
                raise ValueError(
                    "Peptide evidence was provided but structure Peptide attention is disabled"
                )
            scores = scores.masked_fill(~node_mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled.append((hidden * attention).sum(dim=1))
        if self.input_skip:
            pooled.extend(
                [
                    masked_mean(input_hidden, node_mask),
                    masked_max(input_hidden, node_mask),
                ]
            )
        return torch.cat(pooled, dim=-1)

    def peptide_attention_alpha(self) -> torch.Tensor | None:
        if self.peptide_attention_alpha_logit is None:
            return None
        return self.peptide_attention_alpha_max * torch.sigmoid(
            self.peptide_attention_alpha_logit
        )


class StructureClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 192,
        layers: int = 4,
        message_dropout: float = 0.1,
        head_dropout: float = 0.2,
        residual: bool = False,
        layer_mix: bool = False,
        input_skip: bool = False,
        pooling: str = "meanmax",
        peptide_attention: bool = False,
        peptide_attention_alpha_init: float = 0.1,
        peptide_attention_alpha_max: float = 1.0,
    ):
        super().__init__()
        self.encoder = StructureEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            layers=layers,
            message_dropout=message_dropout,
            residual=residual,
            layer_mix=layer_mix,
            input_skip=input_skip,
            pooling=pooling,
            peptide_attention=peptide_attention,
            peptide_attention_alpha_init=peptide_attention_alpha_init,
            peptide_attention_alpha_max=peptide_attention_alpha_max,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_size),
            nn.Linear(self.encoder.output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        peptide_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.encoder(
            node_features,
            adjacency,
            node_mask,
            peptide_evidence=peptide_evidence,
        )
        return self.head(pooled).squeeze(-1)


class AmyloidAPRModel(nn.Module):
    """The complete model: sequence auxiliary/APR tasks plus structure classification."""

    def __init__(
        self,
        sequence_backbone: str = "esm2_t6_8m",
        sequence_hidden_size: int = 320,
        freeze_sequence_backbone: bool = True,
        chunk_overlap: int = 64,
        structure_input_size: int = 22,
        structure_hidden_size: int = 192,
        structure_layers: int = 4,
        structure_message_dropout: float = 0.1,
        structure_head_dropout: float = 0.2,
        structure_residual: bool = False,
        structure_layer_mix: bool = False,
        structure_input_skip: bool = False,
        structure_pooling: str = "meanmax",
        structure_geometric_features: bool = False,
        extra_feature_dim: int = 0,
        sequence_model_path: str | Path | None = None,
        embedding_cache_dir: str | Path | None = None,
        lora_rank: int = 0,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 2,
        lora_target_projections: Sequence[str] = ("query", "value"),
        residue_feature_table: str | Path | None = None,
        residue_feature_sets: Sequence[str] = (),
        residue_feature_gate_init: float = -2.0,
        residue_feature_dropout: float = 0.0,
        peptide_residue_features: bool = False,
        peptide_backbone_gradients: bool = False,
        peptide_short_cnn: bool = False,
        peptide_short_cnn_hidden_size: int = 96,
        peptide_short_cnn_kernels: Sequence[int] = (3, 5, 7),
        peptide_short_cnn_dropout: float = 0.1,
        peptide_short_cnn_gate_init: float = -1.5,
        peptide_length_conditioned_gate: bool = False,
        peptide_length_gate_hidden_size: int = 16,
        peptide_output_fusion: str = "main",
        peptide_output_main_weight: float = 0.5,
        peptide_output_detach_propensity: bool = False,
        peptide_output_fusion_hidden_size: int = 8,
        peptide_attention_residual: bool = False,
        peptide_attention_residual_hidden_size: int = 32,
        peptide_attention_residual_heads: int = 2,
        peptide_attention_residual_dropout: float = 0.2,
        peptide_attention_residual_max_delta: float = 0.5,
        peptide_global_physchem: bool = False,
        peptide_global_physchem_hidden_size: int = 128,
        peptide_global_physchem_dropout: float = 0.2,
        peptide_global_physchem_max_delta: float = 1.0,
        peptide_length6_residual: bool = False,
        peptide_length6_hidden_size: int = 64,
        peptide_length6_dropout: float = 0.2,
        peptide_length6_max_delta: float = 1.0,
        peptide_local_propensity: bool = False,
        peptide_local_propensity_hidden_size: int = 64,
        peptide_local_propensity_kernels: Sequence[int] = (3, 5, 7),
        peptide_local_propensity_dropout: float = 0.1,
        sequence_channel_dropout: float = 0.0,
        sequence_projection_mode: str = "mlp",
        peptide_apr_fusion: str = "none",
        peptide_apr_beta_init: float = 0.1,
        peptide_apr_beta_max: float = 1.0,
        peptide_apr_window_size: int = 7,
        peptide_apr_feature_residual_hidden_size: int = 32,
        structure_sequence_embeddings: bool = False,
        structure_apr_probabilities: bool = False,
        structure_apr_gradient_beta: float = 0.0,
        structure_peptide_attention: bool = False,
        structure_peptide_attention_alpha_init: float = 0.1,
        structure_peptide_attention_alpha_max: float = 1.0,
        apr_context_kernel: int = 0,
        apr_head_mode: str = "mlp",
        apr_local_residual: bool = False,
        apr_local_residual_kernels: Sequence[int] = (3, 7, 15),
        apr_local_residual_hidden_size: int = 32,
        apr_local_residual_dropout: float = 0.1,
        apr_local_residual_max_delta: float = 0.5,
        apr_region_head: bool = False,
        apr_region_lengths: Sequence[int] = (6, 8, 10, 12, 16, 20),
        apr_region_dropout: float = 0.1,
        apr_region_max_residue_delta: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= structure_apr_gradient_beta <= 1.0:
            raise ValueError("structure_apr_gradient_beta must be in [0, 1]")
        self.sequence = SequenceMultiTaskModel(
            backbone=sequence_backbone,
            hidden_size=sequence_hidden_size,
            freeze_backbone=freeze_sequence_backbone,
            chunk_overlap=chunk_overlap,
            extra_feature_dim=extra_feature_dim,
            model_path=sequence_model_path,
            embedding_cache_dir=embedding_cache_dir,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_last_n_layers=lora_last_n_layers,
            lora_target_projections=lora_target_projections,
            residue_feature_table=residue_feature_table,
            residue_feature_sets=residue_feature_sets,
            residue_feature_gate_init=residue_feature_gate_init,
            residue_feature_dropout=residue_feature_dropout,
            peptide_residue_features=peptide_residue_features,
            peptide_residue_feature_gate_init=residue_feature_gate_init,
            peptide_residue_feature_dropout=residue_feature_dropout,
            peptide_backbone_gradients=peptide_backbone_gradients,
            peptide_short_cnn=peptide_short_cnn,
            peptide_short_cnn_hidden_size=peptide_short_cnn_hidden_size,
            peptide_short_cnn_kernels=peptide_short_cnn_kernels,
            peptide_short_cnn_dropout=peptide_short_cnn_dropout,
            peptide_short_cnn_gate_init=peptide_short_cnn_gate_init,
            peptide_length_conditioned_gate=peptide_length_conditioned_gate,
            peptide_length_gate_hidden_size=peptide_length_gate_hidden_size,
            peptide_output_fusion=peptide_output_fusion,
            peptide_output_main_weight=peptide_output_main_weight,
            peptide_output_detach_propensity=peptide_output_detach_propensity,
            peptide_output_fusion_hidden_size=peptide_output_fusion_hidden_size,
            peptide_attention_residual=peptide_attention_residual,
            peptide_attention_residual_hidden_size=(
                peptide_attention_residual_hidden_size
            ),
            peptide_attention_residual_heads=peptide_attention_residual_heads,
            peptide_attention_residual_dropout=peptide_attention_residual_dropout,
            peptide_attention_residual_max_delta=(
                peptide_attention_residual_max_delta
            ),
            peptide_global_physchem=peptide_global_physchem,
            peptide_global_physchem_hidden_size=(
                peptide_global_physchem_hidden_size
            ),
            peptide_global_physchem_dropout=peptide_global_physchem_dropout,
            peptide_global_physchem_max_delta=peptide_global_physchem_max_delta,
            peptide_length6_residual=peptide_length6_residual,
            peptide_length6_hidden_size=peptide_length6_hidden_size,
            peptide_length6_dropout=peptide_length6_dropout,
            peptide_length6_max_delta=peptide_length6_max_delta,
            peptide_local_propensity=peptide_local_propensity,
            peptide_local_propensity_hidden_size=(
                peptide_local_propensity_hidden_size
            ),
            peptide_local_propensity_kernels=peptide_local_propensity_kernels,
            peptide_local_propensity_dropout=peptide_local_propensity_dropout,
            sequence_channel_dropout=sequence_channel_dropout,
            projection_mode=sequence_projection_mode,
            apr_context_kernel=apr_context_kernel,
            apr_head_mode=apr_head_mode,
            apr_local_residual=apr_local_residual,
            apr_local_residual_kernels=apr_local_residual_kernels,
            apr_local_residual_hidden_size=apr_local_residual_hidden_size,
            apr_local_residual_dropout=apr_local_residual_dropout,
            apr_local_residual_max_delta=apr_local_residual_max_delta,
            apr_region_head=apr_region_head,
            apr_region_lengths=apr_region_lengths,
            apr_region_dropout=apr_region_dropout,
            apr_region_max_residue_delta=apr_region_max_residue_delta,
            peptide_apr_fusion=peptide_apr_fusion,
            peptide_apr_beta_init=peptide_apr_beta_init,
            peptide_apr_beta_max=peptide_apr_beta_max,
            peptide_apr_window_size=peptide_apr_window_size,
            peptide_apr_feature_residual_hidden_size=(
                peptide_apr_feature_residual_hidden_size
            ),
        )
        self.structure_sequence_embeddings = structure_sequence_embeddings
        self.structure_apr_probabilities = structure_apr_probabilities
        self.structure_apr_gradient_beta = float(structure_apr_gradient_beta)
        if structure_geometric_features:
            structure_input_size += STRUCTURE_GEOMETRIC_FEATURE_DIM
        if structure_sequence_embeddings:
            structure_input_size += self.sequence.encoder.backbone_size
        if structure_apr_probabilities:
            structure_input_size += 1
        self.structure = StructureClassifier(
            input_size=structure_input_size,
            hidden_size=structure_hidden_size,
            layers=structure_layers,
            message_dropout=structure_message_dropout,
            head_dropout=structure_head_dropout,
            residual=structure_residual,
            layer_mix=structure_layer_mix,
            input_skip=structure_input_skip,
            pooling=structure_pooling,
            peptide_attention=structure_peptide_attention,
            peptide_attention_alpha_init=structure_peptide_attention_alpha_init,
            peptide_attention_alpha_max=structure_peptide_attention_alpha_max,
        )

    def forward_sequence(self, sequences: Sequence[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.sequence.forward_sequence(sequences, **kwargs)

    def forward_structure(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        chain_sequences: Sequence[Sequence[str]] | None = None,
        apr_gradient_beta: float | None = None,
        peptide_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.structure_sequence_embeddings or self.structure_apr_probabilities:
            if chain_sequences is None:
                raise ValueError("Structure sequence features require PDB chain sequences")
            sequence_features, apr_probabilities = self._encode_structure_chains(
                chain_sequences,
                node_mask=node_mask,
                include_embeddings=self.structure_sequence_embeddings,
                include_apr_probabilities=self.structure_apr_probabilities,
                apr_gradient_beta=apr_gradient_beta,
            )
            features = [node_features]
            if sequence_features is not None:
                features.append(sequence_features)
            if apr_probabilities is not None:
                features.append(apr_probabilities)
            node_features = torch.cat(features, dim=-1)
        return self.structure(
            node_features,
            adjacency,
            node_mask,
            peptide_evidence=peptide_evidence,
        )

    def _encode_structure_chains(
        self,
        chain_sequences: Sequence[Sequence[str]],
        node_mask: torch.Tensor,
        include_embeddings: bool,
        include_apr_probabilities: bool,
        apr_gradient_beta: float | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if len(chain_sequences) != node_mask.shape[0]:
            raise ValueError(
                f"Structure batch mismatch: {len(chain_sequences)} sequence groups for "
                f"{node_mask.shape[0]} graphs"
            )
        flat_sequences = [
            sequence
            for sample_sequences in chain_sequences
            for sequence in sample_sequences
        ]
        if not flat_sequences:
            raise ValueError("Structure batch contains no PDB chain sequences")
        backbone_hidden, residue_mask = self.sequence.encoder.backbone(flat_sequences)
        sequence_output = None
        if include_embeddings:
            sequence_output = backbone_hidden.new_zeros(
                node_mask.shape[0],
                node_mask.shape[1],
                self.sequence.encoder.backbone_size,
            )
        apr_output = None
        flat_apr_probabilities = None
        if include_apr_probabilities:
            beta = (
                self.structure_apr_gradient_beta
                if apr_gradient_beta is None
                else float(apr_gradient_beta)
            )
            if not 0.0 <= beta <= 1.0:
                raise ValueError("apr_gradient_beta must be in [0, 1]")
            projected_hidden = self.sequence.encoder.projection(backbone_hidden)
            apr_logits = self.sequence.apr_head(projected_hidden).squeeze(-1)
            raw_probabilities = torch.sigmoid(apr_logits)
            flat_apr_probabilities = (
                raw_probabilities.detach()
                + beta * (raw_probabilities - raw_probabilities.detach())
            )
            apr_output = backbone_hidden.new_zeros(
                node_mask.shape[0], node_mask.shape[1], 1
            )
        chain_index = 0
        for sample_index, sample_sequences in enumerate(chain_sequences):
            offset = 0
            for sequence in sample_sequences:
                length = len(sequence)
                if length == 0 or not torch.all(residue_mask[chain_index, :length]):
                    raise ValueError("ESM residue mask does not match a PDB chain sequence")
                end = offset + length
                if end > node_mask.shape[1]:
                    raise ValueError("PDB chain sequences exceed the padded structure graph")
                if sequence_output is not None:
                    sequence_output[sample_index, offset:end] = backbone_hidden[
                        chain_index, :length
                    ]
                if apr_output is not None and flat_apr_probabilities is not None:
                    apr_output[sample_index, offset:end, 0] = flat_apr_probabilities[
                        chain_index, :length
                    ]
                offset = end
                chain_index += 1
            graph_residue_count = int(node_mask[sample_index].sum().item())
            if offset != graph_residue_count:
                raise ValueError(
                    f"ESM/PDB residue mismatch in batch row {sample_index}: "
                    f"embeddings={offset}, graph={graph_residue_count}"
                )
        return sequence_output, apr_output

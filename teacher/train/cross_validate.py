from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from teacher.my_loader.data import APRDataset, AggregatingPeptideDataset, AmyloidStructureDataset, BalancedBinaryBatchSampler, collate_apr, collate_peptides, collate_structures
from teacher.model.apr_regions import apr_region_losses
from teacher.model.models import AmyloidAPRModel
from teacher.utils.optimization import GRADIENT_CLIPPING_MODES, clip_model_gradients, trainable_parameters
from teacher.utils.transfer import load_peptide_transfer_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random K-fold baseline for the Amyloid/APR multi-task model"
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "random10")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "splits" / "random10_seed2026" / "fold_manifest.json",
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--task",
        choices=["all", "sequence", "peptide", "apr", "structure"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--apr-validation-batch-size",
        type=int,
        default=None,
        help="Optional smaller APR validation batch; does not change training updates",
    )
    parser.add_argument("--structure-batch-size", type=int, default=2)
    parser.add_argument(
        "--sequence-backbone",
        choices=["esm2_t6_8m", "esm2_t12_35m", "esm2_t30_150m", "protbert", "tiny"],
        default="esm2_t6_8m",
    )
    parser.add_argument("--sequence-hidden-size", type=int, default=320)
    parser.add_argument(
        "--sequence-projection-mode",
        choices=["mlp", "identity", "residual"],
        default="mlp",
        help=(
            "Projection applied to frozen token embeddings. identity avoids a "
            "random square projection when backbone and hidden sizes match."
        ),
    )
    parser.add_argument("--structure-hidden-size", type=int, default=192)
    parser.add_argument("--structure-layers", type=int, default=4)
    parser.add_argument("--structure-message-dropout", type=float, default=0.1)
    parser.add_argument("--structure-head-dropout", type=float, default=0.2)
    parser.add_argument(
        "--structure-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--structure-layer-mix",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--structure-input-skip",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--structure-pooling",
        choices=["meanmax", "meanmaxstd", "meanmaxstdattn"],
        default="meanmax",
    )
    parser.add_argument("--structure-graph-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--structure-edge-weighting",
        choices=["binary", "gaussian", "inverse"],
        default="binary",
    )
    parser.add_argument("--structure-distance-scale", type=float, default=4.0)
    parser.add_argument(
        "--structure-normalize-position",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--structure-geometric-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--freeze-sequence-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequence-lora-rank", type=int, default=0)
    parser.add_argument("--sequence-lora-alpha", type=int, default=8)
    parser.add_argument("--sequence-lora-dropout", type=float, default=0.05)
    parser.add_argument("--sequence-lora-last-n-layers", type=int, default=2)
    parser.add_argument(
        "--sequence-lora-target-projections",
        default="query,value",
        help="Comma-separated ESM attention projections for LoRA",
    )
    parser.add_argument(
        "--sequence-backbone-learning-rate",
        type=float,
        default=None,
        help="Optional lower learning rate for trainable ESM/LoRA parameters",
    )
    parser.add_argument(
        "--sequence-model-path",
        type=Path,
        default=None,
        help="Optional local Hugging Face model directory (avoids network access)",
    )
    parser.add_argument(
        "--peptide-transfer-checkpoint",
        type=Path,
        default=None,
        help=(
            "Single full-data Peptide checkpoint used to initialize the shared "
            "encoder, Peptide adapter/CNN/features, and local propensity head"
        ),
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help="Optional disk cache for frozen per-residue sequence embeddings",
    )
    parser.add_argument(
        "--residue-feature-table",
        type=Path,
        default=None,
        help="Portable JSON lookup table for optional sequence-derived residue features",
    )
    parser.add_argument(
        "--residue-feature-sets",
        default="",
        help="Comma-separated optional sets such as aaindex_pca15,vhse8,zscale5",
    )
    parser.add_argument("--residue-feature-gate-init", type=float, default=-2.0)
    parser.add_argument("--residue-feature-dropout", type=float, default=0.0)
    parser.add_argument(
        "--peptide-residue-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fuse selected residue descriptors into the isolated Peptide adapter",
    )
    parser.add_argument(
        "--peptide-backbone-gradients",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Peptide losses to update trainable ESM/LoRA backbone parameters",
    )
    parser.add_argument(
        "--peptide-short-cnn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a multi-scale amino-acid CNN branch for short-peptide motifs",
    )
    parser.add_argument("--peptide-short-cnn-hidden-size", type=int, default=96)
    parser.add_argument(
        "--peptide-short-cnn-kernels",
        default="3,5,7",
        help="Comma-separated odd convolution kernels for the short-peptide branch",
    )
    parser.add_argument("--peptide-short-cnn-dropout", type=float, default=0.1)
    parser.add_argument("--peptide-short-cnn-gate-init", type=float, default=-1.5)
    parser.add_argument(
        "--peptide-length-conditioned-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Learn length-dependent multipliers for Peptide CNN and residue features",
    )
    parser.add_argument("--peptide-length-gate-hidden-size", type=int, default=16)
    parser.add_argument(
        "--peptide-output-fusion",
        choices=["main", "fixed_average", "length_gate"],
        default="main",
        help="Probability-level fusion of main and local propensity Peptide heads",
    )
    parser.add_argument(
        "--peptide-output-main-weight",
        type=float,
        default=0.5,
        help=(
            "Main-head probability weight used by fixed_average Peptide output "
            "fusion; 0.5 preserves the original equal average"
        ),
    )
    parser.add_argument(
        "--peptide-output-detach-propensity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop the Peptide classification loss at the local propensity output "
            "while retaining probability-level fusion"
        ),
    )
    parser.add_argument("--peptide-output-fusion-hidden-size", type=int, default=8)
    parser.add_argument(
        "--peptide-attention-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a bounded attention-pooling correction to Peptide logits",
    )
    parser.add_argument("--peptide-attention-residual-hidden-size", type=int, default=32)
    parser.add_argument("--peptide-attention-residual-heads", type=int, default=2)
    parser.add_argument("--peptide-attention-residual-dropout", type=float, default=0.2)
    parser.add_argument("--peptide-attention-residual-max-delta", type=float, default=0.5)
    parser.add_argument(
        "--peptide-global-physchem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a bounded neural residual from composition and dipeptide features",
    )
    parser.add_argument("--peptide-global-physchem-hidden-size", type=int, default=128)
    parser.add_argument("--peptide-global-physchem-dropout", type=float, default=0.2)
    parser.add_argument("--peptide-global-physchem-max-delta", type=float, default=1.0)
    parser.add_argument(
        "--peptide-length6-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a bounded position-specific neural residual for six-residue peptides",
    )
    parser.add_argument("--peptide-length6-hidden-size", type=int, default=64)
    parser.add_argument("--peptide-length6-dropout", type=float, default=0.2)
    parser.add_argument("--peptide-length6-max-delta", type=float, default=1.0)
    parser.add_argument(
        "--peptide-local-propensity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict a multi-scale residue propensity map from Peptide supervision",
    )
    parser.add_argument("--peptide-local-propensity-hidden-size", type=int, default=64)
    parser.add_argument("--peptide-local-propensity-kernels", default="3,5,7")
    parser.add_argument("--peptide-local-propensity-dropout", type=float, default=0.1)
    parser.add_argument(
        "--sequence-channel-dropout",
        type=float,
        default=0.0,
        help="Training-only channel dropout on projected per-residue representations",
    )
    parser.add_argument(
        "--structure-sequence-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fuse residue embeddings from the sequence backbone into the PDB graph",
    )
    parser.add_argument(
        "--structure-apr-probabilities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fuse predicted APR probability as one additional PDB graph node feature",
    )
    parser.add_argument(
        "--structure-apr-gradient-beta",
        type=float,
        default=0.0,
        help="Fraction of Structure loss gradient passed through APR probabilities",
    )
    parser.add_argument(
        "--structure-apr-gradient-warmup-epochs",
        type=int,
        default=0,
        help="Use beta=0 for this many initial epochs before enabling configured beta",
    )
    parser.add_argument(
        "--structure-peptide-evidence",
        type=Path,
        default=None,
        help="Precomputed fold-aligned OOF Peptide residue evidence JSON",
    )
    parser.add_argument(
        "--structure-peptide-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use detached Peptide evidence as a bounded structure attention prior",
    )
    parser.add_argument(
        "--structure-peptide-attention-alpha-init", type=float, default=0.1
    )
    parser.add_argument(
        "--structure-peptide-attention-alpha-max", type=float, default=1.0
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
        help="Optimizer learning-rate schedule applied per update",
    )
    parser.add_argument(
        "--learning-rate-warmup-fraction",
        type=float,
        default=0.10,
        help="Fraction of the configured maximum training horizon used for warm-up",
    )
    parser.add_argument(
        "--learning-rate-min-factor",
        type=float,
        default=0.05,
        help="Final learning-rate factor for warmup_cosine",
    )
    parser.add_argument("--peptide-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--peptide-label-smoothing",
        type=float,
        default=0.0,
        help="Symmetric label smoothing for Peptide BCE; zero preserves hard labels",
    )
    parser.add_argument("--peptide-focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--peptide-rdrop-weight",
        type=float,
        default=0.0,
        help="Symmetric Bernoulli KL weight for two dropout views of each Peptide",
    )
    parser.add_argument(
        "--peptide-stepwise-dropout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a distinct reproducible dropout seed for every Peptide train step",
    )
    parser.add_argument("--peptide-soft-mcc-weight", type=float, default=0.0)
    parser.add_argument("--peptide-pairwise-rank-weight", type=float, default=0.0)
    parser.add_argument("--peptide-pairwise-rank-temperature", type=float, default=1.0)
    parser.add_argument(
        "--peptide-mutation-pair-rank-weight",
        type=float,
        default=0.0,
        help="Fold-local ranking loss weight for opposite-label Hamming-1 pairs",
    )
    parser.add_argument(
        "--peptide-mutation-pair-consistency-weight",
        type=float,
        default=0.0,
        help="Fold-local embedding consistency weight for same-label Hamming-1 pairs",
    )
    parser.add_argument("--peptide-mutation-pair-margin", type=float, default=0.0)
    parser.add_argument("--peptide-mutation-pair-temperature", type=float, default=1.0)
    parser.add_argument(
        "--peptide-mutation-pair-lengths",
        default="6",
        help="Comma-separated Peptide lengths used to construct fold-local Hamming-1 pairs",
    )
    parser.add_argument(
        "--peptide-auxiliary-workbook",
        type=Path,
        default=None,
        help=(
            "Optional workbook of positive-only APR-derived peptide records. "
            "Records are appended only to each fold's training loader."
        ),
    )
    parser.add_argument(
        "--apr-positive-contrastive-workbook",
        type=Path,
        default=None,
        help=(
            "Optional APR-positive-only peptide workbook. Records are kept in a "
            "separate train-only loader and never enter Peptide BCE or validation."
        ),
    )
    parser.add_argument(
        "--apr-positive-contrastive-weight",
        type=float,
        default=0.0,
        help="Low weight for APR-positive representation ranking loss",
    )
    parser.add_argument(
        "--apr-positive-contrastive-temperature",
        type=float,
        default=0.20,
        help="Temperature for APR-positive representation InfoNCE loss",
    )
    parser.add_argument(
        "--peptide-apr-fusion",
        choices=[
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
        ],
        default="none",
        help="Optional explicit residue-propensity path from Peptide supervision to APR logits",
    )
    parser.add_argument("--peptide-propensity-loss-weight", type=float, default=1.0)
    parser.add_argument("--peptide-propensity-consistency-weight", type=float, default=0.0)
    parser.add_argument("--peptide-propensity-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--peptide-apr-beta-init", type=float, default=0.1)
    parser.add_argument("--peptide-apr-beta-max", type=float, default=1.0)
    parser.add_argument("--peptide-apr-window-size", type=int, default=7)
    parser.add_argument(
        "--peptide-apr-feature-residual-hidden-size", type=int, default=32
    )
    parser.add_argument(
        "--peptide-apr-fusion-warmup-epochs",
        type=int,
        default=0,
        help="Keep the Peptide-to-APR correction disabled for these APR epochs",
    )
    parser.add_argument(
        "--peptide-apr-fusion-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp the bounded Peptide-to-APR correction over these epochs",
    )
    parser.add_argument(
        "--peptide-apr-late-fusion-epochs",
        type=int,
        default=0,
        help=(
            "After APR base training, freeze both branches and fit only the "
            "Peptide-to-APR beta for this many epochs"
        ),
    )
    parser.add_argument(
        "--peptide-apr-late-fusion-learning-rate",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--peptide-apr-pretrain-epochs",
        type=int,
        default=0,
        help=(
            "Optional fold-local pretraining of the isolated Peptide classifier and "
            "residue-propensity branch before APR training"
        ),
    )
    parser.add_argument("--apr-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--apr-context-kernel",
        type=int,
        default=0,
        help="Optional odd kernel size for a residual local APR context head",
    )
    parser.add_argument(
        "--apr-local-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a zero-initialized bounded local-composition residual to APR logits",
    )
    parser.add_argument("--apr-local-residual-kernels", default="3,7,15")
    parser.add_argument("--apr-local-residual-hidden-size", type=int, default=32)
    parser.add_argument("--apr-local-residual-dropout", type=float, default=0.1)
    parser.add_argument("--apr-local-residual-max-delta", type=float, default=0.5)
    parser.add_argument("--apr-local-residual-penalty", type=float, default=0.01)
    parser.add_argument(
        "--apr-region-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the multi-scale candidate-region APR residual head",
    )
    parser.add_argument("--apr-region-lengths", default="6,8,10,12,16,20")
    parser.add_argument("--apr-region-dropout", type=float, default=0.1)
    parser.add_argument("--apr-region-max-residue-delta", type=float, default=0.5)
    parser.add_argument("--apr-region-loss-weight", type=float, default=1.0)
    parser.add_argument("--apr-region-consistency-weight", type=float, default=0.2)
    parser.add_argument("--apr-region-rank-weight", type=float, default=0.1)
    parser.add_argument("--apr-region-positive-fraction", type=float, default=0.8)
    parser.add_argument("--apr-region-max-candidates-per-class", type=int, default=64)
    parser.add_argument("--apr-region-rank-margin", type=float, default=0.2)
    parser.add_argument("--apr-region-rank-temperature", type=float, default=1.0)
    parser.add_argument(
        "--apr-positive-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier for valid APR-positive residue BCE terms. Values below 1 "
            "counter the APR residue positive-class skew without changing labels."
        ),
    )
    parser.add_argument(
        "--apr-sequence-balanced-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Give each APR sequence equal loss mass instead of weighting residues globally",
    )
    parser.add_argument(
        "--apr-focal-gamma",
        type=float,
        default=0.0,
        help="Focal BCE gamma for hard APR residues; zero is ordinary BCE",
    )
    parser.add_argument(
        "--apr-head-mode",
        choices=["mlp", "linear"],
        default="mlp",
        help="APR token classifier on top of the shared sequence representation",
    )
    parser.add_argument(
        "--apr-hard-negative-gamma",
        type=float,
        default=0.0,
        help=(
            "Detached within-batch hard-negative reweighting exponent. Negative "
            "residue weights are normalized to preserve their total loss mass."
        ),
    )
    parser.add_argument(
        "--apr-mixed-sequence-weight",
        type=float,
        default=1.0,
        help="Loss multiplier for training sequences containing both APR-positive and negative labels",
    )
    parser.add_argument(
        "--apr-mixed-negative-weight",
        type=float,
        default=1.0,
        help=(
            "Loss multiplier applied only to negative residues originating from "
            "mixed APR-positive/negative sequences"
        ),
    )
    parser.add_argument(
        "--apr-training-window-flank",
        type=int,
        default=None,
        help=(
            "Training-only local context on each labeled APR run. The value is the "
            "number of true sequence residues retained on each side; validation stays full-length."
        ),
    )
    parser.add_argument(
        "--apr-training-window-full-mix",
        type=float,
        default=0.0,
        help=(
            "Fraction of APR label weight kept on full training sequences when local "
            "windows are enabled; the remainder is assigned once across the windows."
        ),
    )
    parser.add_argument(
        "--apr-soft-mcc-weight",
        type=float,
        default=0.0,
        help="Training-only weight for a differentiable APR MCC auxiliary loss",
    )
    parser.add_argument(
        "--apr-smoothness-weight",
        type=float,
        default=0.0,
        help="Training-only probability smoothness weight within adjacent same-label APR regions",
    )
    parser.add_argument(
        "--apr-pairwise-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only weight for ranking labeled APR-positive residues above "
            "labeled negatives from the same sequence"
        ),
    )
    parser.add_argument(
        "--apr-pairwise-rank-temperature",
        type=float,
        default=1.0,
        help="Temperature for the within-sequence APR pairwise logistic loss",
    )
    parser.add_argument("--structure-loss-weight", type=float, default=1.0)
    parser.add_argument("--structure-positive-weight-scale", type=float, default=1.0)
    parser.add_argument("--structure-focal-gamma", type=float, default=0.0)
    parser.add_argument("--structure-label-smoothing", type=float, default=0.0)
    parser.add_argument("--structure-pairwise-rank-weight", type=float, default=0.0)
    parser.add_argument("--structure-pairwise-rank-temperature", type=float, default=1.0)
    parser.add_argument("--structure-rank-memory-size", type=int, default=0)
    parser.add_argument("--structure-rank-hard-k", type=int, default=0)
    parser.add_argument(
        "--structure-balanced-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--epoch-anchor",
        choices=["max", "apr"],
        default="max",
        help="Define an epoch by the largest loader or one complete APR pass",
    )
    parser.add_argument(
        "--gradient-clipping",
        choices=GRADIENT_CLIPPING_MODES,
        default="separate",
        help="separate prevents sequence gradients from scaling structure gradients",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--parameter-ema-decay",
        type=float,
        default=0.0,
        help=(
            "Exponential moving-average decay for trainable model parameters; "
            "zero disables EMA and preserves the original training behavior"
        ),
    )
    parser.add_argument(
        "--parameter-ema-start-epoch",
        type=int,
        default=1,
        help="First epoch whose optimizer updates contribute to parameter EMA",
    )
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-aggregating-negatives", action="store_true")
    parser.add_argument(
        "--apr-weak-negatives",
        type=Path,
        default=None,
        help="Optional training-only APR weak-negative workbook",
    )
    parser.add_argument("--apr-weak-negative-weight", type=float, default=0.25)
    parser.add_argument(
        "--apr-mapped-negative-weight",
        type=float,
        default=1.0,
        help=(
            "Training-loss multiplier for experimental negative residues mapped "
            "into protein context; standalone negative peptides remain at weight 1"
        ),
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select the epoch on an inner validation split, then retrain on the full outer fold",
    )
    parser.add_argument("--early-stopping-validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--peptide-inner-group-disjoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep Peptide protein_id groups disjoint in the inner validation "
            "split used for early stopping"
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=5)
    parser.add_argument(
        "--early-stopping-selection-start-epoch",
        type=int,
        default=1,
        help="First epoch eligible to become the full-fold retraining length",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping-monitor",
        choices=[
            "total_loss",
            "peptide_loss",
            "peptide_propensity_loss",
            "peptide_mcc_at_0_5",
            "peptide_transfer_score",
            "apr_loss",
            "apr_mcc_at_0_5",
            "structure_loss",
            "structure_mcc_at_0_5",
        ],
        default="apr_loss",
        help=(
            "Inner-validation metric used to select the epoch. Metrics ending in "
            "mcc_at_0_5 keep the decision threshold fixed at 0.5."
        ),
    )
    parser.add_argument("--save-fold-checkpoints", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_parameter_ema(
    shadow: dict[str, torch.Tensor],
    parameters: dict[str, nn.Parameter],
    decay: float,
) -> None:
    with torch.no_grad():
        if not shadow:
            shadow.update(
                {
                    name: parameter.detach().clone()
                    for name, parameter in parameters.items()
                }
            )
            return
        for name, parameter in parameters.items():
            shadow[name].lerp_(parameter.detach(), 1.0 - decay)


def warmup_cosine_factor(
    step: int,
    total_steps: int,
    warmup_fraction: float,
    min_factor: float,
) -> float:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))
    if step < warmup_steps:
        return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
    progress = min(
        1.0,
        (step - warmup_steps) / max(total_steps - warmup_steps, 1),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def structure_apr_beta(args: argparse.Namespace, epoch: int | None = None) -> float:
    if not args.structure_apr_probabilities:
        return 0.0
    if epoch is not None and epoch <= args.structure_apr_gradient_warmup_epochs:
        return 0.0
    return float(args.structure_apr_gradient_beta)


def peptide_apr_fusion_scale(args: argparse.Namespace, epoch: int) -> float:
    if args.peptide_apr_fusion == "none":
        return 0.0
    if args.peptide_apr_late_fusion_epochs > 0:
        return 0.0
    if epoch <= args.peptide_apr_fusion_warmup_epochs:
        return 0.0
    if args.peptide_apr_fusion_ramp_epochs == 0:
        return 1.0
    ramp_epoch = epoch - args.peptide_apr_fusion_warmup_epochs
    return min(1.0, ramp_epoch / args.peptide_apr_fusion_ramp_epochs)


def masked_apr_bce_components(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
    positive_weight: float = 1.0,
    sequence_balanced: bool = False,
    focal_gamma: float = 0.0,
    hard_negative_gamma: float = 0.0,
    mixed_sequence_weight: float = 1.0,
    mixed_negative_weight: float = 1.0,
    source_mixed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = targets != -100
    if not torch.any(valid):
        zero = logits.sum() * 0.0
        return zero, zero.detach()

    safe_targets = targets.masked_fill(~valid, 0.0)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )
    if focal_gamma > 0:
        probabilities = torch.sigmoid(logits)
        target_probabilities = torch.where(
            safe_targets == 1, probabilities, 1.0 - probabilities
        )
        losses = losses * (1.0 - target_probabilities).pow(focal_gamma)
    if hard_negative_gamma > 0:
        negative_mask = valid & (safe_targets == 0)
        if torch.any(negative_mask):
            negative_label_weights = label_weights * negative_mask.to(
                label_weights.dtype
            )
            hardness = torch.sigmoid(logits.detach()).pow(hard_negative_gamma)
            mean_hardness = (
                hardness * negative_label_weights
            ).sum() / negative_label_weights.sum().clamp_min(1e-12)
            negative_multipliers = hardness / mean_hardness.clamp_min(1e-12)
            losses = torch.where(
                negative_mask, losses * negative_multipliers, losses
            )
    has_positive = ((targets == 1) & valid).any(dim=1)
    has_negative = ((targets == 0) & valid).any(dim=1)
    mixed_rows = has_positive & has_negative
    if source_mixed is not None:
        if source_mixed.shape != mixed_rows.shape:
            raise ValueError("source_mixed must have one value per APR batch row")
        mixed_rows = source_mixed.to(device=targets.device, dtype=torch.bool)
    mixed_negative_mask = (
        valid & (safe_targets == 0) & mixed_rows.unsqueeze(1)
    )
    class_weights = torch.where(
        safe_targets == 1,
        torch.full_like(safe_targets, positive_weight),
        torch.ones_like(safe_targets),
    )
    class_weights = torch.where(
        mixed_negative_mask,
        class_weights * mixed_negative_weight,
        class_weights,
    )
    weights = label_weights * class_weights * valid.to(label_weights.dtype)
    row_weights = torch.where(
        has_positive & has_negative,
        torch.full(
            (targets.shape[0],),
            mixed_sequence_weight,
            dtype=label_weights.dtype,
            device=label_weights.device,
        ),
        torch.ones(
            targets.shape[0], dtype=label_weights.dtype, device=label_weights.device
        ),
    )
    if not sequence_balanced:
        weights = weights * row_weights.unsqueeze(1)
        return (losses * weights).sum(), weights.sum()

    row_weight_sums = weights.sum(dim=1)
    row_losses = (losses * weights).sum(dim=1) / row_weight_sums.clamp_min(1e-12)
    active_rows = row_weight_sums > 0
    active_row_weights = row_weights[active_rows]
    return (
        (row_losses[active_rows] * active_row_weights).sum(),
        active_row_weights.sum(),
    )


def masked_apr_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
    positive_weight: float = 1.0,
    sequence_balanced: bool = False,
    focal_gamma: float = 0.0,
    hard_negative_gamma: float = 0.0,
    mixed_sequence_weight: float = 1.0,
    mixed_negative_weight: float = 1.0,
    source_mixed: torch.Tensor | None = None,
) -> torch.Tensor:
    numerator, denominator = masked_apr_bce_components(
        logits,
        targets,
        label_weights,
        positive_weight=positive_weight,
        sequence_balanced=sequence_balanced,
        focal_gamma=focal_gamma,
        hard_negative_gamma=hard_negative_gamma,
        mixed_sequence_weight=mixed_sequence_weight,
        mixed_negative_weight=mixed_negative_weight,
        source_mixed=source_mixed,
    )
    return numerator / denominator.clamp_min(1e-12)


def masked_apr_soft_mcc(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
) -> torch.Tensor:
    valid = targets != -100
    if not torch.any(valid):
        return logits.sum() * 0.0
    probabilities = torch.sigmoid(logits[valid])
    labels = targets[valid]
    weights = label_weights[valid]
    true_positive = (weights * probabilities * labels).sum()
    true_negative = (weights * (1.0 - probabilities) * (1.0 - labels)).sum()
    false_positive = (weights * probabilities * (1.0 - labels)).sum()
    false_negative = (weights * (1.0 - probabilities) * labels).sum()
    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = torch.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
        + 1e-8
    )
    return 1.0 - numerator / denominator


def masked_apr_smoothness(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
) -> torch.Tensor:
    left_targets = targets[:, :-1]
    right_targets = targets[:, 1:]
    valid_pairs = (
        (left_targets != -100)
        & (right_targets != -100)
        & (left_targets == right_targets)
    )
    if not torch.any(valid_pairs):
        return logits.sum() * 0.0
    pair_weights = torch.minimum(label_weights[:, :-1], label_weights[:, 1:])
    pair_weights = pair_weights * valid_pairs.to(pair_weights.dtype)
    probabilities = torch.sigmoid(logits)
    squared_differences = (probabilities[:, 1:] - probabilities[:, :-1]).pow(2)
    return (squared_differences * pair_weights).sum() / pair_weights.sum().clamp_min(
        1e-12
    )


def masked_apr_pairwise_rank(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_weights: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Rank labeled positives above negatives within each mixed APR sequence."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    row_losses: list[torch.Tensor] = []
    for row_logits, row_targets, row_weights in zip(logits, targets, label_weights):
        positive = (row_targets == 1) & (row_weights > 0)
        negative = (row_targets == 0) & (row_weights > 0)
        if not torch.any(positive) or not torch.any(negative):
            continue
        differences = (
            row_logits[positive].unsqueeze(1) - row_logits[negative].unsqueeze(0)
        ) / temperature
        pair_weights = (
            row_weights[positive].unsqueeze(1) * row_weights[negative].unsqueeze(0)
        )
        pair_losses = nn.functional.softplus(-differences)
        row_losses.append(
            (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)
        )
    if not row_losses:
        return logits.sum() * 0.0
    return torch.stack(row_losses).mean()


def masked_logmeanexp_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    lengths = mask.sum(dim=1).clamp_min(1).to(logits.dtype)
    return torch.logsumexp(values, dim=1) - torch.log(lengths)


def peptide_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    soft_mcc_weight: float = 0.0,
    pairwise_rank_weight: float = 0.0,
    pairwise_rank_temperature: float = 1.0,
) -> torch.Tensor:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    bce_targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, bce_targets, reduction="none"
    )
    if focal_gamma > 0:
        probabilities = torch.sigmoid(logits)
        target_probabilities = torch.where(
            labels >= 0.5, probabilities, 1.0 - probabilities
        )
        bce = bce * (1.0 - target_probabilities).pow(focal_gamma)
    loss = bce.mean()

    if soft_mcc_weight > 0:
        probabilities = torch.sigmoid(logits)
        positives = labels
        negatives = 1.0 - labels
        true_positive = (probabilities * positives).sum()
        false_positive = (probabilities * negatives).sum()
        false_negative = ((1.0 - probabilities) * positives).sum()
        true_negative = ((1.0 - probabilities) * negatives).sum()
        numerator = true_positive * true_negative - false_positive * false_negative
        denominator = torch.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
            + 1e-8
        )
        loss = loss + soft_mcc_weight * (1.0 - numerator / denominator)

    if pairwise_rank_weight > 0:
        positive_logits = logits[labels >= 0.5]
        negative_logits = logits[labels < 0.5]
        if positive_logits.numel() and negative_logits.numel():
            margins = (
                positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)
            ) / pairwise_rank_temperature
            loss = loss + pairwise_rank_weight * nn.functional.softplus(
                -margins
            ).mean()
    return loss


def symmetric_bernoulli_kl(
    logits_a: torch.Tensor, logits_b: torch.Tensor
) -> torch.Tensor:
    """Symmetric KL between Bernoulli distributions parameterized by logits."""
    probability_a = torch.sigmoid(logits_a)
    probability_b = torch.sigmoid(logits_b)
    kl_a_b = (
        probability_a * (nn.functional.logsigmoid(logits_a) - nn.functional.logsigmoid(logits_b))
        + (1.0 - probability_a)
        * (nn.functional.logsigmoid(-logits_a) - nn.functional.logsigmoid(-logits_b))
    )
    kl_b_a = (
        probability_b * (nn.functional.logsigmoid(logits_b) - nn.functional.logsigmoid(logits_a))
        + (1.0 - probability_b)
        * (nn.functional.logsigmoid(-logits_b) - nn.functional.logsigmoid(-logits_a))
    )
    return 0.5 * (kl_a_b + kl_b_a).mean()


def local_propensity_smoothness(
    residue_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Encourage contiguous evidence without supervising unknown residue labels."""
    if residue_logits.ndim != 2 or mask.shape != residue_logits.shape:
        raise ValueError("Local propensity logits and mask must have matching rank-2 shapes")
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    if not valid_pairs.any():
        return residue_logits.sum() * 0.0
    probabilities = torch.sigmoid(residue_logits)
    squared_difference = (probabilities[:, 1:] - probabilities[:, :-1]).pow(2)
    weights = valid_pairs.to(squared_difference.dtype)
    return (squared_difference * weights).sum() / weights.sum().clamp_min(1.0)


def apr_positive_representation_loss(
    apr_embeddings: torch.Tensor,
    peptide_embeddings: torch.Tensor,
    peptide_labels: torch.Tensor,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Align APR-positive embeddings with labeled Peptide positives.

    This is deliberately not a classification loss: APR-derived records only
    provide positive anchors in a separate loader.  Original Peptide records
    supply the positive/negative reference sets, while their labels continue
    to be consumed by the ordinary Peptide BCE path only.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if apr_embeddings.ndim != 2 or peptide_embeddings.ndim != 2:
        raise ValueError("contrastive embeddings must be rank-2 tensors")
    positive = peptide_labels >= 0.5
    negative = ~positive
    if not torch.any(positive) or not torch.any(negative):
        return apr_embeddings.sum() * 0.0
    anchors = nn.functional.normalize(apr_embeddings, dim=-1)
    references = nn.functional.normalize(peptide_embeddings, dim=-1)
    similarities = anchors @ references.transpose(0, 1) / temperature
    # Every reliable positive in the Peptide batch is a valid numerator. The
    # APR-derived anchor itself is never assigned a BCE target.
    positive_logits = similarities[:, positive]
    all_logits = similarities
    return (
        -torch.logsumexp(positive_logits, dim=1)
        + torch.logsumexp(all_logits, dim=1)
    ).mean()


def peptide_mutation_pair_losses(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank opposite-label point mutants and align same-label neighbors."""
    if margin < 0:
        raise ValueError("Mutation-pair margin must be non-negative")
    if temperature <= 0:
        raise ValueError("Mutation-pair temperature must be positive")
    opposite = labels_a != labels_b
    same = ~opposite
    if torch.any(opposite):
        direction = torch.where(
            labels_a[opposite] > labels_b[opposite], 1.0, -1.0
        )
        signed_difference = direction * (
            logits_a[opposite] - logits_b[opposite]
        )
        rank_loss = nn.functional.softplus(
            (margin - signed_difference) / temperature
        ).mean()
    else:
        rank_loss = (logits_a.sum() + logits_b.sum()) * 0.0
    if torch.any(same):
        normalized_a = nn.functional.normalize(embeddings_a[same], dim=-1)
        normalized_b = nn.functional.normalize(embeddings_b[same], dim=-1)
        consistency_loss = (
            1.0 - (normalized_a * normalized_b).sum(dim=-1)
        ).mean()
    else:
        consistency_loss = (embeddings_a.sum() + embeddings_b.sum()) * 0.0
    return rank_loss, consistency_loss


def structure_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    pairwise_rank_weight: float = 0.0,
    pairwise_rank_temperature: float = 1.0,
    positive_logit_memory: torch.Tensor | None = None,
    negative_logit_memory: torch.Tensor | None = None,
    rank_hard_k: int = 0,
) -> torch.Tensor:
    targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    if focal_gamma > 0:
        hard_targets = (labels >= 0.5).to(logits.dtype)
        probabilities = torch.sigmoid(logits)
        target_probabilities = torch.where(
            hard_targets == 1,
            probabilities,
            1.0 - probabilities,
        )
        losses = losses * (1.0 - target_probabilities).pow(focal_gamma)
    loss = losses.mean()
    if pairwise_rank_weight > 0:
        positive_logits = logits[labels >= 0.5]
        negative_logits = logits[labels < 0.5]
        rank_losses: list[torch.Tensor] = []
        if positive_logits.numel() and negative_logits.numel():
            margins = (
                positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0)
            ) / pairwise_rank_temperature
            rank_losses.append(nn.functional.softplus(-margins).mean())
        if positive_logits.numel() and negative_logit_memory is not None and negative_logit_memory.numel():
            references = negative_logit_memory
            if rank_hard_k > 0 and references.numel() > rank_hard_k:
                references = torch.topk(references, rank_hard_k).values
            margins = (
                positive_logits.unsqueeze(1) - references.unsqueeze(0)
            ) / pairwise_rank_temperature
            rank_losses.append(nn.functional.softplus(-margins).mean())
        if negative_logits.numel() and positive_logit_memory is not None and positive_logit_memory.numel():
            references = positive_logit_memory
            if rank_hard_k > 0 and references.numel() > rank_hard_k:
                references = torch.topk(
                    references, rank_hard_k, largest=False
                ).values
            margins = (
                references.unsqueeze(1) - negative_logits.unsqueeze(0)
            ) / pairwise_rank_temperature
            rank_losses.append(nn.functional.softplus(-margins).mean())
        if rank_losses:
            loss = loss + pairwise_rank_weight * torch.stack(rank_losses).mean()
    return loss


def pretrain_peptide_branch(
    model: AmyloidAPRModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    phase: str,
) -> None:
    if epochs <= 0:
        return
    if model.sequence.peptide_adapter is None:
        raise ValueError(
            "Peptide branch pretraining requires --peptide-apr-fusion"
        )
    peptide_modules = [
        model.sequence.peptide_adapter,
        model.sequence.peptide_head,
    ]
    if model.sequence.peptide_short_cnn is not None:
        peptide_modules.extend(
            [
                model.sequence.peptide_short_cnn,
                model.sequence.peptide_short_cnn_projection,
            ]
        )
    if model.sequence.peptide_attention_residual is not None:
        peptide_modules.append(model.sequence.peptide_attention_residual)
    if model.sequence.peptide_global_physchem is not None:
        peptide_modules.append(model.sequence.peptide_global_physchem)
    if model.sequence.peptide_length6_residual is not None:
        peptide_modules.append(model.sequence.peptide_length6_residual)
    if model.sequence.aggregation_propensity_head is not None:
        peptide_modules.append(model.sequence.aggregation_propensity_head)
    parameters = trainable_parameters(peptide_modules)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and device.type == "cuda"
    )
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None
    )
    for epoch in range(1, epochs + 1):
        model.train()
        classification_sum = 0.0
        propensity_sum = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                output = model.forward_sequence(batch["sequences"])
                labels = batch["labels"].to(device)
                classification_loss = nn.functional.binary_cross_entropy_with_logits(
                    output["peptide_logits"], labels
                )
                propensity_loss = nn.functional.binary_cross_entropy_with_logits(
                    output["propensity_peptide_logits"], labels
                )
                loss = classification_loss + propensity_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_model_gradients(
                model,
                mode=args.gradient_clipping,
                max_norm=args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            classification_sum += float(classification_loss.detach().cpu())
            propensity_sum += float(propensity_loss.detach().cpu())
        print(
            f"phase={phase} epoch={epoch:03d}/{epochs:03d} "
            f"peptide_classification_loss="
            f"{classification_sum / max(len(loader), 1):.5f} "
            f"peptide_propensity_loss="
            f"{propensity_sum / max(len(loader), 1):.5f}",
            flush=True,
        )
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)


def fit_peptide_apr_late_fusion(
    model: AmyloidAPRModel,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    epochs = args.peptide_apr_late_fusion_epochs
    if epochs <= 0:
        return {}
    beta_logit = model.sequence.peptide_apr_beta_logit
    if beta_logit is None:
        raise ValueError("Late Peptide-to-APR fusion requires a fusion mode")

    requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    beta_logit.requires_grad_(True)
    late_fusion_parameters: list[nn.Parameter] = [beta_logit]
    feature_residual_head = model.sequence.peptide_apr_feature_residual_head
    if feature_residual_head is not None:
        for parameter in feature_residual_head.parameters():
            parameter.requires_grad_(True)
            late_fusion_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        late_fusion_parameters,
        lr=args.peptide_apr_late_fusion_learning_rate,
        weight_decay=0.0,
    )
    model.sequence.set_peptide_apr_fusion_scale(1.0)
    model.eval()
    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_sequence(
                batch["sequences"], isolate_peptide_rng=True
            )
            loss = masked_apr_bce(
                output["apr_logits"],
                batch["targets"].to(device),
                batch["label_weights"].to(device),
                positive_weight=args.apr_positive_weight,
                sequence_balanced=args.apr_sequence_balanced_loss,
                focal_gamma=args.apr_focal_gamma,
                hard_negative_gamma=args.apr_hard_negative_gamma,
                mixed_sequence_weight=args.apr_mixed_sequence_weight,
                mixed_negative_weight=args.apr_mixed_negative_weight,
                source_mixed=batch["source_mixed"].to(device),
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
        last_loss = loss_sum / max(len(loader), 1)
        active_beta = float(
            (
                model.sequence.peptide_apr_beta_max
                * torch.sigmoid(beta_logit.detach())
            ).cpu()
        )
        print(
            f"phase=peptide_apr_late_fusion epoch={epoch:03d}/{epochs:03d} "
            f"apr_loss={last_loss:.5f} peptide_apr_beta={active_beta:.5f}",
            flush=True,
        )

    for parameter, enabled in requires_grad.items():
        parameter.requires_grad_(enabled)
    model.train()
    return {
        "peptide_apr_late_fusion_loss": last_loss,
        "peptide_apr_beta": float(
            (
                model.sequence.peptide_apr_beta_max
                * torch.sigmoid(beta_logit.detach())
            ).cpu()
        ),
    }


def apr_finetune_loaders(
    loaders: dict[str, DataLoader], args: argparse.Namespace
) -> dict[str, DataLoader]:
    if (
        args.peptide_loss_weight == 0
        and args.peptide_propensity_loss_weight == 0
    ):
        return {name: loader for name, loader in loaders.items() if name != "peptide"}
    return loaders


def structure_positive_weight(dataset: AmyloidStructureDataset, indices: np.ndarray) -> float:
    labels = [dataset.records[int(index)].label for index in indices]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Each structure training fold must contain both labels")
    return float(negatives / positives)


def make_loader(
    dataset,
    indices,
    batch_size,
    collate_fn,
    shuffle,
    num_workers,
    balanced_binary: bool = False,
    seed: int | None = None,
):
    subset = Subset(dataset, indices)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    if balanced_binary:
        labels = [dataset.records[int(index)].label for index in indices]
        return DataLoader(
            subset,
            batch_sampler=BalancedBinaryBatchSampler(labels, batch_size),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


class PeptideMutationPairDataset(Dataset):
    def __init__(
        self,
        peptide_dataset: AggregatingPeptideDataset,
        pairs: list[tuple[int, int]],
    ):
        self.peptide_dataset = peptide_dataset
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        first, second = self.pairs[index]
        return (
            self.peptide_dataset.records[first],
            self.peptide_dataset.records[second],
        )


def collate_peptide_mutation_pairs(records) -> dict[str, Any]:
    records = list(records)
    first = [item[0] for item in records]
    second = [item[1] for item in records]
    return {
        "sequences_a": [record.sequence for record in first],
        "sequences_b": [record.sequence for record in second],
        "labels_a": torch.tensor(
            [record.label for record in first], dtype=torch.float32
        ),
        "labels_b": torch.tensor(
            [record.label for record in second], dtype=torch.float32
        ),
    }


def hamming_one_peptide_pairs(
    dataset: AggregatingPeptideDataset,
    indices: np.ndarray,
    lengths: tuple[int, ...],
) -> tuple[list[tuple[int, int]], int, int]:
    allowed_lengths = set(lengths)
    pattern_groups: dict[tuple[int, str], list[int]] = {}
    for raw_index in indices:
        index = int(raw_index)
        record = dataset.records[index]
        length = len(record.sequence)
        if length not in allowed_lengths:
            continue
        for position in range(length):
            pattern = record.sequence[:position] + "*" + record.sequence[position + 1 :]
            pattern_groups.setdefault((length, pattern), []).append(index)

    pairs: set[tuple[int, int]] = set()
    for group in pattern_groups.values():
        unique = sorted(set(group))
        for first_position in range(len(unique)):
            for second_position in range(first_position + 1, len(unique)):
                first = unique[first_position]
                second = unique[second_position]
                if dataset.records[first].sequence == dataset.records[second].sequence:
                    continue
                pairs.add((first, second))
    ordered = sorted(pairs)
    opposite = sum(
        dataset.records[first].label != dataset.records[second].label
        for first, second in ordered
    )
    return ordered, opposite, len(ordered) - opposite


def make_peptide_mutation_pair_loader(
    dataset: AggregatingPeptideDataset,
    indices: np.ndarray,
    lengths: tuple[int, ...],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, int, int]:
    pairs, opposite, same = hamming_one_peptide_pairs(dataset, indices, lengths)
    if not pairs:
        raise ValueError(
            f"No fold-local Hamming-1 Peptide pairs found for lengths={lengths}"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        PeptideMutationPairDataset(dataset, pairs),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_peptide_mutation_pairs,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return loader, opposite, same


def leakage_filtered_auxiliary_indices(
    dataset: AggregatingPeptideDataset,
    held_out_sequences: set[str],
) -> np.ndarray:
    """Exclude exact and substring-related auxiliary sequences from a holdout."""
    return np.asarray(
        [
            index
            for index, record in enumerate(dataset.records)
            if not any(
                record.sequence in held_out or held_out in record.sequence
                for held_out in held_out_sequences
            )
        ],
        dtype=np.int64,
    )


def load_fixed_splits(
    manifest_path: Path,
    datasets: dict[str, Any],
    folds: int,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Fixed split manifest not found: {manifest_path}. "
            "Run prepare_random_splits.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("folds", -1)) != folds:
        raise ValueError(
            f"Manifest has {manifest.get('folds')} folds but --folds={folds}. "
            "Use the matching --folds value or generate a new manifest."
        )
    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, dataset in datasets.items():
        config = manifest.get("datasets", {}).get(name)
        if config is None:
            raise ValueError(f"Manifest does not contain dataset {name!r}")
        fold_by_id = config["fold_by_id"]
        ids = []
        if name == "peptide":
            ids = [record.peptide_id for record in dataset.records]
        elif name == "apr":
            evaluation_rows = [
                (index, record.sequence_id)
                for index, record in enumerate(dataset.records)
                if not record.training_only
            ]
            training_only_indices = np.asarray(
                [
                    index
                    for index, record in enumerate(dataset.records)
                    if record.training_only
                ],
                dtype=np.int64,
            )
            ids = [sequence_id for _, sequence_id in evaluation_rows]
            if set(ids) != set(fold_by_id):
                raise ValueError("Manifest IDs do not match current apr evaluation database")
            result[name] = []
            for fold in range(folds):
                base_train = np.asarray(
                    [index for index, sequence_id in evaluation_rows if int(fold_by_id[sequence_id]) != fold],
                    dtype=np.int64,
                )
                base_valid = np.asarray(
                    [index for index, sequence_id in evaluation_rows if int(fold_by_id[sequence_id]) == fold],
                    dtype=np.int64,
                )
                train_indices = np.concatenate([base_train, training_only_indices])
                result[name].append((train_indices, base_valid))
            continue
        elif name == "structure":
            ids = [record.sample_id for record in dataset.records]
        if set(ids) != set(fold_by_id):
            raise ValueError(f"Manifest IDs do not match current {name} database")
        assignments = np.asarray([int(fold_by_id[item_id]) for item_id in ids])
        result[name] = [
            (np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold))
            for fold in range(folds)
        ]
    return result


def _inner_stratum(name: str, record: Any) -> int:
    if name in {"peptide", "structure"}:
        return int(record.label)
    if name == "apr":
        has_positive = bool(torch.any(record.target == 1).item())
        has_negative = bool(torch.any(record.target == 0).item())
        return int(has_positive) * 2 + int(has_negative)
    raise ValueError(f"Unsupported dataset for inner split: {name}")


def make_inner_validation_split(
    name: str,
    dataset: Any,
    outer_train_indices: np.ndarray,
    validation_fraction: float,
    seed: int,
    group_disjoint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    training_only: list[int] = []
    eligible: list[int] = []
    strata: list[int] = []
    for raw_index in outer_train_indices:
        index = int(raw_index)
        record = dataset.records[index]
        if name == "apr" and record.training_only:
            training_only.append(index)
            continue
        eligible.append(index)
        strata.append(_inner_stratum(name, record))
    if len(eligible) < 2:
        raise ValueError(f"Not enough {name} records for inner validation")
    if group_disjoint:
        if name != "peptide":
            raise ValueError("Group-disjoint inner validation is only supported for Peptide")
        groups = [
            dataset.records[index].split_group_id
            or dataset.records[index].protein_id
            for index in eligible
        ]
        if any(not group_id for group_id in groups):
            raise ValueError(
                "Peptide group-disjoint inner validation requires protein_id for every record"
            )
        unique_group_n = len(set(groups))
        requested_splits = max(2, int(round(1.0 / validation_fraction)))
        inner_folds = min(requested_splits, unique_group_n)
        splitter = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed,
        )
        eligible_array = np.asarray(eligible, dtype=np.int64)
        strata_array = np.asarray(strata, dtype=np.int64)
        groups_array = np.asarray(groups, dtype=object)
        target_size = validation_fraction * len(eligible)
        target_positive_fraction = float(np.mean(strata_array == 1))
        candidates = []
        for train_positions, valid_positions in splitter.split(
            eligible_array, strata_array, groups_array
        ):
            valid_labels = strata_array[valid_positions]
            missing_label_penalty = float(len(np.unique(valid_labels)) < 2)
            size_error = abs(len(valid_positions) - target_size) / max(target_size, 1.0)
            prevalence_error = abs(
                float(np.mean(valid_labels == 1)) - target_positive_fraction
            )
            candidates.append(
                (
                    missing_label_penalty,
                    size_error + prevalence_error,
                    train_positions,
                    valid_positions,
                )
            )
        _, _, train_positions, valid_positions = min(
            candidates, key=lambda item: item[:2]
        )
        inner_train = eligible_array[train_positions]
        inner_valid = eligible_array[valid_positions]
        train_groups = {
            dataset.records[int(index)].split_group_id
            or dataset.records[int(index)].protein_id
            for index in inner_train
        }
        valid_groups = {
            dataset.records[int(index)].split_group_id
            or dataset.records[int(index)].protein_id
            for index in inner_valid
        }
        if train_groups & valid_groups:
            raise RuntimeError("Protein groups overlap in inner validation")
        return inner_train, inner_valid
    counts = {label: strata.count(label) for label in set(strata)}
    stratify = strata if len(counts) > 1 and min(counts.values()) >= 2 else None
    inner_train, inner_valid = train_test_split(
        np.asarray(eligible, dtype=np.int64),
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    inner_train = np.concatenate(
        [inner_train, np.asarray(training_only, dtype=np.int64)]
    )
    return inner_train, np.asarray(inner_valid, dtype=np.int64)


def binary_metrics(labels: list[float], probabilities: list[float]) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    result: dict[str, float] = {
        "n": float(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    result.update(
        {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "balanced_accuracy": float((sensitivity + specificity) / 2.0),
            "predicted_positive_rate": float(np.mean(y_pred)),
            "true_positive": float(true_positive),
            "false_negative": float(false_negative),
            "true_negative": float(true_negative),
            "false_positive": float(false_positive),
        }
    )
    for name, metric in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
        try:
            result[name] = float(metric(y_true, y_prob))
        except ValueError:
            result[name] = float("nan")
    return result


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return zero-based half-open intervals for true runs in a 1D mask."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    starts = [int(indices[0])]
    ends: list[int] = []
    for previous, current in zip(indices[:-1], indices[1:]):
        if int(current) != int(previous) + 1:
            ends.append(int(previous) + 1)
            starts.append(int(current))
    ends.append(int(indices[-1]) + 1)
    return list(zip(starts, ends))


def apr_region_profile_statistics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    true_regions = contiguous_regions(targets == 1)
    predicted_regions = contiguous_regions(probabilities >= threshold)
    recalled = 0
    best_iou_sum = 0.0
    boundary_error_sum = 0.0
    boundary_match_count = 0
    for true_start, true_end in true_regions:
        best_iou = 0.0
        best_region: tuple[int, int] | None = None
        for pred_start, pred_end in predicted_regions:
            intersection = max(0, min(true_end, pred_end) - max(true_start, pred_start))
            union = max(true_end, pred_end) - min(true_start, pred_start)
            iou = intersection / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_region = (pred_start, pred_end)
        best_iou_sum += best_iou
        if best_iou >= 0.5:
            recalled += 1
        if best_region is not None and best_iou > 0:
            boundary_error_sum += 0.5 * (
                abs(best_region[0] - true_start) + abs(best_region[1] - true_end)
            )
            boundary_match_count += 1
    trusted_negative_regions = contiguous_regions(
        (targets == 0) & (probabilities >= threshold)
    )
    return {
        "true_region_count": float(len(true_regions)),
        "recalled_region_count": float(recalled),
        "best_iou_sum": best_iou_sum,
        "boundary_error_sum": boundary_error_sum,
        "boundary_match_count": float(boundary_match_count),
        "trusted_negative_false_region_count": float(
            len(trusted_negative_regions)
        ),
        "has_trusted_negative": float(np.any(targets == 0)),
    }


@torch.no_grad()
def evaluate(
    model,
    loaders,
    device: torch.device,
    args: argparse.Namespace,
    prediction_rows: dict[str, list[dict[str, Any]]] | None = None,
    fold: int | None = None,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}

    if "peptide" in loaders:
        peptide_labels: list[float] = []
        peptide_probs: list[float] = []
        propensity_probs: list[float] = []
        for batch in loaders["peptide"]:
            output = model.forward_sequence(batch["sequences"])
            probabilities = torch.sigmoid(output["peptide_logits"]).cpu()
            propensity_probabilities = (
                torch.sigmoid(output["propensity_peptide_logits"]).cpu()
                if output["propensity_peptide_logits"] is not None
                else None
            )
            residue_propensity_probabilities = (
                torch.sigmoid(output["aggregation_propensity_logits"]).cpu()
                if output["aggregation_propensity_logits"] is not None
                else None
            )
            peptide_labels.extend(batch["labels"].tolist())
            peptide_probs.extend(probabilities.tolist())
            if propensity_probabilities is not None:
                propensity_probs.extend(propensity_probabilities.tolist())
            if prediction_rows is not None:
                batch_propensity_probs = (
                    propensity_probabilities.tolist()
                    if propensity_probabilities is not None
                    else [None] * len(batch["ids"])
                )
                batch_propensity_profiles = (
                    residue_propensity_probabilities.tolist()
                    if residue_propensity_probabilities is not None
                    else [None] * len(batch["ids"])
                )
                for (
                    identifier,
                    sequence,
                    label,
                    probability,
                    propensity_probability,
                    propensity_profile,
                ) in zip(
                    batch["ids"],
                    batch["sequences"],
                    batch["labels"].tolist(),
                    probabilities.tolist(),
                    batch_propensity_probs,
                    batch_propensity_profiles,
                ):
                    row = {
                        "fold": fold,
                        "item_id": identifier,
                        "sequence": sequence,
                        "label": int(label),
                        "probability": float(probability),
                        "prediction_at_0_5": int(probability >= 0.5),
                    }
                    if propensity_probability is not None:
                        row["propensity_probability"] = float(
                            propensity_probability
                        )
                        row["propensity_prediction_at_0_5"] = int(
                            propensity_probability >= 0.5
                        )
                        row["local_propensity_profile"] = ";".join(
                            f"{value:.6f}"
                            for value in propensity_profile[: len(sequence)]
                        )
                    prediction_rows["peptide"].append(row)
        metrics.update(
            {f"peptide_{key}": value for key, value in binary_metrics(peptide_labels, peptide_probs).items()}
        )
        if propensity_probs:
            metrics.update(
                {
                    f"peptide_propensity_{key}": value
                    for key, value in binary_metrics(
                        peptide_labels, propensity_probs
                    ).items()
                }
            )

    if "apr" in loaders:
        apr_labels: list[float] = []
        apr_probs: list[float] = []
        region_stat_totals: dict[str, float] = {}
        for batch in loaders["apr"]:
            output = model.forward_sequence(batch["sequences"])
            targets = batch["targets"].to(device, non_blocking=True)
            valid = targets != -100
            probabilities = torch.sigmoid(output["apr_logits"])
            apr_labels.extend(targets[valid].cpu().tolist())
            apr_probs.extend(probabilities[valid].cpu().tolist())
            if args.apr_region_head:
                for row_index, sequence in enumerate(batch["sequences"]):
                    sequence_targets = targets[
                        row_index, : len(sequence)
                    ].cpu().numpy()
                    sequence_probabilities = probabilities[
                        row_index, : len(sequence)
                    ].cpu().numpy()
                    statistics = apr_region_profile_statistics(
                        sequence_targets, sequence_probabilities
                    )
                    for name, value in statistics.items():
                        region_stat_totals[name] = (
                            region_stat_totals.get(name, 0.0) + value
                        )
            if prediction_rows is not None:
                for row_index, (identifier, sequence) in enumerate(
                    zip(batch["ids"], batch["sequences"])
                ):
                    sequence_targets = targets[row_index, : len(sequence)].cpu().numpy()
                    sequence_probabilities = (
                        probabilities[row_index, : len(sequence)].cpu().numpy()
                    )
                    if args.apr_region_head:
                        if "apr_region_profile" in prediction_rows:
                            labels = "".join(
                                "1" if value == 1 else "0" if value == 0 else "?"
                                for value in sequence_targets
                            )
                            prediction_rows["apr_region_profile"].append(
                                {
                                    "fold": fold,
                                    "item_id": identifier,
                                    "sequence": sequence,
                                    "label_mask": labels,
                                    "probability_profile": ";".join(
                                        f"{value:.6f}"
                                        for value in sequence_probabilities
                                    ),
                                }
                            )
                    positions = torch.nonzero(valid[row_index], as_tuple=False).flatten()
                    for position in positions.tolist():
                        probability = float(probabilities[row_index, position].cpu())
                        label = int(targets[row_index, position].cpu())
                        prediction_rows["apr"].append(
                            {
                                "fold": fold,
                                "item_id": identifier,
                                "residue_index": position + 1,
                                "amino_acid": sequence[position],
                                "label": label,
                                "probability": probability,
                                "prediction_at_0_5": int(probability >= 0.5),
                            }
                        )
        metrics.update(
            {
                f"apr_{key}": value
                for key, value in binary_metrics(apr_labels, apr_probs).items()
            }
        )
        if args.apr_region_head:
            true_region_count = region_stat_totals.get("true_region_count", 0.0)
            matched_count = region_stat_totals.get("boundary_match_count", 0.0)
            negative_proteins = region_stat_totals.get("has_trusted_negative", 0.0)
            metrics.update(
                {
                    "apr_region_recall_at_0_5_iou": (
                        region_stat_totals.get("recalled_region_count", 0.0)
                        / max(true_region_count, 1.0)
                    ),
                    "apr_region_mean_best_iou": (
                        region_stat_totals.get("best_iou_sum", 0.0)
                        / max(true_region_count, 1.0)
                    ),
                    "apr_region_mean_boundary_error": (
                        region_stat_totals.get("boundary_error_sum", 0.0)
                        / max(matched_count, 1.0)
                    ),
                    "apr_region_false_regions_per_trusted_negative_protein": (
                        region_stat_totals.get(
                            "trusted_negative_false_region_count", 0.0
                        )
                        / max(negative_proteins, 1.0)
                    ),
                    "apr_region_true_region_n": true_region_count,
                }
            )

    if "structure" in loaders:
        structure_labels: list[float] = []
        structure_probs: list[float] = []
        for batch in loaders["structure"]:
            logits = model.forward_structure(
                batch["node_features"].to(device, non_blocking=True),
                batch["adjacency"].to(device, non_blocking=True),
                batch["node_mask"].to(device, non_blocking=True),
                chain_sequences=batch["chain_sequences"],
                apr_gradient_beta=0.0,
                peptide_evidence=(
                    batch["peptide_evidence"].to(device, non_blocking=True)
                    if batch["peptide_evidence"] is not None
                    else None
                ),
            )
            structure_labels.extend(batch["labels"].tolist())
            probabilities = torch.sigmoid(logits).cpu()
            structure_probs.extend(probabilities.tolist())
            if prediction_rows is not None:
                for identifier, label, probability in zip(
                    batch["ids"], batch["labels"].tolist(), probabilities.tolist()
                ):
                    prediction_rows["structure"].append(
                        {
                            "fold": fold,
                            "item_id": identifier,
                            "label": int(label),
                            "probability": float(probability),
                            "prediction_at_0_5": int(probability >= 0.5),
                        }
                    )
        metrics.update(
            {f"structure_{key}": value for key, value in binary_metrics(structure_labels, structure_probs).items()}
        )
    model.train()
    return metrics


@torch.no_grad()
def evaluate_validation_losses(
    model: AmyloidAPRModel,
    loaders: dict[str, DataLoader],
    device: torch.device,
    structure_pos_weight: float | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    loss_sums: dict[str, float] = {}
    weight_sums: dict[str, float] = {}
    apr_validation_labels: list[float] = []
    apr_validation_probabilities: list[float] = []
    peptide_validation_labels: list[float] = []
    peptide_validation_probabilities: list[float] = []
    peptide_main_validation_probabilities: list[float] = []
    peptide_propensity_validation_probabilities: list[float] = []
    structure_validation_labels: list[float] = []
    structure_validation_probabilities: list[float] = []

    if "peptide" in loaders:
        for batch in loaders["peptide"]:
            output = model.forward_sequence(batch["sequences"])
            logits = output["peptide_logits"]
            labels = batch["labels"].to(device, non_blocking=True)
            peptide_validation_labels.extend(labels.cpu().tolist())
            peptide_validation_probabilities.extend(
                torch.sigmoid(logits).cpu().tolist()
            )
            peptide_main_validation_probabilities.extend(
                torch.sigmoid(output["peptide_main_logits"]).cpu().tolist()
            )
            peptide_losses = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )
            loss_sums["peptide"] = loss_sums.get("peptide", 0.0) + float(
                peptide_losses.sum().cpu()
            )
            weight_sums["peptide"] = weight_sums.get("peptide", 0.0) + labels.numel()
            propensity_logits = output["propensity_peptide_logits"]
            if propensity_logits is not None:
                peptide_propensity_validation_probabilities.extend(
                    torch.sigmoid(propensity_logits).cpu().tolist()
                )
                propensity_losses = nn.functional.binary_cross_entropy_with_logits(
                    propensity_logits, labels, reduction="none"
                )
                loss_sums["peptide_propensity"] = loss_sums.get(
                    "peptide_propensity", 0.0
                ) + float(propensity_losses.sum().cpu())
                weight_sums["peptide_propensity"] = weight_sums.get(
                    "peptide_propensity", 0.0
                ) + labels.numel()
                consistency_losses = nn.functional.binary_cross_entropy_with_logits(
                    propensity_logits,
                    torch.sigmoid(output["peptide_main_logits"].detach()),
                    reduction="none",
                )
                loss_sums["peptide_propensity_consistency"] = loss_sums.get(
                    "peptide_propensity_consistency", 0.0
                ) + float(consistency_losses.sum().cpu())
                weight_sums["peptide_propensity_consistency"] = weight_sums.get(
                    "peptide_propensity_consistency", 0.0
                ) + labels.numel()
                smoothness = local_propensity_smoothness(
                    output["aggregation_propensity_logits"],
                    output["residue_mask"],
                )
                loss_sums["peptide_propensity_smoothness"] = loss_sums.get(
                    "peptide_propensity_smoothness", 0.0
                ) + float(smoothness.cpu())
                weight_sums["peptide_propensity_smoothness"] = weight_sums.get(
                    "peptide_propensity_smoothness", 0.0
                ) + 1.0

    if "apr" in loaders:
        for batch in loaders["apr"]:
            output = model.forward_sequence(batch["sequences"])
            logits = output["apr_logits"]
            targets = batch["targets"].to(device, non_blocking=True)
            label_weights = batch["label_weights"].to(device, non_blocking=True)
            valid = targets != -100
            apr_validation_labels.extend(targets[valid].cpu().tolist())
            apr_validation_probabilities.extend(torch.sigmoid(logits)[valid].cpu().tolist())
            numerator, denominator = masked_apr_bce_components(
                logits,
                targets,
                label_weights,
                positive_weight=args.apr_positive_weight,
                sequence_balanced=args.apr_sequence_balanced_loss,
                focal_gamma=args.apr_focal_gamma,
                hard_negative_gamma=args.apr_hard_negative_gamma,
                mixed_sequence_weight=args.apr_mixed_sequence_weight,
                mixed_negative_weight=args.apr_mixed_negative_weight,
                source_mixed=batch["source_mixed"].to(device, non_blocking=True),
            )
            loss_sums["apr"] = loss_sums.get("apr", 0.0) + float(numerator.cpu())
            weight_sums["apr"] = weight_sums.get("apr", 0.0) + float(
                denominator.cpu()
            )
            if args.apr_region_head:
                region_components = apr_region_losses(
                    output["apr_region_logits"],
                    output["apr_region_valid"],
                    output["apr_region_lengths"],
                    logits,
                    targets,
                    label_weights,
                    training_only=batch["training_only"].to(
                        device, non_blocking=True
                    ),
                    positive_fraction=args.apr_region_positive_fraction,
                    max_candidates_per_class=(
                        args.apr_region_max_candidates_per_class
                    ),
                    rank_margin=args.apr_region_rank_margin,
                    rank_temperature=args.apr_region_rank_temperature,
                )
                for name, value in region_components.items():
                    key = f"apr_region_{name}" if name != "region" else "apr_region"
                    loss_sums[key] = loss_sums.get(key, 0.0) + float(value.cpu())
                    weight_sums[key] = weight_sums.get(key, 0.0) + 1.0

    if "structure" in loaders:
        if structure_pos_weight is None:
            raise RuntimeError("Structure validation requires a positive-class weight")
        pos_weight = torch.tensor(
            structure_pos_weight, dtype=torch.float32, device=device
        )
        for batch in loaders["structure"]:
            logits = model.forward_structure(
                batch["node_features"].to(device, non_blocking=True),
                batch["adjacency"].to(device, non_blocking=True),
                batch["node_mask"].to(device, non_blocking=True),
                chain_sequences=batch["chain_sequences"],
                apr_gradient_beta=0.0,
                peptide_evidence=(
                    batch["peptide_evidence"].to(device, non_blocking=True)
                    if batch["peptide_evidence"] is not None
                    else None
                ),
            )
            labels = batch["labels"].to(device, non_blocking=True)
            structure_validation_labels.extend(labels.cpu().tolist())
            structure_validation_probabilities.extend(
                torch.sigmoid(logits).cpu().tolist()
            )
            loss_sums["structure"] = loss_sums.get("structure", 0.0) + float(
                nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=pos_weight,
                    reduction="sum",
                ).cpu()
            )
            weight_sums["structure"] = (
                weight_sums.get("structure", 0.0) + labels.numel()
            )

    losses = {
        f"{name}_loss": loss_sums[name] / max(weight_sums[name], 1e-12)
        for name in loss_sums
    }
    loss_weights = {
        "peptide": args.peptide_loss_weight,
        "peptide_propensity": args.peptide_propensity_loss_weight,
        "peptide_propensity_consistency": (
            args.peptide_propensity_consistency_weight
        ),
        "peptide_propensity_smoothness": (
            args.peptide_propensity_smoothness_weight
        ),
        "apr": args.apr_loss_weight,
        "apr_region": args.apr_region_loss_weight,
        "apr_region_consistency": args.apr_region_consistency_weight,
        "apr_region_rank": args.apr_region_rank_weight,
        "structure": args.structure_loss_weight,
    }
    losses["total_loss"] = sum(
        float(loss_weights[name]) * losses[f"{name}_loss"] for name in loss_sums
    )
    if peptide_validation_labels:
        peptide_metrics = binary_metrics(
            peptide_validation_labels, peptide_validation_probabilities
        )
        losses["peptide_mcc_at_0_5"] = peptide_metrics["mcc"]
        main_metrics = binary_metrics(
            peptide_validation_labels, peptide_main_validation_probabilities
        )
        losses["peptide_main_auprc"] = main_metrics["auprc"]
        if peptide_propensity_validation_probabilities:
            propensity_metrics = binary_metrics(
                peptide_validation_labels,
                peptide_propensity_validation_probabilities,
            )
            losses["peptide_propensity_auprc"] = propensity_metrics["auprc"]
            components = (
                max(float(peptide_metrics["mcc"]), 1e-12),
                max(float(main_metrics["auprc"]), 1e-12),
                max(float(propensity_metrics["auprc"]), 1e-12),
            )
            losses["peptide_transfer_score"] = len(components) / sum(
                1.0 / component for component in components
            )
    if apr_validation_labels:
        losses["apr_mcc_at_0_5"] = binary_metrics(
            apr_validation_labels, apr_validation_probabilities
        )["mcc"]
    if structure_validation_labels:
        losses["structure_mcc_at_0_5"] = binary_metrics(
            structure_validation_labels, structure_validation_probabilities
        )["mcc"]
    model.train()
    return losses


def train_one_fold(
    model: AmyloidAPRModel,
    loaders: dict[str, DataLoader],
    args: argparse.Namespace,
    device: torch.device,
    structure_pos_weight: float | None,
    epochs: int,
    validation_loaders: dict[str, DataLoader] | None = None,
    validation_structure_pos_weight: float | None = None,
    phase: str = "train",
) -> tuple[dict[str, float], int, float | None]:
    enabled_modules = []
    if args.task in {"all", "sequence", "peptide"}:
        enabled_modules.append(model.sequence)
    elif args.task == "apr":
        if model.sequence.apr_region_head is not None:
            # The transferred Peptide ESM/LoRA and local propensity path are
            # evidence providers, not APR optimisation targets.
            for parameter in model.sequence.encoder.parameters():
                parameter.requires_grad_(False)
            enabled_modules.extend(
                [model.sequence.apr_head, model.sequence.apr_region_head]
            )
        else:
            enabled_modules.extend([model.sequence.encoder, model.sequence.apr_head])
    if args.task in {"all", "structure"}:
        enabled_modules.append(model.structure)
    parameters = trainable_parameters(enabled_modules)
    trainable_parameter_ids = {id(parameter) for parameter in parameters}
    ema_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if id(parameter) in trainable_parameter_ids
    }
    ema_shadow: dict[str, torch.Tensor] = {}
    ema_updates = 0
    if args.sequence_backbone_learning_rate is not None:
        backbone_module = model.sequence.encoder.backbone.backbone
        backbone_ids = {id(parameter) for parameter in backbone_module.parameters()}
        backbone_parameters = [
            parameter for parameter in parameters if id(parameter) in backbone_ids
        ]
        head_parameters = [
            parameter for parameter in parameters if id(parameter) not in backbone_ids
        ]
        parameter_groups = []
        if head_parameters:
            parameter_groups.append(
                {"params": head_parameters, "lr": args.learning_rate}
            )
        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": args.sequence_backbone_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    structure_pos_weight_tensor = (
        torch.tensor(
            structure_pos_weight * args.structure_positive_weight_scale,
            dtype=torch.float32,
            device=device,
        )
        if structure_pos_weight is not None
        else None
    )
    structure_positive_memory = torch.empty(0, device=device)
    structure_negative_memory = torch.empty(0, device=device)
    if args.epoch_anchor == "apr":
        if "apr" not in loaders:
            raise ValueError("--epoch-anchor apr requires the APR dataset")
        steps = len(loaders["apr"])
    else:
        core_loaders = {
            name: loader
            for name, loader in loaders.items()
            if name in {"peptide", "apr", "structure"}
        }
        if not core_loaders:
            raise ValueError("At least one primary task loader is required")
        # Auxiliary pair/contrastive loaders must not silently lengthen an epoch
        # and increase exposure to the primary supervised datasets.
        steps = max(len(loader) for loader in core_loaders.values())
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.learning_rate_schedule == "warmup_cosine":
        # Epoch selection and full-fold retraining must replay the same LR prefix.
        # Keep the schedule horizon at the configured maximum, even when the
        # selected retraining length is shorter.
        schedule_steps = max(1, args.epochs * steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: warmup_cosine_factor(
                step,
                schedule_steps,
                args.learning_rate_warmup_fraction,
                args.learning_rate_min_factor,
            ),
        )
    iterators = {name: infinite(loader) for name, loader in loaders.items()}
    peptide_rng_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    model.train()
    last_statistics: dict[str, float] = {}
    best_epoch = epochs
    best_monitor: float | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if args.task == "apr" and model.sequence.apr_region_head is not None:
            model.sequence.set_peptide_transfer_eval()
        active_peptide_apr_scale = peptide_apr_fusion_scale(args, epoch)
        model.sequence.set_peptide_apr_fusion_scale(active_peptide_apr_scale)
        totals: dict[str, float] = {"loss": 0.0}
        for step_index in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses: dict[str, torch.Tensor] = {}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                if "peptide" in loaders:
                    peptide_batch = next(iterators["peptide"])
                    with torch.random.fork_rng(devices=peptide_rng_devices):
                        if args.peptide_stepwise_dropout:
                            dropout_seed = (
                                args.seed
                                + epoch * 1_000_003
                                + step_index * 10_007
                            )
                            torch.manual_seed(dropout_seed)
                            if device.type == "cuda":
                                torch.cuda.manual_seed(dropout_seed)
                        peptide_outputs = [
                            model.forward_sequence(peptide_batch["sequences"])
                        ]
                        if args.peptide_rdrop_weight > 0:
                            peptide_outputs.append(
                                model.forward_sequence(peptide_batch["sequences"])
                            )
                    peptide_output = peptide_outputs[0]
                    peptide_labels = peptide_batch["labels"].to(device)
                    losses["peptide"] = torch.stack(
                        [
                            peptide_classification_loss(
                                output["peptide_logits"],
                                peptide_labels,
                                label_smoothing=args.peptide_label_smoothing,
                                focal_gamma=args.peptide_focal_gamma,
                                soft_mcc_weight=args.peptide_soft_mcc_weight,
                                pairwise_rank_weight=args.peptide_pairwise_rank_weight,
                                pairwise_rank_temperature=(
                                    args.peptide_pairwise_rank_temperature
                                ),
                            )
                            for output in peptide_outputs
                        ]
                    ).mean()
                    if len(peptide_outputs) == 2:
                        losses["peptide_rdrop"] = symmetric_bernoulli_kl(
                            peptide_outputs[0]["peptide_logits"],
                            peptide_outputs[1]["peptide_logits"],
                        )
                    propensity_logits = [
                        output["propensity_peptide_logits"]
                        for output in peptide_outputs
                    ]
                    if propensity_logits[0] is not None:
                        propensity_targets = (
                            peptide_labels * (1.0 - args.peptide_label_smoothing)
                            + 0.5 * args.peptide_label_smoothing
                        )
                        losses["peptide_propensity"] = torch.stack(
                            [
                                nn.functional.binary_cross_entropy_with_logits(
                                    logits, propensity_targets
                                )
                                for logits in propensity_logits
                            ]
                        ).mean()
                        losses["peptide_propensity_consistency"] = torch.stack(
                            [
                                nn.functional.binary_cross_entropy_with_logits(
                                    propensity_logit,
                                    torch.sigmoid(
                                        output["peptide_main_logits"].detach()
                                    ),
                                )
                                for output, propensity_logit in zip(
                                    peptide_outputs, propensity_logits
                                )
                            ]
                        ).mean()
                        losses["peptide_propensity_smoothness"] = torch.stack(
                            [
                                local_propensity_smoothness(
                                    output["aggregation_propensity_logits"],
                                    output["residue_mask"],
                                )
                                for output in peptide_outputs
                            ]
                        ).mean()

                if "peptide_mutation_pairs" in loaders:
                    mutation_batch = next(iterators["peptide_mutation_pairs"])
                    sequences_a = mutation_batch["sequences_a"]
                    sequences_b = mutation_batch["sequences_b"]
                    mutation_output = model.forward_sequence(
                        sequences_a + sequences_b,
                        isolate_peptide_rng=True,
                    )
                    pair_count = len(sequences_a)
                    logits_a = mutation_output["peptide_logits"][:pair_count]
                    logits_b = mutation_output["peptide_logits"][pair_count:]
                    embeddings_a = mutation_output["peptide_embedding"][:pair_count]
                    embeddings_b = mutation_output["peptide_embedding"][pair_count:]
                    rank_loss, consistency_loss = peptide_mutation_pair_losses(
                        logits_a,
                        logits_b,
                        embeddings_a,
                        embeddings_b,
                        mutation_batch["labels_a"].to(device),
                        mutation_batch["labels_b"].to(device),
                        margin=args.peptide_mutation_pair_margin,
                        temperature=args.peptide_mutation_pair_temperature,
                    )
                    losses["peptide_mutation_rank"] = rank_loss
                    losses["peptide_mutation_consistency"] = consistency_loss

                if "apr_positive_contrastive" in loaders:
                    if "peptide" not in loaders:
                        raise RuntimeError(
                            "APR-positive contrastive training requires Peptide batches"
                        )
                    apr_positive_batch = next(
                        iterators["apr_positive_contrastive"]
                    )
                    apr_positive_output = model.forward_sequence(
                        apr_positive_batch["sequences"],
                        isolate_peptide_rng=True,
                    )
                    losses["apr_positive_contrastive"] = (
                        apr_positive_representation_loss(
                            apr_positive_output["peptide_embedding"],
                            peptide_output["peptide_embedding"],
                            peptide_labels,
                            temperature=(
                                args.apr_positive_contrastive_temperature
                            ),
                        )
                    )

                if "apr" in loaders:
                    apr_batch = next(iterators["apr"])
                    apr_output = model.forward_sequence(
                        apr_batch["sequences"], isolate_peptide_rng=True
                    )
                    apr_targets = apr_batch["targets"].to(device)
                    apr_label_weights = apr_batch["label_weights"].to(device)
                    losses["apr"] = masked_apr_bce(
                        apr_output["apr_logits"],
                        apr_targets,
                        apr_label_weights,
                        positive_weight=args.apr_positive_weight,
                        sequence_balanced=args.apr_sequence_balanced_loss,
                        focal_gamma=args.apr_focal_gamma,
                        hard_negative_gamma=args.apr_hard_negative_gamma,
                        mixed_sequence_weight=args.apr_mixed_sequence_weight,
                        mixed_negative_weight=args.apr_mixed_negative_weight,
                        source_mixed=apr_batch["source_mixed"].to(device),
                    )
                    if args.apr_region_head:
                        region_components = apr_region_losses(
                            apr_output["apr_region_logits"],
                            apr_output["apr_region_valid"],
                            apr_output["apr_region_lengths"],
                            apr_output["apr_logits"],
                            apr_targets,
                            apr_label_weights,
                            training_only=apr_batch["training_only"].to(device),
                            positive_fraction=args.apr_region_positive_fraction,
                            max_candidates_per_class=(
                                args.apr_region_max_candidates_per_class
                            ),
                            rank_margin=args.apr_region_rank_margin,
                            rank_temperature=args.apr_region_rank_temperature,
                        )
                        losses["apr_region"] = region_components["region"]
                        losses["apr_region_consistency"] = region_components[
                            "consistency"
                        ]
                        losses["apr_region_rank"] = region_components["rank"]
                    if args.apr_soft_mcc_weight > 0:
                        losses["apr"] = losses["apr"] + (
                            args.apr_soft_mcc_weight
                            * masked_apr_soft_mcc(
                                apr_output["apr_logits"],
                                apr_targets,
                                apr_label_weights,
                            )
                        )
                    if args.apr_smoothness_weight > 0:
                        losses["apr"] = losses["apr"] + (
                            args.apr_smoothness_weight
                            * masked_apr_smoothness(
                                apr_output["apr_logits"],
                                apr_targets,
                                apr_label_weights,
                            )
                        )
                    if args.apr_pairwise_rank_weight > 0:
                        losses["apr"] = losses["apr"] + (
                            args.apr_pairwise_rank_weight
                            * masked_apr_pairwise_rank(
                                apr_output["apr_logits"],
                                apr_targets,
                                apr_label_weights,
                                temperature=args.apr_pairwise_rank_temperature,
                            )
                        )
                    apr_local_delta = apr_output["apr_local_residual_delta"]
                    if (
                        apr_local_delta is not None
                        and args.apr_local_residual_penalty > 0
                    ):
                        valid_apr = apr_targets != -100
                        losses["apr"] = losses["apr"] + (
                            args.apr_local_residual_penalty
                            * apr_local_delta[valid_apr].pow(2).mean()
                        )

                if "structure" in loaders:
                    if structure_pos_weight_tensor is None:
                        raise RuntimeError("Structure training requires a positive-class weight")
                    structure_batch = next(iterators["structure"])
                    structure_logits = model.forward_structure(
                        structure_batch["node_features"].to(device),
                        structure_batch["adjacency"].to(device),
                        structure_batch["node_mask"].to(device),
                        chain_sequences=structure_batch["chain_sequences"],
                        apr_gradient_beta=structure_apr_beta(args, epoch),
                        peptide_evidence=(
                            structure_batch["peptide_evidence"].to(device)
                            if structure_batch["peptide_evidence"] is not None
                            else None
                        ),
                    )
                    structure_labels = structure_batch["labels"].to(device)
                    losses["structure"] = structure_classification_loss(
                        structure_logits,
                        structure_labels,
                        structure_pos_weight_tensor,
                        focal_gamma=args.structure_focal_gamma,
                        label_smoothing=args.structure_label_smoothing,
                        pairwise_rank_weight=args.structure_pairwise_rank_weight,
                        pairwise_rank_temperature=args.structure_pairwise_rank_temperature,
                        positive_logit_memory=structure_positive_memory,
                        negative_logit_memory=structure_negative_memory,
                        rank_hard_k=args.structure_rank_hard_k,
                    )
                    if args.structure_rank_memory_size > 0:
                        current_positive = structure_logits[structure_labels >= 0.5].detach()
                        current_negative = structure_logits[structure_labels < 0.5].detach()
                        if current_positive.numel():
                            structure_positive_memory = torch.cat(
                                [structure_positive_memory, current_positive]
                            )[-args.structure_rank_memory_size :]
                        if current_negative.numel():
                            structure_negative_memory = torch.cat(
                                [structure_negative_memory, current_negative]
                            )[-args.structure_rank_memory_size :]

                loss_weights = {
                    "peptide": args.peptide_loss_weight,
                    "peptide_propensity": args.peptide_propensity_loss_weight,
                    "peptide_propensity_consistency": (
                        args.peptide_propensity_consistency_weight
                    ),
                    "peptide_propensity_smoothness": (
                        args.peptide_propensity_smoothness_weight
                    ),
                    "peptide_rdrop": args.peptide_rdrop_weight,
                    "peptide_mutation_rank": (
                        args.peptide_mutation_pair_rank_weight
                    ),
                    "peptide_mutation_consistency": (
                        args.peptide_mutation_pair_consistency_weight
                    ),
                    "apr_positive_contrastive": (
                        args.apr_positive_contrastive_weight
                    ),
                    "apr": args.apr_loss_weight,
                    "apr_region": args.apr_region_loss_weight,
                    "apr_region_consistency": args.apr_region_consistency_weight,
                    "apr_region_rank": args.apr_region_rank_weight,
                    "structure": args.structure_loss_weight,
                }
                weighted_losses = []
                for name, loss in losses.items():
                    weight = float(loss_weights[name])
                    weighted_losses.append(weight * loss)
                total_loss = sum(weighted_losses[1:], weighted_losses[0])

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            gradient_norms = clip_model_gradients(
                model,
                mode=args.gradient_clipping,
                max_norm=args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if (
                args.parameter_ema_decay > 0
                and epoch >= args.parameter_ema_start_epoch
            ):
                update_parameter_ema(
                    ema_shadow, ema_parameters, args.parameter_ema_decay
                )
                ema_updates += 1
            totals["loss"] += float(total_loss.detach().cpu())
            for name, loss in losses.items():
                totals[f"{name}_loss"] = totals.get(f"{name}_loss", 0.0) + float(loss.detach().cpu())
            for name, value in gradient_norms.items():
                totals[name] = totals.get(name, 0.0) + value
        last_statistics = {name: value / steps for name, value in totals.items()}
        last_statistics["head_learning_rate"] = float(
            optimizer.param_groups[0]["lr"]
        )
        if len(optimizer.param_groups) > 1:
            last_statistics["backbone_learning_rate"] = float(
                optimizer.param_groups[1]["lr"]
            )
        if model.sequence.peptide_apr_beta_logit is not None:
            last_statistics["peptide_apr_fusion_scale"] = active_peptide_apr_scale
            last_statistics["peptide_apr_beta"] = float(
                (
                    active_peptide_apr_scale
                    *
                    model.sequence.peptide_apr_beta_max
                    * torch.sigmoid(model.sequence.peptide_apr_beta_logit.detach())
                ).cpu()
            )
        peptide_feature_gate = model.sequence.peptide_residue_feature_gate_logit
        if peptide_feature_gate is not None:
            last_statistics["peptide_residue_feature_gate"] = float(
                torch.sigmoid(peptide_feature_gate.detach()).cpu()
            )
        if args.structure_apr_probabilities:
            last_statistics["structure_apr_beta"] = structure_apr_beta(args, epoch)
        peptide_attention_alpha = model.structure.encoder.peptide_attention_alpha()
        if peptide_attention_alpha is not None:
            last_statistics["structure_peptide_attention_alpha"] = float(
                peptide_attention_alpha.detach().cpu()
            )
        validation_losses: dict[str, float] | None = None
        if validation_loaders is not None:
            ema_backup: dict[str, torch.Tensor] = {}
            if ema_shadow:
                with torch.no_grad():
                    for name, parameter in ema_parameters.items():
                        ema_backup[name] = parameter.detach().clone()
                        parameter.copy_(ema_shadow[name])
            try:
                validation_losses = evaluate_validation_losses(
                    model,
                    validation_loaders,
                    device,
                    structure_pos_weight=validation_structure_pos_weight,
                    args=args,
                )
            finally:
                if ema_backup:
                    with torch.no_grad():
                        for name, parameter in ema_parameters.items():
                            parameter.copy_(ema_backup[name])
            monitor_name = args.early_stopping_monitor
            if monitor_name not in validation_losses:
                raise RuntimeError(
                    f"{monitor_name} is required for early stopping"
                )
            monitor = validation_losses[monitor_name]
            maximize_monitor = monitor_name in {
                "peptide_mcc_at_0_5",
                "peptide_transfer_score",
                "apr_mcc_at_0_5",
                "structure_mcc_at_0_5",
            }
            if epoch >= args.early_stopping_selection_start_epoch:
                improved = best_monitor is None or (
                    monitor > best_monitor + args.early_stopping_min_delta
                    if maximize_monitor
                    else monitor < best_monitor - args.early_stopping_min_delta
                )
                if improved:
                    best_monitor = monitor
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
        if epoch % args.log_every_epochs == 0 or epoch == epochs:
            details = " ".join(f"{name}={value:.5f}" for name, value in last_statistics.items())
            validation_text = ""
            if validation_losses is not None:
                validation_text = " " + " ".join(
                    f"val_{name}={value:.5f}"
                    for name, value in validation_losses.items()
                )
            print(
                f"phase={phase} epoch={epoch:03d}/{epochs:03d} "
                f"{details}{validation_text}",
                flush=True,
            )
        if (
            validation_loaders is not None
            and epoch >= args.early_stopping_min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping selection at epoch {epoch}; "
                f"best_epoch={best_epoch} "
                f"best_{args.early_stopping_monitor}={best_monitor:.6f}",
                flush=True,
            )
            break
    if ema_shadow:
        with torch.no_grad():
            for name, parameter in ema_parameters.items():
                parameter.copy_(ema_shadow[name])
        last_statistics["parameter_ema_updates"] = float(ema_updates)
    return last_statistics, best_epoch, best_monitor


def build_model(args: argparse.Namespace, device: torch.device) -> AmyloidAPRModel:
    model = AmyloidAPRModel(
        sequence_backbone=args.sequence_backbone,
        sequence_hidden_size=args.sequence_hidden_size,
        sequence_projection_mode=args.sequence_projection_mode,
        freeze_sequence_backbone=args.freeze_sequence_backbone,
        structure_hidden_size=args.structure_hidden_size,
        structure_layers=args.structure_layers,
        structure_geometric_features=args.structure_geometric_features,
        structure_message_dropout=args.structure_message_dropout,
        structure_head_dropout=args.structure_head_dropout,
        structure_residual=args.structure_residual,
        structure_layer_mix=args.structure_layer_mix,
        structure_input_skip=args.structure_input_skip,
        structure_pooling=args.structure_pooling,
        sequence_model_path=args.sequence_model_path,
        embedding_cache_dir=args.embedding_cache_dir,
        lora_rank=args.sequence_lora_rank,
        lora_alpha=args.sequence_lora_alpha,
        lora_dropout=args.sequence_lora_dropout,
        lora_last_n_layers=args.sequence_lora_last_n_layers,
        lora_target_projections=tuple(
            item.strip()
            for item in args.sequence_lora_target_projections.split(",")
            if item.strip()
        ),
        residue_feature_table=args.residue_feature_table,
        residue_feature_sets=tuple(
            item.strip()
            for item in args.residue_feature_sets.split(",")
            if item.strip()
        ),
        residue_feature_gate_init=args.residue_feature_gate_init,
        residue_feature_dropout=args.residue_feature_dropout,
        peptide_residue_features=getattr(args, "peptide_residue_features", False),
        peptide_backbone_gradients=getattr(
            args, "peptide_backbone_gradients", False
        ),
        peptide_short_cnn=getattr(args, "peptide_short_cnn", False),
        peptide_short_cnn_hidden_size=getattr(
            args, "peptide_short_cnn_hidden_size", 96
        ),
        peptide_short_cnn_kernels=tuple(
            int(item.strip())
            for item in getattr(args, "peptide_short_cnn_kernels", "3,5,7").split(",")
            if item.strip()
        ),
        peptide_short_cnn_dropout=getattr(args, "peptide_short_cnn_dropout", 0.1),
        peptide_short_cnn_gate_init=getattr(args, "peptide_short_cnn_gate_init", -1.5),
        peptide_length_conditioned_gate=getattr(
            args, "peptide_length_conditioned_gate", False
        ),
        peptide_length_gate_hidden_size=getattr(
            args, "peptide_length_gate_hidden_size", 16
        ),
        peptide_output_fusion=getattr(args, "peptide_output_fusion", "main"),
        peptide_output_main_weight=getattr(
            args, "peptide_output_main_weight", 0.5
        ),
        peptide_output_detach_propensity=getattr(
            args, "peptide_output_detach_propensity", False
        ),
        peptide_output_fusion_hidden_size=getattr(
            args, "peptide_output_fusion_hidden_size", 8
        ),
        peptide_attention_residual=getattr(
            args, "peptide_attention_residual", False
        ),
        peptide_attention_residual_hidden_size=getattr(
            args, "peptide_attention_residual_hidden_size", 32
        ),
        peptide_attention_residual_heads=getattr(
            args, "peptide_attention_residual_heads", 2
        ),
        peptide_attention_residual_dropout=getattr(
            args, "peptide_attention_residual_dropout", 0.2
        ),
        peptide_attention_residual_max_delta=getattr(
            args, "peptide_attention_residual_max_delta", 0.5
        ),
        peptide_global_physchem=getattr(args, "peptide_global_physchem", False),
        peptide_global_physchem_hidden_size=getattr(
            args, "peptide_global_physchem_hidden_size", 128
        ),
        peptide_global_physchem_dropout=getattr(
            args, "peptide_global_physchem_dropout", 0.2
        ),
        peptide_global_physchem_max_delta=getattr(
            args, "peptide_global_physchem_max_delta", 1.0
        ),
        peptide_length6_residual=getattr(args, "peptide_length6_residual", False),
        peptide_length6_hidden_size=getattr(
            args, "peptide_length6_hidden_size", 64
        ),
        peptide_length6_dropout=getattr(args, "peptide_length6_dropout", 0.2),
        peptide_length6_max_delta=getattr(args, "peptide_length6_max_delta", 1.0),
        peptide_local_propensity=getattr(
            args, "peptide_local_propensity", False
        ),
        peptide_local_propensity_hidden_size=getattr(
            args, "peptide_local_propensity_hidden_size", 64
        ),
        peptide_local_propensity_kernels=tuple(
            int(item.strip())
            for item in getattr(
                args, "peptide_local_propensity_kernels", "3,5,7"
            ).split(",")
            if item.strip()
        ),
        peptide_local_propensity_dropout=getattr(
            args, "peptide_local_propensity_dropout", 0.1
        ),
        sequence_channel_dropout=args.sequence_channel_dropout,
        peptide_apr_fusion=args.peptide_apr_fusion,
        peptide_apr_beta_init=args.peptide_apr_beta_init,
        peptide_apr_beta_max=args.peptide_apr_beta_max,
        peptide_apr_window_size=args.peptide_apr_window_size,
        peptide_apr_feature_residual_hidden_size=(
            args.peptide_apr_feature_residual_hidden_size
        ),
        apr_context_kernel=args.apr_context_kernel,
        apr_head_mode=args.apr_head_mode,
        apr_local_residual=args.apr_local_residual,
        apr_local_residual_kernels=tuple(
            int(item.strip())
            for item in args.apr_local_residual_kernels.split(",")
            if item.strip()
        ),
        apr_local_residual_hidden_size=args.apr_local_residual_hidden_size,
        apr_local_residual_dropout=args.apr_local_residual_dropout,
        apr_local_residual_max_delta=args.apr_local_residual_max_delta,
        apr_region_head=args.apr_region_head,
        apr_region_lengths=tuple(
            int(item.strip())
            for item in args.apr_region_lengths.split(",")
            if item.strip()
        ),
        apr_region_dropout=args.apr_region_dropout,
        apr_region_max_residue_delta=args.apr_region_max_residue_delta,
        structure_sequence_embeddings=args.structure_sequence_embeddings,
        structure_apr_probabilities=args.structure_apr_probabilities,
        structure_apr_gradient_beta=args.structure_apr_gradient_beta,
        structure_peptide_attention=getattr(
            args, "structure_peptide_attention", False
        ),
        structure_peptide_attention_alpha_init=(
            getattr(args, "structure_peptide_attention_alpha_init", 0.1)
        ),
        structure_peptide_attention_alpha_max=(
            getattr(args, "structure_peptide_attention_alpha_max", 1.0)
        ),
    ).to(device)
    transfer_checkpoint = getattr(args, "peptide_transfer_checkpoint", None)
    if transfer_checkpoint is not None:
        report = load_peptide_transfer_checkpoint(
            model, transfer_checkpoint, map_location=device
        )
        print(
            "peptide_transfer_loaded=true "
            f"parameters={report['loaded_parameter_n']} "
            f"apr_head_loaded={report['apr_head_loaded']} "
            f"structure_loaded={report['structure_loaded']} "
            f"source={transfer_checkpoint}",
            flush=True,
        )
    return model


def main(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.log_every_epochs < 1:
        raise ValueError("--log-every-epochs must be at least 1")
    if args.apr_validation_batch_size is not None and args.apr_validation_batch_size < 1:
        raise ValueError("--apr-validation-batch-size must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive")
    if not 0.0 <= args.learning_rate_warmup_fraction < 1.0:
        raise ValueError("--learning-rate-warmup-fraction must be in [0, 1)")
    if not 0.0 < args.learning_rate_min_factor <= 1.0:
        raise ValueError("--learning-rate-min-factor must be in (0, 1]")
    if (
        args.peptide_transfer_checkpoint is not None
        and not args.peptide_transfer_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"Peptide transfer checkpoint not found: {args.peptide_transfer_checkpoint}"
        )
    if not 0.0 <= args.structure_message_dropout < 1.0:
        raise ValueError("--structure-message-dropout must be in [0, 1)")
    if not 0.0 <= args.structure_head_dropout < 1.0:
        raise ValueError("--structure-head-dropout must be in [0, 1)")
    if args.structure_graph_cutoff <= 0:
        raise ValueError("--structure-graph-cutoff must be positive")
    if args.structure_distance_scale <= 0:
        raise ValueError("--structure-distance-scale must be positive")
    if not 0.0 <= args.parameter_ema_decay < 1.0:
        raise ValueError("--parameter-ema-decay must be in [0, 1)")
    if not 1 <= args.parameter_ema_start_epoch <= args.epochs:
        raise ValueError(
            "--parameter-ema-start-epoch must be between 1 and --epochs"
        )
    if args.apr_positive_weight <= 0:
        raise ValueError("--apr-positive-weight must be positive")
    if args.apr_positive_contrastive_weight < 0:
        raise ValueError(
            "--apr-positive-contrastive-weight must be non-negative"
        )
    if args.apr_positive_contrastive_temperature <= 0:
        raise ValueError(
            "--apr-positive-contrastive-temperature must be positive"
        )
    if (
        args.apr_positive_contrastive_workbook is None
        and args.apr_positive_contrastive_weight > 0
    ):
        raise ValueError(
            "--apr-positive-contrastive-weight requires "
            "--apr-positive-contrastive-workbook"
        )
    if (
        args.apr_positive_contrastive_workbook is not None
        and args.apr_positive_contrastive_weight <= 0
    ):
        raise ValueError(
            "--apr-positive-contrastive-workbook requires a positive "
            "--apr-positive-contrastive-weight"
        )
    if (
        args.peptide_auxiliary_workbook is not None
        and args.apr_positive_contrastive_workbook is not None
    ):
        raise ValueError(
            "Use either BCE peptide auxiliary records or APR-positive "
            "contrastive records, not both"
        )
    if args.structure_positive_weight_scale <= 0:
        raise ValueError("--structure-positive-weight-scale must be positive")
    if args.structure_focal_gamma < 0:
        raise ValueError("--structure-focal-gamma must be non-negative")
    if not 0.0 <= args.structure_label_smoothing < 1.0:
        raise ValueError("--structure-label-smoothing must be in [0, 1)")
    if args.structure_pairwise_rank_weight < 0:
        raise ValueError("--structure-pairwise-rank-weight must be non-negative")
    if args.structure_pairwise_rank_temperature <= 0:
        raise ValueError("--structure-pairwise-rank-temperature must be positive")
    if args.structure_rank_memory_size < 0:
        raise ValueError("--structure-rank-memory-size must be non-negative")
    if args.structure_rank_hard_k < 0:
        raise ValueError("--structure-rank-hard-k must be non-negative")
    if args.structure_rank_memory_size and args.structure_pairwise_rank_weight == 0:
        raise ValueError("--structure-rank-memory-size requires pairwise rank loss")
    if args.apr_focal_gamma < 0:
        raise ValueError("--apr-focal-gamma must be non-negative")
    if args.apr_hard_negative_gamma < 0:
        raise ValueError("--apr-hard-negative-gamma must be non-negative")
    if args.apr_pairwise_rank_weight < 0:
        raise ValueError("--apr-pairwise-rank-weight must be non-negative")
    if args.apr_pairwise_rank_temperature <= 0:
        raise ValueError("--apr-pairwise-rank-temperature must be positive")
    if args.apr_mixed_sequence_weight <= 0:
        raise ValueError("--apr-mixed-sequence-weight must be positive")
    if args.apr_mixed_negative_weight <= 0:
        raise ValueError("--apr-mixed-negative-weight must be positive")
    if args.apr_training_window_flank is not None and args.apr_training_window_flank < 0:
        raise ValueError("--apr-training-window-flank must be non-negative")
    if not 0.0 <= args.apr_training_window_full_mix <= 1.0:
        raise ValueError("--apr-training-window-full-mix must be in [0, 1]")
    if args.apr_training_window_full_mix > 0 and args.apr_training_window_flank is None:
        raise ValueError(
            "--apr-training-window-full-mix requires --apr-training-window-flank"
        )
    if args.apr_soft_mcc_weight < 0:
        raise ValueError("--apr-soft-mcc-weight must be non-negative")
    if args.apr_smoothness_weight < 0:
        raise ValueError("--apr-smoothness-weight must be non-negative")
    if args.apr_context_kernel < 0 or (
        args.apr_context_kernel > 0 and args.apr_context_kernel % 2 == 0
    ):
        raise ValueError("--apr-context-kernel must be zero or a positive odd number")
    try:
        local_residual_kernels = tuple(
            int(item.strip())
            for item in args.apr_local_residual_kernels.split(",")
            if item.strip()
        )
    except ValueError:
        raise ValueError(
            "--apr-local-residual-kernels must be comma-separated integers"
        ) from None
    if not local_residual_kernels or any(
        kernel <= 0 or kernel % 2 == 0 for kernel in local_residual_kernels
    ):
        raise ValueError(
            "--apr-local-residual-kernels must contain positive odd integers"
        )
    if args.apr_local_residual_hidden_size <= 0:
        raise ValueError("--apr-local-residual-hidden-size must be positive")
    if not 0.0 <= args.apr_local_residual_dropout < 1.0:
        raise ValueError("--apr-local-residual-dropout must be in [0, 1)")
    if args.apr_local_residual_max_delta <= 0:
        raise ValueError("--apr-local-residual-max-delta must be positive")
    if args.apr_local_residual_penalty < 0:
        raise ValueError("--apr-local-residual-penalty must be non-negative")
    try:
        region_lengths = tuple(
            int(item.strip())
            for item in args.apr_region_lengths.split(",")
            if item.strip()
        )
    except ValueError:
        raise ValueError(
            "--apr-region-lengths must be comma-separated integers"
        ) from None
    if not region_lengths or any(length < 2 for length in region_lengths):
        raise ValueError("--apr-region-lengths must contain integers >= 2")
    if len(region_lengths) != len(set(region_lengths)):
        raise ValueError("--apr-region-lengths must not contain duplicates")
    if not 0.0 <= args.apr_region_dropout < 1.0:
        raise ValueError("--apr-region-dropout must be in [0, 1)")
    if args.apr_region_max_residue_delta <= 0:
        raise ValueError("--apr-region-max-residue-delta must be positive")
    for name in (
        "apr_region_loss_weight",
        "apr_region_consistency_weight",
        "apr_region_rank_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.5 <= args.apr_region_positive_fraction <= 1.0:
        raise ValueError("--apr-region-positive-fraction must be in [0.5, 1]")
    if args.apr_region_max_candidates_per_class < 1:
        raise ValueError("--apr-region-max-candidates-per-class must be positive")
    if args.apr_region_rank_margin < 0:
        raise ValueError("--apr-region-rank-margin must be non-negative")
    if args.apr_region_rank_temperature <= 0:
        raise ValueError("--apr-region-rank-temperature must be positive")
    if args.apr_region_head and args.task not in {"all", "sequence", "apr"}:
        raise ValueError("--apr-region-head requires an APR-containing task")
    if args.sequence_lora_rank < 0:
        raise ValueError("--sequence-lora-rank must be non-negative")
    if args.sequence_lora_rank > 0 and args.sequence_backbone == "tiny":
        raise ValueError("LoRA requires an ESM sequence backbone")
    if args.sequence_lora_rank > 0 and args.embedding_cache_dir is not None:
        raise ValueError("LoRA cannot be combined with --embedding-cache-dir")
    if args.sequence_backbone_learning_rate is not None and args.sequence_backbone_learning_rate <= 0:
        raise ValueError("--sequence-backbone-learning-rate must be positive")
    selected_residue_features = tuple(
        item.strip() for item in args.residue_feature_sets.split(",") if item.strip()
    )
    if selected_residue_features and args.residue_feature_table is None:
        raise ValueError("--residue-feature-sets require --residue-feature-table")
    if not 0.0 <= args.residue_feature_dropout < 1.0:
        raise ValueError("--residue-feature-dropout must be in [0, 1)")
    if args.peptide_residue_features:
        if args.task not in {"all", "sequence", "peptide"} and not (
            args.task == "apr" and args.peptide_transfer_checkpoint is not None
        ):
            raise ValueError("--peptide-residue-features requires a Peptide task")
        if not selected_residue_features:
            raise ValueError(
                "--peptide-residue-features requires --residue-feature-sets"
            )
        if args.peptide_apr_fusion == "none":
            raise ValueError(
                "--peptide-residue-features requires the isolated Peptide adapter"
            )
    if args.peptide_length_gate_hidden_size <= 0:
        raise ValueError("--peptide-length-gate-hidden-size must be positive")
    if args.peptide_length_conditioned_gate and not (
        args.peptide_short_cnn or args.peptide_residue_features
    ):
        raise ValueError(
            "--peptide-length-conditioned-gate requires --peptide-short-cnn "
            "or --peptide-residue-features"
        )
    if args.peptide_output_fusion_hidden_size <= 0:
        raise ValueError("--peptide-output-fusion-hidden-size must be positive")
    if not 0.0 <= args.peptide_output_main_weight <= 1.0:
        raise ValueError("--peptide-output-main-weight must be in [0, 1]")
    if (
        args.peptide_output_detach_propensity
        and args.peptide_output_fusion == "main"
    ):
        raise ValueError(
            "--peptide-output-detach-propensity requires Peptide output fusion"
        )
    if (
        args.peptide_output_fusion != "main"
        and args.peptide_apr_fusion == "none"
        and not args.peptide_local_propensity
    ):
        raise ValueError("--peptide-output-fusion requires a propensity head")
    try:
        local_propensity_kernels = tuple(
            int(item.strip())
            for item in args.peptide_local_propensity_kernels.split(",")
            if item.strip()
        )
    except ValueError:
        raise ValueError(
            "--peptide-local-propensity-kernels must be comma-separated integers"
        ) from None
    if not local_propensity_kernels or any(
        kernel <= 0 or kernel % 2 == 0 for kernel in local_propensity_kernels
    ):
        raise ValueError(
            "--peptide-local-propensity-kernels must contain positive odd integers"
        )
    if args.peptide_local_propensity_hidden_size <= 0:
        raise ValueError("--peptide-local-propensity-hidden-size must be positive")
    if not 0.0 <= args.peptide_local_propensity_dropout < 1.0:
        raise ValueError("--peptide-local-propensity-dropout must be in [0, 1)")
    if args.peptide_attention_residual_hidden_size <= 0:
        raise ValueError("--peptide-attention-residual-hidden-size must be positive")
    if args.peptide_attention_residual_heads <= 0:
        raise ValueError("--peptide-attention-residual-heads must be positive")
    if not 0.0 <= args.peptide_attention_residual_dropout < 1.0:
        raise ValueError("--peptide-attention-residual-dropout must be in [0, 1)")
    if args.peptide_attention_residual_max_delta <= 0:
        raise ValueError("--peptide-attention-residual-max-delta must be positive")
    if args.peptide_backbone_gradients:
        if args.task not in {"all", "sequence", "peptide"}:
            raise ValueError("--peptide-backbone-gradients requires a Peptide task")
        if args.sequence_lora_rank <= 0 and args.freeze_sequence_backbone:
            raise ValueError(
                "--peptide-backbone-gradients requires LoRA or an unfrozen backbone"
            )
    if not 0.0 <= args.sequence_channel_dropout < 1.0:
        raise ValueError("--sequence-channel-dropout must be in [0, 1)")
    if args.peptide_propensity_loss_weight < 0:
        raise ValueError("--peptide-propensity-loss-weight cannot be negative")
    if args.peptide_propensity_consistency_weight < 0:
        raise ValueError(
            "--peptide-propensity-consistency-weight cannot be negative"
        )
    if args.peptide_propensity_smoothness_weight < 0:
        raise ValueError(
            "--peptide-propensity-smoothness-weight cannot be negative"
        )
    if (
        (
            args.peptide_propensity_consistency_weight > 0
            or args.peptide_propensity_smoothness_weight > 0
        )
        and args.peptide_apr_fusion == "none"
        and not args.peptide_local_propensity
    ):
        raise ValueError(
            "Peptide propensity losses require --peptide-local-propensity or "
            "--peptide-apr-fusion"
        )
    if args.peptide_loss_weight < 0:
        raise ValueError("--peptide-loss-weight cannot be negative")
    if not 0.0 <= args.peptide_label_smoothing < 1.0:
        raise ValueError("--peptide-label-smoothing must be in [0, 1)")
    if args.peptide_rdrop_weight < 0:
        raise ValueError("--peptide-rdrop-weight must be non-negative")
    if (
        args.apr_positive_contrastive_workbook is not None
        and args.peptide_loss_weight <= 0
    ):
        raise ValueError(
            "APR-positive contrastive training requires Peptide BCE supervision"
        )
    if args.peptide_focal_gamma < 0:
        raise ValueError("--peptide-focal-gamma must be non-negative")
    if args.peptide_soft_mcc_weight < 0:
        raise ValueError("--peptide-soft-mcc-weight must be non-negative")
    if args.peptide_pairwise_rank_weight < 0:
        raise ValueError("--peptide-pairwise-rank-weight must be non-negative")
    if args.peptide_pairwise_rank_temperature <= 0:
        raise ValueError("--peptide-pairwise-rank-temperature must be positive")
    if args.peptide_mutation_pair_rank_weight < 0:
        raise ValueError(
            "--peptide-mutation-pair-rank-weight must be non-negative"
        )
    if args.peptide_mutation_pair_consistency_weight < 0:
        raise ValueError(
            "--peptide-mutation-pair-consistency-weight must be non-negative"
        )
    if args.peptide_mutation_pair_margin < 0:
        raise ValueError("--peptide-mutation-pair-margin must be non-negative")
    if args.peptide_mutation_pair_temperature <= 0:
        raise ValueError(
            "--peptide-mutation-pair-temperature must be positive"
        )
    try:
        mutation_pair_lengths = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in args.peptide_mutation_pair_lengths.split(",")
                    if item.strip()
                }
            )
        )
    except ValueError:
        raise ValueError(
            "--peptide-mutation-pair-lengths must contain integers"
        ) from None
    if not mutation_pair_lengths or any(length <= 0 for length in mutation_pair_lengths):
        raise ValueError(
            "--peptide-mutation-pair-lengths must contain positive integers"
        )
    args.peptide_mutation_pair_lengths_tuple = mutation_pair_lengths
    mutation_pair_enabled = (
        args.peptide_mutation_pair_rank_weight > 0
        or args.peptide_mutation_pair_consistency_weight > 0
    )
    if mutation_pair_enabled and args.task not in {"all", "sequence", "peptide"}:
        raise ValueError("Peptide mutation-pair losses require a Peptide task")
    if args.peptide_apr_pretrain_epochs < 0:
        raise ValueError("--peptide-apr-pretrain-epochs must be non-negative")
    if args.peptide_apr_pretrain_epochs > 0 and args.task not in {"all", "sequence"}:
        raise ValueError(
            "--peptide-apr-pretrain-epochs requires a task containing Peptide and APR data"
        )
    if args.peptide_apr_pretrain_epochs > 0 and args.peptide_apr_fusion == "none":
        raise ValueError(
            "--peptide-apr-pretrain-epochs requires --peptide-apr-fusion"
        )
    if args.peptide_apr_fusion != "none" and args.task not in {
        "all",
        "sequence",
        "peptide",
    } and not (
        args.task == "apr" and args.peptide_transfer_checkpoint is not None
    ):
        raise ValueError(
            "Peptide modules require a Peptide task or a Peptide transfer checkpoint"
        )
    if args.peptide_apr_fusion_warmup_epochs < 0:
        raise ValueError("--peptide-apr-fusion-warmup-epochs cannot be negative")
    if args.peptide_apr_fusion_ramp_epochs < 0:
        raise ValueError("--peptide-apr-fusion-ramp-epochs cannot be negative")
    if args.peptide_apr_late_fusion_epochs < 0:
        raise ValueError("--peptide-apr-late-fusion-epochs cannot be negative")
    if args.peptide_apr_late_fusion_learning_rate <= 0:
        raise ValueError(
            "--peptide-apr-late-fusion-learning-rate must be positive"
        )
    if args.peptide_apr_late_fusion_epochs > 0 and args.peptide_apr_fusion == "none":
        raise ValueError(
            "--peptide-apr-late-fusion-epochs requires --peptide-apr-fusion"
        )
    if args.peptide_apr_late_fusion_epochs > 0 and args.task == "peptide":
        raise ValueError("APR late fusion is not available for --task peptide")
    if (
        args.peptide_apr_fusion_warmup_epochs
        + args.peptide_apr_fusion_ramp_epochs
        > args.epochs
    ):
        raise ValueError(
            "Peptide-to-APR fusion warmup plus ramp cannot exceed --epochs"
        )
    if args.peptide_apr_beta_max <= 0:
        raise ValueError("--peptide-apr-beta-max must be positive")
    if args.peptide_apr_window_size <= 0 or args.peptide_apr_window_size % 2 == 0:
        raise ValueError("--peptide-apr-window-size must be a positive odd integer")
    if args.peptide_apr_feature_residual_hidden_size <= 0:
        raise ValueError(
            "--peptide-apr-feature-residual-hidden-size must be positive"
        )
    if not 0 < args.peptide_apr_beta_init < args.peptide_apr_beta_max:
        raise ValueError("--peptide-apr-beta-init must be in (0, beta_max)")
    if (
        args.peptide_apr_fusion.endswith("detached")
        and args.peptide_propensity_loss_weight <= 0
        and args.peptide_apr_pretrain_epochs <= 0
    ):
        raise ValueError(
            "Detached Peptide-to-APR fusion requires propensity supervision or pretraining"
        )
    if not 0 < args.apr_weak_negative_weight <= 1:
        raise ValueError("--apr-weak-negative-weight must be in (0, 1]")
    if args.apr_mapped_negative_weight <= 0:
        raise ValueError("--apr-mapped-negative-weight must be positive")
    if not 0.0 <= args.structure_apr_gradient_beta <= 1.0:
        raise ValueError("--structure-apr-gradient-beta must be in [0, 1]")
    if args.structure_apr_gradient_warmup_epochs < 0:
        raise ValueError("--structure-apr-gradient-warmup-epochs cannot be negative")
    if args.structure_apr_gradient_warmup_epochs > args.epochs:
        raise ValueError("--structure-apr-gradient-warmup-epochs cannot exceed --epochs")
    if args.structure_apr_gradient_beta > 0 and not args.structure_apr_probabilities:
        raise ValueError(
            "--structure-apr-gradient-beta requires --structure-apr-probabilities"
        )
    if args.structure_apr_probabilities and args.task != "all":
        raise ValueError("APR-to-Structure fusion currently requires --task all")
    if args.structure_peptide_attention:
        if args.task not in {"all", "structure"}:
            raise ValueError("Structure Peptide attention requires a Structure task")
        if args.structure_peptide_evidence is None:
            raise ValueError(
                "--structure-peptide-attention requires --structure-peptide-evidence"
            )
        if args.structure_pooling != "meanmaxstdattn":
            raise ValueError(
                "Structure Peptide attention requires --structure-pooling meanmaxstdattn"
            )
    if args.structure_peptide_attention_alpha_max <= 0:
        raise ValueError("Structure Peptide attention alpha max must be positive")
    if not (
        0
        < args.structure_peptide_attention_alpha_init
        < args.structure_peptide_attention_alpha_max
    ):
        raise ValueError(
            "Structure Peptide attention alpha init must be in (0, alpha max)"
        )
    if args.early_stopping:
        task_monitors = {
            "peptide": {
                "total_loss",
                "peptide_loss",
                "peptide_propensity_loss",
                "peptide_mcc_at_0_5",
                "peptide_transfer_score",
            },
            "apr": {"total_loss", "apr_loss", "apr_mcc_at_0_5"},
            "structure": {
                "total_loss",
                "structure_loss",
                "structure_mcc_at_0_5",
            },
        }
        if args.task in task_monitors and args.early_stopping_monitor not in task_monitors[args.task]:
            raise ValueError(
                f"--early-stopping-monitor {args.early_stopping_monitor} is not "
                f"available for --task {args.task}"
            )
        if not 0 < args.early_stopping_validation_fraction < 0.5:
            raise ValueError("--early-stopping-validation-fraction must be in (0, 0.5)")
        if args.early_stopping_patience < 1:
            raise ValueError("--early-stopping-patience must be at least 1")
        if not 1 <= args.early_stopping_min_epochs <= args.epochs:
            raise ValueError("--early-stopping-min-epochs must be between 1 and --epochs")
        if not 1 <= args.early_stopping_selection_start_epoch <= args.epochs:
            raise ValueError(
                "--early-stopping-selection-start-epoch must be between 1 and --epochs"
            )
        if args.early_stopping_min_delta < 0:
            raise ValueError("--early-stopping-min-delta cannot be negative")
        if (
            args.early_stopping_monitor == "peptide_transfer_score"
            and not args.peptide_local_propensity
        ):
            raise ValueError(
                "--early-stopping-monitor peptide_transfer_score requires "
                "--peptide-local-propensity"
            )
    set_seed(args.seed)
    device = choose_device(args.device)
    root = args.data_root
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config["data_root"] = str(args.data_root)
    run_config["output_root"] = str(args.output_root)
    run_config["split_manifest"] = str(args.split_manifest)
    run_config["sequence_model_path"] = (
        str(args.sequence_model_path) if args.sequence_model_path is not None else None
    )
    run_config["peptide_transfer_checkpoint"] = (
        str(args.peptide_transfer_checkpoint)
        if args.peptide_transfer_checkpoint is not None
        else None
    )
    run_config["embedding_cache_dir"] = (
        str(args.embedding_cache_dir) if args.embedding_cache_dir is not None else None
    )
    run_config["residue_feature_table"] = (
        str(args.residue_feature_table) if args.residue_feature_table is not None else None
    )
    run_config["apr_weak_negatives"] = (
        str(args.apr_weak_negatives) if args.apr_weak_negatives is not None else None
    )
    run_config["peptide_auxiliary_workbook"] = (
        str(args.peptide_auxiliary_workbook)
        if args.peptide_auxiliary_workbook is not None
        else None
    )
    run_config["apr_positive_contrastive_workbook"] = (
        str(args.apr_positive_contrastive_workbook)
        if args.apr_positive_contrastive_workbook is not None
        else None
    )
    run_config["structure_peptide_evidence"] = (
        str(args.structure_peptide_evidence)
        if args.structure_peptide_evidence is not None
        else None
    )
    run_config["device"] = str(device)
    (args.output_root / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    datasets: dict[str, Any] = {}
    if args.task in {"all", "sequence", "peptide"}:
        datasets["peptide"] = AggregatingPeptideDataset(
            root / "aggregating_peptides_model_database.xlsx"
        )
    auxiliary_peptide_records = []
    if args.peptide_auxiliary_workbook is not None:
        if "peptide" not in datasets:
            raise ValueError("--peptide-auxiliary-workbook requires a Peptide task")
        auxiliary_dataset = AggregatingPeptideDataset(args.peptide_auxiliary_workbook)
        auxiliary_peptide_records = list(auxiliary_dataset.records)
    apr_positive_contrastive_dataset: AggregatingPeptideDataset | None = None
    if args.apr_positive_contrastive_workbook is not None:
        if "peptide" not in datasets:
            raise ValueError(
                "--apr-positive-contrastive-workbook requires a Peptide task"
            )
        apr_positive_contrastive_dataset = AggregatingPeptideDataset(
            args.apr_positive_contrastive_workbook
        )
        if any(record.label != 1 for record in apr_positive_contrastive_dataset.records):
            raise ValueError(
                "APR-positive contrastive workbook must contain only label=1 records"
            )
    if args.task in {"all", "sequence", "apr"}:
        datasets["apr"] = APRDataset(
            root / "apr_model_database.xlsx",
            aggregating_workbook=root / "aggregating_peptides_model_database.xlsx",
            use_aggregating_negatives=not args.no_aggregating_negatives,
            weak_negative_workbook=args.apr_weak_negatives,
            weak_negative_weight=args.apr_weak_negative_weight,
            mapped_negative_weight=args.apr_mapped_negative_weight,
        )
    if args.task in {"all", "structure"}:
        datasets["structure"] = AmyloidStructureDataset(
            root / "amyloid_structure_model_database.xlsx",
            root / "pdb",
            graph_cutoff=args.structure_graph_cutoff,
            graph_edge_weighting=args.structure_edge_weighting,
            graph_distance_scale=args.structure_distance_scale,
            normalize_position=args.structure_normalize_position,
            geometric_features=args.structure_geometric_features,
            peptide_evidence_file=args.structure_peptide_evidence,
        )
    counts = ", ".join(f"{name}={len(dataset)}" for name, dataset in datasets.items())
    print(f"Random {args.folds}-fold task={args.task} on {counts}")
    if "apr" in datasets:
        weak_count = sum(record.training_only for record in datasets["apr"].records)
        weak_residues = sum(
            int((record.target == 0).sum().item())
            for record in datasets["apr"].records
            if record.training_only
        )
        print(
            f"APR training-only weak negatives={weak_count} "
            f"residues={weak_residues} "
            f"residue_weight={args.apr_weak_negative_weight:.4f} "
            f"effective_residues={weak_residues * args.apr_weak_negative_weight:.2f}"
        )
    if apr_positive_contrastive_dataset is not None:
        print(
            "apr_positive_contrastive_records="
            f"{len(apr_positive_contrastive_dataset)} "
            f"weight={args.apr_positive_contrastive_weight:.4f} "
            f"temperature={args.apr_positive_contrastive_temperature:.4f} "
            "supervision=representation_only bce=false"
        )
    print(
        f"gradient_clipping={args.gradient_clipping} max_grad_norm={args.max_grad_norm:.4f}"
    )
    if "apr" in datasets:
        print(
            f"apr_positive_weight={args.apr_positive_weight:.4f} "
            f"apr_sequence_balanced_loss={args.apr_sequence_balanced_loss} "
            f"apr_focal_gamma={args.apr_focal_gamma:.4f} "
            f"apr_hard_negative_gamma={args.apr_hard_negative_gamma:.4f} "
            f"apr_mixed_sequence_weight={args.apr_mixed_sequence_weight:.4f} "
            f"apr_mixed_negative_weight={args.apr_mixed_negative_weight:.4f} "
            f"apr_training_window_flank={args.apr_training_window_flank} "
            f"apr_training_window_full_mix={args.apr_training_window_full_mix:.4f} "
            f"apr_soft_mcc_weight={args.apr_soft_mcc_weight:.4f} "
            f"apr_smoothness_weight={args.apr_smoothness_weight:.4f} "
            f"apr_pairwise_rank_weight={args.apr_pairwise_rank_weight:.4f} "
            f"apr_pairwise_rank_temperature={args.apr_pairwise_rank_temperature:.4f} "
            f"apr_mapped_negative_weight={args.apr_mapped_negative_weight:.4f} "
            f"apr_context_kernel={args.apr_context_kernel} "
            f"apr_local_residual={args.apr_local_residual} "
            f"apr_local_residual_kernels={args.apr_local_residual_kernels} "
            f"apr_local_residual_max_delta={args.apr_local_residual_max_delta:.4f} "
            f"apr_local_residual_penalty={args.apr_local_residual_penalty:.4f}"
        )
    if args.sequence_lora_rank > 0:
        print(
            "sequence_lora=true "
            f"rank={args.sequence_lora_rank} "
            f"last_n_layers={args.sequence_lora_last_n_layers} "
            f"backbone_lr={args.sequence_backbone_learning_rate}"
        )
    if selected_residue_features:
        print(
            "residue_features="
            f"{','.join(selected_residue_features)} "
            f"gate_init={args.residue_feature_gate_init:.4f} "
            f"dropout={args.residue_feature_dropout:.4f}"
        )
    if args.peptide_length_conditioned_gate:
        print(
            "peptide_length_conditioned_gate=true "
            f"hidden_size={args.peptide_length_gate_hidden_size}"
        )
    if args.peptide_output_fusion != "main":
        print(
            f"peptide_output_fusion={args.peptide_output_fusion} "
            f"hidden_size={args.peptide_output_fusion_hidden_size} "
            "probability_space=true fixed_threshold=0.5"
        )
    if args.peptide_attention_residual:
        print(
            "peptide_attention_residual=true "
            f"heads={args.peptide_attention_residual_heads} "
            f"hidden_size={args.peptide_attention_residual_hidden_size} "
            f"max_delta={args.peptide_attention_residual_max_delta:.4f}"
        )
    if mutation_pair_enabled:
        print(
            "peptide_mutation_pairs=true "
            f"lengths={args.peptide_mutation_pair_lengths_tuple} "
            f"rank_weight={args.peptide_mutation_pair_rank_weight:.4f} "
            "consistency_weight="
            f"{args.peptide_mutation_pair_consistency_weight:.4f} "
            f"margin={args.peptide_mutation_pair_margin:.4f} "
            f"temperature={args.peptide_mutation_pair_temperature:.4f}"
        )
    if args.sequence_channel_dropout > 0:
        print(f"sequence_channel_dropout={args.sequence_channel_dropout:.4f}")
    if args.peptide_apr_fusion != "none":
        print(
            f"peptide_apr_fusion={args.peptide_apr_fusion} "
            f"classification_loss_weight={args.peptide_loss_weight:.4f} "
            f"propensity_loss_weight={args.peptide_propensity_loss_weight:.4f} "
            f"focal_gamma={args.peptide_focal_gamma:.4f} "
            f"soft_mcc_weight={args.peptide_soft_mcc_weight:.4f} "
            f"pairwise_rank_weight={args.peptide_pairwise_rank_weight:.4f} "
            f"pairwise_rank_temperature={args.peptide_pairwise_rank_temperature:.4f} "
            f"beta_init={args.peptide_apr_beta_init:.4f} "
            f"beta_max={args.peptide_apr_beta_max:.4f} "
            f"window_size={args.peptide_apr_window_size} "
            "feature_residual_hidden_size="
            f"{args.peptide_apr_feature_residual_hidden_size} "
            f"warmup_epochs={args.peptide_apr_fusion_warmup_epochs} "
            f"ramp_epochs={args.peptide_apr_fusion_ramp_epochs} "
            f"late_fusion_epochs={args.peptide_apr_late_fusion_epochs} "
            f"late_fusion_lr={args.peptide_apr_late_fusion_learning_rate:.6f}"
        )
    if args.peptide_apr_pretrain_epochs > 0:
        print(
            "peptide_apr_pretraining=true "
            f"epochs={args.peptide_apr_pretrain_epochs} "
            "isolated_adapter=true pooling=logmeanexp"
        )
    if args.structure_apr_probabilities:
        print(
            "structure_apr_probabilities=true "
            f"gradient_beta={args.structure_apr_gradient_beta:.4f} "
            f"warmup_epochs={args.structure_apr_gradient_warmup_epochs}"
        )
    manifest_metadata = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    print(f"split_method={manifest_metadata.get('method', 'unknown')}")
    if manifest_metadata.get("warning"):
        print(f"WARNING: {manifest_metadata['warning']}")
    peptide_manifest = manifest_metadata.get("datasets", {}).get("peptide", {})
    if (
        args.early_stopping
        and peptide_manifest.get("group_column")
        and not args.peptide_inner_group_disjoint
    ):
        raise ValueError(
            "Grouped Peptide manifests require --peptide-inner-group-disjoint "
            "when early stopping is enabled"
        )

    splits = load_fixed_splits(args.split_manifest, datasets, args.folds)
    # Keep the manifest-defined Peptide records for inner validation and OOF
    # evaluation. APR-derived positive records are training-only and are added
    # fold by fold after removing records related to that fold's held-out peptides.
    selection_splits = {
        name: list(fold_splits) for name, fold_splits in splits.items()
    }
    if auxiliary_peptide_records:
        peptide_dataset = datasets["peptide"]
        base_peptide_count = len(peptide_dataset.records)
        peptide_dataset.records.extend(auxiliary_peptide_records)
        auxiliary_indices = np.arange(
            base_peptide_count,
            base_peptide_count + len(auxiliary_peptide_records),
            dtype=np.int64,
        )
        augmented_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold, (base_train, base_valid) in enumerate(selection_splits["peptide"]):
            held_out_sequences = {
                peptide_dataset.records[int(index)].sequence for index in base_valid
            }
            keep = np.asarray(
                [
                    int(index)
                    for index in auxiliary_indices
                    if not any(
                        peptide_dataset.records[int(index)].sequence in held_out
                        or held_out in peptide_dataset.records[int(index)].sequence
                        for held_out in held_out_sequences
                    )
                ],
                dtype=np.int64,
            )
            augmented_splits.append(
                (np.concatenate([base_train, keep]), base_valid)
            )
        splits["peptide"] = augmented_splits
        print(
            f"peptide_auxiliary_positive_records={len(auxiliary_peptide_records)} "
            f"source={args.peptide_auxiliary_workbook}"
        )
    all_results: list[dict[str, Any]] = []
    oof_prediction_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in datasets
    }
    if args.apr_region_head:
        oof_prediction_rows["apr_region_profile"] = []
    loader_configs = {
        "peptide": (args.batch_size, collate_peptides),
        "apr": (args.batch_size, collate_apr),
        "structure": (args.structure_batch_size, collate_structures),
    }
    validation_loader_configs = dict(loader_configs)
    if args.apr_validation_batch_size is not None:
        validation_loader_configs["apr"] = (
            args.apr_validation_batch_size,
            collate_apr,
        )
    training_collates = dict(loader_configs)
    if args.apr_training_window_flank is not None:
        training_collates["apr"] = (
            args.batch_size,
            partial(
                collate_apr,
                training_window_flank=args.apr_training_window_flank,
                training_window_full_mix=args.apr_training_window_full_mix,
            ),
        )

    for fold in range(args.folds):
        print(f"\n===== Fold {fold + 1}/{args.folds} =====")
        fold_seed = args.seed + fold
        structure_pos_weight: float | None = None
        if "structure" in datasets:
            structure_pos_weight = structure_positive_weight(
                datasets["structure"], splits["structure"][fold][0]
            )
            print(f"structure_pos_weight={structure_pos_weight:.6f}")
        train_loaders: dict[str, DataLoader] = {}
        valid_loaders: dict[str, DataLoader] = {}
        for task_number, (name, dataset) in enumerate(datasets.items()):
            batch_size, collate_fn = training_collates[name]
            train_loaders[name] = make_loader(
                dataset,
                splits[name][fold][0],
                batch_size,
                collate_fn,
                True,
                args.num_workers,
                balanced_binary=(
                    name == "structure" and args.structure_balanced_batches
                ),
                seed=fold_seed + 10_000 * (task_number + 1),
            )
            validation_batch_size, validation_collate_fn = validation_loader_configs[name]
            valid_loaders[name] = make_loader(
                dataset,
                splits[name][fold][1],
                validation_batch_size,
                validation_collate_fn,
                False,
                args.num_workers,
                seed=fold_seed + 10_000 * (task_number + 1) + 5_000,
            )

        if mutation_pair_enabled:
            mutation_loader, opposite_pairs, same_pairs = (
                make_peptide_mutation_pair_loader(
                    datasets["peptide"],
                    selection_splits["peptide"][fold][0],
                    args.peptide_mutation_pair_lengths_tuple,
                    args.batch_size,
                    args.num_workers,
                    seed=fold_seed + 80_000,
                )
            )
            train_loaders["peptide_mutation_pairs"] = mutation_loader
            print(
                f"fold_{fold + 1}_peptide_mutation_pairs: "
                f"opposite={opposite_pairs} same={same_pairs} "
                f"lengths={args.peptide_mutation_pair_lengths_tuple}"
            )

        if apr_positive_contrastive_dataset is not None:
            peptide_dataset = datasets["peptide"]
            held_out_sequences = {
                peptide_dataset.records[int(index)].sequence
                for index in selection_splits["peptide"][fold][1]
            }
            apr_positive_indices = leakage_filtered_auxiliary_indices(
                apr_positive_contrastive_dataset,
                held_out_sequences,
            )
            if apr_positive_indices.size == 0:
                raise ValueError(
                    f"Fold {fold + 1} has no leakage-free APR-positive records"
                )
            train_loaders["apr_positive_contrastive"] = make_loader(
                apr_positive_contrastive_dataset,
                apr_positive_indices,
                args.batch_size,
                collate_peptides,
                True,
                args.num_workers,
                seed=fold_seed + 90_000,
            )
            print(
                f"fold_{fold + 1}_apr_positive_contrastive: "
                f"kept={len(apr_positive_indices)} "
                f"filtered={len(apr_positive_contrastive_dataset) - len(apr_positive_indices)}"
            )

        selected_epoch = args.epochs
        best_inner_monitor: float | None = None
        if args.early_stopping:
            inner_train_indices: dict[str, np.ndarray] = {}
            selection_train_loaders: dict[str, DataLoader] = {}
            selection_valid_loaders: dict[str, DataLoader] = {}
            for task_number, (name, dataset) in enumerate(datasets.items()):
                inner_train, inner_valid = make_inner_validation_split(
                    name,
                    dataset,
                    selection_splits[name][fold][0],
                    validation_fraction=args.early_stopping_validation_fraction,
                    seed=fold_seed + 100 * (task_number + 1),
                    group_disjoint=(
                        name == "peptide" and args.peptide_inner_group_disjoint
                    ),
                )
                inner_train_indices[name] = inner_train
                batch_size, collate_fn = training_collates[name]
                selection_train_loaders[name] = make_loader(
                    dataset,
                    inner_train,
                    batch_size,
                    collate_fn,
                    True,
                    args.num_workers,
                    balanced_binary=(
                        name == "structure" and args.structure_balanced_batches
                    ),
                    seed=fold_seed + 20_000 * (task_number + 1),
                )
                validation_batch_size, validation_collate_fn = validation_loader_configs[name]
                selection_valid_loaders[name] = make_loader(
                    dataset,
                    inner_valid,
                    validation_batch_size,
                    validation_collate_fn,
                    False,
                    args.num_workers,
                    seed=fold_seed + 20_000 * (task_number + 1) + 5_000,
                )
                print(
                    f"inner_{name}: train={len(inner_train)} valid={len(inner_valid)}"
                )
            if mutation_pair_enabled:
                mutation_loader, opposite_pairs, same_pairs = (
                    make_peptide_mutation_pair_loader(
                        datasets["peptide"],
                        inner_train_indices["peptide"],
                        args.peptide_mutation_pair_lengths_tuple,
                        args.batch_size,
                        args.num_workers,
                        seed=fold_seed + 85_000,
                    )
                )
                selection_train_loaders["peptide_mutation_pairs"] = mutation_loader
                print(
                    "inner_peptide_mutation_pairs: "
                    f"opposite={opposite_pairs} same={same_pairs} "
                    f"lengths={args.peptide_mutation_pair_lengths_tuple}"
                )
            if apr_positive_contrastive_dataset is not None:
                peptide_dataset = datasets["peptide"]
                outer_valid = selection_splits["peptide"][fold][1]
                inner_valid = selection_valid_loaders["peptide"].dataset.indices
                held_out_sequences = {
                    peptide_dataset.records[int(index)].sequence
                    for index in itertools.chain(outer_valid, inner_valid)
                }
                selection_apr_positive_indices = (
                    leakage_filtered_auxiliary_indices(
                        apr_positive_contrastive_dataset,
                        held_out_sequences,
                    )
                )
                if selection_apr_positive_indices.size == 0:
                    raise ValueError(
                        f"Fold {fold + 1} inner split has no leakage-free "
                        "APR-positive records"
                    )
                selection_train_loaders["apr_positive_contrastive"] = (
                    make_loader(
                        apr_positive_contrastive_dataset,
                        selection_apr_positive_indices,
                        args.batch_size,
                        collate_peptides,
                        True,
                        args.num_workers,
                        seed=fold_seed + 95_000,
                    )
                )
                print(
                    f"inner_apr_positive_contrastive: "
                    f"train={len(selection_apr_positive_indices)} valid=0"
                )
            selection_structure_pos_weight: float | None = None
            if "structure" in datasets:
                selection_structure_pos_weight = structure_positive_weight(
                    datasets["structure"], inner_train_indices["structure"]
                )
                print(
                    "selection_structure_pos_weight="
                    f"{selection_structure_pos_weight:.6f}"
                )
            set_seed(fold_seed)
            selection_model = build_model(args, device)
            if args.peptide_apr_pretrain_epochs > 0:
                pretrain_peptide_branch(
                    selection_model,
                    selection_train_loaders["peptide"],
                    args,
                    device,
                    epochs=args.peptide_apr_pretrain_epochs,
                    phase="peptide_apr_pretrain_selection",
                )
            _, selected_epoch, best_inner_monitor = train_one_fold(
                selection_model,
                apr_finetune_loaders(selection_train_loaders, args),
                args,
                device,
                structure_pos_weight=selection_structure_pos_weight,
                epochs=args.epochs,
                validation_loaders=apr_finetune_loaders(
                    selection_valid_loaders, args
                ),
                validation_structure_pos_weight=selection_structure_pos_weight,
                phase="epoch_selection",
            )
            print(
                f"Selected epoch {selected_epoch}/{args.epochs} "
                f"from inner {args.early_stopping_monitor}="
                f"{best_inner_monitor:.6f}"
            )
            del selection_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        set_seed(fold_seed)
        model = build_model(args, device)
        if args.peptide_apr_pretrain_epochs > 0:
            pretrain_peptide_branch(
                model,
                train_loaders["peptide"],
                args,
                device,
                epochs=args.peptide_apr_pretrain_epochs,
                phase="peptide_apr_pretrain_full",
            )
        losses, _, _ = train_one_fold(
            model,
            apr_finetune_loaders(train_loaders, args),
            args,
            device,
            structure_pos_weight=structure_pos_weight,
            epochs=selected_epoch,
            phase="full_retrain" if args.early_stopping else "train",
        )
        if args.peptide_apr_late_fusion_epochs > 0:
            losses.update(
                fit_peptide_apr_late_fusion(
                    model,
                    train_loaders["apr"],
                    args,
                    device,
                )
            )
        metrics = evaluate(
            model,
            valid_loaders,
            device,
            args,
            prediction_rows=oof_prediction_rows,
            fold=fold + 1,
        )
        result = {
            "fold": fold + 1,
            **losses,
            **metrics,
        }
        if args.early_stopping:
            result["selected_epoch"] = selected_epoch
            result[f"best_inner_{args.early_stopping_monitor}"] = best_inner_monitor
        if structure_pos_weight is not None:
            result["structure_pos_weight"] = structure_pos_weight
        all_results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
        if args.save_fold_checkpoints:
            torch.save(
                {"fold": fold + 1, "model": model.state_dict(), "args": vars(args)},
                args.output_root / f"fold_{fold + 1:02d}.pt",
            )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_names = [name for name in all_results[0] if name not in {"fold"} and name != "n"]
    summary: dict[str, Any] = {
        "folds": args.folds,
        "seed": args.seed,
        "device": str(device),
        "task": args.task,
        "decision_threshold": 0.5,
        "gradient_clipping": args.gradient_clipping,
        "max_grad_norm": args.max_grad_norm,
    }
    for name in metric_names:
        values = np.asarray([row[name] for row in all_results], dtype=float)
        summary[f"{name}_mean"] = float(np.nanmean(values))
        summary[f"{name}_std"] = float(np.nanstd(values))
    for dataset_name, rows in oof_prediction_rows.items():
        if not rows:
            continue
        if dataset_name == "apr_region_profile":
            output_path = args.output_root / "oof_apr_region_profiles.csv"
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            continue
        pooled = binary_metrics(
            [float(row["label"]) for row in rows],
            [float(row["probability"]) for row in rows],
        )
        summary.update(
            {
                f"{dataset_name}_oof_{name}": value
                for name, value in pooled.items()
            }
        )
        if dataset_name == "peptide" and "propensity_probability" in rows[0]:
            pooled_propensity = binary_metrics(
                [float(row["label"]) for row in rows],
                [float(row["propensity_probability"]) for row in rows],
            )
            summary.update(
                {
                    f"peptide_propensity_oof_{name}": value
                    for name, value in pooled_propensity.items()
                }
            )
        output_path = args.output_root / f"oof_{dataset_name}_predictions.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "fold_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    with (args.output_root / "fold_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print("\n===== Mean +/- SD =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main(parse_args())

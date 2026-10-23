# Residue-level APR encoder

Six stride-1 scales (6, 8, 10, 12, 16, 20 aa) provide frozen teacher window tokens. Within-scale residue mapping, scale fusion and boundary-aware context produce **HAPR `[L,192]`**, with pAPR and rAPR readouts `[L]`. The retained encoder has zero-valued ESM-shaped base inputs; this encoder stage is separate from the protein classifier's 650M ESM-2 inputs.

| Directory/file | Responsibility |
| --- | --- |
| `model/residue_encoder.py` | Final V3 encoder and strict checkpoint loader |
| `model/context.py`, `scale_fusion.py`, `local_encoder.py`, `window_attention.py` | Required inherited network components |
| `my_loader/window_data.py`, `labels.py` | APR sequences, masks, chunks and cache collation |
| `utils/objectives.py`, `support.py`, `contrastive.py` | Training objectives and consistency helpers |
| `train/train.py` | Fold-based encoder fitting |
| `train/refit.py`, `full_data.py` | Full-data recipe with released data paths and explicit external caches |
| `scratch/get_feature/` | Window tokens, zero base inputs and HAPR extraction |
| `pretrained/encoder_epoch20.pt` | Unchanged existing retained encoder weights |

## Inputs and extraction

A workbook has a `sequences` sheet with `sequence_id`, `sequence`; supervised training also needs an `apr_positive_mask` string of one 0/1 character per residue. A training fold CSV supplies `dataset=apr`, `item_id`, `fold`. Preserve study identifiers such as the CPADPEP prefix, which controls the 0.15 source weight.

```bash
python -m encoder.scratch.get_feature.window_tokens \
  --backbone /assets/esm2_t6_8M_UR50D \
  --peptide-checkpoint teacher/pretrained/teacher_canya_cpad.pt \
  --apr-workbook /data/sequences.xlsx --output-dir /cache/new_window_tokens \
  --window-lengths 6,8,10,12,16,20 --stride 1 --batch-size 128 --device cuda
python -m encoder.scratch.get_feature.zero_base \
  --apr-workbook /data/sequences.xlsx --output-dir /cache/zero_base
```

A trained encoder must use its matching teacher and feature definition. The example above generates **masked-v2 inputs for a new encoder fit**, not for the unchanged retained encoder. Read [the enforced artifact distinction](pretrained/README.md). No refit or regeneration has been performed by this repair.

```bash
python -m encoder.scratch.get_feature.extract_hapr \
  --workbook /data/sequences.xlsx --window-cache /cache/compatible_window_tokens \
  --esm-cache /cache/zero_base --checkpoint encoder/pretrained/encoder_epoch20.pt \
  --output-dir /external/source/embeddings/hapr/epoch20 --device cuda
```

The extractor chunks at 512 residues with overlap64 and averages overlaps. It exports an ordered manifest and `.npz` files containing `hapr`, `pAPR`, `rAPR`. None of these generated files are included in this repository.

## Training

```bash
python -m encoder.train.train --apr-workbook encoder/data/cpad_full_v1/apr_mixed_unified_v1.xlsx \
  --fold-csv /data/encoder_splits/homology10/apr_folds.csv --peptide-window-cache /cache/window_tokens \
  --esm-cache /cache/zero_base --output-dir /runs/encoder_cv \
  --boundary-detection-weight 0.10 --cpad-weight 0.15 \
  --pos-weight-power 0.35 --supcon-weight 0 --device cuda
```

Retained settings are in `utils/training_config.json`. Full-data fitting defaults to the released workbook and homology folds in `encoder/data`; override them with `--workbook` and `--fold-csv` if needed:

```bash
python -m encoder.train.refit \
  --teacher-checkpoint teacher/pretrained/teacher_canya_cpad.pt \
  --window-cache /cache/new_window_tokens --base-cache /cache/zero_base \
  --output /runs/new_canonical_encoder \
  --epochs 20 --device cuda
```

Teacher hash and feature-policy checks remain enabled. Using a frozen final teacher does not require its CAnYA training initializer. Cache readers check exact sequences, including same-ID mutations/deletions. Imports do not start fitting or extraction.

# Peptide teacher

**Training:** CAnYA shared-supervised pretraining → CPAD peptide fine-tuning → full-data teacher. The supplied final checkpoint is `pretrained/teacher_canya_cpad.pt` (36 final epochs, 2,031 recorded peptide examples). It contains the saved arguments and complete teacher model state.

| File | Responsibility |
| --- | --- |
| `model/models.py` | ESM-2/LoRA, sequence/CNN and propensity components required by the teacher checkpoint |
| `model/apr_regions.py` | Shared model components imported by the original architecture |
| `my_loader/data.py` | Original peptide workbook schema, datasets and collators |
| `train/canya.py` | Original CAnYA shared output and dataset-specific auxiliary supervision |
| `train/cross_validate.py` | Original CV, optimization and selection implementation |
| `train/full_data.py` | Full-data teacher fit using CV-selected epoch counts |
| `utils/` | Path preparation, training records, optimizer and legacy rotary compatibility |
| `scratch/get_feature/` | CAnYA data preparation and peptide inference |
| `feature/` | Fixed residue descriptor definitions |

## 1. CAnYA pretraining

The retained recipe starts from an existing peptide initializer, not an undocumented random initialization. Supply that initializer and its compatible configuration separately. The CAnYA TSV columns are `aa_seq`, `seed_bh`, `dataset`, `train_test`.

```bash
python -m teacher.train.pretrain --backbone /assets/esm2_t6_8M_UR50D \
  --project-root teacher --input /data/teacher_splits/canya_random90_10.tsv \
  --checkpoint /assets/initial_peptide_transfer.pt \
  --compat-config teacher/utils/canya_initialization_config.json \
  --output-dir /runs/canya --official-split --evaluate-test \
  --epochs 3 --scheduler-epochs 15 --batch-size 256 \
  --head-lr 0.0005 --lora-lr 0.00005 --backbone-lr 0.00001 \
  --dataset-heads --dataset-head-lr 0.001 --dataset-head-hidden-size 256 \
  --dataset-head-dropout 0.1 --shared-loss-weight 1 --dataset-aux-weight 0.3 \
  --dataset-residual-penalty 0.01 --weight-decay 0.0001 \
  --propensity-weight 0.2 --pos-weight 1 --train-mode full \
  --seed 2026 --patience 4 --device cuda
```

`--official-split` reads the supplied train/test column. The recorded preparation is a random within-dataset 90/10 split, not homology isolation. The original fixed evaluation threshold is recorded in `utils/canya_training_record.json`.

## 2. CPAD fine-tuning

The source workbook is `teacher/data/aggregating_peptides_model_database.xlsx`; fold outputs are generated outside the repository. Relocate the saved configuration before training:

```bash
python -m teacher.utils.prepare_config --data-root teacher/data \
  --backbone /assets/esm2_t6_8M_UR50D \
  --canya-checkpoint /runs/canya/best_transfer.pt \
  --split-manifest /data/teacher_splits/cpad_random10/fold_manifest.json \
  --output-root /runs/cpad --output /runs/cpad_local.json
python -m teacher.train.train --config /runs/cpad_local.json
python -m teacher.train.train_full --cv-run /runs/completed_cpad_cv \
  --output /runs/teacher_canya_cpad.pt --device cuda
```

The full-data trainer uses the rounded median of the CV-selected epochs. It does not select epochs from downstream protein test results. Shared supporting architecture classes remain where the original factory/checkpoint requires them; they are not additional released downstream classifiers.

## 3. Peptide inference

```bash
python -m teacher.scratch.get_feature.predict \
  --checkpoint teacher/pretrained/teacher_canya_cpad.pt \
  --sequence-model-path /assets/esm2_t6_8M_UR50D \
  --fasta /data/peptides.fasta --output /results/peptides.json --device cuda
```

Full-checkpoint inference bypasses stale initialization file paths, restores the supplied complete state and handles the one previously documented unused rotary-position tensor. No other parameter mismatch is suppressed.

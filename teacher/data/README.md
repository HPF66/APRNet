# Peptide teacher input data

The source workbooks and fixed fold outputs are intentionally not bundled in
the code repository. Supply the original CPAD workbook and CAnYA table from a
separate data location. `make_canya_split.py` creates the CAnYA random 90/10
table; `make_cpad_folds.py` creates the CPAD peptide K-fold manifest.

```bash
python -m teacher.my_loader.make_canya_split \
  --input /data/canya/adt5111_SD2_official_split.tsv \
  --output /data/teacher_splits/canya_random90_10.tsv --fraction 0.1 --seed 2026
python -m teacher.my_loader.make_cpad_folds \
  --workbook /data/cpad/aggregating_peptides_model_database.xlsx \
  --output-root /data/teacher_splits/cpad_random10 --folds 10 --seed 2026
```

The generated manifest is passed explicitly to the trainer. Fold files,
intermediate TSVs, split workbooks and extraction caches should remain outside
the checkout.

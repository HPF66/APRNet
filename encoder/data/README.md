# Residue encoder input data

`cpad_full_v1/apr_mixed_unified_v1.xlsx` contains 3,031 source records corresponding to 3,025 distinct sequences:

| Record class | Count |
| --- | ---: |
| APR-annotated long sequences | 332 |
| Screened UniProt background proteins (presumed negative) | 668 |
| Positive CPAD peptides | 917 |
| Negative CPAD peptides | 1,114 |

The original sheets retain sequences, per-residue `apr_positive_mask`, APR intervals, source records and selection audits. The accompanying `source_records.csv`, `negative_protein_audit.csv`, `peptide_audit.csv` and `summary.json` are copied unchanged.

`homology10/` and `random10/` contain the original `apr_folds.csv`, `fold_manifest.json` and `group_membership.csv`. Use these explicit paths with the fold trainer. The reference settings in `encoder/utils/training_config.json` point to the bundled workbook and homology fold CSV using repository-relative paths; the training CLI does not load this file automatically.

The full-data refit entry point defaults to this bundled workbook and `homology10/apr_folds.csv`. Supply the prepared window and zero-base caches through `--window-cache` and `--base-cache`; use `--workbook` and `--fold-csv` to override the data paths. The retained refit uses all 3,031 records for a fixed 20 epochs. See the [encoder training commands](../README.md#training).

No HAPR, pAPR/rAPR outputs, teacher window tokens, ESM inputs or caches are bundled. This directory contains the source data and original fixed partitions.

# Residue encoder input data

`cpad_full_v1/apr_mixed_unified_v1.xlsx` is the unchanged 3,031-record merged training workbook:

| Record class | Count |
| --- | ---: |
| APR-annotated long sequences | 332 |
| Background negative long proteins | 668 |
| Positive CPAD peptides | 917 |
| Negative CPAD peptides | 1,114 |

The original sheets retain sequences, per-residue `apr_positive_mask`, APR intervals, source records and selection audits. The accompanying `source_records.csv`, `negative_protein_audit.csv`, `peptide_audit.csv` and `summary.json` are copied unchanged.

`homology10/` and `random10/` contain the original `apr_folds.csv`, `fold_manifest.json` and `group_membership.csv`. Use these explicit paths with the fold trainer. `utils/training_config.json` already points to the bundled workbook and homology fold CSV. The specialized full-data refit command still expects the work-directory layout stated in its README; bundling these data does not start a refit.

No HAPR, pAPR/rAPR outputs, teacher window tokens, ESM inputs or caches are bundled. This directory contains the source data and original fixed partitions.

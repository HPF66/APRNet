# Residue encoder input data

`cpad_full_v1/apr_mixed_unified_v1.xlsx` is the unchanged 3,031-record merged training workbook:

| Record class | Count |
| --- | ---: |
| APR-annotated long sequences | 332 |
| Background negative long proteins | 668 |
| Positive CPAD peptides | 917 |
| Negative CPAD peptides | 1,114 |

The original sheets retain sequences, per-residue `apr_positive_mask`, APR intervals, source records and selection audits. The accompanying `source_records.csv`, `negative_protein_audit.csv`, `peptide_audit.csv` and `summary.json` are copied unchanged.

Fold outputs are intentionally not bundled. `tools/make_folds.py` creates a
random K-fold manifest, or a grouped manifest when supplied with a separately
generated `sequence_id,group_id` CSV. Keep homology clustering and its mapping
outside the checkout, then pass the generated `apr_folds.csv` explicitly to
the trainer.

No fold outputs, HAPR, pAPR/rAPR outputs, teacher window tokens, ESM inputs or
caches are bundled. The source workbook remains available separately from the
generated partitions.

# Code repair: 2026-10-20

- New window-output probabilities average real residues only and record `residue_mask_mean_v2`. Teacher/CNN forward padding is unchanged. Historical caches are not overwritten.
- HAPR extraction verifies teacher identity and feature policy against encoder inputs. It rejects the incompatible retained-encoder/bundled-teacher combination. No weight is replaced or refitted.
- New encoder exports record their input contract. Full-data fitting uses the released workbook/folds and explicit external cache paths; consuming the frozen final teacher no longer requires its training initializer.
- Cache loading checks exact sequences, duplicate IDs, NPZ identity, window geometry, finite values and optional profile geometry.
- Both graph builders resolve recorded or relocated coordinate paths, validate supplied checksums and honor `--workers`.
- DSSP is configurable/discoverable. Its absence reaches the existing masked-secondary-structure/SASA fallback.
- Protein results record cohort and partition identity. Mean/SD summaries reject mixed conditions and missing folds by default; explicitly requested interim summaries record missing folds in a sidecar.

Sequence/label workbooks, partitions, pretrained weight bytes, model layers and existing server results are unchanged. This repair performs no training or production-cache regeneration.

# Manuscript-aligned defaults and descriptions: 2026-11-13

- Set encoder training defaults to boundary-supervision weight 0.10 and CPAD-source weight 0.15, matching the manuscript, reference configuration and full-data refit recipe.
- Corrected V3 comments and export descriptions to identify the trunk boundary predictions and their role in context attention and input/output residuals. Network computation, parameter names and checkpoint files were not changed.
- Updated the encoder README and corrected the primary Cold-family validation description to the fixed five-block rotation. Split tables and group assignments were not changed.
- No training, inference or tests were run for this update.

# Window-input configuration: 2026-11-13

- At the author's direction, aligned the retained encoder's configured input policy with the manuscript's valid-residue mean (`residue_mask_mean_v2`). The existing window extractor already computes this mean and is unchanged.
- Added the shared-aux teacher and window settings to `encoder/pretrained/metadata.json`; the retained-encoder loader now reads that explicit contract instead of requiring padded-width means.
- Updated usage examples and existing regression expectations. No tests, training or inference were run. Checkpoint bytes and existing cache contents were not changed.

# Teacher-source correction: 2026-11-13

- Shared teacher defaults are defined in `teacher/utils/checkpoints.py` and point to `teacher/pretrained/teacher_canya_cpad.pt` (`shared_aux_transfer_v1`).
- At the author's direction, replaced the retained encoder's old teacher binding with the configured shared-aux teacher and removed hash-based inference of window-probability policy.
- Updated the existing regression expectations and usage notes. Tests were not run for this change.
- Window aggregation, the independent retained-encoder policy restriction, checkpoint bytes and existing caches were unchanged by this source correction. The subsequent window-input configuration above updates the policy restriction separately.

# Code repair: 2026-10-20

- New window-output probabilities average real residues only and record `residue_mask_mean_v2`. Teacher/CNN forward padding is unchanged. Historical caches are not overwritten.
- HAPR extraction verifies teacher identity and feature policy against encoder inputs. The initial teacher binding was superseded by the 2026-11-13 source correction above. No weight was replaced or refitted.
- New encoder exports record their input contract. Full-data fitting uses the released workbook/folds and explicit external cache paths; consuming the frozen final teacher no longer requires its training initializer.
- Cache loading checks exact sequences, duplicate IDs, NPZ identity, window geometry, finite values and optional profile geometry.
- Both graph builders resolve recorded or relocated coordinate paths, validate supplied checksums and honor `--workers`.
- DSSP is configurable/discoverable. Its absence reaches the existing masked-secondary-structure/SASA fallback.
- Protein results record cohort and partition identity. Mean/SD summaries reject mixed conditions and missing folds by default; explicitly requested interim summaries record missing folds in a sidecar.

Sequence/label workbooks, partitions, pretrained weight bytes, model layers and existing server results are unchanged. This repair performs no training or production-cache regeneration.

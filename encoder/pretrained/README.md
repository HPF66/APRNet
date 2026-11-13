# Supplied retained encoder

`encoder_epoch20.pt` is the exact existing retained Boundary-Aware Context V3 export. Its identity is recorded in `metadata.json`; it is not refitted during this source reorganization.

## Configured teacher

The configured teacher is `teacher/pretrained/teacher_canya_cpad.pt`, from `peptide_binary_random10_nnk123_shared_aux_transfer_v1` (36 full-data epochs). Its SHA-256 is `7b059283e57e54dcff3e48ad1e339deb7251aedc55b977d0bd79a489aa35b327`. Entry points share the default path defined in `teacher/utils/checkpoints.py`.

The teacher-source binding was corrected at the author's direction on 2026-11-13. HAPR extraction checks the window cache against this teacher. Existing cache manifests and checkpoint bytes are not rewritten by changing the configured source.

## Window-feature policy

The configured window policy is `residue_mask_mean_v2`, matching the manuscript definition: apply sigmoid to the teacher's residue logits, sum only valid residue probabilities, and divide by the number of valid residues in that window. Padding positions do not enter the numerator or denominator. This is already the calculation used by `window_tokens.py`; its calculation is unchanged.

The retained encoder's configured `input_contract` is stored in `metadata.json` and read by `extract_hapr` through `lineage.py`. It specifies the shared-aux teacher, six stride-1 window scales, width 320 and the valid-residue mean. It was set at the author's direction on 2026-11-13 without modifying the checkpoint. Every cache must explicitly record the matching policy; changing this configuration does not convert existing cache values or relabel their sources. New full-data/CV exports record their own `input_contract`.

No tests, training, feature regeneration or inference were run for this configuration change.

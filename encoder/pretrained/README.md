# Supplied retained encoder

**Input checks are now enforced. The bundled encoder and separately distributed teacher are not a verified end-to-end pretrained pair for new sequences.**

`encoder_epoch20.pt` is the exact existing retained Boundary-Aware Context V3 export. Its identity is recorded in `metadata.json`; it is not refitted during this source reorganization.

## Original artifact distinction

The existing retained encoder's export/refit records refer to the historical `peptide_window_tokens_cpad_full_v1` cache, whose manifest records teacher SHA256 `cf5374914dee17fb143b64e6b62dff34057e58a7c9c76e05bb8ad3feff8c8957`. The canonical CAnYA→CPAD teacher supplied in this release instead has SHA256 `7b059283e57e54dcff3e48ad1e339deb7251aedc55b977d0bd79a489aa35b327`.

These are distinct original artifacts. Copying the canonical teacher into `teacher/pretrained` does not retroactively change how the historical encoder was fitted. Use the retained encoder with its compatible teacher inputs, or explicitly fit a new canonical encoder with the supplied canonical teacher. Historical weights, cached features and reported experimental results are not rewritten by this package. This note preserves the source records rather than silently asserting that the two hashes are interchangeable.

`extract_hapr` now verifies teacher SHA-256 and the feature policy before inference. The identified retained encoder accepts its historical cache inputs; it rejects the Release-teacher/masked-v2 combination. Old absolute paths are provenance and are not required installation paths.

New window extraction corrects a padded-output mean and records `local_probability_policy=residue_mask_mean_v2`. It does not change CNN input padding. This changes one input feature definition, so new caches must be used with an encoder fitted on that definition. New full-data/CV exports record their own `input_contract`.

The repair supplies neither a replacement teacher nor a refitted encoder. To reproduce retained outputs, supply the original compatible historical window caches externally. Creating a newly paired pretrained pipeline remains separate work. Checkpoint identities do not by themselves establish or refute the CAnYA→CPAD training ancestry.

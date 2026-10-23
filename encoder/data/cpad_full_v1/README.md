# Unified mixed APR database v1

All 3,031 sequences are formal samples for joint training, validation, and evaluation.

- APR-labelled long sequences: 332
- Random non-amyloid long proteins: 668
- Positive short peptides: 917
- Negative short peptides: 1,114
- Positive residues: 22,744
- Negative residues: 456,317
- Negative:positive ratio: 20.063:1
- Different source records are retained as separate samples even when their sequences are identical
- Exact-sequence duplicates are grouped into the same fold to prevent train/test leakage

The `sequences` sheet is directly compatible with the current APR loader. Every mask contains only `0` and `1`.

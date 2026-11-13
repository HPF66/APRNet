# Unified mixed APR database v1

The collection contains 3,031 source records and 3,025 distinct sequences. Fold assignments determine which records are used for training, validation and evaluation in each run. The final full-data refit uses all 3,031 records.

- APR-labelled long sequences: 332
- Screened UniProt background proteins, treated as presumed negatives: 668
- Positive short peptides: 917
- Negative short peptides: 1,114
- Positive residue labels: 22,744
- Negative residue labels: 456,317
- Negative:positive ratio: 20.063:1
- Different source records are retained as separate samples even when their sequences are identical
- Exact-sequence duplicates are assigned to the same fold, so identical sequences do not cross the encoder's training, validation and test partitions within a run

The `sequences` sheet is directly compatible with the APR loader. Each `apr_positive_mask` has one `0` or `1` per residue. Annotated long sequences use regional APR labels; positive and negative CPAD peptides receive all-one and all-zero masks from their peptide-level labels; screened background proteins receive all-zero masks as presumed negatives. These supervision sources are distinguished in the manuscript. Residue-label counts above are before chunking or source weighting.

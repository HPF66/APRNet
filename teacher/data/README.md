# Peptide teacher input data

- `aggregating_peptides_model_database.xlsx`: the unchanged CPAD-derived merged input workbook; 2,031 rows in `peptides`, plus `mapped_regions` and `protein_sequences`.
- `canya_random90_10.tsv`: the actual CAnYA NNK1/2/3 random90/10 input used by the retained recipe (111,692 source rows before the trainer skips six empty sequences).
- `canya_original_sd2.tsv`: the original extracted SD2 table before random re-partitioning. Training uses `canya_random90_10.tsv`.
- `splits/peptide_folds.csv`, `splits/fold_manifest.json`: unchanged row-order fold definitions. The manifest contains the original multi-task project metadata; the teacher runner selects the peptide task.
- `splits/aggregating_peptides/split_01.xlsx` through `split_10.xlsx`: the original ten peptide split workbooks.

Sample IDs, labels, row order and original workbook sheets are preserved. The source data and original splits are included here. No learned peptide/residue embeddings or extraction caches are included.

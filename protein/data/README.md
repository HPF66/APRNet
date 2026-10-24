# Protein classification input data

## Excel workbooks

- `sources/primary/amyloid_structure_model_database.xlsx`: the unchanged merged protein dataset, with `samples` (1,417), `chains` (2,654), `sequences` (619), `residue_codes` and `provenance` sheets.
| Cohort | Positive | Negative | Total |
| --- | ---: | ---: | ---: |
| primary | 1,218 | 199 | 1,417 |
| ratio1 (1:1) | 1,218 | 1,218 | 2,436 |
| ratio2 (1:2) | 1,218 | 2,436 | 3,654 |
| ratio3 (1:3) | 1,218 | 3,654 | 4,872 |
| ratio4 (1:4) | 1,218 | 4,872 | 6,090 |
| ratio5 (1:5) | 1,218 | 6,090 | 7,308 |

The additional Excel is a view, not a newly selected dataset. Each sample includes its binary label, structure/accession metadata where present, sequence identifiers, source-table path and original zero-based test-fold indices. Source metadata stay empty when absent; nothing is inferred from a protein name. `source_table` is relative to this `data` directory. `total_unique_sequence_length_aa` sums the listed distinct chain inputs and is not the length of a single mother protein or a count of physical assembly copies.

## Original source tables

`sources/primary/` contains `labels.tsv`, `chains.tsv`, `sequences.tsv`, `sequences.fasta`, provenance/evidence tables and audits. `sources/background_ratio1/` through `sources/background_ratio5/` contain the exact matched background releases used by their respective ratios. Original row order, IDs, label values and chain-to-sequence alignments are preserved. Multiple proteins sharing one chain are retained as distinct samples. Every cohort retains all 199 original negative samples.

## Fixed study partitions

Fixed cohort partitions are generated artifacts and are intentionally omitted
from this repository. Generate them in an external data directory and pass
that directory to the trainer. The required layout is documented in
`protein/README.md` and `protein/tools/README.md`.

## External structure and feature assets

PDB/mmCIF coordinates, structure graph caches, learned HAPR/pAPR/rAPR outputs and ESM-2 embeddings are not copied into this code/data package. Source tables preserve structure IDs, original file/path references, residue alignments, and background download URLs/checksums where present. Coordinate assets must be supplied separately to the graph builder. No protein-classifier checkpoint is bundled.

Training consumes the included original TSVs and externally generated features; the Excel view does not change any model result. Source and partition descriptions are provided above.

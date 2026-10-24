# Protein classification input data

## Included source data

- `sources/primary/amyloid_structure_model_database.xlsx`: the unchanged merged protein dataset, with `samples` (1,417), `chains` (2,654), `sequences` (619), `residue_codes` and `provenance` sheets.
- `sources/primary/`: the corresponding labels, chain table, sequence table, FASTA file, provenance and audit records.

The primary source is the only protein cohort shipped in this repository. The
ratio cohorts are external generated data because their five background tables
are redundant and add roughly 110 MB to the checkout. They remain available
outside the repository and can be passed to the same loaders through an
external `--sources` root.

## Cohort sizes

| Cohort | Positive | Negative | Total |
| --- | ---: | ---: | ---: |
| primary | 1,218 | 199 | 1,417 |
| ratio1 (1:1) | 1,218 | 1,218 | 2,436 |
| ratio2 (1:2) | 1,218 | 2,436 | 3,654 |
| ratio3 (1:3) | 1,218 | 3,654 | 4,872 |
| ratio4 (1:4) | 1,218 | 4,872 | 6,090 |
| ratio5 (1:5) | 1,218 | 6,090 | 7,308 |

The ratio rows describe external cohort releases used in the study; they are
not additional files in this checkout. Each sample includes its binary label,
structure/accession metadata where present, sequence identifiers, source-table
path and original zero-based test-fold indices. Source metadata stay empty
when absent; nothing is inferred from a protein name. `source_table` is
relative to the supplied external data root. `total_unique_sequence_length_aa`
sums the listed distinct chain inputs and is not the length of a single mother
protein or a count of physical assembly copies.

## Original source tables

`sources/primary/` contains `labels.tsv`, `chains.tsv`, `sequences.tsv`,
`sequences.fasta`, provenance/evidence tables and audits. The external ratio
releases preserve their original row order, IDs, label values and
chain-to-sequence alignments. Multiple proteins sharing one chain are retained
as distinct samples. Every cohort retains all 199 original negative samples.
Keep the external ratio directories outside the code checkout and record their
source URL, release date and checksums in the run manifest.

## Rebuild the ratio data

The repository ships the unchanged primary source tables and the scripts used
to obtain and select the large background cohorts. The generated ratio tables
are kept outside GitHub because they include thousands of repeated structure
and sequence records. Start from the English workflow in
[`../tools/README.md`](../tools/README.md): acquire a frozen RCSB candidate
catalogue, build nested `ratio1`--`ratio5`, then run the validation command.
The scripts preserve all 199 original negatives and never use model scores to
select a background record.

The public RCSB query is only an acquisition mechanism. For an exact release,
use the saved candidate TSV and manifest as an offline input; this records the
candidate snapshot instead of silently changing the cohort when the archive is
updated.

## Fixed study partitions

Fixed cohort partitions are generated artifacts and are intentionally omitted
from this repository. Generate them in an external data directory and pass
that directory to the trainer. The required layout is documented in
`protein/README.md` and `protein/tools/README.md`.

## External structure and feature assets

PDB/mmCIF coordinates, structure graph caches, learned HAPR/pAPR/rAPR outputs and ESM-2 embeddings are not copied into this code/data package. Source tables preserve structure IDs, original file/path references, residue alignments, and background download URLs/checksums where present. Coordinate assets must be supplied separately to the graph builder. No protein-classifier checkpoint is bundled.

Training consumes the included primary TSVs, or equivalent external cohort
TSVs, together with externally generated features. The Excel workbook is a
source view and does not change any model result. Source and partition
descriptions are provided above.

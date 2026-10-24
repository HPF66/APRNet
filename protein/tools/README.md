# Rebuilding protein cohorts

The repository contains the source-screened cohort construction code. Large
candidate tables, coordinates, embeddings, graphs, fixed folds and training
outputs stay outside the Git checkout. This keeps the release small while
preserving a reproducible route from public structure records to the five
background ratios.

## 1. Acquire candidates

Online mode queries the public RCSB Search/Data APIs and can download the
corresponding mmCIF files:

```bash
python protein/tools/download_background_candidates.py \
  --output /data/apr_candidates_20261024 \
  --max-entries 5000 \
  --download-structures
```

For a frozen or offline rebuild, use a checked-in query or an ID list instead:

```bash
python protein/tools/download_background_candidates.py \
  --output /data/apr_candidates_20261024 \
  --ids-file /data/rcsb_ids.txt \
  --download-structures
```

The output contains `candidates.tsv`, `manifest.json`, optional `errors.tsv`
and a `structures/` directory. The manifest records the API request, query
date, candidate hash and coordinate download status. A candidate is not a
label; it is only a structure-linked input eligible for source screening.

## 2. Build nested ratios

Point the builder at the unchanged primary tables and the candidate TSV. The
output is written outside the repository:

```bash
python protein/tools/build_ratio_cohorts.py \
  --primary protein/data/sources/primary \
  --candidates /data/apr_candidates_20261024/candidates.tsv \
  --output /data/apr_cohorts_20261024
```

This creates `ratio1` through `ratio5`. Every ratio contains all 1,218 primary
positive samples and all 199 original negative samples. It then adds exactly
`1,218 * ratio - 199` source-screened background records when the candidate
pool has enough eligible records. The added records are selected without model
scores using:

- joint length bin, human/other species group and deposited/full-parent scope;
- scaled amino-acid-composition distance to the positive/original-negative
  reference;
- one selected view per PDB/UniProt pair, with deterministic seed and order;
- a nested prefix, so `ratio1` is retained in `ratio2`, and so on.

If a public snapshot cannot satisfy a quota, the command stops and reports the
shortfall. `--allow-shortfall` is available for an explicitly incomplete audit
run; it must not be used for the final benchmark. `construction_manifest.json`
and each `ratio*/selection_audit.json` retain the protocol, counts and hashes.

The label wording is deliberately conservative:
`source-screened assumed background negative; no direct nonaggregation
experiment asserted`. These records are a matched computational background,
not experimentally confirmed non-amyloid proteins.

## 3. Validate before training

```bash
python protein/tools/validate_background_candidates.py \
  --primary protein/data/sources/primary \
  --cohorts /data/apr_cohorts_20261024
```

Validation checks the 199 original negatives, primary labels and sequence
content, ratio counts, nested membership, unique background PDB IDs and
UniProt accessions. It writes `validation.json` and exits nonzero on any
failure.

## Candidate TSV contract

`candidates.tsv` must contain at least `pdb_id`, `uniprot_id` and `sequence`.
The downloader also writes `entity_id`, `chain_ids`, `organism_id`,
`species_group`, `scope`, `structure_url`, `coordinate_url` and optional
coordinate checksum fields. User-supplied offline candidates should use the
same names. Sequence strings must be canonical one-letter amino-acid codes and
at least 50 residues long.

## Reproducibility limits

RCSB search results can change as the public archive grows. For a paper or
release, preserve the generated candidate TSV, `manifest.json`, structure
checksums, query JSON/ID list, seed and construction manifest alongside the
external data snapshot. The scripts intentionally do not promise that a fresh
online query in a later year produces the same candidate set.

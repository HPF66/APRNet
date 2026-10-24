# Protein amyloid classifier: ESM-2 650M + HAPR + PDB

This stage contains one model. ESM-2 residue inputs have width1280; HAPR has width192. ESM-2 Queries interact globally with HAPR Keys/Values, with pAPR/rAPR-conditioned gating and APR-aware chain pooling. Aligned structure messages and spatial readout preserve physical chain instances before protein-level pooling. The retained sequence and structure hyperparameters are in `utils/config.json`.

| File | Responsibility |
| --- | --- |
| `model/aprnet.py` | Final structure-aware forward computation |
| `model/sequence_context.py` | Shared residue-context and chain/protein pooling backbone; no standalone sequence classifier |
| `my_loader/data.py`, `graphs.py` | Protein records, membership/split loading, padded chains and graph collation |
| `feature/store.py` | Existing external ESM-2 650M and HAPR/pAPR/rAPR inputs |
| `utils/training.py` | Optimizer groups, retained updates and validation evaluation |
| `utils/metrics.py`, `checkpoint.py`, `model.py` | Fixed-threshold metrics, checkpoint I/O and single-model construction |
| `train/train.py`, `predict.py`, `report.py` | Fold fitting, checkpoint inference and mean±SD export |
| `scratch/get_feature/` | 650M extraction and aligned PDB/mmCIF graph preparation |

## Input contract

Membership and split TSV files require `sample_id`, `label`, `sequence_ids` (semicolon-separated). A biological sample may contain several distinct chain sequences; proteins sharing one chain are not automatically merged. The sequence ID is `SEQ-` plus the first12 uppercase SHA256 hex digits of the exact chain sequence.

```text
cohort/cohort_membership.tsv
cohort/<protocol>/outer_fold0/{train,val,test}.tsv
source/sequences.tsv              # sequence_id, sequence, sequence_length
source/embeddings/esm2/esm2_650m/manifest.json
source/embeddings/hapr/epoch20/manifest.json
graph_root/manifest.json
```

ESM tensors are `[L,1280]`. HAPR inputs contain `hapr[L,192]`, `pAPR[L]`, `rAPR[L]`. Graph manifests must have `validation=passed`; each record has `sample_id`, `file`, ordered `sequence_ids`. Final `.npz` graph fields are `node_seq`, `positions` (0-based), `instance`, `instance_to_seq`, `node`, `edge_index`, `edge`; node and edge dimensions are22. Labels and data-source categories are not graph inputs.

## Prepare external features

```bash
python -m protein.scratch.get_feature.esm2_650m --dataset /data/source \
  --output-root /data/source/embeddings/esm2 --device cuda:0 --batch-size 2
python -m protein.scratch.get_feature.build_graphs --stage backbone \
  --source /data/source --structure-root /data/original_coordinates \
  --backbone-output /cache/backbone --output /data/graphs --workers 8
python -m protein.scratch.get_feature.build_graphs --stage context \
  --source /data/source --structure-root /data/original_coordinates \
  --backbone-output /cache/backbone --output /data/graphs --workers 8
```

ESM extraction retains full chains, excludes BOS/EOS, uses float32 last-layer residue vectors and reduces the batch on OOM. HAPR extraction is described in `encoder/README.md`. Feature output paths should be outside the source repository.

Structure preparation needs existing residue-to-sequence alignments, not arbitrary unaligned PDB files. `chains.tsv` supplies `sample_id`, `chain_id`, `sequence_id`, `coordinate_residue_keys`, `coordinate_seqres_positions` (1-based), `coordinate_sequence`. `labels.tsv` supplies `structure_file`, `selected_model`, `sequence_ids`, `assembly_mode`, `target_chain_ids`. The builders retain the original missing-coordinate masks and only use contacts supported by the supplied assembly metadata. Both PDB and mmCIF inputs are supported; format conversion is unnecessary.

## Fit a supplied fold

Coordinate resolution checks repeatable `--structure-root` directories, `source/pdb`, then the recorded `structure_path`. Supplied structure checksums must match. A root may contain coordinate files directly or in its `pdb` subdirectory. Both builder stages honor `--workers`. DSSP is discovered on `PATH` or set with `--dssp`; its absence uses masked secondary structure and the SASA fallback, with diagnostics recorded in graph metadata.

```bash
python -m protein.train.train --membership /data/cohort/cohort_membership.tsv \
  --fold-dir /data/cohort/cold-cluster/outer_fold0 --fold 0 \
  --sources /data/protein_source /data/background_source \
  --graph-roots /data/protein_graphs /data/background_graphs \
  --output /runs/cohort_coldcluster_fold0 --threads 2 --device cuda
```

For primary, supply only its relevant source/graph roots. Run each fold/cohort/protocol in a distinct output directory. The program uses the supplied partitions and refuses sample-ID overlap; it does not rebuild or resample them. Train loss uses `(N_negative_train/N_positive_train)^0.35`. Checkpoint selection uses validation AUROC, with unweighted BCE as a tie-break within1e-5. The final test predictions use a fixed0.5 threshold. Seed and the separate geometry optimizer settings remain those in the retained implementation.

## Existing checkpoint inference

```bash
python -m protein.train.predict --checkpoint /runs/fold0/checkpoints/best.pt.gz \
  --membership /data/inference.tsv --sources /data/source \
  --graph-roots /data/graphs --output /results/protein_scores.tsv --device cuda
```

Checkpoints must identify `esm=650m`, `arm=epoch20` and a structure configuration. State loading is strict. The membership label column may be a dummy0 for unlabeled inference; prediction does not use it.

```bash
python -m protein.train.report --runs /runs/fold0 /runs/fold1 /runs/fold2 /runs/fold3 /runs/fold4 \
  --output /results/mean_sd.tsv
```

The example assumes a five-fold protocol. Supply all folds for **one** cohort/protocol/configuration (ten for random/cold-protein/cold-cluster). Reporting verifies membership and partition hashes, threshold, hyperparameters and completeness before writing mean/sample-SD with three decimals. Explicit interim reporting requires `--allow-partial` and records missing folds in a JSON sidecar. No classifier weight or generated feature array is bundled.

## Reproducible data construction

`protein/tools/download_background_candidates.py`,
`protein/tools/build_ratio_cohorts.py` and
`protein/tools/validate_background_candidates.py` provide the complete route
from a frozen RCSB candidate snapshot to the nested 1:1--1:5 cohorts. The
workflow and TSV schema are documented in
[`tools/README.md`](tools/README.md). Generated background tables and
coordinates remain external to this source release; the primary tables and
the construction code are the reproducible inputs committed to GitHub.

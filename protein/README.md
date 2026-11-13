# Protein amyloid classifier: ESM-2 650M + HAPR + PDB

This stage provides the ESM-2 650M + HAPR + PDB full model. ESM-2 residue inputs have width 1280; HAPR has width 192. ESM-2 queries interact globally with HAPR keys/values, with pAPR/rAPR-conditioned gating and APR-aware chain pooling. Aligned structure messages and spatial readout preserve physical chain instances before protein-level pooling. The retained sequence and structure hyperparameters are in `utils/config.json`.

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

Membership and split TSV files require `sample_id`, `label`, `sequence_ids` (semicolon-separated). A biological sample may contain several distinct chain sequences; proteins sharing one chain are not automatically merged. The sequence ID is `SEQ-` plus the first 12 uppercase SHA256 hex digits of the exact chain sequence.

```text
cohort/<protocol>/cohort_membership.tsv
cohort/<protocol>/protocol.json
cohort/<protocol>/outer_fold0/{train,val,test}.tsv
cohort/<protocol>/outer_fold1/{train,val,test}.tsv
...                              # all folds for this protocol
source/sequences.tsv              # sequence_id, sequence, sequence_length
source/labels.tsv
source/chains.tsv
source/provenance.tsv
source/embeddings/esm2/esm2_650m/manifest.json
source/embeddings/hapr/epoch20/manifest.json
graph_root/manifest.json
```

The bundled protocol directories are under `protein/data/cohorts/<cohort>/<protocol>/`. Keep `protocol.json`, the protocol-local membership file and all fold directories together; the training entry point reads them to identify the evaluation condition.

ESM tensors are `[L,1280]`. HAPR inputs contain `hapr[L,192]`, `pAPR[L]`, `rAPR[L]`. Graph manifests must have `validation=passed`; each record has `sample_id`, `file`, ordered `sequence_ids`. Final `.npz` graph fields are `node_seq`, `positions` (0-based), `instance`, `instance_to_seq`, `node`, `edge_index`, `edge`; node and edge dimensions are 22. Labels and data-source categories are not graph inputs.

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

ESM extraction retains full chains, excludes BOS/EOS and uses float32 last-layer residue vectors. `--batch-size` controls the number of chains per batch. An out-of-memory error stops extraction; for batch-related memory pressure, lower this value before rerunning. HAPR extraction is described in the [encoder README](../encoder/README.md). Feature output paths should be outside the source repository.

Structure preparation uses existing residue-to-sequence alignments. The source directory must contain `sequences.tsv`, `chains.tsv`, `labels.tsv` and `provenance.tsv`; both builder stages record hashes of these four files. `chains.tsv` supplies `sample_id`, `chain_id`, `sequence_id`, `coordinate_residue_keys`, `coordinate_seqres_positions` (1-based), `coordinate_sequence`. `labels.tsv` supplies `sample_id`, `structure_file`, `selected_model` (`1`), `sequence_ids`, `assembly_mode`, `target_chain_ids`. `provenance.tsv` preserves the source records. The builders retain the original missing-coordinate masks and only use contacts supported by the supplied assembly metadata. Both PDB and mmCIF inputs are supported; format conversion is unnecessary.

Coordinate resolution checks repeatable `--structure-root` directories, `source/pdb`, then the recorded `structure_path`. Supplied structure checksums must match. A root may contain coordinate files directly or in its `pdb` subdirectory. Both builder stages honor `--workers`. DSSP is discovered on `PATH` or set with `--dssp`; its absence uses masked secondary structure and the SASA fallback, with diagnostics recorded in graph metadata.

## Fit a supplied fold

The example uses the bundled primary cold-cluster partitions. `/data/source` contains the primary source tables and prepared sequence features, and `/data/graphs` contains their structure graphs.

```bash
python -m protein.train.train --membership protein/data/cohorts/primary/cold-cluster/cohort_membership.tsv \
  --fold-dir protein/data/cohorts/primary/cold-cluster/outer_fold0 --fold 0 \
  --sources /data/source --graph-roots /data/graphs \
  --output /runs/primary_coldcluster/fold0 --threads 2 --device cuda
```

For a ratio cohort, select its matching partitions and add the corresponding background source and graph roots to `--sources` and `--graph-roots`. Run each fold/cohort/protocol in a distinct output directory. The program uses the supplied partitions and refuses sample-ID overlap; it does not rebuild or resample them. Train loss uses `(N_negative_train/N_positive_train)^0.35`. Checkpoint selection uses validation AUROC, with unweighted BCE as a tie-break within 1e-5. Threshold-based test metrics use a fixed threshold of 0.5. Seed and the separate geometry optimizer settings remain those in the retained implementation.

## Existing checkpoint inference

```bash
python -m protein.train.predict --checkpoint /runs/primary_coldcluster/fold0/checkpoints/best.pt.gz \
  --membership /data/inference.tsv --sources /data/source \
  --graph-roots /data/graphs --output /results/protein_scores.tsv --device cuda
```

Checkpoints must identify `esm=650m`, `arm=epoch20` and a structure configuration. State loading is strict. The membership label column may be a dummy 0 for unlabeled inference; prediction does not use it.

```bash
python -m protein.train.report --runs \
  /runs/primary_coldcluster/fold0 /runs/primary_coldcluster/fold1 \
  /runs/primary_coldcluster/fold2 /runs/primary_coldcluster/fold3 \
  /runs/primary_coldcluster/fold4 /runs/primary_coldcluster/fold5 \
  /runs/primary_coldcluster/fold6 /runs/primary_coldcluster/fold7 \
  /runs/primary_coldcluster/fold8 /runs/primary_coldcluster/fold9 \
  --output /results/mean_sd.tsv
```

Run the report after completing all ten primary cold-cluster folds. Supply all folds for **one** cohort/protocol/configuration: ten for random/cold-protein/cold-cluster, or five for cold-family. Reporting verifies membership and partition hashes, threshold, hyperparameters and completeness before writing mean/sample-SD with three decimals. Explicit interim reporting requires `--allow-partial` and records missing folds in a JSON sidecar. No classifier weight or generated feature array is bundled.

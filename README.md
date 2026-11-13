# APRnet

Short-peptide aggregation knowledge, residue-level APR representations, and structure-aware protein amyloid classification.

This release has three stages. The protein stage provides the **ESM-2 650M + HAPR + PDB full model**, with global cross-attention, pAPR/rAPR-assisted gating and pooling, and aligned structure readout.

```text
APRnet/
├── teacher/                 # CAnYA pretraining → CPAD fine-tuning
│   └── pretrained/teacher_canya_cpad.pt
├── encoder/                 # Window evidence → residue-level APR representation
│   └── pretrained/encoder_epoch20.pt
├── protein/                 # ESM-2 650M + HAPR + PDB → protein probability
├── requirements.txt
└── README.md                # Installation, environment and usage
```

The stage directories organize the implementation by function:

| Directory | Responsibility |
| --- | --- |
| `model/` | Network layers and forward computation |
| `my_loader/` | Input records, chunking, batching and collation |
| `train/` | Training, checkpoint selection, prediction/evaluation entrypoints |
| `utils/` | Configuration, losses, metrics, optimizer/checkpoint helpers |
| `scratch/get_feature/` | Data preparation and feature extraction programs |
| `feature/` | Teacher descriptor definitions or protein feature loading; reserved in the encoder stage |
| `data/` | Merged input workbooks, source tables and original fixed partitions |
| `result/` | Optional output directory; actual destinations are set by each entry point's output arguments |
| `pretrained/` | Supplied weights for teacher and encoder only |

## Installation

Changes are listed in [CHANGELOG.md](CHANGELOG.md). The configured teacher is `teacher/pretrained/teacher_canya_cpad.pt` (`shared_aux_transfer_v1`). Window extraction and the encoder input configuration use `residue_mask_mean_v2`, the mean probability over valid residues described in the manuscript. See [the encoder input settings](encoder/pretrained/README.md).

Use Linux with Python 3.10/3.11 and a CUDA-compatible PyTorch installation. Then run `python -m pip install -r requirements.txt`. The original Linux file-lock and numerical conventions are retained. Run all commands from this repository root using `python -m ...`.

### Recorded reference environment

The original release records the following reference environment:

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch / CUDA build | 2.5.1+cu121 / 12.1 |
| transformers / peft | 4.57.6 / 0.15.2 |
| NumPy / SciPy | 1.26.4 / 1.17.1 |
| Biopython / scikit-learn | 1.88 / 1.9.0 |
| openpyxl | 3.1.5 |

Install a GPU-compatible PyTorch build before installing `requirements.txt`. These recorded versions are a reference for setup; this table does not report testing of the current revisions.

Install DSSP/mkdssp on `PATH`, or pass `build_graphs --dssp /path/to/mkdssp`. When DSSP is absent or fails, secondary structure is masked and exposure uses the existing Shrake–Rupley fallback. Graph metadata record the executable and errors.

Separate assets are needed: the teacher's ESM-2 8M backbone, the protein stage's ESM-2 650M backbone, and aligned original PDB/mmCIF files. Generated embeddings and graph caches remain external outputs.

Run the regression checks without training:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Entry points

- [Teacher: training and peptide inference](teacher/README.md)
- [Encoder: training and residue extraction](encoder/README.md)
- [Protein: 650M + HAPR + PDB](protein/README.md)

These are the retained implementations reorganized into packages. State-dict parameter names are preserved. Existing datasets, partitions, labels and experimental results are not modified by this release. The two supplied pretrained artifacts are copied unchanged; their identities, configured teacher source and input requirements are documented beside the files.

## Bundled study data

Each stage includes sequence/label data and fixed partitions:

- [Teacher data](teacher/data/README.md): CAnYA inputs, the 2,031-peptide merged workbook and original ten-fold files.
- [Encoder data](encoder/data/README.md): the merged APR workbook with 3,031 records and 3,025 distinct sequences, source audits and random/homology folds.
- [Protein data](protein/data/README.md): the 1,417-sample merged primary workbook, exact primary/1:1–1:5 memberships, source sequence/label/chain tables and all 210 retained protocol folds.
- [Protein cohort Excel view](protein/data/merged_protein_cohorts.xlsx): six cohort sheets plus the shared full-sequence sheet.

Original data files are copied byte-for-byte. Source tables, data descriptions and fixed partitions are included in each stage's `data` directory. Structure coordinate assets remain separate; this package supplies their identifiers/alignment tables, not coordinate downloads.

## Inputs and generated features

Generated HAPR arrays, pAPR/rAPR arrays, ESM-2 residue embeddings, structure caches and experiment outputs are **not bundled**. Store generated files in an external data/cache root and pass the paths to the entrypoints. Extraction programs are in each stage's `scratch/get_feature/` directory. `teacher/feature/` contains fixed residue descriptors and their loader, `protein/feature/` contains the feature store, and `encoder/feature/` is reserved; encoder cache loading is in `encoder/my_loader/window_data.py`.

ESM backbone weights are separate model assets, not residue embeddings: teacher inference requires compatible ESM-2 8M assets, and protein feature extraction uses ESM-2 650M. Protein classifier weights are not bundled and must be supplied through `--checkpoint` for prediction.

The root `.gitignore` excludes generated arrays and caches while allowing the two supplied teacher/encoder `.pt` files. A distribution license has not been assigned by the authors.

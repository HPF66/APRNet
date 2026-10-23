# APRnet

Short-peptide aggregation knowledge, residue-level APR representations, and structure-aware protein amyloid classification.

This release has three stages. The protein stage contains **one architecture only: ESM-2 650M + HAPR + PDB**, with the retained global Cross-Attention, pAPR/rAPR-assisted gating and pooling, and aligned structure readout. No baseline, small-ESM, local-attention, Joint Regions, search or queue implementations are bundled.

```text
APRnet/
├── teacher/                 # CAnYA pretraining → CPAD fine-tuning
│   └── pretrained/teacher_canya_cpad.pt
├── encoder/                 # Window evidence → residue-level APR representation
│   └── pretrained/encoder_epoch20.pt
├── protein/                 # ESM-2 650M + HAPR + PDB → protein probability
├── requirements.txt
├── README.md                # Installation, environment and usage
└── README_zh.md
```

Each stage uses the same PHCNet-inspired categories:

| Directory | Responsibility |
| --- | --- |
| `model/` | Network layers and forward computation |
| `my_loader/` | Input records, chunking, batching and collation |
| `train/` | Training, checkpoint selection, prediction/evaluation entrypoints |
| `utils/` | Configuration, losses, metrics, optimizer/checkpoint helpers |
| `scratch/get_feature/` | Data preparation and feature extraction programs |
| `feature/` | Feature loading code or fixed descriptor definitions |
| `data/` | Merged input workbooks, source tables and original fixed partitions |
| `result/` | Output location for explicitly requested runs |
| `pretrained/` | Supplied weights for teacher and encoder only |

## Installation

The 2026-10-20 fixes are listed in [CHANGELOG.md](CHANGELOG.md). See [the enforced teacher/encoder compatibility note](encoder/pretrained/README.md) before using the supplied weights together. The two unchanged weights are not asserted to be a matched pretrained pipeline.

Use Linux with Python 3.10/3.11 and a CUDA-compatible PyTorch installation. Then run `python -m pip install -r requirements.txt`. The original Linux file-lock and numerical conventions are retained. Run all commands from this repository root using `python -m ...`.

### Tested environment

The repair checks used the existing server environment:

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch / CUDA build | 2.5.1+cu121 / 12.1 |
| transformers / peft | 4.57.6 / 0.15.2 |
| NumPy / SciPy | 1.26.4 / 1.17.1 |
| Biopython / scikit-learn | 1.88 / 1.9.0 |
| openpyxl | 3.1.5 |

Install a GPU-compatible PyTorch build before installing `requirements.txt`. This table records the tested environment, rather than every supported dependency combination.

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

These are the retained implementations reorganized into packages. State-dict parameter names and the retained forward operations are preserved. Existing datasets, partitions, labels and experimental results are not modified by this release. The two supplied pretrained artifacts are copied unchanged; their exact identities and original artifact distinction are documented beside the files.

## Bundled study data

Each stage now includes its actual sequence/label data, not only schema notes:

- [Teacher data](teacher/data/README.md): CAnYA inputs, the 2,031-peptide merged workbook and original ten-fold files.
- [Encoder data](encoder/data/README.md): the 3,031-record merged APR workbook, source audits and random/homology folds.
- [Protein data](protein/data/README.md): the 1,417-sample merged primary workbook, exact primary/1:1–1:5 memberships, source sequence/label/chain tables and all 210 retained protocol folds.
- [Protein cohort Excel view](protein/data/merged_protein_cohorts.xlsx): six cohort sheets plus the shared full-sequence sheet.

Original data files are copied byte-for-byte. Source tables, data descriptions and fixed partitions are included in each stage's `data` directory. Structure coordinate assets remain separate; this package supplies their identifiers/alignment tables, not coordinate downloads.

## Inputs and generated features

Generated HAPR arrays, pAPR/rAPR arrays, ESM-2 residue embeddings, structure caches and experiment outputs are **not bundled**. Their extraction/loading code remains. Store generated files in an external data/cache root and pass the paths to the entrypoints. The `feature/` directories contain code and, for the teacher, a fixed residue descriptor table; these are not stored protein embeddings.

ESM backbone weights are separate model assets, not residue embeddings: teacher inference requires the original ESM-2 8M assets, and protein feature extraction uses ESM-2 650M. The 650M backbone is not bundled. Protein classifier weights are supplied explicitly through `--checkpoint`; no arbitrarily selected fold checkpoint is presented as a universal pretrained classifier.

The root `.gitignore` excludes generated arrays and caches while allowing the two supplied teacher/encoder `.pt` files. A distribution license has not been assigned by the authors. No GitHub publication is performed by this reorganization.

# APRnet

Short-peptide aggregation knowledge, residue-level APR representations, and structure-aware protein amyloid classification.

This release has three stages. The protein stage contains **one architecture only: ESM-2 650M + HAPR + PDB**, with the retained global Cross-Attention, pAPR/rAPR-assisted gating and pooling, and aligned structure readout. No baseline, small-ESM, local-attention, Joint Regions, search or queue implementations are bundled.

```text
APRnet/
├── teacher/                 # CAnYA pretraining → CPAD fine-tuning
│   └── pretrained/teacher_canya_cpad.pt   # Download from GitHub Releases
├── encoder/                 # Window evidence → residue-level APR representation
│   └── pretrained/encoder_epoch20.pt
├── protein/                 # ESM-2 650M + HAPR + PDB → protein probability
├── requirements.txt
└── README.md                # Installation, environment and usage
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
| `data/` | Source workbooks and source tables; generated partitions stay external |
| `result/` | Output location for explicitly requested runs |
| `pretrained/` | Bundled encoder checkpoint; teacher download instructions and metadata |

## Installation

The 2026-10-20 fixes are listed in [CHANGELOG.md](CHANGELOG.md). See [the enforced teacher/encoder compatibility note](encoder/pretrained/README.md) before using the checkpoints together. The two unchanged weights are not asserted to be a matched pretrained pipeline.

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

## Pretrained checkpoints

The encoder checkpoint is included at `encoder/pretrained/encoder_epoch20.pt`.
The teacher checkpoint is distributed separately as a
[v1.0.0 Release asset](https://github.com/HPF66/APRNet/releases/tag/v1.0.0).
Download [`teacher_canya_cpad.pt`](https://github.com/HPF66/APRNet/releases/download/v1.0.0/teacher_canya_cpad.pt)
and save it as `teacher/pretrained/teacher_canya_cpad.pt`.
Cloning the repository or downloading its source ZIP does not include this asset.

From the repository root on Linux:

```bash
curl --fail --location --output teacher/pretrained/teacher_canya_cpad.pt \
  https://github.com/HPF66/APRNet/releases/download/v1.0.0/teacher_canya_cpad.pt
sha256sum teacher/pretrained/teacher_canya_cpad.pt
```

Or download it from PowerShell:

```powershell
Invoke-WebRequest -Uri "https://github.com/HPF66/APRNet/releases/download/v1.0.0/teacher_canya_cpad.pt" -OutFile "teacher/pretrained/teacher_canya_cpad.pt"
(Get-FileHash "teacher/pretrained/teacher_canya_cpad.pt" -Algorithm SHA256).Hash.ToLower()
```

The expected teacher SHA-256 is:

```text
7b059283e57e54dcff3e48ad1e339deb7251aedc55b977d0bd79a489aa35b327
```

The exact file size is 36,065,562 bytes. See
[`teacher/pretrained/metadata.json`](teacher/pretrained/metadata.json) for its
identity and [`encoder/pretrained/README.md`](encoder/pretrained/README.md) for
the teacher/encoder input compatibility requirements. Downloading the teacher
does not change those requirements.

## Entry points

- [Teacher: training and peptide inference](teacher/README.md)
- [Encoder: training and residue extraction](encoder/README.md)
- [Protein: 650M + HAPR + PDB](protein/README.md)

These are the retained implementations reorganized into packages. State-dict parameter names and the retained forward operations are preserved. Existing datasets, partitions, labels and experimental results are not modified by this release. The bundled encoder checkpoint and separately distributed teacher checkpoint are unchanged; their exact identities and original artifact distinction are documented in their respective `pretrained/` directories.

## Source data and partitions

The repository keeps the source workbooks and source tables needed to describe
the tasks, but omits generated fold manifests, cohort memberships, split
workbooks and ratio-specific partition directories. This keeps the code
portable and avoids treating one frozen partition as part of the implementation.
Generate partitions into an external data root with the scripts in
`teacher/my_loader`, `encoder/tools` and `protein/tools`, then pass those paths
explicitly to the training entrypoints. Structure coordinate assets and all
generated features remain external as well.

## Inputs and generated features

Generated HAPR arrays, pAPR/rAPR arrays, ESM-2 residue embeddings, structure caches and experiment outputs are **not bundled**. Their extraction/loading code remains. Store generated files in an external data/cache root and pass the paths to the entrypoints. The `feature/` directories contain code and, for the teacher, a fixed residue descriptor table; these are not stored protein embeddings.

ESM backbone weights are separate model assets, not residue embeddings: teacher inference requires the original ESM-2 8M assets, and protein feature extraction uses ESM-2 650M. The 650M backbone is not bundled. Protein classifier weights are supplied explicitly through `--checkpoint`; no arbitrarily selected fold checkpoint is presented as a universal pretrained classifier.

The root `.gitignore` excludes generated arrays, caches and the teacher checkpoint while allowing the bundled encoder `.pt` file. Keep the downloaded teacher checkpoint locally at the documented path. A distribution license has not been assigned by the authors.

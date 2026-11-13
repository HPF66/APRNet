# Protein feature loading

`store.py` resolves external feature manifests and loads residue arrays for each chain sequence: ESM-2 650M `[L,1280]`, HAPR `[L,192]`, and pAPR/rAPR `[L]`. The source sequence tables associate sequence IDs with the corresponding feature files.

ESM-2 extraction is in [`../scratch/get_feature/esm2_650m.py`](../scratch/get_feature/esm2_650m.py); HAPR extraction is described in the [encoder README](../../encoder/README.md). Generated embeddings and graph caches are kept outside this directory.

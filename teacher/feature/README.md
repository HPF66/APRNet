# Fixed residue descriptors

`residue_descriptors.json` stores the fixed amino-acid descriptor lookup: AAindex PCA15, VHSE8 and Z-scale5, totaling 28 values per residue. `residue_features.py` loads the selected descriptor sets for the peptide teacher. These descriptors are model inputs; generated sequence embeddings are stored separately.

Peptide inference and data-preparation entry points are in [`../scratch/get_feature/`](../scratch/get_feature/). See the [teacher README](../README.md) for usage.

"""Construct the single released architecture: ESM-2 650M + HAPR + PDB."""
from protein.model.aprnet import PDBResidueClassifier

def make_model(config, structure, store=None, arm='epoch20'):
    if arm != 'epoch20' or (store is not None and store.esm_dim != 1280):
        raise ValueError('Only ESM-2 650M + HAPR + PDB is included')
    return PDBResidueClassifier(config['hidden'], config['heads'], config['context_layers'],
                                config['dropout'], structure)

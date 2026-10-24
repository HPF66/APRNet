from pathlib import Path
from protein.utils.checkpoint import NODE_DIM, EDGE_DIM, atomic, sha
SOURCE = Path(__file__).resolve().parents[1] / 'data'
GRAPH = SOURCE / 'structure_features/residue_context'

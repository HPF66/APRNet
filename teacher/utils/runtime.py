"""Build complete teacher checkpoints without following stale initialization paths."""
import copy
from pathlib import Path
from teacher.train.cross_validate import build_model as original_build_model
from teacher.utils.compatibility import filter_unused_position

STAGE = Path(__file__).resolve().parents[1]

def build_inference_model(args, device, backbone=None):
    args = copy.deepcopy(args)
    if backbone is not None:
        args.sequence_model_path = Path(backbone)
    descriptors = getattr(args, 'residue_feature_table', None)
    if descriptors is None or not Path(descriptors).is_file():
        args.residue_feature_table = STAGE / 'feature/residue_descriptors.json'
    args.peptide_transfer_checkpoint = None
    model = original_build_model(args, device)
    restore = model.load_state_dict
    def compatible_restore(state, *positional, **keywords):
        return restore(filter_unused_position(model, state), *positional, **keywords)
    model.load_state_dict = compatible_restore
    return model

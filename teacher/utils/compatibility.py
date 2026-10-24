"""One-key compatibility adapter for legacy rotary ESM-2 checkpoints.

The original window-cache loader already tolerates this unused absolute-position
key. Older transfer/prediction entrypoints do not. All other original checks stay
active; originals and source artifacts remain untouched on disk.
"""
from collections import OrderedDict
from pathlib import Path
import tempfile

UNUSED_ROTARY_POSITION = (
    'sequence.encoder.backbone.backbone.base_model.model.embeddings.position_embeddings.weight'
)


def filter_unused_position(model, state):
    if UNUSED_ROTARY_POSITION not in state or UNUSED_ROTARY_POSITION in model.state_dict():
        return state
    # Restrict the compatibility exception to a rotary ESM-2 backbone.
    configs = [getattr(module, 'config', None) for module in model.modules()]
    if not any(getattr(c, 'position_embedding_type', None) == 'rotary' for c in configs):
        return state
    filtered = OrderedDict((name, value) for name, value in state.items() if name != UNUSED_ROTARY_POSITION)
    if hasattr(state, '_metadata'):
        filtered._metadata = state._metadata
    return filtered


def install_transfer_compat(cross_validate):
    import torch
    original = cross_validate.load_peptide_transfer_checkpoint

    def load(model, checkpoint_path, *args, **kwargs):
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or 'model' not in payload:
            return original(model, checkpoint_path, *args, **kwargs)
        state = filter_unused_position(model, payload['model'])
        if state is payload['model']:
            return original(model, checkpoint_path, *args, **kwargs)
        payload['model'] = state
        with tempfile.TemporaryDirectory(prefix='aprnet_rotary_compat_') as directory:
            temporary = Path(directory) / 'transfer.pt'
            torch.save(payload, temporary)
            report = original(model, temporary, *args, **kwargs)
            report['checkpoint'] = str(checkpoint_path)
            report['compatibility_ignored_unused_rotary_position'] = UNUSED_ROTARY_POSITION
            return report

    cross_validate.load_peptide_transfer_checkpoint = load

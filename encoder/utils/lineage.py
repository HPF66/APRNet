"""Content-based teacher/cache contracts; paths are relocatable provenance only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

CANONICAL_PEPTIDE_CHECKPOINT = Path(__file__).resolve().parents[2] / 'teacher/pretrained/teacher_canya_cpad.pt'
CANONICAL_PEPTIDE_SHA256 = '7b059283e57e54dcff3e48ad1e339deb7251aedc55b977d0bd79a489aa35b327'
CANONICAL_WINDOW_CACHE = 'peptide_window_tokens_canya_cpad_v2'
CANONICAL_ZERO_CACHE = 'zero_esm_cpad_full_v1'
MASKED_LOCAL_MEAN = 'residue_mask_mean_v2'
LEGACY_LOCAL_MEAN = 'padded_width_mean_v1'
RETAINED_ENCODER_SHA256 = '77ed79558fec7f4876d7328193b4c8a84e43346a07b3ebd0ebe730b4e8e50d06'
RETAINED_TEACHER_SHA256 = 'cf5374914dee17fb143b64e6b62dff34057e58a7c9c76e05bb8ad3feff8c8957'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_canonical_teacher(checkpoint: Path | None = None) -> None:
    """The frozen final teacher suffices; its training initializer is not an input."""
    path = checkpoint or CANONICAL_PEPTIDE_CHECKPOINT
    actual = sha256(path)
    if actual != CANONICAL_PEPTIDE_SHA256:
        raise ValueError(f'Teacher checkpoint hash mismatch: {actual}')


def window_contract(manifest: dict) -> dict:
    teacher_hash = manifest.get('peptide_checkpoint_sha256')
    if not isinstance(teacher_hash, str) or len(teacher_hash) != 64:
        raise ValueError('Window manifest lacks a teacher SHA-256 identity')
    policy = manifest.get('local_probability_policy')
    # Only the recorded retained artifact may use this pre-versioning schema.
    if policy is None and teacher_hash == RETAINED_TEACHER_SHA256:
        policy = LEGACY_LOCAL_MEAN
    if policy not in (MASKED_LOCAL_MEAN, LEGACY_LOCAL_MEAN):
        raise ValueError('Missing/unsupported local_probability_policy; regenerate a versioned cache')
    if manifest.get('token_width') != 320:
        raise ValueError('Window token_width must be 320')
    if manifest.get('window_lengths') != [6, 8, 10, 12, 16, 20] or manifest.get('stride') != 1:
        raise ValueError('Encoder requires six stride-1 scales: 6,8,10,12,16,20')
    return dict(peptide_checkpoint_sha256=teacher_hash, local_probability_policy=policy,
                window_lengths=manifest['window_lengths'], stride=1, token_width=320,
                batch_size=manifest.get('batch_size') if policy == LEGACY_LOCAL_MEAN else None)


def verify_window_manifest(cache_dir: Path, checkpoint: Path | None = None) -> dict:
    """Verify caches for a new canonical fit without requiring training ancestors."""
    require_canonical_teacher(checkpoint)
    manifest = json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))
    contract = window_contract(manifest)
    if contract['peptide_checkpoint_sha256'] != CANONICAL_PEPTIDE_SHA256:
        raise ValueError('Cache teacher identity differs from the supplied final teacher')
    if contract['local_probability_policy'] != MASKED_LOCAL_MEAN:
        raise ValueError('New fits require residue-masked window means (v2)')
    return manifest


def verify_encoder_inputs(checkpoint: Path, payload: dict, cache_dir: Path, esm_dir: Path) -> dict:
    """Reject silently interchanging teachers or extraction policies at inference."""
    actual = window_contract(json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8')))
    expected = payload.get('input_contract')
    if sha256(checkpoint) == RETAINED_ENCODER_SHA256:
        if actual['peptide_checkpoint_sha256'] != RETAINED_TEACHER_SHA256 or actual['local_probability_policy'] != LEGACY_LOCAL_MEAN:
            raise ValueError('Retained encoder requires its recorded teacher cf537491 and historical padded-width window means. '
                             'The bundled teacher/masked v2 cache is not interchangeable; see encoder/pretrained/README.md.')
        expected = actual
    elif expected is None:
        raise ValueError('Encoder has no input_contract and is not the identified retained artifact; cannot verify its inputs')
    if actual != expected:
        raise ValueError(f'Encoder/cache input contract mismatch: expected={expected}, actual={actual}')
    base = json.loads((esm_dir / 'manifest.json').read_text(encoding='utf-8'))
    if base.get('encoder_signal') != 'none' or base.get('hidden_width') != 480:
        raise ValueError('This encoder requires the explicit zero-base cache (encoder_signal=none, width=480)')
    return actual

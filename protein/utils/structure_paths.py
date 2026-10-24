"""Resolve original or relocated coordinates without changing source tables."""
from pathlib import Path
from protein.utils.checkpoint import sha


def resolve_structure(sample: dict, source: Path, roots=()) -> Path:
    filename = sample.get('structure_file', '')
    if not filename or Path(filename).name != filename or chr(92) in filename:
        raise ValueError(f'Invalid structure_file for {sample.get("sample_id")}: {filename}')
    expected = sample.get('structure_sha256') or sample.get('structure_file_sha256')
    candidates = []
    # Explicit roots take precedence over absolute paths from the source machine.
    for root in roots:
        candidates.extend((Path(root) / filename, Path(root) / 'pdb' / filename))
    recorded = sample.get('structure_path', '')
    candidates.append(Path(source) / 'pdb' / filename)
    if recorded:
        path = Path(recorded)
        candidates.append(path if path.is_absolute() else Path(source) / path)
    seen = set()
    for candidate in candidates:
        key = str(candidate.absolute())
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            if expected and sha(candidate) != expected:
                raise ValueError(f'Structure checksum mismatch for {sample.get("sample_id")}: {candidate}')
            return candidate.resolve()
    raise FileNotFoundError(f'Coordinates missing for {sample.get("sample_id")}; tried {list(seen)}. '
                            'Pass --structure-root with the directory containing the original PDB/mmCIF files.')

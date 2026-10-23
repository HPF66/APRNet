"""Path-independent identity of one cohort and fixed partition plan."""
import hashlib
import json
from pathlib import Path


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_condition(membership: Path, fold_dir: Path, fold: int) -> dict:
    protocol_dir = fold_dir.resolve().parent
    protocol_path = protocol_dir / 'protocol.json'
    if not protocol_path.is_file():
        raise ValueError(f'Missing fixed partition protocol: {protocol_path}')
    plan = json.loads(protocol_path.read_text(encoding='utf-8'))
    expected = list(range(int(plan['outer_folds'])))
    if sorted(item['fold'] for item in plan['folds']) != expected:
        raise ValueError('Protocol fold list is inconsistent with outer_folds')
    if fold not in expected or fold_dir.name != f'outer_fold{fold}':
        raise ValueError('The --fold index must match its outer_fold directory and protocol')
    split_hashes = {
        f'outer_fold{i}/{role}.tsv': file_sha(protocol_dir / f'outer_fold{i}/{role}.tsv')
        for i in expected for role in ('train', 'val', 'test')
    }
    local_membership = protocol_dir / 'cohort_membership.tsv'
    membership_hash = file_sha(membership)
    if file_sha(local_membership) != membership_hash:
        raise ValueError('Membership differs from the fixed protocol membership')
    return dict(format='aprnet_evaluation_condition_v1', cohort=plan['cohort'],
                protocol=plan['protocol'], membership_sha256=membership_hash,
                protocol_sha256=file_sha(protocol_path), expected_folds=expected,
                split_sha256=split_hashes)


def result_condition(folder: Path, result: dict) -> dict:
    """Read embedded provenance or reconstruct verified older packaged runs."""
    config_path = folder / 'run_config.json'
    run = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    condition = result.get('evaluation_condition') or run.get('evaluation_condition')
    if condition is None:
        if not all(key in run for key in ('membership', 'fold_dir', 'input_sha256')):
            raise ValueError(f'{folder}: no verifiable cohort/partition provenance; cannot summarize')
        membership = Path(run['membership'])
        recorded_hashes = run['input_sha256']
        expected_hash = recorded_hashes.get(str(membership.resolve()))
        if expected_hash is None or file_sha(membership) != expected_hash:
            raise ValueError(f'{folder}: missing/changed recorded membership identity')
        condition = evaluation_condition(membership, Path(run['fold_dir']), result['fold'])
        for role in ('train', 'val', 'test'):
            path = Path(run['fold_dir']) / f'{role}.tsv'
            if recorded_hashes.get(str(path.resolve())) != file_sha(path):
                raise ValueError(f'{folder}: missing/changed recorded {role} split identity')
    for saved in (result.get('evaluation_condition'), run.get('evaluation_condition')):
        if saved is not None and saved != condition:
            raise ValueError(f'{folder}: run/result evaluation conditions disagree')
    required = {'cohort', 'protocol', 'membership_sha256', 'protocol_sha256', 'expected_folds', 'split_sha256'}
    if not required.issubset(condition) or condition.get('format') != 'aprnet_evaluation_condition_v1':
        raise ValueError(f'{folder}: incomplete evaluation condition')
    expected = condition['expected_folds']
    if not expected or sorted(set(expected)) != list(range(len(expected))):
        raise ValueError(f'{folder}: invalid expected folds')
    if set(condition['split_sha256']) != {f'outer_fold{i}/{role}.tsv' for i in expected for role in ('train', 'val', 'test')}:
        raise ValueError(f'{folder}: incomplete partition hashes')
    if result['fold'] not in expected:
        raise ValueError(f'{folder}: fold not in protocol')
    return condition

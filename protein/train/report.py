"""Summarize test-fold results for one cohort and one partition protocol."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from protein.utils.evaluation import result_condition

METRICS = ('roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'brier')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='Completed fold output directories for a single condition')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true', help='Explicitly permit incomplete fold summaries; completeness is recorded in a sidecar')
    cli = p.parse_args()
    if len(cli.runs) < 2:
        p.error('At least two folds are required for sample SD')
    records = [json.loads((folder / 'result.json').read_text()) for folder in cli.runs]
    conditions = [result_condition(folder, record) for folder, record in zip(cli.runs, records)]
    if any(condition != conditions[0] for condition in conditions):
        raise ValueError('Different cohorts, partition plans or memberships; summarize each condition separately')
    if len({r['fold'] for r in records}) != len(records):
        raise ValueError('Duplicate fold indices; do not mix different evaluation conditions')
    if any(r['esm'] != '650m' or r['arm'] != 'epoch20' for r in records):
        raise ValueError('Only the released ESM-2 650M + HAPR + PDB model is supported')
    if any((r['config'], r['structure']) != (records[0]['config'], records[0]['structure']) for r in records):
        raise ValueError('Fold hyperparameters differ')
    if any(r.get('threshold') != .5 for r in records):
        raise ValueError('This release reports a fixed 0.5 decision threshold')
    actual = sorted(r['fold'] for r in records)
    expected = conditions[0]['expected_folds']
    if actual != expected and not cli.allow_partial:
        raise ValueError(f'Incomplete fold set: got {actual}, expected {expected}; use --allow-partial only for a labeled interim summary')
    for record in records:
        if any(not math.isfinite(record['test'][metric]) for metric in METRICS):
            raise ValueError('Nonfinite test metric; cannot produce mean/SD')
    sidecar = cli.output.with_suffix(cli.output.suffix + '.json')
    if cli.output.exists() or sidecar.exists():
        raise FileExistsError(cli.output)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['metric', 'folds', 'mean', 'sample_sd', 'mean_sd_3dp'])
        for metric in METRICS:
            values = [r['test'][metric] for r in records]
            avg, sd = mean(values), stdev(values)
            writer.writerow([metric, len(values), avg, sd, f'{avg:.3f} ± {sd:.3f}'])
    sidecar.write_text(json.dumps(dict(evaluation_condition=conditions[0], folds=actual,
        complete=actual == expected, missing_folds=sorted(set(expected) - set(actual)),
        threshold=.5, aggregation='unweighted fold mean and sample SD',
        runs=[str(folder) for folder in cli.runs]), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

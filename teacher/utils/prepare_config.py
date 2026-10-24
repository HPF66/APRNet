"""Relocate an archived teacher configuration; preserve all numerical settings."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, default=ROOT / 'utils/cpad_config.json')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--canya-checkpoint', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    value = json.loads(args.template.read_text(encoding='utf-8'))
    value.update(data_root=str(args.data_root.resolve()),
                 sequence_model_path=str(args.backbone.resolve()),
                 peptide_transfer_checkpoint=str(args.canya_checkpoint.resolve()),
                 residue_feature_table=str(ROOT / 'feature/residue_descriptors.json'),
                 output_root=str(args.output_root.resolve()))
    if args.split_manifest is not None:
        value['split_manifest'] = str(args.split_manifest.resolve())
    elif value.get('split_manifest'):
        p.error('Template has a split manifest; supply --split-manifest explicitly.')
    # Refuse to leave another old machine-specific path active in a portable config.
    # Path keys are repeated here deliberately: preparation needs no PyTorch import.
    relocated = {'data_root', 'sequence_model_path', 'peptide_transfer_checkpoint',
                 'residue_feature_table', 'output_root', 'split_manifest'}
    for name in ('embedding_cache_dir', 'apr_weak_negatives', 'peptide_auxiliary_workbook',
                 'apr_positive_contrastive_workbook', 'structure_peptide_evidence'):
        if value.get(name) and name not in relocated:
            p.error(f'Template has additional active path {name}; relocate it explicitly in a copy first.')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()

"""Regression checks for extraction, input identity and fold aggregation; no training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from openpyxl import Workbook

from encoder.my_loader.window_data import Protein, load_store
from encoder.utils.lineage import (MASKED_LOCAL_MEAN, LEGACY_LOCAL_MEAN,
    CANONICAL_PEPTIDE_SHA256, RETAINED_TEACHER_SHA256, window_contract,
    verify_window_manifest, verify_encoder_inputs)
from protein.utils.evaluation import evaluation_condition
from protein.utils.structure_paths import resolve_structure
from protein.train.report import METRICS

ROOT = Path(__file__).resolve().parents[1]


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ReleaseRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, sequence='AAAAAA'):
        self.windows, self.base = self.root / 'windows', self.root / 'base'
        record = dict(sample_id='SAME_ID', sequence=sequence, length=len(sequence), cache_file='x.npz')
        for root in (self.windows, self.base):
            (root / 'samples').mkdir(parents=True, exist_ok=True)
            put_json(root / 'manifest.json', {'records': [record]})
        self.arrays = dict(sample_id=np.asarray('SAME_ID'), sequence=np.asarray(sequence),
            window_start=np.array([0]), window_length=np.array([6]), window_scale_id=np.array([0]),
            window_embedding=np.zeros((1, 320)), window_probability=np.array([.2]),
            window_local_probability=np.array([.2]))
        np.savez(self.windows / 'samples/x.npz', **self.arrays)
        np.savez(self.base / 'samples/x.npz', hidden=np.zeros((len(sequence), 480)))
        self.protein = Protein('SAME_ID', sequence, np.zeros(len(sequence), dtype=np.int8), 0)

    def load(self):
        return load_store([self.protein], self.windows, self.base)

    def test_valid_cache_and_same_length_sequence_mismatch(self):
        self.cache()
        self.assertEqual(self.load()['SAME_ID']['tokens'].shape, (1, 320))
        self.protein = Protein('SAME_ID', 'VVVVVV', np.zeros(6, dtype=np.int8), 0)
        with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
            self.load()

    def test_npz_sequence_and_duplicate_manifest_ids(self):
        self.cache()
        self.arrays['sequence'] = np.asarray('VVVVVV')
        np.savez(self.windows / 'samples/x.npz', **self.arrays)
        with self.assertRaisesRegex(ValueError, 'NPZ sequence mismatch'):
            self.load()
        manifest = json.loads((self.windows / 'manifest.json').read_text())
        manifest['records'] *= 2
        put_json(self.windows / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate cache manifest ID'):
            self.load()

    def test_invalid_window_geometry_values_and_profiles(self):
        self.cache()
        variants = [
            {'window_start': np.array([-1])},
            {'window_length': np.array([7])},
            {'window_start': np.array([.25])},
            {'window_scale_id': np.array([8])},
            {'window_probability': np.array([1.1])},
            {'window_embedding': np.full((1, 320), np.nan)},
            {'profile_offsets': np.array([0, 5]), 'profile_hidden': np.zeros((6, 320)),
             'profile_local_probability': np.ones(6) * .2},
        ]
        for change in variants:
            with self.subTest(keys=list(change)):
                np.savez(self.windows / 'samples/x.npz', **{**self.arrays, **change})
                with self.assertRaises(ValueError):
                    self.load()
        np.savez(self.windows / 'samples/x.npz', **self.arrays)
        np.savez(self.base / 'samples/x.npz', hidden=np.full((6, 480), np.inf))
        with self.assertRaisesRegex(ValueError, 'Invalid ESM'):
            self.load()

    def test_window_output_uses_real_residues(self):
        from encoder.scratch.get_feature import window_tokens as engine
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = 'sequences'
        sheet.append(['sequence_id', 'sequence'])
        sheet.append(['short', 'AAAAAA'])
        path = self.root / 'input.xlsx'
        workbook.save(path)

        class FakeTeacher:
            def forward_sequence(self, sequences, **kwargs):
                count = len(sequences)
                mask = torch.zeros(count, 20, dtype=torch.bool)
                mask[:, :6] = True
                local = torch.zeros(count, 20)
                local[:, :6] = torch.logit(torch.tensor(.2))
                return dict(peptide_hidden=torch.ones(count, 20, 320),
                    peptide_logits=torch.zeros(count), aggregation_propensity_logits=local,
                    residue_mask=mask)

        checkpoint = self.root / 'fake.pt'
        checkpoint.write_bytes(b'test')
        args = argparse.Namespace(backbone=self.root, peptide_checkpoint=checkpoint,
            apr_workbook=path, output_dir=self.root / 'output', device='cpu',
            window_lengths='6,8,10,12,16,20', stride=1, batch_size=128)
        with patch.object(engine, 'parse_args', return_value=args), patch.object(engine, 'load_peptide_teacher', return_value=FakeTeacher()):
            engine.main()
        manifest = json.loads((args.output_dir / 'manifest.json').read_text())
        self.assertEqual(manifest['local_probability_policy'], MASKED_LOCAL_MEAN)
        with np.load(args.output_dir / 'samples' / manifest['records'][0]['cache_file']) as cached:
            np.testing.assert_allclose(cached['window_local_probability'], .2, atol=1e-6)

    def contract(self, teacher=CANONICAL_PEPTIDE_SHA256, policy=MASKED_LOCAL_MEAN):
        return dict(peptide_checkpoint='/old/machine/teacher.pt', peptide_checkpoint_sha256=teacher,
                    window_lengths=[6, 8, 10, 12, 16, 20], stride=1, token_width=320,
                    local_probability_policy=policy)

    def test_teacher_relocation_without_training_ancestor(self):
        put_json(self.root / 'manifest.json', self.contract())
        # Identity validation is mocked separately; the old absolute path is not an input.
        with patch('encoder.utils.lineage.require_canonical_teacher') as check:
            manifest = verify_window_manifest(self.root, self.root / 'relocated.pt')
        check.assert_called_once_with(self.root / 'relocated.pt')
        self.assertEqual(window_contract(manifest)['local_probability_policy'], MASKED_LOCAL_MEAN)

    def test_encoder_rejects_wrong_teacher_or_policy(self):
        put_json(self.root / 'windows/manifest.json', self.contract())
        put_json(self.root / 'base/manifest.json', {'encoder_signal': 'none', 'hidden_width': 480})
        checkpoint = ROOT / 'encoder/pretrained/encoder_epoch20.pt'
        with self.assertRaisesRegex(ValueError, 'Retained encoder requires'):
            verify_encoder_inputs(checkpoint, {}, self.root / 'windows', self.root / 'base')
        manifest = self.contract(RETAINED_TEACHER_SHA256, LEGACY_LOCAL_MEAN)
        del manifest['local_probability_policy']
        put_json(self.root / 'windows/manifest.json', manifest)
        result = verify_encoder_inputs(checkpoint, {}, self.root / 'windows', self.root / 'base')
        self.assertEqual(result['peptide_checkpoint_sha256'], RETAINED_TEACHER_SHA256)
        # New models have a self-contained input contract.
        checkpoint = self.root / 'new.pt'
        checkpoint.write_bytes(b'new encoder')
        put_json(self.root / 'windows/manifest.json', self.contract())
        verify_encoder_inputs(checkpoint, {'input_contract': window_contract(self.contract())},
                              self.root / 'windows', self.root / 'base')

    def test_coordinates_recorded_and_relocated_and_checksum(self):
        source, relocated = self.root / 'source', self.root / 'relocated'
        relocated.mkdir()
        path = relocated / '1ABC.cif'
        path.write_bytes(b'coordinates')
        sample = dict(sample_id='BGNC_1ABC', structure_file=path.name, structure_path=str(path),
                      structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(resolve_structure(sample, source), path)
        sample['structure_path'] = '/missing/original/1ABC.cif'
        self.assertEqual(resolve_structure(sample, source, [relocated]), path)
        sample['structure_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            resolve_structure(sample, source, [relocated])

    def test_dssp_absence_reaches_sasa_fallback(self):
        from Bio.PDB import PDBParser
        from io import StringIO
        from protein.scratch.get_feature import context_graphs as engine
        pdb = ('ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  \n'
               'ATOM      2  CA  ALA A   1       1.400   0.000   0.000  1.00 20.00           C  \n'
               'ATOM      3  C   ALA A   1       2.000   1.300   0.000  1.00 20.00           C  \nEND\n')
        model = next(PDBParser(QUIET=True).get_structure('x', StringIO(pdb)).get_models())
        with patch.object(engine, 'GRAPH', self.root), patch.object(engine, 'dssp_program', return_value=None):
            values, dssp_error, sasa_error = engine.annotation(model, self.root / 'missing.pdb')
        self.assertIn('FileNotFoundError', dssp_error)
        self.assertIsNone(sasa_error)
        self.assertIsNone(values[('A', '1')][0])
        self.assertTrue(values[('A', '1')][2])

    def report_run(self, folders, output, *extra):
        return subprocess.run([sys.executable, '-m', 'protein.train.report', '--runs',
            *map(str, folders), '--output', str(output), *extra], cwd=ROOT, capture_output=True, text=True)

    def test_reports_require_same_condition_and_complete_folds(self):
        protocol = ROOT / 'protein/data/cohorts/primary/cold-family'
        if not protocol.exists():
            self.skipTest('external protein partitions are not bundled; provide them to run this integration check')
        condition = evaluation_condition(protocol / 'cohort_membership.tsv', protocol / 'outer_fold0', 0)
        folders = []
        for fold in condition['expected_folds']:
            folder = self.root / f'fold{fold}'
            result = dict(esm='650m', arm='epoch20', fold=fold, threshold=.5,
                config={'a': 1}, structure={'b': 2}, evaluation_condition=condition,
                test={metric: .8 + .01 * fold for metric in METRICS})
            put_json(folder / 'result.json', result)
            folders.append(folder)
        output = self.root / 'summary.tsv'
        process = self.report_run(folders, output)
        self.assertEqual(process.returncode, 0, process.stderr)
        sidecar = json.loads(output.with_suffix('.tsv.json').read_text())
        self.assertTrue(sidecar['complete'])
        with output.open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(rows[0]['mean_sd_3dp'], '0.820 ± 0.016')
        process = self.report_run(folders[:2], self.root / 'incomplete.tsv')
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Incomplete fold set', process.stderr)
        process = self.report_run(folders[:2], self.root / 'partial.tsv', '--allow-partial')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(json.loads((self.root / 'partial.tsv.json').read_text())['complete'])
        result_path = folders[1] / 'result.json'
        result = json.loads(result_path.read_text())
        result['evaluation_condition']['cohort'] = 'ratio5'
        put_json(result_path, result)
        process = self.report_run(folders, self.root / 'mixed.tsv')
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Different cohorts', process.stderr)
        self.assertFalse((self.root / 'mixed.tsv').exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

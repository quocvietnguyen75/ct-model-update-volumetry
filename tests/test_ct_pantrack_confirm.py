"""Synthetic checks for the fixed external evaluation; no clinical inputs."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from careport.ct_update_gate import GateError
from careport.ct_pantrack_confirm import (
    MODELS, candidate_records, inspect_instances, summarize_external,
    validate_confirmation_policy, validate_prediction_coverage,
)


def pairs():
    return [dict(patient='PanTrack_001', baseline='PanTrack_001_20200101',
                 followup='PanTrack_001_20200201', label=51,
                 baseline_point=[2, 3, 4], followup_point=[3, 4, 5]),
            dict(patient='PanTrack_002', baseline='PanTrack_002_20200101',
                 followup='PanTrack_002_20200201', label=51,
                 baseline_point=[2, 3, 4], followup_point=[3, 4, 5])]


def prediction_fixture():
    expected = candidate_records(pairs())
    for r in expected:
        r['reference_volume_ml'] = 10.0
    records = []
    for r in expected:
        records.append({**r, 'predictions': {m: dict(volume_ml=10.0, dice=.8,
                        hd95_mm=1.0, empty_prediction=False) for m in MODELS}})
    maps = {name: {'slope': 1.0, 'intercept_ml': 0.0, 'degenerate': False}
            for name in ('old_to_new', 'new_to_old', 'old_to_continued', 'continued_to_old')}
    return expected, records, maps


class ExternalConfirmationTests(unittest.TestCase):
    def test_repeated_visits_get_distinct_pair_keys(self):
        source = pairs()
        second = dict(source[0], baseline=source[0]['followup'], followup='PanTrack_001_20200301')
        records = candidate_records(source + [second])
        self.assertEqual(len(records), 6)
        self.assertEqual(len({(r['pair_id'], r['timepoint']) for r in records}), 6)
        self.assertEqual(len({r['patient'] for r in records}), 2)

    def test_duplicate_cross_patient_or_reversed_pairs_fail(self):
        for change in [dict(followup='PanTrack_002_20200201'),
                       dict(followup='PanTrack_001_20191201'), dict(label=0),
                       dict(baseline_point=[float('nan'), 2, 3])]:
            with self.subTest(change=change), self.assertRaises(GateError):
                candidate_records([dict(pairs()[0], **change)])
        with self.assertRaises(GateError):
            candidate_records(pairs() + [pairs()[0]])

    def test_full_instance_volume_and_xyz_centroid(self):
        labels = np.zeros((8, 9, 10), dtype=np.uint16)
        labels[1:4, 2:5, 3:6] = 51
        affine = np.diag([2., 3., 4., 1.])
        entries = [dict(lesion_id=51, centroid=[2., 3., 4.])]
        records = inspect_instances(labels, affine, entries, 1.05)
        self.assertEqual(records[0]['reference_voxels'], 27)
        self.assertAlmostEqual(records[0]['reference_volume_ml'], .648)
        self.assertEqual(records[0]['native_lesion_box'], [[1, 4], [2, 5], [3, 6]])
        with self.assertRaises(GateError):
            inspect_instances(labels, affine, [dict(lesion_id=51, centroid=[4., 3., 2.])], 1.05)

    def test_missing_float_negative_labels_and_shear_fail(self):
        labels = np.zeros((8, 9, 10), dtype=np.int16)
        entry = [dict(lesion_id=51, centroid=[2., 3., 4.])]
        with self.assertRaises(GateError):
            inspect_instances(labels, np.eye(4), entry, 1.05)
        labels[1:4, 2:5, 3:6] = 51
        with self.assertRaises(GateError):
            inspect_instances(labels.astype(float), np.eye(4), entry, 1.05)
        labels[0, 0, 0] = -1
        with self.assertRaises(GateError):
            inspect_instances(labels, np.eye(4), entry, 1.05)
        labels[0, 0, 0] = 0
        affine = np.eye(4); affine[0, 1] = .5
        with self.assertRaises(GateError):
            inspect_instances(labels, affine, entry, 1.05)

    def test_exact_prediction_coverage_and_reference_identity(self):
        expected, records, _ = prediction_fixture()
        validate_prediction_coverage(records, expected)
        for changed in [records[:-1], records + [records[0]]]:
            with self.assertRaises(GateError):
                validate_prediction_coverage(changed, expected)
        for field, value in [('reference_volume_ml', 11.), ('scan', 'PanTrack_002_20200401')]:
            changed = copy.deepcopy(records); changed[0][field] = value
            with self.assertRaises(GateError):
                validate_prediction_coverage(changed, expected)
        for metric, value in [('dice', float('nan')), ('volume_ml', -1.), ('dice', 2.)]:
            changed = copy.deepcopy(records); changed[0]['predictions']['new'][metric] = value
            with self.assertRaises(GateError):
                validate_prediction_coverage(changed, expected)

    def test_seed_losses_are_averaged_after_absolute_error(self):
        expected, records, maps = prediction_fixture()
        seeds = [copy.deepcopy(records) for _ in range(3)]
        for rows, error in zip(seeds, [2., -2., 0.]):
            for row in rows:
                if row['timepoint'] == 'fu':
                    row['predictions']['new']['volume_ml'] += error
        result = summarize_external(seeds, expected, [maps] * 3, 400, 19, 6, 5.)
        self.assertAlmostEqual(result['seed_averaged']['strategies']['old_new']['mae_ml'], 4/3)
        self.assertEqual(result['patients'], 2)
        self.assertEqual(result['pairs'], 2)
        self.assertEqual(result['seeds'], 3)

    def test_equal_patient_weighting_not_equal_lesion_weighting(self):
        source = pairs()
        source += [dict(source[0], label=61), dict(source[0], label=71)]
        expected = candidate_records(source)
        records = []
        for row in expected:
            row['reference_volume_ml'] = 10.
            err = 2. if row['patient'] == 'PanTrack_001' else 6.
            pred = {m: dict(volume_ml=10., dice=.8, hd95_mm=1., empty_prediction=False) for m in MODELS}
            if row['timepoint'] == 'fu': pred['new']['volume_ml'] += err
            records.append({**row, 'predictions': pred})
        maps = prediction_fixture()[2]
        result = summarize_external([records] * 3, expected, [maps] * 3, 400, 19, 6, 5.)
        self.assertAlmostEqual(result['seed_averaged']['strategies']['old_new']['mae_ml'], 4.)

    def test_common_draws_paired_contrasts_and_multiplicity(self):
        expected, records, maps = prediction_fixture()
        for row in records:
            if row['timepoint'] == 'fu':
                for model in MODELS: row['predictions'][model]['volume_ml'] += 2. if row['patient'].endswith('001') else 6.
        result = summarize_external([records] * 3, expected, [maps] * 3, 400, 19, 6, 5.)
        for contrast in result['seed_averaged']['primary_contrasts'].values():
            self.assertEqual(contrast['mean'], 0.)
            self.assertEqual(contrast['percentile_95_interval'], [0., 0.])
            self.assertEqual(contrast['bonferroni_family_interval'], [0., 0.])
        self.assertEqual(result['primary_family_size'], 6)

    def test_fixed_corrections_apply_without_external_fitting(self):
        expected, records, maps = prediction_fixture()
        maps['old_to_new'] = dict(slope=1., intercept_ml=3., degenerate=False)
        for row in records: row['predictions']['new']['volume_ml'] = 13.
        result = summarize_external([records] * 3, expected, [maps] * 3, 400, 19, 6, 5.)
        self.assertEqual(result['seed_averaged']['strategies']['old_new']['mae_ml'], 3.)
        self.assertEqual(result['seed_averaged']['strategies']['mapped_old_new']['mae_ml'], 0.)

    def test_policy_does_not_enable_internal_test_or_tuning(self):
        path = Path(__file__).resolve().parents[1] / 'config/preanalysis/ct_pantrack_confirmation_010_2026-09-18.json'
        policy = json.loads(path.read_text())
        validate_confirmation_policy(policy)
        for key in ['training_permitted', 'external_fitting_permitted', 'internal_test_values_permitted']:
            changed = dict(policy, **{key: True})
            with self.assertRaises(GateError): validate_confirmation_policy(changed)
        changed = copy.deepcopy(policy); changed['analysis']['primary_family_size'] = 3
        with self.assertRaises(GateError): validate_confirmation_policy(changed)


if __name__ == '__main__':
    unittest.main()

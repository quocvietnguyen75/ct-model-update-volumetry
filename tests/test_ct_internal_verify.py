"""Synthetic checks for the separate CT-UPDATE-014 arithmetic verifier."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from careport.ct_internal_holdout import summarize_holdout
from careport.ct_internal_verify import recompute_patient_values, verify_result
from careport.ct_pantrack_confirm import MAPS, MODELS


def fixture(scale=1):
    rows = []
    for patient, pair, error in [('a', 'one', -2), ('a', 'two', 2), ('b', 'three', 10)]:
        for tp in ('bl', 'fu'):
            predictions = {}
            for model in MODELS:
                # New/new changes are exact; mixed changes have opposite-signed errors.
                volume = 10 if model == 'old' else max(0, 10 + scale * error)
                predictions[model] = dict(volume_ml=volume, dice=0.4 if model == 'old' else 0.7,
                                          hd95_mm=1.0 if volume else None,
                                          empty_prediction=volume == 0)
                if not volume:
                    predictions[model]['dice'] = 0.0
            rows.append(dict(patient=patient, pair_id=pair, scan=patient + tp,
                             timepoint=tp, lesion_id=1, reference_volume_ml=10.0,
                             predictions=predictions))
    maps = {name: dict(slope=1.0, intercept_ml=0.0) for name in MAPS}
    return rows, maps


class InternalVerificationTests(unittest.TestCase):
    def test_absolute_loss_precedes_equal_patient_weighting(self):
        rows, maps = fixture()
        patients, values, _ = recompute_patient_values(rows, maps)
        self.assertEqual(patients, ['a', 'b'])
        self.assertEqual(values['mae:old_new'], [2.0, 10.0])
        self.assertEqual(values['signed:old_new'], [0.0, 10.0])
        self.assertEqual(values['mae:new_new'], [0.0, 0.0])
        self.assertEqual(len([k for k in values if k.startswith('mae:')]), 21)

    def test_map_direction_and_zero_clipping(self):
        rows, maps = fixture()
        maps['old_to_new'] = dict(slope=2.0, intercept_ml=-25.0)
        maps['new_to_old'] = dict(slope=0.0, intercept_ml=1.0)
        _, values, _ = recompute_patient_values(rows, maps)
        self.assertEqual(values['mae:mapped_old_new'], [10.0, 20.0])
        self.assertEqual(values['mae:old_mapped_new'], [9.0, 9.0])

    def test_duplicate_or_incomplete_pairs_fail(self):
        rows, maps = fixture()
        for invalid in (rows + rows[:1], rows[:-1]):
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                recompute_patient_values(invalid, maps)

    def test_empty_prediction_has_zero_dice_and_missing_hd(self):
        rows, maps = fixture()
        rows[1]['predictions']['new'] = dict(volume_ml=0.0, dice=0.0, hd95_mm=None, empty_prediction=True)
        _, values, boundary = recompute_patient_values(rows, maps)
        self.assertEqual(values['dice:new'], [0.35, 0.7])
        self.assertEqual(boundary['new']['empty_followup_predictions'], 1)
        self.assertEqual(boundary['new']['patients_with_defined_hd95'], 2)

    def test_complete_synthetic_result_and_csv_sensitivity(self):
        policy = json.loads(Path('config/preanalysis/ct_internal_holdout_014_2026-09-24.json').read_text())
        seed_rows, maps = zip(*(fixture(scale) for scale in (1, 2, 3)))
        expected = [{k: v for k, v in row.items() if k != 'predictions'} for row in seed_rows[0]]
        for row in expected:
            row['csv_volume_mm3'] = 11000.0 if row['timepoint'] == 'fu' else 10000.0
        result = summarize_holdout(list(seed_rows), expected, list(maps), policy)
        check = verify_result(result, list(seed_rows), expected, list(maps), policy)
        self.assertGreater(check['numeric_values_checked'], 400)
        self.assertLess(check['max_absolute_difference'], 1e-9)
        changed = deepcopy(result)
        changed['seed_averaged']['strategies']['old_new']['mae_ml'] += 0.1
        with self.assertRaises(ValueError):
            verify_result(changed, list(seed_rows), expected, list(maps), policy)
        changed = deepcopy(result)
        changed['csv_reference_sensitivity']['per_seed'][0]['strategies']['old_new']['mae_ml'] += 0.1
        with self.assertRaises(ValueError):
            verify_result(changed, list(seed_rows), expected, list(maps), policy)


if __name__ == '__main__':
    unittest.main()

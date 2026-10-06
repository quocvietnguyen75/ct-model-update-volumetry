from copy import deepcopy
import hashlib
import unittest

import numpy as np

from careport.ct_update_calibration import (
    assign_folds, fit_affine, crossfit_estimates, calibration_analysis,
)
from careport.ct_update_calibration_run import validate_prediction_records
from careport.ct_update_gate import GateError


def row(patient, x, y, lesion=1):
    return dict(patient=patient, lesion_id=lesion, reference_bl=y, reference_fu=y + 3,
                csv_reference_bl=y, csv_reference_fu=y + 3,
                old_bl=x, old_fu=x + 1.5, new_bl=y, new_fu=y + 3,
                continued_bl=x + 1, continued_fu=x + 3,
                old_dice=.5, new_dice=.6, continued_dice=.55)


class CTUpdateCalibrationTests(unittest.TestCase):
    def test_affine_recovers_scale_offset_and_handles_degenerate_and_negative_slope(self):
        fitted = fit_affine([1, 2, 3], [5, 7, 9], [1, 1, 1])
        self.assertAlmostEqual(fitted.slope, 2)
        self.assertAlmostEqual(fitted.intercept, 3)
        self.assertFalse(fitted.degenerate)
        flat = fit_affine([2, 2], [4, 6], [1, 1])
        self.assertEqual(flat.slope, 1)
        self.assertEqual(flat.intercept, 3)
        self.assertTrue(flat.degenerate)
        negative = fit_affine([1, 2], [5, 1], [1, 1])
        self.assertEqual(negative.slope, 0)
        self.assertEqual(negative.intercept, 3)
        clipped = fit_affine([5, 6], [0, 1], [1, 1])
        self.assertEqual(float(clipped.predict(np.array([0]))[0]), 0)
        for x, y, w in [([1], [float('nan')], [1]), ([1], [2], [0]),
                        ([-1], [2], [1]), ([1, 2], [1], [1, 1])]:
            with self.subTest(x=x), self.assertRaises(GateError):
                fit_affine(x, y, w)

    def test_fold_assignment_is_balanced_deterministic_and_patient_grouped(self):
        patients = [f'p{i}' for i in range(27)]
        folds = assign_folds(patients, 3, 'salt:')
        self.assertEqual(folds, assign_folds(patients[::-1], 3, 'salt:'))
        self.assertEqual([list(folds.values()).count(i) for i in range(3)], [9, 9, 9])
        ranked = sorted(patients, key=lambda p: (hashlib.sha256(('salt:' + p).encode()).hexdigest(), p))
        self.assertEqual([folds[p] for p in ranked], [i % 3 for i in range(27)])
        with self.assertRaises(GateError):
            assign_folds(['a', 'a', 'b'], 3, 'salt:')

    def test_fit_excludes_held_out_patient_references_and_all_followup_values(self):
        rows = [row(str(i), i + 1, 2 * (i + 1) + 3) for i in range(6)]
        folds = {str(i): i % 3 for i in range(6)}
        initial = crossfit_estimates(rows, folds)
        changed = deepcopy(rows)
        for r in changed:
            r['reference_bl'] += 100
            r['csv_reference_fu'] += 200
            r['new_fu'] += 500
        self.assertEqual(initial['coefficients'], crossfit_estimates(changed, folds)['coefficients'])
        changed[0]['new_bl'] += 1000
        after = crossfit_estimates(changed, folds)
        self.assertEqual([c for c in initial['coefficients'] if c['fold'] == 0],
                         [c for c in after['coefficients'] if c['fold'] == 0])
        self.assertNotEqual(initial['coefficients'], after['coefficients'])

    def test_patient_weighting_resists_lesion_replication_and_exact_mapping_removes_bias(self):
        rows = [row(str(i), i + 1, 2 * (i + 1) + 3) for i in range(6)]
        folds = {str(i): i % 3 for i in range(6)}
        result = crossfit_estimates(rows, folds)
        self.assertAlmostEqual(result['native']['strategies']['mapped_old_new']['mae_ml'], 0, places=10)
        self.assertGreater(result['native']['strategies']['old_new']['mae_ml'], 0)
        copies = rows + [{**rows[0], 'lesion_id': i} for i in range(2, 11)]
        duplicated = crossfit_estimates(copies, folds)
        for name, metric in result['native']['strategies'].items():
            self.assertAlmostEqual(metric['mae_ml'], duplicated['native']['strategies'][name]['mae_ml'])
        with self.assertRaises(GateError):
            crossfit_estimates(rows + [rows[0]], folds)
        with self.assertRaises(GateError):
            crossfit_estimates(rows, {**folds, 'unexpected': 1})

    def test_bootstrap_matches_independent_patient_resampling_with_refitted_affine(self):
        x = np.arange(1., 7.)
        y = np.array([2., 4., 8., 7., 13., 10.])
        rows = [row(str(i), x[i], y[i]) for i in range(6)]
        folds = {str(i): i // 2 for i in range(6)}
        result = calibration_analysis(rows, folds, 80, 29)
        rng = np.random.default_rng(29)
        differences = []
        for _ in range(80):
            sampled = [rng.choice(np.arange(f * 2, f * 2 + 2), 2, replace=True) for f in range(3)]
            corrected, original = [], []
            for fold in range(3):
                fitting = np.concatenate([sampled[f] for f in range(3) if f != fold])
                xx, yy = x[fitting], y[fitting]
                variance = float(np.mean((xx - xx.mean()) ** 2))
                slope = max(0., float(np.mean((xx - xx.mean()) * (yy - yy.mean()))) / variance)
                intercept = float(yy.mean() - slope * xx.mean())
                held = sampled[fold]
                corrected.extend(np.abs(y[held] - np.maximum(0, intercept + slope * x[held])))
                original.extend(np.abs(y[held] - x[held]))
            differences.append(float(np.mean(original) - np.mean(corrected)))
        actual = result['native']['contrasts']['old_to_new_correction_benefit']['exploratory_95_percentile_interval_ml']
        np.testing.assert_allclose(actual, np.quantile(differences, [.025, .975]), atol=1e-10)
        self.assertEqual(result, calibration_analysis(rows, folds, 80, 29))

    def test_prediction_keys_and_reference_values_must_match_validation_cache_exactly(self):
        index = []
        records = []
        for t in ('bl', 'fu'):
            index.append(dict(patient='a', lesion_id=1, timepoint=t, split='val',
                              reference_volume_ml=1., csv_volume_mm3=1000.))
            records.append(dict(patient='a', lesion_id=1, timepoint=t,
                                reference_volume_ml=1., csv_volume_mm3=1000.,
                                predictions={m: dict(volume_ml=1., dice=.8) for m in ('old', 'new', 'continued')}))
        self.assertEqual(len(validate_prediction_records(records, index)), 1)
        bad_reference = deepcopy(records)
        bad_reference[0]['reference_volume_ml'] = 2.
        for bad in (records[:-1], records + [records[0]],
                    [{**r, 'patient': 'test-patient'} for r in records], bad_reference):
            with self.subTest(records=bad), self.assertRaises(GateError):
                validate_prediction_records(bad, index)
        with self.assertRaises(GateError):
            validate_prediction_records(records, [{**r, 'split': 'test'} for r in index])


if __name__ == '__main__':
    unittest.main()

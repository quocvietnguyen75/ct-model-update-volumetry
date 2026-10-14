"""Synthetic contracts for the CT-UPDATE-019 compatibility and patient-level response analyses (no patient data)."""
import json
import math
import unittest

import numpy as np

from careport.ct_update_gate import GateError

COHORTS = ('internal', 'pantrack')


def row(patient, pair_id, timepoint, scan, reference, old, new, continued=None):
    continued = old if continued is None else continued
    predictions = {m: dict(volume_ml=v, dice=0.5, empty_prediction=v == 0, hd95_mm=None if v == 0 else 2.0)
                   for m, v in zip(('old', 'new', 'continued'), (old, new, continued))}
    return dict(patient=patient, pair_id=pair_id, scan=scan, timepoint=timepoint, lesion_id=1,
                reference_volume_ml=reference, predictions=predictions)


def pair(patient, pair_id, ref_bl, ref_fu, old_bl, new_bl, old_fu, new_fu, bl_scan=None, fu_scan=None):
    bl_scan = bl_scan or f'{patient}_BL_00'
    fu_scan = fu_scan or f'{patient}_FU_00'
    return [row(patient, pair_id, 'bl', bl_scan, ref_bl, old_bl, new_bl),
            row(patient, pair_id, 'fu', fu_scan, ref_fu, old_fu, new_fu)]


def cohort_records(cohort, shift, patients=6):
    """Each patient has three lesion pairs; PanTrack patients have two consecutive scan pairs."""
    records = []
    for k in range(patients):
        p = f'p{k:02d}'
        for j, size in enumerate((0.5, 5.0, 20.0)):
            jitter = 0.01 * (k + 1) * (j + 1)
            if cohort == 'pantrack':
                scans = (f'{p}_s1', f'{p}_s2') if j < 2 else (f'{p}_s2', f'{p}_s3')
            else:
                scans = (f'{p}_BL_0{j}', f'{p}_FU_0{j}')
            records += pair(p, f'{p}-{j}', size, size * (1.9 if j == 0 else 1.1), size * 0.8,
                            size * (0.8 + shift) + jitter, size * 0.9, size * (0.95 + 0.5 * shift) + jitter,
                            bl_scan=scans[0], fu_scan=scans[1])
    return records


def policy(patients=6, draws=200, unet_runs=3, nnunet_runs=2):
    return dict(protocol_id='CT-UPDATE-019',
                cohorts={c: dict(patients=patients, pairs=3 * patients) for c in COHORTS},
                segmenters=dict(unet=dict(runs=unet_runs, seeds=list(range(101, 101 + unet_runs)),
                                          bootstrap_seed=31),
                                nnunet=dict(runs=nnunet_runs, seeds=list(range(201, 201 + nnunet_runs)),
                                            bootstrap_seed=32)),
                bootstrap_draws=draws, limits_of_agreement_multiplier=1.96,
                response_diameter_thresholds=dict(decrease=-0.3, increase=0.2),
                assessment_unit=dict(internal='patient', pantrack='patient_and_consecutive_scan_pair'))


def segmenter_runs(pol=None):
    pol = pol or policy()
    shifts = dict(unet=(0.3, -0.2, 0.6), nnunet=(0.05, 0.1))
    return {s: {c: [(seed, cohort_records(c, shift + (0.1 if c == 'pantrack' else 0.0)))
                    for seed, shift in zip(pol['segmenters'][s]['seeds'], shifts[s])] for c in COHORTS}
            for s in ('unet', 'nnunet')}


class PairAndLogTermTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat
        self.m = ct_compat

    def test_pairs_join_both_timepoints_and_keep_their_scans(self):
        pairs = self.m.join_pairs(pair('a', 'a-0', 2.0, 3.0, 1.5, 2.5, 2.0, 3.5, 'a_s1', 'a_s2'))
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual((p['bl_scan'], p['fu_scan']), ('a_s1', 'a_s2'))
        self.assertEqual((p['ref_bl'], p['ref_fu'], p['bl']['old'], p['bl']['new'], p['fu']['new']),
                         (2.0, 3.0, 1.5, 2.5, 3.5))

    def test_incomplete_or_duplicated_pairs_are_rejected(self):
        records = pair('a', 'a-0', 2.0, 3.0, 1.5, 2.5, 2.0, 3.5)
        with self.assertRaises(GateError):
            self.m.join_pairs(records[:1])
        with self.assertRaises(GateError):
            self.m.join_pairs(records + records[:1])

    def test_mixed_log_error_is_the_same_version_error_plus_the_version_difference(self):
        p = self.m.join_pairs(pair('a', 'a-0', 2.0, 3.0, 1.5, 2.5, 2.0, 3.5))[0]
        terms = self.m.log_terms(p)
        reference = math.log(3.0 / 2.0)
        self.assertAlmostEqual(terms['d'], math.log(2.5 / 1.5))
        self.assertAlmostEqual(terms['e'], math.log(3.5 / 2.5) - reference)
        self.assertAlmostEqual(terms['e_mix'], math.log(3.5 / 1.5) - reference)
        self.assertAlmostEqual(terms['e_mix'], terms['e'] + terms['d'], places=12)

    def test_a_zero_volume_makes_the_pair_run_invalid_on_the_log_scale(self):
        for values in [(0.0, 2.5, 3.5), (1.5, 0.0, 3.5), (1.5, 2.5, 0.0)]:
            old_bl, new_bl, new_fu = values
            p = self.m.join_pairs(pair('a', 'a-0', 2.0, 3.0, old_bl, new_bl, 2.0, new_fu))[0]
            self.assertIsNone(self.m.log_terms(p))
        p = self.m.join_pairs(pair('a', 'a-0', 2.0, 3.0, 1.5, 2.5, 0.0, 3.5))[0]
        self.assertIsNotNone(self.m.log_terms(p))  # the old follow-up volume is not used


class WeightedMeanTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat, ct_replication
        self.m, self.r = ct_compat, ct_replication

    def test_unmasked_two_way_means_equal_the_frozen_replication_bootstrap(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(4, 7))
        run_w, patient_w = self.r.resampling_weights(300, 9, 4, dict(internal=7, pantrack=3))
        expected = ((run_w @ matrix) * patient_w['internal']).sum(axis=1) / (4 * 7)
        got = self.m.two_way_means(matrix, np.ones_like(matrix, dtype=bool), run_w, patient_w['internal'])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(self.m.point_mean(matrix, np.ones_like(matrix, dtype=bool)), matrix.mean())

    def test_masked_cells_leave_their_run_and_runs_stay_equally_weighted(self):
        values = np.array([[1.0, 3.0], [10.0, 99.0]])
        mask = np.array([[True, True], [True, False]])
        # run 1 mean 2, run 2 mean 10 (patient 2 has no valid pair there) -> 6, not (1 + 3 + 10) / 3
        self.assertAlmostEqual(self.m.point_mean(values, mask), 6.0)


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat
        self.m = ct_compat

    def test_relative_cost_is_the_ratio_of_mixed_to_same_version_mae(self):
        # one patient, one pair: same-version error 1 mL, mixed error 2 mL in both runs
        records = pair('a', 'a-0', 2.0, 2.0, 2.0, 3.0, 2.0, 4.0)  # new/new: 4-3-0=1; old/new: 4-2-0=2
        runs_pairs = [self.m.join_pairs(records)] * 2
        matrices = self.m.cohort_matrices(runs_pairs)
        ones = np.ones((1, 2)), np.ones((1, 1))
        result = self.m.relative_cost(matrices, *ones)
        self.assertAlmostEqual(result['mean_percent'], 100.0)
        self.assertEqual(result['runs_positive'], 2)

    def test_decomposition_identity_holds_and_kappa_predicts_inflation_without_cross_term(self):
        # two pairs: e = +a and -a, d = +b for both -> E[e d] = 0, so observed equals predicted inflation
        a, b = 0.3, 0.2
        records = []
        for k, sign in enumerate((1, -1)):
            new_bl = 2.0
            old_bl = new_bl / math.exp(b)
            new_fu = new_bl * math.exp(sign * a)  # reference change zero
            records += pair('a', f'a-{k}', 2.0, 2.0, old_bl, new_bl, 2.0, new_fu)
        matrices = self.m.cohort_matrices([self.m.join_pairs(records)])
        result = self.m.decomposition(matrices, np.ones((1, 1)), np.ones((1, 1)))
        self.assertAlmostEqual(result['m_same'], a * a)
        self.assertAlmostEqual(result['d2'], b * b)
        self.assertAlmostEqual(result['cross'], 0.0, places=12)
        self.assertAlmostEqual(result['identity_residual'], 0.0, places=12)
        self.assertAlmostEqual(result['kappa'], b / a)
        expected = 100 * (math.sqrt(1 + (b / a) ** 2) - 1)
        self.assertAlmostEqual(result['predicted_inflation_percent'], expected)
        self.assertAlmostEqual(result['observed_inflation_percent'], expected)

    def test_identity_holds_on_a_general_cohort(self):
        records = cohort_records('internal', 0.3)
        matrices = self.m.cohort_matrices([self.m.join_pairs(records)])
        result = self.m.decomposition(matrices, np.ones((1, 1)), np.ones((1, 6)))
        self.assertAlmostEqual(result['m_mix'], result['m_same'] + result['d2'] + result['cross'], places=12)

    def test_constant_version_difference_gives_zero_width_limits(self):
        records = []
        for k in range(3):
            records += pair('a', f'a-{k}', 2.0, 2.0, 1.0, 1.25, 2.0, 1.4 + 0.1 * k)
        matrices = self.m.cohort_matrices([self.m.join_pairs(records)])
        result = self.m.version_agreement(matrices, np.ones((1, 1)), np.ones((1, 1)), 1.96)
        self.assertAlmostEqual(result['bias_percent'], 25.0)
        self.assertAlmostEqual(result['sd_log'], 0.0, places=7)
        self.assertAlmostEqual(result['limits_percent'][0], 25.0, places=5)
        self.assertAlmostEqual(result['limits_percent'][1], 25.0, places=5)

    def test_invalid_pair_runs_are_counted_and_excluded(self):
        records = pair('a', 'a-0', 2.0, 2.0, 1.0, 1.25, 2.0, 1.5) + pair('b', 'b-0', 2.0, 2.0, 1.0, 1.0, 2.0, 0.0)
        matrices = self.m.cohort_matrices([self.m.join_pairs(records)])
        self.assertEqual((matrices['valid_pair_runs'], matrices['excluded_pair_runs']), (1, 1))
        self.assertEqual(matrices['log_mask'].tolist(), [[True, False]])


class ResponseTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat
        self.m = ct_compat
        self.thresholds = dict(decrease=-0.3, increase=0.2)

    def test_internal_assessments_sum_a_patients_pairs_across_image_indices(self):
        records = pair('a', 'a-0', 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 'a_BL_00', 'a_FU_00')
        records += pair('a', 'a-1', 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 'a_BL_01', 'a_FU_01')
        found = self.m.assessments(self.m.join_pairs(records), 'internal')
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]['ref_bl'], found[0]['ref_fu']), (4.0, 6.0))

    def test_pantrack_assessments_follow_consecutive_scan_pairs(self):
        records = pair('b', 'b-0', 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 'b_s1', 'b_s2')
        records += pair('b', 'b-1', 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 'b_s1', 'b_s2')
        records += pair('b', 'b-2', 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 'b_s2', 'b_s3')
        found = self.m.assessments(self.m.join_pairs(records), 'pantrack')
        self.assertEqual(len(found), 2)
        self.assertEqual(sorted((a['ref_bl'], a['ref_fu']) for a in found), [(2.0, 1.0), (4.0, 6.0)])

    def test_mixing_that_moves_a_burden_below_the_increase_threshold_is_counted(self):
        # Reference +80% (increase); new/new +80%; old/new from a larger old baseline +20% (stable).
        records = pair('a', 'a-0', 10.0, 18.0, 15.0, 10.0, 15.0, 18.0)
        conditions = self.m.assessment_conditions(self.m.assessments(self.m.join_pairs(records), 'internal')[0],
                                                  self.thresholds)
        self.assertEqual(conditions['differs'], 1)
        self.assertEqual(conditions['progression_split_differs'], 1)
        self.assertEqual(conditions['mixing_loses_reference'], 1)
        self.assertEqual(conditions['mixing_gains_reference'], 0)
        self.assertEqual(conditions['masked_increase'], 1)
        self.assertEqual(conditions['spurious_increase'], 0)
        self.assertEqual((conditions['agrees:new_new'], conditions['agrees:old_new']), (1, 0))

    def test_spurious_increase_from_a_smaller_old_baseline(self):
        # Reference +10% (stable); new/new +10%; old baseline 6 -> old/new +83% (increase).
        records = pair('a', 'a-0', 10.0, 11.0, 6.0, 10.0, 6.6, 11.0)
        conditions = self.m.assessment_conditions(self.m.assessments(self.m.join_pairs(records), 'internal')[0],
                                                  self.thresholds)
        self.assertEqual((conditions['spurious_increase'], conditions['masked_increase']), (1, 0))
        self.assertEqual(conditions['agrees:old_old'], 1)

    def test_patient_values_are_shares_of_their_assessments(self):
        records = pair('b', 'b-0', 10.0, 18.0, 15.0, 10.0, 15.0, 18.0, 'b_s1', 'b_s2')
        records += pair('b', 'b-1', 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 'b_s2', 'b_s3')
        patients, matrices, counts = self.m.response_matrices([self.m.join_pairs(records)], 'pantrack',
                                                              self.thresholds)
        self.assertEqual(patients, ['b'])
        self.assertEqual(matrices['differs'].tolist(), [[0.5]])
        self.assertEqual(counts['assessments'], 2)


class SummaryAndVerificationTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat
        self.m = ct_compat
        self.policy = policy()
        self.runs = segmenter_runs(self.policy)
        self.expected = {c: self.runs['unet'][c][0][1] for c in COHORTS}

    def test_summary_reports_every_planned_analysis_without_patient_values(self):
        result = self.m.summarize_compat(self.runs, self.expected, self.policy)
        self.assertEqual(result['status'], 'COMPAT_ANALYSIS_COMPLETE')
        self.assertEqual(result['protocol_id'], 'CT-UPDATE-019')
        self.assertEqual(set(result['segmenters']), {'unet', 'nnunet'})
        self.assertFalse(result['segmenters_pooled'])
        self.assertFalse(result['cohorts_pooled'])
        cohort = result['segmenters']['nnunet']['cohorts']['pantrack']
        self.assertEqual(set(cohort), {'patients', 'pairs', 'relative_compatibility_cost', 'version_agreement',
                                       'decomposition', 'patient_response'})
        self.assertEqual(result['segmenters']['nnunet']['runs'], 2)
        self.assertEqual(len(cohort['relative_compatibility_cost']['two_way_percentile_95']), 2)
        self.assertEqual(cohort['patient_response']['assessments'], 12)
        text = json.dumps(result)
        self.assertNotIn('p00', text)
        self.assertNotIn('pair_id', text)

    def test_bootstrap_is_reproducible_from_the_policy_seeds(self):
        first = self.m.summarize_compat(self.runs, self.expected, self.policy)
        second = self.m.summarize_compat(self.runs, self.expected, self.policy)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_verification_reproduces_the_summary_and_detects_tampering(self):
        result = self.m.summarize_compat(self.runs, self.expected, self.policy)
        report = self.m.verify_compat(json.loads(json.dumps(result)), self.runs, self.expected, self.policy)
        self.assertEqual(report['status'], 'COMPAT_VERIFIED')
        self.assertGreater(report['values_checked'], 100)
        for path in (('relative_compatibility_cost', 'mean_percent'), ('decomposition', 'kappa'),
                     ('patient_response', 'differs', 'mean')):
            tampered = json.loads(json.dumps(result))
            target = tampered['segmenters']['unet']['cohorts']['internal']
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] += 1e-6
            with self.subTest(path=path), self.assertRaises(GateError):
                self.m.verify_compat(tampered, self.runs, self.expected, self.policy)

    def test_runs_must_match_the_registered_seeds_and_cohort_sizes(self):
        wrong = json.loads(json.dumps(self.policy))
        wrong['segmenters']['nnunet']['seeds'] = [201, 299]
        with self.assertRaises(GateError):
            self.m.summarize_compat(self.runs, self.expected, wrong)
        with self.assertRaises(GateError):
            self.m.summarize_compat(self.runs, self.expected, policy(patients=7))

    def test_policy_validation_rejects_changed_definitions(self):
        for change in (dict(protocol_id='CT-UPDATE-016'), dict(limits_of_agreement_multiplier=2.0),
                       dict(response_diameter_thresholds=dict(decrease=-0.3, increase=0.25)),
                       dict(assessment_unit=dict(internal='patient', pantrack='patient'))):
            with self.subTest(change=change), self.assertRaises(GateError):
                self.m.validate_compat_policy(dict(self.policy, **change))


if __name__ == '__main__':
    unittest.main()

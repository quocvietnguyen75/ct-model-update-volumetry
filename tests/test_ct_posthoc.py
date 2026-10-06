"""Synthetic contracts for the CT-UPDATE-016 post hoc descriptive analyses (no patient data)."""
import json
import math
import unittest

from careport.ct_update_gate import GateError

COHORTS = ('internal', 'pantrack')


def row(patient, pair, timepoint, reference, old, new, continued=None, dice=(0.5, 0.6, 0.55)):
    continued = old if continued is None else continued
    predictions = {m: dict(volume_ml=v, dice=d, empty_prediction=v == 0, hd95_mm=None if v == 0 else 2.0)
                   for m, v, d in zip(('old', 'new', 'continued'), (old, new, continued), dice)}
    return dict(patient=patient, pair_id=pair, scan=f'{patient}-{timepoint}', timepoint=timepoint,
                lesion_id=1, reference_volume_ml=reference, predictions=predictions)


def pair(patient, pair_id, ref_bl, ref_fu, old_bl, new_bl, old_fu, new_fu, dice=(0.5, 0.6, 0.55)):
    return [row(patient, pair_id, 'bl', ref_bl, old_bl, new_bl, dice=dice),
            row(patient, pair_id, 'fu', ref_fu, old_fu, new_fu, dice=dice)]


def cohort_records(shift, patients=6):
    """Each patient has one small (0.5 mL), one medium (5 mL) and one large (20 mL) lesion pair."""
    records = []
    for k in range(patients):
        p = f'p{k:02d}'
        for j, size in enumerate((0.5, 5.0, 20.0)):
            jitter = 0.01 * (k + 1) * (j + 1)
            records += pair(p, f'{p}-{j}', size, size * 1.1, size * 0.8, size * 0.8 + shift + jitter,
                            size * 0.9, size * 0.95 + jitter, dice=(0.5, 0.6 + 0.001 * k + 0.01 * shift, 0.55))
    return records


def policy(patients=6, draws=200):
    return dict(protocol_id='CT-UPDATE-016', runs=3,
                cohorts={c: dict(patients=patients, pairs=3 * patients) for c in COHORTS},
                size_strata_ml=[['lt1', 0.0, 1.0], ['1to10', 1.0, 10.0], ['ge10', 10.0, None]],
                minimum_patients_for_interval=5,
                stratum_bootstrap=dict(draws=draws, seed=11), sensitivity_bootstrap=dict(draws=draws, seed=12,
                                                                                       family_size=6),
                shift_quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                response_diameter_thresholds=dict(decrease=-0.3, increase=0.2))


def runs(shifts=(0.3, -0.2, 0.6)):
    return {c: [(seed, cohort_records(s + (0.1 if c == 'pantrack' else 0))) for seed, s in
                zip((101, 102, 103), shifts)] for c in COHORTS}


class PairAndStratumTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_posthoc
        self.m = ct_posthoc

    def test_pairs_join_both_timepoints_and_give_strategy_errors(self):
        pairs = self.m.pair_rows(pair('a', 'a-0', 2.0, 3.0, 1.5, 2.5, 2.0, 3.5))
        self.assertEqual(len(pairs), 1)
        errors = self.m.strategy_errors(pairs[0])
        target = 3.0 - 2.0
        self.assertAlmostEqual(errors['old_old'], 2.0 - 1.5 - target)
        self.assertAlmostEqual(errors['old_new'], 3.5 - 1.5 - target)
        self.assertAlmostEqual(errors['new_new'], 3.5 - 2.5 - target)
        self.assertAlmostEqual(pairs[0]['baseline_shift'], 2.5 - 1.5)

    def test_incomplete_pairs_are_rejected(self):
        with self.assertRaises(GateError):
            self.m.pair_rows(pair('a', 'a-0', 2.0, 3.0, 1.5, 2.5, 2.0, 3.5)[:1])

    def test_strata_use_reference_baseline_and_exclude_patients_without_pairs(self):
        strata = policy()['size_strata_ml']
        self.assertEqual([self.m.stratum_of(v, strata) for v in (0.2, 0.999, 1.0, 9.99, 10.0, 300.0)],
                         ['lt1', 'lt1', '1to10', '1to10', 'ge10', 'ge10'])
        records = pair('a', 'a-0', 0.5, 0.6, 0.4, 0.6, 0.5, 0.7) + pair('a', 'a-1', 0.8, 0.9, 0.7, 0.9, 0.8, 1.0)
        records += pair('b', 'b-0', 20.0, 22.0, 18.0, 21.0, 19.0, 23.0)
        patients, values = self.m.stratum_patient_values(self.m.pair_rows(records), 'lt1', strata)
        self.assertEqual(patients, ['a'])  # patient b has no small lesion
        first = abs(0.7 - 0.4 - 0.1) - abs(0.7 - 0.6 - 0.1)
        second = abs(1.0 - 0.7 - 0.1) - abs(1.0 - 0.9 - 0.1)
        self.assertAlmostEqual(values['retaining_baseline_cost'][0], (first + second) / 2)


class DescriptiveSummaryTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_posthoc
        self.m = ct_posthoc

    def test_relative_errors_are_percent_of_reference_baseline_averaged_over_runs(self):
        run_a = self.m.pair_rows(pair('a', 'a-0', 2.0, 2.0, 2.0, 2.2, 2.0, 2.2))
        run_b = self.m.pair_rows(pair('a', 'a-0', 2.0, 2.0, 2.0, 2.6, 2.0, 2.6))
        summary = self.m.relative_summary([run_a, run_b])
        # old/new errors 0.2 and 0.6 mL -> 10% and 30% of 2 mL -> run average 20%
        self.assertAlmostEqual(summary['old_new_percent']['median'], 20.0)
        self.assertAlmostEqual(summary['new_new_percent']['median'], 0.0)
        self.assertAlmostEqual(summary['retention_cost_percent']['median'], 20.0)
        self.assertEqual(summary['retention_cost_percent']['share_positive'], 1.0)
        self.assertEqual(summary['pairs'], 1)

    def test_lesion_weighting_gives_every_pair_equal_weight(self):
        records = pair('a', 'a-0', 1.0, 1.0, 1.0, 2.0, 1.0, 2.0) + pair('b', 'b-0', 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        records += pair('b', 'b-1', 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        result = self.m.lesion_weighted_cost([self.m.pair_rows(records)])
        self.assertAlmostEqual(result['mean'], 1.0 / 3)  # patient weighting would give 0.5

    def test_spearman_uses_average_ranks_for_ties(self):
        x, y = [1, 2, 2, 3, 5], [2, 1, 4, 3, 5]
        ranks_x, ranks_y = [1, 2.5, 2.5, 4, 5], [2, 1, 4, 3, 5]
        mean = 3.0
        cov = sum((a - mean) * (b - mean) for a, b in zip(ranks_x, ranks_y))
        expected = cov / math.sqrt(sum((a - mean) ** 2 for a in ranks_x) * sum((b - mean) ** 2 for b in ranks_y))
        self.assertAlmostEqual(self.m.spearman(x, y), expected)
        self.assertAlmostEqual(self.m.spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_shift_distribution_counts_signs_and_quantiles(self):
        records = []
        for k, (old, new) in enumerate([(1.0, 0.5), (1.0, 1.0), (1.0, 1.5), (1.0, 2.0)]):
            records += pair('a', f'a-{k}', 1.0, 1.0, old, new, 1.0, 1.0)
        result = self.m.shift_distribution([self.m.pair_rows(records)], [0.5])
        self.assertEqual((result['share_negative'], result['share_zero'], result['share_positive']), (0.25, 0.25, 0.5))
        self.assertAlmostEqual(result['quantiles']['0.5'], 0.25)
        self.assertAlmostEqual(result['mean_absolute_ml'], (0.5 + 0 + 0.5 + 1.0) / 4)

    def test_response_categories_use_sphere_equivalent_thresholds(self):
        thresholds = self.m.volume_thresholds(dict(decrease=-0.3, increase=0.2))
        self.assertAlmostEqual(thresholds['decrease'], 100 * (0.7 ** 3 - 1))
        self.assertAlmostEqual(thresholds['increase'], 100 * (1.2 ** 3 - 1))
        category = lambda bl, fu: self.m.response_category(bl, fu, thresholds)
        self.assertEqual(category(10.0, 3.4), 'decrease')
        self.assertEqual(category(10.0, 3.5), 'stable')
        self.assertEqual(category(10.0, 17.3), 'increase')
        self.assertEqual(category(10.0, 17.2), 'stable')
        self.assertEqual(category(0.0, 1.0), 'undefined')

    def test_response_illustration_counts_changes_caused_by_mixing(self):
        # Reference grows 80% (increase). new/new measures +80% (increase); old/new starts from a larger
        # old baseline and measures +20% (stable), so mixing loses the correct category.
        records = pair('a', 'a-0', 10.0, 18.0, 15.0, 10.0, 15.0, 18.0)
        result = self.m.response_illustration([self.m.pair_rows(records)], dict(decrease=-0.3, increase=0.2))
        self.assertEqual(result['share_old_new_differs_from_new_new']['mean'], 1.0)
        self.assertEqual(result['share_mixing_loses_reference_category']['mean'], 1.0)
        self.assertEqual(result['share_mixing_gains_reference_category']['mean'], 0.0)
        self.assertEqual(result['agreement_with_reference']['new_new']['mean'], 1.0)
        self.assertEqual(result['agreement_with_reference']['old_new']['mean'], 0.0)

    def test_sign_probability_is_the_binomial_upper_tail(self):
        self.assertAlmostEqual(self.m.sign_probability(19, 1), 21 / 2 ** 20)
        self.assertAlmostEqual(self.m.sign_probability(10, 10), sum(math.comb(20, k) for k in range(10, 21)) / 2 ** 20)
        self.assertEqual(self.m.sign_probability(0, 0), 1.0)


class SummaryAndVerificationTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_posthoc
        self.m = ct_posthoc
        self.runs = runs()
        self.expected = {c: self.runs[c][0][1] for c in COHORTS}
        self.policy = policy()

    def test_summary_reports_every_planned_analysis_without_patient_values(self):
        result = self.m.summarize_posthoc(self.runs, self.expected, self.policy)
        self.assertEqual(result['status'], 'POSTHOC_ANALYSIS_COMPLETE')
        self.assertEqual(result['runs'], 3)
        cohort = result['cohorts']['internal']
        self.assertEqual(set(cohort), {'patients', 'pairs', 'size_strata', 'relative_scale', 'lesion_weighted',
                                       'run_associations', 'baseline_shift_distribution', 'interval_sensitivity',
                                       'response_illustration'})
        self.assertEqual([s['stratum'] for s in cohort['size_strata']], ['lt1', '1to10', 'ge10'])
        self.assertEqual(cohort['size_strata'][0]['patients'], 6)
        self.assertIn('two_way_percentile_95', cohort['size_strata'][0]['retaining_baseline_cost'])
        sensitivity = cohort['interval_sensitivity']['retaining_baseline_cost']
        self.assertEqual(len(sensitivity['two_way_family_interval']), 2)
        self.assertIn('sign_probability', sensitivity)
        text = json.dumps(result)
        self.assertNotIn('p00', text)
        self.assertNotIn('pair_id', text)

    def test_small_strata_get_point_estimates_without_intervals(self):
        small = policy(patients=3)
        few = {c: [(seed, [r for r in records if r['patient'] in ('p00', 'p01', 'p02')])
                   for seed, records in self.runs[c]] for c in COHORTS}
        result = self.m.summarize_posthoc(few, {c: few[c][0][1] for c in COHORTS}, small)
        stratum = result['cohorts']['pantrack']['size_strata'][2]
        self.assertNotIn('two_way_percentile_95', stratum['retaining_baseline_cost'])
        self.assertIn('runs_positive', stratum['retaining_baseline_cost'])

    def test_verification_reproduces_the_summary_and_detects_tampering(self):
        result = self.m.summarize_posthoc(self.runs, self.expected, self.policy)
        report = self.m.verify_posthoc(json.loads(json.dumps(result)), self.runs, self.expected, self.policy)
        self.assertEqual(report['status'], 'POSTHOC_VERIFIED')
        self.assertGreater(report['values_checked'], 50)
        tampered = json.loads(json.dumps(result))
        tampered['cohorts']['pantrack']['lesion_weighted']['mean'] += 1e-6
        with self.assertRaises(GateError):
            self.m.verify_posthoc(tampered, self.runs, self.expected, self.policy)

    def test_cohort_sizes_must_match_the_policy(self):
        wrong = policy(patients=7)
        with self.assertRaises(GateError):
            self.m.summarize_posthoc(self.runs, self.expected, wrong)


if __name__ == '__main__':
    unittest.main()

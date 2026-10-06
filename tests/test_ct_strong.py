"""Synthetic contracts for the CT-UPDATE-017 analysis (nnU-Net runs) and its timing-only schedule rule."""
import copy
import json
from pathlib import Path
import unittest

from careport.ct_update_gate import GateError
from test_ct_posthoc import COHORTS, runs as synthetic_runs

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT/'config/preanalysis/ct_strong_017_2026-10-02.json').read_text(encoding='utf-8'))


def small_policy(seeds=(101, 102, 103), minimum=2, patients=6):
    policy = copy.deepcopy(POLICY)
    policy.update(seeds=list(seeds), runs=len(seeds), minimum_completed_runs=minimum,
                  cohorts={c: dict(patients=patients, pairs=3 * patients) for c in COHORTS})
    policy['analysis']['bootstrap_draws'] = 200
    policy['descriptive']['stratum_bootstrap'] = dict(draws=200, seed=11)
    policy['descriptive']['sensitivity_bootstrap'] = dict(draws=200, seed=12, family_size=6)
    return policy


class ScheduleRuleTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_strong
        self.m = ct_strong

    def test_long_schedule_only_when_the_projected_run_fits_thirty_hours(self):
        fast = self.m.choose_schedule([300.0, 90.0, 100.0], POLICY)  # first epoch is warm-up and ignored
        self.assertEqual(fast['schedule'], 'long')
        self.assertEqual(fast['epochs'], dict(old=500, new=250, continued=250))
        self.assertAlmostEqual(fast['projected_long_run_hours'], 1000 * 95 / 3600 + 1.5)
        slow = self.m.choose_schedule([300.0, 110.0, 110.0], POLICY)
        self.assertEqual(slow['schedule'], 'short')
        self.assertEqual(slow['epochs'], dict(old=250, new=125, continued=125))
        boundary = (30.0 - 1.5) * 3600 / 1000
        self.assertEqual(self.m.choose_schedule([1.0, boundary, boundary], POLICY)['schedule'], 'long')

    def test_invalid_timing_is_rejected(self):
        for seconds in ([100.0, 100.0], [100.0, 0.0, 100.0], [100.0, float('nan'), 100.0]):
            with self.subTest(seconds=seconds), self.assertRaises(GateError):
                self.m.choose_schedule(seconds, POLICY)


class PolicyTests(unittest.TestCase):
    def test_committed_policy_is_valid_and_inconsistent_ones_are_not(self):
        from careport.ct_strong import validate_strong_policy
        validate_strong_policy(POLICY)
        for change in (dict(runs=9), dict(minimum_completed_runs=11), dict(protocol_id='CT-UPDATE-015')):
            with self.subTest(change=change), self.assertRaises(GateError):
                validate_strong_policy(dict(copy.deepcopy(POLICY), **change))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_strong
        self.m = ct_strong
        self.runs = synthetic_runs()
        self.expected = {c: self.runs[c][0][1] for c in COHORTS}
        self.policy = small_policy()

    def test_summary_has_the_primary_family_secondary_contrasts_and_descriptives(self):
        result = self.m.summarize_strong(self.runs, self.expected, self.policy)
        self.assertEqual(result['status'], 'STRONG_REPLICATION_ANALYSIS_COMPLETE')
        self.assertEqual(result['runs'], 3)
        cohort = result['cohorts']['pantrack']
        self.assertEqual(set(cohort['primary_contrasts']), {'retaining_baseline_cost', 'ordinary_update_benefit',
                                                            'contour_benefit'})
        self.assertIn('two_way_family_interval', cohort['primary_contrasts']['retaining_baseline_cost'])
        self.assertIn('baseline_version_shift_ml', cohort['secondary_contrasts'])
        self.assertEqual(set(cohort['followup_dice']), {'old', 'new', 'continued'})
        self.assertEqual(result['descriptive']['protocol_definitions'], 'CT-UPDATE-016')
        self.assertIn('size_strata', result['descriptive']['cohorts']['internal'])
        self.assertNotIn('p00', json.dumps(result))

    def test_verification_reproduces_and_rechecks_point_estimates(self):
        result = json.loads(json.dumps(self.m.summarize_strong(self.runs, self.expected, self.policy)))
        report = self.m.verify_strong(result, self.runs, self.expected, self.policy)
        self.assertEqual(report['status'], 'STRONG_REPLICATION_VERIFIED')
        self.assertGreater(report['values_checked'], 40)
        tampered = copy.deepcopy(result)
        tampered['cohorts']['internal']['primary_contrasts']['retaining_baseline_cost']['mean'] += 1e-6
        with self.assertRaises(GateError):
            self.m.verify_strong(tampered, self.runs, self.expected, self.policy)

    def test_runs_must_be_registered_ordered_and_numerous_enough(self):
        with self.assertRaises(GateError):
            self.m.summarize_strong(self.runs, self.expected, small_policy(seeds=(101, 102, 103, 104), minimum=4))
        with self.assertRaises(GateError):
            self.m.summarize_strong(self.runs, self.expected, small_policy(seeds=(103, 102, 101)))
        # Three of four registered runs completed (the descriptive correlations need at least three).
        result = self.m.summarize_strong(self.runs, self.expected, small_policy(seeds=(101, 102, 103, 104), minimum=3))
        self.assertEqual(result['missing_seeds'], [104])


if __name__ == '__main__':
    unittest.main()

"""Synthetic contracts for CT manuscript table rendering and the number-consistency check."""
import unittest

from careport.ct_manuscript import (MANUSCRIPT_CONTRASTS, decimal_tokens, latex_interval, latex_number, latex_table,
                                    model_comparison_rows, percent_values, posthoc_sensitivity_rows,
                                    posthoc_stratum_rows, primary_contrast_rows, source_values, unmatched_intervals,
                                    unmatched_numbers, unmatched_percentages)
from careport.ct_update_gate import GateError


class ContrastOrderTests(unittest.TestCase):
    def test_manuscript_tables_list_the_retention_cost_first(self):
        self.assertEqual(MANUSCRIPT_CONTRASTS, ('retaining_baseline_cost', 'ordinary_update_benefit', 'contour_benefit'))


class LatexTableTests(unittest.TestCase):
    def test_table_has_one_line_per_row_between_rules(self):
        text = latex_table(['Cohort', 'MAE (mL)'], [['Internal', '4.279'], ['PanTrack', '8.895']], 'lr')
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], r'\begin{tabular}{lr}')
        self.assertEqual(lines[-1], r'\end{tabular}')
        self.assertEqual(sum(line.endswith(r'\\') for line in lines), 3)
        self.assertEqual(lines.count(r'\hline'), 3)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(GateError):
            latex_table(['a', 'b'], [['1']], 'lr')
        with self.assertRaises(GateError):
            latex_table(['a', 'b'], [['1', '2']], 'l')


class NumberCheckTests(unittest.TestCase):
    TEXT = (r'MAE rose from 4.279 to 11.103 mL ($-$0.813 to 1.281), Dice 0.5604; MONAI 1.5.1 on 12 patients, '
            r'2,024 pairs and a 9.999 typo.')

    def test_tokens_are_decimals_only_and_skip_version_strings(self):
        tokens = [t for t, _, _ in decimal_tokens(self.TEXT)]
        self.assertEqual(tokens, ['4.279', '11.103', '0.813', '1.281', '0.5604', '9.999'])
        self.assertEqual(decimal_tokens('Dice 0.5604')[0][2], 4)

    def test_tokens_must_round_from_a_source_value_ignoring_sign(self):
        sources = [4.278521, 11.10339, -0.81325, 1.28124, 0.560365]
        self.assertEqual(unmatched_numbers(self.TEXT, sources), ['9.999'])
        self.assertEqual(unmatched_numbers(self.TEXT, sources, exempt={'9.999'}), [])

    SUMMARIES = {'cohorts': {'internal': {'cost': dict(mean=0.4453, two_way_percentile_95=[0.1145, 0.8389],
                                                       per_run_means=[-0.32, 0.5, 1.19]),
                                          'benefit': dict(mean=-1.6471, two_way_family_interval=[-3.8256, 0.1571])}}}

    def test_each_interval_must_match_one_reported_interval(self):
        text = r'cost $+0.45$\,mL internally ($+0.11$ to $+0.84$), ranging from $-0.32$ to $+1.19$\,mL'
        self.assertEqual(unmatched_intervals(text, self.SUMMARIES), [])
        wrong = text.replace('$+0.11$', '$+0.12$')
        self.assertEqual(unmatched_intervals(wrong, self.SUMMARIES), ['+0.12 to +0.84'])

    def test_a_reported_interval_may_appear_with_reversed_sign(self):
        text = 'difference 1.65 mL; family interval $-$0.16 to 3.83; direct ($-3.826$ to $+0.157$)'
        self.assertEqual(unmatched_intervals(text, self.SUMMARIES), [])
        self.assertEqual(unmatched_intervals('interval $-$0.16 to 3.84', self.SUMMARIES), ['-0.16 to 3.84'])

    def test_source_values_collect_numbers_from_nested_json_and_csv_text(self):
        values = source_values([{'a': [1.5, {'b': -2.25}], 'c': 'x', 'd': True}], ['k,v\nm,0.125\nn,text\n'])
        self.assertEqual(sorted(values), [-2.25, 0.125, 1.5])


class PercentageCheckTests(unittest.TestCase):
    def test_percent_values_come_only_from_keys_named_percent(self):
        objects = [{'share': 0.7689, 'relative_percent': {'median': 8.86, 'q25': -32.23},
                    'two_way_percentile_95': [57.07, 60.0], 'thresholds_percent': {'decrease': -65.7}}]
        self.assertEqual(sorted(percent_values(objects)), [-65.7, -32.23, 8.86])

    def test_decimal_percentages_round_from_a_fraction_or_a_value_stored_in_percent(self):
        text = r'shares 76.9\% and 8.9\%, quartile $-32.2\%$, threshold $-65.7\%$; not 57.1\% or 12.0\%'
        fractions = [0.7689, 57.07, 60.0]
        self.assertEqual(unmatched_percentages(text, fractions, [8.86, -32.23, -65.7]), ['57.1%', '12.0%'])
        self.assertEqual(unmatched_percentages(r'share 76.9\%', fractions), [])


class LatexNumberTests(unittest.TestCase):
    def test_signed_numbers_use_a_typeset_minus_and_an_explicit_plus(self):
        self.assertEqual(latex_number(-0.0577, 3, True), '$-$0.058')
        self.assertEqual(latex_number(2.07618, 3, True), '+2.076')
        self.assertEqual(latex_number(4.8, 3), '4.800')
        self.assertEqual(latex_number(-1.5, 1), '$-$1.5')
        self.assertEqual(latex_interval(-0.0577, 4.3034, 3, True), '[$-$0.058, +4.303]')


def _summary(mean, interval=None, positive=10, negative=10):
    found = dict(mean=mean, runs_positive=positive, runs_negative=negative)
    if interval is not None:
        found['two_way_percentile_95'] = interval
    return found


class PosthocTableTests(unittest.TestCase):
    LARGE = dict(stratum='lt1', lower_ml=0.0, upper_ml=1.0, patients=25, pairs=205,
                 mae_ml=dict(old_old=0.60434, old_new=0.67167, new_new=0.51877),
                 retaining_baseline_cost=_summary(0.15290, [0.02495, 0.32022], 18, 2),
                 ordinary_update_benefit=_summary(-0.06733, [-0.26219, 0.07488], 9, 11))
    SMALL = dict(stratum='1to10', lower_ml=1.0, upper_ml=10.0, patients=3, pairs=4,
                 mae_ml=dict(old_old=4.8337, old_new=4.6082, new_new=4.1307),
                 retaining_baseline_cost=_summary(0.47750, None, 2, 1),
                 ordinary_update_benefit=_summary(0.22544, None, 1, 2))
    EMPTY = dict(stratum='ge10', lower_ml=10.0, upper_ml=None, patients=0, pairs=0)
    RESULTS = {'cohorts': {'internal': {'size_strata': [LARGE, SMALL, EMPTY]}, 'pantrack': {'size_strata': [LARGE]}}}

    def test_stratum_rows_follow_cohort_and_policy_order_with_volume_labels(self):
        rows = posthoc_stratum_rows(self.RESULTS, 'retaining_baseline_cost')
        self.assertEqual([row[:3] for row in rows],
                         [['Internal', '$<1$', '25 / 205'], ['Internal', '1 to $<10$', '3 / 4'],
                          ['Internal', r'$\geq10$', '0 / 0'], ['PanTrack', '$<1$', '25 / 205']])
        self.assertEqual(rows[0][3:], ['0.604', '0.672', '0.519', '+0.153', '[+0.025, +0.320]', '18 / 2'])

    def test_small_strata_have_no_interval_and_empty_strata_no_values(self):
        rows = posthoc_stratum_rows(self.RESULTS, 'ordinary_update_benefit', with_mae=False)
        self.assertEqual(rows[0][3:], ['$-$0.067', '[$-$0.262, +0.075]', '9 / 11'])
        self.assertEqual(rows[1][3:], ['+0.225', '---', '1 / 2'])
        self.assertEqual(rows[2][3:], ['---', '---', '---'])
        self.assertEqual(posthoc_stratum_rows(self.RESULTS, 'retaining_baseline_cost')[2][3:], ['---'] * 6)

    @staticmethod
    def _sensitivity_inputs(recomputed_excludes_zero=True):
        published, recomputed = {'cohorts': {}}, {'cohorts': {}}
        for cohort in ('internal', 'pantrack'):
            published['cohorts'][cohort] = {'primary_contrasts': {
                name: dict(two_way_family_interval=[0.0916, 1.0601], family_interval_excludes_zero=True)
                for name in MANUSCRIPT_CONTRASTS}}
            recomputed['cohorts'][cohort] = {'interval_sensitivity': {
                name: dict(two_way_family_interval=[0.08747, 1.05728], family_interval_excludes_zero=recomputed_excludes_zero,
                           runs_positive=19, runs_negative=1, sign_probability=2.0027e-05)
                for name in MANUSCRIPT_CONTRASTS}}
        recomputed['cohorts']['pantrack']['interval_sensitivity']['ordinary_update_benefit'].update(
            runs_positive=11, runs_negative=9, sign_probability=0.41190)
        return published, recomputed

    def test_sensitivity_rows_pair_published_and_recomputed_family_intervals(self):
        rows = posthoc_sensitivity_rows(*self._sensitivity_inputs())
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0], ['Internal', 'Retention cost (mL)', '[+0.092, +1.060]', '[+0.087, +1.057]', 'Yes',
                                   '19 / 1', '$<$0.001'])
        self.assertEqual(rows[2][2:4], ['[+0.0916, +1.0601]', '[+0.0875, +1.0573]'])
        self.assertEqual(rows[4][5:], ['11 / 9', '0.412'])

    def test_a_changed_decision_is_shown_as_changed(self):
        rows = posthoc_sensitivity_rows(*self._sensitivity_inputs(recomputed_excludes_zero=False))
        self.assertEqual({row[4] for row in rows}, {'Changed'})


def _study(scale, runs, dice_old, dice_new):
    """A replication-shaped result whose values are scaled copies of one template."""
    def summary(mean, low, high, family, positive):
        return dict(mean=mean * scale, two_way_percentile_95=[low * scale, high * scale],
                    two_way_family_interval=[family[0] * scale, family[1] * scale],
                    family_interval_excludes_zero=family[0] * family[1] > 0, runs_positive=positive,
                    runs_negative=runs - positive)
    cohorts = {}
    for cohort in ('internal', 'pantrack'):
        cohorts[cohort] = dict(
            followup_dice=dict(old=dice_old, new=dice_new, continued=dice_old),
            primary_contrasts=dict(retaining_baseline_cost=summary(0.52417, 0.21003, 0.90217, (0.09162, 1.06013), runs - 1),
                                   ordinary_update_benefit=summary(0.13933, -0.38986, 1.07037, (-0.52906, 1.60083), runs // 2),
                                   contour_benefit=summary(0.04424, 0.03086, 0.05924, (0.02643, 0.06496), runs)),
            secondary_contrasts=dict(baseline_version_shift_ml=summary(0.48286, -0.01923, 1.20915, (-1.0, 2.0), runs - 5)))
    return dict(runs=runs, cohorts=cohorts)


class StrongModelTableTests(unittest.TestCase):
    UNET = _study(1.0, 20, 0.58021, 0.62445)
    NNUNET = _study(2.0, 10, 0.70519, 0.73388)

    def test_primary_rows_give_mean_both_intervals_the_decision_and_run_signs(self):
        rows = primary_contrast_rows(self.UNET)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0], ['Internal', 'Retention cost (mL)', '+0.524', '[+0.210, +0.902]', '[+0.092, +1.060]',
                                   'Yes', '19 / 1'])
        self.assertEqual(rows[1][3:6], ['[$-$0.390, +1.070]', '[$-$0.529, +1.601]', 'No'])
        self.assertEqual(rows[2][2:5], ['+0.0442', '[+0.0309, +0.0592]', '[+0.0264, +0.0650]'])
        self.assertEqual([row[0] for row in rows], ['Internal'] * 3 + ['PanTrack'] * 3)

    def test_comparison_rows_put_the_two_models_side_by_side_without_pooling(self):
        rows = model_comparison_rows(self.UNET, self.NNUNET)
        self.assertEqual(len(rows), 12)
        self.assertEqual([row[1] for row in rows[:6]],
                         ['Follow-up Dice, old model', 'Follow-up Dice, updated model', 'Retention cost (mL)',
                          'Update benefit (mL)', 'Contour benefit (Dice)', 'Baseline version shift (mL)'])
        self.assertEqual(rows[0][2:], ['0.5802', '---', '0.7052', '---'])
        self.assertEqual(rows[2][2:], ['+0.524 [+0.092, +1.060]', '19 / 1', '+1.048 [+0.183, +2.120]', '9 / 1'])
        # The baseline shift is a secondary contrast: its two-way 95% interval, not a family-wise one.
        self.assertEqual(rows[5][2:4], ['+0.483 [$-$0.019, +1.209]', '15 / 5'])
        self.assertEqual(rows[6][0], 'PanTrack')


if __name__ == '__main__':
    unittest.main()

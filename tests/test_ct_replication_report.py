"""Synthetic contracts for CT-UPDATE-015 manuscript figure and table data."""
import copy
import unittest

from careport.ct_replication_report import PRIMARY, RUN_COLUMNS, markdown_table, panel_data, run_rows, run_table_rows
from careport.ct_update_gate import GateError

SEEDS = list(range(20261001, 20261021))
ORIGINAL = (20260915, 20260917, 20260919)


def contrast(values, family=None):
    result = dict(mean=sum(values)/len(values), per_run_means=list(values),
                  two_way_percentile_95=[min(values), max(values)],
                  runs_positive=sum(v > 0 for v in values), runs_negative=sum(v < 0 for v in values))
    if family is not None:
        result.update(two_way_family_interval=list(family),
                      family_interval_excludes_zero=family[0] > 0 or family[1] < 0)
    return result


def results():
    cohorts = {}
    for offset, cohort in enumerate(('internal', 'pantrack')):
        primary = {}
        for k, name in enumerate(PRIMARY):
            values = [(-1)**(i + k) * 0.1 * (i + 1) + offset for i in range(20)]
            established = offset == 1 and name == 'contour_benefit'
            primary[name] = contrast(values, (0.2 if established else min(values) - 1, max(values) + 1))
        secondary = {name: contrast([0.05 * (i - 7) + offset + k for i in range(20)])
                     for k, name in enumerate(('baseline_version_shift_ml', 'continued_retaining_baseline_cost'))}
        cohorts[cohort] = dict(patients=29 + 16*offset, pairs=277 - 96*offset, primary_contrasts=primary,
                               secondary_contrasts=secondary)
    return dict(status='REPLICATION_ANALYSIS_COMPLETE', protocol_id='CT-UPDATE-015', runs=20, seeds=SEEDS,
                cohorts=cohorts)


def original_rows():
    rows = []
    for cohort in ('internal', 'PanTrack'):
        for seed in (*ORIGINAL, 'fixed_seed_average'):
            for k, name in enumerate(PRIMARY):
                rows.append(dict(cohort=cohort, seed=str(seed), contrast=name, mean=str(0.5 + k)))
    return rows


class PanelDataTests(unittest.TestCase):
    def test_panels_keep_every_run_and_exclude_the_original_average(self):
        panels = panel_data(results(), original_rows())
        self.assertEqual(tuple(panels), PRIMARY)
        for name in PRIMARY:
            for cohort in ('internal', 'pantrack'):
                panel = panels[name][cohort]
                self.assertEqual(len(panel['replication']), 20)
                self.assertEqual(len(panel['original']), 3)
                zeros = sum(v == 0 for v in panel['replication'])
                self.assertEqual(panel['runs_positive'] + panel['runs_negative'] + zeros, 20)
                self.assertEqual(len(panel['family']), 2)
        self.assertEqual(panels['contour_benefit']['pantrack']['original'], [2.5, 2.5, 2.5])

    def test_incomplete_or_inconsistent_inputs_are_rejected(self):
        broken = results()
        broken['cohorts']['internal']['primary_contrasts']['contour_benefit']['per_run_means'].pop()
        with self.assertRaises(GateError):
            panel_data(broken, original_rows())
        with self.assertRaises(GateError):
            panel_data(results(), [r for r in original_rows() if r['seed'] != '20260917'])
        broken = copy.deepcopy(results())
        broken['status'] = 'PENDING'
        with self.assertRaises(GateError):
            panel_data(broken, original_rows())

    def test_markdown_table_reports_all_six_contrasts_with_the_prespecified_rule(self):
        text = markdown_table(results())
        lines = [line for line in text.splitlines() if line.startswith('| ') and '---' not in line]
        self.assertEqual(len(lines), 7)
        self.assertIn('Retention cost (mL)', text)
        self.assertIn('Yes', text)
        self.assertTrue(all(line.count('|') == 8 for line in lines))


class RunRowTests(unittest.TestCase):
    def test_rows_follow_the_fixed_seed_order_and_average_to_the_reported_means(self):
        data = results()
        rows = run_rows(data)
        self.assertEqual(len(rows), 40)
        for cohort in ('internal', 'pantrack'):
            mine = [r for r in rows if r['cohort'] == cohort]
            self.assertEqual([r['seed'] for r in mine], SEEDS)
            for section, name in RUN_COLUMNS:
                reported = data['cohorts'][cohort][section][name]
                self.assertEqual([r[name] for r in mine], reported['per_run_means'])
                self.assertAlmostEqual(sum(r[name] for r in mine) / 20, reported['mean'])

    def test_missing_run_values_are_rejected(self):
        broken = results()
        broken['cohorts']['pantrack']['secondary_contrasts']['baseline_version_shift_ml']['per_run_means'].pop()
        with self.assertRaises(GateError):
            run_rows(broken)
        broken = results()
        broken['seeds'] = SEEDS[:-1]
        with self.assertRaises(GateError):
            run_rows(broken)

    def test_stronger_model_results_give_run_rows_in_their_own_seed_order(self):
        strong = results()
        strong.update(status='STRONG_REPLICATION_ANALYSIS_COMPLETE', protocol_id='CT-UPDATE-017', runs=10,
                      seeds=list(range(20261201, 20261211)))
        for cohort in strong['cohorts'].values():
            for section in ('primary_contrasts', 'secondary_contrasts'):
                for summary in cohort[section].values():
                    summary['per_run_means'] = summary['per_run_means'][:10]
        rows = run_rows(strong)
        self.assertEqual(len(rows), 20)
        self.assertEqual([r['seed'] for r in rows if r['cohort'] == 'pantrack'], list(range(20261201, 20261211)))
        strong['status'] = 'POSTHOC_ANALYSIS_COMPLETE'
        with self.assertRaises(GateError):
            run_rows(strong)

    def test_table_rows_format_every_run_and_a_mean_row(self):
        rows = run_rows(results())
        table = run_table_rows(rows, 'internal')
        self.assertEqual(len(table), 21)
        self.assertEqual(table[0][0], '20261001')
        self.assertEqual(table[-1][0], 'Mean')
        self.assertTrue(all(len(line) == 1 + len(RUN_COLUMNS) for line in table))
        self.assertTrue(all(cell.startswith(('+', '-', '$-$')) for line in table for cell in line[1:]))
        dice = [name for _, name in RUN_COLUMNS].index('contour_benefit') + 1
        self.assertEqual(len(table[0][dice].split('.')[1]), 4)


if __name__ == '__main__':
    unittest.main()

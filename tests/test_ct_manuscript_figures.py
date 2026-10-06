"""Synthetic contracts for the print-size CT manuscript figures (aggregates only)."""
import re
import tempfile
import unittest
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    HAVE_MPL = True
except ImportError:  # pragma: no cover - matplotlib is part of the figure environment
    HAVE_MPL = False

CONTRASTS = ('retaining_baseline_cost', 'ordinary_update_benefit', 'contour_benefit')
COHORTS = (('internal', 'Internal\n29 patients\n277 pairs'), ('pantrack', 'PanTrack\n45 patients\n181 pairs'))


def panels(with_runs):
    data = {}
    for k, name in enumerate(CONTRASTS):
        data[name] = {}
        for j, (cohort, _) in enumerate(COHORTS):
            entry = dict(estimate=0.5 + k + j, interval=(-0.2 + k, 1.4 + k + j), original=[0.1, 0.6, 1.2 + j])
            if with_runs:
                entry.update(runs=[-0.3 + 0.08 * i + k for i in range(20)], positive=19)
            data[name][cohort] = entry
    return data


def batches():
    rows = []
    for shift in (0.9, -0.6, 6.3, 2.2):
        rows.append(dict(components=dict(old_new_minus_old_old=shift, old_new_minus_new_new=shift / 2,
                                         new_new_minus_old_old=shift / 2),
                         strata={'lt1': 0.1 * shift, '1to10': -0.1 * shift, 'ge10': 1.5 * shift}))
    return rows


@unittest.skipUnless(HAVE_MPL, 'matplotlib unavailable')
class PrintFigureTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_manuscript_figures as figures
        self.f = figures

    def check_print_size(self, fig):
        self.assertAlmostEqual(fig.get_size_inches()[0], self.f.TEXT_WIDTH_IN, places=3)
        sizes = self.f.text_sizes(fig)
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), self.f.MIN_TEXT_PT)
        # Nothing may be cut off by the page edge, and value labels must stay inside their own panel.
        self.assertEqual(self.f.text_outside_bounds(fig), [])

    def test_text_width_is_the_iop_column(self):
        self.assertAlmostEqual(self.f.TEXT_WIDTH_IN, 153 / 25.4)
        self.assertEqual(self.f.MIN_TEXT_PT, 8.0)

    def test_contrast_figures_fit_the_text_width_with_legible_text(self):
        for with_runs in (True, False):
            self.check_print_size(self.f.contrast_figure(panels(with_runs), COHORTS, self.f.REPLICATION_LEGEND
                                                         if with_runs else self.f.THREE_RUN_LEGEND))

    def test_contrast_panels_follow_the_retention_first_order(self):
        fig = self.f.contrast_figure(panels(True), COHORTS, self.f.REPLICATION_LEGEND)
        self.assertEqual([ax.get_title(loc='left') for ax in fig.axes],
                         ['Retention cost', 'Update benefit', 'Contour benefit'])
        self.assertEqual(self.f.CONTRASTS, CONTRASTS)

    def test_estimates_original_runs_and_replication_runs_use_distinct_markers(self):
        fig = self.f.contrast_figure(panels(True), COHORTS, self.f.REPLICATION_LEGEND)
        markers = [h.get_marker() for h in fig.legends[0].legend_handles]
        self.assertEqual(len(markers), 3)
        self.assertEqual(len(set(markers)), 3)
        three_run = self.f.contrast_figure(panels(False), COHORTS, self.f.THREE_RUN_LEGEND)
        same = {h.get_label(): h.get_marker() for h in three_run.legends[0].legend_handles}
        self.assertEqual(len(same), 2)
        self.assertEqual(sorted(same.values()), sorted({markers[1], markers[2]}))

    def test_missing_panel_values_are_rejected(self):
        data = panels(True)
        del data['contour_benefit']['pantrack']
        with self.assertRaises(ValueError):
            self.f.contrast_figure(data, COHORTS, self.f.REPLICATION_LEGEND)

    def test_schematic_names_every_strategy_and_contrast(self):
        fig = self.f.strategy_schematic()
        self.check_print_size(fig)
        texts = ' '.join(self.f.figure_texts(fig))
        for word in ('old/old', 'old/new', 'new/new', 'Retention cost', 'Update benefit', 'Contour benefit'):
            self.assertIn(word, texts)

    def test_diagnostic_figure_fits_the_text_width_with_legible_labels(self):
        fig = self.f.diagnostic_figure(batches(), ['20260915', '20260917', '20260919', 'Three-run\naverage'],
                                       {'lt1': 6, '1to10': 24, 'ge10': 36})
        self.check_print_size(fig)
        self.assertEqual(len(fig.axes), 2)

    def test_saved_pdf_page_is_the_text_width_and_reproducible(self):
        # Two independent builds of the same figure must give identical bytes (no dates, stable layout).
        fig = self.f.contrast_figure(panels(True), COHORTS, self.f.REPLICATION_LEGEND)
        again = self.f.contrast_figure(panels(True), COHORTS, self.f.REPLICATION_LEGEND)
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'a.pdf', Path(tmp)/'b.pdf'
            self.f.save_pdf(fig, first)
            self.f.save_pdf(again, second)
            data = first.read_bytes()
            box = re.search(rb'/MediaBox\s*\[\s*0 0 ([\d.]+) ([\d.]+)\s*\]', data)
            self.assertIsNotNone(box)
            self.assertAlmostEqual(float(box.group(1)), 153 / 25.4 * 72, delta=0.5)
            self.assertEqual(data, second.read_bytes())
            with self.assertRaises(FileExistsError):
                self.f.save_pdf(fig, first, overwrite=False)


if __name__ == '__main__':
    unittest.main()

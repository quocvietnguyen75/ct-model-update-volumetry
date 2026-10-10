"""Synthetic contracts for the print-size CT manuscript figures (synthetic values only)."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.text import Text
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


STRATA_LAYOUT = ((0, 1, 25, 205), (1, 10, 19, 51), (10, None, 13, 21))


def replication(runs, scale):
    """Synthetic replication summary shaped like the verified result files (aggregates only)."""
    cohorts = {}
    for j, (cohort, _) in enumerate(COHORTS):
        primary = {}
        for k, name in enumerate(CONTRASTS):
            mean = scale * (0.5 + k + j)
            primary[name] = dict(mean=mean, two_way_family_interval=[mean - 1.0, mean + 1.2],
                                 per_run_means=[mean - 0.4 + 0.8 * i / (runs - 1) for i in range(runs)],
                                 runs_positive=runs, runs_negative=0)
        strata = []
        for lower, upper, patients, pairs in STRATA_LAYOUT:
            m = scale * (0.1 + lower / 5) + j
            strata.append(dict(lower_ml=lower, upper_ml=upper, patients=patients, pairs=pairs,
                               retaining_baseline_cost=dict(mean=m, two_way_percentile_95=[m - 0.3, m + 0.5],
                                                            per_run_means=[m - 0.1 + 0.2 * i / (runs - 1)
                                                                           for i in range(runs)],
                                                            runs_positive=runs, runs_negative=0)))
        cohorts[cohort] = dict(primary_contrasts=primary, size_strata=strata)
    return dict(runs=runs, cohorts=cohorts)


MODEL_LEGEND = {'unet': 'U-Net', 'nnunet': 'nnU-Net'}


def disc(size, centre, radius):
    rows, cols = np.ogrid[:size, :size]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius ** 2


def example():
    """Synthetic example pair shaped like the CT-UPDATE-018 private crops (no study data)."""
    timepoints = {}
    for timepoint, radius in (('bl', 14), ('fu', 17)):
        image = np.full((80, 80), -20.0)
        image[disc(80, (40, 40), radius)] = 60.0
        timepoints[timepoint] = dict(image=image, reference=disc(80, (40, 40), radius),
                                     old=disc(80, (40, 41), radius - 3), new=disc(80, (41, 40), radius),
                                     spacing=(0.75, 0.75))
    volumes = {'reference': {'bl': 12.5, 'fu': 18.25}, 'old': {'bl': 9.75, 'fu': 14.5},
               'new': {'bl': 12.0, 'fu': 18.0}}
    rows = [dict(strategy='old_old', change=4.75, error=-1.0), dict(strategy='old_new', change=8.25, error=2.5),
            dict(strategy='new_new', change=6.0, error=0.25)]
    return dict(timepoints=timepoints, volumes=volumes,
                table=dict(reference_change=5.75, rows=rows, retention_cost=2.25), window_hu=(-160.0, 240.0))


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

    def test_text_one_pixel_past_the_page_edge_is_reported(self):
        # A text box 1 px past the edge loses its descenders in print; text flush with the edge is kept whole.
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(2, 1), dpi=100)
        fig.text(0.5, -1 / 100, 'cut', ha='center', va='bottom', fontsize=8)
        fig.text(0.0, 0.5, 'flush', ha='left', va='center', fontsize=8)
        self.assertEqual(self.f.text_outside_bounds(fig), ['cut'])
        plt.close(fig)

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

    def test_update_explainer_shows_the_identity_and_every_strategy(self):
        fig = self.f.update_explainer()
        self.check_print_size(fig)
        texts = ' '.join(self.f.figure_texts(fig))
        for word in ('Baseline scan', 'Follow-up scan', 'Reference', 'Old model', 'Updated model', 'old/old', 'old/new',
                     'new/new', 'Retention cost', 'Update benefit', 'Contour benefit', 'BL(new) − BL(old)'):
            self.assertIn(word, texts)
        # Two labelled parts: the mechanism and the strategies.
        self.assertIn('a', self.f.figure_texts(fig))
        self.assertIn('b', self.f.figure_texts(fig))

    def test_update_explainer_shades_the_version_difference_and_highlights_its_term(self):
        from matplotlib.path import Path as MplPath
        fig = self.f.update_explainer()
        drawn = {a.get_gid(): a for a in fig.findobj(lambda a: str(a.get_gid()).startswith('explainer:'))}
        for scan in ('baseline', 'followup'):
            self.assertIn(f'explainer:scan:{scan}', drawn)
            for layer in ('reference', 'old', 'new'):
                self.assertIn(f'explainer:outline:{scan}:{layer}', drawn)

        def outline(scan, layer):
            line = drawn[f'explainer:outline:{scan}:{layer}']
            return np.column_stack([line.get_xdata(), line.get_ydata()])

        def area(points):
            x, y = points[:, 0], points[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Lesions are drawn as organic outlines, not circles.
        reference = outline('baseline', 'reference')
        radius = np.hypot(*(reference - reference.mean(axis=0)).T)
        self.assertGreater(radius.max() / radius.min(), 1.1)
        # The shaded band is exactly the region between the two versions' baseline outlines.
        old, new = outline('baseline', 'old'), outline('baseline', 'new')
        self.assertTrue(MplPath(new).contains_points(old).all())
        band = drawn['explainer:version_difference']
        band_area = sum(area(p) * (1 if k == 0 else -1) for k, p in enumerate(band.get_path().to_polygons()))
        self.assertAlmostEqual(band_area, area(new) - area(old), delta=0.01 * area(new))
        # The highlighted term of the identity carries the band's colour.
        term = [t for t in fig.findobj(Text) if t.get_text() == '[BL(new) − BL(old)]']
        self.assertEqual(len(term), 1)
        self.assertEqual(tuple(term[0].get_bbox_patch().get_facecolor()), tuple(band.get_facecolor()))
        # One palette: each strategy badge has the colour of that version's outline.
        for version, label in (('old', 'Old model'), ('new', 'Updated model')):
            badges = [a for a in fig.findobj(lambda a: str(a.get_gid()).startswith(f'explainer:badge:{version}'))]
            self.assertEqual(len(badges), 3)
            colour = matplotlib.colors.to_rgba(drawn[f'explainer:outline:baseline:{version}'].get_color())
            self.assertTrue(all(tuple(b.get_edgecolor()) == colour for b in badges), label)

    def test_update_explainer_key_entries_and_labels_keep_clear(self):
        fig = self.f.update_explainer()
        renderer = fig.canvas.get_renderer()
        key = {a.get_gid(): a.get_window_extent(renderer)
               for a in fig.findobj(lambda a: str(a.get_gid()).startswith('key:'))}
        names = ('Reference', 'Version difference', 'Old model', 'Updated model')
        self.assertEqual(set(key), {f'key:{n}:{part}' for n in names for part in ('handle', 'label')})
        inch = fig.dpi
        for name in names:
            self.assertGreaterEqual((key[f'key:{name}:label'].x0 - key[f'key:{name}:handle'].x1) / inch, 0.04)
        for first, second in zip(names, names[1:]):
            label, handle = key[f'key:{first}:label'], key[f'key:{second}:handle']
            if abs(label.y0 + label.y1 - handle.y0 - handle.y1) / 2 < 2:  # same row
                self.assertGreaterEqual((handle.x0 - label.x1) / inch, 0.15, f'{first} runs into {second}')
        # Every label is placed by hand, so none may overlap another.
        texts = [(t.get_text(), t.get_window_extent(renderer)) for t in self.f._visible_texts(fig)]
        for i, (one, a) in enumerate(texts):
            for two, b in texts[i + 1:]:
                overlap = min(min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0))
                self.assertLessEqual(overlap, 1.0, f'{one!r} overlaps {two!r}')

    def test_model_panels_take_every_value_from_the_results(self):
        unet, nnunet = replication(20, 1.0), replication(10, 0.5)
        panels = self.f.model_comparison_panels(unet, unet, nnunet, nnunet)
        for name in CONTRASTS:
            for cohort, _ in COHORTS:
                for model, source in (('unet', unet), ('nnunet', nnunet)):
                    want = source['cohorts'][cohort]['primary_contrasts'][name]
                    got = panels['primary'][name][cohort][model]
                    self.assertEqual(got['estimate'], want['mean'])
                    self.assertEqual(tuple(got['interval']), tuple(want['two_way_family_interval']))
                    self.assertEqual(list(got['runs']), want['per_run_means'])
        row = panels['strata']['internal'][2]
        self.assertEqual(row['label'], '≥10 mL\n13 / 21')
        want = nnunet['cohorts']['internal']['size_strata'][2]['retaining_baseline_cost']
        self.assertEqual(row['nnunet']['estimate'], want['mean'])
        self.assertEqual(tuple(row['nnunet']['interval']), tuple(want['two_way_percentile_95']))
        self.assertEqual([r['label'] for r in panels['strata']['pantrack']],
                         ['<1 mL\n25 / 205', '1–<10 mL\n19 / 51', '≥10 mL\n13 / 21'])

    def test_model_panels_reject_strata_that_differ_between_replications(self):
        # Both replications measured the same lesions, so each stratum must hold the same patients and pairs.
        unet, nnunet = replication(20, 1.0), replication(10, 0.5)
        nnunet['cohorts']['internal']['size_strata'][0]['pairs'] = 999
        with self.assertRaises(ValueError):
            self.f.model_comparison_panels(unet, unet, nnunet, nnunet)
        nnunet = replication(10, 0.5)
        del nnunet['cohorts']['pantrack']['size_strata'][1]['retaining_baseline_cost']['two_way_percentile_95']
        with self.assertRaises(ValueError):
            self.f.model_comparison_panels(unet, unet, nnunet, nnunet)

    def test_model_comparison_figure_draws_exactly_the_given_values(self):
        unet, nnunet = replication(20, 1.0), replication(10, 0.5)
        panels = self.f.model_comparison_panels(unet, unet, nnunet, nnunet)
        fig = self.f.model_comparison_figure(panels, COHORTS, MODEL_LEGEND)
        self.check_print_size(fig)
        drawn = {a.get_gid(): a for a in fig.findobj(lambda a: a.get_gid() is not None)}
        for name in CONTRASTS:
            for cohort, _ in COHORTS:
                for model in ('unet', 'nnunet'):
                    p = panels['primary'][name][cohort][model]
                    key = f'primary:{name}:{cohort}:{model}'
                    self.assertAlmostEqual(drawn[f'{key}:estimate'].get_offsets()[0][0], p['estimate'])
                    self.assertEqual(tuple(drawn[f'{key}:interval'].get_xdata()), tuple(p['interval']))
                    self.assertEqual([x for x, _ in drawn[f'{key}:runs'].get_offsets()], list(p['runs']))
        for cohort, _ in COHORTS:
            for k, row in enumerate(panels['strata'][cohort]):
                for model in ('unet', 'nnunet'):
                    key = f'strata:{cohort}:{k}:{model}'
                    self.assertAlmostEqual(drawn[f'{key}:estimate'].get_offsets()[0][0], row[model]['estimate'])
                    self.assertEqual(tuple(drawn[f'{key}:interval'].get_xdata()), tuple(row[model]['interval']))

    def test_model_comparison_keeps_the_retention_first_order_and_distinct_models(self):
        unet, nnunet = replication(20, 1.0), replication(10, 0.5)
        fig = self.f.model_comparison_figure(self.f.model_comparison_panels(unet, unet, nnunet, nnunet), COHORTS,
                                             MODEL_LEGEND)
        titles = [ax.get_title(loc='left') for ax in fig.axes if ax.get_title(loc='left')]
        self.assertEqual(titles[:3], ['Retention cost', 'Update benefit', 'Contour benefit'])
        self.assertEqual(titles[3:], ['Internal', 'PanTrack'])
        # The strata are post hoc for the U-Net but were specified before nnU-Net training, so (b) says descriptive.
        heading = [t for t in self.f.figure_texts(fig) if t.startswith('b ')]
        self.assertEqual(len(heading), 1)
        self.assertIn('descriptive', heading[0])
        self.assertNotIn('post hoc', heading[0])
        handles = fig.legends[0].legend_handles
        self.assertEqual(len(handles), 4)
        estimates = [h for h in handles if 'interval' in h.get_label()]
        self.assertEqual(len(estimates), 2)
        self.assertNotEqual(estimates[0].get_marker(), estimates[1].get_marker())
        self.assertNotEqual(estimates[0].get_color(), estimates[1].get_color())

    def test_example_figure_draws_every_contour_and_prints_the_given_values(self):
        data = example()
        fig = self.f.example_figure(data)
        self.check_print_size(fig)
        drawn = {a.get_gid() for a in fig.findobj(lambda a: str(a.get_gid()).startswith('example:'))}
        for timepoint in ('bl', 'fu'):
            self.assertIn(f'example:image:{timepoint}', drawn)
            for layer in ('reference', 'old', 'new'):
                self.assertIn(f'example:contour:{timepoint}:{layer}', drawn)
        texts = {t.get_gid(): t.get_text() for t in fig.findobj(Text) if str(t.get_gid()).startswith('example:')}
        self.assertEqual(texts['example:volume:bl:reference'], '12.50 mL')
        self.assertEqual(texts['example:volume:fu:old'], '14.50 mL')
        self.assertEqual(texts['example:change:old_new'], '+8.25')
        self.assertEqual(texts['example:error:old_old'], '−1.00')
        self.assertEqual(texts['example:reference_change'], '+5.75')
        self.assertIn('+2.25 mL', texts['example:retention_cost'])
        strategies = [t.get_text() for t in sorted((t for t in fig.findobj(Text)
                                                    if str(t.get_gid()).startswith('example:strategy:')),
                                                   key=lambda t: -t.get_position()[1])]
        self.assertEqual(strategies, ['old/old', 'old/new', 'new/new'])

    def test_example_inputs_are_read_from_the_runner_arrays(self):
        import json
        data = example()
        arrays = {}
        for timepoint, view in data['timepoints'].items():
            for layer in ('image', 'reference', 'old', 'new'):
                arrays[f'{timepoint}_{layer}'] = view[layer]
            arrays[f'{timepoint}_spacing_mm'] = np.asarray(view['spacing'])
        arrays['values_json'] = np.array(json.dumps(dict(volumes=data['volumes'], table=data['table'],
                                                         window_hu=list(data['window_hu']), crop_side_mm=60.0,
                                                         seed=20261205, cohort='pantrack')))
        read = self.f.example_from_arrays(arrays)
        self.assertEqual(read['volumes'], data['volumes'])
        self.assertEqual(read['table'], data['table'])
        self.assertEqual(tuple(read['window_hu']), data['window_hu'])
        self.assertEqual(read['timepoints']['fu']['spacing'], (0.75, 0.75))
        self.assertTrue(np.array_equal(read['timepoints']['bl']['old'], data['timepoints']['bl']['old']))
        del arrays['fu_new']
        with self.assertRaises(ValueError):
            self.f.example_from_arrays(arrays)

    def test_example_figure_rejects_an_empty_or_misshapen_layer(self):
        data = example()
        data['timepoints']['fu']['new'] = np.zeros((80, 80), dtype=bool)
        with self.assertRaises(ValueError):
            self.f.example_figure(data)
        data = example()
        data['timepoints']['bl']['old'] = np.ones((70, 80), dtype=bool)
        with self.assertRaises(ValueError):
            self.f.example_figure(data)

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

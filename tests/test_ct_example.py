"""Synthetic contracts for CT-UPDATE-018: choosing and preparing one example lesion pair (illustration only)."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from careport import ct_example as ex
from careport.ct_update_gate import GateError

POLICY = Path(__file__).resolve().parents[1]/'config/preanalysis/ct_example_018_2026-10-09.json'
STRATUM = dict(name='ge10', lower_ml=10.0, upper_ml=None)


def row(patient, pair_id, timepoint, reference, old, new, empty=()):
    """One stored evaluation record, shaped like the replication's private prediction rows."""
    predictions = {}
    for model, volume in (('old', old), ('new', new), ('continued', new)):
        predictions[model] = dict(volume_ml=volume, dice=0.8, hd95_mm=2.0, empty_prediction=model in empty)
    return dict(patient=patient, pair_id=pair_id, scan=f'{patient}_{timepoint}', timepoint=timepoint,
                lesion_id=1, reference_volume_ml=reference, predictions=predictions)


def records():
    # (patient, pair, ref BL, ref FU, old BL, old FU, new BL, new FU); binary-exact values keep ties exact.
    pairs = [('p1', 'a', 12.0, 15.0, 12.0, 13.0, 12.5, 15.5),   # old/new error 15.5-12-3=+0.5, new/new 0 -> cost 0.5
             ('p1', 'b', 4.0, 5.0, 3.0, 4.0, 4.5, 5.0),        # below 10 mL: not in the stratum
             ('p2', 'a', 20.0, 18.0, 17.0, 16.0, 21.0, 19.0),  # old/new 19-17+2=+4, new/new 19-21+2=0 -> cost 4
             ('p3', 'a', 10.0, 11.0, 9.0, 9.5, 10.5, 11.25)]   # old/new 11.25-9-1=+1.25, new/new -0.25 -> cost 1
    rows = []
    for patient, pair, rb, rf, ob, of, nb, nf in pairs:
        rows.append(row(patient, pair, 'bl', rb, ob, nb))
        rows.append(row(patient, pair, 'fu', rf, of, nf))
    return rows


class PolicyTests(unittest.TestCase):
    def test_registered_policy_is_valid(self):
        policy = json.loads(POLICY.read_text(encoding='utf-8'))
        ex.validate_example_policy(policy)
        self.assertEqual(policy['protocol_id'], ex.PROTOCOL_ID)

    def test_execution_resources_are_the_ones_the_017_jobs_used(self):
        # The cluster names its MIG profile in full; job 58269 asked for a shorter name and could never start.
        execution = json.loads(POLICY.read_text(encoding='utf-8'))['execution']
        strong = json.loads((POLICY.parent/'ct_strong_017_2026-10-02.json').read_text(encoding='utf-8'))['execution']
        self.assertEqual(execution['partition'], strong['partition'])
        self.assertEqual(execution['gres'], strong['gres'])
        self.assertEqual(execution['cpus'], strong['cpus_per_task'])
        self.assertEqual(f"{execution['memory_gib']}G", strong['memory'])

    def test_policy_changes_are_rejected(self):
        policy = json.loads(POLICY.read_text(encoding='utf-8'))
        for change in (dict(cohort='internal'), dict(models=['old', 'continued']),
                       dict(permission_required_for_submission=False), dict(patient_level_values_in_record=True),
                       dict(amendment_sha256='abc'), dict(stratum=dict(name='ge10', lower_ml=5.0, upper_ml=None)),
                       dict(check=dict(volume_relative_tolerance=0.2, dice_absolute_tolerance=0.005))):
            changed = copy.deepcopy(policy)
            changed.update(change)
            with self.assertRaises(GateError, msg=str(change)):
                ex.validate_example_policy(changed)


class SelectionTests(unittest.TestCase):
    def test_run_closest_to_the_mean_with_ties_to_the_earlier_seed(self):
        seed, value, mean = ex.choose_run([1.0, 2.0, 3.0, 2.0], [11, 12, 13, 14])
        self.assertEqual((seed, value, mean), (12, 2.0, 2.0))
        # 1.5 and 2.5 are equally close to the mean 2.0: the earlier seed wins.
        self.assertEqual(ex.choose_run([2.5, 1.5, 2.0 + 0.0], [30, 20, 40])[0], 40)
        self.assertEqual(ex.choose_run([2.5, 1.5], [30, 20])[0], 20)
        with self.assertRaises(GateError):
            ex.choose_run([1.0, 2.0], [11])
        with self.assertRaises(GateError):
            ex.choose_run([1.0, 2.0], [11, 11])

    def test_pairs_of_the_stratum_with_their_retention_cost(self):
        pairs = ex.example_pairs(records(), STRATUM)
        self.assertEqual([(p['patient'], p['pair_id']) for p in pairs], [('p1', 'a'), ('p2', 'a'), ('p3', 'a')])
        self.assertEqual([round(p['cost'], 9) for p in pairs], [0.5, 4.0, 1.0])
        self.assertFalse(any(p['empty'] for p in pairs))

    def test_pair_closest_to_the_run_value_skipping_empty_predictions(self):
        rows = records()
        self.assertEqual(ex.choose_pair(ex.example_pairs(rows, STRATUM), 0.9)['patient'], 'p3')
        self.assertEqual(ex.choose_pair(ex.example_pairs(rows, STRATUM), 3.0)['patient'], 'p2')
        # An empty prediction from either model at either timepoint removes the pair from consideration.
        for r in rows:
            if r['patient'] == 'p3' and r['timepoint'] == 'fu':
                r['predictions']['old']['empty_prediction'] = True
        self.assertEqual(ex.choose_pair(ex.example_pairs(rows, STRATUM), 0.9)['patient'], 'p1')

    def test_ties_go_to_the_first_pair_in_patient_and_pair_order(self):
        pairs = ex.example_pairs(records(), STRATUM)
        # 0.75 lies exactly between p1 (0.5) and p3 (1.0).
        self.assertEqual(ex.choose_pair(pairs, 0.75)['patient'], 'p1')


class DisplayTests(unittest.TestCase):
    def test_largest_area_slice_with_ties_nearest_the_centroid(self):
        mask = np.zeros((5, 5, 7), dtype=bool)
        mask[1:4, 1:4, 2] = True   # area 9
        mask[1:4, 1:4, 5] = True   # area 9
        mask[2, 2, 6] = True       # pulls the centroid upward
        self.assertEqual(ex.largest_area_slice(mask), 5)
        mask[0:5, 0:5, 1] = True   # area 25, the unique largest
        self.assertEqual(ex.largest_area_slice(mask), 1)
        with self.assertRaises(GateError):
            ex.largest_area_slice(np.zeros((3, 3, 3), dtype=bool))

    def test_canonical_orientation_follows_the_affine(self):
        array = np.arange(2*3*4).reshape(2, 3, 4)
        # Array axes are (z, x, y), with x pointing left and y pointing posterior.
        affine = np.array([[0., -0.7, 0., 0.], [0., 0., -0.8, 0.], [2.0, 0., 0., 0.], [0., 0., 0., 1.]])
        canonical, spacing = ex.canonical(array, affine)
        self.assertEqual(canonical.shape, (3, 4, 2))
        self.assertEqual(spacing, (0.7, 0.8, 2.0))
        # Canonical index (i, j, k) increases toward the right, anterior and superior.
        self.assertEqual(canonical[0, 0, 0], array[0, 2, 3])
        self.assertEqual(canonical[2, 3, 1], array[1, 0, 0])
        with self.assertRaises(GateError):
            ex.canonical(array, np.diag([1., 1., 1., 1.])[[0, 0, 2, 3]])

    def test_radiological_display_puts_anterior_up_and_patient_right_on_the_left(self):
        axial = np.zeros((4, 3))      # indexed (x toward the right, y toward anterior)
        axial[3, 2] = 1               # most right and most anterior
        shown = ex.radiological(axial)
        self.assertEqual(shown.shape, (3, 4))
        self.assertEqual(shown[0, 0], 1)

    def test_crop_is_square_the_same_size_at_both_timepoints_and_inside_the_image(self):
        first = np.zeros((100, 100), dtype=bool)
        first[40:50, 40:60] = True     # 20 pixel columns at 1 mm: 20 mm
        second = np.zeros((80, 120), dtype=bool)
        second[2:12, 5:15] = True      # near the corner: the window must move inside
        side, windows = ex.crop_windows([first, second], [(1.0, 1.0), (0.5, 0.5)], 40.0, 1.6)
        self.assertAlmostEqual(side, 40.0)  # 1.6 x 20 mm = 32 mm, below the 40 mm minimum
        r0, r1, c0, c1 = windows[0]
        self.assertEqual((r1 - r0, c1 - c0), (40, 40))
        self.assertEqual((r0, c0), (25, 30))
        r0, r1, c0, c1 = windows[1]
        self.assertEqual((r1 - r0, c1 - c0), (80, 80))   # 40 mm at 0.5 mm
        self.assertEqual((r0, c0), (0, 0))
        big = np.zeros((100, 100), dtype=bool)
        big[10:90, 30:40] = True       # 80 mm extent: 1.6 x 80 = 128 mm, larger than the image
        side, windows = ex.crop_windows([big, big], [(1.0, 1.0), (1.0, 1.0)], 40.0, 1.6)
        self.assertAlmostEqual(side, 128.0)
        self.assertEqual(windows[0], (0, 100, 0, 100))

    def test_window_limits(self):
        self.assertEqual(ex.window_limits(400.0, 40.0), (-160.0, 240.0))


class TableTests(unittest.TestCase):
    def test_change_table_matches_the_analysis_errors(self):
        pair = ex.example_pairs(records(), STRATUM)[2]   # p3
        table = ex.change_table(pair)
        self.assertAlmostEqual(table['reference_change'], 1.0)
        self.assertEqual([r['strategy'] for r in table['rows']], ['old_old', 'old_new', 'new_new'])
        changes = {r['strategy']: (round(r['change'], 9), round(r['error'], 9)) for r in table['rows']}
        self.assertEqual(changes, {'old_old': (0.5, -0.5), 'old_new': (2.25, 1.25), 'new_new': (0.75, -0.25)})
        self.assertAlmostEqual(table['retention_cost'], 1.0)

    def test_agreement_with_the_stored_evaluation(self):
        stored = {('bl', 'old'): (10.0, 0.80), ('fu', 'new'): (20.0, 0.90)}
        close = {('bl', 'old'): (10.04, 0.803), ('fu', 'new'): (19.95, 0.896)}
        report = ex.agreement(stored, close, 0.005, 0.005)
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['max_relative_volume_difference'], 0.004)
        far = dict(close)
        far[('fu', 'new')] = (19.0, 0.90)
        self.assertFalse(ex.agreement(stored, far, 0.005, 0.005)['passed'])
        self.assertFalse(ex.agreement(stored, {('bl', 'old'): (10.0, 0.8)}, 0.005, 0.005)['passed'])


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunnerSelectionTests(unittest.TestCase):
    """The runner reads the chosen run only through the verified results' receipt and prediction hashes."""

    def setUp(self):
        import tempfile
        from careport import ct_example_run
        self.run = ct_example_run
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        seeds, per_run = [101, 102, 103], [0.5, 1.25, 2.0]    # mean 1.25: run 102
        folder = self.root/'run_102'/'evaluation'
        folder.mkdir(parents=True)
        self.predictions = folder/'pantrack_predictions_private.jsonl'
        self.predictions.write_bytes(b''.join(json.dumps(r).encode() + b'\n' for r in records()))
        receipt = dict(status='STRONG_SEED_COMPLETE', seed=102,
                       cohorts=dict(pantrack=dict(predictions_sha256=sha256(self.predictions))))
        (folder/'seed_complete.json').write_text(json.dumps(receipt))
        strata = [dict(stratum='lt1', lower_ml=0.0, upper_ml=1.0,
                       retaining_baseline_cost=dict(mean=0.0, per_run_means=[0.0, 0.0, 0.0])),
                  dict(stratum='ge10', lower_ml=10.0, upper_ml=None,
                       retaining_baseline_cost=dict(mean=1.25, per_run_means=per_run))]
        self.results = dict(seeds=seeds, seed_receipt_sha256={'101': '0' * 64, '102': sha256(folder/'seed_complete.json'),
                                                              '103': '0' * 64},
                            descriptive=dict(seeds=seeds, cohorts=dict(pantrack=dict(size_strata=strata))))
        self.policy = json.loads(POLICY.read_text(encoding='utf-8'))

    def tearDown(self):
        self.tmp.cleanup()

    def fresh(self):
        """A clean fixture within one test: the previous temporary folder is removed first."""
        self.tmp.cleanup()
        self.setUp()

    def test_selects_the_registered_run_and_pair(self):
        choice = self.run.select_example(self.results, self.policy, self.root)
        self.assertEqual((choice['seed'], choice['run_value']), (102, 1.25))
        self.assertEqual((choice['pair']['patient'], choice['pair']['pair_id']), ('p3', 'a'))  # cost 1.0
        self.assertEqual((choice['pairs_in_stratum'], choice['eligible_pairs']), (3, 3))
        self.assertEqual(set(choice['stored']), {('bl', 'old'), ('bl', 'new'), ('fu', 'old'), ('fu', 'new')})
        self.assertEqual(choice['stored'][('fu', 'new')], (11.25, 0.8))

    def test_refuses_changed_predictions_receipts_or_strata(self):
        self.predictions.write_bytes(self.predictions.read_bytes() + b'\n')
        with self.assertRaises(GateError):
            self.run.select_example(self.results, self.policy, self.root)
        self.fresh()
        self.results['seed_receipt_sha256']['102'] = '1' * 64
        with self.assertRaises(GateError):
            self.run.select_example(self.results, self.policy, self.root)
        self.fresh()
        self.results['descriptive']['cohorts']['pantrack']['size_strata'][1]['lower_ml'] = 9.0
        with self.assertRaises(GateError):
            self.run.select_example(self.results, self.policy, self.root)
        self.fresh()
        self.results['descriptive']['seeds'] = [101, 103, 102]
        with self.assertRaises(GateError):
            self.run.select_example(self.results, self.policy, self.root)

    def test_display_crops_follow_the_registered_slice_crop_and_orientation(self):
        shape, affine = (40, 30, 12), np.diag([0.8, 1.0, 2.0, 1.0])
        image = np.broadcast_to(np.arange(shape[0], dtype=float)[:, None, None], shape).copy()
        views = {}
        for timepoint, (z, radius) in (('bl', (5, 4)), ('fu', (6, 6))):
            x, y, k = np.ogrid[:shape[0], :shape[1], :shape[2]]
            reference = ((x - 28) ** 2 + (y - 12) ** 2 <= radius ** 2) & (np.abs(k - z) <= 1)
            reference[:, :, z] |= ((x - 28) ** 2 + (y - 12) ** 2 <= (radius + 1) ** 2)[:, :, 0]  # largest at z
            views[timepoint] = dict(image=image, affine=affine,
                                    masks=dict(reference=reference, old=reference & (x < 28), new=reference))
        display = dict(crop_minimum_mm=10.0, crop_factor=1.6, window_width_hu=400.0, window_level_hu=40.0)
        arrays, placement, side = self.run.display_crops(views, display)
        self.assertEqual((placement['bl']['slice'], placement['fu']['slice']), (5, 6))
        self.assertAlmostEqual(side, 1.6 * 15 * 1.0)   # widest extent: 15 rows of 1.0 mm at follow-up
        for timepoint in ('bl', 'fu'):
            crops = {name: arrays[f'{timepoint}_{name}'] for name in ('image', 'reference', 'old', 'new')}
            self.assertEqual(len({c.shape for c in crops.values()}), 1)
            self.assertTrue(crops['reference'].any() and crops['image'].dtype == np.float32)
            self.assertEqual(tuple(arrays[f'{timepoint}_spacing_mm']), (1.0, 0.8))
        # Radiological: the patient's right (high x, bright in this image) is on the viewer's left.
        self.assertGreater(arrays['bl_image'][:, 0].mean(), arrays['bl_image'][:, -1].mean())

    def test_environment_must_bind_the_registered_strong_model_files(self):
        environment = dict(CT_STRONG_POLICY_SHA256=self.policy['strong_policy_sha256'],
                           CT_STRONG_FIX_SHA256=self.policy['strong_fix_sha256'], CT_STRONG_SOURCE_SHA256='a' * 64)
        self.run.check_strong_bindings(self.policy, environment, 'a' * 64)
        for key in environment:
            changed = dict(environment, **{key: 'b' * 64})
            with self.assertRaises(GateError, msg=key):
                self.run.check_strong_bindings(self.policy, changed, 'a' * 64)


if __name__ == '__main__':
    unittest.main()

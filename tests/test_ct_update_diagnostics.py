"""Synthetic diagnostic arithmetic; no clinical inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from careport.ct_pantrack_confirm import MODELS
from careport.ct_update_gate import GateError
from careport.ct_update_diagnostics import diagnose, pair_metrics, size_bin, verify_reproduction
from careport.ct_update_diagnostic_run import read_prediction_records, validate_policy, validate_seed_receipt


def row(patient, pair, timepoint, *, reference=10., old=10., new=10., scan=None):
    return dict(patient=patient, pair_id=pair, timepoint=timepoint,
                scan=scan or pair + timepoint, lesion_id=1, reference_volume_ml=reference,
                predictions={m: dict(volume_ml=new if m == 'new' else old,
                                     dice=.8, hd95_mm=1., empty_prediction=False)
                             for m in MODELS})


def fixture():
    return [row('a', 'a1', 'bl', old=8., new=11.),
            row('a', 'a1', 'fu', reference=12., old=10., new=15.),
            row('b', 'b1', 'bl'), row('b', 'b1', 'fu')]


def run(rows, seeds=None):
    return diagnose(seeds or {1: rows}, rows)


class DiagnosticTests(unittest.TestCase):
    def test_hash_gate_precedes_prediction_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.jsonl'
            path.write_bytes(b'not valid JSON')
            with self.assertRaises(GateError):
                read_prediction_records(path, '0' * 64)
            raw = b'{"synthetic": true}\n'
            path.write_bytes(raw)
            self.assertEqual(read_prediction_records(path, hashlib.sha256(raw).hexdigest()),
                             [dict(synthetic=True)])

    def test_policy_and_receipt_fail_closed(self):
        policy_path = Path('config/preanalysis/ct_diagnostics_013_2026-09-24.json')
        policy = json.loads(policy_path.read_text())
        validate_policy(policy)
        for key, value in [('internal_test_values_permitted', True),
                           ('seeds', [20260915]), ('post_hoc', False)]:
            with self.subTest(key=key), self.assertRaises(GateError):
                validate_policy(dict(policy, **{key: value}))
        receipt_path = Path('newpaper/research_selection_2026-09-13/CT_PANTRACK_SEED_20260915_COMPLETION_2026-09-19.json')
        receipt = json.loads(receipt_path.read_text())
        validate_seed_receipt(receipt, 20260915, policy)
        for key, value in [('status', 'INCOMPLETE'), ('seed', 20260917),
                           ('index_sha256', '0' * 64), ('internal_test_values_read', True)]:
            with self.subTest(key=key), self.assertRaises(GateError):
                validate_seed_receipt(dict(receipt, **{key: value}), 20260915, policy)

    def test_signed_and_absolute_identity_with_cancellation(self):
        values = pair_metrics(fixture()[0], fixture()[1], 'new')
        self.assertEqual(values['mixed_signed_ml'], 5.)
        self.assertEqual(values['same_signed_ml'], 2.)
        self.assertEqual(values['baseline_shift_ml'], 3.)
        self.assertEqual(values['followup_shift_ml'], 5.)
        self.assertEqual(values['mixed_minus_old_ml'], 5.)
        self.assertEqual(values['mixed_minus_same_ml'], 3.)
        self.assertEqual(values['same_minus_old_ml'], 2.)
        # A positive baseline shift can cancel a negative same-version error.
        bl = row('a', 'x', 'bl', old=10., new=13.)
        fu = row('a', 'x', 'fu', old=10., new=11.)
        self.assertEqual(pair_metrics(bl, fu, 'new')['mixed_minus_same_ml'], -1.)

    def test_patient_weighting_and_private_complete_contributions(self):
        rows = fixture()
        rows += [row('a', 'a2', 'bl', old=8., new=11.),
                 row('a', 'a2', 'fu', reference=12., old=10., new=15.)]
        public, private = run(rows)
        self.assertEqual(public['seed_averaged']['models']['new']['mixed_mae_ml'], 2.5)
        self.assertEqual(public['patients'], 2)
        self.assertEqual(public['pairs'], 3)
        self.assertEqual(len(private['patient_contributions']), 2 * len(MODELS))
        self.assertNotIn('patient_contributions', public)

    def test_absolute_loss_precedes_seed_average(self):
        rows = [row('a', 'a1', 'bl'), row('a', 'a1', 'fu', new=14.)]
        other = copy.deepcopy(rows)
        other[1]['predictions']['new']['volume_ml'] = 6.
        public, _ = run(rows, {1: rows, 2: other})
        value = public['seed_averaged']['models']['new']
        self.assertEqual(value['mixed_mae_ml'], 4.)
        self.assertEqual(value['mixed_signed_ml'], 0.)

    def test_size_bins_fixed_and_weighted_within_stratum(self):
        self.assertEqual([size_bin(x) for x in [.9, 1., 9.9, 10.]],
                         ['lt1', '1to10', '1to10', 'ge10'])
        public, _ = run(fixture())
        self.assertEqual(public['seed_averaged']['size_strata']['lt1']['patients'], 0)
        self.assertIsNone(public['seed_averaged']['size_strata']['lt1']['models'])
        self.assertEqual(public['seed_averaged']['size_strata']['ge10']['pairs'], 2)

    def test_repeated_scan_deduplicated_and_inconsistent_metrics_rejected(self):
        rows = [row('a', 'a1', 'bl', scan='first'),
                row('a', 'a1', 'fu', scan='shared'),
                row('a', 'a2', 'bl', scan='shared'),
                row('a', 'a2', 'fu', scan='last')]
        public, _ = run(rows)
        self.assertEqual(public['unique_image_lesions'], 3)
        rows[2]['predictions']['new']['volume_ml'] = 11.
        with self.assertRaises(GateError):
            run(rows)

    def test_missing_timepoint_duplicate_or_reference_drift_fails(self):
        rows = fixture()
        for bad in [rows[:-1], rows + [rows[0]]]:
            with self.subTest(count=len(bad)), self.assertRaises(GateError):
                run(rows, {1: bad})
        bad = copy.deepcopy(rows)
        bad[0]['reference_volume_ml'] += 1
        with self.assertRaises(GateError):
            run(rows, {1: rows, 2: bad})

    def test_influence_does_not_remove_any_patient(self):
        public, _ = run(fixture())
        value = public['seed_averaged']['new_model_influence']['mixed_minus_old_ml']
        self.assertEqual(value['positive_patients'], 1)
        self.assertEqual(value['zero_patients'], 1)
        self.assertEqual(value['leave_one_out_mean_range_ml'], [0., 5.])
        self.assertEqual(value['top1_absolute_contribution_share'], 1.)
        self.assertEqual(public['seed_averaged']['models']['new']['mixed_minus_old_ml'], 2.5)

    def test_reproduction_checks_all_models_and_rejects_mismatch(self):
        public, _ = run(fixture())
        batches = [public['seed_averaged']] + public['per_seed']
        originals = []
        for batch in batches:
            models = batch['models']
            strategies = {'old_old': dict(mae_ml=models['old']['same_mae_ml'],
                                         signed_error_ml=models['old']['same_signed_ml'])}
            for model in MODELS[1:]:
                for name, prefix in [('old_' + model, 'mixed'), (model + '_' + model, 'same')]:
                    strategies[name] = dict(mae_ml=models[model][prefix + '_mae_ml'],
                                            signed_error_ml=models[model][prefix + '_signed_ml'])
            originals.append(dict(strategies=strategies,
                                  patient_mean_followup_dice={m: models[m]['followup_dice'] for m in MODELS}))
        originals[1]['training_seed'] = 1
        original = dict(seed_averaged=originals[0], per_seed=originals[1:])
        self.assertEqual(verify_reproduction(public, original)['values_checked'], 86)
        original['seed_averaged']['strategies']['old_new']['mae_ml'] += .1
        with self.assertRaises(GateError):
            verify_reproduction(public, original)


if __name__ == '__main__':
    unittest.main()

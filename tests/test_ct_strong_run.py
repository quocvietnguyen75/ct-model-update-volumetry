"""Synthetic gates for executing CT-UPDATE-017 (context, schedule record and run receipts)."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from careport.ct_update_gate import GateError

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'config/preanalysis/ct_strong_017_2026-10-02.json'
FIX = 'config/preanalysis/ct_strong_017_fix_a_2026-10-06.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_tree(target: Path) -> Path:
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'config/preanalysis').glob('ct_*.json')]
    names += ['newpaper/plan/CT_EXPERIMENT_AMENDMENT_015_2026-09-25.md',
              'newpaper/plan/CT_EXPERIMENT_AMENDMENT_017_2026-10-02.md',
              'newpaper/plan/CT_EXPERIMENT_AMENDMENT_017A_2026-10-06.md']
    for name in names:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target/name)
    (target/'EXECUTION_SHA256SUMS').write_text(''.join(f'{sha256(target/n)}  {n}\n' for n in sorted(names)),
                                               encoding='utf-8', newline='\n')
    return target


class ContextTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_strong_run
        self.m = ct_strong_run
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.code = code_tree(self.tmp/'code')
        self.project = self.tmp/'project'
        for sub in ('data/longitudinal-ct', 'data/pantrack'):
            (self.project/sub).mkdir(parents=True)
        replication = json.loads((ROOT/'config/preanalysis/ct_replication_015_2026-09-25.json').read_text())
        root = self.project/'runs/strong_017'
        self.env = {'CT_STRONG_POLICY_SHA256': sha256(self.code/POLICY),
                    'CT_STRONG_SOURCE_SHA256': sha256(self.code/'EXECUTION_SHA256SUMS'),
                    'CT_UPDATE_MODEL_FREEZE_SHA256': replication['pantrack_model_freeze_sha256'],
                    'CT_UPDATE_REMOTE_ROOT': str(self.project),
                    'CT_UPDATE_REMOTE_DATA_ROOT': str(self.project/'data/longitudinal-ct'),
                    'CT_UPDATE_PANTRACK_REMOTE_ROOT': str(self.project/'data/pantrack'),
                    'nnUNet_raw': str(root/'nnUNet_raw'), 'nnUNet_preprocessed': str(root/'nnUNet_preprocessed')}

    def load(self, **changes):
        # Inside a cluster job the real correction binding is set; a test sees it only when it passes one itself.
        environment = {k: v for k, v in os.environ.items() if k != 'CT_STRONG_FIX_SHA256'}
        environment.update(self.env, **changes)
        with mock.patch.dict(os.environ, environment, clear=True):
            return self.m.load_context(self.code/POLICY, code=self.code)

    def test_context_binds_policy_bundle_frozen_cohort_gates_and_nnunet_folders(self):
        ctx = self.load()
        self.assertEqual(ctx.root, (self.project/'runs/strong_017').resolve())
        self.assertEqual(ctx.replication.policy_sha, '3f629e057c05896d3bba9ffefdccb45e481c15cd9412e73199e4c9c789171820')
        self.assertEqual(ctx.policy['runs'], 10)

    def test_wrong_bindings_or_nnunet_folders_outside_the_run_root_are_rejected(self):
        for key, value in (('CT_STRONG_POLICY_SHA256', '0' * 64), ('CT_STRONG_SOURCE_SHA256', '0' * 64),
                           ('CT_UPDATE_MODEL_FREEZE_SHA256', '0' * 64), ('nnUNet_raw', str(self.tmp/'elsewhere')),
                           ('nnUNet_preprocessed', str(self.project/'runs/other'))):
            with self.subTest(key=key), self.assertRaises(GateError):
                self.load(**{key: value})


class ScheduleAndReceiptTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_strong_run
        self.m = ct_strong_run
        self.policy = json.loads((ROOT/POLICY).read_text())

    def schedule(self, seconds=(300.0, 120.0, 120.0), **changes):
        from careport.ct_strong import choose_schedule
        value = dict(choose_schedule(list(seconds), self.policy), status='STRONG_SCHEDULE_FIXED',
                     protocol_id='CT-UPDATE-017', policy_sha256='a' * 64, source_manifest_sha256='b' * 64,
                     outcomes_inspected=False)
        value.update(changes)
        return value

    def test_a_schedule_must_follow_from_its_own_timing_under_the_fixed_rule(self):
        self.assertEqual(self.m.validate_schedule(self.schedule(), self.policy, 'a' * 64, 'b' * 64)['schedule'], 'short')
        for changes in (dict(schedule='long'), dict(epochs=dict(old=500, new=250, continued=250)),
                        dict(policy_sha256='c' * 64), dict(outcomes_inspected=True), dict(status='RUNNING')):
            with self.subTest(changes=changes), self.assertRaises(GateError):
                self.m.validate_schedule(self.schedule(**changes), self.policy, 'a' * 64, 'b' * 64)

    def receipt(self, **changes):
        value = dict(status='STRONG_SEED_COMPLETE', protocol_id='CT-UPDATE-017', index=3, seed=20261203,
                     models=['old', 'new', 'continued'], policy_sha256='a' * 64, source_manifest_sha256='b' * 64,
                     schedule_sha256='d' * 64,
                     cohorts={'internal': dict(patients=29, pairs=277, timepoints=554, predictions_sha256='e' * 64),
                              'pantrack': dict(patients=45, pairs=181, timepoints=362, predictions_sha256='f' * 64)})
        value.update(changes)
        return value

    def test_run_receipts_bind_seed_policy_bundle_schedule_and_both_cohorts(self):
        self.m.validate_strong_receipt(self.receipt(), self.policy, 3, 'a' * 64, 'b' * 64, 'd' * 64)
        broken_cohort = copy.deepcopy(self.receipt())
        broken_cohort['cohorts']['pantrack']['pairs'] = 180
        for receipt in (self.receipt(seed=20261204), self.receipt(index=4), self.receipt(schedule_sha256='0' * 64),
                        self.receipt(source_manifest_sha256='0' * 64), self.receipt(models=['old', 'new']),
                        broken_cohort):
            with self.subTest(receipt=receipt), self.assertRaises(GateError):
                self.m.validate_strong_receipt(receipt, self.policy, 3, 'a' * 64, 'b' * 64, 'd' * 64)


class CorrectionTests(unittest.TestCase):
    """Correction A: re-evaluate registered checkpoints with the trained weights loaded, under explicit bindings."""

    def setUp(self):
        from careport import ct_strong_run
        self.m = ct_strong_run
        self.policy = json.loads((ROOT/POLICY).read_text())
        self.policy_sha = sha256(ROOT/POLICY)
        self.fix = json.loads((ROOT/FIX).read_text())
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_the_registered_binding_matches_the_policy_and_names_every_run(self):
        self.m.validate_fix(self.fix, self.policy, self.policy_sha)
        self.assertEqual(sorted(self.fix['training_complete_sha256']), [str(s) for s in self.policy['seeds']])
        self.assertIs(self.fix['retrain'], False)

    def test_incomplete_or_mismatched_bindings_are_rejected(self):
        def changed(**edits):
            fix = copy.deepcopy(self.fix)
            for path, value in edits.items():
                target = fix
                *parents, last = path.split('__')
                for key in parents:
                    target = target[key]
                target[last] = value
            return fix
        dropped = copy.deepcopy(self.fix)
        dropped['training_complete_sha256'].pop('20261210')
        cases = {'retrains': changed(retrain=True), 'other policy': changed(base_policy_sha256='0' * 64),
                 'missing run': dropped, 'bad hash': changed(schedule_sha256='not-a-hash'),
                 'no threshold': changed(sanity__minimum_mean_dice=0), 'unregistered seed': changed(sanity__seed=20261001),
                 'no patches': changed(sanity__patches=0),
                 'evaluation reused': changed(invalid_attempt__evaluation_folder_renamed_to='evaluation'),
                 'patient values': changed(patient_level_values_in_outputs=True)}
        for label, fix in cases.items():
            with self.subTest(case=label), self.assertRaises(GateError):
                self.m.validate_fix(fix, self.policy, self.policy_sha)
        with self.assertRaises(GateError):
            self.m.validate_fix(self.fix, self.policy, '0' * 64)

    def context(self, root, fix=None, source='b' * 64, training_source=None):
        from types import SimpleNamespace
        return SimpleNamespace(root=root, policy=self.policy, policy_sha='a' * 64, source_sha=source, fix=fix,
                               fix_sha=None if fix is None else 'f' * 64,
                               training_source_sha=training_source or source)

    def write_schedule(self, root, source):
        from careport.ct_strong import choose_schedule
        record = dict(choose_schedule([300.0, 120.0, 120.0], self.policy), status='STRONG_SCHEDULE_FIXED',
                      protocol_id='CT-UPDATE-017', policy_sha256='a' * 64, source_manifest_sha256=source,
                      outcomes_inspected=False)
        root.mkdir(parents=True, exist_ok=True)
        (root/'schedule.json').write_text(json.dumps(record))
        return sha256(root/'schedule.json')

    def test_schedule_binds_the_training_source_and_the_registered_hash_under_the_correction(self):
        root = self.tmp/'runs'
        schedule_sha = self.write_schedule(root, 'c' * 64)
        fix = dict(self.fix, schedule_sha256=schedule_sha)
        decision, sha = self.m._schedule(self.context(root, fix, source='b' * 64, training_source='c' * 64))
        self.assertEqual((decision['schedule'], sha), ('short', schedule_sha))
        with self.assertRaises(GateError):  # without the correction, the schedule must match the current bundle
            self.m._schedule(self.context(root, None, source='b' * 64))
        with self.assertRaises(GateError):  # under the correction, only the registered schedule is accepted
            self.m._schedule(self.context(root, dict(self.fix, schedule_sha256='0' * 64), 'b' * 64, 'c' * 64))

    def test_correction_evaluates_registered_training_only_after_the_invalid_evaluation_is_preserved(self):
        folder = self.tmp/'run_20261203'
        folder.mkdir()
        (folder/'training_complete.json').write_text('{"status": "THREE_PHASES_COMPLETE"}')
        fix = copy.deepcopy(self.fix)
        fix['training_complete_sha256']['20261203'] = sha256(folder/'training_complete.json')
        ctx = self.context(self.tmp, fix)
        with self.assertRaises(GateError):  # the invalid evaluation has not been renamed yet
            self.m.correction_ready(ctx, 20261203, folder)
        (folder/'evaluation_attempt002_untrained').mkdir()
        self.m.correction_ready(ctx, 20261203, folder)
        (folder/'evaluation').mkdir()
        with self.assertRaises(GateError):  # a leftover evaluation folder is never reused
            self.m.correction_ready(ctx, 20261203, folder)
        (folder/'evaluation').rmdir()
        (folder/'training_complete.json').write_text('{"status": "CHANGED"}')
        with self.assertRaises(GateError):  # training must be the registered one; nothing is retrained
            self.m.correction_ready(ctx, 20261203, folder)


class SanityGateTests(CorrectionTests):
    def test_evaluation_and_analysis_under_the_correction_need_a_passed_sanity_check_for_this_binding(self):
        ctx = self.context(self.tmp, copy.deepcopy(self.fix))
        with self.assertRaises(GateError):  # not run yet
            self.m._sanity_receipt(ctx)
        path = self.tmp/self.fix['sanity']['receipt_relative']
        for status, binding in (('STRONG_SANITY_FAIL', 'f' * 64), ('STRONG_SANITY_PASS', '0' * 64)):
            path.write_text(json.dumps(dict(status=status, correction_sha256=binding)))
            with self.subTest(status=status, binding=binding[:4]), self.assertRaises(GateError):
                self.m._sanity_receipt(ctx)
        path.write_text(json.dumps(dict(status='STRONG_SANITY_PASS', correction_sha256='f' * 64)))
        self.assertEqual(self.m._sanity_receipt(ctx), sha256(path))
        self.assertIsNone(self.m._sanity_receipt(self.context(self.tmp, None)))


class CorrectionContextTests(ContextTests):
    def test_the_correction_is_loaded_only_from_its_bound_file(self):
        self.assertIsNone(self.load().fix)
        fix_sha = sha256(self.code/FIX)
        ctx = self.load(CT_STRONG_FIX_SHA256=fix_sha)
        self.assertEqual(ctx.fix_sha, fix_sha)
        self.assertEqual(ctx.training_source_sha, ctx.fix['training_source_manifest_sha256'])
        self.assertNotEqual(ctx.training_source_sha, ctx.source_sha)
        with self.assertRaises(GateError):
            self.load(CT_STRONG_FIX_SHA256='0' * 64)
        amendment = self.code/ctx.fix['amendment_file']
        amendment.write_text(amendment.read_text() + 'changed\n')
        with self.assertRaises(GateError):
            self.load(CT_STRONG_FIX_SHA256=fix_sha)


if __name__ == '__main__':
    unittest.main()

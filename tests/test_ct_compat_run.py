"""Synthetic gates for executing CT-UPDATE-019 against the stored CT-UPDATE-015 and -017 outputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from careport.ct_update_gate import GateError

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'config/preanalysis/ct_compat_019_2026-10-12.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_tree(target: Path) -> Path:
    """A bundle-like code root with the bound policies, amendments and an execution manifest."""
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'config/preanalysis').glob('ct_*.json')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'newpaper/plan').glob('CT_EXPERIMENT_AMENDMENT_01[5-9]*.md')]
    for name in names:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target/name)
    manifest = ''.join(f'{sha256(target/n)}  {n}\n' for n in sorted(names))
    (target/'EXECUTION_SHA256SUMS').write_text(manifest, encoding='utf-8', newline='\n')
    return target


class ContextTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat_run
        self.m = ct_compat_run
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.code = code_tree(self.tmp/'code')
        self.project = self.tmp/'project'
        for sub in ('data/longitudinal-ct', 'data/pantrack'):
            (self.project/sub).mkdir(parents=True)
        replication = json.loads((ROOT/'config/preanalysis/ct_replication_015_2026-09-25.json').read_text())
        self.env = {'CT_COMPAT_POLICY_SHA256': sha256(self.code/POLICY),
                    'CT_COMPAT_SOURCE_SHA256': sha256(self.code/'EXECUTION_SHA256SUMS'),
                    'CT_UPDATE_MODEL_FREEZE_SHA256': replication['pantrack_model_freeze_sha256'],
                    'CT_UPDATE_REMOTE_ROOT': str(self.project),
                    'CT_UPDATE_REMOTE_DATA_ROOT': str(self.project/'data/longitudinal-ct'),
                    'CT_UPDATE_PANTRACK_REMOTE_ROOT': str(self.project/'data/pantrack')}

    def load(self, **changes):
        with mock.patch.dict(os.environ, dict(self.env, **changes)):
            return self.m.load_context(self.code/POLICY, code=self.code)

    def test_context_binds_both_replications_through_their_hashes(self):
        ctx = self.load()
        self.assertEqual(ctx.replication.policy_sha, '3f629e057c05896d3bba9ffefdccb45e481c15cd9412e73199e4c9c789171820')
        self.assertEqual(ctx.strong_policy_sha, '7470667550c0ad159c1b0193013790156e804cb701568f7dae0e8ad1c763a3b2')
        self.assertEqual(ctx.strong_root, (self.project/'runs/strong_017').resolve())
        self.assertEqual(ctx.output, (self.project/'runs/compat_019').resolve())

    def test_wrong_policy_source_or_model_freeze_is_rejected(self):
        for key in ('CT_COMPAT_POLICY_SHA256', 'CT_COMPAT_SOURCE_SHA256', 'CT_UPDATE_MODEL_FREEZE_SHA256'):
            with self.subTest(key=key), self.assertRaises(GateError):
                self.load(**{key: '0' * 64})

    def test_changed_amendment_is_rejected(self):
        amendment = self.code/'newpaper/plan/CT_EXPERIMENT_AMENDMENT_019_2026-10-12.md'
        amendment.write_bytes(amendment.read_bytes() + b'\n')
        names = sorted(line.split('  ', 1)[1] for line in
                       (self.code/'EXECUTION_SHA256SUMS').read_text().splitlines())
        (self.code/'EXECUTION_SHA256SUMS').write_text(''.join(f'{sha256(self.code/n)}  {n}\n' for n in names))
        with self.assertRaises(GateError):
            self.load(CT_COMPAT_SOURCE_SHA256=sha256(self.code/'EXECUTION_SHA256SUMS'))


class ResultGateTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_compat_run
        self.m = ct_compat_run
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def write(self, relative, value):
        path = self.tmp/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def unet_ctx(self, result):
        path = self.write('runs/replication_015/analysis/replication_results.json', result)
        policy = dict(replication_results_relative='runs/replication_015/analysis/replication_results.json',
                      replication_results_sha256=sha256(path), replication_policy_sha256='a' * 64,
                      replication_source_manifest_sha256='b' * 64,
                      segmenters=dict(unet=dict(runs=2, seeds=[1, 2])))
        return SimpleNamespace(project=self.tmp, policy=policy)

    def unet_result(self, **changes):
        return dict(dict(status='REPLICATION_ANALYSIS_COMPLETE', missing_seeds=[], seeds=[1, 2],
                         policy_sha256='a' * 64, source_manifest_sha256='b' * 64,
                         seed_receipt_sha256={'1': 'c' * 64, '2': 'd' * 64}), **changes)

    def test_complete_bound_replication_result_is_accepted(self):
        self.assertEqual(self.m.replication_results(self.unet_ctx(self.unet_result()))['seeds'], [1, 2])

    def test_incomplete_or_foreign_replication_results_are_rejected(self):
        for changes in (dict(status='RUNNING'), dict(missing_seeds=[2]), dict(seeds=[1]), dict(seeds=[1, 3]),
                        dict(policy_sha256='e' * 64), dict(source_manifest_sha256='e' * 64)):
            with self.subTest(changes=changes), self.assertRaises(GateError):
                self.m.replication_results(self.unet_ctx(self.unet_result(**changes)))

    def strong_setup(self, records=None, receipt_changes=None, result_changes=None):
        """Two nnU-Net runs with receipts and private predictions, as the CT-UPDATE-017 analysis left them."""
        root = self.tmp/'runs/strong_017'
        receipts = {}
        for seed in (11, 12):
            folder = root/f'run_{seed}'/'evaluation'
            folder.mkdir(parents=True)
            cohorts = {}
            for cohort in ('internal', 'pantrack'):
                path = folder/f'{cohort}_predictions_private.jsonl'
                rows = records or [dict(patient='p', pair_id='x', timepoint='bl')]
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                cohorts[cohort] = dict(predictions_sha256=sha256(path))
            receipt = dict(status='STRONG_SEED_COMPLETE', seed=seed, cohorts=cohorts, **(receipt_changes or {}))
            (folder/'seed_complete.json').write_text(json.dumps(receipt))
            receipts[str(seed)] = sha256(folder/'seed_complete.json')
        result = dict(status='STRONG_REPLICATION_ANALYSIS_COMPLETE', seeds=[11, 12], missing_seeds=[],
                      policy_sha256='f' * 64, seed_receipt_sha256=receipts)
        result.update(result_changes or {})
        path = self.write('runs/strong_017/analysis/strong_results.json', result)
        policy = dict(strong_results_relative='runs/strong_017/analysis/strong_results.json',
                      strong_results_sha256=sha256(path), strong_policy_sha256='f' * 64,
                      segmenters=dict(nnunet=dict(runs=2, seeds=[11, 12])))
        return SimpleNamespace(project=self.tmp, policy=policy, strong_root=root)

    def test_strong_runs_are_read_through_the_bound_receipts(self):
        ctx = self.strong_setup()
        runs, receipts = self.m.strong_runs(ctx)
        self.assertEqual([seed for seed, _ in runs['pantrack']], [11, 12])
        self.assertEqual(set(receipts), {'11', '12'})

    def test_changed_strong_predictions_are_rejected(self):
        ctx = self.strong_setup()
        path = ctx.strong_root/'run_12'/'evaluation'/'internal_predictions_private.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(GateError):
            self.m.strong_runs(ctx)

    def test_changed_strong_receipts_are_rejected(self):
        ctx = self.strong_setup()
        receipt = ctx.strong_root/'run_11'/'evaluation'/'seed_complete.json'
        receipt.write_text(receipt.read_text() + ' ')
        with self.assertRaises(GateError):
            self.m.strong_runs(ctx)

    def test_incomplete_or_foreign_strong_results_are_rejected(self):
        for changes in (dict(status='RUNNING'), dict(missing_seeds=[12]), dict(policy_sha256='e' * 64)):
            self.tmp = Path(tempfile.mkdtemp())
            self.addCleanup(shutil.rmtree, self.tmp)
            ctx = self.strong_setup(result_changes=changes)
            with self.subTest(changes=changes), self.assertRaises(GateError):
                self.m.strong_runs(ctx)


class MetadataTests(unittest.TestCase):
    def test_run_metadata_is_removed_before_reproduction(self):
        from careport.ct_compat_run import RUN_METADATA, core_result
        result = dict(status='COMPAT_ANALYSIS_COMPLETE', segmenters={}, **{k: 'x' for k in RUN_METADATA})
        self.assertEqual(core_result(result), dict(status='COMPAT_ANALYSIS_COMPLETE', segmenters={}))


if __name__ == '__main__':
    unittest.main()

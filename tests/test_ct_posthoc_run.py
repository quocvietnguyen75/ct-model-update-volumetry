"""Synthetic gates for executing CT-UPDATE-016 against the stored CT-UPDATE-015 outputs."""
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
POLICY = 'config/preanalysis/ct_posthoc_016_2026-10-02.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_tree(target: Path) -> Path:
    """A bundle-like code root with the bound policies, amendments and an execution manifest."""
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT/'config/preanalysis').glob('ct_*.json')]
    names += ['newpaper/plan/CT_EXPERIMENT_AMENDMENT_015_2026-09-25.md',
              'newpaper/plan/CT_EXPERIMENT_AMENDMENT_016_2026-10-02.md']
    for name in names:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target/name)
    manifest = ''.join(f'{sha256(target/n)}  {n}\n' for n in sorted(names))
    (target/'EXECUTION_SHA256SUMS').write_text(manifest, encoding='utf-8', newline='\n')
    return target


class ContextTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_posthoc_run
        self.m = ct_posthoc_run
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.code = code_tree(self.tmp/'code')
        self.project = self.tmp/'project'
        for sub in ('data/longitudinal-ct', 'data/pantrack'):
            (self.project/sub).mkdir(parents=True)
        replication = json.loads((ROOT/'config/preanalysis/ct_replication_015_2026-09-25.json').read_text())
        self.env = {'CT_POSTHOC_POLICY_SHA256': sha256(self.code/POLICY),
                    'CT_POSTHOC_SOURCE_SHA256': sha256(self.code/'EXECUTION_SHA256SUMS'),
                    'CT_UPDATE_MODEL_FREEZE_SHA256': replication['pantrack_model_freeze_sha256'],
                    'CT_UPDATE_REMOTE_ROOT': str(self.project),
                    'CT_UPDATE_REMOTE_DATA_ROOT': str(self.project/'data/longitudinal-ct'),
                    'CT_UPDATE_PANTRACK_REMOTE_ROOT': str(self.project/'data/pantrack')}

    def load(self, **changes):
        with mock.patch.dict(os.environ, dict(self.env, **changes)):
            return self.m.load_context(self.code/POLICY, code=self.code)

    def test_context_binds_the_frozen_replication_through_its_hashes(self):
        ctx = self.load()
        self.assertEqual(ctx.replication.policy_sha, '3f629e057c05896d3bba9ffefdccb45e481c15cd9412e73199e4c9c789171820')
        self.assertEqual(ctx.replication.source_sha, ctx.policy['replication_source_manifest_sha256'])
        self.assertNotEqual(ctx.source_sha, ctx.replication.source_sha)
        self.assertEqual(ctx.output, (self.project/'runs/posthoc_016').resolve())

    def test_wrong_policy_source_or_model_freeze_is_rejected(self):
        for key in ('CT_POSTHOC_POLICY_SHA256', 'CT_POSTHOC_SOURCE_SHA256', 'CT_UPDATE_MODEL_FREEZE_SHA256'):
            with self.subTest(key=key), self.assertRaises(GateError):
                self.load(**{key: '0' * 64})

    def test_changed_amendment_is_rejected(self):
        amendment = self.code/'newpaper/plan/CT_EXPERIMENT_AMENDMENT_016_2026-10-02.md'
        amendment.write_bytes(amendment.read_bytes() + b'\n')
        names = sorted(line.split('  ', 1)[1] for line in
                       (self.code/'EXECUTION_SHA256SUMS').read_text().splitlines())
        (self.code/'EXECUTION_SHA256SUMS').write_text(''.join(f'{sha256(self.code/n)}  {n}\n' for n in names))
        with self.assertRaises(GateError):
            self.load(CT_POSTHOC_SOURCE_SHA256=sha256(self.code/'EXECUTION_SHA256SUMS'))


class ReplicationResultTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_posthoc_run
        self.m = ct_posthoc_run
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def ctx_for(self, result):
        path = self.tmp/'runs/replication_015/analysis/replication_results.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        policy = dict(replication_results_relative='runs/replication_015/analysis/replication_results.json',
                      replication_results_sha256=sha256(path), replication_policy_sha256='a' * 64,
                      replication_source_manifest_sha256='b' * 64, runs=2)
        return SimpleNamespace(project=self.tmp, policy=policy)

    def complete(self, **changes):
        return dict(dict(status='REPLICATION_ANALYSIS_COMPLETE', missing_seeds=[], seeds=[1, 2],
                         policy_sha256='a' * 64, source_manifest_sha256='b' * 64,
                         seed_receipt_sha256={'1': 'c' * 64, '2': 'd' * 64}), **changes)

    def test_complete_bound_result_is_accepted(self):
        self.assertEqual(self.m.replication_results(self.ctx_for(self.complete()))['seeds'], [1, 2])

    def test_incomplete_or_foreign_results_are_rejected(self):
        for changes in (dict(status='RUNNING'), dict(missing_seeds=[2]), dict(seeds=[1]),
                        dict(policy_sha256='e' * 64), dict(source_manifest_sha256='e' * 64)):
            with self.subTest(changes=changes), self.assertRaises(GateError):
                self.m.replication_results(self.ctx_for(self.complete(**changes)))

    def test_changed_result_bytes_are_rejected(self):
        ctx = self.ctx_for(self.complete())
        path = self.tmp/ctx.policy['replication_results_relative']
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(GateError):
            self.m.replication_results(ctx)


class MetadataTests(unittest.TestCase):
    def test_run_metadata_is_removed_before_reproduction(self):
        from careport.ct_posthoc_run import RUN_METADATA, core_result
        result = dict(status='POSTHOC_ANALYSIS_COMPLETE', cohorts={}, **{k: 'x' for k in RUN_METADATA})
        self.assertEqual(core_result(result), dict(status='POSTHOC_ANALYSIS_COMPLETE', cohorts={}))


if __name__ == '__main__':
    unittest.main()

"""Synthetic gates for CT-UPDATE-015 training lineage, policies and seed receipts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from careport.ct_replication import MODELS, canonical_sha256, execution_policy
from careport.ct_replication_run import load_training, validate_seed_receipt, write_execution_policy
from careport.ct_update_gate import GateError
from careport.ct_update_train import execution_config

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/'config/preanalysis'/name).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_training(project, policy, base, index, eligible=None, execution_id=None):
    execution = execution_policy(policy, base, index)
    config = execution_config(base, execution)
    config_sha = canonical_sha256(config)
    seed = execution['seed']
    folder = project/policy['run_root_relative']/f'seed_{seed}'/'training'
    folder.mkdir(parents=True)
    (folder/'execution_config.json').write_text(json.dumps(config, indent=2)+'\n')
    hashes = {}
    for phase, steps in (('old', 20000), ('new', 10000), ('continued', 10000)):
        parent = None if phase == 'old' else hashes['old']
        path = folder/f'{phase}.pt'
        torch.save({'state_dict': {'w': torch.full((2,), float(len(phase)))}, 'phase': phase, 'seed': seed,
                    'config_sha256': config_sha, 'parent_sha256': parent, 'steps': steps}, path)
        hashes[phase] = sha256(path)
        counts = dict(policy['training']['eligible_training_patients'], **(eligible or {}))
        receipt = dict(phase=phase, steps=steps, seed=seed, parent_sha256=parent, config_sha256=config_sha,
                       execution_id=execution_id or execution['execution_id'], checkpoint_sha256=hashes[phase],
                       checkpoint_bytes=path.stat().st_size, eligible_training_patients=counts[phase],
                       elapsed_seconds=1., peak_allocated_bytes=1, status='TRAINING_COMPLETE')
        (folder/f'{phase}_receipt.json').write_text(json.dumps(receipt))
    (folder/'training_complete.json').write_text(json.dumps({'status': 'THREE_PHASES_COMPLETE', 'test_used': False}))
    return folder


class TrainingLineageTests(unittest.TestCase):
    def setUp(self):
        self.policy = load('ct_replication_015_2026-09-25.json')
        self.base = load('ct_update_001_2026-09-15.json')

    def test_complete_three_phase_lineage_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_training(Path(tmp), self.policy, self.base, 2)
            seed, config, states, receipts = load_training(Path(tmp), self.policy, self.base, 2)
            self.assertEqual(seed, 20261002)
            self.assertEqual(config['seed'], 20261002)
            self.assertEqual(tuple(states), MODELS)
            self.assertEqual(receipts['new']['parent_sha256'], receipts['old']['checkpoint_sha256'])

    def test_wrong_cache_execution_or_changed_artifacts_are_rejected(self):
        cases = [dict(eligible={'new': 113}), dict(execution_id='CT-REPL-015-99')]
        for case in cases:
            with tempfile.TemporaryDirectory() as tmp:
                make_training(Path(tmp), self.policy, self.base, 1, **case)
                with self.assertRaises(GateError):
                    load_training(Path(tmp), self.policy, self.base, 1)
        for damage in ('completion', 'config', 'checkpoint'):
            with tempfile.TemporaryDirectory() as tmp:
                folder = make_training(Path(tmp), self.policy, self.base, 1)
                if damage == 'completion':
                    (folder/'training_complete.json').unlink()
                elif damage == 'config':
                    config = json.loads((folder/'execution_config.json').read_text())
                    config['training']['update_steps'] = 9999
                    (folder/'execution_config.json').write_text(json.dumps(config))
                else:
                    with (folder/'new.pt').open('ab') as stream:
                        stream.write(b'0')
                with self.assertRaises((GateError, FileNotFoundError)):
                    load_training(Path(tmp), self.policy, self.base, 1)


class PolicyFileAndReceiptTests(unittest.TestCase):
    def setUp(self):
        self.policy = load('ct_replication_015_2026-09-25.json')

    def test_execution_policy_file_is_created_once_and_never_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'execution_policy.json'
            value = dict(execution_id='CT-REPL-015-01', seed=20261001)
            write_execution_policy(path, value)
            first = path.read_bytes()
            write_execution_policy(path, value)
            self.assertEqual(path.read_bytes(), first)
            with self.assertRaises(GateError):
                write_execution_policy(path, dict(value, seed=20261002))

    def receipt(self):
        cohorts = {c: dict(patients=v['patients'], pairs=v['pairs'], timepoints=2*v['pairs'],
                           predictions_sha256='a'*64) for c, v in self.policy['cohorts'].items()}
        return dict(status='REPLICATION_SEED_COMPLETE', protocol_id='CT-UPDATE-015', index=3, seed=20261003,
                    execution_id='CT-REPL-015-03', models=list(MODELS), policy_sha256='b'*64,
                    source_manifest_sha256='c'*64, amendment_sha256=self.policy['amendment_sha256'], cohorts=cohorts)

    def test_seed_receipts_bind_seed_policy_source_and_cohorts(self):
        validate_seed_receipt(self.receipt(), self.policy, 3, 'b'*64, 'c'*64)
        changes = [dict(seed=20261004), dict(index=4), dict(execution_id='CT-REPL-015-04'),
                   dict(status='FAILED'), dict(models=['old', 'new']), dict(policy_sha256='d'*64),
                   dict(source_manifest_sha256='d'*64)]
        for change in changes:
            with self.assertRaises(GateError):
                validate_seed_receipt(dict(self.receipt(), **change), self.policy, 3, 'b'*64, 'c'*64)
        broken = self.receipt()
        broken['cohorts']['pantrack']['pairs'] = 180
        with self.assertRaises(GateError):
            validate_seed_receipt(broken, self.policy, 3, 'b'*64, 'c'*64)
        broken = self.receipt()
        broken['cohorts']['internal']['predictions_sha256'] = 'not-a-hash'
        with self.assertRaises(GateError):
            validate_seed_receipt(broken, self.policy, 3, 'b'*64, 'c'*64)


if __name__ == '__main__':
    unittest.main()

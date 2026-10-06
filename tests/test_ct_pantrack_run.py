"""Synthetic file-integrity and development-only calibration checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from careport.ct_update_gate import GateError
from careport.ct_pantrack_run import checked_path, read_bound_json, fit_development_maps, load_freeze


class ExternalExecutionGateTests(unittest.TestCase):
    def test_bound_json_rejects_changed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.json'
            raw = b'{"status": "PASS"}'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(read_bound_json(path, digest)['status'], 'PASS')
            path.write_bytes(b'{"status": "FAIL"}')
            with self.assertRaises(GateError): read_bound_json(path, digest)

    def test_relative_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(checked_path(root, 'runs/model.pt'), root / 'runs/model.pt')
            for value in ['../outside', str(root / 'absolute'), 'runs/../../outside']:
                with self.subTest(value=value), self.assertRaises(GateError):
                    checked_path(root, value)

    def test_frozen_coefficients_must_match_the_submission_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'control').mkdir()
            policy, registry = root/'policy.json', root/'registry.json'
            policy.write_bytes(b'{}'); registry.write_bytes(b'{}')
            empty_sha = hashlib.sha256(b'{}').hexdigest()
            frozen = dict(status='EXTERNAL_MODELS_AND_CALIBRATION_FROZEN', policy_sha256=empty_sha,
                          checkpoint_registry_sha256=empty_sha, external_arrays_read=False,
                          runs=[dict(seed=s, maps={}) for s in [20260915, 20260917, 20260919]])
            path = root/'control/pantrack_models_010.json'
            path.write_text(json.dumps(frozen))
            submitted = hashlib.sha256(path.read_bytes()).hexdigest()
            args = SimpleNamespace(policy=policy, registry=registry)
            with patch.dict(os.environ, {'CT_UPDATE_MODEL_FREEZE_SHA256': submitted}):
                self.assertEqual(load_freeze(args, root)[1], submitted)
                frozen['runs'][0]['maps'] = {'modified_after_submission': 1}
                path.write_text(json.dumps(frozen))
                with self.assertRaises(GateError): load_freeze(args, root)

    def test_calibration_uses_baseline_model_values_only(self):
        rows = [dict(patient='synthetic_a', old_bl=1., new_bl=3., continued_bl=4.),
                dict(patient='synthetic_b', old_bl=2., new_bl=5., continued_bl=7.),
                dict(patient='synthetic_c', old_bl=3., new_bl=7., continued_bl=10.)]
        maps = fit_development_maps(rows)
        self.assertAlmostEqual(maps['old_to_new']['slope'], 2.)
        self.assertAlmostEqual(maps['old_to_new']['intercept_ml'], 1.)
        changed = copy.deepcopy(rows)
        for row in changed:
            row.update(reference_bl=999999., reference_fu=-123., new_fu=float('nan'))
        self.assertEqual(fit_development_maps(changed), maps)

    def test_duplicate_lesion_repetition_does_not_increase_patient_weight(self):
        rows = [dict(patient='synthetic_a', old_bl=1., new_bl=1., continued_bl=1.),
                dict(patient='synthetic_b', old_bl=2., new_bl=4., continued_bl=2.),
                dict(patient='synthetic_c', old_bl=3., new_bl=5., continued_bl=3.)]
        original, repeated = fit_development_maps(rows), fit_development_maps(rows + [rows[0]]*7)
        for name in original:
            self.assertAlmostEqual(original[name]['slope'], repeated[name]['slope'])
            self.assertAlmostEqual(original[name]['intercept_ml'], repeated[name]['intercept_ml'])


if __name__ == '__main__':
    unittest.main()

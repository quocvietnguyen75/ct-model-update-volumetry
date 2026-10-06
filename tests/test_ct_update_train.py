import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from careport.ct_update_gate import GateError
from careport.ct_update_train import (
    PatchSampler, checkpoint_state, validate_cache_receipt, make_optimizer, execution_config,
)


class CTUpdateTrainTests(unittest.TestCase):
    def test_execution_policy_only_overrides_budget_and_seed_from_bound_source(self):
        base = {"seed": 1, "training": {"old_steps": 4, "update_steps": 4},
                "patch": {"shape": [96, 96, 96]}}
        policy = {"execution_id": "CT-PILOT-002", "amendment": "CT-UPDATE-003",
                  "source_config_sha256": hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
                  "seed": 2, "old_steps": 20, "update_steps": 10}
        merged = execution_config(base, policy)
        self.assertEqual(merged["training"]["old_steps"], 20)
        self.assertEqual(merged["seed"], 2)
        self.assertEqual(base["training"]["old_steps"], 4)
        self.assertEqual(merged["patch"], base["patch"])
        for invalid in ({**policy, "patch": {"shape": [32, 32, 32]}},
                        {**policy, "source_config_sha256": "0" * 64},
                        {**policy, "old_steps": -1}, {**policy, "update_steps": True}):
            with self.subTest(policy=invalid), self.assertRaises(GateError):
                execution_config(base, invalid)

    def test_sampler_is_patient_balanced_and_never_uses_validation(self):
        records = []
        for patient, count in (("a", 1), ("b", 9)):
            for lesion in range(count):
                for timepoint in ("bl", "fu"):
                    records.append(dict(patient=patient, lesion_id=lesion, timepoint=timepoint,
                                        split="train", old_training=True, patch_file="unused"))
        records.append(dict(patient="validation", split="val", old_training=False))
        sampler = PatchSampler(Path("."), records, True, 23)
        selected = [sampler.select_record()["patient"] for _ in range(10000)]
        self.assertLess(abs(selected.count("a") / len(selected) - .5), .03)
        self.assertNotIn("validation", selected)

    def test_model_lineage_is_copied_and_optimizer_does_not_carry_old_moments(self):
        model = torch.nn.Conv3d(1, 1, 1)
        state = checkpoint_state(model)
        descendant = torch.nn.Conv3d(1, 1, 1)
        descendant.load_state_dict(state)
        for a, b in zip(model.parameters(), descendant.parameters()):
            torch.testing.assert_close(a, b)
        with torch.no_grad():
            next(model.parameters()).add_(1.)
        self.assertFalse(torch.equal(next(model.parameters()), next(descendant.parameters())))
        opt = make_optimizer(descendant, {"learning_rate": .0003, "weight_decay": .00001})
        self.assertEqual(len(opt.state), 0)

    def test_cache_requires_matching_configuration_and_private_index(self):
        config = {"test_values_permitted": False}
        payload = b"private cache index"
        report = {"status": "PREPROCESSING_PASS", "test_values_read": False,
                  "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  "index_sha256": hashlib.sha256(payload).hexdigest()}
        validate_cache_receipt(report, payload, config)
        with self.assertRaises(GateError):
            validate_cache_receipt(report, payload + b"bad", config)
        with self.assertRaises(GateError):
            validate_cache_receipt({**report, "test_values_read": True}, payload, config)

    def test_paired_augmentation_and_patch_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patch = np.arange(64, dtype=np.float16).reshape(4, 4, 4)
            mask = (patch > 31).astype(np.uint8)
            np.savez(root / "patch.npz", image=patch, target=mask)
            records = [dict(patient="a", lesion_id=1, timepoint=t, split="train", old_training=True,
                            patch_file="patch.npz") for t in ("bl", "fu")]
            x, y = PatchSampler(root, records, True, 2).batch(4)
            self.assertEqual(tuple(x.shape), (4, 1, 4, 4, 4))
            np.testing.assert_array_equal(x.numpy() > 31, y.numpy().astype(bool))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from careport.ct_update_gate import GateError
from careport.ct_update_preprocess import make_patch_pair, projected_cache_bytes, require_schema, resolve_csv_discrepancy


class CTUpdatePreprocessTests(unittest.TestCase):
    def test_patch_preserves_only_selected_lesion_and_counts_truncation(self):
        image = np.full((9, 9, 9), 2000, dtype=np.int16)
        labels = np.zeros(image.shape, dtype=np.uint8)
        labels[3:6, 3:6, 3:6] = 1
        labels[0:2, 0:2, 0:2] = 2
        record = {"lesion_id": 1, "centroid": [4, 4, 4], "native_affine": np.eye(4).tolist(),
                  "native_lesion_box": [[3, 6], [3, 6], [3, 6]]}
        policy = {"shape": [5, 5, 5], "spacing_mm": [1, 1, 1], "hu_clip": [-1000, 1000]}
        patch, target, geometry, truncated = make_patch_pair(image, labels, record, policy)
        self.assertEqual(patch.dtype, np.float16)
        self.assertEqual(target.dtype, np.uint8)
        self.assertEqual(int(target.sum()), 27)
        self.assertEqual(float(patch.max()), 1.)
        self.assertFalse(truncated)
        record["native_lesion_box"] = [[0, 9], [3, 6], [3, 6]]
        self.assertTrue(make_patch_pair(image, labels, record, policy)[3])

    def test_cache_projection_counts_both_image_and_mask_and_enforces_cap(self):
        self.assertGreaterEqual(projected_cache_bytes(2, (96, 96, 96), 10000000), 2 * 96**3 * 3)
        with self.assertRaises(GateError):
            projected_cache_bytes(100, (96, 96, 96), 1000)

    def test_failed_or_tampered_schema_manifest_cannot_extract(self):
        import hashlib
        import json
        config = {"x": 1}
        payload = b"private synthetic manifest"
        report = {"status": "SCIENTIFIC_SCHEMA_PASS",
                  "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  "private_manifest_sha256": hashlib.sha256(payload).hexdigest()}
        require_schema(report, payload, config)
        with self.assertRaises(GateError):
            require_schema({**report, "status": "SCIENTIFIC_SCHEMA_FAIL"}, payload, config)
        with self.assertRaises(GateError):
            require_schema(report, payload + b"changed", config)

    def test_csv_warning_requires_independent_recount_and_preserves_other_failures(self):
        report = {"status": "SCIENTIFIC_SCHEMA_FAIL", "issues": {"csv_native_volume_mismatch": 15},
                  "private_manifest_sha256": "a" * 64}
        independent = {"status": "INDEPENDENT_NATIVE_COUNT_PASS", "checked": 15,
                       "count_mismatches": 0, "geometry_mismatches": 0,
                       "full_vs_bounded_count_agreements": 15, "source_manifest_sha256": "a" * 64}
        result = resolve_csv_discrepancy(report, independent)
        self.assertEqual(result["status"], "SCIENTIFIC_SCHEMA_PASS")
        self.assertEqual(result["warnings"]["csv_native_volume_mismatch"], 15)
        self.assertEqual(report["status"], "SCIENTIFIC_SCHEMA_FAIL")
        with self.assertRaises(GateError):
            resolve_csv_discrepancy(report, {**independent, "count_mismatches": 1})
        with self.assertRaises(GateError):
            resolve_csv_discrepancy({**report, "issues": {"centroid_xyz_mismatch": 1}}, independent)


if __name__ == "__main__":
    unittest.main()

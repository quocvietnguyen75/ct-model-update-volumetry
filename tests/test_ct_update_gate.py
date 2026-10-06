import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from careport.ct_update_gate import (
    GateError, audit_metadata, choose_old_patients, metadata_fingerprint,
    validate_input_policy, validate_splits, validate_storage,
)


class CTUpdateGateTests(unittest.TestCase):
    def test_split_overlap_and_unassigned_patient_are_rejected(self):
        splits = {"train": ["0000000001"], "val": ["0000000002"], "test": ["0000000003"]}
        patients = set(sum(splits.values(), []))
        self.assertEqual(validate_splits(splits, patients), {"train": 1, "val": 1, "test": 1})
        for bad in ({**splits, "test": ["0000000001"]}, {**splits, "val": []},
                    {**splits, "train": ["0000000001", "0000000001"]}):
            with self.subTest(bad=bad), self.assertRaises(GateError):
                validate_splits(bad, patients)

    def test_old_subset_is_patient_level_deterministic_and_order_independent(self):
        ids = [f"{i:010d}" for i in range(20)]
        selected = choose_old_patients(ids, 10, "fixed:")
        self.assertEqual(selected, choose_old_patients(list(reversed(ids)), 10, "fixed:"))
        self.assertEqual(len(set(selected)), 10)
        self.assertTrue(set(selected) <= set(ids))
        with self.assertRaises(GateError):
            choose_old_patients(ids, 21, "fixed:")

    def test_prohibited_inputs_and_unjustified_time_claim_fail(self):
        policy = {"model_input_roles": ["ct_image", "reference_centroid"],
                  "time_policy": "supplied_BL_FU_pair_only_no_elapsed_time_claim",
                  "test_values_permitted": False,
                  "permitted_development_splits": ["train", "val"]}
        validate_input_policy(policy)
        for key, value in [("model_input_roles", ["ct_image", "volume_bl"]),
                           ("time_policy", "growth_per_day"),
                           ("test_values_permitted", True),
                           ("permitted_development_splits", ["train", "test"])]:
            with self.subTest(key=key), self.assertRaises(GateError):
                validate_input_policy({**policy, key: value})

    def test_storage_counts_existing_raw_copy_and_rejects_insufficient_space(self):
        policy = {"study_cap_bytes": 200, "project_cap_bytes": 300, "planned_additional_bytes": 80}
        self.assertEqual(validate_storage(policy, 100, 100)["projected_study_bytes"], 180)
        for used, free in [(121, 100), (100, 79), (-1, 100)]:
            with self.subTest(used=used, free=free), self.assertRaises(GateError):
                validate_storage(policy, used, free)

    def test_metadata_gate_does_not_read_csv_body_and_binds_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputsTr").mkdir()
            (root / "targetsTr").mkdir()
            splits = {"train": ["0000000001"], "val": [], "test": []}
            (root / "data_split.json").write_text(json.dumps(splits))
            (root / "inputsTr/0000000001.csv").write_bytes(b"lesion_id,linking_unclear\nSECRET,\xff\n")
            for timepoint in ("BL", "FU"):
                (root / f"inputsTr/0000000001_{timepoint}_img_00.nii.gz").write_bytes(b"opaque image")
                target = "inputsTr" if timepoint == "BL" else "targetsTr"
                (root / f"{target}/0000000001_{timepoint}_mask_00.nii.gz").write_bytes(b"opaque mask")
            files = [(p.relative_to(root).as_posix(), p.stat().st_size) for p in root.rglob("*") if p.is_file()]
            config = {
                "protocol_id": "synthetic", "expected_files": len(files),
                "expected_bytes": sum(n for _, n in files),
                "expected_metadata_sha256": metadata_fingerprint(files),
                "expected_split_sha256": hashlib.sha256((root / "data_split.json").read_bytes()).hexdigest(),
                "expected_content_manifest_sha256": "a" * 64,
                "expected_split_counts": {"train": 1, "val": 0, "test": 0},
                "required_csv_columns": ["lesion_id"], "optional_csv_columns": ["linking_unclear"],
                "model_input_roles": ["ct_image", "reference_centroid"],
                "time_policy": "supplied_BL_FU_pair_only_no_elapsed_time_claim",
                "test_values_permitted": False, "permitted_development_splits": ["train", "val"],
                "storage": {"study_cap_bytes": 20000, "project_cap_bytes": 30000, "planned_additional_bytes": 1000},
            }
            receipt = {"status": "CONTENT_HASH_PASS", "content_manifest_sha256": "a" * 64,
                       "files_verified": len(files), "metadata_sha256": config["expected_metadata_sha256"]}
            report = audit_metadata(root, config, receipt, 1000, 10000)
            self.assertEqual(report["status"], "METADATA_GATE_PASS")
            self.assertFalse(report["csv_body_read"])
            self.assertNotIn("SECRET", json.dumps(report))
            self.assertNotIn("0000000001", json.dumps(report))
            with self.assertRaises(GateError):
                audit_metadata(root, config, {**receipt, "status": "PENDING"}, 1000, 10000)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from careport.ct_update_gate import GateError
from careport.ct_update_evaluate import native_metrics, paired_prediction_rows


class CTUpdateEvaluateTests(unittest.TestCase):
    def test_native_reference_outside_crop_counts_as_false_negative(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint8)
        labels[2:7, 2:7, 2:7] = 1
        logits = np.full((3, 3, 3), 10.)
        patch = np.eye(4)
        patch[:3, 3] = [3, 3, 3]
        result = native_metrics(logits, patch, np.eye(4), labels, 1, 125)
        self.assertAlmostEqual(result["volume_ml"], .027)
        self.assertAlmostEqual(result["dice"], 54 / 152)
        self.assertGreater(result["hd95_mm"], 0)

    def test_perfect_mask_and_empty_prediction_are_handled_explicitly(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint8)
        labels[3:6, 3:6, 3:6] = 1
        logits = np.where(labels == 1, 10., -10.)
        result = native_metrics(logits, np.eye(4), np.eye(4), labels, 1, 27)
        self.assertEqual(result["dice"], 1.)
        self.assertEqual(result["hd95_mm"], 0.)
        empty = native_metrics(np.full_like(logits, -10), np.eye(4), np.eye(4), labels, 1, 27)
        self.assertEqual(empty["dice"], 0.)
        self.assertIsNone(empty["hd95_mm"])
        self.assertTrue(empty["empty_prediction"])

    def test_invalid_reference_or_nan_logits_stop_evaluation(self):
        labels = np.ones((3, 3, 3), dtype=np.uint8)
        with self.assertRaises(GateError):
            native_metrics(np.full(labels.shape, np.nan), np.eye(4), np.eye(4), labels, 1, 27)
        with self.assertRaises(GateError):
            native_metrics(np.ones(labels.shape), np.eye(4), np.eye(4), labels, 1, 26)

    def test_pairing_rejects_missing_timepoint_and_preserves_csv_sensitivity(self):
        records = []
        for t, volume in (("bl", 10.), ("fu", 15.)):
            records.append({"patient": "a", "lesion_id": 1, "timepoint": t,
                            "reference_volume_ml": volume, "csv_volume_mm3": volume * 2000,
                            "predictions": {m: {"volume_ml": volume, "dice": .9}
                                            for m in ("old", "new", "continued")}})
        rows = paired_prediction_rows(records, use_csv_reference=False)
        self.assertEqual(rows[0]["reference_fu"], 15.)
        self.assertEqual(paired_prediction_rows(records, True)[0]["reference_fu"], 30.)
        with self.assertRaises(GateError):
            paired_prediction_rows(records[:1], False)


if __name__ == "__main__":
    unittest.main()

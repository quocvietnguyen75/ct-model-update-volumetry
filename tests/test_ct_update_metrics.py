import unittest

from careport.ct_update_gate import GateError
from careport.ct_update_metrics import summarize_predictions, training_groups


class CTUpdateMetricsTests(unittest.TestCase):
    def test_patient_balancing_and_retention_contrast_identity(self):
        rows = []
        for patient, n, delta in (("synthetic_A", 1, 2), ("synthetic_B", 9, 10)):
            for lesion in range(n):
                rows.append({"patient": patient, "lesion_id": lesion, "reference_bl": 10.,
                             "reference_fu": 15., "old_bl": 10., "old_fu": 15. + delta,
                             "new_bl": 12., "new_fu": 16., "continued_bl": 11.,
                             "continued_fu": 16., "old_dice": 0.7, "new_dice": 0.8,
                             "continued_dice": 0.75})
        report = summarize_predictions(rows, 100, 1)
        self.assertEqual(report["strategies"]["old_old"]["mae_ml"], 6.)
        self.assertEqual(report["strategies"]["old_new"]["mae_ml"], 1.)
        self.assertAlmostEqual(report["contrasts"]["ordinary_update_benefit"]["mean_ml"],
                               report["contrasts"]["refresh_benefit"]["mean_ml"]
                               - report["contrasts"]["retaining_baseline_cost"]["mean_ml"])
        self.assertNotIn("synthetic_A", str(report))

    def test_duplicate_lesions_and_nonfinite_predictions_are_rejected(self):
        row = {"patient": "a", "lesion_id": 1, "reference_bl": 1., "reference_fu": 2.,
               "old_bl": 1., "old_fu": 2., "new_bl": 1., "new_fu": 2.,
               "continued_bl": 1., "continued_fu": 2., "old_dice": 1., "new_dice": 1.,
               "continued_dice": 1.}
        with self.assertRaises(GateError):
            summarize_predictions([row, row], 100, 1)
        with self.assertRaises(GateError):
            summarize_predictions([{**row, "new_fu": float("nan")}], 100, 1)

    def test_training_index_rejects_test_and_groups_pairs_by_patient(self):
        records = [dict(patient="a", lesion_id=1, timepoint=t, split="train", old_training=True)
                   for t in ("bl", "fu")]
        self.assertEqual(len(training_groups(records, True)), 1)
        with self.assertRaises(GateError):
            training_groups(records + [{**records[0], "split": "test"}], False)
        with self.assertRaises(GateError):
            training_groups(records[:1], False)


if __name__ == "__main__":
    unittest.main()

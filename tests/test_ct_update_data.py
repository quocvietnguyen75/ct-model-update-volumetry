import unittest

import numpy as np

from careport.ct_update_gate import GateError
from careport.ct_update_data import (
    classify_rows, coordinate, integer, patch_affine, sample_patch, native_prediction,
)


class CTUpdateDataTests(unittest.TestCase):
    def row(self, identity="1", topology="UNCHANGED", clear="False", merged=""):
        return {"lesion_id": identity, "topology_class": topology, "linking_unclear": clear,
                "merged_into": merged, "img_id_bl": "0", "img_id_fu": "1.0",
                "cog_bl": "[2, 3, 4]", "cog_fu": "[4.0 3.0 2.0]",
                "volume_bl": "16", "volume_fu": "32",
                "target_lesion": "False", "use_for_challenge": "False"}

    def test_parse_coordinates_and_integer_indices_without_eval(self):
        self.assertEqual(coordinate("[2.0, 3, 4]"), (2.0, 3.0, 4.0))
        self.assertEqual(coordinate("[2 3 4]"), (2.0, 3.0, 4.0))
        self.assertEqual(integer("1.0"), 1)
        for value in ("[1,2]", "[1,nan,3]", "__import__('os')"):
            with self.subTest(value=value), self.assertRaises(GateError):
                coordinate(value)
        with self.assertRaises(GateError):
            integer("1.5")

    def test_primary_cohort_does_not_filter_target_flags_or_uncertain_links_in(self):
        eligible, counts = classify_rows([self.row()])
        self.assertEqual(len(eligible), 1)
        missing = self.row("2")
        missing.pop("linking_unclear")
        uncertain = self.row("3", clear="True")
        eligible, counts = classify_rows([self.row(), missing, uncertain])
        self.assertEqual([x["lesion_id"] for x in eligible], [1])
        self.assertEqual(counts["link_unknown"], 1)
        self.assertEqual(counts["link_unclear"], 1)

    def test_merger_destination_is_not_one_to_one_even_if_unchanged(self):
        eligible, counts = classify_rows([self.row(), self.row("2", "MERGING", merged="1")])
        self.assertEqual(eligible, [])
        self.assertEqual(counts["merger_destination"], 1)
        with self.assertRaises(GateError):
            classify_rows([self.row(), self.row()])

    def test_patch_uses_physical_spacing_and_fixed_centre_under_flipped_affine(self):
        affine = np.diag([-2., 3., 4., 1.])
        affine[:3, 3] = [10, -20, 50]
        patch = patch_affine(affine, (5, 6, 7), (5, 5, 5), (1, 1, 1))
        np.testing.assert_allclose(patch @ [2, 2, 2, 1], affine @ [5, 6, 7, 1])
        np.testing.assert_allclose(np.linalg.norm(patch[:3, :3], axis=0), [1, 1, 1])

    def test_identity_resampling_recovers_mask_and_native_volume(self):
        image = np.arange(125).reshape(5, 5, 5).astype(np.float32)
        sampled = sample_patch(image, np.eye(4), np.eye(4), (5, 5, 5), 1, -1000)
        np.testing.assert_array_equal(sampled, image)
        logits = np.full((5, 5, 5), -10., dtype=np.float32)
        logits[1:4, 1:4, 1:4] = 10
        slices, prediction = native_prediction(logits, np.eye(4), np.eye(4), (5, 5, 5))
        full = np.zeros((5, 5, 5), dtype=bool)
        full[slices] = prediction
        np.testing.assert_array_equal(full, logits >= 0)
        self.assertEqual(int(full.sum()), 27)

    def test_linear_ct_interpolation_retains_fractional_hu(self):
        image = np.indices((3, 3, 3))[0].astype(np.int16)
        target = np.eye(4)
        target[0, 3] = 0.5
        sampled = sample_patch(image, np.eye(4), target, (1, 1, 1), 1, -1000)
        self.assertEqual(float(sampled[0, 0, 0]), 0.5)

    def test_native_grid_outside_crop_stays_background(self):
        patch = np.eye(4)
        patch[:3, 3] = [3, 3, 3]
        slices, pred = native_prediction(np.ones((3, 3, 3)), patch, np.eye(4), (9, 9, 9))
        full = np.zeros((9, 9, 9), dtype=bool)
        full[slices] = pred
        self.assertEqual(int(full.sum()), 27)
        self.assertFalse(full[0, 0, 0])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from careport.ct_update_recount import independent_count


class CTUpdateRecountTests(unittest.TestCase):
    def test_full_and_bounded_counts_and_physical_volume_agree(self):
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[1:4, 1:4, 1:4] = 2
        record = {"lesion_id": 2, "reference_voxels": 27,
                  "native_lesion_box": [[1, 4], [1, 4], [1, 4]]}
        result = independent_count(labels, np.diag([-2., 3., 4., 1.]), (2, 3, 4), record)
        self.assertEqual(result, {"count_matches": True, "box_count_matches": True, "physical_volume_matches": True})
        record["native_lesion_box"] = [[1, 3], [1, 4], [1, 4]]
        self.assertFalse(independent_count(labels, np.eye(4), (1, 1, 1), record)["box_count_matches"])

    def test_wrong_count_or_header_spacing_remains_a_failure(self):
        labels = np.ones((3, 3, 3), dtype=np.uint8)
        record = {"lesion_id": 1, "reference_voxels": 26,
                  "native_lesion_box": [[0, 3], [0, 3], [0, 3]]}
        result = independent_count(labels, np.eye(4), (1, 1, 2), record)
        self.assertFalse(result["count_matches"])
        self.assertFalse(result["physical_volume_matches"])


if __name__ == "__main__":
    unittest.main()

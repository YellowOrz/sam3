import unittest

import numpy as np

from scripts.select_realsense_overlap_diagnostics import overlap_row


class OverlapSelectionTest(unittest.TestCase):
    def test_empty_has_undefined_ratios(self):
        result = overlap_row(np.zeros((3, 4)), np.zeros((3, 4)), 0)
        self.assertEqual(result["intersection_pixels"], 0)
        self.assertIsNone(result["iou"])
        self.assertIsNone(result["intersection_bbox_xyxy_exclusive"])

    def test_nonzero_union_and_exclusive_bbox(self):
        first = np.array([[0, 2, 1], [0, 0, 1]], dtype=np.uint8)
        second = np.array([[0, 1, 0], [0, 0, 2]], dtype=np.uint8)
        result = overlap_row(first, second, 17)
        self.assertEqual(result["frame_index_zero_based"], 17)
        self.assertEqual(result["intersection_pixels"], 2)
        self.assertEqual(result["intersection_bbox_xyxy_exclusive"], [1, 0, 3, 2])
        self.assertAlmostEqual(result["iou"], 2 / 3)
        self.assertEqual(result["intersection_over_smaller_nonempty_layer"], 1.0)

    def test_disjoint_nonempty_is_zero_not_undefined(self):
        result = overlap_row(np.array([[1, 0]]), np.array([[0, 1]]), 1)
        self.assertEqual(result["intersection_over_first"], 0.0)
        self.assertEqual(result["intersection_over_second"], 0.0)
        self.assertEqual(result["iou"], 0.0)


if __name__ == "__main__":
    unittest.main()

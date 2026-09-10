import unittest

import numpy as np
from pycocotools import mask as mask_utils

from scripts.evaluate_bilateral_tokens import average_precision, decode_gt_mask, dice_iou


class EvaluationMetricsTest(unittest.TestCase):
    def test_tied_scores_do_not_depend_on_ground_truth_order(self):
        rows = [
            {"target_present": True, "top_confidence": 0.5},
            {"target_present": False, "top_confidence": 0.5},
        ]
        self.assertEqual(average_precision(rows), 0.5)
        self.assertEqual(average_precision(rows[::-1]), 0.5)

    def test_compressed_rle_round_trip_and_empty_target(self):
        mask = np.zeros((8, 9), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        decoded = decode_gt_mask({"segmentation": rle}, 8, 9)
        np.testing.assert_array_equal(decoded, mask.astype(bool))
        self.assertEqual(dice_iou(decoded, decoded), (1.0, 1.0))
        empty = decode_gt_mask(None, 8, 9)
        self.assertEqual(dice_iou(empty, decoded), (0.0, 0.0))

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pycocotools import mask as mask_utils

from scripts.evaluate_bilateral_tokens import (
    average_precision, decode_gt_mask, dice_iou, validate_batch_identity,
    get_visual_renderer, parse_args, render_montages,
)


class EvaluationMetricsTest(unittest.TestCase):
    def test_separated_visuals_are_default_and_legacy_overlay_is_explicit(self):
        from scripts.render_separated_masks import render_montages as separated
        argv = ["evaluate", "--data-root", "/data", "--base-checkpoint", "/model.pt",
                "--output-dir", "/output", "--include-ve"]
        with patch("sys.argv", argv):
            self.assertEqual(parse_args().visual_style, "separate")
        with patch("sys.argv", argv + ["--visual-style", "overlay"]):
            self.assertEqual(parse_args().visual_style, "overlay")
        self.assertIs(get_visual_renderer("separate"), separated)
        self.assertIs(get_visual_renderer("overlay"), render_montages)
        with self.assertRaises(ValueError):
            get_visual_renderer("unknown")

    def test_loader_substitution_and_category_mismatch_are_rejected(self):
        metadata = SimpleNamespace(
            coco_image_id=torch.tensor([20, 20, 10, 10]),
            original_category_id=torch.tensor([1, 2, 1, 2]),
        )
        batch = SimpleNamespace(
            find_inputs=[SimpleNamespace(
                img_ids=torch.tensor([0, 0, 1, 1]),
                text_ids=torch.tensor([0, 1, 0, 1]),
            )],
            find_metadatas=[metadata],
        )
        images = [{"id": 10}, {"id": 20}]
        validate_batch_identity(batch, [1, 0], images)
        metadata.coco_image_id[0] = 30
        with self.assertRaisesRegex(RuntimeError, "substituted image"):
            validate_batch_identity(batch, [1, 0], images)
        metadata.coco_image_id[0] = 20
        metadata.original_category_id[0] = 2
        with self.assertRaisesRegex(RuntimeError, "Prompt/category mismatch"):
            validate_batch_identity(batch, [1, 0], images)

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

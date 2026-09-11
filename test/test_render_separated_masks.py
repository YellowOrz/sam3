import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.render_separated_masks import binary_mask, decode_physical_gt, render_montages


class RenderSeparatedMasksTest(unittest.TestCase):
    def test_compressed_and_uncompressed_coco_rle_match(self):
        # Fixed COCO maskApi encoding with negative deltas: runs [3,2,4,1,2].
        compressed = {"segmentation": {"size": [3, 4], "counts": "324ON"}}
        uncompressed = {"segmentation": {"size": [3, 4], "counts": [3, 2, 4, 1, 2]}}
        first = decode_physical_gt(compressed, 3, 4)
        second = decode_physical_gt(uncompressed, 3, 4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.sum(), 3)

    def test_binary_mask_contract_rejects_wrong_shape_values_and_nan(self):
        for invalid in (np.zeros((2, 3)), np.full((3, 4), 255), np.full((3, 4), np.nan)):
            with self.assertRaises(ValueError):
                binary_mask(invalid, 3, 4, "bad")
        self.assertEqual(binary_mask(np.ones((3, 4), dtype=np.uint8), 3, 4, "good").dtype, bool)

    def test_original_rgb_and_binary_masks_are_separate_and_low_score_candidate_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            Image.fromarray(rgb).save(root / "source.png")
            images = [{"id": 7, "file_name": "source.png", "height": 3, "width": 4,
                       "source": "dexycb", "sequence": "seq", "view": "cam", "frame_index": 2}]
            annotation = {"category_id": 1, "segmentation": {"size": [3, 4], "counts": "324ON"}}
            candidate = np.zeros((3, 4), dtype=bool)
            candidate[0, 0] = True
            records = [
                {"model": "model", "dataset_index": 0, "image_id": 7,
                 "prompt_key": prompt, "actual_side": "left_hand",
                 "target_present": prompt == "left_hand", "detected": prompt == "right_hand",
                 "top_confidence": 0.1 if prompt == "left_hand" else 0.9,
                 "top_dice_with_physical_hand": 0.123,
                 "detected_dice_with_physical_hand": 0.0,
                 "original_mapping": {"rgb_video": "source/rgb.mkv", "frame_index": 2}}
                for prompt in ("left_hand", "right_hand")
            ]
            result = render_montages(
                root=root, output_dir=root / "out", render_indices=[0], images=images,
                annotations_by_image={7: annotation}, records=records,
                masks={("model", 0, prompt): candidate for prompt in ("left_hand", "right_hand")},
                model_labels=["model"],
            )[0]
            with Image.open(result["rgb"]) as exported:
                np.testing.assert_array_equal(np.asarray(exported), rgb)
            with Image.open(result["gt_physical_hand"]) as gt:
                self.assertEqual(gt.size, (4, 3))
                self.assertEqual(set(np.unique(np.asarray(gt))), {0, 255})
            left = result["models"]["model"]["left_hand"]
            with Image.open(left["candidate"]) as raw, Image.open(left["detected"]) as detected:
                self.assertEqual(raw.size, (4, 3))
                self.assertEqual(np.asarray(raw)[0, 0], 255)
                self.assertEqual(np.asarray(detected).sum(), 0)
            self.assertTrue(candidate[0, 0])
            self.assertEqual(left["record"]["top_dice_with_physical_hand"], 0.123)
            self.assertEqual(result["comparison_sheets"]["opposite"]["prompt_target"], "empty")
            self.assertEqual(result["original_mapping"]["frame_index"], 2)
            self.assertTrue(Path(result["comparison_sheets"]["correct"]["path"]).is_file())


if __name__ == "__main__":
    unittest.main()

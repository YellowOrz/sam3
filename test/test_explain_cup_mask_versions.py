"""Small pure-CPU checks for cup diagnosis summaries; no dataset or model."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.explain_cup_mask_versions import diagnose, frame_ranges, geometry


class ExplainCupMaskVersionsTest(unittest.TestCase):
    def test_contiguous_intervals_are_inclusive(self):
        self.assertEqual(frame_ranges([187, 171, 172, 180]), [[171, 172], [180, 180], [187, 187]])
        self.assertEqual(frame_ranges([]), [])

    def test_geometry_is_xy_exclusive_without_treating_ids_as_side(self):
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:3, 2:5] = 2
        result = geometry(mask)
        self.assertEqual(result["pixels"], 6)
        self.assertEqual(result["bbox_xyxy_exclusive"], [2, 1, 5, 3])
        self.assertEqual(result["centroid_xy"], [3., 1.5])

    def test_empty_geometry_is_unknown_bbox_not_fabricated_origin(self):
        result = geometry(np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(result["pixels"], 0)
        self.assertIsNone(result["bbox_xyxy_exclusive"])
        self.assertIsNone(result["centroid_xy"])

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("keep")
            with self.assertRaises(FileExistsError):
                diagnose(output)
            self.assertEqual(output.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

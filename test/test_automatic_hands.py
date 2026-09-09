import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_automatic_hands import draw_outputs, merge_instance_masks


class AutomaticHandsTest(unittest.TestCase):
    def test_merge_accepts_predictor_nchw_masks(self):
        masks = np.zeros((2, 1, 4, 5), dtype=bool)
        masks[0, 0, 1, 1] = True
        masks[1, 0, 2, 3] = True

        merged = merge_instance_masks(masks)

        self.assertEqual(merged.shape, (4, 5))
        self.assertTrue(merged[1, 1])
        self.assertTrue(merged[2, 3])

    def test_merge_returns_none_for_no_instances(self):
        self.assertIsNone(merge_instance_masks([]))

    def test_draw_outputs_preserves_frame_shape(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        mask = np.zeros((1, 1, 4, 5), dtype=bool)
        mask[0, 0, 1:3, 1:4] = True
        outputs = {
            "left_hand": {"current": {"out_binary_masks": mask}},
            "right_hand": {"current": {"out_binary_masks": []}},
        }

        rendered = draw_outputs(frame, outputs)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(rendered.sum(), 0)


if __name__ == "__main__":
    unittest.main()

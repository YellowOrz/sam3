import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import render_realsense_diagnostics as render


class DiagnosticRenderTest(unittest.TestCase):
    def test_portable_unique_plan_ids_and_recording_paths(self):
        valid = {"review_id": "D1", "recording": "black_pen", "frame_index": 1}
        render.validate_samples([valid])
        for bad in ("../escape", "/tmp/out", "a/b", "a\\b", "..", "x:ads"):
            for key in ("review_id", "recording"):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    render.validate_samples([{**valid, key: bad}])
        with self.assertRaises(ValueError):
            render.validate_samples([valid, {**valid, "review_id": "d1"}])

    def test_missing_is_unknown_not_zero_overlap(self):
        mask = np.ones((3, 4), dtype=np.uint8)
        self.assertIsNone(render.overlap_or_unknown({"left_hand": mask}))
        self.assertIsNone(render.overlap_or_unknown({}))
        self.assertEqual(render.overlap_or_unknown({"left_hand": mask, "right_hand": mask*0}), 0)
        self.assertEqual(render.overlap_or_unknown({"left_hand": mask, "right_hand": mask*2}), 12)

    def test_native_dimensions_types_counts_and_rates_checked(self):
        correct = dict(width=640, height=480, pix_fmt="gray", avg_frame_rate="30/1", nb_read_packets="642")
        with patch.object(render, "probe", return_value={"streams": [correct]}):
            render.validate_mask_stream(Path("mask"), 640, 480, 30., 642)
        for changes in ({"width": 480, "height": 640}, {"pix_fmt": "gray16le"}, {"pix_fmt": "rgb24"},
                        {"nb_read_packets": "641"}, {"avg_frame_rate": "15/1"}):
            with self.subTest(changes=changes), patch.object(render, "probe", return_value={"streams": [{**correct, **changes}]}):
                with self.assertRaises(ValueError):
                    render.validate_mask_stream(Path("mask"), 640, 480, 30., 642)

    def test_png_native_labels_and_rgb_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, array in enumerate((np.array([[0, 1, 2, 255]], np.uint8),
                                           np.arange(60, dtype=np.uint8).reshape(4,5,3))):
                path = Path(directory) / f"{index}.png"
                render.save_lossless(path, array)
                with Image.open(path) as image:
                    np.testing.assert_array_equal(np.asarray(image), array)


if __name__ == "__main__":
    unittest.main()

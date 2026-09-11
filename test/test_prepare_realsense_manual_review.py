import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts.prepare_realsense_manual_review import panel, sample_frames, stable_record, verify


class ReviewTests(unittest.TestCase):
    def test_sampling_reproducible_and_order_independent(self):
        counts = {"z": 50, "a": 1, "m": 300}
        first = sample_frames(counts, 20260910)
        self.assertEqual(first, sample_frames(dict(reversed(list(counts.items()))), 20260910))
        self.assertEqual([x["recording"] for x in first], ["a", "m", "z"])
        for item in first:
            self.assertLess(item["frame_index"], counts[item["recording"]])
        self.assertEqual(first[0]["frame_index"], 0)

    def test_invalid_counts_rejected(self):
        for counts in ({}, {"a": 0}, {"a": -1}, {"a": True}, {"a": 2.5}):
            with self.assertRaises(ValueError):
                sample_frames(counts, 1)

    def test_missing_stream_not_black_negative(self):
        rgb = np.zeros((200, 640, 3), np.uint8)
        rgb[:, :, 0] = 17
        empty = np.zeros((200, 640), np.uint8)
        mask = empty.copy()
        mask[100:110, 100:110] = 2
        image = np.asarray(panel(rgb, {"left_hand": empty, "object": mask},
                                 {"review_id": "01", "recording": "scene", "frame_index": 5}))
        np.testing.assert_array_equal(image[82:, :640], rgb)
        self.assertTrue((image[82:, 640:1280] == 0).all())
        self.assertTrue((image[-20:, 1280:1920] == 160).all())
        self.assertTrue((image[182:192, 2020:2030] == 255).all())

    def test_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source"
            path.write_bytes(b"original")
            record = stable_record(path)
            verify([record])
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify([record])


if __name__ == "__main__":
    unittest.main()

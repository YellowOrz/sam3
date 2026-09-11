"""CPU-only contracts for full RealSense export; no source-data mutations."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import prepare_realsense_full_test as full


class MetadataTests(unittest.TestCase):
    def test_exact_full_inventory(self):
        self.assertEqual(len(full.FRAME_COUNTS), 10)
        self.assertEqual(sum(full.FRAME_COUNTS.values()), 6204)
        self.assertEqual(full.FRAME_COUNTS["left_hand"], 660)
        self.assertEqual(full.FRAME_COUNTS["right_hand"], 578)

    def test_missing_is_explicit_group_not_empty(self):
        for left, right, group in ((True, True, "both"), (True, False, "left_only"),
                                   (False, True, "right_only"), (False, False, "none")):
            self.assertEqual(full.reference_group(dict(zip(full.SIDES, (left, right)))), group)
        for invalid in ({"left_hand": True}, {"left_hand": 1, "right_hand": True}):
            with self.assertRaises(ValueError):
                full.reference_group(invalid)

    def test_quality_is_side_specific_and_review_not_excluded(self):
        blocked = {"basket": {622: ["D1_confirmed_wrong_side_reference", "previously_displayed_random_review"],
                               609: ["D4_previously_reviewed_temporal_window_uncertainty"]},
                   "cup": {185: ["D3_historical_version_correspondence_uncertainty_not_proven_current_error"],
                           319: ["D2_confirmed_wrong_side_reference"]}}
        flags = full.quality_flags("basket", 622, blocked)
        self.assertEqual(flags, ["D1_confirmed_wrong_side_reference"])
        self.assertEqual(full.side_quality_flags(flags)["right_hand"], [])
        for recording, frame in (("basket", 609), ("cup", 185), ("cup", 319)):
            flags = full.quality_flags(recording, frame, blocked)
            self.assertEqual(full.side_quality_flags(flags)["left_hand"], [])
            self.assertTrue(full.side_quality_flags(flags)["right_hand"])
        self.assertEqual(full.quality_flags("cup", 0, blocked), [])

    def test_render_keeps_legacy_and_unbiased_midpoints(self):
        old = {name: [0, count - 1] for name, count in full.FRAME_COUNTS.items()
               if name not in full.SIDES}
        selected = full.render_selection(full.FRAME_COUNTS, old)
        self.assertEqual(sum(map(len, selected.values())), 26)
        for name, count in full.FRAME_COUNTS.items():
            self.assertIn((count - 1) // 2, selected[name])
            self.assertTrue(set(old.get(name, [])).issubset(selected[name]))
        with self.assertRaises(ValueError):
            full.render_selection({"x": 4}, {"x": [4]})

    def test_nonzero_union_empty_and_rle(self):
        raw = np.array([[0, 2, 0], [4, 0, 255]], dtype=np.uint8)
        annotation = full.side_annotation(raw, 9, 2, 1)
        self.assertEqual(annotation["area"], 3)
        self.assertEqual(annotation["source_instance_values"], [2, 4, 255])
        np.testing.assert_array_equal(mask_utils.decode(annotation["segmentation"]), raw > 0)
        self.assertIsNone(full.side_annotation(np.zeros_like(raw), 9, 2, 1))
        # Missing sides are not passed into side_annotation by the exporter.
        with self.assertRaises((AttributeError, ValueError)):
            full.side_annotation(None, 9, 2, 1)

    def test_png_and_legacy_match_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {"rgb": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
                      "left_hand": np.array([[0, 1, 2], [0, 0, 255]], dtype=np.uint8)}
            files = {}
            for side, array in arrays.items():
                path = root / f"{side}.png"
                files[side] = {"path": path.name, **full.checked_png(path, array)}
            full.check_legacy_pixels(root, {"files": files}, arrays)
            wrong = dict(arrays, left_hand=np.zeros((2, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "pixel mismatch"):
                full.check_legacy_pixels(root, {"files": files}, wrong)
            Image.fromarray(wrong["left_hand"]).save(root / "left_hand.png")
            with self.assertRaisesRegex(ValueError, "output changed"):
                full.check_legacy_pixels(root, {"files": files}, arrays)

    def test_output_must_be_new(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "NEW output"):
                full.export(path / "source", path / "audit", path / "legacy", path)

    def test_legacy_incomplete_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("READY.json", "manifest.json", "annotations.json", "frozen-plan.json"):
                full.write_json(root / name, {"status": "building"})
            with self.assertRaisesRegex(ValueError, "not READY"):
                full.load_legacy(root)

    def test_legacy_metadata_hash_and_identity_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [{"id": i + 1, "recording_id": "basket", "source_frame_index": i}
                      for i in range(128)]
            documents = {"annotations.json": {"images": images}, "frozen-plan.json": {},
                         "manifest.json": {"image_outputs": [{"image_id": i + 1} for i in range(128)]}}
            for name, value in documents.items():
                full.write_json(root / name, value)
            full.write_json(root / "READY.json", {"status": "complete",
                            "annotations_sha256": full.sha256(root / "annotations.json"),
                            "manifest_sha256": full.sha256(root / "manifest.json"),
                            "frozen_plan_sha256": full.sha256(root / "frozen-plan.json")})
            _, lookup, rows, evidence = full.load_legacy(root)
            self.assertEqual((len(lookup), len(rows), len(evidence)), (128, 128, 4))
            (root / "manifest.json").write_text(json.dumps({"image_outputs": []}))
            with self.assertRaisesRegex(ValueError, "metadata hash changed"):
                full.load_legacy(root)

    def test_d4_window_is_uncertain_not_confirmed_and_other_side_retained(self):
        flag = "D4_previously_reviewed_temporal_window_uncertainty"
        for frame in (570, 608, 610, 645):
            flags = full.quality_flags("basket", frame, {"basket": {frame: [flag]}})
            self.assertEqual(flags, [flag])
            self.assertEqual(full.side_quality_flags(flags)["left_hand"], [])
            self.assertFalse(any("confirmed_wrong" in value for value in flags))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "CPU ffmpeg required")
class StreamingTests(unittest.TestCase):
    def make_video(self, path, gray):
        shape = (3, 4, 6) if gray else (3, 4, 6, 3)
        arrays = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-f", "rawvideo",
                   "-pixel_format", "gray" if gray else "rgb24", "-video_size", "6x4",
                   "-framerate", "30", "-i", "pipe:0", "-c:v", "ffv1", "-threads", "1", str(path)]
        subprocess.run(command, input=arrays.tobytes(), check=True, timeout=20)
        return arrays

    def test_bounded_stream_rgb_and_raw_gray_are_pixel_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            for gray in (False, True):
                path = Path(directory) / f"video-{gray}.mkv"
                original = self.make_video(path, gray)
                iterator = full.stream_frames(path, 3, 30., 6, 4, gray)
                self.assertIs(iter(iterator), iterator)
                for index, (array, pts) in enumerate(iterator):
                    np.testing.assert_array_equal(array, original[index])
                    self.assertLess(abs(pts - index / 30), .0012)

    def test_bad_count_and_pts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mkv"
            self.make_video(path, True)
            for count, fps in ((2, 30.), (3, 25.)):
                with self.assertRaisesRegex(ValueError, "frame count/PTS"):
                    list(full.stream_frames(path, count, fps, 6, 4, True))

    def test_early_close_and_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mkv"
            self.make_video(path, True)
            iterator = full.stream_frames(path, 3, 30., 6, 4, True)
            next(iterator)
            iterator.close()
            with self.assertRaisesRegex(ValueError, "Truncated|extra"):
                list(full.stream_frames(path, 3, 30., 8, 4, True))


if __name__ == "__main__":
    unittest.main()

"""CPU checks for fixed nakehand test sampling and bilateral pseudo-label export."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mask_utils

from scripts.prepare_nakehand_test import (
    DIAGNOSTICS, atomic_json, export_dataset, frame_chunk, select_frames, side_annotation,
    source_record, uniform_indices, verify_sources,
)


class PrepareNakehandTest(unittest.TestCase):
    def test_uniform_indices_fixed_cardinality_endpoints_and_round_half_up(self):
        self.assertEqual(uniform_indices(6, 3), [0, 3, 5])
        self.assertEqual(uniform_indices(100, 100), list(range(100)))
        selected = uniform_indices(3449, 100)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len(set(selected)), 100)
        self.assertEqual((selected[0], selected[-1]), (0, 3448))
        for invalid in ((1, 2), (100, 1), (100, 101), (100, 2.5)):
            with self.assertRaises(ValueError):
                uniform_indices(*invalid)

    def test_side_union_includes_label_two_and_single_pixel_without_area_filter(self):
        raw = np.array([[0, 2, 0], [1, 0, 0]], dtype=np.uint8)
        annotation = side_annotation(raw, 7, 2, 11)
        self.assertEqual(annotation["source_instance_values"], [1, 2])
        self.assertEqual(annotation["area"], 2)
        self.assertEqual(annotation["bbox"], [0.0, 0.0, 2.0, 2.0])
        self.assertTrue(np.array_equal(mask_utils.decode(annotation["segmentation"]), raw > 0))
        raw[1, 0] = 0
        self.assertEqual(side_annotation(raw, 7, 2, 11)["area"], 1)
        self.assertIsNone(side_annotation(np.zeros_like(raw), 7, 2, 11))

    def test_each_side_is_independent_for_two_hands_and_empty(self):
        left = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        right = np.array([[0, 0], [2, 0]], dtype=np.uint8)
        annotations = [side_annotation(left, 0, 1, 0), side_annotation(right, 0, 2, 1)]
        self.assertEqual([item["category_id"] for item in annotations], [1, 2])
        self.assertEqual([item["image_id"] for item in annotations], [0, 0])

    def test_chunk_scoped_labels_keep_frame_boundary(self):
        metadata = {"chunks": [{"index": 1, "start_frame": 0, "end_frame_exclusive": 2000},
                               {"index": 2, "start_frame": 2000, "end_frame_exclusive": 3449}]}
        self.assertEqual(frame_chunk(metadata, 1999)["index"], 1)
        self.assertEqual(frame_chunk(metadata, 2000)["index"], 2)
        with self.assertRaises(ValueError):
            frame_chunk(metadata, 3449)

    def test_legacy_video_mapping_does_not_invent_chunk_identity(self):
        metadata = {"source": {"frame_count": 629}, "object_id_to_label": {"0": 1}}
        mapping = frame_chunk(metadata, 0)
        self.assertEqual(mapping["mapping_scope"], "video")
        self.assertIsNone(mapping["index"])
        self.assertEqual(mapping["object_id_to_label"], {"0": 1})
        self.assertEqual(frame_chunk(metadata, 628)["end_frame_exclusive"], 629)
        with self.assertRaises(ValueError):
            frame_chunk(metadata, 629)

    def test_diagnostics_acceptance_scope_exactly_three_original_frames(self):
        self.assertEqual(DIAGNOSTICS[("nakehandego/20260907_142020", 575)], "A")
        self.assertEqual(DIAGNOSTICS[("nakehandexo/20260907_123926", 1459)], "B")
        self.assertEqual(DIAGNOSTICS[("nakehandego/20260907_142020", 1724)], "C")
        self.assertEqual(len(DIAGNOSTICS), 3)

    @patch("scripts.prepare_nakehand_test.subprocess.run")
    def test_selected_decode_checks_every_pts_and_keeps_frame_order(self, run):
        pts = b"pts_time:0\npts_time:0.333333\npts_time:0.666667\npts_time:1\n"
        run.return_value = subprocess.CompletedProcess([], 0, bytes(range(16)), pts)
        values, timestamps = select_frames(Path("mask.mkv"), [0, 10, 20, 30], 30, 2, 2, True)
        self.assertEqual(values.shape, (4, 2, 2))
        self.assertEqual(values[-1].tolist(), [[12, 13], [14, 15]])
        self.assertEqual(timestamps[-1], 1)
        self.assertIn("select=eq(n\\,0)+eq(n\\,10)+eq(n\\,20)+eq(n\\,30),showinfo", run.call_args.args[0])
        run.return_value.stderr = pts.replace(b"0.333333", b"0.366667")
        with self.assertRaisesRegex(ValueError, "PTS mismatch"):
            select_frames(Path("mask.mkv"), [0, 10, 20, 30], 30, 2, 2, True)

    @patch("scripts.prepare_nakehand_test.subprocess.run")
    def test_selected_decode_rejects_truncation(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, bytes(range(15)), b"pts_time:0\n" * 4)
        with self.assertRaisesRegex(ValueError, "size/count mismatch"):
            select_frames(Path("mask.mkv"), [0, 10, 20, 30], 30, 2, 2, True)

    def test_source_rehash_detects_same_size_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source"
            path.write_bytes(b"original")
            sources = [source_record(path)]
            verify_sources(sources)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                verify_sources(sources)

    def test_atomic_json_publication_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "READY.json"
            atomic_json(path, {"status": "complete"})
            self.assertEqual(path.read_text(), '{\n  "status": "complete"\n}\n')
            self.assertFalse(path.with_name("READY.json.tmp").exists())

    def test_export_cannot_write_source_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                export_dataset(root, root / "output")
            output = Path(temporary) / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export_dataset(root, output)

    def test_end_to_end_export_keeps_primary_and_review_cohorts_separate(self):
        names = ["nakehandego/20260907_134035", "nakehandego/20260907_140713",
                 "nakehandego/20260907_142020", "nakehandego/20260907_144324",
                 "nakehandexo/20260907_123926", "nakehandexo/20260907_131154"]
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "source", Path(temporary) / "output"
            for name in names:
                directory = root / name
                directory.mkdir(parents=True)
                (directory / "metadata.json").write_text("{}")

            def fake_recording(source_root, path):
                count = 2001
                chunks = {"chunks": [{"index": 1, "start_frame": 0, "end_frame_exclusive": count,
                                       "object_id_to_label": {"0": 1, "1": 2}}]}
                return {"recording_id": path.parent.relative_to(source_root).as_posix(),
                        "paths": {"rgb": path.parent / "rgb.mkv", "left": path.parent / "left.mkv",
                                  "right": path.parent / "right.mkv", "metadata": path},
                        "metadata": {"frame_count": count,
                                     "frames": [{"source_frame_index": i + 7, "timestamp": 1000 + i / 30}
                                                for i in range(count)]},
                        "sources": [source_record(path)], "width": 2, "height": 2, "fps": 30,
                        "stream_info": {}, "sidecars": {
                            side: {"metadata": chunks, "metadata_path": str(path), "interactions_path": str(path)}
                            for side in ("left", "right")}}

            def fake_frames(path, selected, fps, width, height, gray):
                if gray:
                    values = np.repeat(np.array([[[0, 2], [1, 0]]], dtype=np.uint8), len(selected), axis=0)
                else:
                    values = np.zeros((len(selected), height, width, 3), dtype=np.uint8)
                return values, [frame / fps for frame in selected]

            with patch("scripts.prepare_nakehand_test.load_recording", side_effect=fake_recording), \
                 patch("scripts.prepare_nakehand_test.select_frames", side_effect=fake_frames), \
                 contextlib.redirect_stdout(io.StringIO()):
                manifest = export_dataset(root, output, samples_per_recording=3)
            coco = json.loads((output / "annotations.json").read_text())
            self.assertEqual(manifest["counts"]["primary_images"], 18)
            self.assertEqual(manifest["counts"]["diagnostic_only_images"], 3)
            self.assertEqual(len(coco["images"]), 21)
            accepted = [image for image in coco["images"] if image["human_review"]["status"] == "accepted"]
            self.assertEqual({image["diagnostic_id"] for image in accepted}, {"A", "B", "C"})
            self.assertTrue(all(not image["primary_test"] for image in accepted))
            self.assertEqual(coco["images"][0]["source_frame_index"], 7)
            self.assertEqual(coco["images"][0]["video_pts_seconds"], 0)
            self.assertEqual(coco["images"][0]["source_timestamp_seconds"], 1000)
            self.assertEqual(json.loads((output / "READY.json").read_text())["status"], "complete")
            self.assertFalse(list(output.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()

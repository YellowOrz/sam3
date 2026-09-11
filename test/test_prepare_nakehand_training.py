"""CPU-only checks for full-frame recording-disjoint nakehand development export."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import prepare_nakehand_training as exporter


class PrepareNakehandTrainingTest(unittest.TestCase):
    def test_split_inventory_counts_and_global_ids_are_disjoint_complete(self):
        plan = exporter.split_plan(exporter.FRAME_COUNTS)
        self.assertEqual(plan["total_images"], 18498)
        self.assertEqual({key: item["images"] for key, item in plan["splits"].items()},
                         {"train": 9092, "val": 3449, "development_holdout": 5957})
        self.assertEqual(plan["splits"]["development_holdout"]["coco_split"], "test")
        ids = [offset + frame for item in plan["recordings"].values()
               for offset in [item["global_image_id_offset"]] for frame in range(item["frame_count"])]
        self.assertEqual(ids, list(range(18498)))
        with self.assertRaisesRegex(ValueError, "inventory"):
            exporter.split_plan({**exporter.FRAME_COUNTS, "new": 1})
        with patch.dict(exporter.SPLIT_RECORDINGS, {"val": exporter.SPLIT_RECORDINGS["train"]}):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                exporter.split_plan(exporter.FRAME_COUNTS)

    def test_pts_checks_all_indices_not_capture_timestamps(self):
        log = "[showinfo] n: 0 pts: 0 pts_time:0\n[showinfo] n: 1 pts: 33 pts_time:0.0333333\n"
        self.assertEqual(exporter.parse_pts(log, 2, 30), [0, .0333333])
        with self.assertRaisesRegex(ValueError, "count"):
            exporter.parse_pts(log, 3, 30)
        with self.assertRaisesRegex(ValueError, "index/PTS"):
            exporter.parse_pts(log.replace("n: 1", "n: 2"), 2, 30)
        with self.assertRaisesRegex(ValueError, "index/PTS"):
            exporter.parse_pts(log.replace("0.0333333", "0.0666667"), 2, 30)

    def test_identity_map_rejects_hidden_source_offset(self):
        recording = {"recording_id": "fixture", "metadata": {"frame_count": 2, "frames": [
            {"video_frame_index": 0, "source_frame_index": 0},
            {"video_frame_index": 1, "source_frame_index": 1}]}}
        exporter.recording_identity(recording)
        recording["metadata"]["frames"][1]["source_frame_index"] = 7
        with self.assertRaisesRegex(ValueError, "identity"):
            exporter.recording_identity(recording)

    def test_never_creates_output_inside_source_or_over_existing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                exporter.freeze_plan(root, root / "data", root / "prior.json")
            existing = Path(temporary) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                exporter.freeze_plan(root, existing, root / "prior.json")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_real_sequential_cpu_decode_matches_all_pixels_and_pts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.mkv"
            arrays = np.zeros((4, 6, 8), dtype=np.uint8)
            arrays[1, 1, 2] = 2
            arrays[2, 2:4, 3:5] = 1
            arrays[3, :, :] = 2
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
                       "-f", "rawvideo", "-pixel_format", "gray", "-video_size", "8x6", "-framerate", "30",
                       "-i", "pipe:0", "-frames:v", "4", "-c:v", "ffv1", "-threads", "1", str(path)]
            subprocess.run(command, input=arrays.tobytes(), check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=30)
            with exporter.VideoFrames(path, 4, 30, 8, 6, True) as decoder:
                for expected in arrays:
                    self.assertTrue(np.array_equal(decoder.read(), expected))
                self.assertEqual(decoder.finish(), [0, .033, .067, .1])
            with exporter.VideoFrames(path, 3, 30, 8, 6, True) as decoder:
                for _ in range(3):
                    decoder.read()
                with self.assertRaisesRegex(ValueError, "fewer/more"):
                    decoder.finish()

    def _fixture(self, root, output):
        counts = {name: 3 for name in exporter.FRAME_COUNTS}
        recordings = {}
        for name in sorted(counts):
            directory = root / name
            directory.mkdir(parents=True)
            paths = {}
            sources = []
            for index in range(10):
                path = directory / f"source-{index}"
                path.write_bytes(f"{name}/{index}".encode())
                sources.append(exporter.source_record(path))
                if index < 4:
                    paths[("rgb", "left", "right", "metadata")[index]] = path
            chunk = {"chunks": [{"index": 1, "start_frame": 0, "end_frame_exclusive": 2,
                                  "object_id_to_label": {"0": 1, "1": 2}},
                                 {"index": 2, "start_frame": 2, "end_frame_exclusive": 3,
                                  "object_id_to_label": {"0": 1, "1": 2}}]}
            recordings[name] = {"recording_id": name, "paths": paths, "sources": sources,
                                "metadata": {"frame_count": 3, "frames": [
                                    {"video_frame_index": i, "source_frame_index": i, "timestamp": 1000 + i / 30}
                                    for i in range(3)]}, "width": 3, "height": 2, "fps": 30,
                                "stream_info": {}, "sidecars": {
                                    side: {"metadata": chunk, "metadata_path": str(paths["metadata"]),
                                           "interactions_path": str(paths["metadata"])} for side in ("left", "right")}}
        with patch.object(exporter, "FRAME_COUNTS", counts):
            plan = exporter.split_plan(counts)
        output.mkdir()
        prior = root / "prior.json"
        prior.write_text("{}")
        plan.update(format="nakehand-development-split-plan-v1", root=str(root), output=str(output),
                    sources=[item for recording in recordings.values() for item in recording["sources"]],
                    implementation_sources=[], prior_exposure={"source": exporter.source_record(prior)},
                    png_compress_level=3, person_session_camera_relationship="unknown",
                    human_review="not all manually certified", development_holdout_policy="no model selection")
        exporter.atomic_json(output / "frozen-plan.json", plan)
        return counts, recordings, output / "frozen-plan.json"

    @staticmethod
    def _fake_decoder(path, count, fps, width, height, gray):
        class Decoder:
            def __init__(self):
                self.index = 0

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                pass

            def read(self):
                index = self.index
                self.index += 1
                array = np.zeros((height, width) if gray else (height, width, 3), dtype=np.uint8)
                if gray:
                    if index == 2 or (index == 1 and path.name == "source-1"):
                        array[0, 1] = 2
                else:
                    array[:] = index * 20
                return array

            def finish(self):
                return [index / fps for index in range(count)]
        return Decoder()

    def test_full_export_keeps_empty_single_both_and_chunk_two_then_detects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "source", Path(temporary) / "output"
            counts, recordings, plan_path = self._fixture(root, output)
            with patch.object(exporter, "FRAME_COUNTS", counts), \
                 patch.object(exporter, "load_recording", side_effect=lambda root, path: recordings[path.parent.relative_to(root).as_posix()]), \
                 patch.object(exporter, "VideoFrames", side_effect=self._fake_decoder), \
                 contextlib.redirect_stdout(io.StringIO()):
                manifest = exporter.export_plan(plan_path)
            self.assertEqual(manifest["total_images"], 18)
            ready = json.loads((output / "READY.json").read_text())
            self.assertEqual(ready["status"], "complete")
            ids = []
            for split, number in (("train", 9), ("val", 3), ("development_holdout", 6)):
                coco = json.loads((output / split / "annotations.json").read_text())
                receipt = ready["splits"][split]
                self.assertEqual(receipt["ready_sha256"], exporter.sha256(output / split / "READY.json"))
                self.assertEqual(len(coco["images"]), number)
                self.assertEqual(receipt["counts"]["empty_images"], number // 3)
                self.assertEqual(receipt["counts"]["one_hand_images"], number // 3)
                self.assertEqual(receipt["counts"]["two_hand_images"], number // 3)
                self.assertEqual(coco["info"]["split"], exporter.COCO_SPLITS[split])
                self.assertTrue(all(image["primary_test"] for image in coco["images"]))
                self.assertTrue(all(image["human_review"]["status"] != "accepted" for image in coco["images"]))
                for annotation in coco["annotations"]:
                    self.assertEqual(annotation["source_instance_values"], [2])
                    if annotation["image_id"] % 3 == 2:
                        self.assertEqual(annotation["source_mask"]["chunk"]["index"], 2)
                    self.assertIsNotNone(annotation["source_mask"]["video_pts_seconds"])
                self.assertEqual(exporter.validate_export(output / split)["pngs"], number * 5)
                ids.extend(image["id"] for image in coco["images"])
            self.assertEqual(sorted(ids), list(range(18)))
            with self.assertRaises(FileExistsError):
                exporter.export_plan(plan_path)
            coco = json.loads((output / "train" / "annotations.json").read_text())
            rgb = output / "train" / coco["images"][0]["file_name"]
            rgb.write_bytes(b"tamper")
            with self.assertRaisesRegex(ValueError, "PNG hash"):
                exporter.validate_export(output / "train")

    def test_source_mutation_blocks_ready_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, output = Path(temporary) / "source", Path(temporary) / "output"
            counts, recordings, plan_path = self._fixture(root, output)
            first = next(iter(recordings.values()))["paths"]["rgb"]
            first.write_bytes(b"changed")
            with patch.object(exporter, "FRAME_COUNTS", counts):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    exporter.export_plan(plan_path)
            self.assertFalse((output / "READY.json").exists())
            self.assertFalse((output / "train").exists())


if __name__ == "__main__":
    unittest.main()

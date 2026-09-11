"""CPU-only interrupted-export recovery and immutable-publication checks."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import prepare_nakehand_training as exporter
from scripts import recover_nakehand_export as recovery
import test_prepare_nakehand_training as original_tests


class RecoverNakehandExportTest(unittest.TestCase):
    def fixture(self, temporary):
        root, output = Path(temporary) / "source", Path(temporary) / "output"
        helper = original_tests.PrepareNakehandTrainingTest()
        counts, recordings, plan_path = helper._fixture(root, output)
        plan = recovery.read_json(plan_path)
        plan["implementation_sources"] = [exporter.source_record(Path(exporter.__file__).with_name(name))
                                          for name in exporter.HELPERS]
        snapshot = output / "implementation-snapshot"
        snapshot.mkdir()
        for item in plan["implementation_sources"]:
            shutil.copyfile(item["path"], snapshot / Path(item["path"]).name)
        exporter.atomic_json(plan_path, plan)
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(exporter, "FRAME_COUNTS", counts))
        stack.enter_context(patch.object(exporter, "load_recording", side_effect=lambda root, path: recordings[path.parent.relative_to(root).as_posix()]))
        stack.enter_context(patch.object(exporter, "VideoFrames", side_effect=helper._fake_decoder))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.addCleanup(stack.close)
        digest = exporter.sha256(plan_path)
        for split in ("train", "val"):
            exporter.export_split(output, split, plan, digest, recordings)
        partial = output / "development_holdout"
        partial.mkdir()
        for folder in ("images", "source-masks", "reference-masks"):
            (partial / folder).mkdir()
        return output, plan, plan_path, digest

    def test_exact_source_pixels_required_for_reuse_and_copies_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            partial, staged = Path(temporary) / "partial", Path(temporary) / "staged"
            partial.mkdir()
            staged.mkdir()
            array = np.zeros((2, 3, 3), dtype=np.uint8)
            exporter.save_png(partial / "good.png", array, 3)
            exporter.save_png(partial / "wrong.png", np.ones_like(array), 3)
            (partial / "truncated.png").write_bytes(b"partial")
            old = {path.name: path.read_bytes() for path in partial.iterdir()}
            writer = recovery.ReusingPNGWriter(partial, staged, exporter.save_png)
            for name in ("good.png", "wrong.png", "truncated.png", "missing.png"):
                writer(staged / name, array, 3)
                self.assertTrue(np.array_equal(np.asarray(Image.open(staged / name)), array))
            self.assertEqual(len(writer.reused), 1)
            self.assertEqual(writer.new, 3)
            self.assertEqual(len(writer.unusable), 2)
            self.assertNotEqual((partial / "good.png").stat().st_ino, (staged / "good.png").stat().st_ino)
            self.assertEqual(old, {path.name: path.read_bytes() for path in partial.iterdir()})
            with self.assertRaises(FileExistsError):
                writer(staged / "good.png", array, 3)

    def test_recovery_preserves_partial_and_published_splits_then_publishes_all(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, plan, plan_path, digest = self.fixture(temporary)
            partial = output / "development_holdout"
            first_recording = plan["splits"]["development_holdout"]["recordings"][0]
            basename = first_recording.replace("/", "__") + "__frame-000000"
            good = partial / "images" / f"{basename}.png"
            bad = partial / "source-masks" / f"{basename}__left_instance_raw.png"
            exporter.save_png(good, np.zeros((2, 3, 3), dtype=np.uint8), 3)
            bad.write_bytes(b"interrupted partial PNG")
            (partial / "original-note.txt").write_text("must survive unchanged")
            before = {str(path.relative_to(output)): (path.read_bytes(), path.stat().st_mtime_ns)
                      for split in ("train", "val") for path in (output / split).rglob("*") if path.is_file()}
            manifest = recovery.recover(plan_path, digest)
            ready = recovery.read_json(output / "READY.json")
            self.assertEqual(ready["status"], "complete")
            self.assertEqual(manifest["total_images"], 18)
            receipt = recovery.read_json(manifest["recovery_receipt"])
            self.assertEqual(receipt["reused_pngs"], 1)
            self.assertEqual(receipt["newly_encoded_pngs"], 29)
            self.assertEqual(len(receipt["unusable_partial_pngs"]), 1)
            self.assertTrue(receipt["holdout_fresh_source_decode_all_frames"])
            self.assertFalse(receipt["published_validation"]["train"]["fresh_source_video_decode"])
            backup = Path(manifest["partial_backup"])
            self.assertEqual((backup / "source-masks" / bad.name).read_bytes(), b"interrupted partial PNG")
            self.assertEqual((backup / "original-note.txt").read_text(), "must survive unchanged")
            self.assertNotEqual((backup / "images" / good.name).stat().st_ino, good.stat().st_ino)
            for relative, expected in before.items():
                path = output / relative
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
            self.assertEqual(exporter.validate_export(partial, digest)["images"], 6)
            with self.assertRaisesRegex(FileExistsError, "already published"):
                recovery.recover(plan_path, digest)

    def test_plan_hash_tamper_and_implementation_tamper_block_before_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, unused, plan_path, digest = self.fixture(temporary)
            with self.assertRaisesRegex(ValueError, "explicitly expected"):
                recovery.recover(plan_path, "0" * 64)
            self.assertEqual(list(output.glob("recovery-*")), [])
            snapshot = output / "implementation-snapshot" / exporter.HELPERS[0]
            snapshot.write_text("changed")
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                recovery.recover(plan_path, digest)
            self.assertEqual(list(output.glob("recovery-*")), [])

    def test_published_corrupt_png_blocks_root_ready_and_preserves_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, unused, plan_path, digest = self.fixture(temporary)
            partial = output / "development_holdout"
            note = partial / "preserve.txt"
            note.write_text("do not modify")
            first = next((output / "train" / "images").glob("*.png"))
            first.write_bytes(b"tampered published PNG")
            with self.assertRaisesRegex(ValueError, "PNG hash"):
                recovery.recover(plan_path, digest)
            self.assertFalse((output / "READY.json").exists())
            self.assertEqual(note.read_text(), "do not modify")
            self.assertEqual(first.read_bytes(), b"tampered published PNG")
            self.assertEqual(recovery.read_json(next(output.glob("recovery-*/recovery.json")))["status"], "failed")

    def test_published_allocation_tamper_rejected_even_if_hashes_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, plan, unused, digest = self.fixture(temporary)
            directory = output / "train"
            coco = recovery.read_json(directory / "annotations.json")
            coco["images"][0]["frame_index"] = 123
            exporter.atomic_json(directory / "annotations.json", coco)
            manifest = recovery.read_json(directory / "manifest.json")
            manifest["annotations_sha256"] = exporter.sha256(directory / "annotations.json")
            exporter.atomic_json(directory / "manifest.json", manifest)
            ready = recovery.read_json(directory / "READY.json")
            ready.update(annotations_sha256=manifest["annotations_sha256"], manifest_sha256=exporter.sha256(directory / "manifest.json"))
            exporter.atomic_json(directory / "READY.json", ready)
            with self.assertRaisesRegex(ValueError, "allocation"):
                recovery.validate_published(output, "train", plan, digest)

    def test_source_mutation_rejected_before_output_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, plan, plan_path, digest = self.fixture(temporary)
            Path(plan["sources"][0]["path"]).write_bytes(b"changed original")
            with self.assertRaisesRegex(ValueError, "Source changed"):
                recovery.recover(plan_path, digest)
            self.assertEqual(list(output.glob("recovery-*")), [])


if __name__ == "__main__":
    unittest.main()

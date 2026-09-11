from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import apply_nakehand_review_exclusion as exclusion


def coco_fixture():
    target = {key: value for key, value in exclusion.TARGET.items() if key != "image_id"}
    target.update(id=4713, frame_index=0)
    second = {**target, "id": 4714, "frame_index": 1, "source_frame_index": 1,
              "file_name": "images/nakehandego__20260907_142020__frame-000001.png"}
    return {"images": [target, second], "annotations": [
        {"id": 99, "image_id": 4713, "category_id": 2},
        {"id": 101, "image_id": 4714, "category_id": 1}], "categories": [{"id": 1}, {"id": 2}]}


def records_fixture(coco, model="baseline"):
    return [{"model": model, "image_id": image["id"], "dataset_index": index,
             "observed_coco_image_id": image["id"], "identity_verified": True,
             "file_name": image["file_name"], "recording_id": image["recording_id"],
             "prompt_key": side, "retained_metric": 0.123}
            for index, image in enumerate(coco["images"])
            for side in ("left_hand", "right_hand")]


class ReviewExclusionTests(unittest.TestCase):
    def fixture(self, directory):
        root = directory / "source"
        hashes = {}
        for split in exclusion.SPLITS:
            path = root / split / "annotations.json"
            path.parent.mkdir(parents=True)
            coco = coco_fixture() if split == "val" else {"images": [], "annotations": []}
            path.write_text(json.dumps(coco))
            hashes[split] = exclusion.sha256(path)
        manifest = directory / "request.json"
        manifest.write_text(json.dumps({"format": exclusion.FORMAT,
                                       "exclude": [exclusion.TARGET],
                                       "source_annotations_sha256": hashes}))
        return root, manifest

    def test_filter_removes_associated_annotations_without_mutation_or_renumbering(self):
        source = coco_fixture()
        before = deepcopy(source)
        result = exclusion.filter_coco(source)
        self.assertEqual(source, before)
        self.assertEqual([image["id"] for image in result["images"]], [4714])
        self.assertEqual(result["annotations"], [before["annotations"][1]])
        self.assertEqual(result["categories"], before["categories"])

    def test_exact_identity_and_no_match_rejected(self):
        for key, value in (("id", 999), ("recording_id", "other"),
                           ("source_frame_index", 2), ("file_name", "other.png")):
            source = coco_fixture()
            source["images"][0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                exclusion.filter_coco(source)
        source = coco_fixture()
        source["images"].append(deepcopy(source["images"][0]))
        with self.assertRaises(ValueError):
            exclusion.filter_coco(source)

    def test_records_require_exact_both_sides_and_preserve_rows(self):
        coco = coco_fixture()
        rows = records_fixture(coco)
        before = deepcopy(rows)
        self.assertEqual(exclusion.filter_records(rows, coco, "baseline"), rows[2:])
        self.assertEqual(rows, before)
        for invalid in (rows[1:], rows + [rows[0]], rows[:-1]):
            with self.assertRaises(ValueError):
                exclusion.filter_records(invalid, coco, "baseline")
        for key, value in (("file_name", "wrong"), ("recording_id", "wrong"),
                           ("dataset_index", 9), ("identity_verified", False)):
            invalid = deepcopy(rows)
            invalid[0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                exclusion.filter_records(invalid, coco, "baseline")

    def test_hash_pins_and_cross_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, manifest = self.fixture(directory)
            train = root / "train/annotations.json"
            train.write_text(json.dumps(coco_fixture()))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                exclusion.apply_exclusion(root, directory / "out", manifest)
            request = json.loads(manifest.read_text())
            request["source_annotations_sha256"]["train"] = exclusion.sha256(train)
            manifest.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "unexpectedly appears in train"):
                exclusion.apply_exclusion(root, directory / "out", manifest)
            self.assertFalse((directory / "out").exists())

    def test_end_to_end_sources_unchanged_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, manifest = self.fixture(directory)
            before = {path: exclusion.sha256(path) for path in root.rglob("*.json")}
            evaluation = directory / "evaluation"
            evaluation.mkdir()
            records = evaluation / "baseline.json"
            records.write_text(json.dumps(records_fixture(coco_fixture())))
            summary = {"status": "complete", "full_val_evaluated": True,
                       "annotations_sha256": before[root / "val/annotations.json"],
                       "evaluated_image_ids": [4713, 4714], "evaluated_dataset_indices": [0, 1],
                       "record_files": {"baseline": {"path": str(records),
                                                     "sha256": exclusion.sha256(records)}}}
            (evaluation / "summary.json").write_text(json.dumps(summary))
            out = directory / "out"
            result = exclusion.apply_exclusion(root, out, manifest, [evaluation])
            self.assertEqual(result["filtered_images"], 1)
            self.assertEqual(result["filtered_annotations"], 1)
            self.assertFalse(result["standalone_dataset_root"])
            self.assertFalse(result["metrics_recomputed"])
            self.assertEqual(result["record_sets"][0]["filtered_queries"], 2)
            self.assertEqual(before, {path: exclusion.sha256(path) for path in before})
            self.assertTrue(json.loads((out / "receipt.json").read_text())["source_files_verified_unchanged"])
            with self.assertRaises(ValueError):
                exclusion.apply_exclusion(root, out, manifest)
            summary["status"] = "completed"
            summary["input_sha256"] = {str(root / "val/annotations.json"):
                                        summary.pop("annotations_sha256")}
            (evaluation / "summary.json").write_text(json.dumps(summary))
            boundary = exclusion.apply_exclusion(root, directory / "boundary", manifest, [evaluation])
            self.assertEqual(boundary["record_sets"][0]["filtered_queries"], 2)
            dangling = directory / "dangling"
            dangling.symlink_to(directory / "does-not-exist", target_is_directory=True)
            with self.assertRaises(ValueError):
                exclusion.apply_exclusion(root, dangling, manifest)

    def test_eval_hash_mismatch_and_manifest_widening_rejected(self):
        request = {"format": exclusion.FORMAT, "exclude": [exclusion.TARGET, exclusion.TARGET],
                   "source_annotations_sha256": {split: "a" * 64 for split in exclusion.SPLITS}}
        with self.assertRaises(ValueError):
            exclusion.validate_manifest(request)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root, manifest = self.fixture(directory)
            evaluation = directory / "evaluation"
            evaluation.mkdir()
            records = evaluation / "baseline.json"
            records.write_text(json.dumps(records_fixture(coco_fixture())))
            (evaluation / "summary.json").write_text(json.dumps({
                "status": "complete", "full_val_evaluated": True,
                "annotations_sha256": exclusion.sha256(root / "val/annotations.json"),
                "evaluated_image_ids": [4713, 4714], "evaluated_dataset_indices": [0, 1],
                "record_files": {"baseline": {"path": str(records), "sha256": "0" * 64}}}))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                exclusion.apply_exclusion(root, directory / "out", manifest, [evaluation])
            self.assertFalse((directory / "out").exists())


if __name__ == "__main__":
    unittest.main()

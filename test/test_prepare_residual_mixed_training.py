"""Synthetic, CPU-only publication contracts; never publish the real dataset."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from scripts import prepare_residual_mixed_training as prepare


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")


def _coco(directory, split, image_ids, *, dataset, recording=None):
    coco = {"info": {"split": split}, "categories": deepcopy(prepare.CATEGORIES), "images": [], "annotations": []}
    for index, image_id in enumerate(image_ids):
        image = {"id": image_id, "file_name": f"images/rgb-{image_id}.png", "width": 2, "height": 2,
                 "source": dataset, "sequence": f"subject-{index % 2 + 1:02d}_sequence",
                 "view": "camera", "frame_index": index}
        if recording is not None:
            image.update(recording_id=recording, source_frame_index=index, dataset_role=split,
                         source_rgb_path="/historical/video.mkv", human_review={"scope": "not all certified"})
        coco["images"].append(image)
        rgb = directory / image["file_name"]
        rgb.parent.mkdir(parents=True, exist_ok=True)
        # Only file identity/hashes are in scope here; no decoder or GPU needed.
        rgb.write_bytes(f"synthetic RGB {dataset} {split} {image_id}".encode())
        for side in ((), (1,), (2,), (1, 2))[index % 4]:
            coco["annotations"].append({"id": image_id * 2 + side - 1, "image_id": image_id,
                                        "category_id": side, "area": 4, "bbox": [0, 0, 2, 2], "iscrowd": 0,
                                        "segmentation": {"size": [2, 2], "counts": [0, 4]},
                                        "provenance": "original_reference"})
    return coco


def source_fixture(directory):
    dex, nake = directory / "dex", directory / "nake"
    for split, ids in (("train", [*range(10), 4713]), ("val", [0, 1, 2])):
        coco = _coco(dex / split, split, ids, dataset="dexycb")
        manifest_path = dex / "manifests" / f"{split}_views.json"
        _write_json(manifest_path, {"split": split, "fixture": True})
        coco["info"]["source_manifest"] = str(manifest_path)
        _write_json(dex / split / "annotations.json", coco)
    plan = {"format": "nakehand-development-split-plan-v1", "output": str(nake), "total_images": 6,
            "splits": {split: {"images": 2} for split in prepare.NAKE_SPLITS}}
    _write_json(nake / "frozen-plan.json", plan)
    plan_hash = prepare.sha256(nake / "frozen-plan.json")
    receipts = {}
    inventory = (("train", [0, 1], "nakehandego/20260907_134035"),
                 ("val", [4713, 4714], prepare.EXCLUDED_FRAME["recording"]),
                 ("development_holdout", [2, 3], "nakehandexo/20260907_131154"))
    for split, ids, recording in inventory:
        split_dir = nake / split
        coco = _coco(split_dir, split, ids, dataset="nakehand", recording=recording)
        coco["info"]["frozen_plan_sha256"] = plan_hash
        _write_json(split_dir / "annotations.json", coco)
        counts = prepare._coco_counts(coco)
        manifest = {"status": "complete", "sources_unchanged": True, "frozen_plan_sha256": plan_hash,
                    "annotations_sha256": prepare.sha256(split_dir / "annotations.json"), "counts": counts,
                    "image_outputs": [{"image_id": image["id"], "files": {"rgb": {
                        "path": image["file_name"], "sha256": prepare.sha256(split_dir / image["file_name"])}}}
                                      for image in coco["images"]]}
        _write_json(split_dir / "manifest.json", manifest)
        receipt = {"status": "complete", "annotations_sha256": manifest["annotations_sha256"],
                   "manifest_sha256": prepare.sha256(split_dir / "manifest.json"),
                   "frozen_plan_sha256": plan_hash, "counts": counts}
        _write_json(split_dir / "READY.json", receipt)
        receipts[split] = {**receipt, "ready_sha256": prepare.sha256(split_dir / "READY.json"), "directory": split}
    manifest = {"status": "complete", "sources_unchanged": True, "total_images": 6,
                "frozen_plan_sha256": plan_hash, "splits": receipts}
    _write_json(nake / "manifest.json", manifest)
    _write_json(nake / "READY.json", {"status": "complete", "splits": receipts,
                                     "frozen_plan_sha256": plan_hash,
                                     "manifest_sha256": prepare.sha256(nake / "manifest.json")})
    return dex, nake


class MixedPreparationTests(unittest.TestCase):
    def test_full_train_preserves_validation_and_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dex, nake = source_fixture(root)
            half = prepare.prepare_mixed_training(dex, nake, root / 'half')
            full = prepare.prepare_mixed_training(dex, nake, root / 'full', dex_selection='all')
            self.assertEqual(full['selection_counts']['dex_train_selected'], 11)
            self.assertEqual(full['selection_counts']['nake_selected'], 5)
            self.assertEqual(full['splits']['val']['annotations_sha256'], half['splits']['val']['annotations_sha256'])
            prepare.validate_publication(root / 'full')

    def test_stratified_exact_half_order_independent_and_private_rng(self):
        images = [{"id": index, "sequence": f"subject-{index % 3 + 1:02d}_seq"} for index in range(37)]
        annotations = [{"image_id": index, "category_id": 1 + index % 2} for index in range(37) if index % 4]
        rng_state = random.getstate()
        selected, strata = prepare.stratified_half(images, annotations)
        self.assertEqual(random.getstate(), rng_state)
        reverse, reverse_strata = prepare.stratified_half(list(reversed(images)), list(reversed(annotations)))
        self.assertEqual((selected, strata), (reverse, reverse_strata))
        self.assertEqual(len(selected), 18)
        self.assertEqual(len(set(selected)), 18)
        self.assertEqual(sum(row["available"] for row in strata), 37)
        self.assertEqual(sum(row["selected"] for row in strata), 18)
        self.assertIn("empty", {row["side"] for row in strata})
        self.assertTrue(all(row["subject"].startswith("subject-") for row in strata))
        self.assertNotEqual(selected, prepare.stratified_half(images, annotations, seed=124)[0])
        for row in strata:
            ideal = row["available"] * 18 / 37
            self.assertLess(abs(row["selected"] - ideal), 1)

    def test_publish_deterministic_selection_unique_ids_exact_exclusion_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dex, nake = source_fixture(directory)
            output, second = directory / "derived", directory / "derived-again"
            result = prepare.prepare_mixed_training(dex, nake, output)
            again = prepare.prepare_mixed_training(dex, nake, second)
            self.assertEqual(result["selection"], again["selection"])
            for split in ("train", "val"):
                self.assertEqual(result["splits"][split]["annotations_sha256"], again["splits"][split]["annotations_sha256"])
            self.assertEqual(result["selection_counts"], {
                "dex_train_available": 11, "dex_train_selected": 5, "nake_source_splits": {
                    "train": 2, "val": 2, "development_holdout": 2}, "nake_available": 6,
                "nake_selected": 5, "nake_excluded": 1, "train_images": 10, "dex_val_images": 3})
            train = json.loads((output / "train/annotations.json").read_text())
            val = json.loads((output / "val/annotations.json").read_text())
            ids = [image["id"] for coco in (train, val) for image in coco["images"]]
            annotation_ids = [row["id"] for coco in (train, val) for row in coco["annotations"]]
            self.assertEqual(ids, list(range(1, 14)))
            self.assertEqual(len(annotation_ids), len(set(annotation_ids)))
            nake_images = [image for image in train["images"] if image["source_dataset"] == "nakehand"]
            self.assertEqual({image["source_split"] for image in nake_images}, set(prepare.NAKE_SPLITS))
            self.assertEqual({image["source_image_id"] for image in nake_images}, {0, 1, 2, 3, 4714})
            self.assertEqual(result["excluded_frames"], [prepare.EXCLUDED_FRAME])
            self.assertTrue(all(image["source_dataset"] == "dexycb" and image["source_split"] == "val" for image in val["images"]))
            self.assertTrue(result["no_independent_nakehand_validation"])
            self.assertEqual(result["source_READY_verification"]["dexycb"]["status"], "legacy_source_without_READY")
            self.assertEqual(len(result["source_READY_verification"]["nakehand"]["splits"]), 3)
            for split, coco in (("train", train), ("val", val)):
                for image in coco["images"]:
                    link = output / split / image["file_name"]
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(str(link.resolve()), image["source_rgb_path"])
                    self.assertEqual(prepare.sha256(link), image["source_rgb_sha256"])
                    self.assertEqual(image["dataset_role"], split)
                    self.assertFalse(image["primary_test"])
                    self.assertIn("source_annotations_sha256", image["provenance"])
                    self.assertIn("recording", image)
                    self.assertIn("frame", image)
            approval = json.loads((output / "training-approval.json").read_text())
            self.assertTrue(approval["train"]["exhaustive_hand_labels"])
            self.assertIn("not a claim", approval["label_limitations"])
            self.assertEqual(approval["test_policy"], prepare.TEST_POLICY)
            self.assertEqual(prepare.validate_publication(output), result)
            self.assertFalse((output / "test").exists())

    def test_output_existing_or_inside_sources_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dex, nake = source_fixture(directory)
            with self.assertRaises(FileExistsError):
                prepare.prepare_mixed_training(dex, nake, dex)
            with self.assertRaises(ValueError):
                prepare.prepare_mixed_training(dex, nake, dex / "new-output")
            self.assertFalse((dex / "new-output").exists())
            dangling = directory / "dangling"
            dangling.symlink_to(directory / "not-created")
            with self.assertRaises(FileExistsError):
                prepare.prepare_mixed_training(dex, nake, dangling)
            self.assertFalse((directory / "not-created").exists())

    def test_missing_and_partial_nake_ready_rejected_before_publication(self):
        for split in (None, *prepare.NAKE_SPLITS):
            for status in (None, "in_progress"):
                with self.subTest(split=split, status=status), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    dex, nake = source_fixture(directory)
                    path = nake / (split or "") / "READY.json"
                    if status is None:
                        path.unlink()
                    else:
                        data = json.loads(path.read_text())
                        data["status"] = status
                        _write_json(path, data)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        prepare.prepare_mixed_training(dex, nake, directory / "out")
                    self.assertFalse((directory / "out").exists())

    def test_nake_annotations_or_rgb_sha_changed_rejected(self):
        for kind in ("annotations", "rgb"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                dex, nake = source_fixture(directory)
                path = nake / "train" / ("annotations.json" if kind == "annotations" else "images/rgb-0.png")
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA|binding"):
                    prepare.prepare_mixed_training(dex, nake, directory / "out")
                self.assertFalse((directory / "out/READY.json").exists())

    def test_postpublication_source_annotation_or_rgb_mutation_rejected(self):
        for kind in ("annotations", "rgb"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                dex, nake = source_fixture(directory)
                output = directory / "out"
                prepare.prepare_mixed_training(dex, nake, output)
                path = dex / "train/annotations.json" if kind == "annotations" else nake / "train/images/rgb-0.png"
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA changed"):
                    prepare.validate_publication(output)

    def test_source_symlink_escape_rejected_and_no_ready_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dex, nake = source_fixture(directory)
            foreign = directory / "outside.png"
            foreign.write_bytes(b"outside supplied roots")
            coco = json.loads((dex / "train/annotations.json").read_text())
            selected = prepare.stratified_half(coco["images"], coco["annotations"])[0]
            image = next(row for row in coco["images"] if row["id"] == selected[0])
            rgb = dex / "train" / image["file_name"]
            rgb.unlink()
            rgb.symlink_to(foreign)
            with self.assertRaisesRegex(ValueError, "escapes"):
                prepare.prepare_mixed_training(dex, nake, directory / "out")
            self.assertFalse((directory / "out/READY.json").exists())

    def test_derived_symlink_retarget_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dex, nake = source_fixture(directory)
            output = directory / "out"
            prepare.prepare_mixed_training(dex, nake, output)
            coco = json.loads((output / "train/annotations.json").read_text())
            link = output / "train" / coco["images"][0]["file_name"]
            foreign = directory / "outside.png"
            foreign.write_bytes(b"different")
            link.unlink()
            link.symlink_to(foreign)
            with self.assertRaisesRegex(ValueError, "escapes"):
                prepare.validate_publication(output)

    def test_validation_failure_leaves_partial_output_without_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dex, nake = source_fixture(directory)
            output = directory / "out"
            with mock.patch.object(prepare, "validate_publication", side_effect=ValueError("validation failed")):
                with self.assertRaisesRegex(ValueError, "validation failed"):
                    prepare.prepare_mixed_training(dex, nake, output)
            self.assertTrue((output / "manifest.json").exists())
            self.assertFalse((output / "READY.json").exists())
            self.assertFalse((output / "train/READY.json").exists())
            with self.assertRaises(FileNotFoundError):
                prepare.validate_publication(output)


if __name__ == "__main__":
    unittest.main()

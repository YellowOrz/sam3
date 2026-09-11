import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import analyze_hand_segmentation_failures as analysis


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def save_mask(path, value):
    Image.fromarray(value.astype(np.uint8) * 255).save(path)


def masks():
    own = np.zeros((20, 20), dtype=bool)
    other = own.copy()
    own[5:15, 1:6] = True
    other[5:15, 14:19] = True
    return own, other


def make_fixture(root):
    data = root / "data"
    data.mkdir()
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(data / "rgb.png")
    own, other = masks()
    source_masks = {}
    annotations = []
    for side, mask, category in zip(analysis.SIDES, (own, other), (1, 2)):
        save_mask(data / f"{side}.png", mask)
        source_masks[side.removesuffix("_hand")] = {"binary_reference_png": f"{side}.png"}
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append({"id": category, "image_id": 4, "category_id": category, "segmentation": rle})
    image = {"id": 4, "file_name": "rgb.png", "height": 20, "width": 20,
             "recording_id": "ego/test", "view_type": "ego", "primary_test": True,
             "source_masks": source_masks}
    coco = {"images": [image], "annotations": annotations,
            "categories": [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]}
    save_json(data / "annotations.json", coco)
    visuals = root / "visuals"
    directory = visuals / "image-000004"
    model_dir = directory / "model"
    model_dir.mkdir(parents=True)
    Image.fromarray(rgb).save(directory / "rgb.png")
    # A deliberately high-score wrong-hand prediction must remain wrong in analysis.
    predictions = (other, other)
    records = []
    for side, own_reference, other_reference, prediction in zip(analysis.SIDES, (own, other), (other, own), predictions):
        save_mask(directory / f"{side}__reference.png", own_reference)
        save_mask(model_dir / f"{side}__candidate.png", prediction)
        save_mask(model_dir / f"{side}__detected.png", prediction)
        value = analysis.measure_mask(prediction, own_reference, other_reference)
        records.append({"model": "model", "image_id": 4, "dataset_index": 0, "prompt_key": side,
                        "observed_coco_image_id": 4, "identity_verified": True, "file_name": "rgb.png",
                        "recording_id": "ego/test", "view_type": "ego", "primary_test": True,
                        "top_confidence": .8, "top_class_probability": .8, "presence_probability": 1.,
                        "detected": True, "selected_decoder_query": 9,
                        "top_mask_pixels": value["prediction_pixels"],
                        "reference_pixels": value["own_reference_pixels"],
                        "other_reference_pixels": value["other_reference_pixels"],
                        "top_other_reference_intersection_pixels": value["other_intersection_pixels"],
                        "top_dice_with_own_reference": value["own_dice"],
                        "top_iou_with_own_reference": value["own_iou"]})
    (root / "records").mkdir()
    save_json(root / "records" / "model.json", records)
    manifest = [{"image_id": 4, "dataset_index": 0, "directory": str(directory), "diagnostic_ids": []}]
    save_json(visuals / "manifest.json", manifest)
    summary = {"format": "nakehand-frozen-bilateral-evaluation-v1", "status": "completed",
               "observed_identity_verified": True, "detection_threshold": .5, "mask_threshold": .5,
               "thresholds_fitted_on_nakehand": False, "data_root": str(data),
               "annotations_sha256": hashlib.sha256((data / "annotations.json").read_bytes()).hexdigest(),
               "candidate_selection": "argmax(score); never reference overlap", "visuals": manifest,
               "models": {"model": {}}, "evaluated_images": 1,
               "evaluated_image_ids": [4], "evaluated_dataset_indices": [0]}
    save_json(root / "summary.json", summary)
    return root / "summary.json", directory, data


class HandErrorMetricTest(unittest.TestCase):
    def test_exact_own_reference(self):
        own, other = masks()
        value = analysis.measure_mask(own, own, other)
        for key in ("own_dice", "own_iou", "own_precision", "own_recall", "own_boundary_iou"):
            self.assertEqual(value[key], 1.)
        self.assertEqual(value["own_missed_pixels"], 0)
        self.assertEqual(value["outside_both_references_pixels"], 0)
        self.assertFalse(value["opposite_overlap_dominant_proxy"])

    def test_wrong_hand_is_not_reselected(self):
        own, other = masks()
        value = analysis.measure_mask(other, own, other)
        self.assertEqual(value["own_dice"], 0.)
        self.assertEqual(value["own_missed_fraction"], 1.)
        self.assertEqual(value["other_only_prediction_fraction"], 1.)
        self.assertEqual(value["outside_both_prediction_fraction"], 0.)
        self.assertTrue(value["opposite_overlap_dominant_proxy"])

    def test_merged_hands_separate_wrong_side_area_from_background(self):
        own, other = masks()
        value = analysis.measure_mask(own | other, own, other)
        self.assertAlmostEqual(value["own_dice"], 2 / 3)
        self.assertEqual(value["own_recall"], 1.)
        self.assertEqual(value["own_precision"], .5)
        self.assertEqual(value["other_only_prediction_fraction"], .5)
        self.assertEqual(value["outside_both_references_pixels"], 0)

    def test_background_expansion_is_not_called_forearm(self):
        own, other = masks()
        prediction = own.copy()
        prediction[0:2, 0:5] = True
        value = analysis.measure_mask(prediction, own, other)
        self.assertEqual(value["outside_both_references_pixels"], 10)
        self.assertEqual(value["other_only_intersection_pixels"], 0)
        self.assertEqual(value["own_recall"], 1.)
        self.assertFalse(any("forearm" in key for key in value))

    def test_missing_protrusion_counts_omission_without_finger_claim(self):
        own, other = masks()
        own[2:5, 2:4] = True
        prediction = own.copy()
        prediction[2:5, 2:4] = False
        value = analysis.measure_mask(prediction, own, other)
        self.assertEqual(value["own_missed_pixels"], 6)
        self.assertAlmostEqual(value["own_missed_fraction"], 6 / 56)
        self.assertEqual(value["own_precision"], 1.)

    def test_all_empty_and_missed_reference_have_explicit_denominators(self):
        own, other = masks()
        empty = np.zeros_like(own)
        value = analysis.measure_mask(empty, empty, empty)
        for key in ("own_dice", "own_iou", "own_precision", "own_recall", "own_boundary_iou",
                    "outside_both_prediction_fraction", "other_only_prediction_fraction"):
            self.assertIsNone(value[key])
        value = analysis.measure_mask(empty, own, other)
        for key in ("own_dice", "own_iou", "own_recall", "own_boundary_iou"):
            self.assertEqual(value[key], 0.)
        self.assertIsNone(value["own_precision"])
        self.assertEqual(value["own_missed_fraction"], 1.)

    def test_overlap_priority_makes_prediction_partition_disjoint(self):
        own, _ = masks()
        other = np.roll(own, 2, axis=1)
        value = analysis.measure_mask(own | other, own, other)
        self.assertEqual(value["reference_overlap_pixels"], 30)
        self.assertEqual(value["other_intersection_pixels"], 50)
        self.assertEqual(value["other_only_intersection_pixels"], 20)
        self.assertEqual(value["prediction_pixels"], sum(value[key] for key in (
            "own_intersection_pixels", "other_only_intersection_pixels", "outside_both_references_pixels")))

    def test_inner_boundary_includes_image_edge_and_diagonal_rounding(self):
        full = np.ones((10, 10), dtype=bool)
        band = analysis.inner_boundary(full, .02)
        self.assertEqual(int(band.sum()), 36)
        self.assertTrue(band[0].all())
        self.assertFalse(band[5, 5])
        self.assertEqual(analysis.boundary_width((480, 640), .02), 16)
        self.assertEqual(analysis.boundary_width((1, 1), .02), 1)

    def test_binary_erosion_matches_independent_zero_padded_square_definition(self):
        generator = np.random.default_rng(93)
        for width in (1, 2, 3):
            binary = generator.random((20, 20)) > .2
            padded = np.pad(binary, width, constant_values=False)
            expected_interior = np.ones_like(binary)
            for y in range(2 * width + 1):
                for x in range(2 * width + 1):
                    expected_interior &= padded[y:y + 20, x:x + 20]
            ratio = width / np.hypot(20, 20)
            self.assertTrue(np.array_equal(analysis.inner_boundary(binary, ratio), binary & ~expected_interior))

    def test_bad_input_rejected(self):
        own, other = masks()
        for bad in (np.ones((20, 20)) * .2, np.ones((20, 20, 3)), np.ones((0, 20))):
            with self.assertRaises(ValueError):
                analysis.measure_mask(bad, own, other)
        with self.assertRaises(ValueError):
            analysis.measure_mask(own[:2], own, other)
        for bad_ratio in (0, -1, float("nan"), float("inf"), 2):
            with self.assertRaises(ValueError):
                analysis.measure_mask(own, own, other, bad_ratio)


class HandSavedAnalysisTest(unittest.TestCase):
    def test_end_to_end_preserves_high_score_wrong_hand_and_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            summary, _, _ = make_fixture(root)
            value = analysis.analyze_saved_visuals(summary)
            self.assertEqual(value["analyzed_images"], 1)
            self.assertEqual(len(value["records"]), 4)
            wrong = [row for row in value["records"] if row["prompt_key"] == "left_hand"]
            self.assertTrue(all(row["own_dice"] == 0 and row["selected_decoder_query"] == 9 for row in wrong))
            self.assertTrue(value["source_hashes_before_after_verified"])
            aggregate = value["aggregates"]["model"]["detected"]["overall"]
            self.assertEqual(aggregate["present_macro"]["own_dice"], .5)
            self.assertEqual(aggregate["all_query_pixel_micro"]["other_only_prediction_fraction"], .5)
            self.assertIn("不能直接称为", analysis.render_markdown(value))

    def test_mutated_sources_or_identity_are_rejected(self):
        for mutation in ("reference", "source_reference", "rgb", "candidate", "detected", "annotation", "record", "manifest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                summary, directory, data = make_fixture(root)
                empty = np.zeros((20, 20), dtype=bool)
                if mutation == "reference":
                    save_mask(directory / "left_hand__reference.png", empty)
                elif mutation == "source_reference":
                    save_mask(data / "left_hand.png", empty)
                elif mutation == "rgb":
                    Image.new("RGB", (20, 20), "white").save(directory / "rgb.png")
                elif mutation in ("candidate", "detected"):
                    save_mask(directory / "model" / f"left_hand__{mutation}.png", empty)
                elif mutation == "annotation":
                    (data / "annotations.json").write_bytes((data / "annotations.json").read_bytes() + b" ")
                elif mutation == "record":
                    path = root / "records" / "model.json"
                    rows = json.loads(path.read_text())
                    rows[0]["observed_coco_image_id"] = 123
                    save_json(path, rows)
                else:
                    save_json(root / "visuals" / "manifest.json", [])
                with self.assertRaises(ValueError):
                    analysis.analyze_saved_visuals(summary)

    def test_tracked_read_detects_concurrent_source_change(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "source"
            path.write_bytes(b"original")
            tracked = analysis.TrackedInputs()
            tracked.read(path)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                tracked.verify()

    def test_cli_writes_new_json_markdown_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            summary, _, _ = make_fixture(root)
            output = root / "new-analysis"
            args = ["--summary", str(summary), "--output-dir", str(output)]
            self.assertEqual(analysis.main(args), 0)
            self.assertEqual(json.loads((output / "analysis.json").read_text())["status"], "completed")
            self.assertTrue((output / "REPORT.md").is_file())
            with self.assertRaises(FileExistsError):
                analysis.main(args)


if __name__ == "__main__":
    unittest.main()

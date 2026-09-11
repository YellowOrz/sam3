import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

from scripts import evaluate_nakehand_tokens as evaluate


def masks():
    left = np.zeros((8, 10), dtype=bool)
    right = np.zeros_like(left)
    left[1:5, 1:4] = True
    right[2:6, 6:9] = True
    return left, right, np.zeros_like(left)


def record(image_id, side, own, other, prediction, score, *, primary=True, diagnostic=None,
           recording="ego/one", view="ego"):
    return {
        "model": "epoch2", "image_id": image_id, "prompt_key": side,
        "primary_test": primary, "diagnostic_ids": [] if diagnostic is None else [diagnostic],
        "recording_id": recording, "view_type": view,
        **evaluate.measure_query(prediction, own, other, score),
    }


def four_image_records(swapped=False):
    left, right, empty = masks()
    values = []
    for image_id, (own_left, own_right) in enumerate(((left, right), (left, empty),
                                                   (empty, right), (empty, empty))):
        for index, side in enumerate(evaluate.CLASS_NAMES):
            own, other = (own_left, own_right) if index == 0 else (own_right, own_left)
            prediction = other if swapped and image_id == 0 else own
            values.append(record(image_id, side, own, other, prediction, .8 if own.any() else .2))
    return values


def write_fixture(root):
    left, right, empty = masks()
    images, annotations = [], []
    for image_id, pair in enumerate(((left, right), (left, empty), (empty, right), (empty, empty))):
        image = {
            "id": image_id, "file_name": f"image-{image_id}.png", "height": 8, "width": 10,
            "primary_test": True, "recording_id": "ego/one", "view_type": "ego",
        }
        images.append(image)
        Image.new("RGB", (10, 8), (30 + image_id, 40, 50)).save(root / image["file_name"])
        for category, mask in enumerate(pair, 1):
            if not mask.any():
                continue
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            bbox = mask_utils.toBbox(rle).tolist()
            area = int(mask_utils.area(rle))
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({"id": len(annotations), "image_id": image_id, "category_id": category,
                                "bbox": bbox, "area": area, "segmentation": rle, "iscrowd": 0})
    data = {"images": images, "annotations": annotations,
            "categories": [{"id": category, "name": name} for category, name in evaluate.SIDE_BY_CATEGORY.items()],
            "info": {"split": "external_test"}}
    (root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")
    return data


class BilateralMeasurementsTest(unittest.TestCase):
    def test_both_single_and_empty_have_independent_query_denominators(self):
        result = evaluate.aggregate(four_image_records())
        self.assertEqual(result["images"], 4)
        self.assertEqual(result["queries"], 8)
        self.assertEqual(result["present_queries"], 4)
        self.assertEqual(result["absent_queries"], 4)
        self.assertEqual(result["single_hand_absent_queries"], 2)
        self.assertEqual(result["empty_image_queries"], 2)
        self.assertEqual(result["empty_images"], 1)
        self.assertEqual(result["correct_side_detection_rate"], 1)
        self.assertEqual(result["present_mean_candidate_dice"], 1)
        self.assertEqual(result["present_mean_miss_zero_dice"], 1)
        self.assertEqual(result["absent_side_false_positive_rate"], 0)
        self.assertEqual(result["empty_image_any_detection_rate"], 0)

    def test_swapping_both_masks_does_not_count_as_correct_segmentation(self):
        result = evaluate.aggregate(four_image_records(swapped=True))
        self.assertEqual(result["correct_side_detection_rate"], 1)  # presence only, explicitly not localization.
        self.assertEqual(result["present_mean_candidate_dice"], .5)
        self.assertEqual(result["both_visible_detected_opposite_dominant_queries"], 2)
        self.assertEqual(result["both_visible_detected_opposite_dominant_rate_per_detected_queries"], 1)
        self.assertEqual(result["simultaneous_two_query_swap_proxy_images"], 1)
        self.assertEqual(result["simultaneous_two_query_swap_proxy_image_rate"], 1)

    def test_missing_detection_is_zero_not_removed_from_segmentation_average(self):
        left, right, _ = masks()
        metrics = evaluate.measure_query(left, left, right, .49)
        self.assertEqual(metrics["top_dice"], 1)
        self.assertEqual(metrics["miss_zero_dice"], 0)
        self.assertFalse(metrics["detected"])
        metrics = evaluate.measure_query(left, left, right, .5)
        self.assertEqual(metrics["miss_zero_dice"], 1)
        self.assertTrue(metrics["detected"])

    def test_absent_references_are_null_not_perfect_empty_dice(self):
        _, _, empty = masks()
        metrics = evaluate.measure_query(empty, empty, empty, .7)
        self.assertIsNone(metrics["top_dice"])
        self.assertIsNone(metrics["miss_zero_dice"])
        self.assertTrue(metrics["detected"])  # score-defined FP even if selected mask is empty.
        result = evaluate.aggregate([record(1, "left_hand", empty, empty, empty, .7)])
        self.assertIsNone(result["correct_side_detection_rate"])
        self.assertIsNone(result["present_mean_candidate_dice"])
        self.assertEqual(result["absent_side_false_positive_rate"], 1)
        self.assertIsNone(result["empty_image_any_detection_rate"])  # needs both prompt results.

    def test_wrong_side_proxy_ties_false_and_score_suppression_changes_only_detected_rate(self):
        left, right, empty = masks()
        tie = evaluate.measure_query(empty, left, right, .9)
        self.assertFalse(tie["opposite_overlap_dominant_proxy"])
        low = evaluate.measure_query(right, left, right, .1)
        self.assertTrue(low["opposite_overlap_dominant_proxy"])
        self.assertFalse(low["detected_opposite_overlap_dominant_proxy"])
        self.assertEqual(low["top_iou_with_other_reference"], 1)

    def test_invalid_shapes_and_nonfinite_scores_rejected(self):
        left, right, empty = masks()
        for confidence in (float("nan"), float("inf"), -1, 1.1):
            with self.assertRaises(ValueError):
                evaluate.measure_query(left, left, right, confidence)
        with self.assertRaises(ValueError):
            evaluate.measure_query(left[:2], left, empty, .5)

    def test_diagnostics_do_not_change_primary_and_group_counts(self):
        rows = four_image_records()
        left, right, _ = masks()
        for side, own, other in (("left_hand", left, right), ("right_hand", right, left)):
            rows.append(record(9, side, own, other, other, .8, primary=False, diagnostic="A",
                               recording="exo/two", view="exo"))
        result = evaluate.summarize(rows)["epoch2"]
        self.assertEqual(result["primary_test"]["overall"]["images"], 4)
        self.assertEqual(result["primary_test"]["overall"]["present_mean_candidate_dice"], 1)
        self.assertEqual(result["diagnostic_examples"]["A"]["overall"]["images"], 1)
        self.assertEqual(result["diagnostic_examples"]["A"]["overall"]["present_mean_candidate_dice"], 0)
        self.assertEqual(set(result["primary_test"]["per_recording"]), {"ego/one"})
        self.assertEqual(set(result["diagnostic_only_no_population_claim"]["per_view_type"]), {"exo"})

    def test_missing_or_duplicate_side_records_rejected(self):
        rows = four_image_records()
        with self.assertRaisesRegex(ValueError, "Missing side"):
            evaluate.summarize(rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate query"):
            evaluate.summarize(rows + rows[:1])


class DatasetAndIdentityTest(unittest.TestCase):
    def test_side_category_two_and_disconnected_components_decode_as_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = write_fixture(root)
            duplicate = dict(data["annotations"][1], id=100)
            data["annotations"].append(duplicate)
            (root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")
            images, references, _ = evaluate.load_coco_index(root)
            self.assertEqual(len(images), 4)
            left, right, _ = masks()
            np.testing.assert_array_equal(references[0]["left_hand"], left)
            np.testing.assert_array_equal(references[0]["right_hand"], right)
            self.assertFalse(references[3]["right_hand"].any())

    def test_real_cpu_loader_and_collator_include_two_positive_queries_and_negatives(self):
        from sam3.train.data.collator import collate_fn_api

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            images, _, _ = evaluate.load_coco_index(root)
            dataset = evaluate.shared.make_dataset(root)
            samples = [dataset[index] for index in range(4)]
            self.assertEqual([len(sample.find_queries) for sample in samples], [2, 2, 2, 2])
            self.assertEqual([[len(query.object_ids_output) for query in sample.find_queries]
                              for sample in samples], [[1, 1], [1, 0], [0, 1], [0, 0]])
            batch = collate_fn_api(samples, dict_key="eval", with_seg_masks=True)["eval"]
            evaluate.validate_batch_identity(batch, [0, 1, 2, 3], images)
            self.assertEqual(batch.find_targets[0].num_boxes.tolist(), [1, 1, 1, 0, 0, 1, 0, 0])
            batch.find_metadatas[0].coco_image_id[0] = 100
            with self.assertRaisesRegex(RuntimeError, "substituted image"):
                evaluate.validate_batch_identity(batch, [0, 1, 2, 3], images)

    def test_interactive_and_geometry_prompt_inputs_forbidden(self):
        model = torch.nn.Linear(1, 1)
        model.num_interactive_steps_val = 0
        with self.assertRaises(RuntimeError):
            evaluate.validate_frozen_noninteractive_model(model)
        model.eval()
        evaluate.validate_frozen_noninteractive_model(model)
        model.num_interactive_steps_val = 1
        with self.assertRaises(RuntimeError):
            evaluate.validate_frozen_noninteractive_model(model)
        stage = SimpleNamespace(img_ids=torch.tensor([0, 0]), text_ids=torch.tensor([0, 1]),
                                input_points=torch.tensor([[1., 2., 1.]]))
        batch = SimpleNamespace(find_inputs=[stage], find_text_batch=list(evaluate.CLASS_NAMES),
                                find_metadatas=[SimpleNamespace(coco_image_id=torch.tensor([7, 7]),
                                                              original_category_id=torch.tensor([1, 2]))])
        with self.assertRaisesRegex(RuntimeError, "forbids geometry prompts"):
            evaluate.validate_batch_identity(batch, [0], [{"id": 7}])
        stage.input_points = None
        stage.text_ids = torch.tensor([0, 0])
        batch.find_metadatas[0].original_category_id = torch.tensor([1, 1])
        with self.assertRaisesRegex(RuntimeError, "exactly one left"):
            evaluate.validate_batch_identity(batch, [0], [{"id": 7}])

    def test_checkpoint_epoch_completion_is_derived_from_actual_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pt"
            state = {"class_tokens": torch.zeros(2, 4, 256), "annotation_summary": {"images": 5},
                     "training_config": {"batch_size": 2, "epochs": 2}, "epochs": 2,
                     "base_checkpoint": str(base), "next_step": 3}
            path = root / "misnamed_epoch2_final.pt"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "1 complete epochs"):
                evaluate.learned_checkpoint_metadata(path, base, 2)
            state["next_step"] = 6
            torch.save(state, path)
            self.assertEqual(evaluate.learned_checkpoint_metadata(path, base, 2)["completed_epochs_from_steps"], 2)
            with self.assertRaisesRegex(ValueError, "base path differs"):
                evaluate.learned_checkpoint_metadata(path, root / "other.pt", 2)

    def test_cli_fixed_thresholds_and_unique_models(self):
        args = ["--data-root", "/data", "--base-checkpoint", "/base", "--output-dir", "/new", "--include-ve"]
        parsed = evaluate.parse_args(args)
        self.assertEqual(parsed.batch_size, 1)
        self.assertEqual(parsed.minimum_completed_epochs, 2)
        for invalid in (["--detection-threshold", ".7"], ["--learned-checkpoint", "ve-natural=/checkpoint"]):
            with self.assertRaises(SystemExit):
                evaluate.parse_args(args + invalid)

    def test_separate_visuals_preserve_rgb_and_binary_reference_prediction(self):
        left, right, _ = masks()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = write_fixture(root)
            images, references, _ = evaluate.load_coco_index(root)
            rows = four_image_records()[:2]
            for row in rows:
                row["dataset_index"] = 0
                row["prompt_text"] = row["prompt_key"]
            output = root / "visuals"
            manifest = evaluate.render_results(data_root=root, output_dir=output, images=images,
                                               references=references, render_indices=[0], records=rows,
                                               masks={("epoch2", 0, "left_hand"): left,
                                                      ("epoch2", 0, "right_hand"): right}, labels=["epoch2"])
            target = Path(manifest[0]["directory"])
            np.testing.assert_array_equal(np.array(Image.open(target / "rgb.png")),
                                          np.array(Image.open(root / data["images"][0]["file_name"])))
            for path in (target / "left_hand__reference.png", target / "epoch2/right_hand__detected.png"):
                self.assertEqual(set(np.unique(np.array(Image.open(path)))), {0, 255})
            self.assertTrue((target / "comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()

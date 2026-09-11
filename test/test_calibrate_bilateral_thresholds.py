import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.calibrate_bilateral_thresholds import analyze_summaries, calibrate_records, main


def records(scores=((0.9, 0.7), (0.8, 0.6)), *, label="epoch1", empty_scores=None):
    rows = []
    cases = [("left_hand", pair) for pair in scores]
    if empty_scores is not None:
        cases.append(("empty", empty_scores))
    for index, (side, pair) in enumerate(cases):
        for prompt, score in zip(("left_hand", "right_hand"), pair):
            rows.append({
                "model": label, "dataset_index": index, "image_id": index,
                "file_name": f"image{index}.jpg", "source": "dexycb",
                "sequence": "sequence-01", "view": "camera-01", "frame_index": index,
                "actual_side": side, "prompt_key": prompt,
                "target_present": side == prompt, "physical_hand_present": side != "empty",
                "top_confidence": score, "top_dice_with_physical_hand": 0.4 + index * 0.2,
            })
    return rows


class CalibrateBilateralThresholdsTest(unittest.TestCase):
    def test_equal_scores_are_indivisible_under_strict_budget(self):
        rows = records(((0.9, 0.9), (0.8, 0.1)))
        points = calibrate_records(rows, budgets=(0, 0.5))["operating_points"]
        self.assertEqual(points[0]["correct_true_positives"], 0)
        self.assertGreater(points[0]["selected_threshold"], 0.9)
        self.assertEqual(points[1]["selected_threshold"], 0.8)
        self.assertEqual(points[1]["correct_true_positives"], 2)
        self.assertEqual(points[1]["opposite_false_positives"], 1)

    def test_maximizes_tpr_then_keeps_conservative_flat_region(self):
        result = calibrate_records(records(), budgets=(0,))
        point = result["operating_points"][0]
        self.assertEqual(point["selected_threshold"], 0.8)
        self.assertEqual(point["correct_prompt_tpr"], 1)
        self.assertEqual(point["actual_opposite_fpr"], 0)
        self.assertAlmostEqual(point["correct_prompt_mean_thresholded_dice"], 0.5)
        self.assertAlmostEqual(result["correct_prompt_mean_top_dice"], 0.5)

    def test_misses_count_as_zero_and_empty_fpr_has_separate_denominator(self):
        result = calibrate_records(
            records(((0.9, 0.8), (0.7, 0.1)), empty_scores=(0.95, 0.2)), budgets=(0,)
        )
        point = result["operating_points"][0]
        self.assertEqual(point["correct_prompt_tpr"], 0.5)
        self.assertAlmostEqual(point["correct_prompt_mean_thresholded_dice"], 0.2)
        self.assertEqual(point["empty_prompt_fpr"], 0.5)
        self.assertEqual(result["counts"]["empty_prompts"], 2)

    def test_score_one_ties_have_an_explicit_reject_all_cutoff(self):
        point = calibrate_records(records(((1, 1),)), budgets=(0,))["operating_points"][0]
        self.assertTrue(point["threshold_above_one_requires_reject_all"])
        self.assertEqual(point["correct_true_positives"], 0)

    def test_duplicate_and_missing_prompt_rows_are_rejected(self):
        original = records()
        for rows in (original + [original[0]], original[:-1]):
            with self.assertRaises(ValueError):
                calibrate_records(rows)

    def test_bad_scores_and_presence_labels_are_rejected(self):
        for field, value in (("top_confidence", float("nan")), ("top_confidence", 1.1),
                             ("target_present", False), ("physical_hand_present", False)):
            with self.subTest(field=field, value=value):
                rows = records()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    calibrate_records(rows)

    def fixture(self, root, *, split="val", data_dir="val", label="epoch1"):
        data = root / data_dir
        data.mkdir(exist_ok=True)
        annotation_path = data / "annotations.json"
        annotation_path.write_text(json.dumps({"info": {"split": split}}))
        run = root / label
        (run / "records").mkdir(parents=True)
        rows = records(label=label)
        (run / "records" / f"{label}.json").write_text(json.dumps(rows))
        summary = {
            "data_root": str(data), "annotations_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            "base_checkpoint": "/base.pt", "evaluated_images": 2,
            "evaluated_dataset_indices": [0, 1], "mask_threshold": 0.5,
            "confidence_definition": "class * presence", "models": {label: {}}, "metrics": {label: {}},
        }
        path = run / "summary.json"
        path.write_text(json.dumps(summary))
        return path

    def test_val_metadata_and_file_hashes_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            result = analyze_summaries([path])
            self.assertEqual(result["split"], "val")
            self.assertEqual(result["fpr_budgets"], [0.01, 0.05, 0.1])
            self.assertEqual(len(result["source_files"]), 3)
            self.assertTrue(result["sources_rechecked_unchanged"])
            self.assertIn("not independent test guarantees", result["limitations"][0])

    def test_test_split_is_rejected_even_if_directory_is_named_val(self):
        for split, directory_name in (("test", "val"), ("val", "test")):
            with self.subTest(split=split, directory_name=directory_name):
                with tempfile.TemporaryDirectory() as directory:
                    path = self.fixture(Path(directory), split=split, data_dir=directory_name)
                    with self.assertRaisesRegex(ValueError, "val"):
                        analyze_summaries([path])

    def test_multi_summary_comparison_rejects_misaligned_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.fixture(root, label="epoch1")
            second = self.fixture(root, label="epoch2")
            result = analyze_summaries([first, second])
            self.assertEqual(set(result["models"]), {"epoch1", "epoch2"})
            rows_path = second.parent / "records/epoch2.json"
            rows = json.loads(rows_path.read_text())
            for row in rows:
                row["sequence"] = "another-sequence"
            rows_path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "different images"):
                analyze_summaries([first, second])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.fixture(root)
            output = root / "result.json"
            output.write_text("keep existing result")
            with patch("sys.argv", ["calibrate", "--summary", str(summary), "--output", str(output)]):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(output.read_text(), "keep existing result")


if __name__ == "__main__":
    unittest.main()

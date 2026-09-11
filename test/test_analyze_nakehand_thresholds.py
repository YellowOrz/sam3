from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_nakehand_thresholds as diagnostics
from scripts.evaluate_nakehand_tokens import measure_query


def paired_frame(index, present=(True, True), scores=(.9, .8), model="synthetic"):
    references = [np.zeros((4, 4), bool) for _ in range(2)]
    for side, visible in enumerate(present):
        if visible:
            references[side][side:side + 1, :] = True
    rows = []
    for side, score in enumerate(scores):
        prediction = references[side].copy()
        if not prediction.any():
            prediction[2, 0] = True
        rows.append({
            "image_id": index + 100, "dataset_index": index, "model": model,
            "prompt_key": diagnostics.SIDES[side], "file_name": f"{index}.png",
            "recording_id": "ego/test", "view_type": "ego", "source_frame_index": index,
            "video_pts_seconds": index / 30, "source_mapping": {"video": "fixture"},
            "selected_decoder_query": 10 + side,
            **measure_query(prediction, references[side], references[1 - side], score),
        })
    return rows


class NakehandThresholdTests(unittest.TestCase):
    def sample(self):
        return (paired_frame(0, (True, True), (.9, .8))
                + paired_frame(1, (True, False), (.7, .6))
                + paired_frame(2, (False, False), (.2, .1)))

    def test_two_hands_are_two_positive_queries(self):
        rows = self.sample()
        diagnostics.validate_bilateral_rows(rows)
        result = diagnostics.analyze_rows(rows)["operating_points"][0]["overall"]
        self.assertEqual(result["present_queries"], 3)
        self.assertEqual(result["absent_queries"], 3)
        self.assertEqual(result["single_hand_absent_queries"], 1)
        self.assertEqual(result["empty_image_queries"], 2)
        self.assertEqual(result["empty_images"], 1)

    def test_duplicate_or_missing_prompt_rejected(self):
        for rows in (self.sample() + [self.sample()[0]], self.sample()[:-1]):
            with self.assertRaises(ValueError):
                diagnostics.validate_bilateral_rows(rows)

    def test_pair_presence_mismatch_rejected(self):
        rows = self.sample()
        rows[1]["other_reference_pixels"] = 0
        with self.assertRaises(ValueError):
            diagnostics.validate_bilateral_rows(rows)

    def test_provenance_mismatch_rejected(self):
        rows = self.sample()
        rows[1]["source_mapping"] = {"video": "different"}
        with self.assertRaises(ValueError):
            diagnostics.validate_bilateral_rows(rows)

    def test_metric_corruption_rejected(self):
        for field, value in (("miss_zero_dice", .3), ("top_dice", .3),
                             ("top_iou_with_own_reference", .8), ("detected_mask_pixels", 0)):
            rows = self.sample()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                diagnostics.validate_bilateral_rows(rows)

    def test_nonfinite_score_and_nonboolean_presence_rejected(self):
        for field, value in (("top_confidence", float("nan")), ("top_confidence", float("inf")),
                             ("top_confidence", True), ("target_present", 1)):
            rows = self.sample()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                diagnostics.validate_bilateral_rows(rows)

    def test_candidate_decoder_range_rejected(self):
        rows = self.sample()
        rows[0]["selected_decoder_query"] = 200
        with self.assertRaises(ValueError):
            diagnostics.validate_bilateral_rows(rows)

    def test_exact_decimal_budget_and_ties_are_indivisible(self):
        rows = [{"top_confidence": .9, "target_present": True}]
        rows += [{"top_confidence": .8, "target_present": False}] * 5
        rows += [{"top_confidence": .1, "target_present": False}] * 491
        one = diagnostics.select_threshold(rows, "0.01")
        five = diagnostics.select_threshold(rows, "0.05")
        self.assertEqual(one["allowed_false_positive_queries"], 4)
        self.assertEqual(five["allowed_false_positive_queries"], 24)
        self.assertEqual(one["fit_false_positive_queries"], 0)
        self.assertEqual(one["threshold"], .9)

    def test_tied_positive_and_negative_cannot_split(self):
        rows = [{"top_confidence": .8, "target_present": True},
                {"top_confidence": .8, "target_present": False}]
        point = diagnostics.select_threshold(rows, "0")
        self.assertGreater(point["threshold"], .8)
        self.assertEqual(point["fit_true_positive_queries"], 0)

    def test_reject_all_handles_saturated_score_one(self):
        rows = [{"top_confidence": 1., "target_present": True},
                {"top_confidence": 1., "target_present": False}]
        point = diagnostics.select_threshold(rows, "0")
        self.assertTrue(point["reject_all_above_one"])
        self.assertEqual(point["fit_false_positive_queries"], 0)

    def test_flat_detection_rate_prefers_fewer_false_outputs(self):
        rows = [{"top_confidence": .9, "target_present": True},
                {"top_confidence": .8, "target_present": False}]
        point = diagnostics.select_threshold(rows, "1")
        self.assertEqual(point["threshold"], .9)

    def test_threshold_selection_does_not_consult_dice(self):
        rows = self.sample()
        before = diagnostics.select_threshold(rows, ".05")
        for row in rows:
            row["top_dice"] = 12345
        self.assertEqual(before, diagnostics.select_threshold(rows, ".05"))

    def test_threshold_gates_mask_metric_without_reranking_or_mutation(self):
        rows = self.sample()
        original = deepcopy(rows)
        revised = diagnostics.rethreshold(rows, .85)
        self.assertEqual(rows, original)
        self.assertEqual(revised[1]["top_dice"], 1.)
        self.assertEqual(revised[1]["miss_zero_dice"], 0.)
        self.assertEqual([row["selected_decoder_query"] for row in revised],
                         [row["selected_decoder_query"] for row in rows])

    def test_no_positive_or_negative_denominator_cannot_fit(self):
        for target in (True, False):
            with self.assertRaises(ValueError):
                diagnostics.select_threshold([{"top_confidence": .3, "target_present": target}], "0")

    def test_runs_deduplicate_queries_but_not_recordings(self):
        rows = paired_frame(0, (False, False)) + paired_frame(1, (False, False))
        rows += paired_frame(4, (False, False))
        result = diagnostics.contiguous_ranges(rows)
        self.assertEqual(result["unique_frames"], 3)
        self.assertEqual(result["contiguous_runs"], 2)
        self.assertEqual(result["largest_run_fraction"], 2 / 3)

    def test_empty_frame_false_output_is_image_or_of_two_queries(self):
        rows = paired_frame(0, (True, True)) + paired_frame(1, (False, False), (.9, .8))
        overall = diagnostics.analyze_rows(rows)["operating_points"][0]["overall"]
        self.assertEqual(overall["empty_image_false_positive_queries"], 2)
        self.assertEqual(overall["empty_images_with_any_detection"], 1)

    def test_partial_summary_rejected_before_threshold_analysis(self):
        with self.assertRaises(ValueError):
            diagnostics.checked_report.validate_summary({"format": diagnostics.checked_report.FORMAT,
                                                        "status": "completed", "full_val_evaluated": False}, {})

    def test_existing_output_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            existing.touch()
            with patch.object(diagnostics, "analyze_summary") as analyze:
                with self.assertRaises(FileExistsError):
                    diagnostics.main(["--summary", "missing.json", "--output-json", str(existing),
                                      "--output-md", str(Path(directory) / "new.md")])
                analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import copy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from scripts import evaluate_hand_boundary_diagnostics as boundary
from test_evaluate_nakehand_semantic_tokens import state


def masks(shape=(40, 60)):
    own = np.zeros(shape, dtype=bool)
    other = own.copy()
    own[5:25, 5:20] = True
    other[5:25, 35:50] = True
    return own, other


def fake_output():
    own, other = masks()
    logits = torch.logit(torch.tensor([[[.9], [.8]], [[.9], [.8]]]))
    prediction_masks = torch.full((2, 2, *own.shape), -8.)
    # High-scoring wrong mask; a lower-scoring perfect alternative must be ignored.
    prediction_masks[0, 0][torch.from_numpy(other)] = 8.
    prediction_masks[0, 1][torch.from_numpy(own)] = 8.
    prediction_masks[1, 0][torch.from_numpy(other)] = 8.
    return {"pred_logits": logits, "presence_logit_dec": torch.full((2, 1), 8.), "pred_masks": prediction_masks}


class MultiscaleBoundaryTest(unittest.TestCase):
    def test_distance_bands_equal_existing_erosion_for_shapes_edges_and_holes(self):
        generator = np.random.default_rng(113)
        patterns = [np.ones((40, 60), dtype=bool), np.zeros((40, 60), dtype=bool),
                    generator.random((40, 60)) > .2, masks()[0]]
        holed = patterns[0].copy()
        holed[10:20, 20:30] = False
        patterns.append(holed)
        for binary in patterns:
            bands, widths = boundary.boundary_bands(binary)
            for key, ratio in zip(boundary.RATIO_KEYS, boundary.RATIOS):
                expected = boundary.errors.inner_boundary(binary, ratio)
                np.testing.assert_array_equal(bands[key], expected)
                self.assertEqual(widths[key], boundary.errors.boundary_width(binary.shape, ratio))

    def test_640x480_has_exact_4_8_16_band_width(self):
        binary = np.ones((480, 640), dtype=bool)
        bands, widths = boundary.boundary_bands(binary)
        self.assertEqual(list(widths.values()), [4, 8, 16])
        self.assertTrue(bands["r0.005"][0].all())
        self.assertFalse(bands["r0.005"][10, 10])
        self.assertTrue(bands["r0.020"][10, 10])

    def test_identical_mask_has_one_in_all_bands_and_empty_conventions(self):
        own, other = masks()
        value = boundary.multiscale_metrics(own, own, other)
        self.assertTrue(all(item["iou"] == 1. for item in value["boundary"].values()))
        empty = np.zeros_like(own)
        value = boundary.multiscale_metrics(empty, own, other)
        self.assertTrue(all(item["iou"] == 0. for item in value["boundary"].values()))
        self.assertEqual(value["own_missed_fraction"], 1.)
        self.assertIsNone(value["own_precision"])
        value = boundary.multiscale_metrics(empty, empty, empty)
        self.assertTrue(all(item["iou"] is None for item in value["boundary"].values()))
        self.assertIsNone(value["own_dice"])

    def test_area_decomposition_matches_previous_verified_boolean_metrics(self):
        own, other = masks()
        candidates = [own, other, own | other, np.ones_like(own), np.zeros_like(own)]
        for prediction in candidates:
            actual = boundary.area_metrics(prediction, own, other)
            expected = boundary.errors.measure_mask(prediction, own, other)
            for key, value in actual.items():
                self.assertEqual(value, expected[key], key)

    def test_detected_reuses_candidate_and_empty_avoids_second_distance_transform(self):
        own, other = masks()
        own_bands = boundary.boundary_bands(own)[0]
        for detected in (True, False):
            with patch.object(boundary, "boundary_bands", wraps=boundary.boundary_bands) as wrapped:
                value = boundary.candidate_and_detected_metrics(own, own, other,
                    detected=detected, own_bands=own_bands)
                self.assertEqual(wrapped.call_count, 1)
            if detected:
                self.assertIs(value["candidate"], value["detected"])
            else:
                self.assertEqual(value["detected"]["own_dice"], 0.)
                self.assertEqual(value["detected"]["prediction_pixels"], 0)

    def test_wrong_side_and_background_expansion_have_different_pixel_terms(self):
        own, other = masks()
        wrong = boundary.area_metrics(other, own, other)
        expanded = own.copy()
        expanded[0:2, 0:3] = True
        background = boundary.area_metrics(expanded, own, other)
        self.assertEqual(wrong["other_only_prediction_fraction"], 1.)
        self.assertEqual(wrong["outside_both_references_pixels"], 0)
        self.assertEqual(background["other_only_intersection_pixels"], 0)
        self.assertEqual(background["outside_both_references_pixels"], 6)


class BoundarySelectionAndGroupingTest(unittest.TestCase):
    def test_selection_keeps_high_score_wrong_hand_not_gt_oracle(self):
        selected = boundary.select_predictions(fake_output(), (40, 60))
        own, other = masks()
        self.assertEqual(selected[0]["selected_decoder_query"], 0)
        np.testing.assert_array_equal(selected[0]["prediction"], other)
        self.assertGreater(selected[0]["top_confidence"], .5)
        self.assertEqual(boundary.area_metrics(selected[0]["prediction"], own, other)["own_dice"], 0.)

    def test_resize_and_confidence_match_existing_evaluator_rule(self):
        output = fake_output()
        generator = torch.Generator().manual_seed(51)
        output["pred_masks"] = torch.randn(2, 2, 8, 9, generator=generator)
        result = boundary.select_predictions(output, (40, 60))
        expected = F.interpolate(output["pred_masks"][0, 0][None, None].float(), size=(40, 60),
                                 mode="bilinear", align_corners=False)[0, 0].sigmoid().numpy() >= .5
        np.testing.assert_array_equal(result[0]["prediction"], expected)
        self.assertAlmostEqual(result[0]["top_confidence"], result[0]["top_class_probability"] * result[0]["presence_probability"])

    def test_grouping_reports_empty_masks_without_inflating_boundary_average(self):
        own, other = masks()
        rows = []
        for side, prediction, detected in (("left_hand", own, True), ("right_hand", other, False)):
            reference = own if side == "left_hand" else other
            opposite = other if side == "left_hand" else own
            rows.append({"prompt_key": side, "view_type": "ego", "recording_id": "ego/one", "target_present": True,
                         "spatial": boundary.candidate_and_detected_metrics(prediction, reference, opposite, detected=detected)})
        metrics = boundary.grouped_spatial_summary(rows)
        self.assertEqual(metrics["candidate"]["overall"]["present_macro"]["own_dice"], 1.)
        self.assertEqual(metrics["detected"]["overall"]["present_macro"]["own_dice"], .5)
        self.assertEqual(metrics["detected"]["overall"]["empty_predictions"], 1)
        self.assertEqual(metrics["detected"]["overall"]["boundary_present_macro"]["r0.005"], .5)
        self.assertEqual(metrics["detected"]["prompt_key:right_hand"]["present_macro"]["own_missed_fraction"], 1.)


class BoundaryCheckpointAndCLITest(unittest.TestCase):
    def test_old_pilot_partial_restore_point_is_allowed_but_not_relabelled_complete(self):
        checkpoint = state()
        current, initial = boundary.validate_checkpoint_dispatch(checkpoint, minimum_samples=1,
            base_hash="a" * 64, tokenizer_hash="b" * 64)
        self.assertGreater(float(current.delta.detach().abs().sum()), 0)
        self.assertFalse(checkpoint["progress"]["pilot_complete"])
        self.assertEqual(checkpoint["progress"]["samples_seen"], 20)
        with self.assertRaises(ValueError):
            boundary.validate_checkpoint_dispatch(checkpoint, minimum_samples=2000,
                base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_unknown_trial_dexycb_or_random_format_fails_closed(self):
        for format_name in ("sam3-nakehand-prompt-ablation-training-v99", boundary.semantic.FORMAT, "sam3-learnable-class-tokens-v2"):
            checkpoint = state()
            checkpoint["format"] = format_name
            with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
                boundary.validate_checkpoint_dispatch(checkpoint, minimum_samples=1,
                    base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_boundary_trial_uses_own_schema_without_format_relabelling(self):
        from test_train_nakehand_prompt_ablation import checkpoint_fixture
        checkpoint = checkpoint_fixture(weight=4., steps=1)[0]
        module = boundary.boundary_trial_module()
        with patch.object(module, "validate_checkpoint_schema", wraps=module.validate_checkpoint_schema) as validator:
            current, initial = boundary.validate_checkpoint_dispatch(checkpoint, minimum_samples=1,
                base_hash="a" * 64, tokenizer_hash="b" * 64)
            self.assertEqual(validator.call_count, 1)
            self.assertEqual(validator.call_args.args[0]["format"], boundary.BOUNDARY_TRIAL_FORMAT)
        self.assertEqual(checkpoint["training_config"]["boundary_weight"], 4.)
        self.assertGreater(float(current.delta.detach().abs().sum()), 0.)
        self.assertEqual(float(initial.delta.detach().abs().sum()), 0.)
        checkpoint["training_config"]["boundary_weight"] = 8.
        with self.assertRaises(ValueError):
            boundary.validate_checkpoint_dispatch(checkpoint, minimum_samples=1,
                base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_different_steps_are_explicit_not_a_matched_budget_claim(self):
        first, later = state(), state(1.)
        later["progress"]["samples_seen"] = 100
        later["observed_image_ids"] = list(range(100))
        compared = boundary.compare_checkpoint_inputs({"CP0": first, "CP1": later})
        self.assertFalse(compared["matched_actual_training_budget"])
        self.assertIn("actual_training_progress", compared["differences_excluding_anchor_weight_and_file_paths"]["CP1"])
        changed = copy.deepcopy(later)
        changed["initial_cache_state_dict"]["raw_cache"].add_(1.)
        with self.assertRaises(ValueError):
            boundary.compare_checkpoint_inputs({"CP0": first, "CP1": changed})

    def test_parse_render_limit_frozen_output_and_explicit_aliases(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = ["--data-root", str(root / "data"), "--base-checkpoint", str(root / "base"),
                    "--output-dir", str(root / "output"), "--cp0-checkpoint", str(root / "cp0"),
                    "--variant", "baseline"]
            parsed = boundary.parse_args(args)
            self.assertEqual(parsed.labels, ["baseline"])
            self.assertEqual(parsed.checkpoints["baseline"], root / "cp0")
            self.assertEqual(parsed.render_count, 12)
            self.assertIsNone(parsed.indices)
            with self.assertRaises(SystemExit):
                boundary.parse_args(args + ["--render-count", "13"])


if __name__ == "__main__":
    unittest.main()

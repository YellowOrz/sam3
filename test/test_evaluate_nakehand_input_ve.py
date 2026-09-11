"""CPU-only input VE evaluation contracts, score selection and baseline routing."""
from copy import deepcopy
import contextlib
import inspect
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts import evaluate_nakehand_input_ve as evaluation
from test_train_nakehand_input_ve import checkpoint_fixture


class SpyOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.last = None

    def forward(self, text, input_boxes=None, device=None):
        self.last = (text, input_boxes, device)
        return self.last


class InputVEEvaluationTests(unittest.TestCase):
    def test_baseline_only_aliases_natural_text_and_freezes_original(self):
        original = SpyOriginal()
        wrapper = evaluation.NaturalVEAliases(original)
        result = wrapper(["left_hand", "cup", "right_hand", "knife"], device="cpu")
        self.assertEqual(result[0], ["left hand", "cup", "right hand", "knife"])
        self.assertIs(wrapper.original_ve, original)
        self.assertFalse(original.weight.requires_grad)
        self.assertFalse(any(module.training for module in wrapper.modules()))
        encoded = (torch.ones(1), torch.ones(1), {"inputs_embeds": torch.ones(1)})
        self.assertIs(wrapper(encoded)[0], encoded)

    def test_selector_cannot_accept_gt_or_pick_reference_best_candidate(self):
        self.assertEqual(list(inspect.signature(evaluation.spatial.select_predictions).parameters), ["output", "shape"])
        masks = torch.full((2, 2, 5, 5), -10.)
        # Lower-scored candidate exactly matches the reference, but must not win.
        masks[:, 1, 1:4, 1:4] = 10.
        output = {"pred_logits": torch.tensor([[[6.], [1.]], [[6.], [1.]]]),
                  "presence_logit_dec": torch.tensor([[6.], [6.]]), "pred_masks": masks}
        result = evaluation.spatial.select_predictions(output, (5, 5))
        self.assertEqual([row["selected_decoder_query"] for row in result], [0, 0])
        self.assertFalse(any(row["prediction"].any() for row in result))

    def test_absent_reference_output_remains_measured_without_truth_claim(self):
        prediction = np.ones((5, 5), dtype=bool)
        empty = np.zeros_like(prediction)
        record = evaluation.bilateral.measure_query(prediction, empty, empty, .9)
        self.assertFalse(record["target_present"])
        self.assertTrue(record["detected"])
        self.assertIn("not verified true false detection", evaluation.REFERENCE_ABSENCE_WARNING)
        value = evaluation.spatial.candidate_and_detected_metrics(prediction, empty, empty, detected=True)
        self.assertEqual(value["detected"]["outside_both_references_pixels"], 25)

    def test_comparison_marks_different_lr_and_budget_without_faking_single_factor(self):
        state, *_ = checkpoint_fixture(steps=20)
        output = deepcopy(state)
        result = evaluation.compare_training_inputs(state, output)
        self.assertTrue(result["matched_actual_training_budget"])
        self.assertTrue(result["matched_training_numerics"])
        output["training_config"]["learning_rate"] = .0003
        result = evaluation.compare_training_inputs(state, output)
        self.assertFalse(result["matched_training_numerics"])
        self.assertEqual(result["numeric_configuration_differences"], ["learning_rate"])
        output["progress"]["samples_seen"] = 21
        self.assertFalse(evaluation.compare_training_inputs(state, output)["matched_actual_training_budget"])
        self.assertFalse(evaluation.compare_training_inputs(state)["training_comparison_available"])

    def test_comparison_rejects_source_cache_and_train_order_differences(self):
        state, *_ = checkpoint_fixture(steps=1)
        for mutate in (
            lambda s: s["training_config"].update(annotations_sha256="f" * 64),
            lambda s: s["training_config"].update(base_checkpoint_sha256="f" * 64),
            lambda s: s["initial_cache_state_dict"]["resized_cache"].add_(1),
            lambda s: s["planned_dataset_indices"].reverse(),
        ):
            changed = deepcopy(state)
            mutate(changed)
            with self.assertRaises(ValueError):
                evaluation.compare_training_inputs(state, changed)

    def test_checkpoint_uses_new_format_validator_and_live_code_gate(self):
        state, *_ = checkpoint_fixture(steps=20)
        sources = evaluation.training.implementation_sources()
        state["training_config"]["implementation_sha256"] = {path.name: "a" * 64 for path in sources}
        with patch.object(evaluation.shared, "sha256", return_value="a" * 64), \
             patch.object(evaluation.torch, "load", return_value=state), \
             patch.object(evaluation.training, "verify_training_identity", return_value={"actual_prefix_verified": True}), \
             patch.object(evaluation.semantic, "verify_initial_cache_artifact", return_value={"path": "/cache", "sha256": "c" * 64}):
            loaded, metadata = evaluation.read_checkpoint(Path("/input.pt"), input_format=True,
                minimum_samples=20, base_hash="a" * 64, tokenizer_hash="b" * 64)
            self.assertIs(loaded, state)
            self.assertEqual(metadata["format"], evaluation.training.FORMAT)
            state["training_config"]["implementation_sha256"][sources[0].name] = "f" * 64
            with self.assertRaisesRegex(ValueError, "implementation"):
                evaluation.read_checkpoint(Path("/input.pt"), input_format=True,
                    minimum_samples=20, base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_mutating_checkpoint_is_rejected_before_using_provenance(self):
        state, *_ = checkpoint_fixture(steps=20)
        with patch.object(evaluation.shared, "sha256", side_effect=["a" * 64, "b" * 64]), \
             patch.object(evaluation.torch, "load", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                evaluation.read_checkpoint(Path("/input.pt"), input_format=True,
                    minimum_samples=20, base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_cli_defaults_full_validation_and_controls_optional_comparators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flags = ["--data-root", str(root / "val"), "--base-checkpoint", str(root / "base.pt"),
                     "--input-checkpoint", str(root / "input.pt"), "--output-dir", str(root / "new")]
            args = evaluation.parse_args(flags)
            self.assertEqual(args.labels, ["baseline", "input-ve"])
            self.assertIsNone(args.indices)
            self.assertEqual(args.minimum_samples_seen, 2000)
            self.assertEqual(evaluation.parse_args(flags + ["--output-delta-checkpoint", str(root / "output.pt")]).labels,
                             list(evaluation.LABELS))
            self.assertEqual(evaluation.parse_args(flags + ["--minimum-samples-seen", "20", "--indices", "0,18,20"]).minimum_samples_seen, 20)
            for extras in (["--minimum-samples-seen", "20"], ["--variant", "output-delta"],
                           ["--gpu-memory-fraction", "nan"], ["--render-count", "13"],
                           ["--deadline", "2026-09-11T09:00:00"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    evaluation.parse_args(flags + extras)

    def test_report_explicitly_warns_reference_empty_is_not_real_false_positive(self):
        report = evaluation.render_report({"evaluated_images": 3, "full_val_evaluated": False,
            "spatial_metrics": {}, "comparison": {"matched_actual_training_budget": True, "matched_training_numerics": False}})
        self.assertIn("不能直接当作真实误检率", report)
        self.assertIn("参考漏标与模型错侧同时存在", report)
        self.assertIn("RIGHT hand", evaluation.REFERENCE_ABSENCE_WARNING)
        self.assertIn("LEFT prompt", evaluation.REFERENCE_ABSENCE_WARNING)
        self.assertIn("实际调用原始自然文本 VE", report)
        self.assertIn("不作为单因素架构消融", report)


if __name__ == "__main__":
    unittest.main()

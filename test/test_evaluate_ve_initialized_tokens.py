"""CPU validation of semantic-delta checkpoint and evaluation identities."""

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import torch

from scripts.cached_ve_text_features import CachedVETextEncoder
from scripts.evaluate_ve_initialized_tokens import (
    FORMAT, IdentityCheckedDataset, cache_from_state, validate_checkpoint,
    assert_finite_model_outputs, verify_initial_cache_artifact, verify_training_identities,
)


def state():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, :4] = False
    encoder = CachedVETextEncoder(padding, torch.zeros(32, 2, 256), torch.zeros(32, 2, 1024),
                                 metadata={"base_checkpoint_sha256": "a" * 64,
                                           "tokenizer_sha256": "b" * 64}, mode="zero_delta")
    initial = copy.deepcopy(encoder.state_dict())
    with torch.no_grad():
        encoder.delta.add_(.01)
    return {"format": FORMAT, "base_checkpoint_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "training_config": {"base_checkpoint_sha256": "a" * 64,
                                "tokenizer_sha256": "b" * 64, "batch_size": 1, "amp": True},
            "cache_state_dict": encoder.state_dict(), "initial_cache_state_dict": initial,
            "progress": {"completed_steps": 20, "samples_seen": 20,
                         "planned_steps": 2000, "planned_samples": 2000, "pilot_complete": False},
            "next_step": 20, "planned_dataset_indices": list(range(2000)),
            "observed_image_ids": list(range(20))}


class SemanticDeltaEvaluationTest(unittest.TestCase):
    def validate(self, value, minimum=20):
        return validate_checkpoint(value, minimum_samples=minimum, base_hash="a" * 64,
                                   tokenizer_hash="b" * 64)

    def test_partial_checkpoint_has_actual_counts_and_cannot_pass_full_budget_gate(self):
        trained, initial = self.validate(state())
        self.assertGreater(float(trained.delta.detach().abs().sum()), 0.)
        self.assertEqual(float(initial.delta.detach().abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, "progress"):
            self.validate(state(), minimum=2000)

    def test_random_tokens_changed_base_nonzero_initial_or_modified_cache_rejected(self):
        for mutation in ("class_tokens", "base", "initial", "cache", "count"):
            value = state()
            if mutation == "class_tokens":
                value["class_tokens"] = torch.zeros(2, 4, 256)
            elif mutation == "base":
                value["base_checkpoint_sha256"] = "c" * 64
            elif mutation == "initial":
                value["initial_cache_state_dict"]["delta"].add_(.1)
            elif mutation == "cache":
                value["cache_state_dict"]["resized_cache"].add_(.1)
            else:
                value["progress"]["samples_seen"] = 21
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(value)

    def test_frozen_baseline_has_no_delta_and_keeps_complete_original_features(self):
        value = state()
        baseline = cache_from_state(value["initial_cache_state_dict"], frozen=True)
        self.assertIsNone(baseline.delta)
        self.assertEqual(tuple(baseline.resized_cache.shape), (32, 2, 256))
        self.assertEqual(tuple(baseline.raw_cache.shape), (32, 2, 1024))

    def test_dataset_rejects_fallback_missing_side_and_reference_prompts(self):
        def query(side, image_id=7):
            return SimpleNamespace(query_text=side, image_id=0, input_bbox=None, input_points=None,
                                   inference_metadata=SimpleNamespace(coco_image_id=image_id,
                                       original_category_id=1 if side == "left_hand" else 2))
        sample = SimpleNamespace(find_queries=[query("left_hand"), query("right_hand")])
        checked = IdentityCheckedDataset([sample], [{"id": 7}])
        self.assertIs(checked[0], sample)
        self.assertEqual(checked.observed_indices, [0])
        sample.find_queries[0].inference_metadata.coco_image_id = 8
        with self.assertRaisesRegex(RuntimeError, "substituted"):
            checked[0]
        sample.find_queries[0] = query("left_hand")
        sample.find_queries[0].input_bbox = torch.ones(4)
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            checked[0]
        sample.find_queries.pop()
        with self.assertRaisesRegex(RuntimeError, "two distinct"):
            checked[0]

    def test_training_sample_prefix_is_checked_against_real_annotation_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps({"images": [{"id": index + 5} for index in range(2001)]}))
            digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
            value = state()
            value["training_config"].update(data_root=str(root), annotations_sha256=digest, seed=123)
            value["annotation_summary"] = {"sha256": digest, "images": 2001}
            order = list(range(2001))
            random.Random(123).shuffle(order)
            value["planned_dataset_indices"] = order[:2000]
            value["observed_image_ids"] = [index + 5 for index in order[:20]]
            self.assertTrue(verify_training_identities(value)["actual_prefix_verified"])
            value["observed_image_ids"][0] += 1
            with self.assertRaisesRegex(ValueError, "Observed"):
                verify_training_identities(value)

    def test_initial_cache_artifact_must_reproduce_unchanged_training_features(self):
        with tempfile.TemporaryDirectory() as directory:
            value = state()
            path = Path(directory) / "initial.pt"
            torch.save(value["initial_cache_state_dict"], path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            value["training_config"].update(initial_cache=str(path), initial_cache_sha256=digest)
            value["initial_cache_sha256"] = digest
            self.assertTrue(verify_initial_cache_artifact(value)["verified_against_initial_state"])
            value["initial_cache_state_dict"]["raw_cache"].add_(1.)
            with self.assertRaisesRegex(ValueError, "Initial features"):
                verify_initial_cache_artifact(value)

    def test_nonfinite_masks_are_rejected_before_nan_can_become_empty_prediction(self):
        output = {name: torch.ones(1) for name in ("pred_logits", "presence_logit_dec", "pred_boxes", "pred_masks")}
        assert_finite_model_outputs(None, None, [output])
        output["pred_masks"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "pred_masks"):
            assert_finite_model_outputs(None, None, [output])


if __name__ == "__main__":
    unittest.main()

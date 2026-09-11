from copy import deepcopy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import cached_ve_text_features as cached
from scripts import train_ve_initialized_tokens as training


def encoder():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, :4] = False
    return cached.CachedVETextEncoder(padding, torch.ones(32, 2, 256, dtype=torch.bfloat16),
                                      torch.ones(32, 2, 1024), mode="zero_delta",
                                      metadata={"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64})


def checkpoint_fixture(steps=1):
    model = encoder()
    initial = training.cpu_state(model.state_dict())
    optimizer = torch.optim.AdamW([model.delta], lr=.01, weight_decay=0.)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(list(cached.CLASS_NAMES))[1].float().sum().backward()
        optimizer.step()
    core = {"sam3/fake.py": "e" * 64}
    config = {"learning_rate": .01, "base_checkpoint_sha256": "a" * 64,
              "tokenizer_sha256": "b" * 64, "initial_cache_sha256": "c" * 64,
              "annotations_sha256": "d" * 64, "core_sources_sha256": training.json_hash(core)}
    ids = list(range(10000, 12002))
    order = training.legacy.build_epoch_order(len(ids), 123)[:2000]
    annotations = {"images": len(ids), "sha256": "d" * 64}
    state = training.make_checkpoint(encoder=model, optimizer=optimizer, config=config, initial_state=initial,
                                      annotation_summary=annotations, order=order,
                                      observed_ids=[ids[index] for index in order[:steps]], loss_history=[1.] * steps,
                                      gradient_counts=[steps, steps], core_hashes=core)
    return state, config, order, ids, initial


class SemanticPilotTest(unittest.TestCase):
    def test_initial_cache_exact_semantic_state_and_legacy_loader_disjoint_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified_cache.pt"
            original = encoder()
            torch.save(original.state_dict(), path)
            loaded = training.load_initial_cache(path, base_hash="a" * 64, tokenizer_hash="b" * 64)
            self.assertEqual(training.cache_fingerprint(loaded.state_dict()), training.cache_fingerprint(original.state_dict()))
            state, *_ = checkpoint_fixture()
            self.assertEqual(state["format"], "sam3-ve-initialized-delta-training-v1")
            self.assertNotIn("class_tokens", state)
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                training.load_initial_cache(path, base_hash="f" * 64, tokenizer_hash="b" * 64)
            with torch.no_grad():
                original.delta.fill_(.01)
            torch.save(original.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "exactly zero"):
                training.load_initial_cache(path, base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_resume_reconstructs_only_true_successful_prefix_and_rejects_changes(self):
        state, config, order, ids, initial = checkpoint_fixture()
        self.assertEqual(training.validate_resume(state, config, order, ids, initial), 1)
        variants = []
        changed = deepcopy(state)
        changed["observed_image_ids"][0] = -1
        variants.append(changed)
        changed = deepcopy(state)
        changed["training_config"]["learning_rate"] = .003
        variants.append(changed)
        changed = deepcopy(state)
        changed["cache_state_dict"]["raw_cache"][0, 0, 0] = 99
        variants.append(changed)
        changed = deepcopy(state)
        changed["gradient_nonzero_steps"] = [0, 1]
        variants.append(changed)
        changed = deepcopy(state)
        changed["progress"]["samples_seen"] = 2
        variants.append(changed)
        changed = deepcopy(state)
        changed["planned_dataset_indices_sha256"] = "0" * 64
        variants.append(changed)
        for value in variants:
            with self.assertRaises((ValueError, RuntimeError)):
                training.validate_resume(value, config, order, ids, initial)

    def test_optimizer_cannot_silently_override_learning_rate_or_moment_progress(self):
        state, *_ = checkpoint_fixture()
        training.validate_optimizer_state(state["optimizer"], 1)
        for mutation in ("learning_rate", "step", "nan", "weight_decay"):
            changed = deepcopy(state["optimizer"])
            moments = next(iter(changed["state"].values()))
            if mutation == "learning_rate":
                changed["param_groups"][0]["lr"] = .003
            elif mutation == "step":
                moments["step"] += 1
            elif mutation == "nan":
                moments["exp_avg"][0, 0, 0] = float("nan")
            else:
                changed["param_groups"][0]["weight_decay"] = .01
            with self.assertRaises(ValueError):
                training.validate_optimizer_state(changed, 1)

    def test_atomic_checkpoint_preserves_old_file_and_weights_only_round_trip(self):
        state, config, order, ids, initial = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            training.atomic_save(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(training.validate_resume(loaded, config, order, ids, initial), 1)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                training.atomic_save(path, state)
            self.assertEqual(path.read_bytes(), original)
            with patch.object(torch, "save", side_effect=RuntimeError("simulated write failure")):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    training.atomic_save(path, state, replace=True)
            self.assertEqual(path.read_bytes(), original)

    def test_rng_restores_python_numpy_and_torch_without_inventing_prior_ids(self):
        original = training.rng_state()
        try:
            random.seed(77)
            np.random.seed(77)
            torch.manual_seed(77)
            saved = training.rng_state()
            expected = (random.random(), np.random.rand(), torch.rand(3))
            training.restore_rng(saved)
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual[:2], expected[:2])
            self.assertTrue(torch.equal(actual[2], expected[2]))
        finally:
            training.restore_rng(original)

    def test_pilot_progress_and_first_2000_order_are_not_two_epochs(self):
        result = training.progress(20, 23265)
        self.assertEqual(result["samples_seen"], 20)
        self.assertEqual(result["planned_samples"], 2000)
        self.assertFalse(result["full_epoch_completed"])
        self.assertFalse(result["pilot_complete"])
        final = training.progress(2000, 23265)
        self.assertTrue(final["pilot_complete"])
        self.assertFalse(final["full_epoch_completed"])
        order = training.legacy.build_epoch_order(23265, 123)[:2000]
        self.assertEqual(order[:10], [17951, 14502, 4839, 7629, 4973, 3691, 14261, 14718, 3682, 16008])
        self.assertEqual(len(set(order)), 2000)

    def test_builder_requests_training_matcher_then_keeps_network_eval(self):
        fake = torch.nn.Linear(1, 1)
        fake.matcher = lambda *args: None
        fake.num_interactive_steps_val = 0
        args = SimpleNamespace(base_checkpoint=Path("/base.pt"), tokenizer_path=Path("/tokenizer.gz"))
        with patch("sam3.model_builder.build_sam3_image_model", return_value=fake) as builder:
            result = training.build_model_with_matcher(args)
            self.assertIs(result, fake)
            self.assertFalse(builder.call_args.kwargs["eval_mode"])
            self.assertEqual(builder.call_args.kwargs["text_encoder_type"], "ve")
            self.assertFalse(result.training)
            fake.matcher = None
            with self.assertRaisesRegex(RuntimeError, "matcher"):
                training.build_model_with_matcher(args)

    def test_loss_settings_match_legacy_control_and_geometry_prompts_rejected(self):
        with patch("sam3.train.loss.loss_fns.Masks") as masks, patch("sam3.train.loss.loss_fns.Boxes") as boxes, \
                patch("sam3.train.loss.loss_fns.IABCEMdetr") as classification:
            training.build_loss_functions()
            self.assertEqual(masks.call_args.kwargs["weight_dict"], {"loss_mask": 1., "loss_dice": 1.})
            self.assertEqual(boxes.call_args.kwargs["weight_dict"], {"loss_bbox": 1., "loss_giou": 1.})
            values = classification.call_args.kwargs
            self.assertEqual(values["weight_dict"], {"loss_ce": 1., "presence_loss": 1.})
            self.assertTrue(values["use_presence"])
            self.assertEqual(values["pos_weight"], 5.)
            self.assertEqual(values["presence_gamma"], 0.)
        batch = SimpleNamespace(find_inputs=[SimpleNamespace(input_boxes=torch.empty(0, 4))])
        training.validate_unprompted_batch(batch)
        batch.find_inputs[0].input_boxes = torch.zeros(1, 4)
        with self.assertRaises(RuntimeError):
            training.validate_unprompted_batch(batch)


if __name__ == "__main__":
    unittest.main()

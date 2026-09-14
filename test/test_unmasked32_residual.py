"""Explicit 32-slot unmasking, strict initialization, and save/resume contracts."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import cached_ve_text_features as cached
from scripts import train_residual_ddp as training
from scripts import train_ve_initialized_tokens as shared
from scripts.residual_ddp_checkpoint import configured_delta_shape, validate_resume
from test_cached_ve_text_features import make_cache
from test_train_residual_ddp import _fixture, _loop_patches


class Unmasked32Test(unittest.TestCase):
    def test_zero_delta_changes_only_returned_mask_not_cache_or_features(self):
        for dtype in (torch.float32, torch.bfloat16):
            original = make_cache("zero_delta", dtype)
            model = make_cache("unmasked32_delta", dtype)
            names = ["right_hand", "left_hand", "right_hand"]
            before, after = original(names), model(names)
            self.assertEqual((~after[0]).sum(1).tolist(), [32, 32, 32])
            self.assertTrue(torch.equal(before[1], after[1]))
            self.assertTrue(torch.equal(before[2], after[2]))
            self.assertTrue(torch.equal(model.padding_cache, original.padding_cache))
            self.assertEqual(model.delta.numel(), 16384)
            self.assertEqual(cached.validate_delta_state(model.state_dict()), (2, 32, 256))
            # Returned mask is not an alias of frozen provenance.
            after[0].fill_(True)
            self.assertEqual((~model(names)[0]).sum(1).tolist(), [32, 32, 32])

    def test_all_positions_update_receive_gradient_and_pooling_is_not_original(self):
        model = make_cache("unmasked32_delta")
        names = list(cached.CLASS_NAMES)
        state = deepcopy(model.state_dict())
        mask, features, raw = model(names)
        old_valid = (~model.padding_cache).T.unsqueeze(-1)
        original_pool = (features * old_valid).sum(0) / old_valid.sum(0)
        new_pool = features.mean(0)
        self.assertFalse(torch.allclose(original_pool, new_pool))
        new_pool.sum().backward()
        self.assertTrue(torch.equal(model.delta.grad, torch.full_like(model.delta, 1 / 32)))
        with torch.no_grad():
            model.delta[0].fill_(.25)
            model.delta[1].fill_(.5)
        after = model(names)
        self.assertTrue(torch.equal(after[1][:, 0], features[:, 0] + .25))
        self.assertTrue(torch.equal(after[1][:, 1], features[:, 1] + .5))
        self.assertTrue(torch.equal(raw, after[2]))
        for key in ("resized_cache", "raw_cache", "padding_cache", "valid_positions"):
            self.assertTrue(torch.equal(model.state_dict()[key], state[key]))

    def test_config_explicit_non_equivalence_and_mask_policy_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            args.residual_positions, args.preflight_only = "unmasked32", True
            output = io.StringIO()
            with redirect_stdout(output):
                training.run(args)
            config = json.loads(output.getvalue())["config"]
            self.assertEqual(configured_delta_shape(config), (2, 32, 256))
            self.assertIs(config["zero_delta_original_ve_equivalent"], False)
            for key in ("prompt_mask_policy", "zero_delta_original_ve_equivalent", "residual_mode",
                        "trainable_parameter_count"):
                corrupt = deepcopy(config)
                corrupt.pop(key)
                with self.assertRaises(ValueError):
                    configured_delta_shape(corrupt)
            config["residual_positions"] = "all"
            with self.assertRaises(ValueError):
                configured_delta_shape(config)

    def test_no_unmasked_state_may_masquerade_as_original_initializer(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            source = torch.load(args.initial_cache, weights_only=True)
            hashes = dict(base_hash=source["_extra_state"]["metadata"]["base_checkpoint_sha256"],
                          tokenizer_hash=source["_extra_state"]["metadata"]["tokenizer_sha256"])
            encoder = shared.load_initial_cache(args.initial_cache, residual_positions="unmasked32", **hashes)
            torch.save(encoder.state_dict(), args.initial_cache)
            with self.assertRaises(ValueError):
                shared.load_initial_cache(args.initial_cache, residual_positions="unmasked32", **hashes)
            state = encoder.state_dict()
            state["padding_cache"] = torch.zeros(2, 32, dtype=torch.bool)
            with self.assertRaises(ValueError):
                cached.validate_delta_state(state)

    def test_actual_training_loop_resume_identical_to_continuous_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _fixture(root)
            args.residual_positions, args.stop_after_step = "unmasked32", 1
            with ExitStack() as stack, redirect_stdout(io.StringIO()):
                _loop_patches(stack, [])
                training.run(args)
                resumed = deepcopy(args)
                resumed.resume = args.output_dir / "latest.pt"
                resumed.output_dir, resumed.stop_after_step = root / "resume", None
                training.run(resumed)
                continuous = deepcopy(args)
                continuous.output_dir, continuous.stop_after_step = root / "continuous", None
                training.run(continuous)
            actual = torch.load(resumed.output_dir / "latest.pt", weights_only=True)
            expected = torch.load(continuous.output_dir / "latest.pt", weights_only=True)
            self.assertEqual(actual["trainable_parameter_count"], 16384)
            self.assertEqual(actual["validation_state"], expected["validation_state"])
            self.assertEqual(actual["rank_cache_audit"], expected["rank_cache_audit"])
            self.assertTrue(torch.equal(actual["cache_state_dict"]["delta"], expected["cache_state_dict"]["delta"]))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(actual["optimizer"]["state"][0][key], expected["optimizer"]["state"][0][key]))
            from scripts.evaluate_residual_test import load_residual
            loaded, metadata = load_residual(resumed.output_dir / "latest.pt", args.initial_cache,
                root / "train", base_hash=actual["training_config"]["base_sha256"],
                tokenizer_hash=actual["training_config"]["tokenizer_sha256"])
            self.assertEqual(metadata["trained_parameter_count"], 16384)
            self.assertEqual((~loaded(list(cached.CLASS_NAMES))[0]).sum(1).tolist(), [32, 32])
            self.assertFalse(loaded.delta.requires_grad)
            corrupt = deepcopy(actual)
            corrupt.pop("trainable_parameter_count")
            ids = sorted(row["id"] for row in json.loads((root / "train/annotations.json").read_bytes())["images"])
            with self.assertRaises(ValueError):
                validate_resume(corrupt, actual["training_config"], ids, actual["initial_cache_state_dict"])


if __name__ == "__main__":
    unittest.main()

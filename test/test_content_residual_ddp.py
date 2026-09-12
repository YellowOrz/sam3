"""Content-only layout through the real CPU-mocked DDP orchestration and resume."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from scripts import cached_ve_text_features as cached
from scripts import train_residual_ddp as training
from scripts import train_ve_initialized_tokens as shared
from scripts.residual_ddp_checkpoint import canonical_hash
from test_train_residual_ddp import _fixture, _loop_patches


class ContentResidualEntryTest(unittest.TestCase):
    def test_cli_defaults_and_explicit_content_policy(self):
        required = ["--approval", "/fixture/a", "--base-checkpoint", "/fixture/b",
                    "--initial-cache", "/fixture/c", "--output-dir", "/fixture/o"]
        self.assertEqual(training.parse_args(required).residual_positions, "all")
        self.assertEqual(training.parse_args(required + ["--residual-positions", "content"]).residual_positions,
                         "content")
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            training.parse_args(required + ["--residual-positions", "two"])

    def test_initial_loader_keeps_full_original_features_for_frozen_and_zero_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            original = torch.load(args.initial_cache, weights_only=True)
            for source_mode in ("zero_delta", "frozen"):
                source = deepcopy(original)
                source["_extra_state"]["mode"] = source_mode
                if source_mode == "frozen":
                    source.pop("delta")
                torch.save(source, args.initial_cache)
                hashes = dict(base_hash=source["_extra_state"]["metadata"]["base_checkpoint_sha256"],
                              tokenizer_hash=source["_extra_state"]["metadata"]["tokenizer_sha256"])
                full = shared.load_initial_cache(args.initial_cache, **hashes)
                content = shared.load_initial_cache(args.initial_cache, residual_positions="content", **hashes)
                self.assertEqual(content.mode, "content_delta")
                self.assertEqual(content.delta.numel(), 1024)
                self.assertTrue(bool((content.delta == 0).all()))
                for actual, expected in zip(content(list(cached.CLASS_NAMES)), full(list(cached.CLASS_NAMES))):
                    self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(content.valid_positions, original["valid_positions"]))
                self.assertEqual(tuple(content.padding_cache.shape), (2, 32))
                self.assertEqual((~content.padding_cache).sum(dim=1).tolist(), [4, 4])

    def test_trained_or_content_source_cannot_masquerade_as_original_initializer(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            original = torch.load(args.initial_cache, weights_only=True)
            hashes = dict(base_hash=original["_extra_state"]["metadata"]["base_checkpoint_sha256"],
                          tokenizer_hash=original["_extra_state"]["metadata"]["tokenizer_sha256"])
            for mutation in ("trained", "content", "frozen_with_delta", "missing_delta"):
                state = deepcopy(original)
                if mutation == "trained":
                    state["delta"].fill_(.01)
                elif mutation == "content":
                    state["_extra_state"]["mode"] = "content_delta"
                    state["delta"] = torch.zeros(2, 2, 256)
                elif mutation == "frozen_with_delta":
                    state["_extra_state"]["mode"] = "frozen"
                else:
                    state.pop("delta")
                torch.save(state, args.initial_cache)
                for policy in ("all", "content"):
                    with self.subTest(mutation=mutation, policy=policy), self.assertRaises((ValueError, RuntimeError)):
                        shared.load_initial_cache(args.initial_cache, residual_positions=policy, **hashes)

    def test_content_preflight_binds_layout_without_cuda_and_default_config_stays_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            args.preflight_only = True
            reports = {}
            for policy in ("all", "content"):
                args.residual_positions = policy
                stdout = io.StringIO()
                with redirect_stdout(stdout), mock.patch.object(torch.cuda, "is_available",
                                                                side_effect=AssertionError("CUDA preflight")):
                    training.run(args)
                reports[policy] = json.loads(stdout.getvalue())["config"]
            self.assertEqual(reports["all"]["delta_shape"], [2, 4, 256])
            for field in ("residual_positions", "residual_mode", "trainable_parameter_count"):
                self.assertNotIn(field, reports["all"])
            self.assertEqual(reports["content"]["delta_shape"], [2, 2, 256])
            self.assertEqual(reports["content"]["residual_positions"], "content")
            self.assertEqual(reports["content"]["residual_mode"], "content_delta")
            self.assertEqual(reports["content"]["trainable_parameter_count"], 1024)
            self.assertNotEqual(canonical_hash(reports["all"]), canonical_hash(reports["content"]))
            self.assertFalse(args.output_dir.exists())

    def test_content_real_loop_save_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            args.residual_positions, args.stop_after_step = "content", 1
            calls = []
            with ExitStack() as stack, redirect_stdout(io.StringIO()):
                _loop_patches(stack, calls)
                training.run(args)
                first = torch.load(args.output_dir / "latest.pt", weights_only=True)
                self.assertEqual(first["trainable_parameter_count"], 1024)
                self.assertEqual(first["progress"]["global_step"], 1)
                resumed = deepcopy(args)
                resumed.output_dir, resumed.resume = directory / "resumed", args.output_dir / "latest.pt"
                resumed.stop_after_step = None
                training.run(resumed)
                continuous = deepcopy(args)
                continuous.output_dir, continuous.stop_after_step = directory / "continuous", None
                training.run(continuous)
            actual = torch.load(resumed.output_dir / "latest.pt", weights_only=True)
            expected = torch.load(continuous.output_dir / "latest.pt", weights_only=True)
            self.assertTrue(actual["progress"]["training_complete"])
            self.assertEqual(actual["progress"]["global_step"], 6)
            self.assertEqual(actual["validation_state"], expected["validation_state"])
            self.assertEqual(actual["rank_states"][0]["image_ids"], expected["rank_states"][0]["image_ids"])
            self.assertEqual(actual["rank_cache_audit"], expected["rank_cache_audit"])
            self.assertTrue(torch.equal(actual["cache_state_dict"]["delta"], expected["cache_state_dict"]["delta"]))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(actual["optimizer"]["state"][0][key],
                                            expected["optimizer"]["state"][0][key]))
            summary = json.loads((resumed.output_dir / "summary.json").read_text())
            self.assertEqual(summary["trainable_parameter_count"], 1024)
            self.assertEqual(summary["training_config"]["delta_shape"], [2, 2, 256])


if __name__ == "__main__":
    unittest.main()

"""CPU entry-point contracts; CUDA/model work is replaced by a tiny objective.

These tests exercise the real run loop, sampler, checkpoint schema, optimizer,
RNG restoration and metadata gate. They do not claim a full SAM3 GPU smoke.
Real Gloo/DDP gradient equivalence is covered in test_residual_ddp_runtime.py.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from scripts import cached_ve_text_features as cached
from scripts import residual_ddp_validation as validation
from scripts import train_residual_ddp as training
from scripts.residual_ddp_runtime import DistributedContext


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fixture(directory):
    base, tokenizer, cache = (directory / name for name in ("base.pt", "tokenizer.gz", "initial-cache.pt"))
    base.write_bytes(b"base fixture; full model builder is mocked")
    tokenizer.write_bytes(b"tokenizer fixture")
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, :4] = False
    encoder = cached.CachedVETextEncoder(
        padding, torch.ones(32, 2, 256, dtype=torch.bfloat16), torch.ones(32, 2, 1024),
        mode="zero_delta", metadata={"base_checkpoint_sha256": _digest(base), "tokenizer_sha256": _digest(tokenizer)})
    torch.save(encoder.state_dict(), cache)
    approval = {"format": "sam3-residual-training-selection-v1", "approved_by": "user"}
    for split, size in (("train", 7), ("val", 3)):
        root = directory / split
        (root / "images").mkdir(parents=True)
        images = []
        for index in range(size):
            file_name = f"images/image-{index}.png"
            rgb = root / file_name
            rgb.write_bytes(f"CPU fixture {split} {index}".encode())
            images.append({"id": 1000 + index * 3, "file_name": file_name, "width": 2, "height": 2,
                           "source_dataset": "dexycb", "source_split": split,
                           "source_rgb_sha256": _digest(rgb)})
        _write_json(root / "annotations.json", {"info": {"dataset_role": split}, "images": images,
                                                "annotations": [], "categories": [
                                                    {"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]})
        approval[split] = {"root": str(root), "annotations_sha256": _digest(root / "annotations.json"),
                           "exhaustive_hand_labels": True, "allowed_image_roots": []}
    approval_path = directory / "approval.json"
    _write_json(approval_path, approval)
    return training.parse_args([
        "--approval", str(approval_path), "--base-checkpoint", str(base), "--initial-cache", str(cache),
        "--tokenizer-path", str(tokenizer), "--output-dir", str(directory / "output"),
        "--epochs", "2", "--batch-size-per-rank", "2", "--num-workers", "0",
        "--checkpoint-every", "2", "--log-every", "2", "--no-tensorboard"])


class _TinyObjective(nn.Module):
    check_frozen_contract = training.ResidualObjective.check_frozen_contract

    def __init__(self, encoder):
        super().__init__()
        self.model = nn.Module()
        self.model.frozen = nn.Parameter(torch.tensor(0.75), requires_grad=False)
        self.model.backbone = nn.Module()
        self.model.backbone.language_backbone = encoder
        self.eval()
        # Construction legitimately consumes RNG before checkpoint restoration.
        torch.rand(13)
        random.random()
        np.random.random()

    @property
    def encoder(self):
        return self.model.backbone.language_backbone

    def forward(self, batch):
        noise = torch.rand_like(self.encoder.delta) + random.random() + float(np.random.random())
        target = noise + batch.dataset_indices.float().mean() / 100 + self.model.frozen
        loss = (self.encoder.delta - target).square().mean()
        return loss, (loss.detach() / 6).repeat(6), torch.zeros(())


class _ImageBatch:
    @staticmethod
    def is_pinned():
        return True


class _Batch:
    def __init__(self, indices, contract):
        self.image_ids = tuple(contract.images[index]["id"] for index in indices)
        self.datapoint = SimpleNamespace(img_batch=_ImageBatch(), dataset_indices=torch.tensor(indices))

    def to(self, device, *, non_blocking=True):
        if torch.device(device).type != "cpu" or non_blocking is not True:
            raise AssertionError("CPU orchestration test unexpectedly accessed CUDA")
        return self


class _Event:
    def __init__(self, **unused):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.1


def _validation_metrics(score=.5):
    return {"images": 3, "queries": 6, "targets": 4, "total_loss": 1.,
            **{f"{side}/{key}": value for side in ("left_hand", "right_hand")
               for key, value in {"positive_count": 2, "absent_count": 1,
                                  "false_negative_count": 0, "false_positive_count": 0,
                                  "miss_zero_dice": score}.items()}}


def _loop_patches(stack, validation_calls, *, scores=None, fail_at_step=None):
    context = DistributedContext(0, 0, 1, torch.device("cpu"), None)
    stack.enter_context(mock.patch.object(training, "initialize_distributed", return_value=context))
    stack.enter_context(mock.patch.object(training, "implementation_hashes", return_value={"scripts/fixture.py": "a" * 64}))
    stack.enter_context(mock.patch.object(training, "build_training_model", side_effect=lambda args, encoder, context: _TinyObjective(encoder)))
    stack.enter_context(mock.patch.object(training, "make_identity_dataset", side_effect=lambda root, contract: contract))
    stack.enter_context(mock.patch.object(training, "make_loader", side_effect=lambda dataset, *, batch_sampler, **kwargs:
                                         (_Batch(indices, dataset) for indices in batch_sampler)))
    stack.enter_context(mock.patch.object(training, "gather_gpu_names_local", return_value=["CPU test double"]))
    stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
    # AdamW checks graph-capture state when is_available is true, even for CPU
    # parameters. Keep that check local instead of probing an actual GPU driver.
    stack.enter_context(mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
    stack.enter_context(mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization in CPU test")))
    stack.enter_context(mock.patch.object(torch.cuda, "Event", _Event))
    for name in ("set_per_process_memory_fraction", "reset_peak_memory_stats", "empty_cache"):
        stack.enter_context(mock.patch.object(torch.cuda, name))
    stack.enter_context(mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0))
    stack.enter_context(mock.patch.object(torch.amp, "autocast", side_effect=lambda *args, **kwargs: nullcontext()))

    def evaluate(model, contract, dataset, *, context, output_dir, step, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=False)
        validation_calls.append(step)
        random.random()
        np.random.random(31)
        torch.rand(53)
        if step == fail_at_step:
            raise RuntimeError("Synthetic validation interrupted after epoch checkpoint")
        return _validation_metrics(scores[step] if scores is not None else .5)

    stack.enter_context(mock.patch.object(validation, "evaluate_validation", side_effect=evaluate))
    stack.enter_context(redirect_stdout(io.StringIO()))


class MainEntryTests(unittest.TestCase):
    def test_cli_early_stop_defaults_and_invalid_policies(self):
        required = ["--approval", "/fixture/a", "--base-checkpoint", "/fixture/b",
                    "--initial-cache", "/fixture/c", "--output-dir", "/fixture/o"]
        args = training.parse_args(required)
        self.assertEqual(args.early_stopping_patience, 3)
        self.assertEqual(args.early_stopping_min_delta, .001)
        for options in (("--early-stopping-patience", "0"),
                        ("--early-stopping-min-delta", "-0.1"),
                        ("--early-stopping-min-delta", "nan")):
            with self.subTest(options=options), mock.patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    training.parse_args(required + list(options))

    def test_all_rank_stop_state_must_match_not_just_scalar_best(self):
        state = {"last_validation_step": 3, "selection": {"best_metric": .5, "bad_epochs": 1}}
        training.require_rank_validation_consistency([state, deepcopy(state), deepcopy(state)])
        different = deepcopy(state)
        different["selection"]["bad_epochs"] = 2
        with self.assertRaisesRegex(ValueError, "across ranks"):
            training.require_rank_validation_consistency([state, state, different])
        with self.assertRaises(ValueError):
            training.require_rank_validation_consistency([])

    def test_implementation_hash_paths_are_identical_through_a_repository_symlink(self):
        project = Path(training.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / "repo-alias"
            link.symlink_to(project, target_is_directory=True)
            with mock.patch.object(training.shared.evaluation, "sha256", return_value="a" * 64):
                physical = training.implementation_hashes()
                with mock.patch.object(training, "__file__", str(link / "scripts/train_residual_ddp.py")):
                    symbolic = training.implementation_hashes()
            self.assertEqual(physical, symbolic)
            self.assertIn("scripts/train_residual_ddp.py", physical)
            self.assertIn("scripts/residual_selection.py", physical)

    def test_early_stop_saves_actual_best_even_when_small_gain_exhausts_patience(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            args.epochs, args.early_stopping_patience = 8, 2
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls, scores={0: .5, 3: .6, 6: .6005, 9: .6006})
                training.run(args)
            self.assertEqual(calls, [0, 3, 6, 9])
            latest = torch.load(args.output_dir / "latest.pt", weights_only=True)
            best = torch.load(args.output_dir / "best.pt", weights_only=True)
            self.assertEqual(latest["progress"]["global_step"], 9)
            self.assertFalse(latest["progress"]["training_complete"])
            self.assertEqual(best["progress"]["global_step"], 9)
            self.assertEqual(best["validation_state"]["selection"]["best_metric"], .6006)
            self.assertTrue(latest["validation_state"]["selection"]["stopped"])
            self.assertTrue(torch.equal(best["cache_state_dict"]["delta"], latest["cache_state_dict"]["delta"]))
            summary = json.loads((args.output_dir / "summary.json").read_text())
            self.assertEqual(summary["status"], "early_stopped")
            self.assertEqual(summary["best_checkpoint"], str(args.output_dir / "best.pt"))
            self.assertEqual(summary["selection"]["early_stop_reference_metric"], .6)

    def test_tensorboard_records_baseline_and_epoch_selection_metric_with_actual_steps(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as temporary:
            args = _fixture(Path(temporary))
            args.epochs, args.tensorboard = 1, True
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls, scores={0: .5, 3: .6})
                training.run(args)
            events = EventAccumulator(str(args.output_dir / "tensorboard"))
            events.Reload()
            values = events.Scalars("validation/dexycb_val_selection/macro_miss_zero_dice")
            self.assertEqual([event.step for event in values], [0, 3])
            self.assertAlmostEqual(values[0].value, .5)
            self.assertAlmostEqual(values[1].value, .6)
            self.assertIn("train/total_loss", events.Tags()["scalars"])
            best = torch.load(args.output_dir / "best.pt", weights_only=True)
            self.assertEqual(best["progress"]["global_step"], 3)

    def test_epoch_prevalidation_recovery_services_pending_validation_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls, fail_at_step=3)
                with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                    training.run(args)
            pending_path = args.output_dir / "step-00000003-recovery.pt"
            pending = torch.load(pending_path, weights_only=True)
            self.assertEqual(pending["progress"]["global_step"], 3)
            self.assertEqual(pending["validation_state"]["selection"]["last_validation_step"], 0)
            calls.clear()
            resumed = deepcopy(args)
            resumed.output_dir, resumed.resume = directory / "resumed-pending", pending_path
            continuous = deepcopy(args)
            continuous.output_dir = directory / "continuous"
            with ExitStack() as stack:
                _loop_patches(stack, calls)
                training.run(resumed)
                self.assertEqual(calls, [3, 6])
                training.run(continuous)
            actual = torch.load(resumed.output_dir / "latest.pt", weights_only=True)
            expected = torch.load(continuous.output_dir / "latest.pt", weights_only=True)
            self.assertEqual(actual["validation_state"], expected["validation_state"])
            self.assertTrue(torch.equal(actual["cache_state_dict"]["delta"], expected["cache_state_dict"]["delta"]))
            self.assertTrue(torch.equal(actual["optimizer"]["state"][0]["exp_avg"], expected["optimizer"]["state"][0]["exp_avg"]))

    def test_resume_inherits_earlier_best_and_terminal_resume_does_not_revalidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            args.epochs, args.early_stopping_patience = 5, 1
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls, scores={0: .8, 3: .7})
                training.run(args)
                resumed = deepcopy(args)
                resumed.output_dir, resumed.resume = directory / "terminal-resume", args.output_dir / "latest.pt"
                calls.clear()
                training.run(resumed)
                self.assertEqual(calls, [])
            best = torch.load(resumed.output_dir / "best.pt", weights_only=True)
            latest = torch.load(resumed.output_dir / "latest.pt", weights_only=True)
            self.assertEqual(best["progress"]["global_step"], 0)
            self.assertEqual(latest["progress"]["global_step"], 3)
            self.assertTrue(torch.equal(best["cache_state_dict"]["delta"], torch.zeros(2, 4, 256)))
            self.assertFalse(torch.equal(best["cache_state_dict"]["delta"], latest["cache_state_dict"]["delta"]))

    def test_missing_historical_best_rejected_instead_of_exporting_latest_as_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls)
                training.run(args)
                (args.output_dir / "step-00000000-best.pt").unlink()
                resumed = deepcopy(args)
                resumed.output_dir, resumed.resume = directory / "missing-best", args.output_dir / "latest.pt"
                with self.assertRaisesRegex(RuntimeError, "FileNotFoundError"):
                    training.run(resumed)

    def test_validation_can_be_disabled_and_partial_diagnostics_do_not_enter_selector(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            args.stop_after_step = 2
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls)
                training.run(args)
                partial = torch.load(args.output_dir / "latest.pt", weights_only=True)
                self.assertEqual(partial["validation_state"]["last_validation_step"], 2)
                self.assertEqual([row["step"] for row in partial["validation_state"]["selection"]["history"]], [0])
                disabled = deepcopy(args)
                disabled.validation = False
                disabled.output_dir = directory / "disabled"
                calls.clear()
                training.run(disabled)
                self.assertEqual(calls, [])
            saved = torch.load(disabled.output_dir / "latest.pt", weights_only=True)
            self.assertIsNone(saved["validation_state"])
            self.assertEqual(saved["training_config"]["validation"], {"enabled": False})
            self.assertFalse((disabled.output_dir / "best.pt").exists())

    def test_preflight_uses_real_contract_and_cache_without_cuda_groups_or_file_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            args.preflight_only = True
            before = {str(path.relative_to(directory)): _digest(path) for path in directory.rglob("*") if path.is_file()}
            output = io.StringIO()
            with ExitStack() as stack:
                for name in ("is_available", "_lazy_init", "set_device", "init"):
                    stack.enter_context(mock.patch.object(torch.cuda, name, side_effect=AssertionError("Preflight touched CUDA")))
                for name in ("initialize_distributed", "build_training_model", "atomic_checkpoint", "make_identity_dataset"):
                    stack.enter_context(mock.patch.object(training, name, side_effect=AssertionError("Preflight entered training")))
                stack.enter_context(mock.patch.object(Path, "mkdir", side_effect=AssertionError("Preflight created a directory")))
                stack.enter_context(redirect_stdout(output))
                training.run(args)
            report = json.loads(output.getvalue())
            self.assertEqual(report["status"], "preflight_only_no_training")
            self.assertEqual(report["config"]["steps_per_epoch"], 3)
            self.assertEqual(report["config"]["global_batch_size"], 2)
            self.assertFalse(args.output_dir.exists())
            after = {str(path.relative_to(directory)): _digest(path) for path in directory.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_two_steps_resume_matches_six_continuous_steps_and_validation_preserves_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base_args = _fixture(directory)
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls)
                uninterrupted = deepcopy(base_args)
                uninterrupted.output_dir = directory / "uninterrupted"
                training.run(uninterrupted)
                self.assertEqual(calls, [0, 3, 6])
                calls.clear()
                first = deepcopy(base_args)
                first.output_dir, first.stop_after_step = directory / "first-two", 2
                training.run(first)
                self.assertEqual(calls, [0, 2])
                calls.clear()
                resumed = deepcopy(base_args)
                resumed.output_dir = directory / "resumed"
                resumed.resume = first.output_dir / "step-00000002-final.pt"
                training.run(resumed)
                self.assertEqual(calls, [3, 6])
            expected = torch.load(uninterrupted.output_dir / "step-00000006-final.pt", weights_only=True)
            actual = torch.load(resumed.output_dir / "step-00000006-final.pt", weights_only=True)
            self.assertEqual(expected["progress"], actual["progress"])
            self.assertTrue(actual["progress"]["training_complete"])
            self.assertEqual(actual["progress"]["samples_seen"], 12)
            self.assertTrue(torch.equal(actual["cache_state_dict"]["delta"], expected["cache_state_dict"]["delta"]))
            for name in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(actual["optimizer"]["state"][0][name], expected["optimizer"]["state"][0][name]))
            self.assertEqual(actual["rank_states"][0]["image_ids"], expected["rank_states"][0]["image_ids"])
            self.assertEqual(actual["rank_states"][0]["rng"]["python"], expected["rank_states"][0]["rng"]["python"])
            self.assertTrue(torch.equal(actual["rank_states"][0]["rng"]["torch_cpu"], expected["rank_states"][0]["rng"]["torch_cpu"]))
            self.assertTrue(torch.equal(actual["rank_states"][0]["rng"]["numpy"]["keys"], expected["rank_states"][0]["rng"]["numpy"]["keys"]))
            self.assertEqual(len(list(resumed.output_dir.glob("step-*-recovery.pt"))), 3)
            self.assertTrue((resumed.output_dir / "summary.json").exists())
            self.assertFalse((resumed.output_dir / "failure.json").exists())

    def test_completed_checkpoint_resume_does_not_repeat_epoch_or_overwrite_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            args.epochs = 1
            calls = []
            with ExitStack() as stack:
                _loop_patches(stack, calls)
                training.run(args)
                resume = deepcopy(args)
                resume.resume, resume.output_dir = args.output_dir / "step-00000003-final.pt", directory / "completed-resume"
                calls.clear()
                training.run(resume)
                self.assertEqual(calls, [])
                before = {path.name: _digest(path) for path in args.output_dir.iterdir() if path.is_file()}
                with self.assertRaisesRegex(RuntimeError, "FileExistsError"):
                    training.run(args)
                after = {path.name: _digest(path) for path in args.output_dir.iterdir() if path.is_file()}
                self.assertEqual(before, after)

    def test_realsense_legacy_source_field_is_rejected_by_training_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            path = directory / "train/annotations.json"
            coco = json.loads(path.read_text())
            for image in coco["images"]:
                image.pop("source_dataset")
                image["source"] = "realsense"
            _write_json(path, coco)
            approval = json.loads(args.approval.read_text())
            approval["train"]["annotations_sha256"] = _digest(path)
            _write_json(args.approval, approval)
            with self.assertRaisesRegex(ValueError, "RealSense"):
                training.approval_contracts(args)

    def test_approval_sha_role_and_actual_train_val_overlap_are_enforced(self):
        for corruption in ("sha", "role", "overlap"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                args = _fixture(directory)
                approval = json.loads(args.approval.read_text())
                if corruption == "sha":
                    approval["train"]["annotations_sha256"] = "0" * 64
                elif corruption == "role":
                    path = directory / "train/annotations.json"
                    coco = json.loads(path.read_text())
                    coco["info"]["dataset_role"] = "test"
                    _write_json(path, coco)
                    approval["train"]["annotations_sha256"] = _digest(path)
                else:
                    image = directory / "val/images/image-0.png"
                    image.unlink()
                    image.symlink_to(directory / "train/images/image-0.png")
                    approval["val"]["allowed_image_roots"] = [str(directory / "train")]
                _write_json(args.approval, approval)
                with self.assertRaises(ValueError):
                    training.approval_contracts(args)

    def test_relocated_files_and_loader_settings_keep_semantic_training_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = _fixture(directory)
            _, approval_hash, contracts = training.approval_contracts(args)
            initial = {"delta": torch.zeros(2, 4, 256)}
            fingerprints = {"base": "a" * 64, "tokenizer": "b" * 64, "cache": "c" * 64,
                            "implementation": {"scripts/example.py": "d" * 64}}
            expected = training.training_configuration(args, contracts["train"], approval_hash, 3, initial, fingerprints,
                                                        contracts["val"])
            relocated = deepcopy(args)
            relocated.data_root, relocated.val_root = directory / "new-server/train", directory / "new-server/val"
            shutil.copytree(directory / "train", relocated.data_root)
            shutil.copytree(directory / "val", relocated.val_root)
            relocated.num_workers, relocated.prefetch_factor = 5, 3
            _, relocated_hash, relocated_contracts = training.approval_contracts(relocated)
            actual = training.training_configuration(relocated, relocated_contracts["train"], relocated_hash, 3, initial, fingerprints,
                                                      relocated_contracts["val"])
            self.assertEqual(expected, actual)

    def test_builder_moves_entire_objective_to_rank_device_after_legacy_cuda_string_handling(self):
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.language_backbone = nn.Linear(2, 2)
        model.frozen = nn.Parameter(torch.ones(2))
        model.matcher = lambda *args: None
        model.num_interactive_steps_val = 0
        padding = torch.ones(2, 32, dtype=torch.bool)
        padding[:, :4] = False
        encoder = cached.CachedVETextEncoder(padding, torch.ones(32, 2, 256), torch.ones(32, 2, 1024),
                                            mode="zero_delta", metadata={"base_checkpoint_sha256": "a" * 64,
                                                                          "tokenizer_sha256": "b" * 64})
        context = DistributedContext(3, 3, 4, torch.device("cuda", 3), "nccl")
        builder = mock.Mock(return_value=model)
        args = SimpleNamespace(base_checkpoint=Path("/fixture/base.pt"), tokenizer_path=Path("/fixture/bpe.gz"), anchor_weight=0.)
        with mock.patch.dict(sys.modules, {"sam3.model_builder": SimpleNamespace(build_sam3_image_model=builder)}), \
             mock.patch.object(encoder, "to", return_value=encoder), \
             mock.patch.object(training.shared, "build_loss_functions", return_value=()), \
             mock.patch.object(training.ResidualObjective, "to", autospec=True, side_effect=lambda self, device: self) as move, \
             mock.patch.object(torch.cuda, "empty_cache"):
            result = training.build_training_model(args, encoder, context)
        self.assertIs(result.model, model)
        self.assertFalse(builder.call_args.kwargs["eval_mode"])
        self.assertFalse(builder.call_args.kwargs["enable_inst_interactivity"])
        self.assertEqual(move.call_args.args[1], torch.device("cuda", 3))
        self.assertEqual([name for name, value in model.named_parameters() if value.requires_grad], ["backbone.language_backbone.delta"])
        self.assertTrue(all(not module.training for module in model.modules()))


if __name__ == "__main__":
    unittest.main()

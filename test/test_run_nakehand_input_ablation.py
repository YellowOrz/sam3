"""CPU-only queued scheduling, PID ownership, finite wait and result gates."""
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from scripts import run_nakehand_input_ablation as queue

NOW = datetime.fromisoformat("2026-09-10T16:10:00+00:00")


def arguments(root):
    return Namespace(data_root=root / "data", base_checkpoint=root / "base.pt", initial_cache=root / "cache.pt",
        old_unconstrained=root / "old0.pt", wait_for_run=root / "prior", output_dir=root / "output",
        project_root=root / "code", python=sys.executable, gpus=[0, 2], deadline=queue.LATEST_DEADLINE,
        queue_wait_seconds=10800, gpu_wait_seconds=1800)


def prepare(root):
    args = arguments(root)
    for directory in (args.output_dir, args.output_dir / "logs", args.wait_for_run, args.project_root):
        directory.mkdir(parents=True, exist_ok=True)
    return args


def save_state(root, status="complete", commands=None):
    (root / "state.json").write_text(json.dumps({"format": queue.PREDECESSOR_FORMAT,
        "status": status, "commands": {} if commands is None else commands}))


class FakeClock:
    def __init__(self):
        self.elapsed = 0.

    def monotonic(self):
        return self.elapsed

    def now(self):
        return NOW + timedelta(seconds=self.elapsed)

    def sleep(self, seconds):
        self.elapsed += seconds


class InputQueueTests(unittest.TestCase):
    def test_commands_package_frozen_lr_budget_and_execution_only_probe(self):
        runner = queue.InputRunner(arguments(Path("/tmp/input-queue")), now=lambda: NOW)
        train = runner.training_command(Path("/tmp/output"), .0003, 20)
        self.assertEqual(train[2:4], ["-m", "scripts.train_nakehand_input_ve"])
        for flag, value in (("--learning-rate", "0.0003"), ("--seed", "123"),
                            ("--checkpoint-every", "100"), ("--max-steps", "20"), ("--gpu-memory-fraction", "0.35")):
            self.assertEqual(train[train.index(flag) + 1], value)
        continuation = runner.training_command(Path("/tmp/new"), .0003, 2000, Path("/tmp/step20.pt"))
        self.assertEqual(continuation[-2:], ["--resume", "/tmp/step20.pt"])
        probe = runner.evaluation_command(Path("/tmp/probe"), Path("/tmp/step20.pt"), full=False)
        self.assertEqual(probe[probe.index("--indices") + 1], "0,18,20,66")
        self.assertNotIn("--output-delta-checkpoint", probe)
        full = runner.evaluation_command(Path("/tmp/full"), Path("/tmp/step2000.pt"), full=True)
        self.assertNotIn("--indices", full)
        self.assertIn("--output-delta-checkpoint", full)

    def test_predecessor_terminal_with_running_row_does_not_unlock(self):
        state = {"format": queue.PREDECESSOR_FORMAT, "status": "complete", "commands": {
            "train": {"status": "running", "command": ["python"]}, "result": {"checkpoint": "x"}}}
        self.assertEqual(queue.predecessor_status(state), (False, ["train"]))
        state["commands"]["train"]["status"] = "completed"
        self.assertEqual(queue.predecessor_status(state), (True, []))
        state["status"] = "stopping_after_failure"
        self.assertFalse(queue.predecessor_status(state)[0])
        state["status"] = "failed_or_stopped"
        self.assertTrue(queue.predecessor_status(state)[0])
        with self.assertRaises(ValueError):
            queue.predecessor_status({"format": "unknown", "commands": {}})

    def test_live_predecessor_process_blocks_then_its_exit_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            save_state(args.wait_for_run)
            clock = FakeClock()
            probe = lambda _path: [{"pid": 987, "command": ["prior"]}] if clock.elapsed < 30 else []
            runner = queue.InputRunner(args, now=clock.now, sleep=clock.sleep, monotonic=clock.monotonic, process_probe=probe)
            with patch.object(runner, "verify_sources") as verify:
                runner.wait_for_predecessor()
            self.assertEqual(clock.elapsed, 30)
            self.assertEqual(runner.state["status"], "predecessor_finished")
            self.assertEqual(runner.state["predecessor"]["matching_live_child_processes"], [])
            verify.assert_called_once()

    def test_predecessor_failure_is_not_retried_and_changed_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            save_state(args.wait_for_run, "failed_or_stopped")
            runner = queue.InputRunner(args, now=lambda: NOW, process_probe=lambda _: [])
            with patch.object(runner, "verify_sources", side_effect=RuntimeError("changed source")), patch.object(runner, "child") as child:
                with self.assertRaisesRegex(RuntimeError, "changed source"):
                    runner.wait_for_predecessor()
                child.assert_not_called()
            with patch.object(runner, "verify_sources"):
                runner.wait_for_predecessor()
            self.assertEqual(runner.state["predecessor"]["status"], "failed_or_stopped")

    def test_queue_wait_has_finite_upper_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            args.queue_wait_seconds = 61
            save_state(args.wait_for_run, "running")
            clock = FakeClock()
            runner = queue.InputRunner(args, now=clock.now, sleep=clock.sleep, monotonic=clock.monotonic)
            with self.assertRaises(TimeoutError):
                runner.wait_for_predecessor()
            self.assertEqual(clock.elapsed, 61)

    def test_gpu_threshold_12000_and_thirty_minute_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            clock = FakeClock()
            calls = []
            def query(command, **_kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "11999\n")
            runner = queue.InputRunner(args, now=clock.now, sleep=clock.sleep, monotonic=clock.monotonic, run=query)
            with self.assertRaises(TimeoutError):
                runner.wait_for_gpu(2)
            self.assertEqual(clock.elapsed, 1800)
            self.assertTrue(all(command[1] == "--id=2" for command in calls))
            runner.run = lambda command, **_: subprocess.CompletedProcess(command, 0, "12000\n")
            runner.wait_for_gpu(0)

    def test_peer_failure_cancels_waiters_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            runner = queue.InputRunner(args, now=lambda: NOW)
            waiting = threading.Event()
            attempts = []
            def waiter():
                waiting.set()
                runner.stop_event.wait(timeout=2)
                runner.check_stopping()
            def fail():
                self.assertTrue(waiting.wait(timeout=2))
                attempts.append(1)
                raise RuntimeError("peer failed")
            with self.assertRaises(RuntimeError):
                runner.paired([("waiter", waiter, ()), ("fail", fail, ())])
            self.assertEqual(attempts, [1])
            self.assertEqual(runner.state["status"], "stopping_after_failure")

    def test_actual_cpu_child_records_pid_and_completes_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            runner = queue.InputRunner(args, now=lambda: NOW)
            command = [sys.executable, "-c", "print('cpu-only')", "--output-dir", str(args.output_dir / "child")]
            with patch.object(runner, "verify_sources"), patch.object(runner, "wait_for_gpu") as gpu:
                runner.child("cpu", command, gpu=0, cap=10, needs_gpu=False)
            row = runner.state["commands"]["cpu"]
            self.assertGreater(row["pid"], 0)
            self.assertEqual(row["status"], "completed")
            self.assertTrue(row["process_exited"])
            self.assertEqual(row["command"], command)
            gpu.assert_not_called()

    def test_timeout_signals_only_own_new_cpu_child_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            runner = queue.InputRunner(args, now=lambda: NOW)
            command = [sys.executable, "-c", "import time; time.sleep(20)", "--output-dir", str(args.output_dir / "child")]
            with patch.object(runner, "verify_sources"):
                with self.assertRaises(TimeoutError):
                    runner.child("timeout", command, gpu=0, cap=.1, needs_gpu=False)
            row = runner.state["commands"]["timeout"]
            self.assertEqual(row["status"], "timed_out")
            self.assertTrue(row["process_exited"])
            self.assertEqual(row["process_group_id"], row["pid"])
            self.assertIn("own", row["termination"])

    def test_active_peer_completes_its_finite_stage_after_other_lane_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            runner = queue.InputRunner(args, now=lambda: NOW)
            running = threading.Event()
            class Process:
                pid = 12345
                returncode = None
                def poll(self):
                    return self.returncode
                def wait(self, timeout):
                    running.set()
                    self_test.assertTrue(runner.stop_event.wait(timeout=2))
                    self.returncode = 0
                    return 0
            self_test = self
            runner.popen = lambda *a, **k: Process()
            def active():
                runner.child("active", ["python", "--output-dir", str(args.output_dir / "active")], gpu=0, cap=600, needs_gpu=False)
            def fail():
                self.assertTrue(running.wait(timeout=2))
                raise ValueError("fail peer")
            with patch.object(runner, "verify_sources"), patch.object(runner, "_stop_own_child") as stop:
                with self.assertRaises(ValueError):
                    runner.paired([("active", active, ()), ("fail", fail, ())])
                stop.assert_not_called()
            self.assertEqual(runner.state["commands"]["active"]["status"], "completed")

    def test_smoke_failure_stops_before_probe_or_continuation(self):
        runner = queue.InputRunner(arguments(Path("/tmp/input-queue")), now=lambda: NOW)
        with patch.object(runner, "child") as child, patch.object(queue, "checkpoint_result", side_effect=ValueError("bad smoke")):
            with self.assertRaises(ValueError):
                runner.train_one("lr1e-3", .001, 0)
            self.assertEqual(child.call_count, 1)

    def test_small_probe_receipt_is_not_published_as_a_full_verified_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = prepare(Path(tmp))
            runner = queue.InputRunner(args, now=lambda: NOW)
            final = {"progress": {"samples_seen": 2000}}
            with patch.object(runner, "child"), \
                 patch.object(queue, "checkpoint_result", side_effect=[(Path("/smoke.pt"), {}), (Path("/final.pt"), final)]), \
                 patch.object(queue, "validate_evaluation_result", return_value={"actual_images_per_variant": 4}), \
                 patch.object(queue.runner, "file_hash", return_value="a" * 64):
                runner.train_one("lr1e-3", .001, 0)
            probe = runner.state["commands"]["lr1e-3-probe"]
            self.assertIn("verified_execution_probe", probe)
            self.assertNotIn("verified_result", probe)
            self.assertEqual(runner.state["verified_reports"], {})

    def test_compare_lr_only_and_exact_initial_and_order_contract(self):
        first = {"training_config": {"learning_rate": .001, "loss_weights": {"mask": 1.}},
                 "next_step": 2000, "progress": {"samples_seen": 2000}, "observed_image_ids": [1, 2],
                 "planned_dataset_indices": [2, 1], "initial_input_residual_state": {"delta": torch.zeros(2)},
                 "initial_cache_state_dict": {"base": torch.ones(2)}}
        second = deepcopy(first)
        second["training_config"]["learning_rate"] = .0003
        self.assertEqual(queue.compare_input_trials(first, second)["actual_steps_each"], 2000)
        for mutate in (lambda s: s["training_config"]["loss_weights"].update(mask=4.),
                       lambda s: s["observed_image_ids"].reverse(),
                       lambda s: s["initial_input_residual_state"]["delta"].add_(1)):
            changed = deepcopy(second)
            mutate(changed)
            with self.assertRaises(ValueError):
                queue.compare_input_trials(first, changed)

    def test_incomplete_full_evaluation_header_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(json.dumps({"format": queue.evaluation.FORMAT, "status": "completed",
                "evaluated_images": 3449, "full_val_evaluated": True, "all_sources_unchanged": True,
                "completed_images_per_model": {name: 3449 for name in queue.evaluation.LABELS}}))
            with self.assertRaises(ValueError):
                queue.validate_evaluation_result(root, full=True)

    def test_cli_enforces_fixed_cards_cutoff_and_wait_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prior").mkdir()
            flags = ["--data-root", str(root / "data"), "--base-checkpoint", str(root / "base.pt"),
                "--initial-cache", str(root / "cache.pt"), "--old-unconstrained", str(root / "old.pt"),
                "--wait-for-run", str(root / "prior"), "--output-dir", str(root / "new"),
                "--deadline", "2026-09-11T08:50:00+08:00"]
            with patch.object(queue, "datetime", wraps=datetime) as date:
                date.now.return_value = NOW
                self.assertEqual(queue.parse_args(flags).gpus, [0, 2])
                for extra in (["--gpus", "0", "3"], ["--gpus", "0", "0"], ["--queue-wait-seconds", "10801"],
                              ["--gpu-wait-seconds", "1801"], ["--deadline", "2026-09-11T08:51:00+08:00"]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        queue.parse_args(flags + extra)

    def test_latest_user_review_preserves_simultaneous_missing_label_and_wrong_side(self):
        runner = queue.InputRunner(arguments(Path("/tmp/input-queue")), now=lambda: NOW)
        review = runner.state["latest_user_review"]
        self.assertEqual(review["human_side"], "right_hand")
        self.assertEqual(review["inspected_output_delta_prompt"], "left_hand")
        self.assertFalse(review["labels_modified"])


if __name__ == "__main__":
    unittest.main()

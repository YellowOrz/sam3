"""CPU-only queue, budget, completion and failure gates."""
from argparse import Namespace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_nakehand_boundary_ablation as queue


def arguments(root):
    return Namespace(data_root=root / "data", base_checkpoint=root / "base.pt", initial_cache=root / "cache.pt",
                     old_unconstrained=root / "old0.pt", old_anchored=root / "old1.pt", output_dir=root / "output",
                     project_root=root / "code", python="python", gpus=[0, 2],
                     deadline=datetime.now(timezone.utc) + timedelta(hours=8))


class BoundaryQueueTest(unittest.TestCase):
    def test_commands_use_package_entry_and_frozen_budget(self):
        args = arguments(Path("/tmp/queue-test"))
        runner = queue.BoundaryRunner(args)
        command = runner.training_command(args.output_dir / "probe", 4., 20)
        self.assertEqual(command[2:4], ["-m", "scripts.train_nakehand_prompt_ablation"])
        self.assertEqual(command[command.index("--max-steps") + 1], "20")
        self.assertEqual(command[command.index("--gpu-memory-fraction") + 1], "0.25")
        self.assertNotIn("--resume", command)
        resumed = runner.training_command(args.output_dir / "full", 4., 2000, Path("/tmp/probe.pt"))
        self.assertEqual(resumed[-2:], ["--resume", "/tmp/probe.pt"])

    def test_training_continuation_requires_actual_twenty_step_gate(self):
        args = arguments(Path("/tmp/queue-test"))
        runner = queue.BoundaryRunner(args)
        with patch.object(runner, "child") as child, patch.object(queue, "checkpoint_result", side_effect=ValueError("bad checkpoint")):
            with self.assertRaises(ValueError):
                runner.train_one("weight0", 0., 0)
            self.assertEqual(child.call_count, 1)

    def test_comparison_rejects_hidden_hyperparameter_or_training_order_changes(self):
        first = {"training_config": {"boundary_weight": 0., "learning_rate": .001},
                 "progress": {"completed_steps": 2000}, "next_step": 2000, "observed_image_ids": [1, 2]}
        second = deepcopy(first)
        second["training_config"]["boundary_weight"] = 4.
        with patch.object(queue.diagnostic, "compare_checkpoint_inputs"):
            self.assertEqual(queue.compare_boundary_trials(first, second)["actual_steps_each"], 2000)
            second["training_config"]["learning_rate"] = .01
            with self.assertRaises(ValueError):
                queue.compare_boundary_trials(first, second)
            second["training_config"]["learning_rate"] = .001
            second["observed_image_ids"] = [2, 1]
            with self.assertRaises(ValueError):
                queue.compare_boundary_trials(first, second)

    def test_gpu_wait_queries_only_selected_card_and_failure_stops_unstarted_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            args.output_dir.mkdir()
            runner = queue.BoundaryRunner(args, run=lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "22000\n", ""))
            runner.wait_for_gpu(2)
            runner.stop_event.set()
            with self.assertRaises(queue.runner.CancelledError):
                runner.wait_for_gpu(2)

    def test_deadline_cap_and_insufficient_remaining_time(self):
        self.assertEqual(queue.runner.stage_timeout(300, 600), 270)
        self.assertEqual(queue.runner.stage_timeout(5000, 600), 600)
        with self.assertRaises(TimeoutError):
            queue.runner.stage_timeout(119, 600)

    def test_complete_header_without_actual_records_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(json.dumps({"status": "completed", "full_val_evaluated": True,
                "evaluated_images": 3449, "all_sources_unchanged": True,
                "completed_images_per_model": {label: 3449 for label in queue.diagnostic.LABELS},
                "record_files": {}}))
            with self.assertRaises(KeyError):
                queue.validate_full_spatial_result(root)

    def test_paired_failure_sets_cancellation_and_does_not_retry(self):
        runner = queue.BoundaryRunner(arguments(Path("/tmp/queue-test")))
        attempts = []
        def fail():
            attempts.append(1)
            raise ValueError("intentional failure")
        with self.assertRaises(ValueError):
            runner.paired([("failure", fail, ())])
        self.assertEqual(attempts, [1])
        self.assertTrue(runner.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()

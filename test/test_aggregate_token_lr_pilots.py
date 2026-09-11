from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.aggregate_token_lr_pilots import (
    DEADLINE, SHARED_SCRIPTS, aggregate, completed_inputs, parse_args, validate_completed_runs,
)


class AggregateTokenLrPilotsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.run_dirs = []
        for rate in (0.01, 0.003, 0.001):
            run_dir = self.root / f"run-{rate}"
            (run_dir / "snapshots").mkdir(parents=True)
            artifacts = {}
            for name in SHARED_SCRIPTS:
                path = run_dir / "snapshots" / name
                path.write_bytes(b"# same source\n")
                artifacts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            checkpoint, summary = run_dir / "token.pt", run_dir / "summary.json"
            checkpoint.write_bytes(b"tokens")
            summary.write_text("{}")
            observed = {"image_ids_sha256": "observed", "start_step": 0, "end_step": 2000,
                        "samples": 2000, "queries": 4000}
            state = {
                "format": "sam3-token-lr-pilot-v1", "status": "complete",
                "comparison_valid": True,
                "core_python_sources": {"sam3/core.py": hashlib.sha256(b"source").hexdigest()},
                "fixed_training": {"learning_rates": [rate], "steps": 2000, "samples": 2000, "batch_size": 1},
                "train_annotations_sha256": "train", "val_annotations_sha256": "val",
                "base_checkpoint_sha256": "base", "epoch_order_sha256": "order",
                "sample_prefix_sha256": "prefix", "formal_initial_class_tokens_sha256": "initial",
                "expected_observed_image_ids_sha256": "observed", "dataset_size": 23265, "val_images": 2909,
                "config": {"data_root": "/data/train", "val_root": "/data/val", "base_checkpoint": "/base.pt"},
                "artifacts": artifacts,
                "trials": [{
                    "learning_rate": rate, "status": "evaluated", "actual_samples": 2000,
                    "actual_steps": 2000, "completed_full_epochs": 0,
                    "initial_class_tokens_sha256": "initial", "epoch_order_sha256": "order",
                    "sample_prefix_sha256": "prefix", "observed_identity": observed,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "evaluation_summary": str(summary), "evaluation_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                }],
            }
            state["core_python_sources_sha256"] = hashlib.sha256(json.dumps(
                state["core_python_sources"], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            (run_dir / "state.json").write_text(json.dumps(state))
            self.run_dirs.append(run_dir)
        self.current = DEADLINE - timedelta(hours=1)
        self.sleeps = []
        self.calls = []
        self.on_sleep = None
        self.exit_code = 0

    def update(self, index, change):
        path = self.run_dirs[index] / "state.json"
        state = json.loads(path.read_text())
        change(state)
        path.write_text(json.dumps(state))

    def args(self):
        argv = ["--output-dir", str(self.root / "aggregate"), "--project-root", str(self.project)]
        for path in self.run_dirs:
            argv.extend(["--run-dir", str(path)])
        return parse_args(argv)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)
        if self.on_sleep:
            self.on_sleep()

    def runner(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.exit_code == 0:
            Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=self.exit_code)

    def execute(self):
        return aggregate(self.args(), now=lambda: self.current, sleep=self.sleep, run=self.runner)

    def test_combines_all_three_only_after_verified_success_on_cpu(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.calls), 1)
        command, kwargs = self.calls[0]
        self.assertEqual(command.count("--summary"), 3)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        state = json.loads((self.root / "aggregate/state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["returncode"], 0)
        self.assertEqual(len(state["source_states"]), 3)

    def test_waits_for_last_complete_run_and_keeps_unchanged_logs_quiet(self):
        self.update(2, lambda state: state.update(status="running"))
        self.on_sleep = lambda: self.update(2, lambda state: state.update(status="complete")) if len(self.sleeps) == 2 else None
        self.assertEqual(self.execute(), 0)
        log = (self.root / "aggregate/aggregate.log").read_text()
        self.assertEqual(log.count("waiting_runs"), 1)
        self.assertTrue(all(seconds <= 30 for seconds in self.sleeps))

    def test_failed_or_stopped_run_never_counts_as_success(self):
        self.update(0, lambda state: state.update(status="failed", detail="OOM"))
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, [])

    def test_deadline_stops_wait_without_calibrating_partial_set(self):
        self.current = DEADLINE - timedelta(seconds=65)
        self.update(1, lambda state: state.update(status="running"))
        self.assertEqual(self.execute(), 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.sleeps, [30, 30, 5])

    def test_different_initialization_or_sample_budget_is_rejected(self):
        self.update(1, lambda state: state.update(formal_initial_class_tokens_sha256="other"))
        loaded, _ = completed_inputs(self.run_dirs)
        with self.assertRaisesRegex(ValueError, "initial"):
            validate_completed_runs(loaded)

    def test_duplicate_learning_rate_does_not_replace_missing_control(self):
        def duplicate(state):
            state["fixed_training"]["learning_rates"] = [0.003]
            state["trials"][0]["learning_rate"] = 0.003
        self.update(2, duplicate)
        loaded, _ = completed_inputs(self.run_dirs)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_completed_runs(loaded)

    def test_changed_checkpoint_is_rejected(self):
        (self.run_dirs[0] / "token.pt").write_bytes(b"changed")
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, [])

    def test_different_core_source_maps_cannot_be_aggregated(self):
        def change_core(state):
            state["core_python_sources"]["sam3/added.py"] = hashlib.sha256(b"extra").hexdigest()
            state["core_python_sources_sha256"] = hashlib.sha256(json.dumps(
                state["core_python_sources"], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
        self.update(1, change_core)
        loaded, _ = completed_inputs(self.run_dirs)
        with self.assertRaisesRegex(ValueError, "core_python_sources"):
            validate_completed_runs(loaded)

    def test_calibration_failure_and_existing_output_are_explicit(self):
        self.exit_code = 7
        self.assertEqual(self.execute(), 1)
        state = json.loads((self.root / "aggregate/state.json").read_text())
        self.assertEqual(state["returncode"], 7)
        self.assertEqual(state["status"], "failed")
        with self.assertRaises(FileExistsError):
            self.execute()


if __name__ == "__main__":
    unittest.main()

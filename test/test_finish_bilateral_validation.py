import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from scripts.finish_bilateral_validation import (
    AUTHORIZED_DEADLINE, LIMITATION, Supervisor, parse_args,
)


class FakeClock:
    def __init__(self, seconds_remaining=10800):
        self.current = AUTHORIZED_DEADLINE - timedelta(seconds=seconds_remaining)
        self.sleeps = []
        self.on_sleep = None

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)
        if self.on_sleep:
            self.on_sleep()


class FinishBilateralValidationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project = self.root / "project"
        (self.project / "scripts").mkdir(parents=True)
        for name in ("report_bilateral_evaluation.py", "evaluate_bilateral_tokens.py", "render_separated_masks.py"):
            (self.project / "scripts" / name).write_text("# frozen test script\n")
        self.data = self.root / "val"
        self.data.mkdir()
        (self.data / "annotations.json").write_text("{}")
        self.base = self.root / "base.pt"
        self.base.write_bytes(b"base")
        self.token = self.root / "epoch2_complete.pt"
        self.token.write_bytes(b"tokens")
        self.epoch1 = self.root / "epoch1_summary.json"
        self.reference = {
            "data_root": str(self.data), "base_checkpoint": str(self.base),
            "annotations_sha256": hashlib.sha256(b"{}").hexdigest(),
            "evaluated_dataset_indices": [0, 1], "evaluated_images": 2,
            "detection_threshold": 0.5, "mask_threshold": 0.5,
            "confidence_definition": "class * presence",
            "models": {"epoch1": {}, "ve-natural": {}, "ve-underscore": {}},
            "metrics": {"epoch1": {}, "ve-natural": {}, "ve-underscore": {}},
        }
        self.write_epoch1()
        self.argv = [
            "--epoch1-summary", str(self.epoch1), "--epoch2-checkpoint", str(self.token),
            "--output-dir", str(self.root / "run"), "--project-root", str(self.project),
            "--data-root", str(self.data), "--base-checkpoint", str(self.base),
            "--unified-root", str(self.root / "unified"),
        ]
        self.clock = FakeClock()
        self.commands = []
        self.gpu_values = []
        self.eval_failure = None

    def write_epoch1(self):
        self.epoch1.write_text(json.dumps(self.reference))

    def runner(self, command, **kwargs):
        self.commands.append((list(command), kwargs))
        if command[0] == "nvidia-smi":
            free = self.gpu_values.pop(0) if self.gpu_values else 9000
            return SimpleNamespace(returncode=0, stdout=f"{free}\n", stderr="")
        if "--learned-checkpoint" in command:
            if self.eval_failure == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            if self.eval_failure:
                return SimpleNamespace(returncode=self.eval_failure)
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            summary = copy.deepcopy(self.reference)
            summary["models"] = {"epoch2": {}}
            summary["metrics"] = {"epoch2": {}}
            (output / "summary.json").write_text(json.dumps(summary))
        else:
            output = Path(command[command.index("--output") + 1])
            output.write_text("# Report\n")
        return SimpleNamespace(returncode=0)

    def supervisor(self, extra=()):
        return Supervisor(parse_args(self.argv + list(extra)), now=self.clock.now,
                          sleep=self.clock.sleep, run=self.runner)

    def test_success_runs_epoch2_once_without_ve_and_freezes_inputs(self):
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 0)
        self.assertEqual(supervisor.state["status"], "complete")
        self.assertEqual(supervisor.state["epoch2_attempts"], 1)
        evaluations = [(command, kwargs) for command, kwargs in self.commands if "--learned-checkpoint" in command]
        self.assertEqual(len(evaluations), 1)
        command, kwargs = evaluations[0]
        self.assertNotIn("--include-ve", command)
        self.assertIn("--amp", command)
        self.assertEqual(command[command.index("--visual-style") + 1], "separate")
        self.assertEqual(command[command.index("--indices") + 1], "0,1")
        self.assertEqual(command[command.index("--gpu-memory-fraction") + 1], "0.25")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertLessEqual(kwargs["timeout"], 7200)
        self.assertTrue((self.root / "run/snapshots/render_separated_masks.py").exists())
        self.assertIn(LIMITATION, (self.root / "run/reports/combined_report.md").read_text())
        self.assertTrue(all(row["returncode"] == 0 for row in supervisor.state["commands"]))

    def test_waits_for_summary_checkpoint_and_memory_without_repeated_logs(self):
        self.epoch1.unlink()
        self.token.unlink()
        self.gpu_values = [7000, 7200, 9000, 9000]

        def publish_inputs():
            if len(self.clock.sleeps) == 2:
                self.write_epoch1()
            if len(self.clock.sleeps) == 4:
                self.token.write_bytes(b"tokens")

        self.clock.on_sleep = publish_inputs
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 0)
        log = (self.root / "run/supervisor.log").read_text()
        self.assertEqual(log.count("waiting_epoch1_summary"), 1)
        self.assertEqual(log.count("waiting_epoch2_checkpoint"), 1)
        self.assertEqual(log.count("waiting_gpu_memory"), 1)
        self.assertTrue(all(0 < seconds <= 30 for seconds in self.clock.sleeps))

    def test_deadline_stops_wait_without_running_any_subprocess(self):
        self.epoch1.unlink()
        self.clock = FakeClock(65)
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 2)
        self.assertEqual(supervisor.state["status"], "stopped_deadline")
        self.assertEqual(self.commands, [])
        self.assertEqual(self.clock.sleeps, [30, 30, 5])

    def test_near_deadline_makes_epoch1_report_but_does_not_launch_epoch2(self):
        self.clock = FakeClock(1000)
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 2)
        self.assertEqual(supervisor.state["status"], "stopped_insufficient_time")
        self.assertEqual(supervisor.state["epoch2_attempts"], 0)
        self.assertEqual(len(self.commands), 1)

    def test_timeout_is_bounded_by_remaining_time_and_stops_without_retry(self):
        self.clock = FakeClock(2000)
        self.eval_failure = "timeout"
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 1)
        record = supervisor.state["commands"][-1]
        self.assertEqual(record["label"], "epoch2_evaluation")
        self.assertEqual(record["timeout_seconds"], 1940)
        self.assertTrue(record["timed_out"])
        self.assertEqual(supervisor.state["status"], "failed")
        self.assertEqual(supervisor.state["epoch2_attempts"], 1)

    def test_child_failure_records_exit_code_and_does_not_retry(self):
        self.eval_failure = 7
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 1)
        self.assertEqual(supervisor.state["commands"][-1]["returncode"], 7)
        self.assertEqual(supervisor.state["epoch2_attempts"], 1)
        self.assertFalse((self.root / "run/reports/combined_report.md").exists())

    def test_annotation_change_prevents_any_epoch2_evaluation(self):
        (self.data / "annotations.json").write_text('{"changed": true}')
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 1)
        self.assertIn("annotations changed", supervisor.state["detail"])
        self.assertEqual(supervisor.state["epoch2_attempts"], 0)

    def test_time_spent_waiting_does_not_allow_a_late_start(self):
        self.clock = FakeClock(1810)
        self.gpu_values = [7000]
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 2)
        self.assertEqual(supervisor.state["epoch2_attempts"], 0)
        self.assertEqual(supervisor.state["status"], "stopped_insufficient_time")

    def test_existing_run_is_never_restarted(self):
        supervisor = self.supervisor()
        self.assertEqual(supervisor.execute(), 0)
        with self.assertRaises(FileExistsError):
            self.supervisor().execute()

    def test_deadline_cannot_extend_authorized_budget(self):
        with self.assertRaises(SystemExit):
            parse_args(self.argv + ["--deadline", "2026-09-11T21:40:00+08:00"])

    def test_output_cannot_be_inside_git_project(self):
        with self.assertRaises(SystemExit):
            parse_args(self.argv + ["--output-dir", str(self.project / "results")])


if __name__ == "__main__":
    unittest.main()

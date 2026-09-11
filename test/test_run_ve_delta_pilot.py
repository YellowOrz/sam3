"""CPU checks for bounded semantic pilot supervision; no child GPU launch."""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.run_ve_delta_pilot import DeltaPilot, CHECKPOINT_FORMAT, parse_args, verify_completed_pilot


class VEDeltaPilotRunnerTest(unittest.TestCase):
    def args(self, output):
        return parse_args(["--resume", "/tmp/resume20.pt", "--initial-cache", "/tmp/cache.pt",
                           "--base-checkpoint", "/tmp/base.pt", "--output-dir", str(output)])

    def test_complete_gate_rejects_partial_pilot_before_evaluation(self):
        progress = {"completed_steps": 2000, "samples_seen": 2000, "planned_steps": 2000,
                    "planned_samples": 2000, "pilot_complete": True}
        summary = {"format": CHECKPOINT_FORMAT, "progress": progress}
        checkpoint = {**summary, "next_step": 2000, "observed_image_ids": list(range(2000))}
        verify_completed_pilot(summary, checkpoint)
        with self.assertRaises(ValueError):
            verify_completed_pilot(summary, dict(checkpoint, next_step=20))
        with self.assertRaises(ValueError):
            verify_completed_pilot(summary, dict(checkpoint, progress={**progress, "pilot_complete": False}))

    def test_cannot_extend_deadline_or_reduce_memory_margin(self):
        base = ["--resume", "/tmp/r.pt", "--initial-cache", "/tmp/c.pt",
                "--base-checkpoint", "/tmp/b.pt", "--output-dir", "/tmp/semantic-run-new-test"]
        for extra in (["--deadline", "2026-09-10T21:41:00+08:00"], ["--minimum-free-mib", "7499"]):
            with self.assertRaises(SystemExit):
                parse_args(base + extra)

    def test_stage_uses_cap_reserve_and_never_retries_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "output")
            runner = DeltaPilot(args, now=lambda: args.deadline - timedelta(seconds=1000))
            runner.check_frozen_sources = Mock()
            runner.gpu_free = Mock(return_value=9000)
            runner.child = Mock(side_effect=RuntimeError("GPU child failed"))
            with self.assertRaisesRegex(RuntimeError, "child failed"):
                runner.stage("train", ["python", "train.py"], 2700, needs_gpu=True)
            runner.child.assert_called_once_with("train", ["python", "train.py"], maximum_seconds=2700, reserve_seconds=30)
            self.assertEqual(runner.check_frozen_sources.call_count, 2)

    def test_no_stage_starts_inside_final_120_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "output")
            runner = DeltaPilot(args, now=lambda: args.deadline - timedelta(seconds=119))
            runner.transition = Mock()
            runner.child = Mock()
            runner.gpu_free = Mock()
            self.assertFalse(runner.stage("eval", [], 3600, needs_gpu=True))
            runner.child.assert_not_called()
            runner.gpu_free.assert_not_called()

    def test_low_memory_waits_without_touching_other_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "output")
            runner = DeltaPilot(args, now=lambda: args.deadline - timedelta(seconds=1000))
            runner.transition = Mock()
            runner.pause = Mock()
            runner.gpu_free = Mock(side_effect=[7000, 9000])
            runner.child = Mock()
            runner.check_frozen_sources = Mock()
            self.assertTrue(runner.stage("eval", [], 3600, needs_gpu=True))
            runner.pause.assert_called_once()
            runner.child.assert_called_once()


if __name__ == "__main__":
    unittest.main()

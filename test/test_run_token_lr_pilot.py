from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

from scripts.run_token_lr_pilot import DEADLINE, LOSS_NAMES, Pilot, expected_order, parse_args


class FakeClock:
    def __init__(self, remaining=10800):
        self.current = DEADLINE - timedelta(seconds=remaining)
        self.sleeps = []
        self.callback = None

    def now(self):
        return self.current

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += timedelta(seconds=seconds)
        if self.callback:
            self.callback()


class TokenLrPilotTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.project = self.root / "project"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "sam3").mkdir()
        (self.project / "sam3" / "core.py").write_text("# stable model implementation\n")
        for name in ("train_learnable_tokens.py", "evaluate_bilateral_tokens.py",
                     "render_separated_masks.py", "calibrate_bilateral_thresholds.py"):
            (self.project / "scripts" / name).write_text("# test snapshot\n")
        self.data = self.root / "train"
        self.val = self.root / "val"
        for path, count in ((self.data, 2001), (self.val, 6)):
            path.mkdir()
            (path / "annotations.json").write_text(json.dumps({
                "info": {"split": path.name}, "images": [{"id": i} for i in range(count)]
            }))
        self.annotation_hash = hashlib.sha256((self.data / "annotations.json").read_bytes()).hexdigest()
        self.val_hash = hashlib.sha256((self.val / "annotations.json").read_bytes()).hexdigest()
        self.base = self.root / "base.pt"
        self.base.write_bytes(b"frozen base")
        self.formal_summary = self.root / "formal_summary.json"
        self.formal_checkpoint = self.root / "formal_complete.pt"
        self.write_formal()
        self.argv = [
            "--formal-summary", str(self.formal_summary), "--formal-checkpoint", str(self.formal_checkpoint),
            "--data-root", str(self.data), "--val-root", str(self.val),
            "--base-checkpoint", str(self.base), "--project-root", str(self.project),
            "--output-dir", str(self.root / "pilot"), "--unified-root", str(self.root / "unified"),
        ]
        self.clock = FakeClock()
        self.commands = []
        self.gpu_free = []
        self.failure = None
        self.initial_mismatch = False
        self.observed_mismatch = False
        self.change_core = False

    def checkpoint(self, lr=0.01, steps=2002, batch_size=2):
        config = {
            "tokens_per_class": 4, "batch_size": batch_size, "amp": True, "epochs": 2,
            "seed": 123, "learning_rate": lr, "data_root": str(self.data), "base_checkpoint": str(self.base),
            **{f"{name}_weight": 1.0 for name in LOSS_NAMES},
        }
        return {
            "format": "sam3-learnable-class-tokens-v2", "class_names": ["left_hand", "right_hand"],
            "training_config": config, "next_step": steps, "epoch_order": expected_order(2001),
            "annotation_summary": {"sha256": self.annotation_hash},
            "class_tokens": torch.ones(2, 4, 256), "initial_class_tokens": torch.zeros(2, 4, 256),
        }

    def write_formal(self):
        self.formal_summary.write_text(json.dumps({
            "epochs": 2, "steps": 2002, "samples_seen": 4002, "dataset_size": 2001,
            "batch_size": 2, "amp": True, "annotation_summary": {"sha256": self.annotation_hash},
        }))
        torch.save(self.checkpoint(), self.formal_checkpoint)

    def runner(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if command[0] == "nvidia-smi":
            free = self.gpu_free.pop(0) if self.gpu_free else 20000
            return SimpleNamespace(returncode=0, stdout=f"{free}\n", stderr="")
        if "--learning-rate" in command:
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir(parents=True)
            lr = float(command[command.index("--learning-rate") + 1])
            steps = int(command[command.index("--max-steps") + 1])
            batch_size = int(command[command.index("--batch-size") + 1])
            if self.failure:
                torch.save(self.checkpoint(lr, 100, batch_size), destination / "latest.pt")
                if self.failure == "timeout":
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                return SimpleNamespace(returncode=self.failure)
            state = self.checkpoint(lr, steps, batch_size)
            if self.initial_mismatch and lr == 0.003:
                state["initial_class_tokens"][0, 0, 0] = 1
            observed = {
                "format": "sam3-training-observed-identities-v1",
                "scope": "current_process_completed_steps_only", "start_step": 0,
                "end_step": steps, "samples": 2000, "queries": 4000,
                "image_ids_sha256": hashlib.sha256("".join(
                    f"{index}\n" for index in expected_order(2001)[:2000]
                ).encode("utf-8")).hexdigest(),
                "image_ids_hash_encoding": "decimal_id_newline_utf8",
            }
            if self.observed_mismatch:
                observed["image_ids_sha256"] = "wrong actual IDs"
            state["observed_identity"] = observed
            checkpoint = destination / f"k4_epoch2_step{steps}_partial.pt"
            torch.save(state, checkpoint)
            (destination / "k4_epoch2_summary.json").write_text(json.dumps({
                "epochs": 2, "steps": steps, "samples_seen": 2000, "dataset_size": 2001,
                "epochs_planned": 2, "epochs_completed": 0, "full_training_complete": False,
                "final_checkpoint": str(checkpoint),
                "observed_identity": observed,
                "initial_class_tokens_sha256": hashlib.sha256(
                    state["initial_class_tokens"].numpy().tobytes()
                ).hexdigest(),
            }))
            if self.change_core:
                (self.project / "sam3" / "core.py").write_text("# changed model implementation\n")
                (self.project / "sam3" / "added.py").write_text("# new imported source\n")
        elif "--learned-checkpoint" in command:
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir(parents=True)
            label = command[command.index("--learned-checkpoint") + 1].split("=", 1)[0]
            (destination / "summary.json").write_text(json.dumps({
                "evaluated_images": 6, "evaluated_dataset_indices": list(range(6)),
                "annotations_sha256": self.val_hash, "models": {label: {}},
            }))
        else:
            Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0)

    def pilot(self, extra=()):
        return Pilot(parse_args(self.argv + list(extra)), now=self.clock.now,
                     sleep=self.clock.sleep, run=self.runner)

    def test_three_fresh_same_prefix_runs_then_full_val_and_calibration(self):
        pilot = self.pilot()
        self.assertEqual(pilot.execute(), 0)
        self.assertEqual(pilot.state["status"], "complete")
        self.assertEqual([trial["learning_rate"] for trial in pilot.state["trials"]], [0.01, 0.003, 0.001])
        self.assertEqual({trial["actual_samples"] for trial in pilot.state["trials"]}, {2000})
        self.assertEqual({trial["completed_full_epochs"] for trial in pilot.state["trials"]}, {0})
        self.assertEqual(len({trial["initial_class_tokens_sha256"] for trial in pilot.state["trials"]}), 1)
        self.assertEqual(len({trial["sample_prefix_sha256"] for trial in pilot.state["trials"]}), 1)
        train_commands = [cmd for cmd, _ in self.commands if "--learning-rate" in cmd]
        eval_commands = [cmd for cmd, _ in self.commands if "--learned-checkpoint" in cmd]
        self.assertEqual(len(train_commands), 3)
        self.assertEqual(len(eval_commands), 3)
        for command in train_commands:
            self.assertNotIn("--resume", command)
            self.assertEqual(command[command.index("--gpu-memory-fraction") + 1], "0.35")
            self.assertEqual(command[command.index("--expected-initial-token-sha256") + 1],
                             pilot.state["formal_initial_class_tokens_sha256"])
        for command in eval_commands:
            self.assertNotIn("--include-ve", command)
            self.assertEqual(command[command.index("--samples-per-group") + 1], "0")
        self.assertEqual(len(pilot.state["commands"]), 7)
        for record in pilot.state["commands"]:
            self.assertEqual(record["environment"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(record["environment"]["OMP_NUM_THREADS"], "2")
            self.assertEqual(record["returncode"], 0)
        self.assertIn("partial.pt", pilot.state["trials"][0]["checkpoint"])

    def test_waits_for_actual_formal_completion_without_repeated_status_log(self):
        self.formal_summary.unlink()
        self.clock.callback = lambda: self.write_formal() if len(self.clock.sleeps) == 2 else None
        self.assertEqual(self.pilot().execute(), 0)
        log = (self.root / "pilot/pilot.log").read_text()
        self.assertEqual(log.count("waiting_formal_training"), 1)
        self.assertTrue(all(seconds <= 30 for seconds in self.clock.sleeps))

    def test_partial_formal_summary_is_rejected_despite_complete_filename(self):
        summary = json.loads(self.formal_summary.read_text())
        summary["steps"] = 1000
        self.formal_summary.write_text(json.dumps(summary))
        pilot = self.pilot()
        self.assertEqual(pilot.execute(), 1)
        self.assertIn("two complete epochs", pilot.state["detail"])
        self.assertEqual(self.commands, [])

    def test_initial_mismatch_stops_before_second_evaluation(self):
        self.initial_mismatch = True
        pilot = self.pilot()
        self.assertEqual(pilot.execute(), 1)
        self.assertIn("Initial class tokens differ", pilot.state["detail"])
        self.assertEqual(len([cmd for cmd, _ in self.commands if "--learning-rate" in cmd]), 2)
        self.assertEqual(len([cmd for cmd, _ in self.commands if "--learned-checkpoint" in cmd]), 1)

    def test_failure_preserves_recovery_checkpoint_and_never_retries(self):
        self.failure = 7
        pilot = self.pilot()
        self.assertEqual(pilot.execute(), 1)
        self.assertEqual(pilot.state["commands"][-1]["returncode"], 7)
        self.assertTrue((self.root / "pilot/trials/lr-0.01-step1000/train/latest.pt").is_file())
        self.assertEqual(len(pilot.state["trials"]), 1)

    def test_observed_image_mismatch_stops_before_evaluation(self):
        self.observed_mismatch = True
        pilot = self.pilot(["--learning-rates", "0.003"])
        self.assertEqual(pilot.execute(), 1)
        self.assertIn("Observed training image/query provenance", pilot.state["detail"])
        self.assertFalse(any("--learned-checkpoint" in command for command, _ in self.commands))

    def test_core_content_and_file_set_changes_invalidate_comparison(self):
        self.change_core = True
        pilot = self.pilot(["--learning-rates", "0.003"])
        self.assertEqual(pilot.execute(), 1)
        self.assertFalse(pilot.state["comparison_valid"])
        self.assertEqual(pilot.state["core_source_change"]["modified"], ["sam3/core.py"])
        self.assertEqual(pilot.state["core_source_change"]["added"], ["sam3/added.py"])
        self.assertFalse(any("--learned-checkpoint" in command for command, _ in self.commands))

    def test_timeout_is_bounded_by_remaining_budget(self):
        self.clock = FakeClock(1300)
        self.failure = "timeout"
        pilot = self.pilot()
        self.assertEqual(pilot.execute(), 1)
        record = pilot.state["commands"][-1]
        self.assertEqual(record["timeout_seconds"], 1300)
        self.assertTrue(record["timed_out"])

    def test_insufficient_memory_and_time_do_not_start_training(self):
        self.clock = FakeClock(1210)
        self.gpu_free = [4000]
        pilot = self.pilot()
        self.assertEqual(pilot.execute(), 2)
        self.assertEqual(pilot.state["trials"], [])

    def test_early_gpu3_batch1_uses_same_2000_sample_budget(self):
        self.formal_summary.unlink()
        pilot = self.pilot([
            "--gpu", "3", "--batch-size", "1", "--max-steps", "2000",
            "--start-before-formal", "--initial-reference-checkpoint", str(self.formal_checkpoint),
        ])
        self.assertEqual(pilot.execute(), 0)
        self.assertTrue(pilot.state["early_start_authorized"])
        self.assertEqual({trial["actual_steps"] for trial in pilot.state["trials"]}, {2000})
        for record in pilot.state["commands"]:
            self.assertEqual(record["environment"]["CUDA_VISIBLE_DEVICES"], "3")

    def test_existing_run_cannot_be_restarted(self):
        self.assertEqual(self.pilot().execute(), 0)
        with self.assertRaises(FileExistsError):
            self.pilot().execute()

    def test_single_learning_rate_runs_one_train_eval_and_calibration(self):
        pilot = self.pilot(["--learning-rates", "0.003", "--train-timeout-seconds", "3600"])
        self.assertEqual(pilot.execute(), 0)
        self.assertEqual(len(pilot.state["trials"]), 1)
        self.assertEqual(pilot.state["trials"][0]["learning_rate"], 0.003)
        self.assertEqual(len(pilot.state["commands"]), 3)
        self.assertEqual(pilot.state["commands"][0]["timeout_seconds"], 3600)
        self.assertEqual(pilot.state["fixed_training"]["learning_rates"], [0.003])
        self.assertIn("1 same-prefix", pilot.state["detail"])

    def test_configuration_cannot_extend_deadline_or_break_same_sample_budget(self):
        for extra in (["--deadline", "2026-09-11T21:40:00+08:00"],
                      ["--batch-size", "1", "--max-steps", "1000"],
                      ["--start-before-formal"], ["--gpu", "1"],
                      ["--learning-rates", "0.02"], ["--learning-rates", "0.003", "0.003"],
                      ["--train-timeout-seconds", "3601"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_args(self.argv + extra)


if __name__ == "__main__":
    unittest.main()

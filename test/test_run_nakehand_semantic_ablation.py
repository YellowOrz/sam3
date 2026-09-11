"""CPU-only supervisor/deadline and full paired-report contract checks."""

from argparse import Namespace
import contextlib
from copy import deepcopy
from datetime import timedelta
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from scripts import report_nakehand_semantic_ablation as report
from scripts import run_nakehand_semantic_ablation as runner


class NakehandAblationSupervisorTest(unittest.TestCase):
    @staticmethod
    def checkpoint(weight=0., steps=2000):
        return {"format": "sam3-nakehand-semantic-delta-training-v1", "next_step": steps,
                "training_config": {"anchor_weight": weight, "learning_rate": .001, "seed": 123,
                                    "batch_size": 1, "implementation_sha256": {"core": "same"}},
                "progress": {"samples_seen": steps, "completed_steps": steps, "pilot_complete": steps == 2000},
                "planned_dataset_indices": list(range(2000)), "observed_image_ids": list(range(steps)),
                "initial_cache_state_dict": {"delta": torch.zeros(2, 4, 256),
                                             "resized_cache": torch.ones(32, 2, 256, dtype=torch.bfloat16)}}

    def test_deadline_timeout_never_starts_last_two_minutes(self):
        self.assertEqual(runner.stage_timeout(500, 100), 100)
        self.assertEqual(runner.stage_timeout(500, 1000), 470)
        self.assertEqual(runner.stage_timeout(120, 1000), 90)
        with self.assertRaises(TimeoutError):
            runner.stage_timeout(119, 1000)

    def test_user_removed_cutoff_but_explicit_timezone_and_gpu_isolation_remain(self):
        base = ["--data-root", "/tmp/semantic-dataset", "--base-checkpoint", "/tmp/base.pt",
                "--initial-cache", "/tmp/cache.pt", "--unconstrained-resume", "/tmp/un.pt",
                "--constrained-resume", "/tmp/an.pt", "--output-dir", "/tmp/ablation-nonexistent-output-20260910"]
        self.assertIsNone(runner.parse_args(base).deadline)
        self.assertGreater(runner.parse_args(base + ["--deadline", "2026-09-11T08:00:00+08:00"]).deadline,
                           runner.LEGACY_DEADLINE)
        self.assertEqual(runner.stage_timeout(float("inf"), 2700), 2700)
        for additional in (["--deadline", "2026-09-10T21:00:00"], ["--gpus", "0", "0"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(base + additional)

    def test_child_timeout_is_own_command_with_gpu_isolation_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "logs").mkdir()
            args = Namespace(output_dir=output, project_root=output, data_root=output / "data",
                             deadline=runner.LEGACY_DEADLINE, gpus=[0, 3])
            command = ["python", "owned-training.py"]
            run = Mock(side_effect=subprocess.TimeoutExpired(command, 470))
            supervisor = runner.AblationRunner(args, now=lambda: runner.LEGACY_DEADLINE - timedelta(seconds=500), run=run)
            supervisor.verify_sources = Mock()
            with self.assertRaises(subprocess.TimeoutExpired):
                supervisor.child("fixture", command, gpu=3, cap=1000, needs_gpu=False)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], command)
            self.assertEqual(run.call_args.kwargs["timeout"], 470)
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["commands"]["fixture"]["status"], "failed")
            self.assertFalse((output / "state.json.tmp").exists())

    @staticmethod
    def supervisor_args(output):
        return Namespace(output_dir=output, project_root=output, data_root=output / "data",
                         unconstrained_resume=output / "un.pt", constrained_resume=output / "an.pt",
                         deadline=None, gpus=[0, 3])

    def test_gpu_wait_has_finite_limit_without_deadline_and_never_launches_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            elapsed = [0.]

            def advance(seconds):
                elapsed[0] += seconds

            run = Mock(return_value=subprocess.CompletedProcess(["nvidia-smi"], 0, "7000\n"))
            supervisor = runner.AblationRunner(self.supervisor_args(output), run=run,
                                               sleep=advance, monotonic=lambda: elapsed[0])
            supervisor.verify_sources = Mock()
            with self.assertRaisesRegex(TimeoutError, "1800 seconds"):
                supervisor.child("waiting", ["must-not-run.py"], gpu=3, cap=2700)
            self.assertEqual(elapsed[0], runner.GPU_WAIT_TIMEOUT_SECONDS)
            self.assertEqual(run.call_count, 60)
            self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in run.call_args_list))
            self.assertTrue(all(0 < call.kwargs["timeout"] <= 10 for call in run.call_args_list))
            supervisor.verify_sources.assert_not_called()
            saved = json.loads((output / "state.json").read_text())
            self.assertEqual(saved["commands"]["waiting"]["status"], "failed")

    def test_either_trial_failure_cancels_peer_waiting_for_gpu(self):
        # Exercise both completion orders: the second submitted trial must not
        # wait for the first future before its error can stop resource polling.
        for failed_label in ("unconstrained", "anchored"):
            with self.subTest(failed_label=failed_label), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                peer_waiting = threading.Event()
                failures = []

                def query(command, **kwargs):
                    self.assertEqual(command[0], "nvidia-smi")
                    peer_waiting.set()
                    return subprocess.CompletedProcess(command, 0, "7000\n")

                supervisor = runner.AblationRunner(self.supervisor_args(output), run=query)
                supervisor.verify_sources = Mock()

                def trial(label, _weight, _resume, gpu):
                    if label == failed_label:
                        if not peer_waiting.wait(timeout=2):
                            raise TimeoutError("Test peer did not start polling")
                        raise RuntimeError("original paired failure")
                    supervisor.child(label, ["must-not-launch.py"], gpu=gpu, cap=2700)

                supervisor.train_trial = trial

                def run_pair():
                    try:
                        supervisor.run_training_trials()
                    except BaseException as error:
                        failures.append(error)

                worker = threading.Thread(target=run_pair)
                worker.start()
                try:
                    worker.join(timeout=2)
                    self.assertFalse(worker.is_alive(), "GPU waiter was not cancelled promptly")
                    self.assertEqual(len(failures), 1)
                    self.assertEqual(str(failures[0]), "original paired failure")
                    state = json.loads((output / "state.json").read_text())
                    self.assertEqual(state["status"], "stopping_after_failure")
                    self.assertEqual(state["failed_trial"], failed_label)
                    peer = "anchored" if failed_label == "unconstrained" else "unconstrained"
                    self.assertEqual(state["commands"][peer]["status"], "cancelled")
                    supervisor.verify_sources.assert_not_called()
                finally:
                    supervisor.stop_event.set()
                    worker.join(timeout=2)

    def test_failure_is_published_while_already_running_peer_finishes_normally(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "logs").mkdir()
            peer_running, release_peer = threading.Event(), threading.Event()
            failures, invocations = [], []

            def active_child(command, **kwargs):
                invocations.append((command, kwargs))
                peer_running.set()
                if not release_peer.wait(timeout=3):
                    raise TimeoutError("Test did not release the running peer")
                return subprocess.CompletedProcess(command, 0)

            supervisor = runner.AblationRunner(self.supervisor_args(output), run=active_child)
            supervisor.verify_sources = Mock()

            def trial(label, _weight, _resume, gpu):
                if label == "anchored":
                    if not peer_running.wait(timeout=2):
                        raise TimeoutError("Test peer did not launch")
                    raise RuntimeError("anchored failed")
                supervisor.child(label, ["owned-running-training.py"], gpu=gpu, cap=2700, needs_gpu=False)
                return "retained-peer-checkpoint", {}

            supervisor.train_trial = trial

            def run_pair():
                try:
                    supervisor.run_training_trials()
                except BaseException as error:
                    failures.append(error)

            worker = threading.Thread(target=run_pair)
            worker.start()
            try:
                self.assertTrue(supervisor.stop_event.wait(timeout=2))
                # stop_event precedes the atomic status write. Wait for the
                # published status without relying on a thread scheduling race.
                published = threading.Event()
                for _ in range(100):
                    with supervisor.lock:
                        if supervisor.state["status"] == "stopping_after_failure":
                            published.set()
                    if published.wait(timeout=.01):
                        break
                self.assertTrue(published.is_set())
                self.assertTrue(worker.is_alive(), "Running peer was not allowed to finish")
                state = json.loads((output / "state.json").read_text())
                self.assertEqual(state["status"], "stopping_after_failure")
                self.assertEqual(state["commands"]["unconstrained"]["status"], "running")
                self.assertEqual(len(invocations), 1)
                self.assertEqual(invocations[0][1]["timeout"], 2700)
            finally:
                release_peer.set()
                worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            self.assertEqual([str(error) for error in failures], ["anchored failed"])
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(state["commands"]["unconstrained"]["status"], "completed")
            self.assertEqual(state["status"], "stopping_after_failure")
            self.assertEqual(len(invocations), 1)

    def test_completed_pairs_keep_label_order_despite_completion_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = runner.AblationRunner(self.supervisor_args(Path(temporary)))
            second_completed = threading.Event()

            def trial(label, _weight, _resume, _gpu):
                if label == "unconstrained":
                    if not second_completed.wait(timeout=2):
                        raise TimeoutError("Test second trial did not complete")
                else:
                    second_completed.set()
                return label, {"label": label}

            supervisor.train_trial = trial
            first, second = supervisor.run_training_trials()
            self.assertEqual(first[0], "unconstrained")
            self.assertEqual(second[0], "anchored")
            self.assertFalse(supervisor.stop_event.is_set())

    def test_training_comparison_requires_only_anchor_difference_and_actual_2000_prefix(self):
        first, second = self.checkpoint(0.), self.checkpoint(1.)
        self.assertTrue(report.compare_training_contracts(first, second)["only_anchor_weight_differs"])
        for mutate in (lambda value: value["training_config"].update(learning_rate=.01),
                       lambda value: value["observed_image_ids"].__setitem__(0, 7),
                       lambda value: value["initial_cache_state_dict"]["delta"].fill_(.01),
                       lambda value: value.update(next_step=20)):
            changed = deepcopy(second)
            mutate(changed)
            with self.assertRaises(ValueError):
                report.compare_training_contracts(first, changed)

    def test_source_guard_catches_modified_executed_snapshot_and_new_core_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sam3").mkdir()
            core, snapshot = root / "sam3" / "model.py", root / "snapshot.py"
            core.write_text("model = 1\n")
            snapshot.write_text("train = 1\n")
            args = Namespace(output_dir=root, project_root=root, data_root=root / "data",
                             deadline=runner.LEGACY_DEADLINE, gpus=[0, 3])
            supervisor = runner.AblationRunner(args)
            supervisor.state["sources"] = [{"path": str(path), "kind": kind, "sha256": report.file_hash(path)}
                                           for path, kind in ((core, "core"), (snapshot, "snapshot"))]
            supervisor.verify_sources()
            snapshot.write_text("train = 2\n")
            with self.assertRaisesRegex(RuntimeError, "Frozen source changed"):
                supervisor.verify_sources()
            snapshot.write_text("train = 1\n")
            (root / "sam3" / "extra.py").write_text("extra = 1\n")
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                supervisor.verify_sources()

    def test_completed_checkpoint_rejects_partial_or_outside_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "checkpoint.pt"
            state = self.checkpoint()
            torch.save(state, path)
            summary = {"format": state["format"], "final_checkpoint": str(path),
                       "final_checkpoint_sha256": report.file_hash(path), "progress": state["progress"]}
            self.assertEqual(runner.completed_checkpoint(summary, output)[0], path)
            with self.assertRaisesRegex(ValueError, "outside"):
                runner.completed_checkpoint(summary, output / "other-trial")
            state = self.checkpoint(steps=20)
            torch.save(state, path)
            summary.update(final_checkpoint_sha256=report.file_hash(path), progress=state["progress"])
            with self.assertRaisesRegex(ValueError, "2000"):
                runner.completed_checkpoint(summary, output)

    @staticmethod
    def summary_fixture():
        own = np.array([[1, 0], [0, 0]], dtype=bool)
        other = np.array([[0, 0], [0, 1]], dtype=bool)
        measure = report.metrics.measure_query(own, own, other, .72)
        records = {label: [{**measure, "model": label, "image_id": image_id, "prompt_key": side,
                            "identity_verified": True, "presence_probability": .9, "top_class_probability": .8,
                            "observed_coco_image_id": image_id, "split": "val", "dataset_role": "validation", "primary_test": True,
                            "recording_id": "nakehandego/20260907_142020", "view_type": "ego"}
                           for image_id in range(3449) for side in report.metrics.CLASS_NAMES]
                   for label in report.LABELS}
        summary = {"format": report.FORMAT, "status": "completed", "full_val_evaluated": True,
                   "evaluated_images": 3449, "evaluated_image_ids": list(range(3449)),
                   "diagnostic_training_checkpoint": False, "detection_threshold": .5, "mask_threshold": .5,
                   "dataset_role": "validation", "training_performed": False, "thresholds_fitted_on_nakehand": False,
                   "observed_identity_verified": True, "all_sources_unchanged": True, "actual_training_prefix_verified": True,
                   "models": {label: {"kind": "frozen_natural_ve_cache" if index == 0 else "semantic_delta",
                                      "anchor_weight": None if index == 0 else float(index == 2),
                                      "training_applied_to_this_variant": index != 0}
                              for index, label in enumerate(report.LABELS)},
                   "metrics": {label: {"validation": report.metrics.grouped_summary(rows)}
                               for label, rows in records.items()}}
        return summary, records

    def test_report_recalculates_all_three_models_on_full_validation(self):
        summary, records = self.summary_fixture()
        actual = report.validate_summary(summary, records)
        self.assertEqual(set(actual), set(report.LABELS))
        self.assertEqual(actual[report.LABELS[0]]["overall"]["queries"], 6898)
        self.assertEqual(actual[report.LABELS[0]]["overall"]["present_mean_candidate_dice"], 1.)
        broken = deepcopy(summary)
        broken["metrics"][report.LABELS[0]]["validation"]["overall"]["present_mean_candidate_dice"] = .8
        with self.assertRaisesRegex(ValueError, "metrics differ"):
            report.validate_summary(broken, records)
        records[report.LABELS[1]][0]["detected"] = False
        with self.assertRaisesRegex(ValueError, "confidence or threshold"):
            report.validate_summary(summary, records)

    def test_report_rejects_partial_diagnostic_or_missing_query(self):
        summary, records = self.summary_fixture()
        for changes in ({"status": "failed"}, {"evaluated_images": 10}, {"diagnostic_training_checkpoint": True},
                        {"detection_threshold": .3}, {"models": {report.LABELS[0]: {}}},
                        {"thresholds_fitted_on_nakehand": True}, {"all_sources_unchanged": False}):
            with self.assertRaises(ValueError):
                report.validate_summary({**summary, **changes}, records)
        records[report.LABELS[2]].pop()
        with self.assertRaisesRegex(ValueError, "actual query identities"):
            report.validate_summary(summary, records)


if __name__ == "__main__":
    unittest.main()

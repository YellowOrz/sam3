#!/usr/bin/env python3
"""Run two independent, bounded nakehand delta continuations and one paired val.

Use caller-owned tmux. Only own child processes may be killed on timeout.
No random initialization, geometry training, holdout evaluation, or retries.
GPU resource waits are capped at 1800 seconds and cancelled when a peer fails;
already-running peers retain their original finite timeout and checkpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from scripts.report_nakehand_semantic_ablation import compare_training_contracts, file_hash

LEGACY_DEADLINE = datetime.fromisoformat("2026-09-10T21:40:00+08:00")
GPU_WAIT_TIMEOUT_SECONDS = 1800
SCRIPTS = (
    "run_nakehand_semantic_ablation.py", "report_nakehand_semantic_ablation.py",
    "train_nakehand_semantic_tokens.py", "evaluate_nakehand_semantic_tokens.py",
    "prepare_nakehand_training.py", "prepare_nakehand_test.py", "audit_nakehand_dataset.py",
    "cached_ve_text_features.py", "train_ve_initialized_tokens.py", "evaluate_ve_initialized_tokens.py",
    "train_learnable_tokens.py", "evaluate_bilateral_tokens.py", "evaluate_nakehand_tokens.py",
    "run_token_lr_pilot.py", "finish_bilateral_validation.py", "render_separated_masks.py",
)


def stage_timeout(remaining, cap):
    if remaining < 120:
        raise TimeoutError("Do not start work within the last 120 authorized seconds")
    return min(cap, remaining - 30)


def completed_checkpoint(summary, output_dir):
    import torch

    if summary.get("format") != "sam3-nakehand-semantic-delta-training-v1":
        raise ValueError("Wrong training summary format")
    path = Path(summary["final_checkpoint"]).resolve()
    if path.parent != output_dir.resolve() or file_hash(path) != summary["final_checkpoint_sha256"]:
        raise ValueError("Checkpoint is outside this trial or its hash differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("progress") != summary.get("progress"):
        raise ValueError("Checkpoint and summary progress differ")
    progress = state.get("progress", {})
    if (progress.get("samples_seen") != 2000 or progress.get("completed_steps") != 2000
            or progress.get("pilot_complete") is not True or state.get("next_step") != 2000):
        raise ValueError("Only actual 2000-sample completion permits full validation")
    return path, state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "unconstrained-resume", "constrained-resume", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 3])
    parser.add_argument("--deadline", help="Optional timezone-aware cutoff; user removed the previous fixed cutoff")
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    if args.deadline is not None and args.deadline.tzinfo is None:
        parser.error("An explicit deadline must include its timezone")
    if len(set(args.gpus)) != 2 or any(gpu not in (0, 1, 2, 3) for gpu in args.gpus):
        parser.error("Choose two distinct authorized GPUs from 0,1,2,3")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists():
        parser.error("Output must be new")
    for root in (args.project_root, args.data_root):
        if args.output_dir == root or root in args.output_dir.parents:
            parser.error("Do not put experiment outputs inside code or source data")
    return args


class AblationRunner:
    def __init__(self, args, *, now=None, run=None, sleep=None, monotonic=None):
        self.args = args
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.run = run or subprocess.run
        self.stop_event = threading.Event()
        # Production polling wakes immediately when a peer fails. Tests may
        # inject a clock-advancing wait without actually sleeping.
        self.sleep = sleep or self.stop_event.wait
        self.monotonic = monotonic or time.monotonic
        self.lock = threading.Lock()
        self.state = {"format": "sam3-nakehand-semantic-ablation-supervisor-v1", "status": "created",
                      "created_at": self.now().isoformat(), "deadline": args.deadline.isoformat() if args.deadline else None,
                      "authorization": "User removed the fixed cutoff on 2026-09-10; finite per-stage limits remain",
                      "data_root": str(args.data_root), "gpus": args.gpus,
                      "commands": {}, "sources": [], "trials": {}, "planned_samples_per_trial": 2000,
                      "gpu_wait_timeout_seconds": GPU_WAIT_TIMEOUT_SECONDS,
                      "peer_failure_policy": "cancel waiting work; retain already-running child through its finite stage cap",
                      "geometry_training": False, "development_holdout_used": False}

    def remaining(self):
        return (self.args.deadline - self.now()).total_seconds() if self.args.deadline else float("inf")

    def save(self):
        temporary = self.args.output_dir / "state.json.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.args.output_dir / "state.json")

    def update(self, key, **values):
        with self.lock:
            if key is None:
                self.state.update(values)
            else:
                self.state["commands"].setdefault(key, {}).update(values)
            self.state["updated_at"] = self.now().isoformat()
            self.save()

    def verify_sources(self):
        expected_core = {row["path"] for row in self.state["sources"] if row["kind"] == "core"}
        actual_core = {str(path.resolve()) for path in (self.args.project_root / "sam3").rglob("*.py")}
        if expected_core != actual_core:
            raise RuntimeError("Core source file inventory changed")
        for row in self.state["sources"]:
            if file_hash(row["path"]) != row["sha256"]:
                raise RuntimeError(f"Frozen source changed: {row['path']}")

    def check_stopping(self):
        if self.stop_event.is_set():
            raise CancelledError("Paired trial failed; cancel work that has not started a child process")

    def wait_for_gpu(self, gpu):
        started = self.monotonic()
        while self.remaining() >= 120:
            self.check_stopping()
            wait_remaining = GPU_WAIT_TIMEOUT_SECONDS - (self.monotonic() - started)
            if wait_remaining <= 0:
                raise TimeoutError(f"GPU {gpu} did not have 7500 MiB free within {GPU_WAIT_TIMEOUT_SECONDS} seconds")
            result = self.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=min(10, self.remaining(), wait_remaining), check=False)
            self.check_stopping()
            if result.returncode or not result.stdout.strip().isdigit():
                raise RuntimeError(f"Cannot safely query GPU {gpu}")
            wait_remaining = GPU_WAIT_TIMEOUT_SECONDS - (self.monotonic() - started)
            if wait_remaining <= 0:
                raise TimeoutError(f"GPU {gpu} resource-wait limit reached")
            if int(result.stdout.strip()) >= 7500:
                return
            self.sleep(min(30, wait_remaining, max(0, self.remaining() - 119)))
        raise TimeoutError("Insufficient remaining authorization while waiting for GPU")

    def child(self, label, command, *, gpu, cap, needs_gpu=True):
        try:
            self.check_stopping()
            if needs_gpu:
                self.update(label, status="waiting_for_gpu", gpu=gpu,
                            gpu_wait_timeout_seconds=GPU_WAIT_TIMEOUT_SECONDS)
                self.wait_for_gpu(gpu)
            self.verify_sources()
            self.check_stopping()
            timeout = stage_timeout(self.remaining(), cap)
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
            environment["PYTHONPATH"] = os.pathsep.join((str(self.args.project_root), str(self.args.project_root / "scripts")))
            log = self.args.output_dir / "logs" / f"{label}.log"
            self.update(label, status="running", started_at=self.now().isoformat(), gpu=gpu,
                        command=list(command), timeout_seconds=timeout, log=str(log))
            self.check_stopping()
            with log.open("x") as stream:
                result = self.run(command, cwd=self.args.project_root, env=environment,
                                  stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            if result.returncode:
                raise RuntimeError(f"{label} failed with exit {result.returncode}; see {log}")
            self.verify_sources()
            self.update(label, status="completed", returncode=0, completed_at=self.now().isoformat())
        except BaseException as error:
            self.update(label, status="cancelled" if isinstance(error, CancelledError) else "failed",
                        error=f"{type(error).__name__}: {error}",
                        completed_at=self.now().isoformat())
            raise

    def snapshot(self):
        for split in ("train", "val", "development_holdout"):
            if not (self.args.data_root / split / "READY.json").is_file():
                raise ValueError(f"Missing fully exported split READY: {split}")
        if not (self.args.data_root / "READY.json").is_file():
            raise ValueError("Missing complete dataset READY")
        paths = [(path, "core") for path in sorted((self.args.project_root / "sam3").rglob("*.py"))]
        if not paths:
            raise ValueError("Missing core source tree")
        for name in SCRIPTS:
            source = self.args.project_root / "scripts" / name
            target = self.args.output_dir / "snapshots" / name
            digest = file_hash(source)
            shutil.copy2(source, target)
            if file_hash(target) != digest or file_hash(source) != digest:
                raise RuntimeError("Script changed while snapshotting")
            paths.append((source, "script"))
            paths.append((target, "snapshot"))
        inputs = [self.args.base_checkpoint, self.args.initial_cache,
                  self.args.unconstrained_resume, self.args.constrained_resume,
                  self.args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"]
        inputs += [self.args.data_root / name for name in ("READY.json", "manifest.json", "frozen-plan.json")]
        for split in ("train", "val"):
            inputs += [self.args.data_root / split / name for name in ("READY.json", "manifest.json", "annotations.json")]
        paths += [(path, "input") for path in inputs]
        self.state["sources"] = [{"path": str(path.resolve()), "kind": kind, "sha256": file_hash(path)} for path, kind in paths]
        self.update(None, status="ready_to_train")

    def train_trial(self, label, weight, resume, gpu):
        import torch

        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (checkpoint.get("format") != "sam3-nakehand-semantic-delta-training-v1"
                or checkpoint.get("next_step") != 20
                or checkpoint.get("training_config", {}).get("anchor_weight") != weight):
            raise ValueError("Each continuation requires its own verified 20-step semantic checkpoint")
        output = self.args.output_dir / label
        command = [self.args.python, "-u", str(self.args.output_dir / "snapshots/train_nakehand_semantic_tokens.py"),
                   "--data-root", str(self.args.data_root / "train"), "--base-checkpoint", str(self.args.base_checkpoint),
                   "--initial-cache", str(self.args.initial_cache), "--anchor-weight", str(weight),
                   "--resume", str(resume), "--project-root", str(self.args.project_root),
                   "--max-steps", "2000", "--gpu-memory-fraction", "0.25", "--output-dir", str(output)]
        self.child(label, command, gpu=gpu, cap=2700)
        path, state = completed_checkpoint(json.loads((output / "summary.json").read_text()), output)
        with self.lock:
            self.state["trials"][label] = {"checkpoint": str(path), "sha256": file_hash(path), "progress": state["progress"]}
            self.save()
        return path, state

    def run_training_trials(self):
        """Surface either peer's failure before waiting for an active child.

        The executor still waits for already-running subprocesses so their
        finite timeout/recovery handling remains intact. Idle GPU waiters wake
        through stop_event instead of delaying failure indefinitely.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.train_trial, "unconstrained", 0., self.args.unconstrained_resume,
                                self.args.gpus[0]): "unconstrained",
                executor.submit(self.train_trial, "anchored", 1., self.args.constrained_resume,
                                self.args.gpus[1]): "anchored",
            }
            current_label = None
            try:
                for future in as_completed(futures):
                    current_label = futures[future]
                    results[current_label] = future.result()
            except BaseException as error:
                self.stop_event.set()
                self.update(None, status="stopping_after_failure", failed_trial=current_label,
                            error=f"{type(error).__name__}: {error}")
                for pending, label in futures.items():
                    if pending.cancel():
                        self.update(label, status="cancelled", completed_at=self.now().isoformat(),
                                    error="Cancelled before execution because a paired trial failed")
                raise
        return results["unconstrained"], results["anchored"]

    def execute(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("logs", "snapshots"):
            (self.args.output_dir / name).mkdir()
        self.update(None, status="created")
        try:
            self.snapshot()
            (un_path, un_state), (an_path, an_state) = self.run_training_trials()
            comparison = compare_training_contracts(un_state, an_state)
            self.update(None, status="training_complete", training_comparability=comparison)
            output = self.args.output_dir / "validation"
            command = [self.args.python, "-u", str(self.args.output_dir / "snapshots/evaluate_nakehand_semantic_tokens.py"),
                       "--data-root", str(self.args.data_root / "val"), "--base-checkpoint", str(self.args.base_checkpoint),
                       "--baseline-checkpoint", str(un_path), "--unconstrained-checkpoint", str(un_path),
                       "--constrained-checkpoint", str(an_path), "--variant", "all",
                       "--project-root", str(self.args.project_root), "--minimum-samples-seen", "2000",
                       "--gpu-memory-fraction", "0.25",
                       "--output-dir", str(output)]
            if self.args.deadline is not None:
                command.extend(("--deadline", self.args.deadline.isoformat()))
            self.child("full_validation", command, gpu=self.args.gpus[0], cap=5400)
            report = self.args.output_dir / "NAKEHAND_SEMANTIC_REPORT.md"
            command = [self.args.python, str(self.args.output_dir / "snapshots/report_nakehand_semantic_ablation.py"),
                       "--summary", str(output / "summary.json"), "--unconstrained-checkpoint", str(un_path),
                       "--constrained-checkpoint", str(an_path), "--output", str(report)]
            self.child("report", command, gpu=self.args.gpus[0], cap=120, needs_gpu=False)
            self.update(None, status="complete", report=str(report), report_sha256=file_hash(report))
            return 0
        except BaseException as error:
            self.update(None, status="failed_or_stopped", error=f"{type(error).__name__}: {error}")
            return 1


def main():
    raise SystemExit(AblationRunner(parse_args()).execute())


if __name__ == "__main__":
    main()

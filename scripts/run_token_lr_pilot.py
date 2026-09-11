#!/usr/bin/env python3
"""Queue a fixed, single-factor token-LR pilot on one explicitly selected GPU.

Fresh runs differ only in AdamW LR: .01/.003/.001, each seeing 2000 identical
training images (default 1000 steps/batch2; optional 2000 steps/batch1). Each run is
followed by full-val learned-token-only evaluation. This is a short-budget
comparison, not three completed two-epoch trainings. No MANO or GPU1 jobs run.
By default wait for formal training; optional early GPU2/3 work needs an explicit
existing reference checkpoint to verify identical initialization and data order.
Use tmux; a fresh output directory is mandatory. Failures stop without retry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


DEADLINE = datetime.fromisoformat("2026-09-10T21:40:00+08:00")
LEARNING_RATES = (0.01, 0.003, 0.001)
PILOT_STEPS = 1000
PILOT_SAMPLES = 2000
LOSS_NAMES = ("mask", "dice", "bbox", "giou", "classification", "presence")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def core_source_hashes(project_root):
    """Include both the Python source file set and each file's exact contents."""
    project_root = Path(project_root)
    return {
        path.relative_to(project_root).as_posix(): file_hash(path)
        for path in sorted((project_root / "sam3").rglob("*.py")) if path.is_file()
    }


def token_hash(tensor):
    import torch

    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (2, 4, 256)
            or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item()):
        raise ValueError("Expected finite float32 token tensor with shape (2,4,256)")
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_checkpoint(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def expected_order(dataset_size):
    order = []
    for seed in (123, 124):
        epoch = list(range(dataset_size))
        random.Random(seed).shuffle(epoch)
        order.extend(epoch)
    return order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("formal-summary", "formal-checkpoint", "data-root", "val-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--unified-root", type=Path, default=Path("/data/xuzhefeng/Datasets/uni-hoi-dataset"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, choices=(0, 2, 3), default=0)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=list(LEARNING_RATES))
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-steps", type=int, default=PILOT_STEPS)
    parser.add_argument("--train-memory-fraction", type=float, default=0.35)
    parser.add_argument("--eval-memory-fraction", type=float, default=0.25)
    parser.add_argument("--start-before-formal", action="store_true")
    parser.add_argument("--initial-reference-checkpoint", type=Path)
    parser.add_argument("--minimum-free-mib", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--minimum-start-seconds", type=float, default=1200)
    parser.add_argument("--train-timeout-seconds", type=float, default=1800)
    parser.add_argument("--deadline", default=DEADLINE.isoformat())
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or args.deadline > DEADLINE:
        parser.error("deadline must be timezone-aware and no later than 2026-09-10T21:40:00+08:00")
    if not 0 < args.poll_seconds <= 30 or args.minimum_free_mib < 1:
        parser.error("poll interval must be (0,30] seconds and minimum free memory positive")
    if args.batch_size * args.max_steps != PILOT_SAMPLES:
        parser.error("All runs must see 2000 samples: batch2/1000steps or batch1/2000steps")
    if (any(rate not in LEARNING_RATES for rate in args.learning_rates)
            or len(set(args.learning_rates)) != len(args.learning_rates)):
        parser.error("learning-rates must be distinct values chosen from .01/.003/.001")
    if not 0 < args.train_timeout_seconds <= 3600:
        parser.error("train-timeout-seconds must be within (0,3600]")
    if not 0 < args.train_memory_fraction <= 0.35 or not 0 < args.eval_memory_fraction <= 0.25:
        parser.error("Allocator caps must be positive and no greater than train .35 / eval .25")
    if args.start_before_formal and (args.gpu not in (2, 3) or args.initial_reference_checkpoint is None):
        parser.error("Early runs require GPU2/3 and an explicit initial-reference-checkpoint")
    if not math.isfinite(args.minimum_start_seconds) or args.minimum_start_seconds < 1200:
        parser.error("minimum start budget must be >=1200 seconds")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    for protected in (args.project_root, args.unified_root, args.data_root, args.val_root):
        if args.output_dir == protected or protected in args.output_dir.parents:
            parser.error("output must be outside the project and input dataset directories")
    return args


class Pilot:
    def __init__(self, args, *, now=None, sleep=None, run=None):
        self.args = args
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or time.sleep
        self.run = run or subprocess.run
        self.last_status = None
        self.state = {
            "format": "sam3-token-lr-pilot-v1", "status": "created",
            "created_at": self.now().isoformat(), "deadline": args.deadline.isoformat(),
            "config": {key: value.isoformat() if isinstance(value, datetime)
                       else str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
            "fixed_training": {"learning_rates": list(args.learning_rates), "steps": args.max_steps,
                               "samples": PILOT_SAMPLES, "batch_size": args.batch_size, "amp": "BF16",
                               "seed": 123, "K": 4, "planned_epochs": 2,
                               "loss_weights": {name: 1.0 for name in LOSS_NAMES}},
            "commands": [], "artifacts": {}, "trials": [],
            "limitations": ["Equal short-run budget measures early optimization, not final LR convergence.",
                            "No VE initialization, loss changes, MANO, or GPU1 tasks.",
                            "Validation thresholds are fitted on val and are not independent test guarantees."],
        }

    def remaining(self):
        return (self.args.deadline - self.now()).total_seconds()

    def save(self):
        temporary = self.args.output_dir / ".state.json.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.args.output_dir / "state.json")

    def transition(self, status, detail=""):
        if (status, detail) == self.last_status:
            return
        self.last_status = (status, detail)
        self.state.update(status=status, detail=detail, updated_at=self.now().isoformat())
        self.save()
        with (self.args.output_dir / "pilot.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{self.now().isoformat()} {status} {detail}\n")

    def pause(self):
        self.sleep(max(0, min(self.args.poll_seconds, self.remaining())))

    def snapshot(self, source, name):
        target = self.args.output_dir / "snapshots" / name
        raw = Path(source).read_bytes()
        with target.open("xb") as stream:
            stream.write(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if file_hash(source) != digest:
            raise RuntimeError(f"Source changed during snapshot: {source}")
        self.state["artifacts"][name] = {"source": str(source), "path": str(target), "sha256": digest}
        self.save()
        return target

    def verify_core_sources(self):
        expected = self.state["core_python_sources"]
        try:
            current = core_source_hashes(self.args.project_root)
        except OSError as error:
            self.state["comparison_valid"] = False
            self.state["core_source_change"] = {"read_error": str(error)}
            self.save()
            raise RuntimeError("Could not verify sam3 sources; comparison invalid") from error
        if current == expected:
            return
        self.state["comparison_valid"] = False
        self.state["core_source_change"] = {
            "added": sorted(current.keys() - expected.keys()),
            "removed": sorted(expected.keys() - current.keys()),
            "modified": sorted(key for key in current.keys() & expected.keys()
                               if current[key] != expected[key]),
        }
        self.save()
        raise RuntimeError("sam3 Python sources changed; pilot results are not comparable")

    def child(self, label, command, maximum_seconds, *, guard_core=False):
        if guard_core:
            self.verify_core_sources()
        timeout = min(maximum_seconds, self.remaining())
        if timeout <= 0:
            raise TimeoutError("Deadline reached before subprocess launch")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.args.gpu), OMP_NUM_THREADS="2")
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(self.args.project_root), str(self.args.project_root / "scripts"),
            environment.get("PYTHONPATH"),
        )))
        log_path = self.args.output_dir / "logs" / f"{label}.log"
        record = {"label": label, "command": command, "timeout_seconds": timeout,
                  "cwd": str(self.args.project_root), "started_at": self.now().isoformat(),
                  "log": str(log_path), "returncode": None,
                  "environment": {key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")}}
        self.state["commands"].append(record)
        self.transition("running", label)
        try:
            with log_path.open("x", encoding="utf-8") as stream:
                result = self.run(command, cwd=self.args.project_root, env=environment,
                                  stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
            record["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{label} exited {result.returncode}; checkpoints retained")
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            # subprocess.run cleans up its own child only; no external PID operations.
            raise
        finally:
            record["finished_at"] = self.now().isoformat()
            self.save()
            if guard_core:
                self.verify_core_sources()

    def gpu_ready(self):
        if self.remaining() <= 0:
            return False
        result = self.run(
            ["nvidia-smi", f"--id={self.args.gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=min(10, self.remaining()), check=False,
        )
        if result.returncode or not result.stdout.strip().isdigit():
            raise RuntimeError(f"GPU{self.args.gpu} free-memory query failed")
        return int(result.stdout.strip()) >= self.args.minimum_free_mib

    def wait_for_gpu(self, *, new_trial):
        minimum = self.args.minimum_start_seconds if new_trial else 60
        while self.remaining() >= minimum:
            if self.gpu_ready() and self.remaining() >= minimum:
                return True
            self.transition("waiting_gpu", f"training/evaluation stays on GPU{self.args.gpu}; GPU1 is reserved")
            self.pause()
        self.transition("stopped_insufficient_time", "No more child work within the budget")
        return False

    def verify_checkpoint(self, state, *, learning_rate, steps, batch_size=None):
        batch_size = self.args.batch_size if batch_size is None else batch_size
        config = state["training_config"]
        expected = {"tokens_per_class": 4, "batch_size": batch_size, "amp": True, "epochs": 2,
                    "seed": 123, "learning_rate": learning_rate,
                    "data_root": str(self.args.data_root), "base_checkpoint": str(self.args.base_checkpoint)}
        expected.update({f"{name}_weight": 1.0 for name in LOSS_NAMES})
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint training configuration differs from fixed pilot settings")
        if (state.get("format") != "sam3-learnable-class-tokens-v2"
                or state.get("class_names") != ["left_hand", "right_hand"]
                or state.get("next_step") != steps
                or state["annotation_summary"]["sha256"] != self.state["train_annotations_sha256"]
                or state.get("epoch_order") != self.order):
            raise ValueError("Checkpoint steps/data/order/class contract mismatch")
        token_hash(state["class_tokens"])
        return token_hash(state["initial_class_tokens"])

    def verify_formal(self):
        summary_path = self.snapshot(self.args.formal_summary, "formal_summary.json")
        checkpoint_path = self.snapshot(self.args.formal_checkpoint, "formal_complete.pt")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_steps = 2 * ((self.dataset_size + 1) // 2)
        if (summary.get("epochs") != 2 or summary.get("steps") != expected_steps
                or summary.get("samples_seen") != 2 * self.dataset_size
                or summary.get("dataset_size") != self.dataset_size
                or summary.get("batch_size") != 2 or summary.get("amp") is not True
                or summary["annotation_summary"]["sha256"] != self.state["train_annotations_sha256"]):
            raise ValueError("Formal summary does not prove two complete epochs on this dataset")
        state = load_checkpoint(checkpoint_path)
        self.initial_hash = self.verify_checkpoint(state, learning_rate=0.01, steps=expected_steps, batch_size=2)
        self.state["formal_initial_class_tokens_sha256"] = self.initial_hash
        self.state["formal_verified_complete_steps"] = expected_steps
        self.save()

    def verify_early_reference(self):
        path = self.snapshot(self.args.initial_reference_checkpoint, "initial_reference.pt")
        state = load_checkpoint(path)
        steps = state.get("next_step")
        if type(steps) is not int or not 0 < steps <= 2 * ((self.dataset_size + 1) // 2):
            raise ValueError("Early reference checkpoint has invalid training progress")
        self.initial_hash = self.verify_checkpoint(state, learning_rate=0.01, steps=steps, batch_size=2)
        self.state["formal_initial_class_tokens_sha256"] = self.initial_hash
        self.state["early_start_authorized"] = True
        self.state["initial_reference_steps"] = steps
        self.save()

    def execute(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("snapshots", "logs", "trials"):
            (self.args.output_dir / name).mkdir()
        os.environ["OMP_NUM_THREADS"] = "2"
        self.transition("created")
        try:
            return self._execute()
        except Exception as error:
            self.transition("failed", f"{type(error).__name__}: {error}")
            return 1

    def _execute(self):
        core_sources = core_source_hashes(self.args.project_root)
        if not core_sources:
            raise FileNotFoundError("No sam3 Python package sources found in project-root")
        self.state.update(core_python_sources=core_sources,
                          core_python_sources_sha256=object_hash(core_sources),
                          comparison_valid=True)
        self.save()
        train_annotation = self.args.data_root / "annotations.json"
        val_annotation = self.args.val_root / "annotations.json"
        train = json.loads(train_annotation.read_text(encoding="utf-8"))
        val = json.loads(val_annotation.read_text(encoding="utf-8"))
        if train.get("info", {}).get("split") != "train" or val.get("info", {}).get("split") != "val":
            raise ValueError("Pilot requires explicit train and val COCO splits")
        self.dataset_size = len(train["images"])
        if self.dataset_size <= PILOT_SAMPLES or not val["images"]:
            raise ValueError("Pilot requires >2000 train images and nonempty full val")
        self.order = expected_order(self.dataset_size)
        sorted_image_ids = sorted(image["id"] for image in train["images"])
        if (len(set(sorted_image_ids)) != self.dataset_size
                or any(type(image_id) is not int for image_id in sorted_image_ids)):
            raise ValueError("Train COCO must contain unique integer image IDs")
        self.observed_ids_hash = hashlib.sha256("".join(
            f"{sorted_image_ids[index]}\n" for index in self.order[:PILOT_SAMPLES]
        ).encode("utf-8")).hexdigest()
        self.state.update(train_annotations_sha256=file_hash(train_annotation),
                          val_annotations_sha256=file_hash(val_annotation),
                          val_images=len(val["images"]), dataset_size=self.dataset_size,
                          epoch_order_sha256=object_hash(self.order),
                          sample_prefix_sha256=object_hash(self.order[:PILOT_SAMPLES]),
                          expected_observed_image_ids_sha256=self.observed_ids_hash)
        self.snapshot(Path(__file__).resolve(), "run_token_lr_pilot.py")
        scripts = {}
        for name in ("train_learnable_tokens.py", "evaluate_bilateral_tokens.py",
                     "render_separated_masks.py", "calibrate_bilateral_thresholds.py"):
            scripts[name] = self.snapshot(self.args.project_root / "scripts" / name, name)
        if self.args.start_before_formal:
            self.verify_early_reference()
        else:
            while not (self.args.formal_summary.is_file() and self.args.formal_checkpoint.is_file()):
                if self.remaining() < self.args.minimum_start_seconds:
                    self.transition("stopped_insufficient_time", "Formal training not ready before pilot cutoff")
                    return 2
                self.transition("waiting_formal_training", "Require matching complete checkpoint AND actual two-epoch summary")
                self.pause()
            self.verify_formal()
        self.state["base_checkpoint_sha256"] = file_hash(self.args.base_checkpoint)
        self.save()
        summaries = []
        for learning_rate in self.args.learning_rates:
            if not self.wait_for_gpu(new_trial=True):
                return 2
            if (file_hash(train_annotation) != self.state["train_annotations_sha256"]
                    or file_hash(val_annotation) != self.state["val_annotations_sha256"]
                    or file_hash(self.args.base_checkpoint) != self.state["base_checkpoint_sha256"]):
                raise RuntimeError("Train/val/base snapshot changed during pilot")
            # Base hashing can take time on shared storage; refresh availability.
            if not self.wait_for_gpu(new_trial=True):
                return 2
            label = f"lr-{learning_rate:g}-step{self.args.max_steps}"
            trial_dir = self.args.output_dir / "trials" / label
            trial_dir.mkdir()
            train_dir, eval_dir = trial_dir / "train", trial_dir / "val"
            trial = {"label": label, "learning_rate": learning_rate, "requested_steps": self.args.max_steps,
                     "planned_epochs": 2, "status": "training"}
            self.state["trials"].append(trial)
            command = [self.args.python, "-u", str(scripts["train_learnable_tokens.py"]),
                       "--data-root", str(self.args.data_root), "--base-checkpoint", str(self.args.base_checkpoint),
                       "--output-dir", str(train_dir), "--tokens-per-class", "4", "--batch-size", str(self.args.batch_size),
                       "--amp", "--epochs", "2", "--seed", "123", "--learning-rate", str(learning_rate),
                       "--expected-initial-token-sha256", self.initial_hash,
                       "--max-steps", str(self.args.max_steps), "--save-every", "100", "--log-every", "50",
                       "--gpu-memory-fraction", str(self.args.train_memory_fraction)]
            for loss in LOSS_NAMES:
                command.extend([f"--{loss}-weight", "1.0"])
            self.child(f"{label}-train", command, self.args.train_timeout_seconds, guard_core=True)
            summary_path = train_dir / "k4_epoch2_summary.json"
            training = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint = Path(training["final_checkpoint"]).resolve()
            if train_dir not in checkpoint.parents:
                raise ValueError("Training checkpoint escaped its new trial directory")
            if (training.get("steps") != self.args.max_steps or training.get("samples_seen") != PILOT_SAMPLES
                    or training.get("epochs") != 2 or training.get("dataset_size") != self.dataset_size
                    or training.get("epochs_planned") != 2 or training.get("epochs_completed") != 0
                    or training.get("full_training_complete") is not False):
                raise ValueError("Short training did not complete the identical requested sample budget")
            state = load_checkpoint(checkpoint)
            initial_hash = self.verify_checkpoint(state, learning_rate=learning_rate, steps=self.args.max_steps)
            if initial_hash != self.initial_hash:
                raise ValueError("Initial class tokens differ across formal/pilot runs; single-factor comparison invalid")
            if training.get("initial_class_tokens_sha256") != self.initial_hash:
                raise ValueError("Training summary initial-token hash differs from reference")
            observed = {
                "format": "sam3-training-observed-identities-v1",
                "scope": "current_process_completed_steps_only",
                "start_step": 0, "end_step": self.args.max_steps,
                "samples": PILOT_SAMPLES, "queries": 2 * PILOT_SAMPLES,
                "image_ids_sha256": self.observed_ids_hash,
                "image_ids_hash_encoding": "decimal_id_newline_utf8",
            }
            for source in (state, training):
                if source.get("observed_identity") != observed:
                    raise ValueError("Observed training image/query provenance differs from the requested sample prefix")
            trial.update(status="trained", actual_steps=self.args.max_steps, actual_samples=PILOT_SAMPLES,
                         observed_identity=observed,
                         completed_full_epochs=0, initial_class_tokens_sha256=initial_hash,
                         epoch_order_sha256=object_hash(state["epoch_order"]),
                         sample_prefix_sha256=object_hash(state["epoch_order"][:PILOT_SAMPLES]),
                         checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                         training_summary=str(summary_path), training_summary_sha256=file_hash(summary_path))
            self.save()
            if not self.wait_for_gpu(new_trial=False):
                return 2
            command = [self.args.python, "-u", str(scripts["evaluate_bilateral_tokens.py"]),
                       "--data-root", str(self.args.val_root), "--base-checkpoint", str(self.args.base_checkpoint),
                       "--learned-checkpoint", f"{label}={checkpoint}", "--output-dir", str(eval_dir),
                       "--unified-root", str(self.args.unified_root), "--batch-size", "1",
                       "--amp", "--gpu-memory-fraction", str(self.args.eval_memory_fraction), "--samples-per-group", "0",
                       "--detection-threshold", "0.5", "--mask-threshold", "0.5",
                       "--render-count-per-group", "2", "--visual-style", "separate"]
            self.child(f"{label}-eval", command, 2400, guard_core=True)
            eval_summary_path = eval_dir / "summary.json"
            evaluation = json.loads(eval_summary_path.read_text(encoding="utf-8"))
            if (evaluation.get("evaluated_images") != len(val["images"])
                    or sorted(evaluation.get("evaluated_dataset_indices", [])) != list(range(len(val["images"])))
                    or evaluation.get("annotations_sha256") != self.state["val_annotations_sha256"]
                    or set(evaluation.get("models", {})) != {label}):
                raise ValueError("Evaluation did not cover exactly the full shared val set/model")
            trial.update(status="evaluated", evaluation_summary=str(eval_summary_path),
                         evaluation_summary_sha256=file_hash(eval_summary_path))
            summaries.append(eval_summary_path)
            self.transition("trial_complete", label)
        calibration_path = self.args.output_dir / "lr_calibration.json"
        command = [self.args.python, str(scripts["calibrate_bilateral_thresholds.py"])]
        for summary_path in summaries:
            command.extend(["--summary", str(summary_path)])
        command.extend(["--output", str(calibration_path)])
        self.child("calibration", command, 120)
        if not calibration_path.is_file():
            raise RuntimeError("Calibration process did not publish its output")
        self.state["artifacts"]["calibration"] = {"path": str(calibration_path), "sha256": file_hash(calibration_path)}
        self.transition("complete", f"{len(self.state['trials'])} same-prefix, same-initialization 2000-sample LR trial(s); not two-epoch completions")
        return 0


def main():
    raise SystemExit(Pilot(parse_args()).execute())


if __name__ == "__main__":
    main()

"""Bounded two-GPU boundary pilot followed by full spatial validation.

Run in a caller-owned tmux session. Preserve original experiments, execute
package entry points, checkpoint every 100 steps, and never signal other jobs.
This queue does not require Windows or an active assistant connection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from scripts import run_nakehand_semantic_ablation as runner
from scripts import train_nakehand_prompt_ablation as training
from scripts import evaluate_hand_boundary_diagnostics as diagnostic


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "old-unconstrained",
                 "old-anchored", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--deadline", required=True, help="Timezone-aware cutoff, at most nine hours ahead")
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or not 120 < (args.deadline - datetime.now(timezone.utc)).total_seconds() <= 9 * 3600:
        parser.error("Deadline must be timezone-aware and 120 seconds to nine hours ahead")
    if len(set(args.gpus)) != 2 or any(gpu not in (0, 1, 2, 3) for gpu in args.gpus):
        parser.error("Require two distinct authorized GPU indices")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists() or any(args.output_dir == root or root in args.output_dir.parents
                                     for root in (args.project_root, args.data_root)):
        parser.error("Output must be a new directory outside source code and training data")
    return args


def checkpoint_result(output, expected_steps, expected_weight):
    import torch

    summary = json.loads((output / "summary.json").read_text())
    path = Path(summary["final_checkpoint"]).resolve()
    if path.parent != output.resolve() or runner.file_hash(path) != summary["final_checkpoint_sha256"]:
        raise ValueError("Trial checkpoint path/fingerprint differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    info = training.validate_checkpoint_schema(state, minimum_samples=expected_steps)
    if (info["completed_steps"] != expected_steps or info["boundary_weight"] != expected_weight
            or summary.get("status") != "completed_requested_steps"
            or summary.get("progress") != state["progress"]):
        raise ValueError("Actual successful steps/weight differ from requested stage")
    return path, state


def compare_boundary_trials(first, second):
    diagnostic.compare_checkpoint_inputs({"weight0": first, "weight4": second})
    left, right = dict(first["training_config"]), dict(second["training_config"])
    if left.pop("boundary_weight") != 0 or right.pop("boundary_weight") != 4 or left != right:
        raise ValueError("Boundary trials must differ only in boundary_weight")
    if (first["progress"] != second["progress"] or first["next_step"] != 2000
            or first["observed_image_ids"] != second["observed_image_ids"]):
        raise ValueError("Boundary trials did not consume the same 2000-image prefix")
    return {"only_training_factor": "boundary_weight: 0 versus 4", "actual_steps_each": 2000,
            "same_initial_cache": True, "same_ordered_training_images": True}


def validate_full_spatial_result(output):
    value = json.loads((output / "summary.json").read_text())
    if (value.get("status") != "completed" or value.get("full_val_evaluated") is not True
            or value.get("evaluated_images") != 3449 or value.get("all_sources_unchanged") is not True
            or value.get("completed_images_per_model") != {name: 3449 for name in diagnostic.LABELS}):
        raise ValueError("Spatial evaluation did not actually complete three full validation passes")
    for label in diagnostic.LABELS:
        entry = value["record_files"][label]
        path = Path(entry["path"]).resolve()
        if path.parent != (output / "records").resolve() or runner.file_hash(path) != entry["sha256"]:
            raise ValueError("Final records changed or escaped the evaluation output")
        rows = json.loads(path.read_text())
        expected = {(image_id, side) for image_id in value["evaluated_image_ids"]
                    for side in training.cached.CLASS_NAMES}
        actual = [(row["image_id"], row["prompt_key"]) for row in rows]
        if len(actual) != 6898 or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Final spatial records have missing/duplicate actual image-query pairs")
    report = output / "REPORT.md"
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError("Completed spatial result has no nonempty report")
    return {"summary": str(output / "summary.json"), "report": str(report), "report_sha256": runner.file_hash(report),
            "summary_sha256": runner.file_hash(output / "summary.json"), "actual_images_per_variant": 3449}


class BoundaryRunner(runner.AblationRunner):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.state.update(format="sam3-nakehand-boundary-supervisor-v1", retry_policy="none",
                          authorization="User authorized controlled overnight experiments through morning",
                          geometry_training=False, memory_training=False, realsense_used=False,
                          smoke_steps=20, checkpoint_interval=100, boundary_weights=[0., 4.],
                          maximum_parallel_gpu_jobs=2, gpu_memory_fraction=.25,
                          stage_caps_seconds={"smoke": 600, "continuation": 3600, "full_spatial_validation": 7200},
                          package_execution=True)

    def snapshot(self):
        paths = {path.resolve(): "core" for path in (self.args.project_root / "sam3").rglob("*.py")}
        scripts = set(runner.SCRIPTS) | {
            "run_nakehand_boundary_ablation.py", "train_nakehand_prompt_ablation.py",
            "evaluate_hand_boundary_diagnostics.py", "analyze_hand_segmentation_failures.py"}
        for name in sorted(scripts):
            source = self.args.project_root / "scripts" / name
            target = self.args.output_dir / "snapshots" / name
            digest = runner.file_hash(source)
            shutil.copy2(source, target)
            if runner.file_hash(target) != digest or runner.file_hash(source) != digest:
                raise RuntimeError("Script changed during backup")
            paths[source.resolve()], paths[target.resolve()] = "script", "snapshot"
        inputs = [self.args.base_checkpoint, self.args.initial_cache, self.args.old_unconstrained,
                  self.args.old_anchored, self.args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"]
        inputs += [self.args.data_root / name for name in ("READY.json", "manifest.json", "frozen-plan.json")]
        for split in ("train", "val"):
            inputs += [self.args.data_root / split / name for name in ("READY.json", "manifest.json", "annotations.json")]
        for path in inputs:
            paths[path.resolve()] = "input"
        self.state["sources"] = [{"path": str(path), "kind": kind, "sha256": runner.file_hash(path)}
                                 for path, kind in sorted(paths.items())]
        self.update(None, status="ready_to_train")

    def training_command(self, output, weight, steps, resume=None):
        command = [self.args.python, "-u", "-m", "scripts.train_nakehand_prompt_ablation",
                   "--data-root", str(self.args.data_root / "train"),
                   "--base-checkpoint", str(self.args.base_checkpoint), "--initial-cache", str(self.args.initial_cache),
                   "--boundary-weight", str(weight), "--project-root", str(self.args.project_root),
                   "--max-steps", str(steps), "--gpu-memory-fraction", "0.25", "--output-dir", str(output)]
        if resume is not None:
            command.extend(("--resume", str(resume)))
        return command

    def train_one(self, label, weight, gpu):
        smoke = self.args.output_dir / f"{label}-smoke20"
        self.child(f"{label}-smoke20", self.training_command(smoke, weight, 20), gpu=gpu, cap=600)
        resume, _ = checkpoint_result(smoke, 20, weight)
        output = self.args.output_dir / f"{label}-train2000"
        self.child(f"{label}-train2000", self.training_command(output, weight, 2000, resume), gpu=gpu, cap=3600)
        path, state = checkpoint_result(output, 2000, weight)
        self.update(label, checkpoint=str(path), checkpoint_sha256=runner.file_hash(path), progress=state["progress"])
        return path, state

    def paired(self, calls):
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(function, *arguments): label for label, function, arguments in calls}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException as error:
                self.stop_event.set()
                if self.args.output_dir.exists():
                    self.update(None, status="stopping_after_failure", error=f"{type(error).__name__}: {error}")
                for future in futures:
                    future.cancel()
                raise
        return results

    def spatial(self, label, first, second, gpu):
        output = self.args.output_dir / label
        command = [self.args.python, "-u", "-m", "scripts.evaluate_hand_boundary_diagnostics",
                   "--data-root", str(self.args.data_root / "val"), "--base-checkpoint", str(self.args.base_checkpoint),
                   "--baseline-checkpoint", str(first), "--cp0-checkpoint", str(first), "--cp1-checkpoint", str(second),
                   "--variant", "all", "--minimum-samples-seen", "2000", "--render-count", "12",
                   "--gpu-memory-fraction", "0.25", "--project-root", str(self.args.project_root),
                   "--output-dir", str(output)]
        self.child(label, command, gpu=gpu, cap=7200)
        result = validate_full_spatial_result(output)
        self.update(label, verified_result=result)
        return result

    def execute(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("logs", "snapshots"):
            (self.args.output_dir / name).mkdir()
        self.update(None, status="created")
        try:
            self.snapshot()
            trained = self.paired([
                ("weight0", self.train_one, ("weight0", 0., self.args.gpus[0])),
                ("weight4", self.train_one, ("weight4", 4., self.args.gpus[1]))])
            self.update(None, status="training_complete", training_comparison=compare_boundary_trials(
                trained["weight0"][1], trained["weight4"][1]))
            reports = self.paired([
                ("boundary-comparison", self.spatial, ("boundary-comparison", trained["weight0"][0],
                                                      trained["weight4"][0], self.args.gpus[0])),
                ("semantic-comparison", self.spatial, ("semantic-comparison", self.args.old_unconstrained,
                                                      self.args.old_anchored, self.args.gpus[1]))])
            self.update(None, status="complete", verified_reports=reports)
            return 0
        except BaseException as error:
            self.stop_event.set()
            self.update(None, status="failed_or_stopped", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    raise SystemExit(BoundaryRunner(parse_args()).execute())

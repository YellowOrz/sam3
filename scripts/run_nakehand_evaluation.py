#!/usr/bin/env python3
"""One bounded, token-only external test; never train or tune on nakehand.

Run in tmux. Wait for published data and the original complete second epoch,
snapshot code/checkpoint, check shared GPU headroom, evaluate three fixed models.
Failures stop, existing outputs are not overwritten, deadline cannot be extended.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

try:
    from finish_bilateral_validation import Supervisor, AUTHORIZED_DEADLINE, sha256
    from run_token_lr_pilot import core_source_hashes, object_hash
except ModuleNotFoundError:
    from scripts.finish_bilateral_validation import Supervisor, AUTHORIZED_DEADLINE, sha256
    from scripts.run_token_lr_pilot import core_source_hashes, object_hash


def verify_epoch2(state, summary):
    """Old formal v2 has no epochs_completed field: verify actual work instead."""
    import torch

    size = summary.get("dataset_size")
    if type(size) is not int or size < 1:
        raise ValueError("Invalid formal dataset size")
    expected_steps = 2 * ((size + 1) // 2)
    config = state.get("training_config", {})
    tokens = state.get("class_tokens")
    if (not state.get("annotation_summary", {}).get("sha256")
            or summary.get("epochs") != 2 or summary.get("batch_size") != 2
            or summary.get("steps") != expected_steps or summary.get("samples_seen") != 2 * size
            or state.get("next_step") != expected_steps
            or config.get("epochs") != 2 or config.get("batch_size") != 2
            or state.get("format") != "sam3-learnable-class-tokens-v2"
            or state.get("class_names") != ["left_hand", "right_hand"]
            or not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (2, 4, 256)
            or not tokens.is_floating_point() or not torch.isfinite(tokens).all().item()
            or state.get("annotation_summary", {}).get("sha256") != summary.get("annotation_summary", {}).get("sha256")):
        raise ValueError("Checkpoint/summary do not prove the same complete original two-epoch run")
    return expected_steps


def verify_published_data(root):
    """READY is written last; annotations alone are not a completed publication."""
    root = Path(root)
    ready = json.loads((root / "READY.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    progress = json.loads((root / "export-in-progress.json").read_text())
    actual = sha256(root / "annotations.json")
    if (any(item.get("status") != "complete" for item in (ready, manifest, progress))
            or any(item.get("annotations_sha256") != actual for item in (ready, manifest, progress))
            or ready.get("manifest_sha256") != sha256(root / "manifest.json")
            or manifest.get("sources_unchanged") is not True):
        raise ValueError("External data publication is incomplete or its hashes changed")
    return actual


def readable_summary(path):
    # The already-running old formal trainer publishes this small JSON directly.
    # A just-created/partly-written file is pending, not a failed completed run.
    try:
        return isinstance(json.loads(Path(path).read_text()), dict)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "token-checkpoint", "formal-summary", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--minimum-free-mib", type=int, default=7500)
    parser.add_argument("--deadline", default=AUTHORIZED_DEADLINE.isoformat())
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or args.deadline > AUTHORIZED_DEADLINE:
        parser.error("deadline cannot exceed 2026-09-10 21:40 +08:00")
    if args.minimum_free_mib < 7500:
        parser.error("at least 7500 MiB free memory is required")
    args.poll_seconds, args.child_timeout_seconds = 30, 3600
    args.minimum_start_seconds = 1800
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    for protected in (args.project_root, args.data_root):
        if args.output_dir == protected or protected in args.output_dir.parents:
            parser.error("output must be outside the project and input data root")
    return args


class ExternalTest(Supervisor):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.state.update(format="sam3-nakehand-external-test-v1", comparison_limitations=[
            "SAM3-assisted reference labels; only A/B/C accepted by user, not fully manual GT.",
            "600 recording-balanced systematic frames plus separate diagnostics; not all 18498 frames.",
            "Fixed thresholds .5, no nakehand training or calibration; correlated frames are not independent people.",
        ])
        self.state.pop("epoch2_attempts")

    def check_core(self):
        try:
            actual = core_source_hashes(self.args.project_root)
        except OSError:
            self.state["comparison_valid"] = False
            self.save()
            raise
        if actual != self.state["core_python_sources"]:
            self.state["comparison_valid"] = False
            self.save()
            raise RuntimeError("Core SAM3 Python files changed during the external test")

    def execute(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("snapshots", "logs"):
            (self.args.output_dir / name).mkdir()
        self.transition("created")
        try:
            return self._execute()
        except Exception as error:
            self.transition("failed", f"{type(error).__name__}: {error}")
            return 1

    def _execute(self):
        scripts = self.args.project_root / "scripts"
        for name in ("run_nakehand_evaluation.py", "finish_bilateral_validation.py", "run_token_lr_pilot.py",
                     "evaluate_bilateral_tokens.py", "render_separated_masks.py", "evaluate_nakehand_tokens.py"):
            self.snapshot(scripts / name, name)
        self.state["core_python_sources"] = core_source_hashes(self.args.project_root)
        if not self.state["core_python_sources"]:
            raise ValueError("Missing SAM3 core source files")
        self.state["core_python_sources_sha256"] = object_hash(self.state["core_python_sources"])
        self.state["comparison_valid"] = True
        self.save()
        annotation = self.args.data_root / "annotations.json"
        ready = (annotation, self.args.data_root / "READY.json", self.args.token_checkpoint, self.args.formal_summary)
        while not (all(path.is_file() for path in ready) and readable_summary(self.args.formal_summary)):
            if self.remaining() < self.args.minimum_start_seconds:
                self.transition("stopped_insufficient_time", "Inputs not ready before the start cutoff")
                return 2
            missing = [path.name for path in ready if not path.is_file()]
            if not readable_summary(self.args.formal_summary):
                missing.append("complete formal summary JSON")
            self.transition("waiting_inputs", ", ".join(missing))
            self.pause()
        verify_published_data(self.args.data_root)
        data_path = self.snapshot(annotation, "annotations.json")
        self.snapshot(self.args.data_root / "manifest.json", "data_manifest.json")
        self.snapshot(self.args.data_root / "READY.json", "data_READY.json")
        data = json.loads(data_path.read_text())
        if (data.get("info", {}).get("split") != "external_test"
                or sum(image.get("primary_test") is True for image in data["images"]) != 600):
            raise ValueError("Expected published external_test data with exactly 600 primary frames")
        checkpoint = self.snapshot(self.args.token_checkpoint, "epoch2_complete.pt")
        formal = self.snapshot(self.args.formal_summary, "formal_summary.json")
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.state["verified_epoch2_steps"] = verify_epoch2(state, json.loads(formal.read_text()))
        if Path(state["training_config"]["base_checkpoint"]).resolve() != self.args.base_checkpoint:
            raise ValueError("Formal token base checkpoint differs from the requested base")
        self.state["base_checkpoint_sha256"] = sha256(self.args.base_checkpoint)
        self.save()
        while self.remaining() >= self.args.minimum_start_seconds:
            if self.gpu_free() >= self.args.minimum_free_mib:
                break
            self.transition("waiting_gpu", f"GPU {self.args.gpu}, require {self.args.minimum_free_mib} MiB")
            self.pause()
        else:
            self.transition("stopped_insufficient_time", "Insufficient safe runtime/GPU headroom")
            return 2
        if sha256(annotation) != self.state["artifacts"]["annotations.json"]["sha256"]:
            raise RuntimeError("External annotations changed after snapshot")
        self.check_core()
        result_dir = self.args.output_dir / "results"
        command = [self.args.python, "-u", str(self.args.output_dir / "snapshots/evaluate_nakehand_tokens.py"),
                   "--data-root", str(self.args.data_root), "--base-checkpoint", str(self.args.base_checkpoint),
                   "--learned-checkpoint", f"epoch2={checkpoint}", "--include-ve",
                   "--output-dir", str(result_dir), "--batch-size", "1", "--amp", "--gpu-memory-fraction", "0.25",
                   "--render-per-recording", "4"]
        try:
            self.child("nakehand_three_models", command, maximum_seconds=3600)
        finally:
            self.check_core()
        summary_path = result_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (summary.get("evaluated_images") != len(data["images"])
                or sorted(summary.get("evaluated_dataset_indices", [])) != list(range(len(data["images"])))
                or set(summary.get("models", {})) != {"epoch2", "ve-underscore", "ve-natural"}
                or summary.get("annotations_sha256") != self.state["artifacts"]["annotations.json"]["sha256"]
                or summary.get("base_checkpoint_sha256") != self.state["base_checkpoint_sha256"]
                or sha256(self.args.base_checkpoint) != self.state["base_checkpoint_sha256"]
                or sha256(annotation) != self.state["artifacts"]["annotations.json"]["sha256"]):
            raise ValueError("Incomplete or changed model/data results")
        self.state["artifacts"]["evaluation_summary"] = {"path": str(summary_path), "sha256": sha256(summary_path)}
        self.transition("complete", "600 fixed primary frames plus diagnostics, three fixed models, no training/tuning")
        return 0


def main():
    raise SystemExit(ExternalTest(parse_args()).execute())


if __name__ == "__main__":
    main()

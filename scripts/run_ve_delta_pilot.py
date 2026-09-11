#!/usr/bin/env python3
"""One deadline-bounded semantic-delta continuation, validation and CPU report.

Launch in caller-owned tmux. Child failure stops without retry. subprocess.run
terminates only its own timed-out child immediately (zero-second kill grace).
No nakehand training, MANO updates, other-user process signalling or budget
extension. Existing output directories are never reused.
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


CHECKPOINT_FORMAT = "sam3-ve-initialized-delta-training-v1"
SCRIPT_NAMES = (
    "run_ve_delta_pilot.py", "finish_bilateral_validation.py", "run_token_lr_pilot.py",
    "train_ve_initialized_tokens.py", "evaluate_ve_initialized_tokens.py",
    "cached_ve_text_features.py", "train_learnable_tokens.py", "evaluate_bilateral_tokens.py",
    "evaluate_nakehand_tokens.py", "render_separated_masks.py", "report_bilateral_evaluation.py",
)


def verify_completed_pilot(summary, checkpoint):
    if summary.get("format") != CHECKPOINT_FORMAT or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Not a semantic VE-delta pilot")
    expected = {"completed_steps": 2000, "samples_seen": 2000,
                "planned_steps": 2000, "planned_samples": 2000, "pilot_complete": True}
    for document in (summary, checkpoint):
        if any(document.get("progress", {}).get(key) != value for key, value in expected.items()):
            raise ValueError("Full validation requires exactly 2000 successful pilot steps/samples")
    if checkpoint.get("next_step") != 2000 or len(checkpoint.get("observed_image_ids", [])) != 2000:
        raise ValueError("Pilot checkpoint does not record 2000 successful observations")
    if summary.get("progress") != checkpoint.get("progress"):
        raise ValueError("Training summary and checkpoint progress disagree")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resume", "initial-cache", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=Path("/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/train"))
    parser.add_argument("--val-root", type=Path, default=Path("/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/val"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--minimum-free-mib", type=int, default=7500)
    parser.add_argument("--deadline", default=AUTHORIZED_DEADLINE.isoformat())
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or args.deadline > AUTHORIZED_DEADLINE:
        parser.error("Deadline cannot exceed 2026-09-10 21:40 +08:00")
    if args.minimum_free_mib < 7500:
        parser.error("Shared GPU requires at least 7500 MiB free")
    args.poll_seconds, args.child_timeout_seconds, args.minimum_start_seconds = 30, 3600, 120
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists():
        parser.error("Output must be a new directory")
    for protected in (args.project_root, args.train_root, args.val_root):
        if args.output_dir == protected or protected in args.output_dir.parents:
            parser.error("Output must be outside the project and input dataset splits")
    return args


class DeltaPilot(Supervisor):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.state.update(format="sam3-ve-delta-pilot-supervisor-v1", comparison_limitations=[
            "Exactly 2000 training images is a short pilot, not two completed epochs.",
            "Only DexYCB train/val; no nakehand tuning and no MANO training.",
            "Timeout kills only this supervisor's own child; zero-second kill grace.",
        ])
        self.state.pop("epoch2_attempts")

    def check_frozen_sources(self):
        if core_source_hashes(self.args.project_root) != self.state["core_sources"]:
            raise RuntimeError("SAM3 core changed during the controlled pilot")
        for name in SCRIPT_NAMES:
            artifact = self.state["artifacts"][name]
            if sha256(artifact["source"]) != artifact["sha256"]:
                raise RuntimeError(f"Controlled script changed: {name}")
        for source in self.state["inputs"]:
            if sha256(source["path"]) != source["sha256"]:
                raise RuntimeError(f"Controlled input changed: {source['path']}")

    def stage(self, label, command, seconds, *, needs_gpu):
        while self.remaining() >= self.args.minimum_start_seconds:
            if not needs_gpu or self.gpu_free() >= self.args.minimum_free_mib:
                break
            self.transition(f"waiting_gpu_{label}", f"GPU {self.args.gpu}: need {self.args.minimum_free_mib} MiB")
            self.pause()
        else:
            self.transition("stopped_insufficient_time", f"Not starting {label} near deadline")
            return False
        self.check_frozen_sources()
        # Hashing a large base file can consume time; recheck the minimum window.
        if self.remaining() < self.args.minimum_start_seconds:
            self.transition("stopped_insufficient_time", f"Source verification exhausted {label} start window")
            return False
        try:
            self.child(label, command, maximum_seconds=seconds, reserve_seconds=30)
        finally:
            self.check_frozen_sources()
        return True

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
        import torch

        for name in SCRIPT_NAMES:
            self.snapshot(self.args.project_root / "scripts" / name, name)
        self.state["core_sources"] = core_source_hashes(self.args.project_root)
        if not self.state["core_sources"]:
            raise ValueError("Missing SAM3 core sources")
        self.state["core_sources_sha256"] = object_hash(self.state["core_sources"])
        inputs = (self.args.resume, self.args.initial_cache, self.args.base_checkpoint,
                  self.args.train_root / "annotations.json", self.args.val_root / "annotations.json",
                  self.args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        self.state["inputs"] = [{"path": str(path), "sha256": sha256(path)} for path in inputs]
        self.save()
        resume = self.snapshot(self.args.resume, "resume_step20.pt")
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (state.get("format") != CHECKPOINT_FORMAT or state.get("next_step") != 20
                or state.get("progress", {}).get("samples_seen") != 20
                or state.get("progress", {}).get("pilot_complete") is not False):
            raise ValueError("This continuation requires the explicitly verified 20-step semantic delta smoke")
        # The trainer checks the full cache/config/optimizer/RNG/observed-prefix contract.
        train_dir = self.args.output_dir / "training"
        scripts = self.args.output_dir / "snapshots"
        train_command = [self.args.python, "-u", str(scripts / "train_ve_initialized_tokens.py"),
                         "--data-root", str(self.args.train_root), "--base-checkpoint", str(self.args.base_checkpoint),
                         "--initial-cache", str(self.args.initial_cache), "--resume", str(resume),
                         "--project-root", str(self.args.project_root), "--max-steps", "2000",
                         "--gpu-memory-fraction", "0.25", "--output-dir", str(train_dir)]
        if not self.stage("train_to_2000", train_command, 2700, needs_gpu=True):
            return 2
        training_summary = json.loads((train_dir / "summary.json").read_text())
        final_checkpoint = Path(training_summary["final_checkpoint"]).resolve()
        if final_checkpoint.parent != train_dir.resolve():
            raise ValueError("Training final checkpoint is outside this new run")
        if sha256(final_checkpoint) != training_summary.get("final_checkpoint_sha256"):
            raise ValueError("Final training checkpoint hash differs from its summary")
        verify_completed_pilot(training_summary, torch.load(final_checkpoint, map_location="cpu", weights_only=True))
        self.state["artifacts"]["completed_training_checkpoint"] = {
            "path": str(final_checkpoint), "sha256": sha256(final_checkpoint)}
        self.transition("training_complete", "2000 actual samples; accuracy still unevaluated")
        eval_dir = self.args.output_dir / "validation"
        evaluation_command = [self.args.python, "-u", str(scripts / "evaluate_ve_initialized_tokens.py"),
                              "--data-root", str(self.args.val_root), "--base-checkpoint", str(self.args.base_checkpoint),
                              "--delta-checkpoint", str(final_checkpoint), "--project-root", str(self.args.project_root),
                              "--output-dir", str(eval_dir), "--minimum-samples-seen", "2000",
                              "--gpu-memory-fraction", "0.25", "--deadline", self.args.deadline.isoformat()]
        if not self.stage("full_val_frozen_and_delta", evaluation_command, 3600, needs_gpu=True):
            return 2
        summary_path = eval_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (summary.get("status") != "completed" or summary.get("full_val_evaluated") is not True
                or summary.get("evaluated_images") != 2909
                or summary.get("diagnostic_training_checkpoint") is not False
                or set(summary.get("models", {})) != {"ve-frozen-cache", "ve-delta"}
                or summary.get("delta_checkpoint_sha256") != sha256(final_checkpoint)):
            raise ValueError("Validation did not finish the full two-mode 2909-image comparison")
        self.state["artifacts"]["evaluation_summary"] = {"path": str(summary_path), "sha256": sha256(summary_path)}
        report = self.args.output_dir / "validation_report.md"
        command = [self.args.python, str(scripts / "report_bilateral_evaluation.py"),
                   "--summary", str(summary_path), "--output", str(report)]
        if not self.stage("cpu_report", command, 120, needs_gpu=False):
            return 2
        self.state["artifacts"]["report"] = {"path": str(report), "sha256": sha256(report)}
        self.transition("complete", "2000-sample semantic pilot and both full-val modes completed; report saved")
        return 0


def main():
    raise SystemExit(DeltaPilot(parse_args()).execute())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finish token validation once, within a fixed deadline; never launch training.

Run inside tmux on the server. Wait for epoch1's atomic summary, make its report,
then wait for epoch2's complete checkpoint and enough free GPU memory. Evaluate
only epoch2 learned tokens (no repeated VE), then combine reports. Existing jobs
are never inspected, signalled, or stopped. Any child failure stops this run;
reusing an output directory is forbidden, so there is no automatic restart.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


AUTHORIZED_DEADLINE = datetime.fromisoformat("2026-09-10T21:40:00+08:00")
LIMITATION = (
    "旧 epoch1 summary 仅记录基础权重路径，未记录其 SHA256 和 AMP 配置；"
    "路径、样本及阈值匹配不能严格证明两轮数值配置一致。"
    "本次 epoch2 命令与文件哈希已记录，比较结论应保留这一限制。"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch1-summary", type=Path, required=True)
    parser.add_argument("--epoch2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="COCO val directory")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--unified-root", type=Path, default=Path("/data/xuzhefeng/Datasets/uni-hoi-dataset"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--minimum-free-mib", type=int, default=7500)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-count-per-group", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--minimum-start-seconds", type=float, default=1800)
    parser.add_argument("--child-timeout-seconds", type=float, default=7200)
    parser.add_argument("--deadline", default=AUTHORIZED_DEADLINE.isoformat())
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or args.deadline > AUTHORIZED_DEADLINE:
        parser.error("deadline must be timezone-aware and no later than 2026-09-10T21:40:00+08:00")
    if not 0 < args.poll_seconds <= 30:
        parser.error("poll-seconds must be within (0, 30]")
    if (not 0 < args.child_timeout_seconds <= 7200
            or not math.isfinite(args.minimum_start_seconds)
            or args.minimum_start_seconds < 60):
        parser.error("child timeout must be within (0,7200]; minimum start must be >=60 seconds")
    if args.gpu < 0 or args.minimum_free_mib < 7500:
        parser.error("GPU index must be nonnegative and minimum-free-mib >=7500")
    if not 0 < args.gpu_memory_fraction <= 0.25 or args.batch_size != 1:
        parser.error("This shared-GPU run requires memory fraction <=0.25 and batch size 1")
    if args.render_count_per_group < 0:
        parser.error("render-count-per-group must be nonnegative")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir == args.project_root or args.project_root in args.output_dir.parents:
        parser.error("Output must be outside the Git project")
    if args.output_dir == args.unified_root or args.unified_root in args.output_dir.parents:
        parser.error("Output must be outside the shared unified dataset")
    return args


class Supervisor:
    def __init__(self, args, *, now=None, sleep=None, run=None):
        self.args = args
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or time.sleep
        self.run = run or subprocess.run
        self.state = {
            "format": "sam3-token-validation-supervisor-v1",
            "status": "created", "created_at": self.now().isoformat(),
            "config": {key: value.isoformat() if isinstance(value, datetime)
                       else str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
            "commands": [], "artifacts": {}, "comparison_limitations": [LIMITATION],
            "epoch2_attempts": 0,
        }
        self.last_transition = None

    def remaining(self):
        return (self.args.deadline - self.now()).total_seconds()

    def save(self):
        temporary = self.args.output_dir / ".state.json.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.args.output_dir / "state.json")

    def transition(self, status, detail=""):
        if (status, detail) == self.last_transition:
            return
        self.last_transition = (status, detail)
        self.state.update(status=status, detail=detail, updated_at=self.now().isoformat())
        self.save()
        with (self.args.output_dir / "supervisor.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{self.now().isoformat()} {status} {detail}\n")

    def pause(self):
        self.sleep(max(0, min(self.args.poll_seconds, self.remaining())))

    def snapshot(self, source, name):
        target = self.args.output_dir / "snapshots" / name
        raw = Path(source).read_bytes()
        with target.open("xb") as stream:
            stream.write(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if sha256(source) != digest:
            raise RuntimeError(f"Source changed during snapshot: {source}")
        self.state["artifacts"][name] = {"source": str(source), "path": str(target), "sha256": digest}
        self.save()
        return target

    def child(self, label, command, *, maximum_seconds=7200, reserve_seconds=0):
        timeout = min(maximum_seconds, self.args.child_timeout_seconds, self.remaining() - reserve_seconds)
        if timeout <= 0:
            raise TimeoutError("Deadline leaves no time for a new subprocess")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(self.args.project_root), str(self.args.project_root / "scripts"),
            environment.get("PYTHONPATH"),
        )))
        log_path = self.args.output_dir / "logs" / f"{label}.log"
        record = {
            "label": label, "command": list(command), "cwd": str(self.args.project_root),
            "started_at": self.now().isoformat(), "timeout_seconds": timeout,
            "log": str(log_path), "returncode": None,
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "PYTHONPATH": environment["PYTHONPATH"],
        }
        self.state["commands"].append(record)
        self.transition(f"running_{label}")
        try:
            with log_path.open("x", encoding="utf-8") as stream:
                result = self.run(
                    command, cwd=self.args.project_root, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                )
            record["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{label} exited with code {result.returncode}; see {log_path}")
        except subprocess.TimeoutExpired:
            # subprocess.run terminates only its own child and waits for it.
            record["timed_out"] = True
            raise
        finally:
            record["finished_at"] = self.now().isoformat()
            self.save()

    def gpu_free(self):
        timeout = min(10, self.remaining())
        if timeout <= 0:
            raise TimeoutError("Deadline reached before GPU query")
        result = self.run(
            ["nvidia-smi", f"--id={self.args.gpu}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"GPU memory query failed: {result.stderr.strip()}")
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1 or not lines[0].strip().isdigit():
            raise ValueError(f"Unexpected GPU memory response: {result.stdout!r}")
        return int(lines[0].strip())

    def report(self, reporter, summaries, name):
        output = self.args.output_dir / "reports" / f"{name}.md"
        command = [self.args.python, str(reporter)]
        for summary in summaries:
            command.extend(["--summary", str(summary)])
        command.extend(["--output", str(output)])
        self.child(name, command, maximum_seconds=120)
        if not output.is_file():
            raise RuntimeError(f"Report subprocess did not create {output}")
        with output.open("a", encoding="utf-8") as stream:
            stream.write(f"\n## 配置可比性限制\n\n{LIMITATION}\n")
        self.state["artifacts"][name] = {"path": str(output), "sha256": sha256(output)}
        self.save()

    def execute(self):
        # Exclusive creation prevents accidental restart/overwrite of any attempt.
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("snapshots", "logs", "reports"):
            (self.args.output_dir / name).mkdir()
        self.transition("created")
        try:
            self.snapshot(Path(__file__).resolve(), "supervisor.py")
            return self._execute()
        except Exception as error:
            self.transition("failed", f"{type(error).__name__}: {error}")
            return 1

    def _execute(self):
        while not self.args.epoch1_summary.is_file():
            if self.remaining() <= 0:
                self.transition("stopped_deadline", "epoch1 summary did not arrive")
                return 2
            self.transition("waiting_epoch1_summary")
            self.pause()
        if self.remaining() <= 0:
            self.transition("stopped_deadline")
            return 2
        epoch1_path = self.snapshot(self.args.epoch1_summary, "epoch1_summary.json")
        reference = json.loads(epoch1_path.read_text(encoding="utf-8"))
        if Path(reference["base_checkpoint"]).resolve() != self.args.base_checkpoint:
            raise ValueError("Epoch1 base checkpoint path differs from requested base")
        if Path(reference["data_root"]).resolve() != self.args.data_root:
            raise ValueError("Epoch1 data root differs from requested data")
        if "epoch2" in reference["models"]:
            raise ValueError("Epoch1 summary already includes epoch2; refusing duplicate evaluation")
        indices = reference["evaluated_dataset_indices"]
        if (not indices or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)
                or len(indices) != reference["evaluated_images"]):
            raise ValueError("Invalid epoch1 evaluated indices/count")
        reporter = self.snapshot(self.args.project_root / "scripts/report_bilateral_evaluation.py", "reporter.py")
        evaluator = self.snapshot(self.args.project_root / "scripts/evaluate_bilateral_tokens.py", "evaluator.py")
        self.snapshot(self.args.project_root / "scripts/render_separated_masks.py", "render_separated_masks.py")
        self.report(reporter, [epoch1_path], "epoch1_report")
        self.transition("epoch1_report_complete")

        token_snapshot = None
        while True:
            if self.remaining() < self.args.minimum_start_seconds:
                self.transition("stopped_insufficient_time", "No new epoch2 evaluation near deadline")
                return 2
            if not self.args.epoch2_checkpoint.is_file():
                self.transition("waiting_epoch2_checkpoint")
            elif self.gpu_free() < self.args.minimum_free_mib:
                self.transition("waiting_gpu_memory", f"GPU {self.args.gpu} needs >= {self.args.minimum_free_mib} MiB free")
            else:
                if token_snapshot is None:
                    token_snapshot = self.snapshot(self.args.epoch2_checkpoint, "epoch2_tokens.pt")
                    self.state["base_checkpoint_sha256_at_epoch2_preparation"] = sha256(self.args.base_checkpoint)
                    self.save()
                    # Hashing the base may take time; check deadline and free memory again.
                    continue
                if sha256(self.args.data_root / "annotations.json") != reference["annotations_sha256"]:
                    raise ValueError("COCO annotations changed since epoch1 evaluation")
                if self.remaining() < self.args.minimum_start_seconds:
                    continue
                break
            self.pause()

        output = self.args.output_dir / "epoch2"
        command = [
            self.args.python, "-u", str(evaluator),
            "--data-root", str(self.args.data_root),
            "--base-checkpoint", str(self.args.base_checkpoint),
            "--learned-checkpoint", f"epoch2={token_snapshot}",
            "--output-dir", str(output), "--unified-root", str(self.args.unified_root),
            "--batch-size", str(self.args.batch_size),
            "--gpu-memory-fraction", str(self.args.gpu_memory_fraction),
            "--indices", ",".join(str(index) for index in indices),
            "--detection-threshold", str(reference["detection_threshold"]),
            "--mask-threshold", str(reference["mask_threshold"]),
            "--render-count-per-group", str(self.args.render_count_per_group),
            "--visual-style", "separate",
            "--amp" if self.args.amp else "--no-amp",
        ]
        self.state["epoch2_attempts"] = 1
        self.child("epoch2_evaluation", command, reserve_seconds=60)
        epoch2_summary = output / "summary.json"
        if not epoch2_summary.is_file():
            raise RuntimeError("Epoch2 process exited without a completed summary")
        self.state["artifacts"]["epoch2_summary"] = {"path": str(epoch2_summary), "sha256": sha256(epoch2_summary)}
        self.report(reporter, [epoch1_path, epoch2_summary], "combined_report")
        self.transition("complete", "Epoch1, epoch2 and existing VE results reported")
        return 0


def main():
    raise SystemExit(Supervisor(parse_args()).execute())


if __name__ == "__main__":
    main()

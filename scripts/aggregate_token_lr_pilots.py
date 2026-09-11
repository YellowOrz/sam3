#!/usr/bin/env python3
"""Wait for three successful LR pilot results and calibrate them on CPU only.

This does not schedule training/evaluation or touch GPU jobs. All three learning
rates must be complete, with identical initialization, sample prefix, training
budget, data/base hashes and frozen trainer/evaluator sources. A failed/stopped
run or the hard deadline ends aggregation explicitly; partial runs never count
as a successful three-way comparison. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEADLINE = datetime.fromisoformat("2026-09-10T21:40:00+08:00")
EXPECTED_RATES = {0.01, 0.003, 0.001}
SHARED_KEYS = (
    "train_annotations_sha256", "val_annotations_sha256", "base_checkpoint_sha256",
    "epoch_order_sha256", "sample_prefix_sha256", "formal_initial_class_tokens_sha256",
    "expected_observed_image_ids_sha256", "dataset_size", "val_images",
    "core_python_sources", "core_python_sources_sha256",
)
SHARED_SCRIPTS = (
    "train_learnable_tokens.py", "evaluate_bilateral_tokens.py",
    "render_separated_masks.py", "calibrate_bilateral_thresholds.py",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_inputs(run_dirs):
    loaded, progress = [], []
    for run_dir in run_dirs:
        path = Path(run_dir) / "state.json"
        if not path.is_file():
            progress.append(f"{Path(run_dir).name}:missing")
            continue
        raw = path.read_bytes()
        state = json.loads(raw)
        status = state.get("status")
        if status == "failed" or str(status).startswith("stopped"):
            raise RuntimeError(f"Pilot did not succeed: {run_dir}: {status}: {state.get('detail', '')}")
        progress.append(f"{Path(run_dir).name}:{status}")
        if status == "complete":
            loaded.append((Path(run_dir), state, raw))
    if len(loaded) != len(run_dirs):
        return None, "; ".join(progress)
    return loaded, "; ".join(progress)


def validate_completed_runs(loaded):
    if not loaded:
        raise ValueError("No completed runs")
    reference = loaded[0][1]
    summaries, seen_rates = [], set()
    fixed = {key: value for key, value in reference["fixed_training"].items() if key != "learning_rates"}
    for run_dir, state, _ in loaded:
        if state.get("format") != "sam3-token-lr-pilot-v1" or state.get("status") != "complete":
            raise ValueError("Expected an explicitly successful token LR pilot state")
        if state.get("comparison_valid") is not True:
            raise ValueError("Pilot declared its comparison invalid")
        core_map = state["core_python_sources"]
        digest = hashlib.sha256(json.dumps(core_map, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if not core_map or digest != state["core_python_sources_sha256"]:
            raise ValueError("Invalid core Python source map/digest")
        for key in SHARED_KEYS:
            if state[key] != reference[key]:
                raise ValueError(f"Incomparable pilot runs: {key}")
        if {key: value for key, value in state["fixed_training"].items() if key != "learning_rates"} != fixed:
            raise ValueError("Incomparable fixed training budgets/configurations")
        for key in ("data_root", "val_root", "base_checkpoint"):
            if state["config"][key] != reference["config"][key]:
                raise ValueError(f"Incomparable pilot input path: {key}")
        for script in SHARED_SCRIPTS:
            if state["artifacts"][script]["sha256"] != reference["artifacts"][script]["sha256"]:
                raise ValueError(f"Incomparable frozen source: {script}")
        trials = state["trials"]
        if sorted(trial["learning_rate"] for trial in trials) != sorted(state["fixed_training"]["learning_rates"]):
            raise ValueError("Completed trials do not match the requested learning rates")
        for trial in trials:
            rate = trial["learning_rate"]
            if rate not in EXPECTED_RATES or rate in seen_rates:
                raise ValueError("Require each of .01/.003/.001 exactly once")
            seen_rates.add(rate)
            if (trial.get("status") != "evaluated" or trial.get("actual_samples") != 2000
                    or trial.get("actual_steps") != fixed["steps"]
                    or trial.get("completed_full_epochs") != 0):
                raise ValueError("Trial did not actually finish the fixed short-run/evaluation budget")
            for trial_key, shared_key in (
                ("initial_class_tokens_sha256", "formal_initial_class_tokens_sha256"),
                ("epoch_order_sha256", "epoch_order_sha256"),
                ("sample_prefix_sha256", "sample_prefix_sha256"),
            ):
                if trial[trial_key] != reference[shared_key]:
                    raise ValueError(f"Trial actual artifact mismatch: {trial_key}")
            observed = trial.get("observed_identity", {})
            if (observed.get("image_ids_sha256") != reference["expected_observed_image_ids_sha256"]
                    or observed.get("start_step") != 0 or observed.get("end_step") != fixed["steps"]
                    or observed.get("samples") != 2000 or observed.get("queries") != 4000):
                raise ValueError("Trial actual observed image/query budget differs")
            for name in ("checkpoint", "evaluation_summary"):
                path = Path(trial[name]).resolve()
                if run_dir.resolve() not in path.parents or file_hash(path) != trial[f"{name}_sha256"]:
                    raise ValueError(f"Trial artifact path/hash mismatch: {name}")
            summaries.append(Path(trial["evaluation_summary"]))
    if seen_rates != EXPECTED_RATES:
        raise ValueError("Successful comparison requires all three distinct learning rates")
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--deadline", default=DEADLINE.isoformat())
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or args.deadline > DEADLINE:
        parser.error("deadline may not exceed 2026-09-10T21:40:00+08:00")
    if not 0 < args.poll_seconds <= 30:
        parser.error("poll-seconds must be within (0,30]")
    args.project_root, args.output_dir = args.project_root.resolve(), args.output_dir.resolve()
    args.run_dir = [path.resolve() for path in args.run_dir]
    if len(set(args.run_dir)) != len(args.run_dir):
        parser.error("run-dir values must be unique")
    if args.output_dir == args.project_root or args.project_root in args.output_dir.parents:
        parser.error("Aggregation output must be outside the Git project")
    return args


def aggregate(args, *, now=None, sleep=None, run=None):
    now = now or (lambda: datetime.now(timezone.utc))
    sleep, run = sleep or time.sleep, run or subprocess.run
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = args.output_dir / "snapshots"
    snapshots.mkdir()
    state = {"format": "sam3-token-lr-aggregate-v1", "deadline": args.deadline.isoformat(),
             "run_dirs": [str(path) for path in args.run_dir], "command": None, "returncode": None}
    last = None

    def transition(status, detail=""):
        nonlocal last
        if last == (status, detail):
            return
        last = (status, detail)
        state.update(status=status, detail=detail, updated_at=now().isoformat())
        temporary = args.output_dir / ".state.json.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output_dir / "state.json")
        with (args.output_dir / "aggregate.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{now().isoformat()} {status} {detail}\n")

    transition("created")
    try:
        while True:
            remaining = (args.deadline - now()).total_seconds()
            if remaining <= 0:
                transition("stopped_deadline", "Not all requested LR results were verified before deadline")
                return 2
            loaded, progress = completed_inputs(args.run_dir)
            if loaded is not None:
                break
            transition("waiting_runs", progress)
            sleep(min(args.poll_seconds, remaining))
        summaries = validate_completed_runs(loaded)
        state["source_states"] = []
        for index, (path, _, raw) in enumerate(loaded):
            destination = snapshots / f"run_{index}_state.json"
            destination.write_bytes(raw)
            state["source_states"].append({"path": str(path / "state.json"), "sha256": hashlib.sha256(raw).hexdigest()})
        source = Path(loaded[0][1]["artifacts"]["calibrate_bilateral_thresholds.py"]["path"]).resolve()
        if loaded[0][0].resolve() not in source.parents:
            raise ValueError("Calibration snapshot is outside its pilot directory")
        raw = source.read_bytes()
        script = snapshots / "calibrate_bilateral_thresholds.py"
        script.write_bytes(raw)
        script_hash = hashlib.sha256(raw).hexdigest()
        if script_hash != loaded[0][1]["artifacts"]["calibrate_bilateral_thresholds.py"]["sha256"]:
            raise ValueError("Calibration script snapshot hash differs from pilot metadata")
        state["calibration_script_sha256"] = script_hash
        command = [args.python, str(script)]
        for summary in summaries:
            command.extend(["--summary", str(summary)])
        output = args.output_dir / "lr_calibration.json"
        command.extend(["--output", str(output)])
        timeout = min(120, (args.deadline - now()).total_seconds())
        if timeout <= 0:
            transition("stopped_deadline", "No time remains for CPU calibration")
            return 2
        state.update(command=command, timeout_seconds=timeout)
        transition("calibrating")
        with (args.output_dir / "calibration.log").open("x", encoding="utf-8") as stream:
            result = run(command, cwd=args.project_root,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"),
                         stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        state["returncode"] = result.returncode
        if result.returncode or not output.is_file():
            raise RuntimeError(f"CPU calibration did not complete successfully: exit={result.returncode}")
        state.update(calibration=str(output), calibration_sha256=file_hash(output))
        transition("complete", "Three distinct LR results verified and calibrated on the same val set")
        return 0
    except Exception as error:
        transition("failed", f"{type(error).__name__}: {error}")
        return 1


def main():
    raise SystemExit(aggregate(parse_args()))


if __name__ == "__main__":
    main()

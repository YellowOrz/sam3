#!/usr/bin/env python3
"""Bounded queued input-VE comparison; no GPU starts until predecessor exits.

Two lanes, GPUs 0/2, LR .001/.0003, each 20-step smoke -> execution-only small
validation -> same-checkpoint continuation to total 2000 -> three-model full val.
No retries, relabeling, threshold tuning, geometry or memory training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from scripts import run_nakehand_semantic_ablation as runner
from scripts import train_nakehand_input_ve as training
from scripts import evaluate_nakehand_input_ve as evaluation

FORMAT = "sam3-nakehand-input-ablation-supervisor-v1"
PREDECESSOR_FORMAT = "sam3-nakehand-boundary-supervisor-v1"
LATEST_DEADLINE = datetime.fromisoformat("2026-09-11T08:50:00+08:00")
LANES = (("lr1e-3", .001), ("lr3e-4", .0003))
PROBE_INDICES = (0, 18, 20, 66)
MIN_FREE_MIB = 12000
QUEUE_WAIT_SECONDS = 10800
GPU_WAIT_SECONDS = 1800
TERMINAL_CHILD_STATUSES = {"completed", "failed", "cancelled", "timed_out"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "old-unconstrained", "wait-for-run", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 2])
    parser.add_argument("--deadline", required=True, help="Timezone-aware cutoff no later than 2026-09-11 08:50 +08:00")
    parser.add_argument("--queue-wait-seconds", type=int, default=QUEUE_WAIT_SECONDS)
    parser.add_argument("--gpu-wait-seconds", type=int, default=GPU_WAIT_SECONDS)
    args = parser.parse_args(argv)
    try:
        args.deadline = datetime.fromisoformat(args.deadline)
    except ValueError:
        parser.error("Invalid ISO deadline")
    if (args.deadline.tzinfo is None or args.deadline > LATEST_DEADLINE
            or not 120 < (args.deadline - datetime.now(timezone.utc)).total_seconds() <= 9 * 3600):
        parser.error("Require 120 seconds to nine hours remaining, ending no later than the authorized 08:50 cutoff")
    if sorted(args.gpus) != [0, 2]:
        parser.error("This queue authorizes exactly two distinct GPUs: 0 and 2")
    if not 1 <= args.queue_wait_seconds <= QUEUE_WAIT_SECONDS or not 1 <= args.gpu_wait_seconds <= GPU_WAIT_SECONDS:
        parser.error("Queue wait must be <=3h; GPU resource wait must be <=30min")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if not args.wait_for_run.is_dir():
        parser.error("--wait-for-run must be the existing predecessor queue directory")
    if (args.output_dir.exists() or any(args.output_dir.is_relative_to(root)
            for root in (args.project_root, args.data_root, args.wait_for_run))):
        parser.error("Use a new output directory outside code, frozen data and predecessor outputs")
    return args


def read_json(path, maximum_bytes=64 * 1024 * 1024):
    path = Path(path)
    size = path.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"JSON exceeds the bounded audit limit: {path}")
    with path.open("rb") as stream:
        raw = stream.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise ValueError("JSON grew beyond the audit limit")
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(raw, parse_constant=invalid)


def predecessor_status(state):
    if state.get("format") != PREDECESSOR_FORMAT or not isinstance(state.get("commands"), dict):
        raise ValueError("Predecessor state does not match the expected boundary supervisor")
    active = []
    for name, value in state["commands"].items():
        if not isinstance(value, dict):
            raise ValueError("Malformed predecessor command state")
        # Metadata-only result entries have no command or execution status.
        if value.get("status") is not None and value.get("status") not in TERMINAL_CHILD_STATUSES:
            active.append(name)
        elif value.get("command") and value.get("status") not in TERMINAL_CHILD_STATUSES:
            active.append(name)
    terminal = state.get("status") in {"complete", "failed_or_stopped"}
    return terminal and not active, active


def active_predecessor_processes(directory):
    """Read-only Linux process check for same-user children targeting this run.

    The old supervisor does not record child PIDs. Its terminal state plus no
    active command rows is checked separately; this catches lingering Python or
    timeout-wrapped children with an output directory inside the predecessor.
    Never signal any predecessor process or any other user's process.
    """
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            with (entry / "cmdline").open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise RuntimeError("Cannot safely inspect oversized same-user command line")
            argv = [piece.decode("utf-8", errors="surrogateescape") for piece in raw.split(b"\0") if piece]
            if "scripts.run_nakehand_boundary_ablation" in argv or "scripts.run_nakehand_input_ablation" in argv:
                continue
            for index, value in enumerate(argv[:-1]):
                if value == "--output-dir":
                    output = Path(argv[index + 1])
                    if output.is_absolute() and output.resolve().is_relative_to(directory.resolve()):
                        found.append({"pid": int(entry.name), "command": argv, "output_dir": str(output)})
                        break
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError("Cannot verify a same-user process while checking predecessor exit") from error
    return found


def checkpoint_result(output, steps, learning_rate):
    import torch
    summary = read_json(output / "summary.json")
    path = Path(summary["final_checkpoint"]).resolve()
    digest = runner.file_hash(path)
    if path.parent != output.resolve() or digest != summary["final_checkpoint_sha256"]:
        raise ValueError("Trial checkpoint path/fingerprint differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if runner.file_hash(path) != digest:
        raise RuntimeError("Checkpoint changed while validating the completed stage")
    info = training.validate_checkpoint_schema(state, minimum_samples=steps)
    if (info["completed_steps"] != steps or info["learning_rate"] != learning_rate
            or summary.get("format") != training.FORMAT or summary.get("status") != "completed_requested_steps"
            or summary.get("progress") != state["progress"]):
        raise ValueError("Actual steps/LR differ from the requested stage")
    return path, state


def compare_input_trials(first, second):
    left, right = dict(first["training_config"]), dict(second["training_config"])
    if left.pop("learning_rate") != .001 or right.pop("learning_rate") != .0003 or left != right:
        raise ValueError("Input trials must differ only in learning rate")
    for key in ("initial_input_residual_state", "initial_cache_state_dict"):
        if training.shared.cache_fingerprint(first[key]) != training.shared.cache_fingerprint(second[key]):
            raise ValueError("Input trials have different natural zero initializations")
    if (first["next_step"] != 2000 or second["next_step"] != 2000
            or first["progress"] != second["progress"] or first["observed_image_ids"] != second["observed_image_ids"]
            or first["planned_dataset_indices"] != second["planned_dataset_indices"]):
        raise ValueError("Input trials did not consume the same complete 2000-image prefix")
    return {"only_training_factor": "learning_rate: .001 versus .0003", "actual_steps_each": 2000,
            "same_initial_input_residual": True, "same_initial_cache": True, "same_ordered_training_images": True,
            "architecture_comparison_note": "LR .001 shares old output-delta data/loss/numerics, but input adaptation also changes cross-side sharing; not a pure Transformer-only factor."}


def validate_evaluation_result(output, *, full):
    value = read_json(output / "summary.json")
    labels = list(evaluation.LABELS) if full else ["baseline", "input-ve"]
    count = 3449 if full else len(PROBE_INDICES)
    if (value.get("format") != evaluation.FORMAT or value.get("status") != "completed"
            or value.get("evaluated_images") != count or value.get("all_sources_unchanged") is not True
            or value.get("observed_identity_verified") is not True
            or value.get("parameters_unchanged_by_version_counter") is not True
            or value.get("completed_images_per_model") != {label: count for label in labels}
            or value.get("full_val_evaluated") is not full
            or set(value.get("metrics", {})) != set(labels) or set(value.get("spatial_metrics", {})) != set(labels)):
        raise ValueError("Evaluation did not actually finish the requested variants and scope")
    if not full and value.get("evaluated_dataset_indices") != list(PROBE_INDICES):
        raise ValueError("Execution-only probe indices changed")
    image_ids = value.get("evaluated_image_ids", [])
    if len(image_ids) != count or len(set(image_ids)) != count:
        raise ValueError("Evaluation image identities are missing/duplicated")
    expected = {(image_id, side) for image_id in image_ids for side in training.SIDE_NAMES}
    if set(value.get("record_files", {})) != set(labels):
        raise ValueError("Missing exact final per-model record inventory")
    for label in labels:
        entry = value["record_files"][label]
        path = Path(entry["path"]).resolve()
        if path.parent != (output / "records").resolve() or runner.file_hash(path) != entry["sha256"]:
            raise ValueError("Final records changed or escaped the output")
        rows = read_json(path)
        actual = [(row["image_id"], row["prompt_key"]) for row in rows]
        if (len(actual) != 2 * count or set(actual) != expected
                or any(row.get("model") != label or row.get("identity_verified") is not True
                       or set(row.get("spatial", {})) != {"candidate", "detected"} for row in rows)):
            raise ValueError("Actual image-query records are incomplete or mismatched")
        for row in rows:
            confidence = row.get("top_confidence")
            if (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
                    or row.get("detected") != (confidence >= .5)):
                raise ValueError("Invalid confidence/detection fields in final records")
            for stage in ("candidate", "detected"):
                measured = row["spatial"][stage]
                if set(measured.get("boundary", {})) != set(evaluation.spatial.RATIO_KEYS):
                    raise ValueError("Final records lack the three actual spatial boundary measurements")
                for band in measured["boundary"].values():
                    intersection, union = band.get("intersection_pixels"), band.get("union_pixels")
                    if (type(intersection) is not int or type(union) is not int
                            or not 0 <= intersection <= union or type(band.get("band_pixels")) is not int
                            or band["band_pixels"] < 1
                            or band.get("iou") != (intersection / union if union else None)):
                        raise ValueError("Malformed boundary intersection/union evidence")
        if runner.file_hash(path) != entry["sha256"]:
            raise RuntimeError("Records changed while checking final coverage")
    report = output / "REPORT.md"
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError("Completed evaluation has no report")
    return {"summary": str(output / "summary.json"), "report": str(report),
            "report_sha256": runner.file_hash(report), "summary_sha256": runner.file_hash(output / "summary.json"),
            "actual_images_per_variant": count, "models": labels,
            "probe_interpretation": "execution/finite-output gate only; no accuracy threshold or hyperparameter selection" if not full else None}


class InputRunner(runner.AblationRunner):
    def __init__(self, args, *, popen=None, process_probe=None, **kwargs):
        super().__init__(args, **kwargs)
        self.popen = popen or subprocess.Popen
        self.process_probe = process_probe or active_predecessor_processes
        self.state.update(format=FORMAT, retry_policy="none", wait_for_run=str(args.wait_for_run),
            queue_wait_timeout_seconds=args.queue_wait_seconds, gpu_wait_timeout_seconds=args.gpu_wait_seconds,
            maximum_parallel_gpu_jobs=2, gpu_memory_fraction=.35, min_free_gpu_mib=MIN_FREE_MIB,
            learning_rates={label: rate for label, rate in LANES}, smoke_steps=20, checkpoint_interval=100,
            smoke_validation_indices=list(PROBE_INDICES), smoke_quality_selection=False, package_execution=True,
            stage_caps_seconds={"smoke": 600, "smoke_validation": 600, "continuation": 3600, "full_validation": 7200},
            verified_reports={}, geometry_training=False, memory_training=False, realsense_used=False,
            latest_user_review={"validation_image_id": 4713, "source_frame_index": 0,
                "human_side": "right_hand", "both_source_masks_empty": True,
                "inspected_output_delta_prompt": "left_hand",
                "conclusion": "Reference omission and model anatomical-side confusion coexist; neither all reference-relative FP are true errors nor all are unfairly penalized good predictions.",
                "labels_modified": False},
            authorization="Controlled queued experiments only; complete/stop no later than 2026-09-11 08:50 +08:00")

    def snapshot(self):
        paths = {path.resolve(): "core" for path in (self.args.project_root / "sam3").rglob("*.py")}
        if not paths:
            raise ValueError("Missing core source inventory")
        scripts = set(runner.SCRIPTS) | {"run_nakehand_input_ablation.py", "run_nakehand_boundary_ablation.py",
            "train_nakehand_input_ve.py", "evaluate_nakehand_input_ve.py", "soft_ve_prompt.py",
            "evaluate_hand_boundary_diagnostics.py", "analyze_hand_segmentation_failures.py"}
        for name in sorted(scripts):
            source, target = self.args.project_root / "scripts" / name, self.args.output_dir / "snapshots" / name
            digest = runner.file_hash(source)
            shutil.copy2(source, target)
            if runner.file_hash(source) != digest or runner.file_hash(target) != digest:
                raise RuntimeError("Script changed while snapshotting")
            paths[source.resolve()], paths[target.resolve()] = "script", "snapshot"
        inputs = [self.args.base_checkpoint, self.args.initial_cache, self.args.old_unconstrained,
                  self.args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"]
        inputs += [self.args.data_root / name for name in ("READY.json", "manifest.json", "frozen-plan.json")]
        for split in ("train", "val"):
            inputs += [self.args.data_root / split / name for name in ("READY.json", "manifest.json", "annotations.json")]
        for path in inputs:
            paths[path.resolve()] = "input"
        self.state["sources"] = [{"path": str(path), "kind": kind, "sha256": runner.file_hash(path)}
                                 for path, kind in sorted(paths.items())]
        self.update(None, status="waiting_for_predecessor")

    def verify_sources(self):
        if self.remaining() < 30:
            raise TimeoutError("No remaining authorized time for source verification")
        expected = {row["path"] for row in self.state["sources"] if row["kind"] == "core"}
        actual = {str(path.resolve()) for path in (self.args.project_root / "sam3").rglob("*.py")}
        if expected != actual:
            raise RuntimeError("Core source inventory changed")
        for row in self.state["sources"]:
            if self.remaining() < 30:
                raise TimeoutError("Authorization exhausted during finite source verification")
            if runner.file_hash(row["path"]) != row["sha256"]:
                raise RuntimeError(f"Frozen source changed: {row['path']}")

    def wait_for_predecessor(self):
        started = self.monotonic()
        state_path = self.args.wait_for_run / "state.json"
        while self.remaining() >= 120:
            self.check_stopping()
            left = self.args.queue_wait_seconds - (self.monotonic() - started)
            if left <= 0:
                raise TimeoutError("Predecessor did not finish within the three-hour bounded queue wait")
            predecessor = read_json(state_path, 16 * 1024 * 1024)
            ready, active_commands = predecessor_status(predecessor)
            live = self.process_probe(self.args.wait_for_run) if ready else []
            if ready and not live:
                # Read once more after process inspection to reject a restarted queue.
                current = read_json(state_path, 16 * 1024 * 1024)
                current_ready, _ = predecessor_status(current)
                if not current_ready or current != predecessor:
                    continue
                if self.monotonic() - started >= self.args.queue_wait_seconds:
                    raise TimeoutError("Predecessor readiness arrived after queue wait limit")
                self.verify_sources()
                self.update(None, status="predecessor_finished", predecessor={
                    "run": str(self.args.wait_for_run), "state": str(state_path), "state_sha256": runner.file_hash(state_path),
                    "status": predecessor["status"], "running_commands": [], "matching_live_child_processes": [],
                    "checked_at": self.now().isoformat(), "elapsed_wait_seconds": self.monotonic() - started,
                    "failure_policy": "A predecessor failure is not retried; this method may proceed independently if sources are unchanged"})
                return
            self.update(None, status="waiting_for_predecessor", predecessor_observed_status=predecessor.get("status"),
                        predecessor_active_commands=active_commands, predecessor_live_children=live)
            self.sleep(min(30, left, max(0, self.remaining() - 119)))
        raise TimeoutError("Authorization exhausted before predecessor completion")

    def wait_for_gpu(self, gpu):
        started = self.monotonic()
        while self.remaining() >= 120:
            self.check_stopping()
            left = self.args.gpu_wait_seconds - (self.monotonic() - started)
            if left <= 0:
                raise TimeoutError(f"GPU {gpu} did not have {MIN_FREE_MIB} MiB free within the bounded GPU wait")
            result = self.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=min(10, self.remaining(), left), check=False)
            self.check_stopping()
            if result.returncode or not result.stdout.strip().isdigit():
                raise RuntimeError(f"Cannot safely query GPU {gpu}")
            left = self.args.gpu_wait_seconds - (self.monotonic() - started)
            if left <= 0:
                raise TimeoutError("GPU readiness arrived after wait timeout")
            if int(result.stdout.strip()) >= MIN_FREE_MIB:
                return
            self.sleep(min(30, left, max(0, self.remaining() - 119)))
        raise TimeoutError("Insufficient remaining authorization for a GPU stage")

    def _stop_own_child(self, process, label):
        if process.poll() is not None:
            return
        self.update(label, termination="SIGTERM own newly-created process group", termination_started_at=self.now().isoformat())
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    def child(self, label, command, *, gpu, cap, needs_gpu=True):
        process = None
        try:
            self.check_stopping()
            if needs_gpu:
                self.update(label, status="waiting_for_gpu", gpu=gpu, command=list(command), min_free_gpu_mib=MIN_FREE_MIB)
                self.wait_for_gpu(gpu)
            self.verify_sources()
            self.check_stopping()
            timeout = runner.stage_timeout(self.remaining(), cap)
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                PYTHONPATH=os.pathsep.join((str(self.args.project_root), str(self.args.project_root / "scripts"))))
            log = self.args.output_dir / "logs" / f"{label}.log"
            output = command[command.index("--output-dir") + 1]
            self.update(label, status="starting", gpu=gpu, command=list(command), log=str(log), output_dir=output,
                        timeout_seconds=timeout, stage_cap_seconds=cap, started_at=self.now().isoformat())
            self.check_stopping()
            with log.open("x") as stream:
                process = self.popen(command, cwd=self.args.project_root, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                started = self.monotonic()
                self.update(label, status="running", pid=process.pid, process_group_id=process.pid,
                            owns_new_process_group=True)
                while process.poll() is None:
                    if self.monotonic() - started >= timeout or self.remaining() <= 30:
                        raise TimeoutError(f"{label} exceeded its finite stage/deadline budget")
                    # A peer failure stops later stages, not an already-running finite child.
                    try:
                        process.wait(timeout=min(5, max(.01, timeout - (self.monotonic() - started)), max(.01, self.remaining() - 30)))
                    except subprocess.TimeoutExpired:
                        pass
                returncode = process.returncode
            if returncode:
                raise RuntimeError(f"{label} failed with exit {returncode}; see {log}")
            self.verify_sources()
            self.update(label, status="completed", returncode=0, process_exited=True, completed_at=self.now().isoformat())
        except BaseException as error:
            if process is not None and process.poll() is None:
                self._stop_own_child(process, label)
            self.update(label, status="cancelled" if isinstance(error, runner.CancelledError) else
                        "timed_out" if isinstance(error, TimeoutError) else "failed",
                        pid=process.pid if process is not None else None,
                        process_exited=process.poll() is not None if process is not None else True,
                        returncode=process.returncode if process is not None else None,
                        error=f"{type(error).__name__}: {error}", completed_at=self.now().isoformat())
            raise

    def training_command(self, output, learning_rate, steps, resume=None):
        command = [self.args.python, "-u", "-m", "scripts.train_nakehand_input_ve",
            "--data-root", str(self.args.data_root / "train"), "--base-checkpoint", str(self.args.base_checkpoint),
            "--initial-cache", str(self.args.initial_cache), "--project-root", str(self.args.project_root),
            "--learning-rate", str(learning_rate), "--seed", "123", "--max-steps", str(steps),
            "--checkpoint-every", "100", "--gpu-memory-fraction", "0.35", "--output-dir", str(output)]
        if resume is not None:
            command.extend(("--resume", str(resume)))
        return command

    def evaluation_command(self, output, checkpoint, *, full):
        command = [self.args.python, "-u", "-m", "scripts.evaluate_nakehand_input_ve",
            "--data-root", str(self.args.data_root / "val"), "--base-checkpoint", str(self.args.base_checkpoint),
            "--input-checkpoint", str(checkpoint), "--project-root", str(self.args.project_root),
            "--variant", "all", "--minimum-samples-seen", "2000" if full else "20",
            "--render-count", "8" if full else "0", "--gpu-memory-fraction", "0.35",
            "--deadline", self.args.deadline.isoformat(), "--output-dir", str(output)]
        if full:
            command.extend(("--output-delta-checkpoint", str(self.args.old_unconstrained)))
        else:
            command.extend(("--indices", ",".join(map(str, PROBE_INDICES))))
        return command

    def train_one(self, label, learning_rate, gpu):
        smoke = self.args.output_dir / f"{label}-smoke20"
        self.child(f"{label}-smoke20", self.training_command(smoke, learning_rate, 20), gpu=gpu, cap=600)
        resume, _ = checkpoint_result(smoke, 20, learning_rate)
        probe = self.args.output_dir / f"{label}-probe"
        self.child(f"{label}-probe", self.evaluation_command(probe, resume, full=False), gpu=gpu, cap=600)
        self.update(f"{label}-probe", verified_execution_probe=validate_evaluation_result(probe, full=False))
        output = self.args.output_dir / f"{label}-train2000"
        self.child(f"{label}-train2000", self.training_command(output, learning_rate, 2000, resume), gpu=gpu, cap=3600)
        path, state = checkpoint_result(output, 2000, learning_rate)
        with self.lock:
            self.state["trials"][label] = {"checkpoint": str(path), "sha256": runner.file_hash(path),
                "progress": state["progress"], "learning_rate": learning_rate}
            self.save()
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
                self.update(None, status="stopping_after_failure", error=f"{type(error).__name__}: {error}")
                for future in futures:
                    future.cancel()
                raise
        return results

    def full_validation(self, label, checkpoint, gpu):
        output = self.args.output_dir / f"{label}-validation"
        self.child(f"{label}-validation", self.evaluation_command(output, checkpoint, full=True), gpu=gpu, cap=7200)
        result = validate_evaluation_result(output, full=True)
        with self.lock:
            self.state["verified_reports"][label] = result
            self.save()
        return result

    def execute(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        for name in ("logs", "snapshots"):
            (self.args.output_dir / name).mkdir()
        self.update(None, status="created")
        try:
            self.snapshot()
            self.wait_for_predecessor()
            trained = self.paired([(label, self.train_one, (label, rate, gpu))
                for (label, rate), gpu in zip(LANES, self.args.gpus)])
            comparison = compare_input_trials(trained["lr1e-3"][1], trained["lr3e-4"][1])
            self.update(None, status="training_complete", training_comparison=comparison)
            reports = self.paired([(label, self.full_validation, (label, trained[label][0], gpu))
                for (label, _rate), gpu in zip(LANES, self.args.gpus)])
            self.verify_sources()
            self.update(None, status="complete", verified_reports=reports)
            return 0
        except BaseException as error:
            self.stop_event.set()
            self.update(None, status="failed_or_stopped", error=f"{type(error).__name__}: {error}")
            return 1


if __name__ == "__main__":
    raise SystemExit(InputRunner(parse_args()).execute())

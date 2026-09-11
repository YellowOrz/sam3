"""Recover only the failed validation of two already-completed semantic pilots.

Preserve the old run, checkpoints, snapshots and all dependencies. The sole
execution change is a coherent ``python -m scripts.evaluate_...`` package entry.
Use a caller-owned tmux session. No training, retries, or other-process signals.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

from scripts.report_nakehand_semantic_ablation import compare_training_contracts, file_hash
from scripts.run_nakehand_semantic_ablation import completed_checkpoint


FORMAT = "sam3-nakehand-semantic-validation-recovery-v1"
EVALUATOR = "scripts.evaluate_nakehand_semantic_tokens"
REPORTER = "scripts.report_nakehand_semantic_ablation"
VALIDATION_TIMEOUT = 5400
REPORT_TIMEOUT = 120
GPU_WAIT_TIMEOUT = 1800


def required_argument(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f"Expected exactly one {flag} in historical command")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Missing historical argument: {flag}")
    return command[index + 1]


def verify_sources(rows, project_root):
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Source fingerprints must be nonempty and unique")
    expected_core = {row["path"] for row in rows if row["kind"] == "core"}
    actual_core = {str(path.resolve()) for path in (project_root / "sam3").rglob("*.py")}
    if not expected_core or expected_core != actual_core:
        raise ValueError("Frozen core source inventory changed")
    for row in rows:
        if file_hash(row["path"]) != row["sha256"]:
            raise RuntimeError(f"Frozen source changed: {row['path']}")


def inspect_completed_run(old_run, project_root):
    state_path = old_run / "state.json"
    digest = file_hash(state_path)
    state = json.loads(state_path.read_text())
    if (state.get("format") != "sam3-nakehand-semantic-ablation-supervisor-v1"
            or state.get("status") != "failed_or_stopped"
            or state.get("commands", {}).get("full_validation", {}).get("status") != "failed"):
        raise ValueError("Only the existing failed validation, not an active/new training run, can be recovered")
    verify_sources(state["sources"], project_root)
    old_command = state["commands"]["full_validation"]["command"]
    if Path(required_argument(old_command, "--project-root")).resolve() != project_root:
        raise ValueError("Historical project root differs")
    data_root = Path(state["data_root"]).resolve()
    if Path(required_argument(old_command, "--data-root")).resolve() != data_root / "val":
        raise ValueError("Historical command was not the fixed validation split")
    if (required_argument(old_command, "--variant") != "all"
            or required_argument(old_command, "--minimum-samples-seen") != "2000"
            or required_argument(old_command, "--gpu-memory-fraction") != "0.25"):
        raise ValueError("Historical full-validation contract differs")
    base = Path(required_argument(old_command, "--base-checkpoint")).resolve()
    source_paths = {row["path"] for row in state["sources"]}
    required_sources = [base, *(project_root / "scripts" / name for name in (
        "evaluate_nakehand_semantic_tokens.py", "report_nakehand_semantic_ablation.py", "cached_ve_text_features.py"))]
    if not all(str(path) in source_paths for path in required_sources):
        raise ValueError("Actual package entry/dependencies/base are not bound by the old source contract")
    checked, loaded, added_sources = {}, [], [{"path": str(state_path), "kind": "historical_state", "sha256": digest}]
    for label in ("unconstrained", "anchored"):
        if state["commands"].get(label, {}).get("status") != "completed":
            raise ValueError(f"Training is not already completed: {label}")
        summary_path = old_run / label / "summary.json"
        summary_hash = file_hash(summary_path)
        summary = json.loads(summary_path.read_text())
        checkpoint, checkpoint_state = completed_checkpoint(summary, old_run / label)
        historical = state.get("trials", {}).get(label, {})
        checkpoint_hash = file_hash(checkpoint)
        if (historical.get("checkpoint") != str(checkpoint) or historical.get("sha256") != checkpoint_hash
                or historical.get("progress") != checkpoint_state["progress"]):
            raise ValueError(f"Historical trial checkpoint/progress differs: {label}")
        flag = "--unconstrained-checkpoint" if label == "unconstrained" else "--constrained-checkpoint"
        if Path(required_argument(old_command, flag)).resolve() != checkpoint:
            raise ValueError("Recovery checkpoint differs from previously intended validation")
        checked[label] = {"checkpoint": str(checkpoint), "sha256": checkpoint_hash,
                          "progress": checkpoint_state["progress"], "summary_sha256": summary_hash}
        loaded.append(checkpoint_state)
        added_sources.extend(({"path": str(summary_path), "kind": "completed_training_summary", "sha256": summary_hash},
                              {"path": str(checkpoint), "kind": "completed_checkpoint", "sha256": checkpoint_hash}))
    comparison = compare_training_contracts(*loaded)
    if file_hash(state_path) != digest:
        raise RuntimeError("Historical state changed during verification")
    return {"old_state_sha256": digest, "sources": state["sources"] + added_sources,
            "checkpoints": checked, "training_comparison": comparison,
            "data_root": str(data_root), "base_checkpoint": str(base),
            "historical_failed_command": old_command}


def package_identity_probe(checkpoint_path):
    """Execute the real evaluator module's imports without its GPU-only main.

    runpy uses the same package/spec resolution as -m. A real cache reconstructed
    through evaluator.semantic must install via evaluator.cached's strict class
    check into a tiny CPU model. No isinstance weakening or class alias patch.
    """
    import torch

    namespace = runpy.run_module(EVALUATOR, run_name="__cpu_package_identity_probe__")
    cached, semantic = namespace["cached"], namespace["semantic"]
    if namespace["__package__"] != "scripts" or cached is not semantic.cached:
        raise RuntimeError("Package evaluator and semantic helper have divergent cached-module identities")
    if "cached_ve_text_features" in sys.modules:
        raise RuntimeError("Unexpected top-level cached_ve_text_features module alias")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder = semantic.cache_from_state(state["initial_cache_state_dict"], frozen=True)
    if type(encoder) is not cached.CachedVETextEncoder:
        raise RuntimeError("Actual reconstructed encoder has a different class identity")
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.language_backbone = torch.nn.Linear(1, 1)
    original = cached.install_cached_ve_text_encoder(model, encoder)
    cached.set_cached_ve_training_mode(model, train_delta=False)
    outputs = encoder(list(cached.CLASS_NAMES), device="cpu")
    if any(output.device.type != "cpu" for output in outputs):
        raise RuntimeError("Identity probe unexpectedly used non-CPU tensors")
    cached.restore_original_ve_text_encoder(model, original)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only identity probe unexpectedly initialized CUDA")
    return {"module": EVALUATOR, "entry_package": namespace["__package__"],
            "entry_source": namespace["__file__"], "cached_module": cached.__name__,
            "semantic_cached_is_same_module": True, "reconstructed_class_matches_exactly": True,
            "strict_install_and_restore_passed": True, "cuda_initialized": False,
            "method": "runpy real package-module imports with non-main name; actual -m --help is a separate logged execution-stage check"}


def environment(project_root, gpu):
    result = dict(os.environ)
    result.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1",
                  PYTHONPATH=str(project_root))
    return result


def commands(args, evidence):
    paths = evidence["checkpoints"]
    un = paths["unconstrained"]["checkpoint"]
    an = paths["anchored"]["checkpoint"]
    evaluation = [args.python, "-u", "-m", EVALUATOR,
                  "--data-root", str(Path(evidence["data_root"]) / "val"),
                  "--base-checkpoint", evidence["base_checkpoint"], "--baseline-checkpoint", un,
                  "--unconstrained-checkpoint", un, "--constrained-checkpoint", an,
                  "--variant", "all", "--project-root", str(args.project_root),
                  "--minimum-samples-seen", "2000", "--gpu-memory-fraction", "0.25",
                  "--output-dir", str(args.output_dir / "validation")]
    reporting = [args.python, "-u", "-m", REPORTER, "--summary", str(args.output_dir / "validation/summary.json"),
                 "--unconstrained-checkpoint", un, "--constrained-checkpoint", an,
                 "--output", str(args.output_dir / "NAKEHAND_SEMANTIC_REPORT.md")]
    return evaluation, reporting


@contextmanager
def singleton(old_run):
    identity = hashlib.sha256(str(old_run).encode()).hexdigest()[:20]
    path = Path(tempfile_root()) / f"sam3-semantic-val-{os.getuid()}-{identity}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise PermissionError("Recovery lock is not owned by this user")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield str(path)
    finally:
        os.close(descriptor)


def tempfile_root():
    return "/tmp"


class Recovery:
    def __init__(self, args):
        self.args = args
        self.state = {"format": FORMAT, "status": "created", "created_at": self.now(),
                      "old_run": str(args.old_run), "supervisor_pid": os.getpid(), "gpu": args.gpu,
                      "training_performed": False, "realsense_used": False,
                      "retry_policy": "none", "validation_timeout_seconds": VALIDATION_TIMEOUT,
                      "gpu_memory_fraction": .25, "commands": {}}

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def save(self, **values):
        self.state.update(values, updated_at=self.now())
        temporary = self.args.output_dir / "state.json.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.args.output_dir / "state.json")

    def wait_gpu(self):
        deadline = time.monotonic() + GPU_WAIT_TIMEOUT
        self.save(status="waiting_for_gpu")
        while time.monotonic() < deadline:
            result = subprocess.run(["nvidia-smi", f"--id={self.args.gpu}", "--query-gpu=memory.free",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    check=True, timeout=10)
            if not result.stdout.strip().isdigit():
                raise ValueError("Cannot parse GPU free memory")
            free = int(result.stdout.strip())
            self.save(last_gpu_free_mib=free)
            if free >= 7500:
                return
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        raise TimeoutError("GPU did not have 7500 MiB free within 1800 seconds")

    def child(self, label, command, timeout):
        verify_sources(self.state["evidence"]["sources"], self.args.project_root)
        log = self.args.output_dir / "logs" / f"{label}.log"
        record = {"status": "starting", "command": command, "cwd": str(self.args.project_root),
                  "log": str(log), "timeout_seconds": timeout, "started_at": self.now()}
        self.state["commands"][label] = record
        self.save(status=label)
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=self.args.project_root,
                                       env=environment(self.args.project_root, self.args.gpu),
                                       stdout=stream, stderr=subprocess.STDOUT)
            record.update(status="running", pid=process.pid)
            self.save()
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                # Only this owned direct child. Evaluator has no worker children.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                record.update(status="failed", error="parent interruption or finite timeout", completed_at=self.now())
                self.save()
                raise
        record.update(returncode=code, completed_at=self.now(), status="completed" if code == 0 else "failed")
        self.save()
        if code:
            raise RuntimeError(f"{label} failed with exit {code}; see {log}")
        verify_sources(self.state["evidence"]["sources"], self.args.project_root)

    def execute(self):
        # Output must be exclusively created before any subprocess is started.
        self.args.output_dir.mkdir(parents=True, exist_ok=False)
        (self.args.output_dir / "logs").mkdir()
        self.save()
        try:
            with singleton(self.args.old_run) as lock_path:
                self.save(status="verifying_completed_training", singleton_lock=lock_path)
                evidence = inspect_completed_run(self.args.old_run, self.args.project_root)
                snapshot = self.args.output_dir / "recovery-script.py"
                shutil.copyfile(Path(__file__), snapshot)
                old_snapshot = self.args.output_dir / "historical-state.json"
                shutil.copyfile(self.args.old_run / "state.json", old_snapshot)
                for path, kind in ((Path(__file__).resolve(), "recovery_source"), (snapshot, "recovery_snapshot"),
                                   (old_snapshot, "historical_state_snapshot")):
                    evidence["sources"].append({"path": str(path), "kind": kind, "sha256": file_hash(path)})
                probe_result = package_identity_probe(evidence["checkpoints"]["unconstrained"]["checkpoint"])
                self.save(evidence=evidence, package_identity_probe=probe_result)
                self.child("actual_module_help_cpu", [self.args.python, "-m", EVALUATOR, "--help"], 120)
                self.wait_gpu()
                evaluation, reporting = commands(self.args, evidence)
                self.child("full_validation", evaluation, VALIDATION_TIMEOUT)
                self.child("report", reporting, REPORT_TIMEOUT)
                report = self.args.output_dir / "NAKEHAND_SEMANTIC_REPORT.md"
                self.save(status="complete", report=str(report), report_sha256=file_hash(report), completed_at=self.now())
                return 0
        except BaseException as error:
            self.save(status="failed_or_stopped", error=f"{type(error).__name__}: {error}", completed_at=self.now())
            return 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, choices=(0,), default=0)
    parser.add_argument("--check-only", action="store_true", help="read-only sources/checkpoint/module identity verification, no CUDA/output")
    args = parser.parse_args(argv)
    for name in ("old_run", "output_dir", "project_root"):
        setattr(args, name, getattr(args, name).resolve())
    historical_data = Path(json.loads((args.old_run / "state.json").read_text())["data_root"]).resolve()
    if args.output_dir.exists() or any(args.output_dir == root or root in args.output_dir.parents
                                     for root in (args.old_run, args.project_root, historical_data)):
        parser.error("Choose a new output outside the old run and project tree")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.check_only:
        evidence = inspect_completed_run(args.old_run, args.project_root)
        identity = package_identity_probe(evidence["checkpoints"]["unconstrained"]["checkpoint"])
        print(json.dumps({"verified_source_count": len(evidence["sources"]),
                          "training_comparison": evidence["training_comparison"], "identity": identity,
                          "checkpoints": evidence["checkpoints"]}, ensure_ascii=False, indent=2))
        return 0
    return Recovery(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())

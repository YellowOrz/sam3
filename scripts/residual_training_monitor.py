"""Rank-zero TensorBoard and JSONL monitoring for residual training.

Callers supply loss/performance tags and keep RGB, reference masks and predictions
under separate image tags. Steps are optimizer steps, while progress/samples_seen
records the sample counter on that same axis. A resumed run may start at any step.
No CUDA synchronization or performance measurement is performed by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Integral, Real
from pathlib import Path
import time


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _finite_scalar(value, name):
    # Scalar tensors are detached before conversion; callers should aggregate
    # losses before logging instead of passing a whole batch or autograd graph.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "ndim") and value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _tag(value, name="tag"):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    if any(not part or part in (".", "..") for part in value.split("/")):
        raise ValueError(f"{name} must have nonempty path components")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} cannot contain control characters")
    return value


def _scalars(values):
    if not isinstance(values, Mapping):
        raise ValueError("metrics must be a mapping of tags to scalar values")
    return {_tag(key): _finite_scalar(value, key) for key, value in values.items()}


class TrainingMonitor:
    """Write rank-zero logs lazily, flushing both formats on close.

    ``enabled=False`` or ``rank != 0`` disables all output and optional imports.
    Calls in one monitor must use nondecreasing steps (equal steps allow images
    and validation alongside training). For checkpoint rollback, use a new
    output directory so previously logged future steps remain distinguishable.
    """

    def __init__(self, output_dir, rank=0, enabled=True, flush_seconds=30):
        self.log_dir = Path(output_dir) / "tensorboard"
        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self.enabled = bool(enabled) and rank == 0
        self.flush_seconds = _finite_scalar(flush_seconds, "flush_seconds")
        if self.flush_seconds <= 0:
            raise ValueError("flush_seconds must be positive")
        self._writer = None
        self._jsonl = None
        self._closed = False
        self._last_step = None
        self._last_samples_seen = None
        self._last_flush = time.monotonic()

    def _check_step(self, step):
        if self._closed:
            raise RuntimeError("TrainingMonitor is closed")
        step = _nonnegative_integer(step, "step")
        if self._last_step is not None and step < self._last_step:
            raise ValueError(f"step decreased from {self._last_step} to {step}")
        return step

    def _open(self):
        if self._writer is not None:
            return
        # Import only when an enabled rank-zero monitor actually emits a log.
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires tensorboard; install it or use enabled=False"
            ) from exc
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        try:
            self._writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=self.flush_seconds)
        except Exception:
            self._jsonl.close()
            self._jsonl = None
            raise

    def _record(self, record):
        self._jsonl.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        self._last_step = record["global_step"]
        if time.monotonic() - self._last_flush >= self.flush_seconds:
            self.flush()

    def log_scalars(self, step, values, samples_seen, wall_seconds):
        """Record supplied tags, samples consumed and elapsed run wall time."""
        if not self.enabled:
            return
        step = self._check_step(step)
        values = _scalars(values)
        samples_seen = _nonnegative_integer(samples_seen, "samples_seen")
        if self._last_samples_seen is not None and samples_seen < self._last_samples_seen:
            raise ValueError("samples_seen must be nondecreasing")
        wall_seconds = _finite_scalar(wall_seconds, "wall_seconds")
        if wall_seconds < 0:
            raise ValueError("wall_seconds must be nonnegative")
        reserved = {"progress/samples_seen", "progress/wall_seconds"}
        if reserved.intersection(values):
            raise ValueError("progress counters are supplied through the dedicated arguments")
        self._open()
        for key, value in values.items():
            self._writer.add_scalar(key, value, global_step=step)
        self._writer.add_scalar("progress/samples_seen", samples_seen, global_step=step)
        self._writer.add_scalar("progress/wall_seconds", wall_seconds, global_step=step)
        self._record({
            "kind": "train", "global_step": step, "samples_seen": samples_seen,
            "wall_seconds": wall_seconds, "values": values,
        })
        self._last_samples_seen = samples_seen

    def log_validation(self, step, metrics, scope):
        """Record validation/{scope}/{metric}, retaining the declared scope."""
        if not self.enabled:
            return
        step = self._check_step(step)
        scope = _tag(scope, "scope")
        metrics = _scalars(metrics)
        self._open()
        for key, value in metrics.items():
            self._writer.add_scalar(f"validation/{scope}/{key}", value, global_step=step)
        self._record({
            "kind": "validation", "scope": scope, "global_step": step,
            "samples_seen": self._last_samples_seen, "values": metrics,
        })

    def log_images(self, step, images):
        """Log separate images; tensors prefer CHW and NumPy arrays prefer HWC.

        HW grayscale masks are also accepted. Floating images must already be
        in [0, 1], integer images in [0, 255]; boolean masks become 0/255. This
        method never normalizes each mask independently or creates overlays.
        """
        if not self.enabled:
            return
        step = self._check_step(step)
        if not isinstance(images, Mapping):
            raise ValueError("images must be a mapping of separate image tags to arrays")
        import numpy as np

        prepared = []
        for key, value in images.items():
            key = _tag(key)
            is_tensor = hasattr(value, "detach")
            if is_tensor:
                value = value.detach().cpu()
                # NumPy does not accept torch.bfloat16 directly.
                if value.is_floating_point():
                    value = value.float()
                value = value.numpy()
            elif not isinstance(value, np.ndarray):
                raise ValueError(f"{key} must be a tensor or NumPy array")
            if value.ndim == 2:
                dataformats = "HW"
            elif value.ndim == 3:
                first_channel, last_channel = value.shape[0] in (1, 3, 4), value.shape[-1] in (1, 3, 4)
                if first_channel and (is_tensor or not last_channel):
                    dataformats = "CHW"
                elif last_channel:
                    dataformats = "HWC"
                else:
                    raise ValueError(f"{key} must have 1, 3 or 4 channels")
            else:
                raise ValueError(f"{key} must be one HW, CHW or HWC image")
            if not value.size or not np.isfinite(value).all():
                raise ValueError(f"{key} must be nonempty and finite")
            if value.dtype.kind == "b":
                value = value.astype(np.uint8) * 255
            elif value.dtype.kind in "iu":
                if value.min() < 0 or value.max() > 255:
                    raise ValueError(f"{key} integer pixels must be in [0, 255]")
                value = value.astype(np.uint8, copy=False)
            elif value.dtype.kind == "f":
                if value.min() < 0 or value.max() > 1:
                    raise ValueError(f"{key} floating pixels must be in [0, 1]")
            else:
                raise ValueError(f"{key} has an unsupported pixel dtype")
            prepared.append((key, value, dataformats))
        self._open()
        for key, value, dataformats in prepared:
            self._writer.add_image(key, value, global_step=step, dataformats=dataformats)
        self._record({
            "kind": "images", "global_step": step,
            "images": [{"tag": key, "shape": list(value.shape), "dataformats": fmt}
                       for key, value, fmt in prepared],
        })

    def flush(self):
        """Make queued events and JSONL records available to live readers."""
        if self._writer is not None:
            self._writer.flush()
        if self._jsonl is not None:
            self._jsonl.flush()
        self._last_flush = time.monotonic()

    def close(self):
        if self._closed:
            return
        try:
            self.flush()
        finally:
            try:
                if self._writer is not None:
                    self._writer.close()
            finally:
                if self._jsonl is not None:
                    self._jsonl.close()
                self._writer = None
                self._jsonl = None
                self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("TrainingMonitor is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

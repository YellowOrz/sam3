"""Shared, policy-free helpers for the repository's video CLIs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


COLORS = (
    (60, 60, 255),
    (60, 220, 60),
    (255, 100, 60),
    (60, 220, 220),
    (220, 60, 220),
    (220, 180, 60),
    (120, 60, 255),
    (255, 160, 60),
    (60, 160, 255),
    (180, 255, 60),
    (255, 60, 160),
    (160, 60, 255),
)
MASK_ALPHA = 0.30


def utc_now(timespec: str = "seconds") -> str:
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def expand_path(value: str) -> Path:
    """Expand a user path without resolving symlink components."""
    return Path(value).expanduser().absolute()


def parse_device(value: str) -> Tuple[str, int]:
    if value == "cuda":
        return "cuda:0", 0
    if not value.startswith("cuda:"):
        raise argparse.ArgumentTypeError("device must look like 'cuda:0'")
    try:
        index = int(value.split(":", 1)[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("device must look like 'cuda:0'") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("CUDA device index must be non-negative")
    return value, index


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def probe_video(
    video_path: Path, *, fallback_fps: Optional[float] = None
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video dimensions: {video_path}")
    if not np.isfinite(fps) or fps <= 0:
        if fallback_fps is None:
            raise RuntimeError(f"invalid video FPS: {video_path}")
        fps = fallback_fps
    return {"width": width, "height": height, "frame_count": frame_count, "fps": fps}


def extract_png_frames(
    video_path: Path, frame_dir: Path, max_frames: Optional[int]
) -> int:
    """Decode one video to sequential, lossless PNG files."""
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_path = frame_dir / f"{count:06d}.png"
            if not cv2.imwrite(
                str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
            ):
                raise RuntimeError(f"failed to write temporary frame: {frame_path}")
            count += 1
    finally:
        cap.release()
    if count == 0:
        raise RuntimeError(f"video contains no readable frames: {video_path}")
    return count


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically replace a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_output_arrays(
    outputs: Optional[Dict[str, Any]], score_key: str = "out_probs"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = outputs or {}
    obj_ids = as_numpy(outputs.get("out_obj_ids")).reshape(-1)
    scores = as_numpy(outputs.get(score_key)).reshape(-1)
    masks = as_numpy(outputs.get("out_binary_masks"))
    if masks.size == 0:
        masks = np.empty((0, 0, 0), dtype=bool)
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise RuntimeError(f"unexpected mask shape: {masks.shape}")
    if len(obj_ids) != len(masks):
        raise RuntimeError(
            f"object/mask count mismatch: {len(obj_ids)} IDs and {len(masks)} masks"
        )
    if len(scores) not in (0, len(obj_ids)):
        raise RuntimeError(
            f"object/score count mismatch: {len(obj_ids)} IDs and {len(scores)} scores"
        )
    return obj_ids, scores, masks


def color_for_label(label: int) -> Tuple[int, int, int]:
    return COLORS[(label - 1) % len(COLORS)]


def lighter_color(
    color: Tuple[int, int, int], amount: float = 0.45
) -> Tuple[int, int, int]:
    return tuple(round(channel + (255 - channel) * amount) for channel in color)

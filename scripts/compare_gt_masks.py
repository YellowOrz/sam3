"""Compare aligned RGB and lossless foreground mask videos without a model."""

import csv
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

if __package__:
    from scripts.video_utils import MASK_ALPHA, probe_video, write_json
else:
    from video_utils import MASK_ALPHA, probe_video, write_json


def metric_summary(rows: list) -> Dict[str, Any]:
    """Aggregate frame scores and pixel counts; empty batches have no score."""
    count = len(rows)
    intersection = sum(row["intersection"] for row in rows)
    prediction = sum(row["prediction_pixels"] for row in rows)
    ground_truth = sum(row["gt_pixels"] for row in rows)
    union = prediction + ground_truth - intersection
    return {
        "frames_evaluated": count,
        "intersection": intersection,
        "prediction_pixels": prediction,
        "gt_pixels": ground_truth,
        "mean_iou": sum(row["iou"] for row in rows) / count if count else None,
        "mean_dice": sum(row["dice"] for row in rows) / count if count else None,
        "pixel_iou": (intersection / union if union else 1.0) if count else None,
        "pixel_dice": (
            2 * intersection / (prediction + ground_truth)
            if prediction + ground_truth
            else 1.0
        )
        if count
        else None,
    }


def foreground(frame: np.ndarray, path: Path) -> np.ndarray:
    if frame.dtype != np.uint8:
        raise ValueError(f"mask must contain uint8 labels: {path}")
    if frame.ndim == 3:
        if not (
            np.array_equal(frame[:, :, 0], frame[:, :, 1])
            and np.array_equal(frame[:, :, 0], frame[:, :, 2])
        ):
            raise ValueError(f"mask channels differ: {path}")
        frame = frame[:, :, 0]
    return frame != 0


def compare_masks(
    video_path: Path,
    gt_path: Path,
    output_dir: Path,
    max_frames: Optional[int],
    skip_frames: int,
) -> Dict[str, Any]:
    """Write comparison artifacts only after successful aligned decoding."""
    if skip_frames < 0:
        raise ValueError("skip_frames must be non-negative")
    prediction_path = output_dir / "masks.mkv"
    paths = (video_path, prediction_path, gt_path)
    infos = [probe_video(path) for path in paths]
    source = infos[0]
    frame_count = source["frame_count"]
    if frame_count <= 0:
        raise ValueError(f"unknown or empty frame count: {video_path}")
    limit = min(frame_count, max_frames) if max_frames is not None else frame_count
    for path, info in zip(paths[1:], infos[1:]):
        if any(info[key] != source[key] for key in ("width", "height")):
            raise ValueError(f"dimensions do not match RGB: {path}")
        if not math.isclose(info["fps"], source["fps"], rel_tol=1e-5):
            raise ValueError(f"FPS does not match RGB: {path}")
        expected = limit if path == prediction_path else frame_count
        if info["frame_count"] != expected:
            raise ValueError(f"frame count does not match expected {expected}: {path}")

    stride = skip_frames + 1
    rows = []
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    try:
        with tempfile.TemporaryDirectory(prefix=".gt_compare_", dir=output_dir) as temp:
            temporary = Path(temp)
            writer = cv2.VideoWriter(
                str(temporary / "comparison.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source["fps"] / stride,
                (source["width"] * 3, source["height"]),
            )
            try:
                if not writer.isOpened():
                    raise RuntimeError("cannot create comparison.mp4")
                for frame_index in range(limit):
                    frames = []
                    for path, capture in zip(paths, captures):
                        ok, frame = capture.read()
                        if not ok or frame.shape[:2] != (
                            source["height"],
                            source["width"],
                        ):
                            raise ValueError(
                                f"cannot decode frame {frame_index}: {path}"
                            )
                        frames.append(frame)
                    prediction = foreground(frames[1], prediction_path)
                    gt = foreground(frames[2], gt_path)
                    if frame_index % stride:
                        continue
                    intersection = int(np.count_nonzero(prediction & gt))
                    prediction_pixels = int(np.count_nonzero(prediction))
                    gt_pixels = int(np.count_nonzero(gt))
                    union = prediction_pixels + gt_pixels - intersection
                    total = prediction_pixels + gt_pixels
                    row = {
                        "frame_index": frame_index,
                        "time_seconds": frame_index / source["fps"],
                        "intersection": intersection,
                        "prediction_pixels": prediction_pixels,
                        "gt_pixels": gt_pixels,
                        "iou": intersection / union if union else 1.0,
                        "dice": 2 * intersection / total if total else 1.0,
                    }
                    rows.append(row)
                    panels = [frames[0].copy() for _ in range(3)]
                    for panel, mask in zip(panels[1:], (prediction, gt)):
                        panel[mask] = (
                            panel[mask] * (1 - MASK_ALPHA)
                            + np.array((60, 220, 60)) * MASK_ALPHA
                        ).astype(np.uint8)
                    titles = (
                        f"RGB frame={frame_index}",
                        f"Prediction IoU={row['iou']:.3f} Dice={row['dice']:.3f}",
                        "GT",
                    )
                    for panel, title in zip(panels, titles):
                        for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                            cv2.putText(
                                panel,
                                title,
                                (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                thickness,
                                cv2.LINE_AA,
                            )
                    writer.write(np.concatenate(panels, axis=1))
                # With a frame limit, later source frames are outside this evaluation.
                for path, capture in zip(paths, captures):
                    if (
                        path == prediction_path or limit == frame_count
                    ) and capture.read()[0]:
                        raise ValueError(f"more decoded frames than declared: {path}")
            finally:
                writer.release()
            result_info = probe_video(temporary / "comparison.mp4")
            if result_info["frame_count"] != len(rows):
                raise RuntimeError("comparison video frame count is incomplete")
            with (temporary / "gt_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                csv_writer.writeheader()
                csv_writer.writerows(rows)
            summary = {
                "status": "success",
                "input_video": str(video_path),
                "gt_video": str(gt_path),
                "prediction_video": str(prediction_path),
                "skip_frames": skip_frames,
                "frame_stride": stride,
                "frames_compared_before_sampling": limit,
                "empty_masks_score": 1.0,
                **metric_summary(rows),
            }
            write_json(temporary / "gt_metrics.json", summary)
            for name in ("comparison.mp4", "gt_metrics.csv", "gt_metrics.json"):
                (temporary / name).replace(output_dir / name)
    finally:
        for capture in captures:
            capture.release()
    return {"summary": summary, "rows": rows}

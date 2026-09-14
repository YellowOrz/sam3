"""Compare aligned RGB and lossless foreground mask videos without a model."""

import argparse
import csv
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .video_utils import MASK_ALPHA, probe_video, write_json


def metric_summary(rows: list) -> Dict[str, Any]:
    """! @brief 汇总逐帧前景重叠统计；空批次不给出分数。

    每帧的 ``intersection``、``prediction_pixels``、``gt_pixels`` 先按像素求和，
    再据此计算整批 micro 指标；``iou`` / ``dice`` 则对帧求算术平均。

    @param rows 逐帧记录，需含 ``intersection``、``prediction_pixels``、
        ``gt_pixels``、``iou``、``dice``。
    @return 汇总字典，字段含义如下：
        - ``frames_evaluated``：参与评测的帧数。
        - ``intersection``：各帧预测与 GT 前景交集像素数之和。
        - ``prediction_pixels``：各帧预测前景像素数之和。
        - ``gt_pixels``：各帧 GT 前景像素数之和。
        - ``mean_iou``：逐帧 IoU 的算术平均；空批次为 ``None``。
          单帧 IoU 为交集 / 并集，预测与 GT 均为空时记 1。
        - ``mean_dice``：逐帧 Dice 的算术平均；空批次为 ``None``。
          单帧 Dice 为 ``2 * 交集 / (预测像素 + GT 像素)``，分母为 0 时记 1。
        - ``pixel_iou``：整批 micro IoU，
          ``sum(intersection) / (sum(prediction) + sum(GT) - sum(intersection))``；
          空批次为 ``None``，并集为 0 时记 1。大掩码帧权重大于小掩码帧。
        - ``pixel_dice``：整批 micro Dice，
          ``2 * sum(intersection) / (sum(prediction) + sum(GT))``；
          空批次为 ``None``，分母为 0 时记 1。同样按像素加权。
    """
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


def file_name(value: str) -> str:
    if not value or value in (".", "..") or any(c in value for c in "/\\*?[]"):
        raise argparse.ArgumentTypeError(
            "expected a single name, without paths or globs"
        )
    return value


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def add_gt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compare-gt", action="store_true", help="Compare predictions with GT"
    )
    parser.add_argument("--gt-dir-name", type=file_name, default="masks_sam3")
    parser.add_argument(
        "--gt-mask-name", type=file_name, help="GT filename; required with --compare-gt"
    )
    parser.add_argument(
        "--compare-skip-frames",
        type=nonnegative_int,
        default=0,
        help="Skip N frames after each evaluated frame; 2 selects frames 0, 3, 6, ...",
    )


def validate_gt_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.compare_gt and not args.gt_mask_name:
        parser.error("--gt-mask-name is required with --compare-gt")


class GTComparison:
    """Collect one batch's comparisons, failures and per-direction frame metrics."""

    def __init__(self, args: argparse.Namespace, directions: tuple[str, ...]) -> None:
        self.args = args
        self.sequences: list[Dict[str, Any]] = []
        self.rows: Dict[str, list] = {direction: [] for direction in directions}

    def compare(self, video: Path, output: Path, direction: str) -> None:
        if not self.args.compare_gt:
            return
        result = compare_masks(
            video,
            video.parent / self.args.gt_dir_name / self.args.gt_mask_name,
            output,
            self.args.max_frames,
            self.args.compare_skip_frames,
        )
        self.sequences.append({"direction": direction, **result["summary"]})
        self.rows[direction].extend(result["rows"])

    def record_failure(
        self, video: Path, output: Path, direction: str, error: Exception
    ) -> None:
        if not self.args.compare_gt:
            return
        failure = {
            "status": "failed",
            "input_video": str(video),
            "direction": direction,
            "error": str(error),
        }
        self.sequences.append(failure)
        for name in ("comparison.mp4", "gt_metrics.csv"):
            (output / name).unlink(missing_ok=True)
        write_json(output / "gt_metrics.json", failure)

    def write_summary(self, output_root: Path) -> None:
        if not self.args.compare_gt:
            return
        write_json(
            output_root / "gt_summary.json",
            {
                "status": "failed"
                if any(s["status"] == "failed" for s in self.sequences)
                else "success",
                "skip_frames": self.args.compare_skip_frames,
                "empty_masks_score": 1.0,
                "directions": {
                    direction: metric_summary(rows)
                    for direction, rows in self.rows.items()
                },
                "sequences": self.sequences,
            },
        )

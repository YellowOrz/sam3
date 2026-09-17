"""Compare aligned RGB and lossless foreground mask videos without a model."""

import argparse
import csv
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .video_utils import draw_geometry, MASK_ALPHA, probe_video, write_json

PREDICTION_COLOR = np.array((60, 220, 60), dtype=np.float64)
DETECTOR_COLOR = np.array((0, 165, 255), dtype=np.float64)


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
    summary = {
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
    if count and "detector_iou" in rows[0]:
        detector_intersection = sum(row["detector_intersection"] for row in rows)
        detector_prediction = sum(row["detector_prediction_pixels"] for row in rows)
        detector_union = detector_prediction + ground_truth - detector_intersection
        summary["detector_intersection"] = detector_intersection
        summary["detector_prediction_pixels"] = detector_prediction
        summary["detector_mean_iou"] = sum(row["detector_iou"] for row in rows) / count
        summary["detector_mean_dice"] = (
            sum(row["detector_dice"] for row in rows) / count
        )
        summary["detector_pixel_iou"] = (
            detector_intersection / detector_union if detector_union else 1.0
        )
        summary["detector_pixel_dice"] = (
            2 * detector_intersection / (detector_prediction + ground_truth)
            if detector_prediction + ground_truth
            else 1.0
        )
    return summary


def _overlap(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    intersection = int(np.count_nonzero(prediction & ground_truth))
    prediction_pixels = int(np.count_nonzero(prediction))
    gt_pixels = int(np.count_nonzero(ground_truth))
    union = prediction_pixels + gt_pixels - intersection
    total = prediction_pixels + gt_pixels
    return {
        "intersection": intersection,
        "prediction_pixels": prediction_pixels,
        "gt_pixels": gt_pixels,
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / total if total else 1.0,
    }


def _tint(panel: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
    panel[mask] = (panel[mask] * (1 - MASK_ALPHA) + color * MASK_ALPHA).astype(np.uint8)


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
    geometry_prompts: Optional[Dict[int, Dict[str, Any]]] = None,
    detector_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write comparison artifacts only after successful aligned decoding."""
    if skip_frames < 0:
        raise ValueError("skip_frames must be non-negative")
    prediction_path = output_dir / "masks.mkv"
    paths = [video_path, prediction_path]
    if detector_path is not None:
        paths.append(detector_path)
    paths.append(gt_path)
    panel_count = 4 if detector_path is not None else 3
    infos = [probe_video(path) for path in paths]
    source = infos[0]
    frame_count = source["frame_count"]
    if frame_count <= 0:
        raise ValueError(f"unknown or empty frame count: {video_path}")
    limit = min(frame_count, max_frames) if max_frames is not None else frame_count
    aligned = {prediction_path, detector_path} if detector_path else {prediction_path}
    for path, info in zip(paths[1:], infos[1:]):
        if any(info[key] != source[key] for key in ("width", "height")):
            raise ValueError(f"dimensions do not match RGB: {path}")
        if not math.isclose(info["fps"], source["fps"], rel_tol=1e-5):
            raise ValueError(f"FPS does not match RGB: {path}")
        expected = limit if path in aligned else frame_count
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
                (source["width"] * panel_count, source["height"]),
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
                    gt = foreground(frames[-1], gt_path)
                    detector = (
                        foreground(frames[2], detector_path)
                        if detector_path is not None
                        else None
                    )
                    if frame_index % stride:
                        continue
                    row = {
                        "frame_index": frame_index,
                        "time_seconds": frame_index / source["fps"],
                        **_overlap(prediction, gt),
                    }
                    if detector is not None:
                        detector_overlap = _overlap(detector, gt)
                        row.update(
                            detector_intersection=detector_overlap["intersection"],
                            detector_prediction_pixels=detector_overlap[
                                "prediction_pixels"
                            ],
                            detector_iou=detector_overlap["iou"],
                            detector_dice=detector_overlap["dice"],
                        )
                    rows.append(row)
                    panels = [frames[0].copy() for _ in range(panel_count)]
                    draw_geometry(
                        panels[0], (geometry_prompts or {}).get(frame_index, {})
                    )
                    if detector is None:
                        overlay_masks = (prediction, gt)
                        overlay_colors = (PREDICTION_COLOR, PREDICTION_COLOR)
                        titles = (
                            f"RGB frame={frame_index}",
                            f"Prediction IoU={row['iou']:.3f} Dice={row['dice']:.3f}",
                            "GT",
                        )
                    else:
                        overlay_masks = (detector, prediction, gt)
                        overlay_colors = (
                            DETECTOR_COLOR,
                            PREDICTION_COLOR,
                            PREDICTION_COLOR,
                        )
                        titles = (
                            f"RGB frame={frame_index}",
                            (
                                f"Detector IoU={row['detector_iou']:.3f} "
                                f"Dice={row['detector_dice']:.3f}"
                            ),
                            f"Prediction IoU={row['iou']:.3f} Dice={row['dice']:.3f}",
                            "GT",
                        )
                    for panel, mask, color in zip(
                        panels[1:], overlay_masks, overlay_colors
                    ):
                        _tint(panel, mask, color)
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
                    if (path in aligned or limit == frame_count) and capture.read()[0]:
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
            if detector_path is not None:
                summary["detector_video"] = str(detector_path)
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

    def compare(
        self,
        video: Path,
        output: Path,
        direction: str,
        geometry_prompts: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        if not self.args.compare_gt:
            return
        detector_path = (
            output / "detector_masks.mkv"
            if getattr(self.args, "save_detector", False)
            else None
        )
        result = compare_masks(
            video,
            video.parent / self.args.gt_dir_name / self.args.gt_mask_name,
            output,
            self.args.max_frames,
            self.args.compare_skip_frames,
            geometry_prompts,
            detector_path,
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

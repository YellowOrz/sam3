#!/usr/bin/env python3
"""Zero-training SAM3 bidirectional memory fusion.

Independent forward/backward sessions export encoded memories. Each frame reads
past forward and future backward memories with a fixed budget, then uses the
original SAM3 decoder once. The source banks are never updated by fused masks.

Examples:
    python scripts/process_bidirectional_videos.py --input-root DATA \
        --output-root OUT --prompt "left hand" --backward-mode physical
    python scripts/process_bidirectional_videos.py --input-root DATA \
        --output-root OUT --prompt "left hand" --chunk-frames 120 --context-frames 30

chunk-frames=0 processes the full sequence. Otherwise each independent window
contains a unique output core and optional context on either side. Outputs are
lossless masks, a four-panel comparison, per-frame memory provenance and optional
GT evaluation. No Viterbi or mask averaging is used. Standard SAM3 only, one GPU.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__:
    from scripts.process_dataset_videos import discover_color_videos, output_dir_for
    from scripts.video_utils import (
        as_numpy,
        expand_path,
        extract_png_frames,
        normalize_output_arrays,
        parse_device,
        positive_int,
        probe_video,
        utc_now,
        write_json,
    )
else:
    from process_dataset_videos import discover_color_videos, output_dir_for
    from video_utils import (
        as_numpy,
        expand_path,
        extract_png_frames,
        normalize_output_arrays,
        parse_device,
        positive_int,
        probe_video,
        utc_now,
        write_json,
    )

LOGGER = logging.getLogger("sam3_bidirectional_processor")
SCHEMA_VERSION = 2


@dataclass
class DirectionResult:
    name: str
    frames: List[Dict[int, np.ndarray]]
    detector_scores: List[Dict[int, float]]
    tracker_scores: List[Dict[int, float]]
    primary_obj_id: Optional[int] = None
    memories: Dict[int, Dict[int, Dict[str, Any]]] = field(default_factory=dict)


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return number


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def output_frame(
    outputs: Optional[Dict[str, Any]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, float]]:
    obj_ids, detector, masks = normalize_output_arrays(outputs)
    tracker = as_numpy((outputs or {}).get("out_tracker_probs")).reshape(-1)
    if len(tracker) not in (0, len(obj_ids)):
        raise RuntimeError("object/tracker score count mismatch")
    return (
        {
            int(obj_id): np.asarray(mask, dtype=bool).copy()
            for obj_id, mask in zip(obj_ids, masks)
        },
        {int(obj_id): float(score) for obj_id, score in zip(obj_ids, detector)},
        {int(obj_id): float(score) for obj_id, score in zip(obj_ids, tracker)},
    )


def reverse_frame_directory(
    frame_dir: Path, reverse_dir: Path, frame_count: int
) -> None:
    reverse_dir.mkdir()
    for reverse_index, source_index in enumerate(range(frame_count - 1, -1, -1)):
        source = frame_dir / f"{source_index:06d}.png"
        target = reverse_dir / f"{reverse_index:06d}.png"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def run_direction(
    predictor: Any,
    frame_dir: Path,
    frame_count: int,
    prompt: str,
    name: str,
    *,
    propagation_direction: str = "forward",
    reverse_index: bool = False,
    memory_dir: Optional[Path] = None,
) -> DirectionResult:
    """Run one isolated session, processing the prompt frame exactly once."""
    frames: List[Dict[int, np.ndarray]] = [{} for _ in range(frame_count)]
    detector_scores: List[Dict[int, float]] = [{} for _ in range(frame_count)]
    tracker_scores: List[Dict[int, float]] = [{} for _ in range(frame_count)]
    start_index = frame_count - 1 if propagation_direction == "backward" else 0
    session_id: Optional[str] = None
    memories = {}
    capture_started = False
    try:
        session = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frame_dir),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            }
        )
        session_id = session["session_id"]
        if memory_dir is not None:
            predictor.begin_memory_capture(session_id, memory_dir)
            capture_started = True
        prompt_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": start_index,
                "text": prompt,
            }
        )
        prompt_masks, prompt_detector, prompt_tracker = output_frame(
            prompt_response.get("outputs")
        )
        prompt_source_index = (
            frame_count - 1 - start_index if reverse_index else start_index
        )
        frames[prompt_source_index] = prompt_masks
        detector_scores[prompt_source_index] = prompt_detector
        tracker_scores[prompt_source_index] = prompt_tracker

        remaining_frames = frame_count - 1
        propagation_start = (
            start_index if propagation_direction == "backward" else start_index + 1
        )
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": propagation_start,
            "max_frame_num_to_track": remaining_frames,
        }
        responses = predictor.handle_stream_request(request) if remaining_frames else ()
        for response in responses:
            returned_index = int(response["frame_index"])
            source_index = (
                frame_count - 1 - returned_index if reverse_index else returned_index
            )
            if not 0 <= source_index < frame_count:
                continue
            masks, detector, tracker = output_frame(response.get("outputs"))
            frames[source_index] = masks
            detector_scores[source_index] = detector
            tracker_scores[source_index] = tracker
    finally:
        if session_id is not None:
            try:
                if capture_started:
                    exported = predictor.finish_memory_capture(session_id)
                    for index, records in exported.items():
                        mapped = frame_count - 1 - index if reverse_index else index
                        memories[mapped] = {
                            obj_id: {**entry, "frame_index": mapped, "direction": name}
                            for obj_id, entry in records.items()
                        }
            finally:
                predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
    return DirectionResult(
        name, frames, detector_scores, tracker_scores, memories=memories
    )


def track_ids(result: DirectionResult) -> List[int]:
    return sorted({obj_id for frame in result.frames for obj_id in frame})


def object_quality(result: DirectionResult, obj_id: int) -> float:
    scores = []
    visible = 0
    for masks, detector, tracker in zip(
        result.frames, result.detector_scores, result.tracker_scores
    ):
        mask = masks.get(obj_id)
        if mask is None or not mask.any():
            continue
        visible += 1
        scores.append(tracker.get(obj_id, detector.get(obj_id, 0.0)))
    if not result.frames:
        return 0.0
    return 0.75 * (float(np.mean(scores)) if scores else 0.0) + 0.25 * visible / len(
        result.frames
    )


def pair_track_iou(
    forward: DirectionResult,
    forward_id: int,
    backward: DirectionResult,
    backward_id: int,
) -> float:
    values = []
    for forward_frame, backward_frame in zip(forward.frames, backward.frames):
        forward_mask = forward_frame.get(forward_id)
        backward_mask = backward_frame.get(backward_id)
        if forward_mask is None or backward_mask is None:
            continue
        if forward_mask.any() or backward_mask.any():
            values.append(mask_iou(forward_mask, backward_mask))
    return float(np.mean(values)) if values else 0.0


def match_primary_tracks(
    forward: DirectionResult, backward: DirectionResult
) -> Tuple[Optional[int], Optional[int], List[Dict[str, float]]]:
    """Match independent IDs using temporal mask overlap plus per-track quality."""
    pairs = []
    for forward_id in track_ids(forward):
        for backward_id in track_ids(backward):
            overlap = pair_track_iou(forward, forward_id, backward, backward_id)
            quality = 0.5 * (
                object_quality(forward, forward_id)
                + object_quality(backward, backward_id)
            )
            pairs.append(
                {
                    "forward_id": float(forward_id),
                    "backward_id": float(backward_id),
                    "mean_iou": overlap,
                    "quality": quality,
                    "rank_score": overlap + 0.25 * quality,
                }
            )
    pairs.sort(key=lambda item: item["rank_score"], reverse=True)
    if pairs:
        best = pairs[0]
        return int(best["forward_id"]), int(best["backward_id"]), pairs
    forward_ids = track_ids(forward)
    backward_ids = track_ids(backward)
    best_forward = max(
        forward_ids, key=lambda obj: object_quality(forward, obj), default=None
    )
    best_backward = max(
        backward_ids, key=lambda obj: object_quality(backward, obj), default=None
    )
    return best_forward, best_backward, pairs


def compare_direction_results(
    physical: DirectionResult,
    api: DirectionResult,
    shape: Tuple[int, int],
    minimum_iou: float,
) -> Dict[str, Any]:
    physical_id, api_id, _ = match_primary_tracks(physical, api)
    physical.primary_obj_id = physical_id
    api.primary_obj_id = api_id
    physical_masks, physical_scores = primary_track(physical, shape)
    api_masks, api_scores = primary_track(api, shape)
    ious = [mask_iou(first, second) for first, second in zip(physical_masks, api_masks)]
    presence_mismatches = int(
        sum(
            first.any() != second.any()
            for first, second in zip(physical_masks, api_masks)
        )
    )
    score_differences = [
        abs(first - second) for first, second in zip(physical_scores, api_scores)
    ]
    return {
        "equivalent": presence_mismatches == 0
        and min(ious, default=1.0) >= minimum_iou
        and max(score_differences, default=0.0) <= 1e-4,
        "minimum_required_iou": minimum_iou,
        "mean_mask_iou": float(np.mean(ious)) if ious else 1.0,
        "minimum_mask_iou": min(ious, default=1.0),
        "presence_mismatches": presence_mismatches,
        "maximum_tracker_score_difference": max(score_differences, default=0.0),
        "physical_object_id": physical_id,
        "api_object_id": api_id,
    }


def primary_track(
    result: DirectionResult, shape: Tuple[int, int]
) -> Tuple[List[np.ndarray], List[float]]:
    masks = []
    scores = []
    for frame_masks, detector, tracker in zip(
        result.frames, result.detector_scores, result.tracker_scores
    ):
        mask = (
            frame_masks.get(result.primary_obj_id)
            if result.primary_obj_id is not None
            else None
        )
        masks.append(np.zeros(shape, dtype=bool) if mask is None else mask)
        if result.primary_obj_id is None:
            scores.append(0.0)
        else:
            scores.append(
                float(
                    tracker.get(
                        result.primary_obj_id,
                        detector.get(result.primary_obj_id, 0.0),
                    )
                )
            )
    return masks, scores


def write_label_video(path: Path, masks: Sequence[np.ndarray], fps: float) -> None:
    if not masks:
        raise ValueError("cannot write empty mask video")
    height, width = masks[0].shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height), isColor=False
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create lossless mask video: {path}")
    try:
        for mask in masks:
            writer.write(mask.astype(np.uint8))
    finally:
        writer.release()


def overlay(
    frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]
) -> np.ndarray:
    result = frame.copy()
    if mask.any():
        result[mask] = (
            result[mask].astype(np.float32) * 0.65
            + np.asarray(color, dtype=np.float32) * 0.35
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    return result


def header(panel: np.ndarray, text: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def write_result_video(
    path: Path,
    frame_dir: Path,
    forward_masks: Sequence[np.ndarray],
    backward_masks: Sequence[np.ndarray],
    fused_masks: Sequence[np.ndarray],
    statuses: Sequence[str],
    fps: float,
) -> None:
    height, width = forward_masks[0].shape
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * 2)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create result video: {path}")
    try:
        for index, (forward, backward, fused) in enumerate(
            zip(forward_masks, backward_masks, fused_masks)
        ):
            frame = cv2.imread(str(frame_dir / f"{index:06d}.png"), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"cannot read extracted frame {index}")
            forward_panel = overlay(frame, forward, (0, 80, 255))
            backward_panel = overlay(frame, backward, (255, 100, 0))
            fused_panel = overlay(frame, fused, (0, 220, 120))
            header(forward_panel, f"Forward  frame={index}")
            header(backward_panel, "Backward")
            header(
                fused_panel,
                f"Memory fusion  {statuses[index]}",
            )

            diagnostic = (frame.astype(np.float32) * 0.35).astype(np.uint8)
            overlap = np.logical_and(forward, backward)
            forward_only = np.logical_and(forward, ~backward)
            backward_only = np.logical_and(backward, ~forward)
            diagnostic[overlap] = (0, 190, 0)
            diagnostic[forward_only] = (0, 0, 255)
            diagnostic[backward_only] = (255, 0, 0)
            contours, _ = cv2.findContours(
                fused.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(diagnostic, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
            header(
                diagnostic,
                f"Difference  IoU={mask_iou(forward, backward):.2f}",
            )
            writer.write(
                np.concatenate(
                    (
                        np.concatenate((forward_panel, backward_panel), axis=1),
                        np.concatenate((fused_panel, diagnostic), axis=1),
                    ),
                    axis=0,
                )
            )
    finally:
        writer.release()


def read_label_video(
    path: Path, frame_count: int, shape: Tuple[int, int], label: int
) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    masks = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open ground-truth video: {path}")
        while len(masks) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            gray = frame if frame.ndim == 2 else frame[:, :, 0]
            if gray.shape != shape:
                raise RuntimeError(
                    f"ground-truth dimensions differ: {gray.shape} != {shape}"
                )
            masks.append(gray == label)
    finally:
        capture.release()
    if len(masks) != frame_count:
        raise RuntimeError(
            f"ground-truth frame count differs: {len(masks)} != {frame_count}"
        )
    return masks


def boundary_f_score(
    prediction: np.ndarray, target: np.ndarray, tolerance: int = 2
) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    prediction_boundary = np.logical_xor(
        prediction,
        cv2.erode(prediction.astype(np.uint8), kernel).astype(bool),
    )
    target_boundary = np.logical_xor(
        target, cv2.erode(target.astype(np.uint8), kernel).astype(bool)
    )
    if not prediction_boundary.any() and not target_boundary.any():
        return 1.0
    size = tolerance * 2 + 1
    tolerance_kernel = np.ones((size, size), dtype=np.uint8)
    target_dilated = cv2.dilate(
        target_boundary.astype(np.uint8), tolerance_kernel
    ).astype(bool)
    prediction_dilated = cv2.dilate(
        prediction_boundary.astype(np.uint8), tolerance_kernel
    ).astype(bool)
    precision = np.logical_and(prediction_boundary, target_dilated).sum() / max(
        int(prediction_boundary.sum()), 1
    )
    recall = np.logical_and(target_boundary, prediction_dilated).sum() / max(
        int(target_boundary.sum()), 1
    )
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def evaluate_masks(
    ground_truth: Sequence[np.ndarray], methods: Mapping[str, Sequence[np.ndarray]]
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"methods": {}}
    per_method = {}
    for name, masks in methods.items():
        j = [mask_iou(mask, target) for mask, target in zip(masks, ground_truth)]
        f = [
            boundary_f_score(mask, target) for mask, target in zip(masks, ground_truth)
        ]
        jf = [(j_value + f_value) / 2 for j_value, f_value in zip(j, f)]
        per_method[name] = jf
        report["methods"][name] = {
            "J": float(np.mean(j)),
            "F": float(np.mean(f)),
            "J_and_F": float(np.mean(jf)),
            "worst_10_percent_J_and_F": float(
                np.mean(sorted(jf)[: max(1, math.ceil(len(jf) * 0.1))])
            ),
        }
    oracle = [
        max(per_method["forward"][i], per_method["backward"][i])
        for i in range(len(ground_truth))
    ]
    report["per_frame_oracle_J_and_F"] = float(np.mean(oracle))
    report["best_single_direction_J_and_F"] = max(
        report["methods"]["forward"]["J_and_F"],
        report["methods"]["backward"]["J_and_F"],
    )
    report["per_video_oracle_J_and_F"] = report["best_single_direction_J_and_F"]
    return report


def bidirectional_audit(
    forward_masks: Sequence[np.ndarray], backward_masks: Sequence[np.ndarray]
) -> Dict[str, Any]:
    ious = [
        mask_iou(first, second) for first, second in zip(forward_masks, backward_masks)
    ]
    low_conflict = [value < 0.5 for value in ious]
    longest_run = run = 0
    for conflict in low_conflict:
        run = run + 1 if conflict else 0
        longest_run = max(longest_run, run)
    return {
        "frame_count": len(ious),
        "mean_forward_backward_iou": float(np.mean(ious)) if ious else 1.0,
        "median_forward_backward_iou": float(np.median(ious)) if ious else 1.0,
        "frames_below_iou_0_5": sum(low_conflict),
        "longest_run_below_iou_0_5": longest_run,
        "both_empty_frames": int(
            sum(
                not first.any() and not second.any()
                for first, second in zip(forward_masks, backward_masks)
            )
        ),
        "presence_disagreement_frames": int(
            sum(
                first.any() != second.any()
                for first, second in zip(forward_masks, backward_masks)
            )
        ),
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def config_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def resolve_ground_truth(
    root: Optional[Path], relative_parent: Path, single_video: bool
) -> Optional[Path]:
    if root is None:
        return None
    if root.is_file():
        if not single_video:
            raise ValueError(
                "a ground-truth file can only be used with one input video"
            )
        return root
    candidate = root / relative_parent / "masks.mkv"
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class MemoryConfig:
    chunk_frames: int = 0
    context_frames: int = 0
    spatial_frames: int = 6
    pointer_frames: int = 16
    side: str = "both"
    min_quality: float = 0.0
    min_match_iou: float = 0.1

    def __post_init__(self):
        if self.chunk_frames < 0 or self.context_frames < 0:
            raise ValueError("chunk/context frames must be non-negative")
        if self.chunk_frames == 0 and self.context_frames:
            raise ValueError("--context-frames requires --chunk-frames")
        if self.spatial_frames < 1 or self.pointer_frames < 0:
            raise ValueError(
                "spatial frames must be positive; pointer frames non-negative"
            )
        if self.side not in {"both", "past", "future"}:
            raise ValueError("memory side must be both, past or future")
        if not 0 <= self.min_quality <= 1 or not 0 <= self.min_match_iou <= 1:
            raise ValueError("quality and match thresholds must be in [0, 1]")


def processing_windows(frame_count: int, config: MemoryConfig):
    """Yield (core_start, core_end, window_start, window_end), half-open."""
    if frame_count < 1:
        raise ValueError("Video contains no frames")
    step = config.chunk_frames or frame_count
    for start in range(0, frame_count, step):
        end = min(start + step, frame_count)
        yield (
            start,
            end,
            max(0, start - config.context_frames),
            min(frame_count, end + config.context_frames),
        )


def window_frame_directory(source: Path, target: Path, start: int, end: int):
    target.mkdir()
    for local, original in enumerate(range(start, end)):
        src, dst = source / f"{original:06d}.png", target / f"{local:06d}.png"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def select_memories(entries, frame_idx, budget, config):
    """Select disjoint source frames; split a fixed budget then refill spare slots.

    Quality is predicted IoU, not presence: low visibility is not automatically
    an error. Unknown quality is admitted only when no positive cutoff was set.
    """
    pools = {"F": [], "B": []}
    seen = set()
    for entry in entries:
        direction = entry["direction"]
        index = entry["frame_index"]
        if direction not in pools or index == frame_idx:
            continue
        if (direction == "F" and index > frame_idx) or (
            direction == "B" and index < frame_idx
        ):
            continue
        if config.side == "past" and direction != "F":
            continue
        if config.side == "future" and direction != "B":
            continue
        quality = entry.get("quality")
        if quality is None or not math.isfinite(quality):
            if config.min_quality > 0:
                continue
        elif quality < config.min_quality:
            continue
        key = (index, direction, entry["object_id"])
        if key in seen:
            continue
        seen.add(key)
        pools[direction].append(entry)

    def rank(entry):
        quality = entry.get("quality")
        quality = quality if quality is not None and math.isfinite(quality) else -1.0
        return (
            not entry["conditioning"],
            abs(frame_idx - entry["frame_index"]),
            -quality,
        )

    for pool in pools.values():
        pool.sort(key=rank)
    if budget == 0:
        return []
    past_slots = (budget + 1) // 2
    chosen = pools["F"][:past_slots] + pools["B"][: budget // 2]
    remainder = pools["F"][past_slots:] + pools["B"][budget // 2 :]
    chosen.extend(sorted(remainder, key=rank)[: budget - len(chosen)])
    return sorted(chosen, key=lambda item: (item["direction"], item["frame_index"]))


def memory_diagnostics(entries, offset):
    return [
        {
            key: (value + offset if key == "frame_index" else value)
            for key, value in entry.items()
            if key != "path"
        }
        for entry in entries
    ]


def run_memory_window(
    predictor,
    frame_dir,
    work_dir,
    count,
    prompt,
    shape,
    config,
    backward_mode,
    equivalence_iou,
    core_start,
    core_end,
):
    """Build isolated banks, then decode the core against immutable snapshots."""
    forward = run_direction(
        predictor, frame_dir, count, prompt, "F", memory_dir=work_dir / "memory_forward"
    )
    physical = api = None
    if backward_mode in {"physical", "verify"}:
        reverse_dir = work_dir / "frames_reversed"
        reverse_frame_directory(frame_dir, reverse_dir, count)
        physical = run_direction(
            predictor,
            reverse_dir,
            count,
            prompt,
            "B",
            reverse_index=True,
            memory_dir=work_dir / "memory_backward",
        )
    if backward_mode in {"api", "verify"}:
        api = run_direction(
            predictor,
            frame_dir,
            count,
            prompt,
            "B",
            propagation_direction="backward",
            memory_dir=work_dir / "memory_api",
        )
    equivalence = None
    if backward_mode == "verify":
        equivalence = compare_direction_results(physical, api, shape, equivalence_iou)
        equivalence["scope"] = "public masks and scores only; not memory equivalence"
        # Always retain physical source memories. Public-output equality does not
        # prove internal pointer/position/memory equality.
        if not equivalence["equivalent"]:
            write_json(work_dir.parent / "backward_equivalence.json", equivalence)
            raise RuntimeError(
                "Backward implementations differ; inspect backward_equivalence.json "
                "and explicitly choose --backward-mode physical or api"
            )
    backward = physical if physical is not None else api
    forward_id, backward_id, matches = match_primary_tracks(forward, backward)
    forward.primary_obj_id, backward.primary_obj_id = forward_id, backward_id
    forward_masks, forward_scores = primary_track(forward, shape)
    backward_masks, backward_scores = primary_track(backward, shape)
    identity_conflict = bool(matches and matches[0]["mean_iou"] < config.min_match_iou)
    entries = []
    for direction in (forward, backward):
        if direction.primary_obj_id is not None:
            for records in direction.memories.values():
                entry = records.get(direction.primary_obj_id)
                if entry is not None:
                    entries.append(entry)
    source_count = len(entries)
    if source_count == 0 and (forward_id is not None or backward_id is not None):
        raise RuntimeError(
            "Detected target has no exported memories; check the SAM3 capture path"
        )

    fused, diagnostics = [], []
    session_id = None
    try:
        session_id = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(frame_dir),
                "offload_video_to_cpu": True,
            }
        )["session_id"]
        for index in range(core_start, core_end):
            spatial = select_memories(entries, index, config.spatial_frames, config)
            pointers = select_memories(entries, index, config.pointer_frames, config)
            if identity_conflict or not spatial:
                # Empty output is an abstention, not a declaration of invisibility.
                # Never rescue this case by copying a direction's final mask.
                result = {
                    "mask": np.zeros(shape, dtype=bool),
                    "spatial_tokens": 0,
                    "pointer_tokens": 0,
                }
                status = "identity_conflict" if identity_conflict else "no_memory"
                spatial, pointers = [], []
            else:
                result = predictor.decode_memory_frame(
                    session_id, index, spatial, pointers
                )
                status = "decoded"
            mask = np.asarray(result.pop("mask"), dtype=bool)
            if mask.shape != shape:
                raise RuntimeError(f"Decoded mask dimensions {mask.shape} != {shape}")
            fused.append(mask)
            diagnostics.append(
                {
                    "frame_index": index,
                    "status": status,
                    "requires_review": status != "decoded",
                    "forward_score": forward_scores[index],
                    "backward_score": backward_scores[index],
                    "forward_backward_iou": mask_iou(
                        forward_masks[index], backward_masks[index]
                    ),
                    "spatial_memory": memory_diagnostics(spatial, 0),
                    "pointer_memory": memory_diagnostics(pointers, 0),
                    **result,
                }
            )
            if (index - core_start + 1) % 50 == 0:
                LOGGER.info(
                    "Memory decoded %d/%d core frames",
                    index - core_start + 1,
                    core_end - core_start,
                )
    finally:
        if session_id is not None:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
    report = {
        "primary_instances": {"forward": forward_id, "backward": backward_id},
        "instance_matches": matches,
        "identity_conflict": identity_conflict,
        "source_memory_count": source_count,
        "backward_equivalence": equivalence,
        "source_memory_bytes": sum(
            path.stat().st_size for path in work_dir.glob("memory_*/*.pt")
        ),
    }
    return (
        forward_masks[core_start:core_end],
        backward_masks[core_start:core_end],
        fused,
        diagnostics,
        report,
    )


def process_video(
    predictor,
    video_path,
    input_root,
    output_dir,
    prompt,
    config,
    *,
    max_frames=None,
    overwrite=False,
    ground_truth_path=None,
    ground_truth_label=1,
    backward_mode="physical",
    equivalence_iou=0.999,
    checkpoint_identity=None,
):
    if not all(
        callable(getattr(predictor, name, None))
        for name in (
            "begin_memory_capture",
            "finish_memory_capture",
            "decode_memory_frame",
        )
    ):
        raise ValueError(
            "Memory fusion requires the updated standard SAM3 single-GPU predictor"
        )
    video_info = probe_video(video_path, fallback_fps=30.0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": "bidirectional_memory",
        "input_video": str(video_path),
        "input_size": video_path.stat().st_size,
        "input_mtime_ns": video_path.stat().st_mtime_ns,
        "prompt": prompt,
        "model_version": "sam3",
        "checkpoint": checkpoint_identity,
        "max_frames": max_frames,
        "memory": asdict(config),
        "ground_truth": str(ground_truth_path) if ground_truth_path else None,
        "ground_truth_mtime_ns": (
            ground_truth_path.stat().st_mtime_ns if ground_truth_path else None
        ),
        "ground_truth_label": ground_truth_label,
        "backward_mode": backward_mode,
        "backward_equivalence_iou": equivalence_iou,
    }
    fingerprint = config_hash(payload)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("config_hash") == fingerprint:
            return "skipped"
        raise FileExistsError(
            "Output exists with incomplete/different configuration; use a new output root or --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # These optional reports must not survive from a different run/config.
        for name in (
            "evaluation.json",
            "backward_equivalence.json",
            "forward/frames.jsonl",
            "backward/frames.jsonl",
        ):
            (output_dir / name).unlink(missing_ok=True)
    metadata = {
        **payload,
        "config_hash": fingerprint,
        "status": "processing",
        "started_at": utc_now(),
    }
    write_json(metadata_path, metadata)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="sam3_memory_") as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            count = extract_png_frames(video_path, frames, max_frames)
            shape = (int(video_info["height"]), int(video_info["width"]))
            forward_masks, backward_masks, fused_masks, diagnostics, reports = (
                [],
                [],
                [],
                [],
                [],
            )
            for block, (core_start, core_end, start, end) in enumerate(
                processing_windows(count, config)
            ):
                LOGGER.info(
                    "Window %d: context [%d,%d), output [%d,%d)",
                    block,
                    start,
                    end,
                    core_start,
                    core_end,
                )
                with tempfile.TemporaryDirectory(
                    prefix=f"block_{block}_", dir=root
                ) as block_dir:
                    work = Path(block_dir)
                    local_frames = work / "frames"
                    window_frame_directory(frames, local_frames, start, end)
                    try:
                        f, b, fused, records, report = run_memory_window(
                            predictor,
                            local_frames,
                            work,
                            end - start,
                            prompt,
                            shape,
                            config,
                            backward_mode,
                            equivalence_iou,
                            core_start - start,
                            core_end - start,
                        )
                    except Exception:
                        equivalence_path = root / "backward_equivalence.json"
                        if equivalence_path.exists():
                            shutil.copy2(
                                equivalence_path,
                                output_dir / "backward_equivalence.json",
                            )
                        raise
                    forward_masks.extend(f)
                    backward_masks.extend(b)
                    fused_masks.extend(fused)
                    for record in records:
                        record["frame_index"] += start
                        record["window_index"] = block
                        for key in ("spatial_memory", "pointer_memory"):
                            for entry in record[key]:
                                entry["frame_index"] += start
                        diagnostics.append(record)
                    reports.append(
                        {
                            "window_index": block,
                            "window": [start, end],
                            "core": [core_start, core_end],
                            **report,
                        }
                    )
            fps = float(video_info["fps"])
            for name, masks in (
                ("forward", forward_masks),
                ("backward", backward_masks),
            ):
                (output_dir / name).mkdir(exist_ok=True)
                write_label_video(output_dir / name / "masks.mkv", masks, fps)
            write_label_video(output_dir / "masks.mkv", fused_masks, fps)
            write_result_video(
                output_dir / "result.mp4",
                frames,
                forward_masks,
                backward_masks,
                fused_masks,
                [record["status"] for record in diagnostics],
                fps,
            )
            write_jsonl(output_dir / "frames.jsonl", diagnostics)
            audit = bidirectional_audit(forward_masks, backward_masks)
            write_json(output_dir / "audit.json", audit)
            evaluation = None
            if ground_truth_path is not None:
                gt = read_label_video(
                    ground_truth_path, count, shape, ground_truth_label
                )
                methods = {
                    "forward": forward_masks,
                    "backward": backward_masks,
                    "memory_fused": fused_masks,
                }
                evaluation = evaluate_masks(gt, methods)
                for name, present in (("visible", True), ("invisible", False)):
                    indices = [
                        i for i, mask in enumerate(gt) if bool(mask.any()) == present
                    ]
                    evaluation[name] = (
                        evaluate_masks(
                            [gt[i] for i in indices],
                            {
                                method: [masks[i] for i in indices]
                                for method, masks in methods.items()
                            },
                        )
                        if indices
                        else None
                    )
                evaluation["oracle_scope"] = (
                    "selection among F/B only; not an upper bound for memory fusion"
                )
                write_json(output_dir / "evaluation.json", evaluation)
            metadata.update(
                {
                    "status": "success",
                    "completed_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "frames_processed": count,
                    "windows": reports,
                    "requires_review_frames": sum(
                        record["requires_review"] for record in diagnostics
                    ),
                    "source_banks_frozen": True,
                    "training": False,
                    "temporal_encoding": "existing unsigned distance codes; no learned direction embedding",
                    "cross_window_memory": False,
                    "identity_protocol": "text prompt and within-window mask association; cross-window identity not guaranteed",
                    "frame_passes": sum(
                        report["window"][1] - report["window"][0] for report in reports
                    )
                    * (3 if backward_mode == "verify" else 2)
                    + count,
                    "source_memory_storage": "temporary CPU tensor files, removed after each window",
                    "audit": audit,
                    "evaluation": evaluation,
                    "mask_labels": "binary target mask, 1=foreground; inspect requires_review for abstentions",
                }
            )
        write_json(metadata_path, metadata)
        return "success"
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )
        write_json(metadata_path, metadata)
        raise


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="不训练 SAM3：独立双向建库、联合 memory attention、单次最终解码。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--version", choices=["sam3"], default="sam3", help="新融合路径仅支持基础 SAM3"
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt",
    )
    parser.add_argument("--device", type=parse_device, default=("cuda:0", 0))
    parser.add_argument("--max-sequences", type=positive_int)
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument(
        "--chunk-frames",
        type=nonnegative_int,
        default=0,
        help="每块输出核心帧数；0 表示全序列",
    )
    parser.add_argument(
        "--context-frames",
        type=nonnegative_int,
        default=0,
        help="每个核心前后附加上下文帧数；重叠区不融合结果",
    )
    parser.add_argument(
        "--memory-frames",
        type=positive_int,
        default=6,
        help="两侧空间记忆总帧数，含条件帧",
    )
    parser.add_argument(
        "--pointer-frames",
        type=nonnegative_int,
        default=16,
        help="两侧 object pointer 总帧数",
    )
    parser.add_argument(
        "--memory-side",
        choices=["both", "past", "future"],
        default="both",
        help="相同预算的方向消融",
    )
    parser.add_argument(
        "--memory-min-quality",
        type=unit_interval,
        default=0.0,
        help="predicted IoU 筛选阈值；0 允许质量未知条目",
    )
    parser.add_argument(
        "--min-match-iou",
        type=unit_interval,
        default=0.1,
        help="两路实例关联最低 IoU；不满足则标记复核，不混合身份",
    )
    parser.add_argument(
        "--backward-mode", choices=["physical", "api", "verify"], default="physical"
    )
    parser.add_argument("--backward-equivalence-iou", type=unit_interval, default=0.999)
    parser.add_argument("--ground-truth-root")
    parser.add_argument("--ground-truth-label", type=positive_int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        config = MemoryConfig(
            args.chunk_frames,
            args.context_frames,
            args.memory_frames,
            args.pointer_frames,
            args.memory_side,
            args.memory_min_quality,
            args.min_match_iou,
        )
    except ValueError as exc:
        parser.error(str(exc))
    input_root, output_root = expand_path(args.input_root), expand_path(
        args.output_root
    )
    if not input_root.is_dir():
        parser.error(f"Input root is not a directory: {input_root}")
    if input_root == output_root or input_root in output_root.parents:
        parser.error("Output root must be outside the input tree")
    videos = discover_color_videos(input_root)
    if args.max_sequences:
        videos = videos[: args.max_sequences]
    if not videos:
        parser.error(f"No color.mp4 found below {input_root}")
    if args.list_only:
        for video in videos:
            print(video.relative_to(input_root))
        return 0
    checkpoint = expand_path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    gt_root = expand_path(args.ground_truth_root) if args.ground_truth_root else None
    if gt_root is not None and not gt_root.exists():
        parser.error(f"Ground truth does not exist: {gt_root}")
    import torch

    device_name, device_index = args.device
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        parser.error(f"Requested CUDA device is unavailable: {device_name}")
    torch.cuda.set_device(device_index)
    from sam3 import build_sam3_predictor

    predictor = build_sam3_predictor(
        version="sam3",
        checkpoint_path=str(checkpoint),
        compile=False,
        async_loading_frames=True,
    )
    counts = {"success": 0, "skipped": 0, "failed": 0}
    try:
        for video in videos:
            try:
                gt = resolve_ground_truth(
                    gt_root, video.parent.relative_to(input_root), len(videos) == 1
                )
                if gt_root is not None and gt is None:
                    raise FileNotFoundError(f"No matching GT masks.mkv for {video}")
                status = process_video(
                    predictor,
                    video,
                    input_root,
                    output_dir_for(video, input_root, output_root),
                    args.prompt,
                    config,
                    max_frames=args.max_frames,
                    overwrite=args.overwrite,
                    ground_truth_path=gt,
                    ground_truth_label=args.ground_truth_label,
                    backward_mode=args.backward_mode,
                    equivalence_iou=args.backward_equivalence_iou,
                    checkpoint_identity={
                        "path": str(checkpoint),
                        "size": checkpoint.stat().st_size,
                        "mtime_ns": checkpoint.stat().st_mtime_ns,
                    },
                )
                counts[status] += 1
            except Exception:
                counts["failed"] += 1
                LOGGER.exception("Sequence failed: %s", video)
    finally:
        predictor.shutdown()
    LOGGER.info("Finished: %s", counts)
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())

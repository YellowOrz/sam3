#!/usr/bin/env python3
"""使用独立的正序、倒序 SAM 推理结果汇总目标手分割。

处理流程：

1. 递归查找输入目录中的 ``color.mp4``；如果输入目录本身包含该文件，则只处理
   这一段视频。
2. 正序从第一帧处理到最后一帧；倒序从最后一帧处理到第一帧。两个方向使用
   完全独立的 session，不共享 tracker 状态或实例 ID。``add_prompt`` 的结果作为
   锚点帧结果直接保存，后续传播从相邻帧开始，保证每帧只处理一次。正式批处理
   推荐 ``--backward-mode physical``，即真正反向编号后再做 forward propagation。
3. 根据整段视频的 mask 重叠和质量分数匹配两个方向的目标实例。
4. 使用 Viterbi 在 Forward、Backward 和目标不可见三种状态间选择时序稳定的
   最终路径。可选的 P0-C 会在高不确定区间两侧寻找可信锚点，建立新 session
   向区间内部重传播；新候选未通过验收时不会替换原结果。

Viterbi 汇总逻辑：

* 每一帧有三个候选状态：``F`` 使用正序 mask，``B`` 使用倒序 mask，``O`` 表示
  目标不可见并输出空 mask。它不是对 F/B 做逐像素平均、并集或交集，而是为每帧
  选择一张完整候选 mask。
* F/B 的“当前帧分数”（发射分数）由模型置信度和面积稳健性组成。面积越接近该
  方向整段视频的非空 mask 面积中位数，面积分越高；空的 F/B 候选不可选择。
* O 是保守的空目标状态：只有 F/B 都为空，或所有非空候选的最高模型分数不超过
  ``--empty-score-threshold`` 时才允许选择，防止轻易把目标标成不可见。
* 相邻帧的“连续性分数”（转移分数）奖励 mask IoU 高、质心移动小的路径；从
  F 切到 B、从 B 切到 F 或进入/离开 O 都会受到惩罚，避免逐帧贪心造成闪烁。
* 动态规划会累计整段视频的发射分数和转移分数，保存每个状态的最佳前驱，最后
  从末帧回溯得到全局总分最高的状态序列。因此某一帧不一定选择当帧分数最高的
  方向，而会兼顾前后帧的一致性。
* ``frames.jsonl`` 记录最终选择的状态、各候选分数、方向切换原因和不确定性；
  ``result.mp4`` 的 Fused 面板显示 Viterbi 选择结果，Difference 面板显示 F/B
  分歧，便于检查切换是否合理。

默认的 ``--backward-mode verify`` 会同时运行两种倒序实现：

* ``physical``：将无损临时帧真正反向编号，再通过 forward API 处理；
* ``api``：保持原帧编号，调用 SAM 的 backward API。

两种结果达到等价门槛时采用较简单的 API backward 结果；不等价时脚本停止，
并保存 ``backward_equivalence.json`` 及两套候选供人工选择。临时 PNG 只用于
模型加载，任务结束后自动删除，不会在输出目录生成 PNG mask。

基本用法：

    python scripts/process_bidirectional_videos.py \
        --input-root DATA \
        --output-root OUT \
        --prompt "left hand" \
        --version sam3 \
        --device cuda:0 \
        --backward-mode physical

使用交互分割产生的 FFV1 ``masks.mkv`` 评测：

    python scripts/process_bidirectional_videos.py \
        --input-root DATA \
        --output-root OUT \
        --prompt "right hand" \
        --ground-truth-root GT_ROOT \
        --ground-truth-label 1

校准阈值后开启保守双锚点修复：

    python scripts/process_bidirectional_videos.py \
        --input-root DATA \
        --output-root OUT \
        --prompt "left hand" \
        --repair-uncertain

主要输出：

* ``forward/masks.mkv``、``backward/masks.mkv``：两个方向的全部实例标签；
* ``masks.mkv``：汇总后的目标实例二值 mask；
* ``result.mp4``：Forward、Backward、Fused、Difference 的 2×2 对比视频；
* ``frames.jsonl``：逐帧选择方向、分数、切换原因和不确定性；
* ``audit.json``：无需 GT 的正反向分歧统计；
* ``evaluation.json``：提供 GT 时生成的 J、F、J&F 和 oracle 指标；
* ``metadata.json``：完整参数、输出结构、实例匹配和修复记录。
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
from dataclasses import asdict, dataclass
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
    from video_utils import (  # type: ignore[no-redef]
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
SCHEMA_VERSION = 1
SOURCES = ("F", "B", "O")


@dataclass(frozen=True)
class FusionConfig:
    model_weight: float = 1.0
    shape_weight: float = 0.25
    temporal_iou_weight: float = 1.25
    centroid_weight: float = 0.25
    switch_penalty: float = 0.20
    empty_transition_penalty: float = 0.35
    empty_score_threshold: float = 0.20
    disagreement_iou_threshold: float = 0.35
    disagreement_min_frames: int = 5
    recovery_iou_threshold: float = 0.60
    recovery_min_frames: int = 3
    anchor_iou_threshold: float = 0.80
    anchor_score_threshold: float = 0.60
    anchor_search_frames: int = 90
    repair_min_gain: float = 0.10


@dataclass
class DirectionResult:
    name: str
    frames: List[Dict[int, np.ndarray]]
    detector_scores: List[Dict[int, float]]
    tracker_scores: List[Dict[int, float]]
    primary_obj_id: Optional[int] = None


@dataclass(frozen=True)
class PathResult:
    sources: Tuple[str, ...]
    score: float
    frame_scores: Tuple[Dict[str, float], ...]


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def stability_from_mask(mask: np.ndarray) -> float:
    """A geometry proxy, not SAM's internal logit stability score."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    return float(np.clip(4.0 * math.pi * area / max(perimeter * perimeter, 1.0), 0, 1))


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
) -> DirectionResult:
    """Run one isolated session, processing the prompt frame exactly once."""
    frames: List[Dict[int, np.ndarray]] = [{} for _ in range(frame_count)]
    detector_scores: List[Dict[int, float]] = [{} for _ in range(frame_count)]
    tracker_scores: List[Dict[int, float]] = [{} for _ in range(frame_count)]
    start_index = frame_count - 1 if propagation_direction == "backward" else 0
    session_id: Optional[str] = None
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
        if remaining_frames == 0:
            return DirectionResult(name, frames, detector_scores, tracker_scores)
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
        for response in predictor.handle_stream_request(request):
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
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
    return DirectionResult(name, frames, detector_scores, tracker_scores)


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


def shape_scores(masks: Sequence[np.ndarray]) -> List[float]:
    nonzero = [float(mask.sum()) for mask in masks if mask.any()]
    median = float(np.median(nonzero)) if nonzero else 0.0
    values = []
    for mask in masks:
        area = float(mask.sum())
        if area == 0 or median == 0:
            values.append(0.0)
        else:
            values.append(float(math.exp(-abs(math.log(area / median)))))
    return values


def transition_score(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    previous_source: str,
    current_source: str,
    config: FusionConfig,
) -> float:
    previous_empty = not previous_mask.any()
    current_empty = not current_mask.any()
    if previous_empty and current_empty:
        score = config.temporal_iou_weight
    elif previous_empty or current_empty:
        score = -config.empty_transition_penalty
    else:
        score = config.temporal_iou_weight * mask_iou(previous_mask, current_mask)
        previous_centroid = mask_centroid(previous_mask)
        current_centroid = mask_centroid(current_mask)
        assert previous_centroid is not None and current_centroid is not None
        diagonal = math.hypot(*previous_mask.shape)
        distance = math.dist(previous_centroid, current_centroid) / max(diagonal, 1.0)
        score -= config.centroid_weight * distance
    if previous_source != current_source:
        score -= config.switch_penalty
    return score


def viterbi_select(
    masks_by_source: Mapping[str, Sequence[np.ndarray]],
    scores_by_source: Mapping[str, Sequence[float]],
    config: FusionConfig,
) -> PathResult:
    frame_count = len(next(iter(masks_by_source.values())))
    states = tuple(masks_by_source)
    if frame_count == 0:
        return PathResult((), 0.0, ())
    shape_by_source = {
        source: shape_scores(masks) if source != "O" else [1.0] * frame_count
        for source, masks in masks_by_source.items()
    }
    dp = np.full((frame_count, len(states)), -np.inf, dtype=np.float64)
    parents = np.full((frame_count, len(states)), -1, dtype=np.int32)
    frame_scores: List[Dict[str, float]] = []

    for frame_index in range(frame_count):
        scores_for_frame = {}
        nonempty_scores = [
            scores_by_source[source][frame_index]
            for source in states
            if source != "O" and masks_by_source[source][frame_index].any()
        ]
        both_empty = all(
            not masks_by_source[source][frame_index].any()
            for source in states
            if source != "O"
        )
        allow_empty = both_empty or (
            nonempty_scores and max(nonempty_scores) <= config.empty_score_threshold
        )
        for state_index, source in enumerate(states):
            if source == "O":
                emission = 0.5 if allow_empty else -np.inf
            else:
                mask = masks_by_source[source][frame_index]
                if not mask.any():
                    emission = -np.inf
                else:
                    emission = (
                        config.model_weight * scores_by_source[source][frame_index]
                        + config.shape_weight * shape_by_source[source][frame_index]
                    )
            scores_for_frame[source] = float(emission)
            if frame_index == 0:
                dp[frame_index, state_index] = emission
                continue
            best_score = -np.inf
            best_parent = -1
            for previous_index, previous_source in enumerate(states):
                candidate_score = dp[frame_index - 1, previous_index]
                if not np.isfinite(candidate_score) or not np.isfinite(emission):
                    continue
                candidate_score += emission + transition_score(
                    masks_by_source[previous_source][frame_index - 1],
                    masks_by_source[source][frame_index],
                    previous_source,
                    source,
                    config,
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_parent = previous_index
            dp[frame_index, state_index] = best_score
            parents[frame_index, state_index] = best_parent
        frame_scores.append(scores_for_frame)

    state_index = int(np.argmax(dp[-1]))
    if not np.isfinite(dp[-1, state_index]):
        raise RuntimeError("no valid Viterbi path")
    path = [states[state_index]]
    for frame_index in range(frame_count - 1, 0, -1):
        state_index = int(parents[frame_index, state_index])
        if state_index < 0:
            raise RuntimeError("broken Viterbi backpointer")
        path.append(states[state_index])
    path.reverse()
    return PathResult(tuple(path), float(np.max(dp[-1])), tuple(frame_scores))


def uncertainty_values(
    forward_masks: Sequence[np.ndarray],
    backward_masks: Sequence[np.ndarray],
    forward_scores: Sequence[float],
    backward_scores: Sequence[float],
) -> List[float]:
    values = []
    for forward, backward, forward_score, backward_score in zip(
        forward_masks, backward_masks, forward_scores, backward_scores
    ):
        disagreement = 1.0 - mask_iou(forward, backward)
        low_quality = 1.0 - max(forward_score, backward_score)
        if not forward.any() and not backward.any():
            disagreement = 0.0
        values.append(float(np.clip(max(disagreement, low_quality), 0, 1)))
    return values


def uncertain_intervals(
    forward_masks: Sequence[np.ndarray],
    backward_masks: Sequence[np.ndarray],
    forward_scores: Sequence[float],
    backward_scores: Sequence[float],
    config: FusionConfig,
) -> List[Tuple[int, int]]:
    flagged = []
    for forward, backward, forward_score, backward_score in zip(
        forward_masks, backward_masks, forward_scores, backward_scores
    ):
        disagreement = mask_iou(forward, backward) < config.disagreement_iou_threshold
        independent_failure = (
            min(forward_score, backward_score) < config.empty_score_threshold
            or (forward.any() != backward.any())
            or abs(stability_from_mask(forward) - stability_from_mask(backward)) > 0.35
        )
        flagged.append(disagreement and independent_failure)
    intervals = []
    index = 0
    while index < len(flagged):
        if not flagged[index]:
            index += 1
            continue
        start = index
        while index < len(flagged) and flagged[index]:
            index += 1
        if index - start < config.disagreement_min_frames:
            continue
        recovery_start = None
        recovery_run = 0
        while index < len(flagged):
            recovered = (
                mask_iou(forward_masks[index], backward_masks[index])
                >= config.recovery_iou_threshold
            )
            recovery_run = recovery_run + 1 if recovered else 0
            if recovery_run >= config.recovery_min_frames:
                recovery_start = index - recovery_run + 1
                break
            index += 1
        end = len(flagged) - 1 if recovery_start is None else recovery_start - 1
        intervals.append((start, end))
        if recovery_start is not None:
            index += 1
    return intervals


def trusted_anchor(
    start: int,
    step: int,
    limit: int,
    forward_masks: Sequence[np.ndarray],
    backward_masks: Sequence[np.ndarray],
    forward_scores: Sequence[float],
    backward_scores: Sequence[float],
    config: FusionConfig,
) -> Optional[int]:
    for offset in range(config.anchor_search_frames + 1):
        index = start + step * offset
        if not 0 <= index < limit:
            break
        if (
            mask_iou(forward_masks[index], backward_masks[index])
            >= config.anchor_iou_threshold
            and min(forward_scores[index], backward_scores[index])
            >= config.anchor_score_threshold
            and forward_masks[index].any()
            and backward_masks[index].any()
        ):
            return index
    return None


def selected_masks(
    path: Sequence[str], masks_by_source: Mapping[str, Sequence[np.ndarray]]
) -> List[np.ndarray]:
    return [masks_by_source[source][index].copy() for index, source in enumerate(path)]


def greedy_sources(
    masks_by_source: Mapping[str, Sequence[np.ndarray]],
    scores_by_source: Mapping[str, Sequence[float]],
) -> List[str]:
    frame_count = len(next(iter(masks_by_source.values())))
    result = []
    for frame_index in range(frame_count):
        candidates = [
            source
            for source in masks_by_source
            if source != "O" and masks_by_source[source][frame_index].any()
        ]
        result.append(
            max(candidates, key=lambda source: scores_by_source[source][frame_index])
            if candidates
            else "O"
        )
    return result


def switch_count(sources: Sequence[str]) -> int:
    return sum(first != second for first, second in zip(sources, sources[1:]))


def interior_point(mask: np.ndarray) -> Tuple[float, float]:
    """Return a normalized point far from the mask boundary."""
    if not mask.any():
        raise ValueError("cannot sample a point from an empty mask")
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    height, width = mask.shape
    return (x + 0.5) / width, (y + 0.5) / height


def run_anchor_propagation(
    predictor: Any,
    frame_dir: Path,
    frame_count: int,
    prompt: str,
    anchor_index: int,
    anchor_mask: np.ndarray,
    interval: Tuple[int, int],
    direction: str,
    name: str,
) -> Optional[Tuple[List[np.ndarray], List[float]]]:
    """Start a clean session at one trusted anchor and propagate into an interval."""
    session_id: Optional[str] = None
    shape = anchor_mask.shape
    masks = [np.zeros(shape, dtype=bool) for _ in range(frame_count)]
    scores = [0.0] * frame_count
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
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor_index,
                "text": prompt,
            }
        )
        anchor_candidates, detector, tracker = output_frame(response.get("outputs"))
        if not anchor_candidates:
            return None
        obj_id = max(
            anchor_candidates,
            key=lambda candidate_id: (
                mask_iou(anchor_candidates[candidate_id], anchor_mask),
                tracker.get(candidate_id, detector.get(candidate_id, 0.0)),
            ),
        )
        if mask_iou(anchor_candidates[obj_id], anchor_mask) < 0.5:
            return None
        x, y = interior_point(anchor_mask)
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor_index,
                "points": [[x, y]],
                "point_labels": [1],
                "clear_old_points": True,
                "obj_id": obj_id,
                "rel_coordinates": True,
            }
        )
        start, end = interval
        max_frames = (
            end - anchor_index if direction == "forward" else anchor_index - start
        )
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": anchor_index,
            "max_frame_num_to_track": max_frames,
        }
        for propagated in predictor.handle_stream_request(request):
            frame_index = int(propagated["frame_index"])
            if not start <= frame_index <= end:
                continue
            frame_masks, frame_detector, frame_tracker = output_frame(
                propagated.get("outputs")
            )
            if obj_id in frame_masks:
                masks[frame_index] = frame_masks[obj_id]
                scores[frame_index] = frame_tracker.get(
                    obj_id, frame_detector.get(obj_id, 0.0)
                )
        return masks, scores
    finally:
        if session_id is not None:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )


def try_repair_interval(
    predictor: Any,
    frame_dir: Path,
    frame_count: int,
    prompt: str,
    interval: Tuple[int, int],
    left_anchor: Optional[int],
    right_anchor: Optional[int],
    base_masks: List[np.ndarray],
    base_sources: List[str],
    masks_by_source: Dict[str, Sequence[np.ndarray]],
    scores_by_source: Dict[str, Sequence[float]],
    config: FusionConfig,
) -> Dict[str, Any]:
    start, end = interval
    attempt: Dict[str, Any] = {
        "start": start,
        "end": end,
        "left_anchor": left_anchor,
        "right_anchor": right_anchor,
        "status": "rejected",
    }
    if left_anchor is None or right_anchor is None:
        attempt["reason"] = "missing_trusted_anchor"
        return attempt
    left_seed = np.logical_and(
        masks_by_source["F"][left_anchor], masks_by_source["B"][left_anchor]
    )
    right_seed = np.logical_and(
        masks_by_source["F"][right_anchor], masks_by_source["B"][right_anchor]
    )
    left = run_anchor_propagation(
        predictor,
        frame_dir,
        frame_count,
        prompt,
        left_anchor,
        left_seed,
        interval,
        "forward",
        "L",
    )
    right = run_anchor_propagation(
        predictor,
        frame_dir,
        frame_count,
        prompt,
        right_anchor,
        right_seed,
        interval,
        "backward",
        "R",
    )
    if left is None or right is None:
        attempt["reason"] = "anchor_prompt_did_not_match"
        return attempt
    local_slice = slice(start, end + 1)
    local_masks = {
        source: list(masks[local_slice]) for source, masks in masks_by_source.items()
    }
    local_scores = {
        source: list(scores[local_slice]) for source, scores in scores_by_source.items()
    }
    local_masks["L"] = list(left[0][local_slice])
    local_masks["R"] = list(right[0][local_slice])
    local_scores["L"] = list(left[1][local_slice])
    local_scores["R"] = list(right[1][local_slice])
    base = viterbi_select(
        {source: masks for source, masks in local_masks.items() if source in SOURCES},
        {
            source: scores
            for source, scores in local_scores.items()
            if source in SOURCES
        },
        config,
    )
    repaired = viterbi_select(local_masks, local_scores, config)
    gain = (repaired.score - base.score) / max(end - start + 1, 1)
    repaired_masks = selected_masks(repaired.sources, local_masks)
    endpoint_iou = min(
        mask_iou(repaired_masks[0], base_masks[start]),
        mask_iou(repaired_masks[-1], base_masks[end]),
    )
    attempt.update(
        {
            "score_gain_per_frame": gain,
            "endpoint_iou": endpoint_iou,
            "candidate_source_counts": {
                source: repaired.sources.count(source) for source in local_masks
            },
        }
    )
    if gain < config.repair_min_gain:
        attempt["reason"] = "insufficient_score_gain"
        return attempt
    if endpoint_iou < config.recovery_iou_threshold:
        attempt["reason"] = "interval_endpoints_worsened"
        return attempt
    for local_index, frame_index in enumerate(range(start, end + 1)):
        base_masks[frame_index] = repaired_masks[local_index]
        base_sources[frame_index] = repaired.sources[local_index]
    masks_by_source["L"] = left[0]
    masks_by_source["R"] = right[0]
    scores_by_source["L"] = left[1]
    scores_by_source["R"] = right[1]
    attempt["status"] = "accepted"
    return attempt


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


def write_all_instance_video(
    path: Path,
    frames: Sequence[Dict[int, np.ndarray]],
    fps: float,
    shape: Optional[Tuple[int, int]] = None,
) -> Dict[int, int]:
    first_mask = next((mask for frame in frames for mask in frame.values()), None)
    if first_mask is None and shape is None:
        raise RuntimeError("mask shape is required when a direction produced no masks")
    height, width = first_mask.shape if first_mask is not None else shape
    mapping = {
        obj_id: index + 1
        for index, obj_id in enumerate(sorted({i for f in frames for i in f}))
    }
    if len(mapping) > 255:
        raise RuntimeError("more than 255 instances cannot be stored in gray8")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height), isColor=False
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create lossless mask video: {path}")
    try:
        for frame in frames:
            labels = np.zeros((height, width), dtype=np.uint8)
            for obj_id, mask in sorted(frame.items()):
                labels[mask] = mapping[obj_id]
            writer.write(labels)
    finally:
        writer.release()
    return mapping


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
    path_sources: Sequence[str],
    uncertainty: Sequence[float],
    repaired_frames: Sequence[bool],
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
                f"Fused  source={path_sources[index]}  uncertainty={uncertainty[index]:.2f}",
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
            repair = " repaired" if repaired_frames[index] else ""
            header(
                diagnostic,
                f"Difference  IoU={mask_iou(forward, backward):.2f}{repair}",
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


def persist_direction(
    output_dir: Path,
    result: DirectionResult,
    fps: float,
    shape: Tuple[int, int],
) -> Dict[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = write_all_instance_video(
        output_dir / "masks.mkv", result.frames, fps, shape
    )
    write_jsonl(
        output_dir / "frames.jsonl",
        (
            {
                "frame_index": index,
                "objects": [
                    {
                        "object_id": obj_id,
                        "label": mapping[obj_id],
                        "detector_score": detector.get(obj_id),
                        "tracker_score": tracker.get(obj_id),
                        "area": int(mask.sum()),
                    }
                    for obj_id, mask in sorted(frame.items())
                ],
            }
            for index, (frame, detector, tracker) in enumerate(
                zip(result.frames, result.detector_scores, result.tracker_scores)
            )
        ),
    )
    return mapping


def process_video(
    predictor: Any,
    video_path: Path,
    input_root: Path,
    output_dir: Path,
    prompt: str,
    model_version: str,
    max_frames: Optional[int],
    overwrite: bool,
    fusion_config: FusionConfig,
    repair_uncertain: bool,
    ground_truth_path: Optional[Path],
    ground_truth_label: int,
    backward_mode: str,
    backward_equivalence_iou: float,
) -> str:
    video_info = probe_video(video_path, fallback_fps=30.0)
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "input_video": str(video_path),
        "input_size": video_path.stat().st_size,
        "input_mtime_ns": video_path.stat().st_mtime_ns,
        "prompt": prompt,
        "model_version": model_version,
        "max_frames": max_frames,
        "fusion": asdict(fusion_config),
        "repair_uncertain": repair_uncertain,
        "ground_truth": str(ground_truth_path) if ground_truth_path else None,
        "ground_truth_label": ground_truth_label,
        "backward_mode": backward_mode,
        "backward_equivalence_iou": backward_equivalence_iou,
    }
    fingerprint = config_hash(fingerprint_payload)
    metadata_path = output_dir / "metadata.json"
    if not overwrite and metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "success"
                and existing.get("config_hash") == fingerprint
            ):
                LOGGER.info("Skipping completed sequence: %s", video_path)
                return "skipped"
        except (OSError, json.JSONDecodeError):
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("forward", "backward", "backward_verification"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for name in (
        "masks.mkv",
        "result.mp4",
        "frames.jsonl",
        "evaluation.json",
        "audit.json",
        "backward_equivalence.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()
    metadata = {
        **fingerprint_payload,
        "config_hash": fingerprint,
        "status": "processing",
        "started_at": utc_now(),
        "source": video_info,
    }
    write_json(metadata_path, metadata)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="sam3_bidirectional_") as temporary:
            temporary_dir = Path(temporary)
            frame_dir = temporary_dir / "frames"
            reverse_dir = temporary_dir / "frames_reversed"
            frame_dir.mkdir()
            frame_count = extract_png_frames(video_path, frame_dir, max_frames)
            reverse_frame_directory(frame_dir, reverse_dir, frame_count)
            LOGGER.info("Running independent forward pass: %s", video_path)
            forward = run_direction(predictor, frame_dir, frame_count, prompt, "F")
            physical_backward = None
            api_backward = None
            if backward_mode in {"physical", "verify"}:
                LOGGER.info(
                    "Running independent physical backward pass: %s", video_path
                )
                physical_backward = run_direction(
                    predictor,
                    reverse_dir,
                    frame_count,
                    prompt,
                    "B-physical",
                    reverse_index=True,
                )
            if backward_mode in {"api", "verify"}:
                LOGGER.info("Running independent API backward pass: %s", video_path)
                api_backward = run_direction(
                    predictor,
                    frame_dir,
                    frame_count,
                    prompt,
                    "B-api",
                    propagation_direction="backward",
                )
            shape = (int(video_info["height"]), int(video_info["width"]))
            equivalence = None
            if backward_mode == "verify":
                assert physical_backward is not None and api_backward is not None
                equivalence = compare_direction_results(
                    physical_backward,
                    api_backward,
                    shape,
                    backward_equivalence_iou,
                )
                write_json(output_dir / "backward_equivalence.json", equivalence)
                if not equivalence["equivalent"]:
                    persist_direction(
                        output_dir / "backward_verification" / "physical",
                        physical_backward,
                        float(video_info["fps"]),
                        shape,
                    )
                    persist_direction(
                        output_dir / "backward_verification" / "api",
                        api_backward,
                        float(video_info["fps"]),
                        shape,
                    )
                    raise RuntimeError(
                        "physical reversal and API backward are not equivalent; "
                        "inspect backward_equivalence.json and "
                        "backward_verification/, then choose "
                        "--backward-mode physical or api"
                    )
                backward = api_backward
            else:
                backward = (
                    physical_backward if physical_backward is not None else api_backward
                )
                assert backward is not None
            forward_id, backward_id, pair_report = match_primary_tracks(
                forward, backward
            )
            forward.primary_obj_id = forward_id
            backward.primary_obj_id = backward_id
            forward_masks, forward_scores = primary_track(forward, shape)
            backward_masks, backward_scores = primary_track(backward, shape)
            empty_masks = [np.zeros(shape, dtype=bool) for _ in range(frame_count)]
            path_result = viterbi_select(
                {"F": forward_masks, "B": backward_masks, "O": empty_masks},
                {"F": forward_scores, "B": backward_scores, "O": [0.0] * frame_count},
                fusion_config,
            )
            masks_by_source: Dict[str, Sequence[np.ndarray]] = {
                "F": forward_masks,
                "B": backward_masks,
                "O": empty_masks,
            }
            scores_by_source: Dict[str, Sequence[float]] = {
                "F": forward_scores,
                "B": backward_scores,
                "O": [0.0] * frame_count,
            }
            greedy_path = greedy_sources(masks_by_source, scores_by_source)
            uncertainty = uncertainty_values(
                forward_masks, backward_masks, forward_scores, backward_scores
            )
            audit = bidirectional_audit(forward_masks, backward_masks)
            write_json(output_dir / "audit.json", audit)
            intervals = uncertain_intervals(
                forward_masks,
                backward_masks,
                forward_scores,
                backward_scores,
                fusion_config,
            )
            repair_attempts = []
            repaired_frames = [False] * frame_count
            fused_masks = selected_masks(path_result.sources, masks_by_source)
            fused_sources = list(path_result.sources)
            for start, end in intervals:
                left_anchor = trusted_anchor(
                    start - 1,
                    -1,
                    frame_count,
                    forward_masks,
                    backward_masks,
                    forward_scores,
                    backward_scores,
                    fusion_config,
                )
                right_anchor = trusted_anchor(
                    end + 1,
                    1,
                    frame_count,
                    forward_masks,
                    backward_masks,
                    forward_scores,
                    backward_scores,
                    fusion_config,
                )
                if repair_uncertain:
                    attempt = try_repair_interval(
                        predictor,
                        frame_dir,
                        frame_count,
                        prompt,
                        (start, end),
                        left_anchor,
                        right_anchor,
                        fused_masks,
                        fused_sources,
                        masks_by_source,
                        scores_by_source,
                        fusion_config,
                    )
                    if attempt["status"] == "accepted":
                        repaired_frames[start : end + 1] = [True] * (end - start + 1)
                else:
                    attempt = {
                        "start": start,
                        "end": end,
                        "left_anchor": left_anchor,
                        "right_anchor": right_anchor,
                        "status": "disabled",
                    }
                repair_attempts.append(attempt)

            forward_mapping = persist_direction(
                output_dir / "forward", forward, float(video_info["fps"]), shape
            )
            backward_mapping = persist_direction(
                output_dir / "backward", backward, float(video_info["fps"]), shape
            )
            write_label_video(
                output_dir / "masks.mkv", fused_masks, float(video_info["fps"])
            )
            write_result_video(
                output_dir / "result.mp4",
                frame_dir,
                forward_masks,
                backward_masks,
                fused_masks,
                fused_sources,
                uncertainty,
                repaired_frames,
                float(video_info["fps"]),
            )
            write_jsonl(
                output_dir / "frames.jsonl",
                (
                    {
                        "frame_index": index,
                        "selected_source": fused_sources[index],
                        "forward_score": forward_scores[index],
                        "backward_score": backward_scores[index],
                        "forward_backward_iou": mask_iou(
                            forward_masks[index], backward_masks[index]
                        ),
                        "uncertainty": uncertainty[index],
                        "emission_scores": path_result.frame_scores[index],
                        "switch": index > 0
                        and fused_sources[index] != fused_sources[index - 1],
                        "switch_reason": (
                            f"{fused_sources[index - 1]}_to_{fused_sources[index]}"
                            if index > 0
                            and fused_sources[index] != fused_sources[index - 1]
                            else None
                        ),
                        "repair_accepted": repaired_frames[index],
                    }
                    for index in range(frame_count)
                ),
            )

            evaluation = None
            if ground_truth_path is not None:
                ground_truth = read_label_video(
                    ground_truth_path, frame_count, shape, ground_truth_label
                )
                evaluation = evaluate_masks(
                    ground_truth,
                    {
                        "forward": forward_masks,
                        "backward": backward_masks,
                        "union": [
                            np.logical_or(first, second)
                            for first, second in zip(forward_masks, backward_masks)
                        ],
                        "intersection": [
                            np.logical_and(first, second)
                            for first, second in zip(forward_masks, backward_masks)
                        ],
                        "fused": fused_masks,
                    },
                )
                write_json(output_dir / "evaluation.json", evaluation)

        metadata.update(
            {
                "status": "success",
                "completed_at": utc_now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "frames_processed": frame_count,
                "primary_instances": {"forward": forward_id, "backward": backward_id},
                "instance_matches": pair_report,
                "object_id_to_label": {
                    "forward": {str(k): v for k, v in forward_mapping.items()},
                    "backward": {str(k): v for k, v in backward_mapping.items()},
                },
                "viterbi_score": path_result.score,
                "switch_counts": {
                    "greedy_framewise": switch_count(greedy_path),
                    "viterbi": switch_count(fused_sources),
                },
                "source_counts": {
                    source: fused_sources.count(source) for source in masks_by_source
                },
                "uncertain_intervals": [list(interval) for interval in intervals],
                "repair_attempts": repair_attempts,
                "backward_equivalence": equivalence,
                "audit": audit,
                "diagnostics": {
                    "tracker_score": True,
                    "predicted_iou": False,
                    "presence_logit": False,
                    "mask_stability": "geometry_proxy",
                    "low_resolution_logits": False,
                },
                "evaluation": evaluation,
                "outputs": {
                    "forward_masks": "forward/masks.mkv",
                    "backward_masks": "backward/masks.mkv",
                    "fused_masks": "masks.mkv",
                    "visualization": "result.mp4",
                    "frame_diagnostics": "frames.jsonl",
                    "unlabeled_audit": "audit.json",
                },
            }
        )
        write_json(metadata_path, metadata)
        return "success"
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(metadata_path, metadata)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立运行 SAM 正序/倒序分割，并用 Viterbi 汇总目标实例。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="输入目录；直接包含 color.mp4 时处理单段视频，否则递归查找",
    )
    parser.add_argument(
        "--output-root", required=True, help="输出根目录，并保留输入相对目录结构"
    )
    parser.add_argument(
        "--prompt", required=True, help='两个方向共用的文本提示，如 "left hand"'
    )
    parser.add_argument(
        "--version",
        default="sam3",
        choices=["sam3", "sam3.1"],
        help="使用的 SAM 模型版本",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=("cuda:0", 0),
        metavar="cuda:N",
        help="执行推理的 CUDA 设备",
    )
    parser.add_argument("--max-sequences", type=positive_int, help="最多处理多少段视频")
    parser.add_argument(
        "--max-frames", type=positive_int, help="每段视频最多处理多少帧"
    )
    parser.add_argument(
        "--ground-truth-root",
        help=(
            "可选 GT masks.mkv 文件或根目录；目录模式按输入相对路径查找 " "masks.mkv"
        ),
    )
    parser.add_argument(
        "--ground-truth-label",
        type=positive_int,
        default=1,
        help="GT masks.mkv 中作为目标前景的标签值",
    )
    parser.add_argument(
        "--repair-uncertain",
        action="store_true",
        help="开启默认关闭的 P0-C 高不确定区间双锚点重传播",
    )
    parser.add_argument(
        "--backward-mode",
        choices=["verify", "physical", "api"],
        default="verify",
        help=(
            "倒序实现：verify 同时运行 physical/api 并检查等价性；physical "
            "强制使用反编号帧；api 强制使用 SAM backward API"
        ),
    )
    parser.add_argument(
        "--backward-equivalence-iou",
        type=unit_interval,
        default=0.999,
        help="verify 模式判定两种倒序 mask 等价所需的最低逐帧 IoU",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="忽略配置 hash，覆盖已有完整结果"
    )
    parser.add_argument(
        "--list-only", action="store_true", help="只列出待处理视频，不加载模型"
    )
    parser.add_argument(
        "--disagreement-iou-threshold",
        type=unit_interval,
        default=0.35,
        help="F/B IoU 低于该值时视为严重分歧证据",
    )
    parser.add_argument(
        "--disagreement-min-frames",
        type=positive_int,
        default=5,
        help="严重分歧至少持续多少帧才形成不确定区间",
    )
    parser.add_argument(
        "--empty-score-threshold",
        type=unit_interval,
        default=0.20,
        help="允许选择目标不可见状态的最高候选质量分数",
    )
    parser.add_argument(
        "--recovery-iou-threshold",
        type=unit_interval,
        default=0.60,
        help="F/B 一致性恢复所需的最低 IoU",
    )
    parser.add_argument(
        "--recovery-min-frames",
        type=positive_int,
        default=3,
        help="一致性连续恢复多少帧后结束不确定区间",
    )
    parser.add_argument(
        "--anchor-iou-threshold",
        type=unit_interval,
        default=0.80,
        help="可信锚点要求的最低 F/B IoU",
    )
    parser.add_argument(
        "--anchor-score-threshold",
        type=unit_interval,
        default=0.60,
        help="可信锚点要求两个方向均达到的最低质量分数",
    )
    parser.add_argument(
        "--anchor-search-frames",
        type=positive_int,
        default=90,
        help="从不确定区间边缘向外搜索可信锚点的最大帧数",
    )
    parser.add_argument(
        "--repair-min-gain",
        type=nonnegative_float,
        default=0.10,
        help="接受双锚点修复所需的每帧最小 Viterbi 分数提升",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    input_root = expand_path(args.input_root)
    output_root = expand_path(args.output_root)
    ground_truth_root = (
        expand_path(args.ground_truth_root) if args.ground_truth_root else None
    )
    if not input_root.is_dir():
        LOGGER.error("Input root is not a directory: %s", input_root)
        return 2
    if ground_truth_root is not None and not ground_truth_root.exists():
        LOGGER.error("Ground-truth path does not exist: %s", ground_truth_root)
        return 2
    videos = discover_color_videos(input_root)
    if args.max_sequences is not None:
        videos = videos[: args.max_sequences]
    if not videos:
        LOGGER.error("No files named color.mp4 found below %s", input_root)
        return 2
    if args.list_only:
        for video in videos:
            print(video.relative_to(input_root))
        return 0

    checkpoint = expand_path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        LOGGER.error("Checkpoint does not exist: %s", checkpoint)
        return 2
    import torch

    device_name, device_index = args.device
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        LOGGER.error("Requested CUDA device is unavailable: %s", device_name)
        return 2
    torch.cuda.set_device(device_index)
    from sam3 import build_sam3_predictor

    kwargs = {"version": args.version, "compile": False, "async_loading_frames": True}
    if checkpoint is not None:
        kwargs["checkpoint_path"] = str(checkpoint)
    predictor = build_sam3_predictor(**kwargs)
    config = FusionConfig(
        empty_score_threshold=args.empty_score_threshold,
        disagreement_iou_threshold=args.disagreement_iou_threshold,
        disagreement_min_frames=args.disagreement_min_frames,
        recovery_iou_threshold=args.recovery_iou_threshold,
        recovery_min_frames=args.recovery_min_frames,
        anchor_iou_threshold=args.anchor_iou_threshold,
        anchor_score_threshold=args.anchor_score_threshold,
        anchor_search_frames=args.anchor_search_frames,
        repair_min_gain=args.repair_min_gain,
    )
    counts = {"success": 0, "skipped": 0, "failed": 0}
    try:
        for video in videos:
            relative_parent = video.parent.relative_to(input_root)
            output_dir = output_dir_for(video, input_root, output_root)
            try:
                ground_truth = resolve_ground_truth(
                    ground_truth_root, relative_parent, len(videos) == 1
                )
                if ground_truth_root is not None and ground_truth is None:
                    LOGGER.warning("No matching GT masks.mkv for %s", video)
                status = process_video(
                    predictor,
                    video,
                    input_root,
                    output_dir,
                    args.prompt,
                    args.version,
                    args.max_frames,
                    args.overwrite,
                    config,
                    args.repair_uncertain,
                    ground_truth,
                    args.ground_truth_label,
                    args.backward_mode,
                    args.backward_equivalence_iou,
                )
                counts[status] += 1
            except Exception:
                counts["failed"] += 1
                LOGGER.exception("Sequence failed: %s", video)
    finally:
        predictor.shutdown()
    LOGGER.info("Finished: %s", counts)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

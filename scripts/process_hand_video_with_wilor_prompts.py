#!/usr/bin/env python3
"""使用 WiLoR 手关节提示修正单个视频中的 SAM 3 手部分割结果。

脚本从一个输入目录读取 ``color.mp4`` 和
``MANO_wilor_occlusion/hand_joints_occlusion.jsonl``。第一遍先选择可靠的
初始化帧，使用 ``left hand`` 或 ``right hand`` 文本提示锁定一个 SAM 实例，
再向前、向后传播得到基线掩码。随后根据每帧的基线掩码生成点提示：

* 目标手未被遮挡且位于基线掩码内的关节中，选择最多 3 个
  可靠且空间分散的正点；
* 目标手被遮挡且位于基线掩码内的关节中，可靠距离最大者作为负点；
* 反侧手未被遮挡且位于基线掩码内的关节中，可靠距离最大者作为负点；
* 位于腕关节前臂方向、远离所有目标手关节的掩码像素可以作为负点。

正点必须位于基线掩码内，不允许用掩码外关节恢复漏分。只有同一帧存在
正点时才会提交负点；一帧的所有正负点通过一次请求提交。第 1 次分割只用
文本提示；从第 2 次开始，每次都根据上一次分割掩码重新计算点提示，再从
无点提示的基线状态执行分割。``result.mp4`` 在原视频上并排显示候选关节、
最后一次分割所依据的掩码及提示点，以及最终掩码。

TODO：
* 关联视频中同侧的多只手。
* 如果简单腕部几何规则产生误判，为前臂检测增加扩张检测框、最小连通区域
  和跨帧持续性检查。
* 比较基线与修正掩码的质量，并增加带时序平滑的回退机制。
* 评估位于基线掩码外的反侧手负点，以及逐帧传播、逐帧修正的在线模式。

运行示例：
    python scripts/process_hand_video_with_wilor_prompts.py \
        --input-dir /path/to/milk \
        --output-dir /path/to/output \
        --hand-side left \
        --version sam3 \
        --segmentation-passes 2 \
        --device cuda:0
"""

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__:
    from scripts.process_dataset_videos import (
        color_for_label,
        expand_path,
        extract_png_frames,
        lighter_color,
        MASK_ALPHA,
        normalize_outputs,
        parse_device,
        probe_video,
        utc_now,
        write_json,
    )
else:  # Direct execution adds scripts/, not the repo root.
    from process_dataset_videos import (  # type: ignore[no-redef]
        color_for_label,
        expand_path,
        extract_png_frames,
        lighter_color,
        MASK_ALPHA,
        normalize_outputs,
        parse_device,
        probe_video,
        utc_now,
        write_json,
    )


LOGGER = logging.getLogger("sam3_wilor_hand_processor")
JOINTS_PATH = Path("MANO_wilor_occlusion/hand_joints_occlusion.jsonl")
MCP_INDICES = (5, 9, 13, 17)
MAX_POSITIVE_POINTS = 3


@dataclass(frozen=True)
class JointCandidate:
    side: str
    status: str
    joint_index: int
    joint_name: str
    x: int
    y: int
    pixel: Tuple[float, float]
    detection_confidence: float
    reliability_score_px: float


@dataclass(frozen=True)
class PromptPoint:
    x: int
    y: int
    label: int
    kind: str
    joint: Optional[JointCandidate] = None


def finite_xy(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def in_frame(x: float, y: float, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def load_joint_jsonl(
    path: Path, video_info: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and strictly validate frame alignment and the one-hand-per-side limit."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on {path}:{line_number}: {exc}"
                ) from exc

    if not records or records[0].get("record_type") != "metadata":
        raise ValueError(f"first JSONL record must be metadata: {path}")
    metadata = records[0]
    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"unsupported JSONL schema_version: {metadata.get('schema_version')!r}"
        )
    frames = records[1:]
    if any(record.get("record_type") != "frame" for record in frames):
        raise ValueError("all JSONL records after metadata must be frame records")

    expected_frames = int(video_info["frame_count"])
    if len(frames) != expected_frames:
        raise ValueError(
            f"frame count mismatch: video reports {expected_frames}, JSONL has "
            f"{len(frames)}"
        )
    video_metadata = metadata.get("video", {})
    expected_size = (int(video_info["width"]), int(video_info["height"]))
    jsonl_size = (video_metadata.get("width"), video_metadata.get("height"))
    if jsonl_size != expected_size:
        raise ValueError(
            f"resolution mismatch: video is {expected_size}, JSONL is {jsonl_size}"
        )

    for frame_index, frame in enumerate(frames):
        if frame.get("frame_index") != frame_index:
            raise ValueError(
                f"JSONL frame index mismatch at row {frame_index}: "
                f"got {frame.get('frame_index')!r}"
            )
        counts = {"left": 0, "right": 0}
        for hand in frame.get("hands", []):
            side = hand.get("side")
            if side not in counts:
                raise ValueError(f"invalid hand side {side!r} at frame {frame_index}")
            counts[side] += 1
            if counts[side] > 1:
                raise ValueError(
                    f"multiple {side} hands are not supported; first seen at "
                    f"frame {frame_index}"
                )
    return metadata, frames


def hand_for_side(frame: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
    return next(
        (hand for hand in frame.get("hands", []) if hand.get("side") == side), None
    )


def filtered_candidates(
    frame: Dict[str, Any],
    target_side: str,
    width: int,
    height: int,
    confidence_threshold: float,
    reliability_threshold_px: float,
) -> List[JointCandidate]:
    """Return only joint/status combinations that can become SAM prompts."""
    candidates = []
    for hand in frame.get("hands", []):
        side = hand.get("side")
        hand_confidence = float(hand.get("detection_confidence", 0.0))
        if hand_confidence < confidence_threshold:
            continue
        allowed_statuses = (
            {"visible", "occluded"} if side == target_side else {"visible"}
        )
        for joint in hand.get("joints", []):
            joint_confidence = float(joint.get("detection_confidence", hand_confidence))
            if joint_confidence < confidence_threshold:
                continue
            status = joint.get("occlusion_status")
            if status not in allowed_statuses:
                continue
            reliability = joint.get("reliability_score_px")
            if reliability is None or float(reliability) < reliability_threshold_px:
                continue
            sample = finite_xy(joint.get("sample_pixel"))
            pixel = finite_xy(joint.get("pixel"))
            if sample is None or pixel is None:
                continue
            x, y = int(round(sample[0])), int(round(sample[1]))
            if not in_frame(x, y, width, height):
                continue
            candidates.append(
                JointCandidate(
                    side=str(side),
                    status=str(status),
                    joint_index=int(joint["joint_index"]),
                    joint_name=str(joint["joint_name"]),
                    x=x,
                    y=y,
                    pixel=pixel,
                    detection_confidence=joint_confidence,
                    reliability_score_px=float(reliability),
                )
            )
    return candidates


def choose_best(candidates: Sequence[JointCandidate]) -> Optional[JointCandidate]:
    return max(
        candidates,
        key=lambda joint: (joint.reliability_score_px, -joint.joint_index),
        default=None,
    )


def choose_spread_positives(
    candidates: Sequence[JointCandidate], max_points: int = MAX_POSITIVE_POINTS
) -> List[JointCandidate]:
    """Keep the most reliable joint, then add joints farthest from those chosen."""
    remaining = list(candidates)
    first = choose_best(remaining)
    if first is None:
        return []
    selected = [first]
    remaining.remove(first)
    while remaining and len(selected) < max_points:
        best = max(
            remaining,
            key=lambda joint: (
                min(
                    (joint.x - chosen.x) ** 2 + (joint.y - chosen.y) ** 2
                    for chosen in selected
                ),
                joint.reliability_score_px,
                -joint.joint_index,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def target_geometry(
    frame: Dict[str, Any],
    target_side: str,
    width: int,
    height: int,
    confidence_threshold: float,
) -> Dict[int, Tuple[int, int]]:
    """Get in-frame target joints for geometry without a reliability cutoff."""
    hand = hand_for_side(frame, target_side)
    if (
        hand is None
        or float(hand.get("detection_confidence", 0.0)) < confidence_threshold
    ):
        return {}
    result = {}
    for joint in hand.get("joints", []):
        sample = finite_xy(joint.get("sample_pixel"))
        if sample is None:
            continue
        x, y = int(round(sample[0])), int(round(sample[1]))
        if in_frame(x, y, width, height):
            result[int(joint["joint_index"])] = (x, y)
    return result


def arm_negative_point(
    mask: np.ndarray,
    joints: Dict[int, Tuple[int, int]],
    distance_ratio: float,
) -> Optional[Tuple[int, int]]:
    """Pick the closest joint-free mask pixel on the forearm side of the wrist."""
    required = (0,) + MCP_INDICES
    if not mask.any() or any(index not in joints for index in required):
        return None
    wrist = np.asarray(joints[0], dtype=np.float32)
    palm_center = np.mean([joints[index] for index in MCP_INDICES], axis=0)
    forearm_direction = wrist - palm_center
    palm_length = float(np.linalg.norm(forearm_direction))
    if palm_length <= 1e-6:
        return None

    seed_image = np.ones(mask.shape, dtype=np.uint8)
    for x, y in joints.values():
        seed_image[y, x] = 0
    distance_to_joints = cv2.distanceTransform(seed_image, cv2.DIST_L2, 5)
    ys, xs = np.indices(mask.shape)
    on_forearm_side = (xs - wrist[0]) * forearm_direction[0] + (
        ys - wrist[1]
    ) * forearm_direction[1] > 0
    eligible = (
        mask.astype(bool)
        & on_forearm_side
        & (distance_to_joints > distance_ratio * palm_length)
    )
    candidate_y, candidate_x = np.nonzero(eligible)
    if len(candidate_x) == 0:
        return None
    wrist_distances = (candidate_x - wrist[0]) ** 2 + (candidate_y - wrist[1]) ** 2
    best = int(np.argmin(wrist_distances))
    return int(candidate_x[best]), int(candidate_y[best])


def select_frame_prompts(
    candidates: Sequence[JointCandidate],
    geometry: Dict[int, Tuple[int, int]],
    mask: np.ndarray,
    target_side: str,
    arm_distance_ratio: float,
    min_negative_distance_px: float,
) -> List[PromptPoint]:
    inside = [joint for joint in candidates if mask[joint.y, joint.x]]
    positive_joints = choose_spread_positives(
        [
            joint
            for joint in inside
            if joint.side == target_side and joint.status == "visible"
        ]
    )
    if not positive_joints:
        return []

    positives = [
        PromptPoint(joint.x, joint.y, 1, "joint_positive", joint)
        for joint in positive_joints
    ]
    negative_joints = (
        (
            "target_occluded_negative",
            choose_best(
                [
                    joint
                    for joint in inside
                    if joint.side == target_side and joint.status == "occluded"
                ]
            ),
        ),
        (
            "opposite_visible_negative",
            choose_best(
                [
                    joint
                    for joint in inside
                    if joint.side != target_side and joint.status == "visible"
                ]
            ),
        ),
    )
    negatives = [
        PromptPoint(joint.x, joint.y, 0, kind, joint)
        for kind, joint in negative_joints
        if joint is not None
    ]
    arm_point = arm_negative_point(mask, geometry, arm_distance_ratio)
    if arm_point is not None:
        negatives.append(PromptPoint(*arm_point, 0, "arm_negative"))

    min_distance_sq = min_negative_distance_px**2
    negatives = [
        point
        for point in negatives
        if all(
            (point.x - positive.x) ** 2 + (point.y - positive.y) ** 2 >= min_distance_sq
            for positive in positives
        )
    ]
    return positives + negatives


def choose_initial_frame(
    frames: Sequence[Dict[str, Any]],
    target_side: str,
    width: int,
    height: int,
    confidence_threshold: float,
    reliability_threshold_px: float,
) -> Tuple[int, str]:
    ranked = []
    fallback = []
    for frame_index, frame in enumerate(frames):
        hand = hand_for_side(frame, target_side)
        if hand is None:
            continue
        confidence = float(hand.get("detection_confidence", 0.0))
        fallback.append((confidence, -frame_index, frame_index))
        candidates = filtered_candidates(
            frame,
            target_side,
            width,
            height,
            confidence_threshold,
            reliability_threshold_px,
        )
        visible_count = sum(
            joint.side == target_side and joint.status == "visible"
            for joint in candidates
        )
        if confidence >= confidence_threshold:
            ranked.append((visible_count, confidence, -frame_index, frame_index))
    if ranked:
        return max(ranked)[-1], "reliable_target_hand"
    if fallback:
        return max(fallback)[-1], "highest_confidence_target_hand_fallback"
    return 0, "no_target_hand_detection_fallback"


def resized_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask.astype(bool)


def choose_target_object(
    outputs: Dict[str, Any],
    candidates: Sequence[JointCandidate],
    target_side: str,
    width: int,
    height: int,
) -> Optional[int]:
    obj_ids, probabilities, masks = normalize_outputs(outputs)
    ranked = []
    for index, raw_obj_id in enumerate(obj_ids):
        mask = resized_mask(masks[index], width, height)
        target_count = sum(
            joint.side == target_side
            and joint.status == "visible"
            and mask[joint.y, joint.x]
            for joint in candidates
        )
        opposite_count = sum(
            joint.side != target_side
            and joint.status == "visible"
            and mask[joint.y, joint.x]
            for joint in candidates
        )
        probability = float(probabilities[index]) if len(probabilities) else 0.0
        obj_id = int(raw_obj_id)
        ranked.append((target_count, -opposite_count, probability, -obj_id, obj_id))
    return max(ranked)[-1] if ranked else None


def mask_for_object(
    outputs: Dict[str, Any], obj_id: Optional[int], width: int, height: int
) -> np.ndarray:
    if obj_id is None:
        return np.zeros((height, width), dtype=bool)
    obj_ids, _, masks = normalize_outputs(outputs)
    matches = np.nonzero(obj_ids.astype(np.int64) == obj_id)[0]
    if not len(matches):
        return np.zeros((height, width), dtype=bool)
    return resized_mask(masks[int(matches[0])], width, height)


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"failed to write temporary mask: {path}")


def read_mask_png(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cannot read temporary mask: {path}")
    return mask > 0


def propagate_to_mask_dir(
    predictor: Any,
    session_id: str,
    start_frame_index: int,
    frame_count: int,
    obj_id: Optional[int],
    width: int,
    height: int,
    mask_dir: Path,
    initial_outputs: Optional[Dict[str, Any]] = None,
) -> None:
    blank = np.zeros((height, width), dtype=bool)
    for frame_index in range(frame_count):
        write_mask_png(mask_dir / f"{frame_index:06d}.png", blank)
    if initial_outputs is not None:
        write_mask_png(
            mask_dir / f"{start_frame_index:06d}.png",
            mask_for_object(initial_outputs, obj_id, width, height),
        )

    legs = (
        ("forward", frame_count - start_frame_index - 1),
        ("backward", start_frame_index),
    )
    for direction, maximum in legs:
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": start_frame_index,
            "max_frame_num_to_track": maximum,
        }
        for response in predictor.handle_stream_request(request):
            frame_index = int(response["frame_index"])
            if not 0 <= frame_index < frame_count:
                continue
            mask = mask_for_object(response.get("outputs", {}), obj_id, width, height)
            write_mask_png(mask_dir / f"{frame_index:06d}.png", mask)


def joint_to_dict(joint: JointCandidate) -> Dict[str, Any]:
    payload = asdict(joint)
    payload["sample_pixel"] = [payload.pop("x"), payload.pop("y")]
    payload["pixel"] = list(payload["pixel"])
    return payload


def prompt_to_dict(point: PromptPoint) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "sample_pixel": [point.x, point.y],
        "label": point.label,
        "kind": point.kind,
    }
    if point.joint is not None:
        payload["joint"] = joint_to_dict(point.joint)
    return payload


def build_prompt_plan(
    frames: Sequence[Dict[str, Any]],
    baseline_mask_dir: Path,
    target_side: str,
    width: int,
    height: int,
    confidence_threshold: float,
    reliability_threshold_px: float,
    arm_distance_ratio: float,
    min_negative_distance_px: float,
) -> Tuple[List[List[JointCandidate]], List[List[PromptPoint]]]:
    all_candidates = []
    all_prompts = []
    for frame_index, frame in enumerate(frames):
        mask = read_mask_png(baseline_mask_dir / f"{frame_index:06d}.png")
        candidates = filtered_candidates(
            frame,
            target_side,
            width,
            height,
            confidence_threshold,
            reliability_threshold_px,
        )
        prompts = select_frame_prompts(
            candidates,
            target_geometry(frame, target_side, width, height, confidence_threshold),
            mask,
            target_side,
            arm_distance_ratio=arm_distance_ratio,
            min_negative_distance_px=min_negative_distance_px,
        )
        all_candidates.append(candidates)
        all_prompts.append(prompts)
    return all_candidates, all_prompts


def apply_prompt_plan(
    predictor: Any,
    session_id: str,
    obj_id: Optional[int],
    prompts_by_frame: Sequence[Sequence[PromptPoint]],
    width: int,
    height: int,
) -> int:
    if obj_id is None:
        return 0
    prompted_frames = 0
    for frame_index, points in enumerate(prompts_by_frame):
        if not points:
            continue
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "points": [[point.x / width, point.y / height] for point in points],
                "point_labels": [point.label for point in points],
                "clear_old_points": True,
                "obj_id": obj_id,
                "rel_coordinates": True,
            }
        )
        prompted_frames += 1
    return prompted_frames


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    if not mask.any():
        return overlay
    color = color_for_label(1)
    float_color = np.asarray(color, dtype=np.float32)
    mask = mask.astype(bool)
    overlay[mask] = (
        overlay[mask].astype(np.float32) * (1.0 - MASK_ALPHA) + float_color * MASK_ALPHA
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, lighter_color(color), 2, cv2.LINE_AA)
    cv2.drawContours(overlay, contours, -1, color, 1, cv2.LINE_AA)
    return overlay


def draw_header(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_candidates(
    frame: np.ndarray,
    candidates: Sequence[JointCandidate],
    target_side: str,
) -> None:
    for joint in candidates:
        if joint.side == target_side and joint.status == "visible":
            color = (40, 220, 40)
        elif joint.side == target_side:
            color = (0, 165, 255)
        else:
            color = (40, 40, 240)
        cv2.circle(frame, (joint.x, joint.y), 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (joint.x, joint.y), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(joint.joint_index),
            (joint.x + 5, joint.y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_cross(
    frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int]
) -> None:
    x, y = point
    for thickness, draw_color in ((4, (255, 255, 255)), (2, color)):
        cv2.line(
            frame, (x - 7, y - 7), (x + 7, y + 7), draw_color, thickness, cv2.LINE_AA
        )
        cv2.line(
            frame, (x - 7, y + 7), (x + 7, y - 7), draw_color, thickness, cv2.LINE_AA
        )


def draw_plus(
    frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int]
) -> None:
    x, y = point
    for thickness, draw_color in ((4, (255, 255, 255)), (2, color)):
        cv2.line(frame, (x - 8, y), (x + 8, y), draw_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y - 8), (x, y + 8), draw_color, thickness, cv2.LINE_AA)


def draw_prompts(frame: np.ndarray, prompts: Sequence[PromptPoint]) -> None:
    for point in prompts:
        if point.kind == "joint_positive":
            color = (40, 220, 40)
        elif point.kind == "arm_negative":
            color = (0, 165, 255)
        else:
            color = (40, 40, 240)
        if point.label:
            draw_plus(frame, (point.x, point.y), color)
        else:
            draw_cross(frame, (point.x, point.y), color)
        if point.joint is None:
            text = "arm negative"
        else:
            text = (
                f"{point.joint.joint_name} "
                f"conf={point.joint.detection_confidence:.2f} "
                f"dist={point.joint.reliability_score_px:.1f}px"
            )
        cv2.putText(
            frame,
            text,
            (min(point.x + 10, max(0, frame.shape[1] - 250)), max(48, point.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def open_video_writer(
    path: Path, fourcc: str, fps: float, size: Tuple[int, int], is_color: bool
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size, isColor=is_color
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"cannot create output video: {path}")
    return writer


def write_outputs(
    frame_dir: Path,
    baseline_mask_dir: Path,
    prompt_mask_dir: Path,
    final_mask_dir: Path,
    candidates_by_frame: Sequence[Sequence[JointCandidate]],
    prompts_by_frame: Sequence[Sequence[PromptPoint]],
    output_dir: Path,
    target_side: str,
    width: int,
    height: int,
    fps: float,
    prompt_source_pass: int,
    final_pass: int,
) -> None:
    baseline_writer = open_video_writer(
        output_dir / "baseline_masks.mkv", "FFV1", fps, (width, height), False
    )
    final_writer = open_video_writer(
        output_dir / "masks.mkv", "FFV1", fps, (width, height), False
    )
    result_writer = open_video_writer(
        output_dir / "result.mp4", "mp4v", fps, (width * 3, height), True
    )
    try:
        for frame_index, candidates in enumerate(candidates_by_frame):
            frame = cv2.imread(
                str(frame_dir / f"{frame_index:06d}.png"), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise RuntimeError(f"cannot read extracted frame {frame_index}")
            baseline = read_mask_png(baseline_mask_dir / f"{frame_index:06d}.png")
            prompt_mask = read_mask_png(prompt_mask_dir / f"{frame_index:06d}.png")
            final = read_mask_png(final_mask_dir / f"{frame_index:06d}.png")
            baseline_writer.write(baseline.astype(np.uint8))
            final_writer.write(final.astype(np.uint8))

            candidate_panel = frame.copy()
            draw_candidates(candidate_panel, candidates, target_side)
            draw_header(candidate_panel, f"frame={frame_index} filtered joints")
            baseline_panel = overlay_mask(frame, prompt_mask)
            draw_prompts(baseline_panel, prompts_by_frame[frame_index])
            if final_pass == 1:
                prompt_header = "pass 1 SAM mask (no point prompts)"
            else:
                prompt_header = (
                    f"pass {prompt_source_pass} + prompts for pass {final_pass}"
                )
            draw_header(baseline_panel, prompt_header)
            final_panel = overlay_mask(frame, final)
            draw_header(final_panel, f"pass {final_pass} SAM mask")
            result_writer.write(
                np.concatenate((candidate_panel, baseline_panel, final_panel), axis=1)
            )
    finally:
        baseline_writer.release()
        final_writer.release()
        result_writer.release()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    generated = (
        output_dir / "result.mp4",
        output_dir / "baseline_masks.mkv",
        output_dir / "masks.mkv",
        output_dir / "metadata.json",
    )
    existing = [path for path in generated if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"output already exists: {existing[0]} (pass --overwrite to replace it)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return output_dir / "metadata.json"


def process_video(predictor: Any, args: argparse.Namespace) -> None:
    input_dir = expand_path(args.input_dir)
    output_dir = expand_path(args.output_dir)
    video_path = input_dir / "color.mp4"
    joints_path = input_dir / JOINTS_PATH
    if not video_path.is_file():
        raise FileNotFoundError(f"missing input video: {video_path}")
    if not joints_path.is_file():
        raise FileNotFoundError(f"missing joint JSONL: {joints_path}")
    metadata_path = prepare_output_dir(output_dir, args.overwrite)
    video_info = probe_video(video_path)
    joint_metadata, frames = load_joint_jsonl(joints_path, video_info)
    width, height = int(video_info["width"]), int(video_info["height"])
    initial_frame, initial_reason = choose_initial_frame(
        frames,
        args.hand_side,
        width,
        height,
        args.detection_confidence_threshold,
        args.reliability_distance_threshold_px,
    )
    prompt_text = f"{args.hand_side} hand"
    metadata: Dict[str, Any] = {
        "status": "processing",
        "started_at": utc_now(),
        "input_video": str(video_path),
        "joint_jsonl": str(joints_path),
        "model_version": args.version,
        "prompt": prompt_text,
        "hand_side": args.hand_side,
        "source": video_info,
        "joint_schema_version": joint_metadata["schema_version"],
        "initial_frame_index": initial_frame,
        "initial_frame_reason": initial_reason,
        "parameters": {
            "segmentation_passes": args.segmentation_passes,
            "detection_confidence_threshold": args.detection_confidence_threshold,
            "reliability_distance_threshold_px": args.reliability_distance_threshold_px,
            "arm_distance_ratio": args.arm_distance_ratio,
            "min_opposite_point_distance_px": args.min_opposite_point_distance_px,
        },
    }
    write_json(metadata_path, metadata)

    started = time.monotonic()
    session_id = None
    try:
        with tempfile.TemporaryDirectory(prefix="sam3_wilor_") as temporary:
            temporary_dir = Path(temporary)
            frame_dir = temporary_dir / "frames"
            frame_dir.mkdir()
            pass_dirs = [
                temporary_dir / f"pass_{pass_index:02d}"
                for pass_index in range(1, args.segmentation_passes + 1)
            ]
            for pass_dir in pass_dirs:
                pass_dir.mkdir()
            decoded_frames = extract_png_frames(video_path, frame_dir, None)
            if decoded_frames != len(frames):
                raise ValueError(
                    f"decoded frame count mismatch: decoded {decoded_frames}, "
                    f"JSONL has {len(frames)}"
                )
            session = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_dir),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = session["session_id"]
            initial_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": initial_frame,
                    "text": prompt_text,
                }
            )
            initial_candidates = filtered_candidates(
                frames[initial_frame],
                args.hand_side,
                width,
                height,
                args.detection_confidence_threshold,
                args.reliability_distance_threshold_px,
            )
            obj_id = choose_target_object(
                initial_response.get("outputs", {}),
                initial_candidates,
                args.hand_side,
                width,
                height,
            )
            propagate_to_mask_dir(
                predictor,
                session_id,
                initial_frame,
                len(frames),
                obj_id,
                width,
                height,
                pass_dirs[0],
                initial_response.get("outputs", {}),
            )
            if args.segmentation_passes > 2:
                predictor.handle_request(
                    {
                        "type": "save_checkpoint",
                        "session_id": session_id,
                        "frame_index": initial_frame,
                        "propagation_direction": "both",
                        "exact_frame": True,
                    }
                )

            candidates_by_frame, computed_prompts = build_prompt_plan(
                frames,
                pass_dirs[0],
                args.hand_side,
                width,
                height,
                args.detection_confidence_threshold,
                args.reliability_distance_threshold_px,
                args.arm_distance_ratio,
                args.min_opposite_point_distance_px,
            )
            prompts_by_frame: List[List[PromptPoint]] = [[] for _ in range(len(frames))]
            final_dir = pass_dirs[0]
            prompt_source_dir = pass_dirs[0]
            prompt_source_pass = 1
            completed_passes = 1
            pass_summaries = [
                {"pass_index": 1, "prompted_frames": 0, "point_counts": {}}
            ]

            for pass_index in range(2, args.segmentation_passes + 1):
                if pass_index > 2:
                    restored = predictor.handle_request(
                        {
                            "type": "restore_checkpoint",
                            "session_id": session_id,
                            "frame_index": initial_frame,
                        }
                    )
                    if not restored.get("is_success", False):
                        raise RuntimeError("failed to restore the text-only baseline")
                iteration_prompts = computed_prompts
                prompted_frames = apply_prompt_plan(
                    predictor,
                    session_id,
                    obj_id,
                    iteration_prompts,
                    width,
                    height,
                )
                point_counts: Dict[str, int] = {}
                for points in iteration_prompts:
                    for point in points:
                        point_counts[point.kind] = point_counts.get(point.kind, 0) + 1
                if not prompted_frames:
                    LOGGER.info(
                        "Stopping before pass %d because no valid prompts remain",
                        pass_index,
                    )
                    break

                prompts_by_frame = iteration_prompts
                prompt_source_dir = final_dir
                prompt_source_pass = completed_passes
                final_dir = pass_dirs[pass_index - 1]
                propagate_to_mask_dir(
                    predictor,
                    session_id,
                    initial_frame,
                    len(frames),
                    obj_id,
                    width,
                    height,
                    final_dir,
                )
                completed_passes = pass_index
                pass_summaries.append(
                    {
                        "pass_index": pass_index,
                        "prompted_frames": prompted_frames,
                        "point_counts": point_counts,
                    }
                )
                if pass_index < args.segmentation_passes:
                    candidates_by_frame, computed_prompts = build_prompt_plan(
                        frames,
                        final_dir,
                        args.hand_side,
                        width,
                        height,
                        args.detection_confidence_threshold,
                        args.reliability_distance_threshold_px,
                        args.arm_distance_ratio,
                        args.min_opposite_point_distance_px,
                    )

            write_outputs(
                frame_dir,
                pass_dirs[0],
                prompt_source_dir,
                final_dir,
                candidates_by_frame,
                prompts_by_frame,
                output_dir,
                args.hand_side,
                width,
                height,
                float(video_info["fps"]),
                prompt_source_pass,
                completed_passes,
            )

        final_point_counts = pass_summaries[-1]["point_counts"]
        final_prompted_frames = pass_summaries[-1]["prompted_frames"]
        metadata.update(
            {
                "status": "success",
                "completed_at": utc_now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "frames_processed": len(frames),
                "target_obj_id": obj_id,
                "requested_segmentation_passes": args.segmentation_passes,
                "completed_segmentation_passes": completed_passes,
                "passes": pass_summaries,
                "prompted_frames": final_prompted_frames,
                "point_counts": final_point_counts,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "candidates": [joint_to_dict(joint) for joint in candidates],
                        "prompts": [prompt_to_dict(point) for point in prompts],
                    }
                    for frame_index, (candidates, prompts) in enumerate(
                        zip(candidates_by_frame, prompts_by_frame)
                    )
                ],
                "outputs": {
                    "baseline_masks_video": "baseline_masks.mkv",
                    "final_masks_video": "masks.mkv",
                    "visualization_video": "result.mp4",
                    "mask_codec": "FFV1",
                    "mask_pixel_format": "gray8",
                    "background_label": 0,
                    "target_label": 1,
                },
            }
        )
        write_json(metadata_path, metadata)
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
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
            except Exception:
                LOGGER.exception("Failed to close session %s", session_id)


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine one SAM hand-video mask with WiLoR joint prompts."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hand-side", required=True, choices=["left", "right"])
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    parser.add_argument(
        "--segmentation-passes",
        type=positive_int,
        default=2,
        help="Total segmentation passes: 1 is text-only, later passes recompute prompts",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt",
        help="Checkpoint path (auto-downloads from HuggingFace if omitted)",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=("cuda:0", 0),
        metavar="cuda:N",
    )
    parser.add_argument(
        "--detection-confidence-threshold", type=unit_interval, default=0.7
    )
    parser.add_argument(
        "--reliability-distance-threshold-px", type=nonnegative_float, default=25.0
    )
    parser.add_argument("--arm-distance-ratio", type=nonnegative_float, default=0.5)
    parser.add_argument(
        "--min-opposite-point-distance-px", type=nonnegative_float, default=5.0
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    input_dir = expand_path(args.input_dir)
    output_dir = expand_path(args.output_dir)
    checkpoint = expand_path(args.checkpoint) if args.checkpoint else None
    if not input_dir.is_dir():
        LOGGER.error("Input directory does not exist: %s", input_dir)
        return 2
    if input_dir == output_dir:
        LOGGER.error("Input and output directories must differ")
        return 2
    if checkpoint is not None and not checkpoint.is_file():
        LOGGER.error("Checkpoint does not exist: %s", checkpoint)
        return 2

    import torch

    device_name, device_index = args.device
    if not torch.cuda.is_available():
        LOGGER.error("CUDA is required by the SAM 3 video predictor")
        return 2
    if device_index >= torch.cuda.device_count():
        LOGGER.error(
            "Requested %s, but only %d CUDA device(s) are visible",
            device_name,
            torch.cuda.device_count(),
        )
        return 2
    torch.cuda.set_device(device_index)
    from sam3 import build_sam3_predictor

    build_kwargs = dict(version=args.version, compile=False, async_loading_frames=True)
    if checkpoint is not None:
        build_kwargs["checkpoint_path"] = str(checkpoint)
    predictor = build_sam3_predictor(**build_kwargs)
    try:
        process_video(predictor, args)
    except Exception:
        LOGGER.exception("Processing failed")
        return 1
    finally:
        predictor.shutdown()
    LOGGER.info("Completed %s -> %s", input_dir / "color.mp4", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

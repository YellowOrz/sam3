"""Video decoding, score and result export shared by desktop and headless clients."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sam3.video_io import (
    as_numpy,
    color_for_label,
    COLORS,
    H264Writer,
    lighter_color,
    utc_now as _utc_now,
)

MASK_ALPHA = 0.10

SELECTED_MASK_ALPHA = 0.20

SELECTED_COLOR = (255, 255, 0)

EDGE_HALO_THICKNESS = 2

EDGE_COLOR_THICKNESS = 1

SELECTED_HALO_THICKNESS = 4

SELECTED_EDGE_THICKNESS = 2

MASK_SCORE_SCHEMA = "sam3/mask-scores/1"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class PointEdit:
    sequence: int
    frame_index: int
    obj_id: int
    x: int
    y: int
    label: int

    def as_json(self) -> Dict[str, int]:
        return {
            "sequence": self.sequence,
            "frame_index": self.frame_index,
            "obj_id": self.obj_id,
            "x": self.x,
            "y": self.y,
            "label": self.label,
        }


def utc_now() -> str:
    return _utc_now("milliseconds")


def probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        info = VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        cap.release()
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError(f"invalid video dimensions: {video_path}")
    if not math.isfinite(info.fps) or info.fps <= 0:
        raise RuntimeError(f"invalid video FPS: {video_path}")
    return info


def extract_frames(video_path: Path, output_dir: Path) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video_path))
    index = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_path = output_dir / f"{index:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"failed to write extracted frame: {frame_path}")
            index += 1
    finally:
        cap.release()
    if index == 0:
        raise RuntimeError(f"video contains no readable frames: {video_path}")
    print(f"Extracted {index} frames to {output_dir}")
    return index


def frame_ranges(frame_count: int, chunk_frames: int) -> List[Tuple[int, int]]:
    """Return half-open frame ranges, or one full-video range for zero."""
    if frame_count < 1:
        raise ValueError("frame count must be positive")
    size = chunk_frames or frame_count
    return [
        (start, min(start + size, frame_count)) for start in range(0, frame_count, size)
    ]


def make_chunk_frame_dir(
    source_dir: Path, target_dir: Path, start: int, end: int
) -> None:
    target_dir.mkdir(parents=True)
    for local_index, source_index in enumerate(range(start, end)):
        source = source_dir / f"{source_index:05d}.jpg"
        target = target_dir / f"{local_index:05d}.jpg"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def load_frame_bgr(frame_dir: Path, frame_index: int) -> np.ndarray:
    frame_path = frame_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read frame: {frame_path}")
    return frame


def normalize_masks(outputs: Optional[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    if not outputs:
        return {}
    obj_ids = as_numpy(outputs.get("out_obj_ids")).reshape(-1)
    masks = as_numpy(outputs.get("out_binary_masks"))
    if masks.size == 0:
        return {}
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise RuntimeError(f"unexpected mask shape: {masks.shape}")
    if len(obj_ids) != len(masks):
        raise RuntimeError(
            f"object and mask counts differ: {len(obj_ids)} != {len(masks)}"
        )
    return {
        int(obj_id): np.asarray(mask, dtype=bool).copy()
        for obj_id, mask in zip(obj_ids, masks)
    }


def probability(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("a probability must be a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("a probability must be a finite number in [0, 1]")
    return value


def normalize_frame_outputs(
    outputs: Optional[Dict[str, Any]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, float]]:
    masks = normalize_masks(outputs)
    if not outputs:
        return masks, {}, {}
    obj_ids = as_numpy(outputs.get("out_obj_ids")).reshape(-1)
    scores = []
    for key in ("out_probs", "out_tracker_probs"):
        values = as_numpy(outputs.get(key)).reshape(-1)
        if len(values) not in (0, len(obj_ids)):
            raise RuntimeError(f"object and {key} counts differ")
        scores.append(
            {
                int(obj): probability(value)
                for obj, value in zip(obj_ids, values)
                if value is not None
            }
        )
    return masks, scores[0], scores[1]


def read_mask_scores(metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    board = metadata.get("mask_scores")
    if board is None:
        return None
    if not isinstance(board, dict) or board.get("schema_version") != MASK_SCORE_SCHEMA:
        raise ValueError("unsupported mask score schema")
    frames = board.get("frames")
    count = metadata["frames_processed"]
    start = metadata.get("source", {}).get("frame_start", 0)
    labels = {int(obj): label for obj, label in metadata["object_id_to_label"].items()}
    if (
        type(count) is not int
        or count <= 0
        or type(start) is not int
        or not isinstance(frames, list)
        or len(frames) != count
    ):
        raise ValueError("mask score frame count differs")
    for index, frame in enumerate(frames):
        if (
            not isinstance(frame, dict)
            or type(frame.get("frame_index")) is not int
            or frame["frame_index"] != start + index
            or not isinstance(frame.get("objects"), list)
        ):
            raise ValueError("invalid mask score frame")
        seen = set()
        for item in frame["objects"]:
            obj = item["object_id"]
            if (
                type(obj) is not int
                or obj not in labels
                or obj in seen
                or item["label"] != labels[obj]
            ):
                raise ValueError("mask score identity differs")
            seen.add(obj)
            for key in ("detection", "tracking"):
                probability(item[key])
    return frames


def export_mask_scores(
    app: Any, mapping: Dict[int, int], labels: Dict[int, int]
) -> Dict[str, Any]:
    frames = []
    for frame in range(app.frame_count):
        objects = []
        for obj, mask in sorted(app.cache[frame].items()):
            if not mask.any():
                continue
            exported = mapping[obj]
            objects.append(
                {
                    "object_id": exported,
                    "label": labels[exported],
                    "detection": probability(
                        app.probability_cache.get(frame, {}).get(obj)
                    ),
                    "tracking": probability(
                        app.tracking_probability_cache.get(frame, {}).get(obj)
                    ),
                }
            )
        frames.append({"frame_index": app.frame_offset + frame, "objects": objects})
    return {"schema_version": MASK_SCORE_SCHEMA, "frames": frames}


def render_frame_bgr(
    frame: np.ndarray,
    masks_by_obj: Dict[int, np.ndarray],
    probabilities_by_obj: Optional[Dict[int, float]] = None,
    object_to_label: Optional[Dict[int, int]] = None,
    frame_index: Optional[int] = None,
    prompt: Optional[str] = None,
    points: Sequence[PointEdit] = (),
    active_obj: Optional[int] = None,
    stale: bool = False,
    status: Optional[str] = None,
    object_names: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    probabilities_by_obj = probabilities_by_obj or {}
    object_to_label = object_to_label or {}
    blended = frame.astype(np.float32)
    for obj_id, mask in masks_by_obj.items():
        mask_bool = mask.astype(bool)
        if mask_bool.shape != frame.shape[:2]:
            raise RuntimeError(
                f"mask dimensions {mask_bool.shape} differ from frame {frame.shape[:2]}"
            )
        label = object_to_label.get(obj_id, obj_id)
        selected = obj_id == active_obj
        color = np.asarray(
            SELECTED_COLOR if selected else color_for_label(label), dtype=np.float32
        )
        alpha = SELECTED_MASK_ALPHA if selected else MASK_ALPHA
        blended[mask_bool] = blended[mask_bool] * (1.0 - alpha) + color * alpha
    rendered = blended.astype(np.uint8)
    for obj_id, mask in masks_by_obj.items():
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        label = object_to_label.get(obj_id, obj_id)
        selected = obj_id == active_obj
        color = SELECTED_COLOR if selected else color_for_label(label)
        cv2.drawContours(
            rendered,
            contours,
            -1,
            (255, 255, 255) if selected else lighter_color(color),
            SELECTED_HALO_THICKNESS if selected else EDGE_HALO_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.drawContours(
            rendered,
            contours,
            -1,
            color,
            SELECTED_EDGE_THICKNESS if selected else EDGE_COLOR_THICKNESS,
            cv2.LINE_AA,
        )
        ys, xs = np.where(mask)
        if len(xs):
            center = (int(np.median(xs)), int(np.median(ys)))
            text = f"label={label} id={obj_id}"
            if object_names is not None:
                text = object_names[obj_id]
            if obj_id in probabilities_by_obj:
                text += f" p={probabilities_by_obj[obj_id]:.2f}"
            cv2.putText(
                rendered,
                text,
                (max(0, center[0] - 40), max(18, center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    if frame_index is not None and prompt is not None:
        safe_prompt = prompt.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(
            rendered,
            f"frame={frame_index} prompt={safe_prompt}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for point in points:
        if point.label == 1:
            cv2.line(
                rendered,
                (point.x - 7, point.y),
                (point.x + 7, point.y),
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                rendered,
                (point.x, point.y - 7),
                (point.x, point.y + 7),
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.line(
                rendered,
                (point.x - 6, point.y - 6),
                (point.x + 6, point.y + 6),
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                rendered,
                (point.x - 6, point.y + 6),
                (point.x + 6, point.y - 6),
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        cv2.circle(rendered, (point.x, point.y), 9, (255, 255, 255), 1, cv2.LINE_AA)
    labels = []
    if stale:
        labels.append("STALE")
    if active_obj is not None:
        labels.append(f"obj {active_obj}")
    if status:
        labels.append(status)
    if labels:
        text = " | ".join(labels)
        cv2.putText(
            rendered,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)


def build_label_image(
    masks_by_obj: Dict[int, np.ndarray],
    object_to_label: Dict[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    label_image = np.zeros((height, width), dtype=np.uint8)
    for obj_id, mask in masks_by_obj.items():
        if obj_id not in object_to_label:
            label = len(object_to_label) + 1
            if label > 255:
                raise RuntimeError(
                    "more than 255 tracked instances; uint8 labels overflow"
                )
            object_to_label[obj_id] = label
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        label_image[mask.astype(bool)] = object_to_label[obj_id]
    return label_image


def remap_interaction(item: Dict[str, Any], mapping: Dict[int, int]) -> Dict[str, Any]:
    """Keep tracker IDs for audit; deleted objects have no exported ID."""
    result = dict(item)
    for key in ("obj_id", "active_obj"):
        if key in result and result[key] is not None:
            original = int(result[key])
            result.setdefault(f"original_{key}", original)
            result[key] = mapping.get(original)
    return result


def write_interactive_outputs(app: Any) -> None:
    original_ids = sorted({obj_id for masks in app.cache.values() for obj_id in masks})
    id_mapping = (
        {obj_id: obj_id for obj_id in original_ids}
        if hasattr(app, "loaded_labels")
        else {obj_id: index for index, obj_id in enumerate(original_ids)}
    )
    if len(id_mapping) > 255:
        raise RuntimeError("more than 255 tracked instances; uint8 labels overflow")
    # ponytail: reserve old labels; compact via full re-export if 255 are exhausted.
    object_to_label = (
        dict(app.loaded_labels)
        if hasattr(app, "loaded_labels")
        else {obj_id: obj_id + 1 for obj_id in id_mapping.values()}
    )
    for obj_id in id_mapping.values():
        if obj_id not in object_to_label:
            available = set(range(1, 256)) - set(object_to_label.values())
            if not available:
                raise RuntimeError("more than 255 instance labels; uint8 overflow")
            object_to_label[obj_id] = min(available)
    mask_scores = export_mask_scores(app, id_mapping, object_to_label)
    app.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = app.output_dir / "result.mp4"
    masks_path = app.output_dir / "masks.mkv"
    temporary_result = app.output_dir / ".result.tmp.mp4"
    temporary_masks = app.output_dir / ".masks.tmp.mkv"
    for path in (temporary_result, temporary_masks):
        if path.exists():
            path.unlink()
    result_writer = H264Writer(
        temporary_result,
        app.video_info.fps,
        (app.video_info.width, app.video_info.height),
    )
    if not result_writer.isOpened():
        result_writer.release()
        temporary_result.unlink(missing_ok=True)
        raise RuntimeError(f"cannot create output video: {temporary_result}")
    mask_writer = cv2.VideoWriter(
        str(temporary_masks),
        cv2.VideoWriter_fourcc(*"FFV1"),
        app.video_info.fps,
        (app.video_info.width, app.video_info.height),
        isColor=False,
    )
    if not mask_writer.isOpened():
        result_writer.release()
        mask_writer.release()
        for path in (temporary_result, temporary_masks):
            path.unlink(missing_ok=True)
        raise RuntimeError(f"cannot create lossless mask video: {temporary_masks}")
    try:
        for frame_index in range(app.frame_count):
            masks_by_obj = {
                id_mapping[obj_id]: mask
                for obj_id, mask in app.cache[frame_index].items()
            }
            label_image = build_label_image(
                masks_by_obj,
                object_to_label,
                app.video_info.width,
                app.video_info.height,
            )
            mask_writer.write(label_image)
            result_writer.write(
                render_frame_bgr(
                    load_frame_bgr(app.frame_dir, frame_index),
                    masks_by_obj,
                    probabilities_by_obj={
                        id_mapping[obj_id]: probability
                        for obj_id, probability in app.probability_cache.get(
                            frame_index, {}
                        ).items()
                        if obj_id in id_mapping
                    },
                    object_to_label=object_to_label,
                    frame_index=app.frame_offset + frame_index,
                    prompt=app.prompt,
                )
            )
    except BaseException:
        result_writer.release()
        mask_writer.release()
        for path in (temporary_result, temporary_masks):
            path.unlink(missing_ok=True)
        raise
    finally:
        result_writer.release()
        mask_writer.release()
    temporary_result.replace(result_path)
    temporary_masks.replace(masks_path)
    source = {
        "width": app.video_info.width,
        "height": app.video_info.height,
        "frame_count": app.frame_count,
        "fps": app.video_info.fps,
        "frame_start": app.frame_offset,
    }
    outputs = {
        "instance_masks_video": "masks.mkv",
        "instance_masks_codec": "FFV1",
        "instance_masks_pixel_format": "gray8",
        "visualization_video": "result.mp4",
        "label_dtype": "uint8",
        "background_label": 0,
    }
    atomic_write_json(
        app.output_dir / "metadata.json",
        {
            **(app.loaded_metadata if hasattr(app, "loaded_labels") else {}),
            "status": "success",
            "completed_at": utc_now(),
            "input_video": str(app.video_path),
            "prompt": app.prompt,
            "model_version": app.version,
            "clip_overseg": app.clip_overseg,
            "source": source,
            "frames_processed": app.frame_count,
            **({"propagation": app.propagation} if hasattr(app, "propagation") else {}),
            "mask_scores": mask_scores,
            "original_object_id_to_object_id": id_mapping,
            "object_id_to_label": {
                str(obj_id): label for obj_id, label in sorted(object_to_label.items())
            },
            "outputs": outputs,
        },
    )
    atomic_write_json(
        app.output_dir / "interactions.json",
        {
            "status": "success",
            "created_at": utc_now(),
            "input_video": str(app.video_path),
            "model_version": app.version,
            "text_prompt": app.prompt,
            "propagation_direction": app.initial_propagation_mode,
            "source": source,
            "outputs": outputs,
            "confirmed_points": [
                remap_interaction(
                    {
                        **getattr(app, "loaded_points", {}).get(p.sequence, {}),
                        **p.as_json(),
                    },
                    id_mapping,
                )
                for p in sorted(app.confirmed_points, key=lambda point: point.sequence)
            ],
            "events": [remap_interaction(event, id_mapping) for event in app.events],
        },
    )
    print(f"Saved interactive outputs to {app.output_dir}")


def remap_chunk_interaction(
    item: Dict[str, Any], mapping: Dict[int, int]
) -> Dict[str, Any]:
    result = dict(item)
    for key in ("obj_id", "active_obj"):
        if key in result:
            local_id = result[key]
            result[f"local_{key}"] = local_id
            result[key] = mapping.get(local_id)
    return result


def validate_chunk_mappings(
    mappings: Sequence[Dict[int, int]], local_ids: Sequence[Dict[int, int]]
) -> None:
    if len(mappings) != len(local_ids):
        raise ValueError("one mapping is required per chunk")
    for index, (mapping, ids) in enumerate(zip(mappings, local_ids), 1):
        if mapping.keys() != ids.keys():
            raise ValueError(f"chunk {index}: mapping must cover exactly its local IDs")
        if any(not isinstance(value, int) or value < 0 for value in mapping.values()):
            raise ValueError("global IDs must be non-negative integers")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(
                f"chunk {index}: duplicate global ID; submit swaps together, "
                f"e.g. map {index} 1=2 2=1"
            )
    if len({value for mapping in mappings for value in mapping.values()}) > 255:
        raise ValueError("more than 255 global objects; uint8 labels overflow")


def merge_chunk_outputs(
    chunks: Sequence[Tuple[int, int, Path]],
    output_dir: Path,
    video_path: Path,
    video_info: VideoInfo,
    prompt: str,
    version: str,
    chunk_frames: int,
    frame_dir: Path,
    mappings: Optional[Sequence[Dict[int, int]]] = None,
    confirmed: bool = False,
    clip_overseg: bool = False,
) -> None:
    metadata_by_chunk = [
        json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        for _, _, directory in chunks
    ]
    local_labels = [
        {
            int(obj_id): int(label)
            for obj_id, label in meta["object_id_to_label"].items()
        }
        for meta in metadata_by_chunk
    ]
    if mappings is None:
        mappings = [{obj_id: obj_id for obj_id in labels} for labels in local_labels]
    validate_chunk_mappings(mappings, local_labels)
    global_ids = sorted({obj_id for mapping in mappings for obj_id in mapping.values()})
    global_labels = {obj_id: index + 1 for index, obj_id in enumerate(global_ids)}
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_result = output_dir / ".result.tmp.mp4"
    temporary_masks = output_dir / ".masks.tmp.mkv"
    for path in (temporary_result, temporary_masks):
        path.unlink(missing_ok=True)
    result_writer = H264Writer(
        temporary_result,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    mask_writer = cv2.VideoWriter(
        str(temporary_masks),
        cv2.VideoWriter_fourcc(*"FFV1"),
        video_info.fps,
        (video_info.width, video_info.height),
        isColor=False,
    )
    if not result_writer.isOpened() or not mask_writer.isOpened():
        result_writer.release()
        mask_writer.release()
        for path in (temporary_result, temporary_masks):
            path.unlink(missing_ok=True)
        raise RuntimeError("cannot create merged output videos")

    chunk_metadata = []
    score_frames = []
    confirmed_points = []
    events = []
    sequence_offset = 0
    try:
        for chunk_index, (start, end, chunk_dir) in enumerate(chunks, 1):
            mapping = mappings[chunk_index - 1]
            labels = local_labels[chunk_index - 1]
            local_scores = read_mask_scores(metadata_by_chunk[chunk_index - 1])
            if local_scores is not None and len(local_scores) != end - start:
                raise ValueError("chunk score frame count differs from chunk range")
            mask_capture = cv2.VideoCapture(str(chunk_dir / "masks.mkv"))
            try:
                if not mask_capture.isOpened():
                    raise RuntimeError(f"cannot open outputs for chunk {chunk_index}")
                for frame_index in range(start, end):
                    mask_ok, mask_frame = mask_capture.read()
                    if not mask_ok:
                        raise RuntimeError(f"chunk {chunk_index} output is incomplete")
                    if mask_frame.shape[:2] != (video_info.height, video_info.width):
                        raise RuntimeError(f"chunk {chunk_index} dimensions differ")
                    local_image = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    if not set(np.unique(local_image)).issubset({0, *labels.values()}):
                        raise RuntimeError(
                            f"chunk {chunk_index} contains unknown labels"
                        )
                    masks = {
                        mapping[local_id]: local_image == label
                        for local_id, label in labels.items()
                    }
                    objects = []
                    known = (
                        {
                            item["object_id"]: item
                            for item in local_scores[frame_index - start]["objects"]
                        }
                        if local_scores is not None
                        else {}
                    )
                    if local_scores is not None and (
                        local_scores[frame_index - start]["frame_index"] != frame_index
                        or set(known)
                        != {
                            obj
                            for obj, label in labels.items()
                            if (local_image == label).any()
                        }
                    ):
                        raise ValueError(
                            "chunk scores do not match mask instances/frames"
                        )
                    for local_id, global_id in mapping.items():
                        if (local_image == labels[local_id]).any():
                            item = known.get(local_id, {})
                            objects.append(
                                {
                                    "object_id": global_id,
                                    "label": global_labels[global_id],
                                    "detection": item.get("detection"),
                                    "tracking": item.get("tracking"),
                                }
                            )
                    score_frames.append(
                        {"frame_index": frame_index, "objects": objects}
                    )
                    mask_writer.write(
                        build_label_image(
                            masks, global_labels, video_info.width, video_info.height
                        )
                    )
                    result_writer.write(
                        render_frame_bgr(
                            load_frame_bgr(frame_dir, frame_index),
                            masks,
                            object_to_label=global_labels,
                            frame_index=frame_index,
                            prompt=f"chunk={chunk_index} {prompt}",
                            object_names={
                                global_id: (
                                    f"chunk={chunk_index} local={local_id} "
                                    f"id={global_id}"
                                )
                                for local_id, global_id in mapping.items()
                            },
                        )
                    )
            finally:
                mask_capture.release()

            metadata = metadata_by_chunk[chunk_index - 1]
            interactions = json.loads(
                (chunk_dir / "interactions.json").read_text(encoding="utf-8")
            )
            chunk_metadata.append(
                {
                    "index": chunk_index,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "frame_count": end - start,
                    "object_id_to_label": metadata["object_id_to_label"],
                    "original_object_id_to_object_id": metadata.get(
                        "original_object_id_to_object_id", {}
                    ),
                    "local_object_id_to_global_object_id": mapping,
                }
            )
            local_sequences = [
                int(item.get("sequence", 0))
                for item in [
                    *interactions.get("confirmed_points", []),
                    *interactions.get("events", []),
                ]
            ]
            for item in interactions.get("confirmed_points", []):
                item = remap_chunk_interaction(item, mapping)
                item["sequence"] += sequence_offset
                item["frame_index"] += start
                item["chunk_index"] = chunk_index
                confirmed_points.append(item)
            for item in interactions.get("events", []):
                item = remap_chunk_interaction(item, mapping)
                item["sequence"] += sequence_offset
                for key in ("frame_index", "target_frame", "checkpoint_frame"):
                    if key in item:
                        item[key] += start
                if "point_sequence" in item:
                    item["point_sequence"] += sequence_offset
                if "point_sequences" in item:
                    item["point_sequences"] = [
                        value + sequence_offset for value in item["point_sequences"]
                    ]
                item["chunk_index"] = chunk_index
                events.append(item)
            sequence_offset += max(local_sequences, default=0)
    except BaseException:
        result_writer.release()
        mask_writer.release()
        for path in (temporary_result, temporary_masks):
            path.unlink(missing_ok=True)
        raise
    finally:
        result_writer.release()
        mask_writer.release()

    temporary_result.replace(output_dir / "result.mp4")
    temporary_masks.replace(output_dir / "masks.mkv")
    source = {
        "width": video_info.width,
        "height": video_info.height,
        "frame_count": video_info.frame_count,
        "fps": video_info.fps,
    }
    outputs = {
        "instance_masks_video": "masks.mkv",
        "instance_masks_codec": "FFV1",
        "instance_masks_pixel_format": "gray8",
        "visualization_video": "result.mp4",
        "label_dtype": "uint8",
        "background_label": 0,
        "label_scope": "video",
    }
    common = {
        "status": "success" if confirmed else "unconfirmed",
        "review_confirmed": confirmed,
        "object_id_to_label": global_labels,
        "input_video": str(video_path),
        "model_version": version,
        "clip_overseg": clip_overseg,
        "source": source,
        "chunk_frames": chunk_frames,
        "chunks": chunk_metadata,
        "outputs": outputs,
    }
    atomic_write_json(
        output_dir / "metadata.json",
        {
            **common,
            "completed_at": utc_now(),
            "prompt": prompt,
            "frames_processed": video_info.frame_count,
            "object_identity_scope": "video",
            "mask_scores": {
                "schema_version": MASK_SCORE_SCHEMA,
                "frames": score_frames,
            },
        },
    )
    atomic_write_json(
        output_dir / "interactions.json",
        {
            **common,
            "created_at": utc_now(),
            "text_prompt": prompt,
            "propagation_direction": "forward",
            "confirmed_points": confirmed_points,
            "events": events,
        },
    )
    print(f"Merged {len(chunks)} chunks into {output_dir}")

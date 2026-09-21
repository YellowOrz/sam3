#!/usr/bin/env python3
"""Run SAM 3/3.1 text-prompted video segmentation over RGB video sequences.

The input root and all subdirectories are searched for ``--rgb-name``
(default: ``color.mp4``). Input and output roots must differ.
Each video is decoded to a temporary lossless PNG directory because
the SAM 3 image-directory loader is substantially more memory efficient than its
direct OpenCV video loader.

GT evaluation: append --compare-gt --gt-mask-name left_hand.mkv.
GT directory defaults to masks_sam3; --compare-skip-frames defaults to 0.
The GT filename must be explicit. Complete predictions can be evaluated on CPU.

--list-only lists discovered input videos without loading the model, running
segmentation or GT evaluation, or writing output files.

Example:
    uv run python scripts/process_dataset_videos.py \
        --input-root ~/Datasets \
        --output-root ~/sam3_outputs \
        --version sam3.1 \
        --text-prompt hand \
        --device cuda:2
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np

if __package__:
    from scripts.common.compare_gt_masks import (
        DETECTOR_COLOR,
        GTComparison,
        validate_gt_arguments,
    )
    from scripts.common.video_cli import (
        add_video_arguments,
        discover_rgb_videos,
        validate_input_output,
    )
    from scripts.common.video_utils import (
        color_for_label,
        draw_geometry,
        expand_path,
        extract_png_frames,
        lighter_color,
        normalize_output_arrays,
        positive_int,
        probe_video as _probe_video,
        utc_now,
        write_json,
    )
else:
    from common.compare_gt_masks import (  # type: ignore[no-redef]
        DETECTOR_COLOR,
        GTComparison,
        validate_gt_arguments,
    )
    from common.video_cli import (
        add_video_arguments,
        discover_rgb_videos,
        validate_input_output,
    )
    from common.video_utils import (  # type: ignore[no-redef]
        color_for_label,
        draw_geometry,
        expand_path,
        extract_png_frames,
        lighter_color,
        normalize_output_arrays,
        positive_int,
        probe_video as _probe_video,
        utc_now,
        write_json,
    )


LOGGER = logging.getLogger("sam3_dataset_processor")
# BGR colors for OpenCV rendering. Label zero is the background and never uses
# this palette.
MASK_ALPHA = 0.30
EDGE_HALO_THICKNESS = 2
EDGE_COLOR_THICKNESS = 1


def discover_color_videos(input_root: Path) -> List[Path]:
    """Find a direct color.mp4 or recursively discover dataset sequences."""
    direct_video = input_root / "color.mp4"
    if direct_video.is_file():
        return [direct_video]
    return discover_rgb_videos(input_root, "color.mp4")


def output_dir_for(video_path: Path, input_root: Path, output_root: Path) -> Path:
    if video_path.parent == input_root:
        return output_root
    return output_root / video_path.parent.relative_to(input_root)


def processing_directions(direction: str) -> Tuple[str, ...]:
    return ("forward", "backward") if direction == "both" else (direction,)


def directional_output_dir(
    output_dir: Path, requested_direction: str, direction: str
) -> Path:
    if requested_direction != "both":
        return output_dir
    return output_dir / direction


def probe_video(video_path: Path) -> Dict[str, Any]:
    return _probe_video(video_path, fallback_fps=30.0)


def expected_frame_count(video_info: Dict[str, Any], max_frames: Optional[int]) -> int:
    frame_count = int(video_info["frame_count"])
    if max_frames is None:
        return frame_count
    if frame_count <= 0:
        return max_frames
    return min(frame_count, max_frames)


def is_complete(
    output_dir: Path,
    video_path: Path,
    prompt: str,
    model_version: str,
    expected_frames: int,
    direction: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    save_detector: bool = False,
    detector_only: bool = False,
) -> bool:
    metadata_path = output_dir / "metadata.json"
    result_path = output_dir / "result.mp4"
    masks_path = output_dir / "masks.mkv"
    detector_path = output_dir / "detector_masks.mkv"
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False

    if not (
        metadata.get("status") == "success"
        and metadata.get("input_video") == str(video_path)
        and metadata.get("prompt") == prompt
        and metadata.get("model_version") == model_version
        and metadata.get("detector_only", False) == detector_only
        and metadata.get("propagation_direction", "forward") == direction
        and metadata.get("frames_processed") == expected_frames
        and all(
            metadata.get(key) == value for key, value in (extra_metadata or {}).items()
        )
        and result_path.is_file()
        and masks_path.is_file()
        and (
            not save_detector
            or (
                detector_path.is_file()
                and (output_dir / "detector_result.mp4").is_file()
            )
        )
    ):
        return False
    try:
        mask_info = probe_video(masks_path)
        detector_info = probe_video(detector_path) if save_detector else mask_info
    except RuntimeError:
        return False
    return int(mask_info["frame_count"]) == expected_frames and (
        not save_detector or int(detector_info["frame_count"]) == expected_frames
    )


def prepare_output_dir(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Remove stale generated artifacts and return their output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_path = output_dir / "masks.mkv"
    result_path = output_dir / "result.mp4"
    metadata_path = output_dir / "metadata.json"
    legacy_masks_dir = output_dir / "masks"
    if legacy_masks_dir.exists():
        shutil.rmtree(legacy_masks_dir)
    for generated_path in (
        masks_path,
        result_path,
        output_dir / "detector_masks.mkv",
        output_dir / "detector_result.mp4",
    ):
        if generated_path.exists():
            generated_path.unlink()
    return masks_path, result_path, metadata_path


def normalize_outputs(
    outputs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return normalize_output_arrays(outputs)


def build_label_and_overlay(
    frame: np.ndarray,
    outputs: Dict[str, Any],
    object_to_label: Dict[int, int],
    frame_index: int,
    prompt: str,
    alpha: float = MASK_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert model masks into an 8-bit label image and annotated BGR frame."""
    height, width = frame.shape[:2]
    label_image = np.zeros((height, width), dtype=np.uint8)
    obj_ids, probs, masks = normalize_outputs(outputs)
    objects_in_frame = []

    for index, raw_obj_id in enumerate(obj_ids):
        obj_id = int(raw_obj_id)
        if obj_id not in object_to_label:
            next_label = len(object_to_label) + 1
            if next_label > 255:
                raise RuntimeError(
                    "more than 255 tracked instances; uint8 labels overflow"
                )
            object_to_label[obj_id] = next_label
        label = object_to_label[obj_id]

        mask = masks[index]
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue
        label_image[mask_bool] = label
        probability = float(probs[index]) if len(probs) else None
        objects_in_frame.append((obj_id, label, probability, mask_bool))

    overlay = frame.copy()
    for _, label, _, mask_bool in objects_in_frame:
        color = np.asarray(color_for_label(label), dtype=np.float32)
        blended = overlay[mask_bool].astype(np.float32) * (1.0 - alpha) + color * alpha
        overlay[mask_bool] = blended.astype(np.uint8)

    for obj_id, label, probability, mask_bool in objects_in_frame:
        color = color_for_label(label)
        # RETR_LIST keeps both outer contours and inner boundaries (for example
        # holes in a mask). Use a lighter variant of the instance color for the
        # halo so the highlighted edge stays in the same color family.
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            lighter_color(color),
            EDGE_HALO_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            EDGE_COLOR_THICKNESS,
            cv2.LINE_AA,
        )
        ys, xs = np.nonzero(mask_bool)
        x = int(np.median(xs))
        y = int(np.median(ys))
        text = f"label={label} id={obj_id}"
        if probability is not None:
            text += f" p={probability:.2f}"
        cv2.putText(
            overlay,
            text,
            (max(0, x - 40), max(18, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    safe_prompt = prompt.encode("ascii", errors="replace").decode("ascii")
    header = f"frame={frame_index} prompt={safe_prompt}"
    cv2.putText(
        overlay,
        header,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return label_image, overlay


def empty_outputs() -> Dict[str, np.ndarray]:
    return {
        "out_obj_ids": np.empty((0,), dtype=np.int64),
        "out_probs": np.empty((0,), dtype=np.float32),
        "out_binary_masks": np.empty((0, 0, 0), dtype=bool),
    }


def detector_label_image(
    outputs: Dict[str, Any], height: int, width: int
) -> np.ndarray:
    mask = outputs.get("out_detector_mask")
    if mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.any(mask.reshape(-1, *mask.shape[-2:]), axis=0)
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
    return (mask != 0).astype(np.uint8)


def write_frame_outputs(
    frame_dir: Path,
    mask_writer: cv2.VideoWriter,
    result_writer: cv2.VideoWriter,
    frame_index: int,
    outputs: Dict[str, Any],
    object_to_label: Dict[int, int],
    prompt: str,
    geometry_prompts: Optional[Dict[int, Dict[str, Any]]] = None,
    overlay_geometry: Optional[Dict[int, Dict[str, Any]]] = None,
    detector_writer: Optional[cv2.VideoWriter] = None,
    detector_result_writer: Optional[cv2.VideoWriter] = None,
) -> None:
    frame_path = frame_dir / f"{frame_index:06d}.png"
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read temporary frame: {frame_path}")
    label_image, overlay = build_label_and_overlay(
        frame, outputs, object_to_label, frame_index, prompt
    )
    draw_geometry(
        overlay, (overlay_geometry or geometry_prompts or {}).get(frame_index, {})
    )
    mask_writer.write(label_image)
    result_writer.write(overlay)
    detector = (
        detector_label_image(outputs, frame.shape[0], frame.shape[1])
        if detector_writer is not None or detector_result_writer is not None
        else None
    )
    if detector_writer is not None:
        detector_writer.write(detector)
    if detector_result_writer is not None:
        panel = frame.copy()
        mask = detector != 0
        if mask.any():
            panel[mask] = (
                panel[mask] * (1 - MASK_ALPHA) + DETECTOR_COLOR * MASK_ALPHA
            ).astype(np.uint8)
        draw_geometry(panel, (geometry_prompts or {}).get(frame_index, {}))
        detector_result_writer.write(panel)


def propagate_and_write(
    predictor: Any,
    session_id: str,
    frame_dir: Path,
    masks_path: Path,
    result_path: Path,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    prompt: str,
    direction: str,
    geometry_prompts: Optional[Dict[int, Dict[str, Any]]] = None,
    overlay_geometry: Optional[Dict[int, Dict[str, Any]]] = None,
    save_detector: bool = False,
    prompt_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[int, int]:
    result_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_writer = cv2.VideoWriter(
        str(result_path), result_fourcc, fps, (width, height)
    )
    if not result_writer.isOpened():
        result_writer.release()
        raise RuntimeError(f"cannot create output video: {result_path}")

    mask_fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    mask_writer = cv2.VideoWriter(
        str(masks_path), mask_fourcc, fps, (width, height), isColor=False
    )
    if not mask_writer.isOpened():
        result_writer.release()
        mask_writer.release()
        raise RuntimeError(f"cannot create lossless mask video: {masks_path}")

    detector_writer = None
    detector_result_writer = None
    if save_detector:
        detector_writer = cv2.VideoWriter(
            str(masks_path.with_name("detector_masks.mkv")),
            mask_fourcc,
            fps,
            (width, height),
            isColor=False,
        )
        detector_result_writer = cv2.VideoWriter(
            str(result_path.with_name("detector_result.mp4")),
            result_fourcc,
            fps,
            (width, height),
        )
        if not detector_writer.isOpened() or not detector_result_writer.isOpened():
            result_writer.release()
            mask_writer.release()
            detector_writer.release()
            detector_result_writer.release()
            raise RuntimeError("cannot create detector mask or result video")

    object_to_label: Dict[int, int] = {}
    prompt_frame = 0 if direction == "forward" else frame_count - 1
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": direction,
        "start_frame_index": prompt_frame,
        "max_frame_num_to_track": frame_count,
    }

    def write(frame_index: int, outputs: Dict[str, Any]) -> None:
        if (
            save_detector
            and prompt_outputs is not None
            and frame_index == prompt_frame
            and outputs.get("out_detector_mask") is None
        ):
            outputs = {
                **outputs,
                "out_detector_mask": prompt_outputs.get("out_detector_mask"),
            }
        write_frame_outputs(
            frame_dir,
            mask_writer,
            result_writer,
            frame_index,
            outputs,
            object_to_label,
            prompt,
            geometry_prompts,
            overlay_geometry,
            detector_writer,
            detector_result_writer,
        )

    try:
        if direction == "backward":
            with tempfile.TemporaryDirectory(
                prefix="sam3_backward_outputs_"
            ) as temporary:
                output_cache = Path(temporary)
                previous_frame_index = frame_count
                for response in predictor.handle_stream_request(request):
                    frame_index = int(response["frame_index"])
                    if not 0 <= frame_index < frame_count:
                        continue
                    if frame_index >= previous_frame_index:
                        raise RuntimeError(
                            f"model returned out-of-order frame {frame_index} "
                            "while propagating backward"
                        )
                    previous_frame_index = frame_index
                    outputs = response.get("outputs", empty_outputs())
                    obj_ids, probs, masks = normalize_outputs(outputs)
                    saved = {
                        "out_obj_ids": obj_ids,
                        "out_probs": probs,
                        "out_binary_masks": masks,
                    }
                    if save_detector:
                        saved["out_detector_mask"] = detector_label_image(
                            outputs, height, width
                        )
                    np.savez(output_cache / f"{frame_index:06d}.npz", **saved)

                for frame_index in range(frame_count):
                    cached_path = output_cache / f"{frame_index:06d}.npz"
                    if cached_path.is_file():
                        with np.load(cached_path) as cached:
                            write(frame_index, dict(cached.items()))
                    else:
                        write(frame_index, empty_outputs())
        else:
            next_frame_index = 0
            for response in predictor.handle_stream_request(request):
                frame_index = int(response["frame_index"])
                if not 0 <= frame_index < frame_count:
                    continue
                if frame_index < next_frame_index:
                    raise RuntimeError(
                        f"model returned out-of-order frame {frame_index} after "
                        f"{next_frame_index - 1}"
                    )
                while next_frame_index < frame_index:
                    write(next_frame_index, empty_outputs())
                    next_frame_index += 1
                write(frame_index, response.get("outputs", empty_outputs()))
                next_frame_index = frame_index + 1

            while next_frame_index < frame_count:
                write(next_frame_index, empty_outputs())
                next_frame_index += 1
    finally:
        mask_writer.release()
        result_writer.release()
        if detector_writer is not None:
            detector_writer.release()
        if detector_result_writer is not None:
            detector_result_writer.release()
    return object_to_label


def add_prompt_request(
    session_id: str,
    frame_index: int,
    prompt: str,
    prompt_request_type: str,
) -> Dict[str, Any]:
    if prompt_request_type == "learned":
        return {
            "type": "add_learned_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "target_id": prompt,
        }
    if prompt_request_type == "text":
        return {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_index,
            "text": prompt,
        }
    raise ValueError(f"unsupported prompt request type: {prompt_request_type}")


def process_video(
    predictor: Any,
    video_path: Path,
    output_dir: Path,
    input_root: Path,
    prompt: str,
    model_version: str,
    max_frames: Optional[int],
    overwrite: bool,
    direction: str,
    prompt_request_type: str = "text",
    extra_metadata: Optional[Dict[str, Any]] = None,
    geometry_prompts: Optional[Dict[int, Dict[str, Any]]] = None,
    overlay_geometry: Optional[Dict[int, Dict[str, Any]]] = None,
    predictor_factory: Optional[Callable[[], Any]] = None,
    save_detector: bool = False,
    detector_only: bool = False,
) -> str:
    if detector_only and (model_version != "sam3" or geometry_prompts is None):
        raise ValueError("detector-only requires SAM3 geometry prompts")
    video_info = probe_video(video_path)
    requested_frames = expected_frame_count(video_info, max_frames)
    if not overwrite and is_complete(
        output_dir,
        video_path,
        prompt,
        model_version,
        requested_frames,
        direction,
        extra_metadata if geometry_prompts is not None else None,
        save_detector,
        detector_only,
    ):
        LOGGER.info("Skipping completed sequence: %s", video_path)
        return "skipped"

    if predictor_factory is not None:
        predictor = predictor_factory()

    masks_path, result_path, metadata_path = prepare_output_dir(output_dir)
    relative_video = video_path.relative_to(input_root).as_posix()
    metadata: Dict[str, Any] = {
        "status": "processing",
        "input_video": str(video_path),
        "input_relative_path": relative_video,
        "prompt": prompt,
        "model_version": model_version,
        "detector_only": detector_only,
        "propagation_direction": direction,
        "source": video_info,
        "started_at": utc_now(),
        "frames_processed": 0,
        "object_id_to_label": {},
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(metadata_path, metadata)

    started = time.monotonic()
    session_id: Optional[str] = None
    try:
        with tempfile.TemporaryDirectory(prefix="sam3_color_frames_") as temporary:
            frame_dir = Path(temporary)
            frame_count = extract_png_frames(video_path, frame_dir, max_frames)
            LOGGER.info("Extracted %d frames from %s", frame_count, relative_video)
            session_response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(frame_dir),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = session_response["session_id"]
            request = add_prompt_request(
                session_id,
                0 if direction == "forward" else frame_count - 1,
                prompt,
                prompt_request_type,
            )
            if geometry_prompts is not None:
                request.update(
                    type="add_geometry_prompts",
                    geometry_prompts=geometry_prompts,
                    detector_only=detector_only,
                )
            prompt_response = predictor.handle_request(request)
            object_to_label = propagate_and_write(
                predictor=predictor,
                session_id=session_id,
                frame_dir=frame_dir,
                masks_path=masks_path,
                result_path=result_path,
                frame_count=frame_count,
                width=int(video_info["width"]),
                height=int(video_info["height"]),
                fps=float(video_info["fps"]),
                prompt=prompt,
                direction=direction,
                geometry_prompts=geometry_prompts,
                overlay_geometry=overlay_geometry,
                save_detector=save_detector,
                prompt_outputs=(
                    prompt_response.get("outputs") if save_detector else None
                ),
            )

        metadata.update(
            {
                "status": "success",
                "completed_at": utc_now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "frames_processed": frame_count,
                "object_id_to_label": {
                    str(obj_id): label
                    for obj_id, label in sorted(object_to_label.items())
                },
                "outputs": {
                    "instance_masks_video": masks_path.name,
                    "instance_masks_codec": "FFV1",
                    "instance_masks_pixel_format": "gray8",
                    "visualization_video": result_path.name,
                    "label_dtype": "uint8",
                    "background_label": 0,
                    **({"mask_semantics": "binary_union"} if detector_only else {}),
                    **(
                        {
                            "detector_masks_video": "detector_masks.mkv",
                            "detector_visualization_video": "detector_result.mp4",
                        }
                        if save_detector
                        else {}
                    ),
                },
            }
        )
        write_json(metadata_path, metadata)
        LOGGER.info("Completed %s -> %s", relative_video, result_path)
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
    finally:
        if session_id is not None:
            try:
                predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
            except Exception:
                LOGGER.exception("Failed to close session %s", session_id)


def require_cuda(device_name: str, device_index: int) -> bool:
    import torch

    if not torch.cuda.is_available():
        LOGGER.error("CUDA is required by the SAM 3 video predictor")
        return False
    if device_index >= torch.cuda.device_count():
        LOGGER.error(
            "Requested %s, but only %d CUDA device(s) are visible",
            device_name,
            torch.cuda.device_count(),
        )
        return False
    torch.cuda.set_device(device_index)
    return True


@contextmanager
def lazy_predictor(args: argparse.Namespace) -> Iterator[Callable[[], Any]]:
    """Load once, only when an incomplete sequence needs inference."""
    predictor = None

    def get_predictor() -> Any:
        nonlocal predictor
        if predictor is None:
            if not require_cuda(*args.device):
                raise RuntimeError("CUDA is required for video inference")
            from sam3 import build_sam3_predictor

            kwargs = dict(
                version=args.version, compile=False, async_loading_frames=True
            )
            if args.checkpoint:
                kwargs["checkpoint_path"] = str(expand_path(args.checkpoint))
            LOGGER.info("Loading %s on %s", args.version, args.device[0])
            predictor = build_sam3_predictor(**kwargs)
        return predictor

    try:
        yield get_predictor
    finally:
        if predictor is not None:
            predictor.shutdown()


def run_sequences(
    predictor: Any,
    videos: List[Path],
    input_root: Path,
    output_root: Path,
    prompt: str,
    model_version: str,
    max_frames: Optional[int],
    overwrite: bool,
    requested_direction: str,
    prompt_request_type: str = "text",
    extra_metadata: Optional[Dict[str, Any]] = None,
    comparison: Optional[GTComparison] = None,
    predictor_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, int]:
    counts = {"success": 0, "skipped": 0, "failed": 0}
    directions = processing_directions(requested_direction)
    for index, video_path in enumerate(videos, start=1):
        base_output_dir = output_dir_for(video_path, input_root, output_root)
        LOGGER.info("[%d/%d] Processing %s", index, len(videos), video_path)
        for direction in directions:
            output_dir = directional_output_dir(
                base_output_dir, requested_direction, direction
            )
            try:
                status = process_video(
                    predictor=predictor,
                    video_path=video_path,
                    output_dir=output_dir,
                    input_root=input_root,
                    prompt=prompt,
                    model_version=model_version,
                    max_frames=max_frames,
                    overwrite=overwrite,
                    direction=direction,
                    prompt_request_type=prompt_request_type,
                    extra_metadata=extra_metadata,
                    predictor_factory=predictor_factory,
                )
                if comparison is not None:
                    comparison.compare(video_path, output_dir, direction)
                counts[status] += 1
            except Exception as exc:
                counts["failed"] += 1
                LOGGER.exception("Sequence failed (%s): %s", direction, video_path)
                if comparison is not None:
                    comparison.record_failure(video_path, output_dir, direction, exc)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively process RGB videos with a shared SAM 3/3.1 " "text prompt."
        )
    )
    add_video_arguments(parser)
    parser.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    parser.add_argument(
        "--text-prompt", dest="prompt", required=True, help="Shared text prompt"
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_gt_arguments(parser, args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    input_root = expand_path(args.input_root)
    output_root = expand_path(args.output_root)
    checkpoint = expand_path(args.checkpoint) if args.checkpoint else None

    try:
        validate_input_output(input_root, output_root)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    videos = discover_rgb_videos(input_root, args.rgb_name)
    if args.max_sequences is not None:
        videos = videos[: args.max_sequences]
    if not videos:
        LOGGER.error("No files named %s found below %s", args.rgb_name, input_root)
        return 2

    LOGGER.info("Discovered %d color video(s)", len(videos))
    if args.list_only:
        for video_path in videos:
            print(video_path.relative_to(input_root))
        return 0
    if checkpoint is not None and not checkpoint.is_file():
        LOGGER.error("Checkpoint does not exist: %s", checkpoint)
        return 2

    comparison = GTComparison(args, processing_directions(args.direction))
    with lazy_predictor(args) as get_predictor:
        counts = run_sequences(
            predictor=None,
            videos=videos,
            input_root=input_root,
            output_root=output_root,
            prompt=args.prompt,
            model_version=args.version,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            requested_direction=args.direction,
            comparison=comparison,
            predictor_factory=get_predictor,
        )
    comparison.write_summary(output_root)

    LOGGER.info(
        "Finished: %d succeeded, %d skipped, %d failed",
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

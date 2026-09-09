#!/usr/bin/env python3
"""Export one unified-dataset view into SAM3's image/COCO input format.

The exporter deliberately works on a small, explicit frame range.  It reads
the unified ``rgb.mkv``, ``mask.mkv`` and ``instances.json`` contract, then
writes JPEG frames plus a COCO-style JSON file that ``COCO_FROM_JSON`` can
load.  The default is to keep only frames containing a requested target kind;
this avoids treating an absent annotation as a reliable negative example.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


_MODEL_KIND_NAMES = {
    "hand_left": "left_hand",
    "hand_right": "right_hand",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
}


def _normalize_kind(kind: str) -> str:
    try:
        return _MODEL_KIND_NAMES[kind]
    except KeyError as error:
        raise ValueError(
            f"Unsupported kind {kind!r}; expected hand_left/hand_right"
        ) from error


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    return data if len(data) == size else b""


class _RawFrameReader:
    """Read selected video frames as fixed-size raw arrays from ffmpeg."""

    def __init__(
        self,
        video_path: Path,
        width: int,
        height: int,
        start_frame: int,
        end_frame: int,
        pixel_format: str,
        channels: int,
        dtype: str,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        select = f"select=between(n\\,{start_frame}\\,{end_frame})"
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            select,
            "-fps_mode",
            "vfr",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-an",
            "pipe:1",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._frame_shape = (height, width, channels) if channels > 1 else (height, width)
        self._dtype = np.dtype(dtype)
        self._bytes_per_frame = int(np.prod(self._frame_shape)) * self._dtype.itemsize
        self._expected_frames = end_frame - start_frame + 1
        self._read_frames = 0

    def read(self) -> np.ndarray:
        if self._process.stdout is None:
            raise RuntimeError("ffmpeg stdout is unavailable")
        raw = _read_exact(self._process.stdout, self._bytes_per_frame)
        if not raw:
            error = self._process.stderr.read().decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg returned too few frames ({self._read_frames}/"
                f"{self._expected_frames}): {error.strip()}"
            )
        self._read_frames += 1
        return np.frombuffer(raw, dtype=self._dtype).reshape(self._frame_shape).copy()

    def close(self) -> None:
        _, stderr = self._process.communicate()
        if self._process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {self._process.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )


def _probe_video(
    video_path: Path, ffprobe_bin: str = "ffprobe"
) -> Tuple[int, int, int]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_read_frames",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = result.stdout.strip().split(",")
    if len(values) != 3 or "N/A" in values:
        raise RuntimeError(f"Cannot determine dimensions/frame count for {video_path}")
    return int(values[0]), int(values[1]), int(values[2])


def _frame_ranges(record: Dict) -> Iterable[Tuple[int, int, int]]:
    for frame_map in record.get("frame_map", []):
        frames = frame_map["frames"]
        if len(frames) != 2 or frames[0] > frames[1]:
            raise ValueError(f"Invalid closed frame range: {frames!r}")
        yield int(frames[0]), int(frames[1]), int(frame_map["id"])


def _matching_instance_ids(record: Dict, frame_index: int) -> List[int]:
    return [
        instance_id
        for start, end, instance_id in _frame_ranges(record)
        if start <= frame_index <= end
    ]


def _binary_mask_to_annotation(
    binary_mask: np.ndarray,
    image_id: int,
    annotation_id: int,
    category_id: int,
    record: Dict,
) -> Dict:
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        raise ValueError(
            f"frame_map points to an absent instance: {record.get('track_id')}"
        )

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": int(binary_mask.sum()),
        "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
        "iscrowd": 0,
        "track_id": record.get("track_id"),
        "kind": record["kind"],
        "provenance": record.get("provenance"),
    }


def build_frame_annotations(
    instance_mask: np.ndarray,
    frame_index: int,
    instances: Sequence[Dict],
    category_ids: Dict[str, int],
    image_id: int,
) -> List[Dict]:
    """Build COCO annotations for selected kinds in one uint16 mask frame."""
    if instance_mask.ndim != 2 or instance_mask.dtype != np.uint16:
        raise ValueError("instance_mask must be a 2-D uint16 array")

    annotations = []
    annotation_id = 0
    used_ids = set()
    for record in instances:
        raw_kind = record.get("kind")
        if raw_kind not in _MODEL_KIND_NAMES:
            continue
        kind = _normalize_kind(raw_kind)
        if kind not in category_ids:
            continue
        for instance_id in _matching_instance_ids(record, frame_index):
            if instance_id in used_ids:
                raise ValueError(f"Duplicate instance_id {instance_id} in frame {frame_index}")
            used_ids.add(instance_id)
            annotations.append(
                _binary_mask_to_annotation(
                    instance_mask == instance_id,
                    image_id=image_id,
                    annotation_id=annotation_id,
                    category_id=category_ids[kind],
                    record={**record, "kind": kind},
                )
            )
            annotation_id += 1
    return annotations


def _load_instances(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data.get("instances"), list):
        raise ValueError(f"Missing instances list in {path}")
    return data["instances"]


def export_view(
    view_dir: Path,
    output_dir: Path,
    start_frame: int = 0,
    end_frame: int = None,
    kinds: Sequence[str] = ("hand_right",),
    include_empty: bool = False,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Path:
    """Export a unified view and return the generated annotation path."""
    view_dir = Path(view_dir)
    output_dir = Path(output_dir)
    rgb_path = view_dir / "rgb.mkv"
    mask_path = view_dir / "mask.mkv"
    instances_path = view_dir / "instances.json"
    for path in (rgb_path, mask_path, instances_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    width, height, num_frames = _probe_video(rgb_path, ffprobe_bin)
    mask_width, mask_height, mask_frames = _probe_video(mask_path, ffprobe_bin)
    if (width, height, num_frames) != (mask_width, mask_height, mask_frames):
        raise ValueError("RGB and mask videos do not have matching dimensions/frame counts")

    end_frame = num_frames - 1 if end_frame is None else end_frame
    if start_frame < 0 or end_frame < start_frame or end_frame >= num_frames:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}] for {num_frames} frames")
    kinds = tuple(_normalize_kind(kind) for kind in kinds)
    if not kinds or len(set(kinds)) != len(kinds):
        raise ValueError("kinds must contain at least one unique category")
    category_ids = {kind: index + 1 for index, kind in enumerate(kinds)}
    instances = _load_instances(instances_path)

    image_dir = output_dir / "images"
    annotation_path = output_dir / "annotations.json"
    if annotation_path.exists() and not overwrite:
        raise FileExistsError(annotation_path)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    rgb_reader = _RawFrameReader(
        rgb_path,
        width,
        height,
        start_frame,
        end_frame,
        pixel_format="rgb24",
        channels=3,
        dtype="uint8",
        ffmpeg_bin=ffmpeg_bin,
    )
    mask_reader = _RawFrameReader(
        mask_path,
        width,
        height,
        start_frame,
        end_frame,
        pixel_format="gray16le",
        channels=1,
        dtype="<u2",
        ffmpeg_bin=ffmpeg_bin,
    )
    try:
        for frame_index in range(start_frame, end_frame + 1):
            rgb = rgb_reader.read()
            instance_mask = mask_reader.read()
            frame_annotations = build_frame_annotations(
                instance_mask,
                frame_index=frame_index,
                instances=instances,
                category_ids=category_ids,
                image_id=image_id,
            )
            if not include_empty and not frame_annotations:
                continue

            file_name = f"images/{frame_index:08d}.jpg"
            Image.fromarray(rgb, mode="RGB").save(
                output_dir / file_name,
                format="JPEG",
                quality=95,
            )
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "frame_index": frame_index,
                }
            )
            for annotation in frame_annotations:
                annotation["id"] = annotation_id
                annotation["image_id"] = image_id
                annotations.append(annotation)
                annotation_id += 1
            image_id += 1
    finally:
        rgb_reader.close()
        mask_reader.close()

    result = {
        "info": {
            "description": "SAM3 export from the unified hand-object dataset",
            "source_view": str(view_dir),
            "frame_range": [start_frame, end_frame],
            "include_empty": include_empty,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": kind, "supercategory": "hand"}
            for kind, category_id in category_ids.items()
        ],
    }
    with annotation_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return annotation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--kinds", default="hand_right")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = export_view(
        view_dir=args.view_dir,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        kinds=tuple(args.kinds.split(",")),
        include_empty=args.include_empty,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()

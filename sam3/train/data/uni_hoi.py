"""Load uni-hoi sequences as a single-kind COCO training API."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from pycocotools import mask as mask_util

from sam3.train.data.coco_json_loaders import COCO_FROM_JSON

HAND_KINDS = frozenset({"hand_left", "hand_right"})
MaskDecoder = Callable[[Path, int, int], Iterable[np.ndarray]]


def expand_dataset_root(root: str) -> Path:
    path = Path(root).expanduser()
    if not path.is_dir():
        raise ValueError(f"uni-hoi dataset root does not exist: {path}")
    return path


def instance_ids_on_frame(
    instances: Sequence[dict], frame: int, kind: str
) -> List[int]:
    ids = []
    for instance in instances:
        if instance.get("kind") != kind:
            continue
        for entry in instance.get("frame_map") or []:
            start, end = entry["frames"]
            if start <= frame <= end:
                ids.append(int(entry["id"]))
    return ids


def split_for_sequence(split_index: dict, sequence: dict) -> str:
    source = sequence["source"]
    source_split = split_index.get(source)
    if not isinstance(source_split, dict) or "by_subject" not in source_split:
        raise ValueError(f"metadata/split.json has no by_subject map for {source}")
    subject = sequence.get("subject")
    assigned = source_split["by_subject"].get(subject)
    if assigned is None:
        raise ValueError(f"No split for {source} subject {subject}")
    return assigned


def view_size(sequence: dict, view_id: str, rgb_path: Path) -> tuple[int, int]:
    intrinsics = (sequence.get("intrinsics") or {}).get(view_id) or {}
    width, height = intrinsics.get("w"), intrinsics.get("h")
    if width and height:
        return int(width), int(height)
    import cv2

    capture = cv2.VideoCapture(str(rgb_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot determine resolution for {rgb_path}")
    return width, height


def iter_mask_frames(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to decode uni-hoi mask.mkv")
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_size = width * height * 2
    index = 0
    try:
        while True:
            buffer = process.stdout.read(frame_size)
            if not buffer:
                break
            if len(buffer) != frame_size:
                raise ValueError(
                    f"{path} truncated at frame {index}: {len(buffer)} bytes"
                )
            yield np.frombuffer(buffer, dtype="<u2").reshape(height, width)
            index += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr = (
            process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        )
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {stderr[-400:]}")


def _encode_instance(mask: np.ndarray, instance_id: int) -> Optional[dict]:
    binary = np.asfortranarray(mask == instance_id, dtype=np.uint8)
    if not binary.any():
        return None
    rle = mask_util.encode(binary)
    bbox = mask_util.toBbox(rle).tolist()
    return {
        "segmentation": rle,
        "bbox": bbox,
        "area": float(mask_util.area(rle)),
        "iscrowd": 0,
    }


def build_uni_hoi_raw_data(
    root: Path,
    split_name: str,
    kind: str,
    category_id: int,
    decode_mask: MaskDecoder = iter_mask_frames,
) -> List[Dict]:
    if kind not in HAND_KINDS:
        raise ValueError(f"kind must be hand_left or hand_right, got {kind}")
    split_path = root / "metadata" / "split.json"
    split_index = json.loads(split_path.read_text(encoding="utf-8"))
    raw_data: List[Dict] = []
    positives = 0
    for sequence_file in sorted(root.glob("sequences/*/*/sequence.json")):
        if sequence_file.parent.name.endswith(".tmp"):
            continue
        sequence = json.loads(sequence_file.read_text(encoding="utf-8"))
        if split_for_sequence(split_index, sequence) != split_name:
            continue
        source = sequence["source"]
        seq_id = sequence["seq_id"]
        num_frames = int(sequence["num_frames"])
        print(f"Indexing {source}/{seq_id} ({split_name}, {kind})", flush=True)
        for view_id in sequence["views"]:
            view_dir = sequence_file.parent / view_id
            rgb_path = view_dir / "rgb.mkv"
            mask_path = view_dir / "mask.mkv"
            instances_path = view_dir / "instances.json"
            if not rgb_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing rgb.mkv or mask.mkv in {view_dir}")
            instances = json.loads(instances_path.read_text(encoding="utf-8"))[
                "instances"
            ]
            width, height = view_size(sequence, view_id, rgb_path)
            relative = view_dir.relative_to(root).as_posix()
            decoded = 0
            for frame, mask in enumerate(decode_mask(mask_path, width, height)):
                if mask.shape != (height, width):
                    raise ValueError(
                        f"{mask_path} frame {frame} has shape {mask.shape}, "
                        f"expected {(height, width)}"
                    )
                annotations = []
                for instance_id in instance_ids_on_frame(instances, frame, kind):
                    encoded = _encode_instance(mask, instance_id)
                    if encoded is None:
                        continue
                    encoded.update(
                        {
                            "id": len(annotations) + 1,
                            "image_id": len(raw_data) + 1,
                            "category_id": category_id,
                        }
                    )
                    annotations.append(encoded)
                if annotations:
                    positives += 1
                raw_data.append(
                    {
                        "image": {
                            "id": len(raw_data) + 1,
                            "file_name": f"{relative}/rgb.mkv@{frame}",
                            "width": width,
                            "height": height,
                        },
                        "annotations": annotations,
                    }
                )
                decoded += 1
            if decoded != num_frames:
                raise ValueError(
                    f"{mask_path} has {decoded} frames, sequence.json has {num_frames}"
                )
    if not raw_data:
        raise ValueError(f"No uni-hoi frames for split={split_name} kind={kind}")
    print(
        f"Loaded {len(raw_data)} {split_name} frames "
        f"({positives} with {kind}, {len(raw_data) - positives} negatives)",
        flush=True,
    )
    return raw_data


class UniHoiTargetCOCO(COCO_FROM_JSON):
    """Single-kind uni-hoi loader that keeps frames without the selected hand."""

    def __init__(
        self,
        annotation_file: str,
        category_id: int,
        target_id: str,
        dataset_root: str,
        split_name: str,
        kind: str,
        decode_mask: MaskDecoder = iter_mask_frames,
    ) -> None:
        _ = annotation_file
        self._raw_data = build_uni_hoi_raw_data(
            expand_dataset_root(dataset_root),
            split_name,
            kind,
            category_id,
            decode_mask=decode_mask,
        )
        self._cat_idx_to_text = {category_id: target_id}
        self.prompts = None
        self.include_negatives = True
        self._sorted_cat_ids = [category_id]
        self.category_chunk_size = 1
        self.category_chunks = [[category_id]]

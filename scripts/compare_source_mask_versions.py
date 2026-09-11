"""Read-only paired mask-video audit: raw IDs, nonzero masks and exact PTS.

Decodes only the two explicit inputs, one grayscale frame from each at a time.
No model, GPU, dataset search, mask rewriting, or preferred-version selection.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import tempfile
import time

import numpy as np


FORMAT = "sam3-paired-source-mask-version-comparison-v1"


def file_state(path):
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Not a regular source file: {path}")
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns, "device": info.st_dev, "inode": info.st_ino}


def source_receipt(path):
    path = Path(path).absolute()
    before = file_state(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = file_state(path)
    if before != after:
        raise RuntimeError(f"Source changed during hashing: {path}")
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "stat": after, "sha256": digest.hexdigest()}


def probe(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "stream=index,codec_name,width,height,pix_fmt,time_base,r_frame_rate:frame=pts,best_effort_timestamp,pkt_duration",
               "-of", "json", str(path)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True, timeout=120)
    if result.stderr.strip():
        raise RuntimeError(f"ffprobe reported decode errors: {result.stderr.decode(errors='replace')}")
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Exactly one selected video stream required")
    stream = streams[0]
    if stream.get("pix_fmt") != "gray":
        raise ValueError("This exact raw-ID comparison requires native 8-bit grayscale videos")
    width, height = int(stream["width"]), int(stream["height"])
    if not (0 < width <= 4096 and 0 < height <= 4096):
        raise ValueError("Unsafe or invalid source dimensions")
    frames = metadata.get("frames", [])
    if not frames or len(frames) > 100000:
        raise ValueError("Invalid or excessive decoded frame count")
    timestamps = []
    for row in frames:
        pts = row.get("pts")
        if not isinstance(pts, int) or row.get("best_effort_timestamp", pts) != pts:
            raise ValueError("Missing or ambiguous actual frame PTS")
        timestamps.append(pts)
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Frame PTS are not strictly increasing")
    return {"stream": stream, "frame_count": len(frames), "pts": timestamps,
            "frame_packet_duration": [row.get("pkt_duration") for row in frames], "command": command}


def read_exact(pipe, size, deadline):
    chunks = []
    remaining = size
    while remaining:
        wait = deadline - time.monotonic()
        if wait <= 0:
            raise TimeoutError("Mask decoder exceeded its finite deadline")
        readable, _, _ = select.select([pipe], [], [], min(10., wait))
        if not readable:
            continue
        data = os.read(pipe.fileno(), remaining)
        if not data:
            if not chunks:
                return None
            raise ValueError("Truncated raw mask frame")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


class MaskDecoder:
    def __init__(self, path, width, height, timeout=600):
        self.width, self.height = width, height
        self.deadline = time.monotonic() + timeout
        self.stderr = tempfile.TemporaryFile()
        self.command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-hwaccel", "none",
                        "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-threads", "1",
                        "-filter_threads", "1", "-vsync", "0", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)

    def frame(self):
        raw = read_exact(self.process.stdout, self.width * self.height, self.deadline)
        return None if raw is None else np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width)

    def finish(self):
        status = self.process.wait(timeout=max(.1, self.deadline - time.monotonic()))
        self.stderr.seek(0)
        errors = self.stderr.read().decode(errors="replace")
        if status or errors.strip():
            raise RuntimeError(f"Mask decode failed ({status}): {errors}")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.stderr.close()


def compare_frame(a, b):
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Same-shape two-dimensional uint8 mask arrays required")
    aa, bb = a > 0, b > 0
    raw_changed = a != b
    changed = aa != bb
    intersection = int(np.count_nonzero(aa & bb))
    count_a, count_b = int(np.count_nonzero(aa)), int(np.count_nonzero(bb))
    union = count_a + count_b - intersection
    ys, xs = np.nonzero(changed)
    return {"raw_different_pixels": int(np.count_nonzero(raw_changed)),
            "binary_different_pixels": int(np.count_nonzero(changed)),
            "a_foreground_pixels": count_a, "b_foreground_pixels": count_b,
            "a_only_pixels": count_a - intersection, "b_only_pixels": count_b - intersection,
            "intersection_pixels": intersection, "union_pixels": union,
            "binary_dice": 2 * intersection / (count_a + count_b) if count_a + count_b else 1.,
            "binary_iou": intersection / union if union else 1.,
            "binary_difference_bbox_xyxy_exclusive": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None,
            "a_value_counts": {str(value): int(count) for value, count in zip(*np.unique(a, return_counts=True))},
            "b_value_counts": {str(value): int(count) for value, count in zip(*np.unique(b, return_counts=True))}}


def classification(raw_different, binary_different, source_hashes_equal):
    if binary_different:
        return "visible_foreground_masks_differ; not only container/reencoding or nonzero instance-ID changes"
    if raw_different:
        return "same_nonzero_foreground_masks_but_raw_label_values_differ"
    return "files_identical_and_decoded_frames_identical" if source_hashes_equal else "files_differ_but_all_decoded_raw_mask_frames_identical"


def compare_sources(path_a, path_b):
    paths = [Path(path_a).absolute(), Path(path_b).absolute()]
    before = [source_receipt(path) for path in paths]
    metadata = [probe(path) for path in paths]
    first, second = [row["stream"] for row in metadata]
    for field in ("width", "height", "time_base"):
        if first[field] != second[field]:
            raise ValueError(f"Cannot make frame-paired comparison: stream {field} differs")
    if metadata[0]["pts"] != metadata[1]["pts"]:
        raise ValueError("Actual frame count/PTS differ; no automatic realignment or chosen version")
    width, height = first["width"], first["height"]
    pairs = np.zeros(65536, dtype=np.int64)
    rows = []
    commands = []
    with ExitStack() as cleanup:
        decoders = []
        for path in paths:
            decoder = MaskDecoder(path, width, height)
            cleanup.callback(decoder.close)
            decoders.append(decoder)
            commands.append(decoder.command)
        for frame_index, pts in enumerate(metadata[0]["pts"]):
            arrays = [decoder.frame() for decoder in decoders]
            if any(array is None for array in arrays):
                raise ValueError("Raw decoder ended before independently probed frame count")
            frame = compare_frame(*arrays)
            frame.update(frame_index=frame_index, pts=pts)
            rows.append(frame)
            encoded = arrays[0].astype(np.uint16) * 256 + arrays[1]
            pairs += np.bincount(encoded.ravel(), minlength=65536)
        if any(decoder.frame() is not None for decoder in decoders):
            raise ValueError("Raw decoder produced more frames than independently probed")
        for decoder in decoders:
            decoder.finish()
    after = [source_receipt(path) for path in paths]
    if before != after:
        raise RuntimeError("At least one source changed during decode/probe; refuse a stable-version conclusion")
    raw_changed = [row["frame_index"] for row in rows if row["raw_different_pixels"]]
    binary_changed = [row["frame_index"] for row in rows if row["binary_different_pixels"]]
    sums = {key: sum(row[key] for row in rows) for key in (
        "raw_different_pixels", "binary_different_pixels", "a_foreground_pixels", "b_foreground_pixels",
        "a_only_pixels", "b_only_pixels", "intersection_pixels", "union_pixels")}
    foreground_sum = sums["a_foreground_pixels"] + sums["b_foreground_pixels"]
    pair_counts = [{"a_value": int(code // 256), "b_value": int(code % 256), "pixels": int(pairs[code])}
                   for code in np.flatnonzero(pairs)]
    totals = {**sums, "total_pixels": len(rows) * width * height,
              "raw_changed_frames": len(raw_changed), "binary_changed_frames": len(binary_changed),
              "raw_changed_frame_indices": raw_changed, "binary_changed_frame_indices": binary_changed,
              "binary_micro_dice": 2 * sums["intersection_pixels"] / foreground_sum if foreground_sum else 1.,
              "binary_micro_iou": sums["intersection_pixels"] / sums["union_pixels"] if sums["union_pixels"] else 1.,
              "binary_macro_dice": float(np.mean([row["binary_dice"] for row in rows])),
              "binary_macro_iou": float(np.mean([row["binary_iou"] for row in rows]))}
    return {"format": FORMAT, "status": "completed", "created_at": datetime.now(timezone.utc).isoformat(),
            "source_a": before[0], "source_b": before[1], "source_a_after": after[0], "source_b_after": after[1],
            "sources_unchanged": True, "metadata": metadata, "dimensions_match": True, "exact_pts_match": True,
            "decoded_frame_count": len(rows), "decoder_commands": commands,
            "conclusion": classification(bool(raw_changed), bool(binary_changed), before[0]["sha256"] == before[1]["sha256"]),
            "interpretation": {"nonzero_rule": "mask > 0 independently in each source",
                               "both_empty_dice_and_iou": 1., "preferred_version_selected": False,
                               "source_modified_or_masks_merged": False, "gpu_or_model_used": False,
                               "limits": "A mismatch establishes a version difference, not which mask is accurate; no RGB/human accuracy adjudication."},
            "totals": totals, "all_pixel_value_pair_counts": pair_counts, "frames": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-a", type=Path, required=True)
    parser.add_argument("--source-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Only a new report path is allowed: {args.output}")
    report = compare_sources(args.source_a, args.source_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    compact = {key: report[key] for key in ("status", "conclusion", "decoded_frame_count", "sources_unchanged")}
    compact["totals"] = {key: value for key, value in report["totals"].items() if not key.endswith("frame_indices")}
    compact["complete_per_frame_report"] = str(args.output)
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only inventory and bounded modality audit; never unpickle or convert data.

All video frame metadata are decoded with ffprobe. Every mask pixel is inspected;
RGB/depth pixel values are sampled at five fixed frame indices. NPZ inspection
reads NPY headers only. SQLite uses read-only immutable connections.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from urllib.parse import quote
import zipfile

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def inventory(root):
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            info = path.stat()
            rows[str(path.relative_to(root))] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
                "inode": info.st_ino, "symlink": path.is_symlink(), "suffix": path.suffix.lower()}
    return {"checked_at_utc": now(), "files": rows, "file_count": len(rows),
            "bytes": sum(row["bytes"] for row in rows.values()),
            "extensions": dict(Counter(row["suffix"] for row in rows.values()))}


def differences(left, right):
    a, b = left["files"], right["files"]
    return {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
            "changed": [name for name in sorted(set(a) & set(b)) if a[name] != b[name]]}


def video_metadata(path):
    command = ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
               "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate,avg_frame_rate:format=duration,size:frame=best_effort_timestamp_time",
               "-of", "json", str(path)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if process.returncode or process.stderr.strip():
        raise ValueError(f"ffprobe decode error: {process.stderr[-2000:]}")
    data = json.loads(process.stdout)
    if len(data.get("streams", [])) != 1:
        raise ValueError("Expected one selected video stream")
    times = [frame.get("best_effort_timestamp_time") for frame in data.get("frames", [])]
    if not times or any(value is None for value in times):
        raise ValueError("Missing decoded frame timestamps")
    numeric = np.asarray(times, dtype=np.float64)
    return {**data["streams"][0], "format": data.get("format", {}), "decoded_frames": len(times),
            "first_pts": times[0], "last_pts": times[-1],
            "strictly_increasing_pts": bool(np.all(np.diff(numeric) > 0)),
            "pts_sha256": hashlib.sha256(json.dumps(times).encode()).hexdigest()}, numeric


def full_mask_pixels(path, metadata):
    if metadata["pix_fmt"] != "gray":
        raise ValueError("Expected native 8-bit gray masks; refusing a lossy pixel-format conversion")
    h, w = int(metadata["height"]), int(metadata["width"])
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0",
               "-f", "rawvideo", "-pix_fmt", "gray", "-threads", "1", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    histogram = np.zeros(256, dtype=np.int64)
    empty_indices, areas, frames = [], [], 0
    try:
        while True:
            raw = process.stdout.read(h * w)
            if not raw:
                break
            if len(raw) != h * w:
                raise ValueError("Truncated raw mask frame")
            pixels = np.frombuffer(raw, dtype=np.uint8)
            counts = np.bincount(pixels, minlength=256)
            histogram += counts
            area = int(h * w - counts[0])
            if not area:
                empty_indices.append(frames)
            areas.append(area)
            frames += 1
        stderr = process.stderr.read().decode(errors="replace")
        if process.wait(timeout=30) or stderr.strip():
            raise ValueError(f"ffmpeg mask decode error: {stderr[-2000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    if frames != metadata["decoded_frames"]:
        raise ValueError("Mask raw decode frame count differs from metadata decode")
    return {"scope": "all_decoded_mask_pixels", "frames": frames,
            "value_histogram": {str(index): int(value) for index, value in enumerate(histogram) if value},
            "empty_frames": len(empty_indices), "empty_frame_indices": empty_indices,
            "nonempty_frames": frames - len(empty_indices),
            "nonzero_area_min": min(areas), "nonzero_area_max": max(areas),
            "nonzero_area_mean": float(np.mean(areas)), "area_per_frame": areas}


def sample_pixels(path, metadata):
    count = metadata["decoded_frames"]
    indices = sorted({0, (count - 1) // 4, (count - 1) // 2, 3 * (count - 1) // 4, count - 1})
    is_depth = path.name == "depth.mkv"
    if is_depth and metadata["pix_fmt"] != "gray16le":
        raise ValueError("Depth is not native gray16le; refusing unverified value conversion")
    pix_fmt = "gray16le" if is_depth else "rgb24"
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0",
               "-vf", f"select={expression}", "-fps_mode", "passthrough", "-f", "rawvideo",
               "-pix_fmt", pix_fmt, "-threads", "1", "pipe:1"]
    process = subprocess.run(command, capture_output=True, timeout=300)
    if process.returncode or process.stderr.strip():
        raise ValueError(f"Sample decode failed: {process.stderr.decode(errors='replace')[-2000:]}")
    h, w = int(metadata["height"]), int(metadata["width"])
    dtype, channels = (np.dtype("<u2"), 1) if is_depth else (np.dtype("u1"), 3)
    expected = len(indices) * h * w * channels * dtype.itemsize
    if len(process.stdout) != expected:
        raise ValueError("Sample pixel byte count differs from requested frame count")
    decoded = np.frombuffer(process.stdout, dtype=dtype).reshape(len(indices), h, w, channels)
    rows = []
    for index, image in zip(indices, decoded):
        row = {"frame_index": index, "min": int(image.min()), "max": int(image.max()),
               "pixel_bytes_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        if is_depth:
            positive = image[image > 0]
            row.update(zero_pixel_fraction=float(np.mean(image == 0)),
                       positive_value_percentiles=np.percentile(positive, [1, 50, 99]).tolist() if len(positive) else None)
        rows.append(row)
    return {"scope": "five_fixed_frame_pixel_samples_not_all_rgb_or_depth_pixels", "output_pix_fmt": pix_fmt,
            "rgb_is_decoder_rgb_conversion": not is_depth, "samples": rows}


def npz_headers(path):
    rows = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            row = {"member": member.filename, "bytes_uncompressed": member.file_size,
                   "bytes_compressed": member.compress_size, "crc32_declared": member.CRC}
            if member.filename.endswith(".npy"):
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ValueError(f"Unsupported NPY header version {version}")
                    row.update(shape=list(shape), dtype=str(dtype), fortran_order=fortran, has_object_dtype=dtype.hasobject)
            rows.append(row)
    return {"scope": "zip_directory_and_npy_headers_only_no_pickle_or_array_body", "members": rows,
            "full_crc_or_array_finiteness_verified": False}


def sqlite_summary(path):
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        schema = connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        tables = []
        for name, sql in schema:
            escaped = name.replace('"', '""')
            count = connection.execute(f'SELECT COUNT(*) FROM "{escaped}"').fetchone()[0]
            tables.append({"name": name, "rows": count, "create_sql": sql})
        topics = []
        if any(row[0] == "topics" for row in schema):
            cursor = connection.execute("SELECT * FROM topics")
            names = [description[0] for description in cursor.description]
            topics = [dict(zip(names, row)) for row in cursor.fetchall()]
    return {"scope": "readonly_immutable_schema_row_counts_topics_no_payload_deserialization", "tables": tables, "topics": topics}


def supplementary_checks(root, structural):
    """Small safe metadata follow-up; deliberately excludes large MANO arrays."""
    before = inventory(root)
    rows = {}
    for name, sequence in structural["sequences"].items():
        row = {"image_topic_counts": [], "small_npz_fields": {}}
        rows[name] = row
        for relative in sequence["sqlite"]:
            path = root / name / relative
            uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro&immutable=1"
            with sqlite3.connect(uri, uri=True) as connection:
                cursor = connection.execute("SELECT t.name,COUNT(m.id),MIN(m.timestamp),MAX(m.timestamp) "
                    "FROM topics t LEFT JOIN messages m ON t.id=m.topic_id "
                    "WHERE t.type='sensor_msgs/msg/Image' GROUP BY t.id ORDER BY t.id")
                row["image_topic_counts"] += [{"topic": topic, "messages": count, "first_bag_timestamp": first,
                                                "last_bag_timestamp": last} for topic, count, first, last in cursor]
        for relative in sequence["npz"]:
            selected = {}
            with zipfile.ZipFile(root / name / relative) as archive:
                for key in ("hand", "mask_source", "instance_label", "pose_format", "fps", "width", "height", "total_frames", "frame_indices", "has_hand"):
                    filename = key + ".npy"
                    if filename not in archive.namelist():
                        continue
                    info = archive.getinfo(filename)
                    if info.file_size > 1_000_000:
                        raise ValueError("Small metadata field exceeded read budget")
                    with archive.open(filename) as handle:
                        value = np.load(handle, allow_pickle=False)
                    if value.dtype.hasobject:
                        raise ValueError("Object dtype is forbidden")
                    if value.ndim == 0:
                        selected[key] = value.item()
                    elif key == "frame_indices":
                        selected[key] = {"count": len(value), "first": int(value[0]) if len(value) else None,
                            "last": int(value[-1]) if len(value) else None,
                            "equals_complete_zero_based_index": bool(np.array_equal(value, np.arange(len(value))))}
                    elif key == "has_hand":
                        selected[key] = {"count": len(value), "true": int(np.count_nonzero(value)), "false": int(len(value) - np.count_nonzero(value))}
            row["small_npz_fields"][relative] = selected
    anomalies = []
    for name, sequence in structural["sequences"].items():
        for relative, metadata in sequence["videos"].items():
            if not any(int(value) > 1 for value in metadata.get("pixels", {}).get("value_histogram", {})):
                continue
            path = root / name / relative
            process = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray", "-threads", "1", "pipe:1"],
                capture_output=True, timeout=180)
            if process.returncode or process.stderr.strip():
                raise ValueError("Anomaly localization decode failed")
            values = np.frombuffer(process.stdout, np.uint8).reshape(metadata["decoded_frames"], metadata["height"], metadata["width"])
            for frame in np.flatnonzero((values > 1).any(axis=(1, 2))):
                y, x = np.where(values[frame] > 1)
                found, counts = np.unique(values[frame][y, x], return_counts=True)
                anomalies.append({"file": str(path), "frame_index": int(frame),
                    "values_above_one": {str(v): int(c) for v, c in zip(found, counts)},
                    "bbox_xyxy_exclusive": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]})
    after = inventory(root)
    return {"format": "realsense-readonly-supplement-v1", "scope": "small non-object NPZ metadata, ROS Image topic row counts, localization of observed mask values>1; no large MANO arrays or ROS payload decode",
            "started_at_utc": before["checked_at_utc"], "finished_at_utc": after["checked_at_utc"],
            "sequences": rows, "mask_values_above_one_locations": anomalies,
            "stat_changes": differences(before, after), "original_audit_stat_changes": differences(structural["inventory_after"], after)}


def audit(root):
    root = root.resolve()
    before = inventory(root)
    if any(row["symlink"] for row in before["files"].values()):
        raise ValueError("Symlinks require explicit scope review before modality reads")
    result = {"format": "realsense-hand-object-readonly-audit-v1", "root": str(root), "started_at_utc": now(),
              "scope": "No conversion, training, source writes, unknown script execution, NPZ pickle loading, or ROS message deserialization",
              "inventory_before": before, "sequences": {}, "errors": []}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        name = directory.name
        row = {"files": [path for path in before["files"] if path.startswith(name + "/")],
               "videos": {}, "npz": {}, "sqlite": {}, "pairing": {}}
        result["sequences"][name] = row
        meta_path = directory / "meta.json"
        if meta_path.is_file():
            raw = meta_path.read_bytes()
            row["meta"] = json.loads(raw)
            row["meta_sha256"] = hashlib.sha256(raw).hexdigest()
        times = {}
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(directory))
            try:
                if path.suffix.lower() in (".mkv", ".mp4"):
                    row["videos"][relative], times[relative] = video_metadata(path)
                elif path.suffix.lower() == ".npz":
                    row["npz"][relative] = npz_headers(path)
                elif path.suffix.lower() == ".db3":
                    row["sqlite"][relative] = sqlite_summary(path)
            except Exception as error:
                result["errors"].append({"file": str(path), "phase": "metadata", "error": f"{type(error).__name__}: {error}"})
        color = row["videos"].get("color.mp4")
        if color:
            for relative, metadata in row["videos"].items():
                if relative.startswith("MANO_wilor/") or relative == "depth_preview.mp4":
                    continue
                same_count = metadata["decoded_frames"] == color["decoded_frames"]
                row["pairing"][relative] = {"same_decoded_count_as_color": same_count,
                    "same_shape_as_color": (metadata["height"], metadata["width"]) == (color["height"], color["width"]),
                    "max_abs_pts_difference_seconds": float(np.max(np.abs(times[relative] - times["color.mp4"]))) if same_count else None}
            row["meta_frames_equal_color"] = row.get("meta", {}).get("frames") == color["decoded_frames"]
        row["mask_roles_from_filenames_only"] = [Path(path).stem for path in row["videos"] if path.startswith("masks_sam3/")]
        print(json.dumps({"phase": "metadata", "sequence": name, "videos": len(row["videos"]), "errors": len(result["errors"])}), flush=True)
    result["inventory_midpoint"] = inventory(root)
    for name, row in result["sequences"].items():
        for relative, metadata in row["videos"].items():
            path = root / name / relative
            try:
                if relative.startswith("masks_sam3/"):
                    metadata["pixels"] = full_mask_pixels(path, metadata)
                elif relative in ("color.mp4", "depth.mkv"):
                    metadata["pixels"] = sample_pixels(path, metadata)
            except Exception as error:
                result["errors"].append({"file": str(path), "phase": "pixels", "error": f"{type(error).__name__}: {error}"})
        print(json.dumps({"phase": "pixels", "sequence": name, "errors": len(result["errors"])}), flush=True)
    after = inventory(root)
    result.update(inventory_after=after, finished_at_utc=now())
    result["stat_changes_before_midpoint"] = differences(before, result["inventory_midpoint"])
    result["stat_changes_midpoint_after"] = differences(result["inventory_midpoint"], after)
    result["stable_over_observed_interval"] = not any(values for key in ("stat_changes_before_midpoint", "stat_changes_midpoint_after") for values in result[key].values())
    result["stability_limitation"] = "Three complete file stat snapshots across this audit; no change only means no observed copy/write in this interval, not a promise copying is finished"
    result["status"] = "completed" if not result["errors"] and result["stable_over_observed_interval"] else "completed_with_findings"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Output must be new and outside source dataset")
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "errors": len(result["errors"])}))


if __name__ == "__main__":
    main()

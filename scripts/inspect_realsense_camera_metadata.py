#!/usr/bin/env python3
"""Bounded, read-only extraction of camera strings from ten specified DB3 files.

Never decode image payloads, run recording scripts, or read depth video files.
Optional existing zstandard support is used only with declared size/window caps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct

MAX_MESSAGE_BYTES = 65536
RECORDINGS = ("basket", "black_pen", "blue_pen", "bottle", "bowl", "cup", "left_hand", "milk", "red_pen", "right_hand")
ROOT = Path("/data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh")


def decode_cdr_string(payload: bytes):
    """Accept only bounded plain or single-frame Zstd CDR1 string messages."""
    if not isinstance(payload, bytes) or not 9 <= len(payload) <= MAX_MESSAGE_BYTES:
        raise ValueError("Missing/oversized string payload")
    compression = "none"
    if payload.startswith(bytes.fromhex("28b52ffd")):
        import zstandard  # Existing environment only; no installer or fallback subprocess.

        parameters = zstandard.get_frame_parameters(payload)
        if not 9 <= parameters.content_size <= MAX_MESSAGE_BYTES or parameters.window_size > MAX_MESSAGE_BYTES:
            raise ValueError("Unknown/oversized Zstd content or window size")
        payload = zstandard.ZstdDecompressor(max_window_size=MAX_MESSAGE_BYTES).decompress(
            payload, max_output_size=MAX_MESSAGE_BYTES, allow_extra_data=False)
        if len(payload) != parameters.content_size:
            raise ValueError("Zstd declared/actual sizes differ")
        compression = "zstd"
    if payload[:4] not in (b"\x00\x00\x00\x00", b"\x00\x01\x00\x00"):
        raise ValueError("Unsupported CDR representation/options")
    endian = "<" if payload[1] == 1 else ">"
    length = struct.unpack(endian + "I", payload[4:8])[0]
    if length < 1 or 8 + length != len(payload) or payload[-1:] != b"\x00" or b"\x00" in payload[8:-1]:
        raise ValueError("Invalid exact CDR string length/NUL termination")
    return payload[8:-1].decode("utf-8", errors="strict"), compression


def _numbers(text, length):
    result = [float(item) for item in text.split(",")]
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError("Invalid finite numeric field")
    return result


def interpret_text(text, topic):
    """Parse only the observed simple schema; retain raw text independently."""
    if topic.endswith("/Depth_Units/value"):
        scale = _numbers(text, 1)[0]
        if scale <= 0:
            raise ValueError("Nonpositive depth units")
        return {"kind": "depth_units_text", "value": scale,
                "unit_interpretation": "sensor option numeric value; compare recording meta before applying"}
    parts = text.split(";")
    if any(item.count("=") != 1 for item in parts):
        raise ValueError("Not the observed semicolon key=value camera schema")
    pairs = [item.split("=", 1) for item in parts]
    if len(dict(pairs)) != len(pairs):
        raise ValueError("Duplicate camera metadata keys")
    fields = dict(pairs)
    if topic.endswith("/camera_info"):
        if set(fields) != {"width", "height", "fx", "fy", "ppx", "ppy", "model", "coeffs"}:
            raise ValueError("Unknown camera-info schema")
        width, height = int(fields["width"]), int(fields["height"])
        fx, fy, cx, cy = (_numbers(fields[name], 1)[0] for name in ("fx", "fy", "ppx", "ppy"))
        if width <= 0 or height <= 0 or min(fx, fy) <= 0:
            raise ValueError("Invalid dimensions/focal lengths")
        return {"kind": "native_camera_intrinsics_text", "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy, "K_conventional_from_named_fields": [[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
                "distortion_model_text": fields["model"], "distortion_coefficients": _numbers(fields["coeffs"], 5),
                "scope": "native stream metadata, not a validated aligned-depth video calibration"}
    if "/tf/" in topic:
        if set(fields) != {"rotation", "translation"}:
            raise ValueError("Unknown transform schema")
        return {"kind": "native_tf_text", "rotation_flat_as_stored": _numbers(fields["rotation"], 9),
                "translation_as_stored": _numbers(fields["translation"], 3),
                "matrix_layout": "unverified", "transform_direction": "unverified", "translation_units": "unverified"}
    raise ValueError("Unapproved metadata topic")


def inspect_database(path):
    path = path.resolve(strict=True)
    result = {"path": str(path), "file_bytes": path.stat().st_size, "metadata": [], "image_timestamps": []}
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        # Read only a small topic inventory; images never appear in payload queries.
        topics = connection.execute("SELECT id,name,type,serialization_format FROM topics ORDER BY id LIMIT 257").fetchall()
        if len(topics) > 256:
            raise ValueError("Topic inventory exceeds this bounded audit")
        selected = [row for row in topics if row[1].endswith(("/camera_info", "/Depth_Units/value")) or "/tf/" in row[1]]
        if len(selected) > 12:
            raise ValueError("Too many metadata topics for bounded audit")
        for tid, name, kind, serialization in selected:
            item = {"topic": name, "type": kind, "serialization_format": serialization}
            # First and latest only: do not claim all intermediate metadata agrees.
            samples = []
            for direction in ("ASC", "DESC"):
                row = connection.execute(
                    "SELECT id,timestamp,length(data),CASE WHEN length(data)<=? THEN data ELSE NULL END "
                    f"FROM messages WHERE topic_id=? ORDER BY id {direction} LIMIT 1", (MAX_MESSAGE_BYTES, tid)).fetchone()
                if row is None or any(sample["message_id"] == row[0] for sample in samples):
                    continue
                message_id, timestamp, byte_count, payload = row
                sample = {"message_id": message_id, "timestamp_stored": timestamp, "payload_bytes": byte_count}
                try:
                    if kind != "std_msgs/msg/String" or serialization != "cdr":
                        raise ValueError("Unsupported declared message type/serialization")
                    text, compression = decode_cdr_string(payload)
                    sample.update(payload_sha256=hashlib.sha256(payload).hexdigest(), compression=compression, text=text,
                                  cdr_validated=True, parsed=interpret_text(text, name))
                except Exception as error:
                    sample.update(status="unparsed", reason=f"{type(error).__name__}: {error}")
                samples.append(sample)
            item["first_and_latest_samples"] = samples
            result["metadata"].append(item)
        for tid, name, kind, serialization in topics:
            if not name.endswith("/image/data") or kind != "sensor_msgs/msg/Image":
                continue
            timestamps = {}
            for direction, key in (("ASC", "first_two"), ("DESC", "last_two_descending")):
                timestamps[key] = [list(row) for row in connection.execute(
                    f"SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY id {direction} LIMIT 2", (tid,))]
            result["image_timestamps"].append({"topic": name, "message_id_timestamp_pairs": timestamps,
                                                "timestamp_unit": "not established by this extraction", "image_payload_read": False})
    finally:
        connection.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Do not write audit output inside source data")
    result = {"format": "realsense-bounded-camera-metadata-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "root": str(root), "source_read_only": True, "maximum_message_bytes": MAX_MESSAGE_BYTES,
              "scope": "Ten named DB3 files; first/latest small camera strings and first/last two image timestamps only",
              "depth_video_read": False, "rgb_depth_reprojection_verified": False, "recordings": {}}
    for name in RECORDINGS:
        try:
            result["recordings"][name] = inspect_database(root / name / f"{name}.db3")
        except Exception as error:
            result["recordings"][name] = {"status": "unparsed", "reason": f"{type(error).__name__}: {error}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"output={output} recordings={len(result['recordings'])} depth_video_read=False reprojection_verified=False")


if __name__ == "__main__":
    main()

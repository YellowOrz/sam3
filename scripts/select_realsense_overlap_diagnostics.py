#!/usr/bin/env python3
"""Read-only, targeted independent-layer overlap selection; no image edits.

This deliberately selects extrema for diagnosis, not a random quality sample.
Frame indices are zero-based positions in the currently exported mask videos.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


PAIRS = {"left_right": ("left_hand", "right_hand"),
         "left_object": ("left_hand", "object"),
         "right_object": ("right_hand", "object")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_record(path):
    value = path.stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}


def overlap_row(first, second, frame):
    first, second = first > 0, second > 0
    a, b = int(first.sum()), int(second.sum())
    intersection = first & second
    count = int(intersection.sum())
    y, x = np.nonzero(intersection)
    return {"frame_index_zero_based": frame, "intersection_pixels": count,
            "first_area_pixels": a, "second_area_pixels": b,
            "intersection_over_first": count / a if a else None,
            "intersection_over_second": count / b if b else None,
            "intersection_over_smaller_nonempty_layer": count / min(a, b) if a and b else None,
            "iou": count / (a + b - count) if a + b - count else None,
            "intersection_bbox_xyxy_exclusive":
                [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if count else None}


def scan_sequence(root, name, source):
    videos = source["videos"]
    count = videos["color.mp4"]["decoded_frames"]
    h, w = videos["color.mp4"]["height"], videos["color.mp4"]["width"]
    paths = {role: root / name / "masks_sam3" / f"{role}.mkv"
             for role in ("left_hand", "right_hand", "object")}
    before = {role: {"path": str(path), "stat": stat_record(path), "sha256": sha256(path)}
              for role, path in paths.items()}
    candidates = {pair: [] for pair in PAIRS}
    totals = {pair: {"total_intersection_pixels": 0, "frames_with_overlap": 0} for pair in PAIRS}
    with ExitStack() as stack:
        processes = {}
        for role, path in paths.items():
            metadata = videos[f"masks_sam3/{role}.mkv"]
            if metadata["pix_fmt"] != "gray" or (metadata["height"], metadata["width"]) != (h, w):
                raise ValueError("Expected native gray masks at RGB dimensions")
            stderr = stack.enter_context(tempfile.TemporaryFile())
            process = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray", "-threads", "1", "pipe:1"],
                stdout=subprocess.PIPE, stderr=stderr)
            processes[role] = process, stderr
        try:
            for frame in range(count):
                layers = {}
                for role, (process, _) in processes.items():
                    raw = process.stdout.read(h * w)
                    if len(raw) != h * w:
                        raise ValueError(f"Incomplete frame: {name}/{role}/{frame}")
                    layers[role] = np.frombuffer(raw, np.uint8).reshape(h, w)
                for pair, (first, second) in PAIRS.items():
                    row = overlap_row(layers[first], layers[second], frame)
                    totals[pair]["total_intersection_pixels"] += row["intersection_pixels"]
                    totals[pair]["frames_with_overlap"] += int(row["intersection_pixels"] > 0)
                    if row["intersection_pixels"]:
                        candidates[pair].append(row)
                        candidates[pair].sort(key=lambda value: (-value["intersection_pixels"], value["frame_index_zero_based"]))
                        del candidates[pair][3:]
            for role, (process, stderr) in processes.items():
                if process.stdout.read(1):
                    raise ValueError(f"Extra frames: {name}/{role}")
                code = process.wait(timeout=30)
                stderr.seek(0)
                error = stderr.read()
                if code or error:
                    raise ValueError(f"Decode error: {name}/{role}: {error[-1000:]}")
        finally:
            for process, _ in processes.values():
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
    after = {role: {"path": str(path), "stat": stat_record(path), "sha256": sha256(path)}
             for role, path in paths.items()}
    if before != after:
        raise ValueError("Source mask changed during read-only scan")
    return {"decoded_frames_per_layer": count, "width": w, "height": h,
            "source_before": before, "source_after": after, "source_unchanged": True,
            "pairs": {pair: {"ordered_roles": list(PAIRS[pair]), **totals[pair],
                             "top3_by_intersection_pixels": candidates[pair]} for pair in PAIRS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prior evidence")
    audit_hash = sha256(args.structural_audit)
    source = json.loads(args.structural_audit.read_text())
    result = {"format": "realsense-overlap-diagnostic-selection-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(args.root.resolve()),
        "selection_rule": "Preselected sequences basket/cup; exhaustively scan their three independent layers; rank each pair by intersection pixels descending, then zero-based frame ascending; retain top 3 per pair.",
        "mask_definition": "Each layer is independently native_value > 0; no exclusivity or replacement. Intersection is not automatically an annotation error or contact GT.",
        "scope": "Targeted maxima for human diagnosis only; not random sample, no full-dataset error-rate claim.",
        "structural_audit": {"path": str(args.structural_audit.resolve()), "sha256": audit_hash},
        "sequences": {}}
    for name in ("basket", "cup"):
        result["sequences"][name] = scan_sequence(args.root, name, source["sequences"][name])
    if sha256(args.structural_audit) != audit_hash:
        raise ValueError("Structural audit changed during scan")
    result["recommended_three_diagnostic_frames"] = []
    for name, pair in (("basket", "left_right"), ("cup", "left_right"), ("basket", "right_object")):
        row = result["sequences"][name]["pairs"][pair]["top3_by_intersection_pixels"]
        if row:
            result["recommended_three_diagnostic_frames"].append({"sequence": name, "pair": pair, **row[0]})
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["recommended_three_diagnostic_frames"], ensure_ascii=False))


if __name__ == "__main__":
    main()

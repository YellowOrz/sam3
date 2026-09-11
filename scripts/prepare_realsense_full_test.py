"""Export all reviewed RealSense RGB frames, retaining missing/uncertain references.

CPU-only bounded streaming; no repairs, selection by prediction, or source writes.
The existing fixed128 publication is checked read-only and is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from scripts.audit_nakehand_dataset import probe, sha256
from scripts.prepare_nakehand_test import side_annotation
from scripts.prepare_realsense_manual_review import stable_record, verify
from scripts.prepare_realsense_test import CATEGORIES, REFERENCE_DESCRIPTION, exclusions, write_json


FORMAT = "sam3-realsense-full-reference-test-v1"
SIDES = ("left_hand", "right_hand")
FRAME_COUNTS = {"basket": 733, "black_pen": 508, "blue_pen": 581,
                "bottle": 796, "bowl": 607, "cup": 642, "left_hand": 660,
                "milk": 667, "red_pen": 432, "right_hand": 578}


def reference_group(provided):
    if set(provided) != set(SIDES) or any(type(v) is not bool for v in provided.values()):
        raise ValueError("Both explicit boolean reference-provided flags required")
    return ("both" if all(provided.values()) else "left_only" if provided["left_hand"]
            else "right_only" if provided["right_hand"] else "none")


def quality_flags(recording, frame, blocked):
    flags = [flag for flag in blocked.get(recording, {}).get(frame, [])
             if flag != "previously_displayed_random_review"]
    if (recording, frame) == ("basket", 609):
        flags.append("D4_exact_reviewed_frame_uncertain_not_confirmed_error")
    return sorted(set(flags))


def side_quality_flags(flags):
    """D1 affects left; D2/D3/D4 affect right, not the unflagged opposite side."""
    return {side: [flag for flag in flags
                   if not (flag.startswith("D1_") and side != "left_hand")
                   and not (flag.startswith(("D2_", "D3_", "D4_")) and side != "right_hand")]
            for side in SIDES}


def render_selection(counts, legacy_selection):
    selected = {name: sorted(set(legacy_selection.get(name, []) + [(count - 1) // 2]))
                for name, count in sorted(counts.items())}
    if sum(map(len, selected.values())) > 30:
        raise ValueError("Frozen display budget exceeds 30")
    if any(type(frame) is not int or not 0 <= frame < counts[name]
           for name, frames in selected.items() for frame in frames):
        raise ValueError("Frozen display frame out of range")
    return selected


def stream_frames(path, count, fps, width, height, gray=False):
    """Yield one frame+source PTS; never buffer an entire video or stderr pipe.

    ffprobe's decoded-frame order supplies the original presentation timestamps.
    ffmpeg emits passthrough raw frames in that same order, with exact EOF count.
    Only the small timestamp list and one raw frame are held for each stream.
    """
    command = ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
               "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
               "-of", "json", str(path)]
    frames = json.loads(subprocess.check_output(command, timeout=180))["frames"]
    pts = [float(row["best_effort_timestamp_time"]) for row in frames]
    if len(pts) != count or any(not np.isfinite(t) or abs(t - i / fps) > .0012
                                for i, t in enumerate(pts)):
        raise ValueError(f"Source frame count/PTS differs: {path}")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-copyts", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
               "-threads", "1", "-filter_threads", "1", "-f", "rawvideo",
               "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"]
    size = width * height * (1 if gray else 3)
    shape = (height, width) if gray else (height, width, 3)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            for index, timestamp in enumerate(pts):
                chunks, remaining = [], size
                while remaining:
                    chunk = process.stdout.read(remaining)
                    if not chunk:
                        raise ValueError(f"Truncated decoded frame {index}: {path}")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                yield np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(shape), timestamp
            if process.stdout.read(1):
                raise ValueError(f"Unexpected extra decoded frame: {path}")
            if process.wait(timeout=60):
                errors.seek(0)
                raise ValueError(f"CPU decoder failed: {errors.read(4096).decode(errors='replace')}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def load_legacy(root):
    paths = {name: root / name for name in ("READY.json", "frozen-plan.json", "manifest.json", "annotations.json")}
    docs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    ready = docs["READY.json"]
    if ready.get("status") != "complete":
        raise ValueError("Legacy fixed128 is not READY")
    for key, name in (("frozen_plan_sha256", "frozen-plan.json"), ("manifest_sha256", "manifest.json"),
                      ("annotations_sha256", "annotations.json")):
        if sha256(paths[name]) != ready[key]:
            raise ValueError("Legacy fixed128 metadata hash changed")
    rows = {row["image_id"]: row for row in docs["manifest.json"]["image_outputs"]}
    images = docs["annotations.json"]["images"]
    lookup = {(row["recording_id"], row["source_frame_index"]): row for row in images}
    if len(images) != 128 or len(lookup) != 128 or len(rows) != 128 or set(rows) != {r["id"] for r in images}:
        raise ValueError("Legacy fixed128 identity coverage differs")
    return docs, lookup, rows, [stable_record(path) for path in paths.values()]


def checked_png(path, array):
    Image.fromarray(array).save(path, compress_level=1)
    with Image.open(path) as decoded:
        if not np.array_equal(np.asarray(decoded), array):
            raise ValueError(f"PNG pixel round-trip mismatch: {path}")
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def check_legacy_pixels(root, row, arrays):
    for key, array in arrays.items():
        record = row["files"][key]
        path = root / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Legacy output changed: {path}")
        with Image.open(path) as image:
            if not np.array_equal(np.asarray(image), array):
                raise ValueError(f"Legacy RGB/raw mask pixel mismatch: {path}")


def export(root, audit_root, legacy_root, output):
    root, audit_root, legacy_root, output = map(lambda x: Path(x).resolve(),
                                               (root, audit_root, legacy_root, output))
    if output.exists() or any(output == p or p in output.parents for p in (root, legacy_root, audit_root)):
        raise ValueError("Use a NEW output outside source/review/fixed128 directories")
    review_path, issue_path = audit_root / "manual-review/frozen-plan.json", audit_root / "review-issues.json"
    review, issues = json.loads(review_path.read_text()), json.loads(issue_path.read_text())
    if review.get("source_root") != str(root) or issues.get("source_root") != str(root):
        raise ValueError("Review evidence source root mismatch")
    if review["frame_counts"] != FRAME_COUNTS or sum(FRAME_COUNTS.values()) != 6204:
        raise ValueError("Require the audited 10 recordings and all 6204 frames")
    blocked = exclusions(review, issues)
    legacy_docs, legacy, legacy_rows, legacy_evidence = load_legacy(legacy_root)
    reviewed_sources = {row["path"]: row for row in review["sources"]}
    sources, streams, missing_paths = [], {}, []
    for name, count in sorted(FRAME_COUNTS.items()):
        paths = {"rgb": root / name / "color.mp4", "metadata": root / name / "meta.json"}
        for side in SIDES:
            path = root / name / "masks_sam3" / f"{side}.mkv"
            if path.is_file():
                paths[side] = path
            else:
                missing_paths.append(path)
        if json.loads(paths["metadata"].read_text())["frames"] != count:
            raise ValueError(f"Recording metadata count changed: {name}")
        expected = {"left_hand"} if name == "left_hand" else {"right_hand"} if name == "right_hand" else set(SIDES)
        if set(paths) - {"rgb", "metadata"} != expected:
            raise ValueError(f"Reviewed reference availability changed: {name}")
        for key, path in paths.items():
            record = stable_record(path)
            if reviewed_sources.get(str(path), {}).get("sha256") != record["sha256"]:
                raise ValueError(f"Source bytes differ from reviewed version: {path}")
            sources.append(record)
            if key != "metadata":
                details = probe(path, count_packets=True)["streams"][0]
                if (details["width"], details["height"], int(details["nb_read_packets"]),
                        Fraction(details["avg_frame_rate"])) != (640, 480, count, Fraction(30)):
                    raise ValueError(f"Source dimensions/frame count/fps changed: {path}")
                if key != "rgb" and (details["pix_fmt"], details["codec_name"]) != ("gray", "ffv1"):
                    raise ValueError(f"Expected lossless uint8 reference stream: {path}")
        streams[name] = paths
    evidence = [stable_record(path) for path in (review_path, issue_path)] + legacy_evidence
    output.mkdir(parents=True, exist_ok=False)
    selection = render_selection(FRAME_COUNTS, legacy_docs["frozen-plan.json"]["render_selection"])
    plan = {"format": FORMAT, "dataset_role": "external_test_only", "evaluation_scope": "fixed_development_benchmark", "frame_counts": FRAME_COUNTS,
            "source_root": str(root), "sources": sources, "review_evidence": evidence,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "selection": "all original video frames",
            "render_selection": selection, "selection_uses_predictions_or_mask_pixels": False,
            "render_method": "legacy fixed128 16 display frames plus the integer midpoint of every recording, deduplicated",
            "quality_flags_policy": "retain D1/D2 confirmed issues and D3/D4 uncertainties; never repair or exclude frames",
            "legacy_fixed128_root": str(legacy_root), "legacy_fixed128_ready_sha256": sha256(legacy_root / "READY.json"),
            "reference_description": REFERENCE_DESCRIPTION,
            "missing_reference_policy": "unknown, not negative; infer every RGB but side mask metrics must be null",
            "deployment": "All output paths relative to data root. Original source paths are provenance only; metadata and RGB/reference assets may be copied separately to identical layout."}
    write_json(output / "frozen-plan.json", plan)
    plan_hash = sha256(output / "frozen-plan.json")
    coco = {"info": {"format": FORMAT, "dataset_role": "external_test_only", "evaluation_scope": "fixed_development_benchmark", "split": "test",
                     "frozen_plan_sha256": plan_hash, "reference_description": REFERENCE_DESCRIPTION,
                     "missing_reference_policy": plan["missing_reference_policy"]},
            "categories": CATEGORIES, "images": [], "annotations": []}
    manifest = {"format": FORMAT, "frozen_plan_sha256": plan_hash, "image_outputs": [],
                "sources": sources, "review_evidence": evidence, "statistics": {}}
    groups, positives, empties, unknowns, flagged = Counter(), Counter(), Counter(), Counter(), Counter()
    matched_legacy, overlap_images = set(), 0
    for name, paths in streams.items():
        provided = {side: side in paths for side in SIDES}
        group = reference_group(provided)
        with ExitStack() as stack:
            iterators = {key: stack.enter_context(closing(stream_frames(path, FRAME_COUNTS[name], 30., 640, 480, key != "rgb")))
                         for key, path in paths.items() if key != "metadata"}
            for frame in range(FRAME_COUNTS[name]):
                pairs = {key: next(iterator) for key, iterator in iterators.items()}
                arrays, pts = {k: p[0] for k, p in pairs.items()}, {k: p[1] for k, p in pairs.items()}
                if max(pts.values()) - min(pts.values()) > .0012:
                    raise ValueError(f"Cross-stream PTS mismatch: {name}/{frame}")
                image_id, old = len(coco["images"]) + 1, legacy.get((name, frame))
                if old is not None:
                    check_legacy_pixels(legacy_root, legacy_rows[old["id"]], arrays)
                    matched_legacy.add(old["id"])
                directory = output / "images" / name / f"frame-{frame:06d}"
                directory.mkdir(parents=True)
                files = {}
                for key, array in arrays.items():
                    path = directory / ("rgb.png" if key == "rgb" else f"{key}_raw.png")
                    files[key] = {"path": path.relative_to(output).as_posix(), **checked_png(path, array)}
                mapping = {"rgb_video": str(paths["rgb"]), "mask_videos": {s: str(paths[s]) for s in SIDES if provided[s]},
                           "source_frame_index": frame, "frame_numbering": "zero_based_exported_video_not_bag_message",
                           "pts_seconds": pts}
                flags = quality_flags(name, frame, blocked)
                common = {"recording_id": name, "source_frame_index": frame, "reference_provided": provided,
                          "has_both_reference": all(provided.values()), "raw_provided_group": group,
                          "quality_flags": flags, "reference_quality_flags": side_quality_flags(flags),
                          "previously_displayed_random_review": "previously_displayed_random_review" in blocked.get(name, {}).get(frame, []),
                          "legacy_fixed128": old is not None, "legacy_fixed128_image_id": old["id"] if old else None,
                          "source_mapping": mapping, "render_preselected": frame in selection[name]}
                coco["images"].append({"id": image_id, "file_name": files["rgb"]["path"], "width": 640, "height": 480,
                                       "source_dataset": "realsense", "frame_index": frame, **common})
                for category, side in enumerate(SIDES, 1):
                    if not provided[side]:
                        unknowns[side] += 1
                        continue
                    annotation = side_annotation(arrays[side], image_id, category, len(coco["annotations"]) + 1, mapping)
                    if annotation is not None:
                        annotation["label_source"] = REFERENCE_DESCRIPTION
                        coco["annotations"].append(annotation)
                        positives[side] += 1
                    else:
                        empties[side] += 1
                overlap = int(((arrays[SIDES[0]] > 0) & (arrays[SIDES[1]] > 0)).sum()) if all(provided.values()) else None
                overlap_images += int(overlap is not None and overlap > 0)
                manifest["image_outputs"].append({"image_id": image_id, "files": files,
                                                   "left_right_overlap_pixels": overlap, **common})
                groups[group] += 1
                flagged.update(flags)
            # Exhaust once more so each generator checks extra output, EOF and return code.
            for iterator in iterators.values():
                if next(iterator, None) is not None:
                    raise ValueError("Unexpected extra source frame")
        print(f"exported {name}: {FRAME_COUNTS[name]} full frames", flush=True)
    if len(coco["images"]) != 6204 or len(matched_legacy) != 128:
        raise ValueError("Full/legacy image coverage differs")
    verify(sources + evidence)
    if any(path.exists() for path in missing_paths):
        raise ValueError("Missing source reference appeared during export")
    manifest.update(status="complete", sources_unchanged=True)
    manifest["statistics"] = {"images": 6204, "recordings": 10, "annotations": len(coco["annotations"]),
                              "raw_provided_groups": dict(groups), "positive_reference_by_side": dict(positives),
                              "empty_reference_by_side": dict(empties), "missing_reference_by_side": dict(unknowns),
                              "quality_flag_counts": dict(flagged), "images_with_left_right_overlap": overlap_images,
                              "legacy_fixed128_pixel_equal_images": len(matched_legacy),
                              "render_images": sum(map(len, selection.values())), "png_roundtrip_verified": True,
                              "rle_roundtrip_verified": True, "all_source_pts_verified": True}
    snapshot = output / "code-snapshot"
    snapshot.mkdir()
    manifest["code"] = []
    for name in (Path(__file__).name, "prepare_realsense_test.py", "prepare_nakehand_test.py",
                 "audit_nakehand_dataset.py", "prepare_realsense_manual_review.py"):
        source = Path(__file__).resolve().with_name(name)
        shutil.copy2(source, snapshot / name)
        manifest["code"].append({"path": f"code-snapshot/{name}", "sha256": sha256(snapshot / name)})
    write_json(output / "annotations.json", coco)
    write_json(output / "manifest.json", manifest)
    write_json(output / "READY.json", {"format": FORMAT, "status": "complete", "dataset_role": "external_test_only",
                                       "evaluation_scope": "fixed_development_benchmark",
                                       "frozen_plan_sha256": plan_hash, "annotations_sha256": sha256(output / "annotations.json"),
                                       "manifest_sha256": sha256(output / "manifest.json")})
    return manifest["statistics"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit-root", "legacy-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.root, args.audit_root, args.legacy_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

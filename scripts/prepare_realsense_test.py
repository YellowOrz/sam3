"""Freeze a recording-balanced RealSense hand-reference test subset, never train.

Missing streams are unknown, not negatives. Existing masks are neither repaired
nor filtered by appearance/model prediction. Sources remain read-only. The
publication is an auxiliary-reference test, not independent human pixel GT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image

from scripts.audit_nakehand_dataset import probe, sha256
from scripts.prepare_nakehand_test import select_frames, side_annotation
from scripts.prepare_realsense_manual_review import stable_record, verify


FORMAT = "sam3-realsense-fixed-reference-test-v1"
SEED = 20260911
PER_RECORDING = 16
RECORDINGS = ("basket", "black_pen", "blue_pen", "bottle", "bowl", "cup", "milk", "red_pen")
FRAME_COUNTS = dict(zip(RECORDINGS, (733, 508, 581, 796, 607, 642, 667, 432)))
CATEGORIES = [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]
REFERENCE_DESCRIPTION = "SAM3-assisted propagated reference masks; not independently annotated human pixel ground truth"


def exclusions(review_plan: dict, issues: dict) -> dict[str, dict[int, list[str]]]:
    """Fixed review-based exclusions, never new model/GT-quality filtering.

    D4's reviewed temporal window is omitted from this restricted score, not
    deleted or relabelled. D3 is version-uncertain, NOT a proven current error.
    Other overlapping pixels remain intact; this is not a claim they are valid.
    """
    if issues.get("format") != "realsense-review-issue-register-v1":
        raise ValueError("Expected existing machine-readable review issues")
    required = {row["id"]: (row["recording"], row["frame_index"])
                for row in issues["confirmed_current_annotation_issues"]}
    if required != {"D1": ("basket", 622), "D2": ("cup", 319)}:
        raise ValueError("Known confirmed review issues changed; review protocol before use")
    if [(row["recording"], row["frame_index"]) for row in issues["uncertainties"]] != [("basket", 609)]:
        raise ValueError("Known uncertainty changed")
    d3 = issues["version_findings_not_current_error_counts"]
    if (len(d3) != 1 or d3[0]["recording"] != "cup"
            or d3[0]["historical_side_correspondence_interval_inclusive"] != [171, 186]
            or d3[0]["current_version_error_confirmed"] is not False):
        raise ValueError("D3 version finding changed")
    result = {name: {} for name in RECORDINGS}
    def add(recording, frame, reason):
        if recording in result:
            result[recording].setdefault(frame, []).append(reason)
    for row in review_plan["samples"]:
        add(row["recording"], row["frame_index"], "previously_displayed_random_review")
    for name, (recording, frame) in required.items():
        add(recording, frame, f"{name}_confirmed_wrong_side_reference")
    for frame in range(570, 646):
        add("basket", frame, "D4_previously_reviewed_temporal_window_uncertainty")
    for frame in range(171, 187):
        add("cup", frame, "D3_historical_version_correspondence_uncertainty_not_proven_current_error")
    return result


def select_indices(counts, excluded, seed=SEED, per_recording=PER_RECORDING):
    if type(seed) is not int or type(per_recording) is not int or per_recording < 1:
        raise ValueError("Integer seed and positive sample count required")
    rng = random.Random(seed)
    result = {}
    for name in sorted(counts):
        count = counts[name]
        if type(count) is not int or count < 1:
            raise ValueError("Positive integer source frame counts required")
        blocked = set(excluded.get(name, {}))
        if any(type(frame) is not int or not 0 <= frame < count for frame in blocked):
            raise ValueError("Excluded frame is outside original video")
        eligible = [frame for frame in range(count) if frame not in blocked]
        if len(eligible) < per_recording:
            raise ValueError("Too few eligible frames")
        result[name] = sorted(rng.sample(eligible, per_recording))
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def export(root, audit_root, output):
    root, audit_root, output = (Path(path).resolve() for path in (root, audit_root, output))
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Output must be a NEW directory outside shared source")
    review_path = audit_root / "manual-review/frozen-plan.json"
    issue_path = audit_root / "review-issues.json"
    review, issues = (json.loads(path.read_text()) for path in (review_path, issue_path))
    if review.get("source_root") != str(root) or issues.get("source_root") != str(root):
        raise ValueError("Review evidence belongs to another source")
    excluded = exclusions(review, issues)
    original_sources = {row["path"]: row for row in review["sources"]}
    sources, streams = [], {}
    for name in RECORDINGS:
        directory = root / name
        paths = {"rgb": directory / "color.mp4", "metadata": directory / "meta.json"}
        paths.update({side: directory / "masks_sam3" / f"{side}.mkv" for side in ("left_hand", "right_hand")})
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"Full RGB and BOTH side streams required: {name}")
        if json.loads(paths["metadata"].read_text()).get("frames") != FRAME_COUNTS[name]:
            raise ValueError(f"Source frame count changed: {name}")
        for key, path in paths.items():
            current = stable_record(path)
            previous = original_sources.get(str(path))
            if previous is None or current["sha256"] != previous["sha256"]:
                raise ValueError(f"Source bytes differ from reviewed version: {path}")
            sources.append(current)
            if key != "metadata":
                details = probe(path, count_packets=True)["streams"][0]
                if (details["width"], details["height"], int(details["nb_read_packets"]),
                        Fraction(details["avg_frame_rate"])) != (640, 480, FRAME_COUNTS[name], Fraction(30)):
                    raise ValueError(f"RGB/mask stream dimensions/count/rate mismatch: {path}")
                if key != "rgb" and (details["pix_fmt"], details["codec_name"]) != ("gray", "ffv1"):
                    raise ValueError(f"Expected lossless grayscale instance mask: {path}")
        streams[name] = paths
    evidence = [stable_record(path) for path in (review_path, issue_path)]
    selection = select_indices(FRAME_COUNTS, excluded)
    output.mkdir(parents=True, exist_ok=False)
    plan = {"format": FORMAT, "dataset_role": "external_test_only", "seed": SEED,
            "samples_per_recording": PER_RECORDING, "frame_counts": FRAME_COUNTS,
            "source_root": str(root), "sources": sources, "review_evidence": evidence,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
            "selection_method": "random.Random(seed).sample(sorted eligible original frame indices,16), sorted recordings",
            "selection_uses_predictions_or_mask_pixels": False,
            "render_selection": {name: [frames[0], frames[len(frames)//2]] for name, frames in selection.items()},
            "exclusions": excluded, "excluded_recordings": {
                "left_hand": "right reference stream missing; unknown is not negative",
                "right_hand": "left reference stream missing; unknown is not negative"},
            "reference_description": REFERENCE_DESCRIPTION,
            "scope": "128-frame restricted hand-reference test; object/MANO/depth not evaluated; no epoch/LR/threshold selection",
            "caveats": ["No automatic repairs, mutual-exclusion overwrite or area filtering",
                "Known-issue neighborhoods outside explicitly excluded windows may still contain errors",
                "Other left/right overlap is recorded, not silently discarded",
                "Not a pristine blind, cross-subject, or full-dataset independent-GT benchmark",
                "Development review influenced eligibility; correlated frames do not establish independent scenes"]}
    write_json(output / "frozen-plan.json", plan)
    plan_hash = sha256(output / "frozen-plan.json")
    coco = {"info": {"dataset_role": "external_test_only", "split": "test", "format": FORMAT,
                       "frozen_plan_sha256": plan_hash, "reference_description": REFERENCE_DESCRIPTION},
            "categories": CATEGORIES, "images": [], "annotations": []}
    manifest = {"format": FORMAT, "image_outputs": [], "sources": sources, "review_evidence": evidence,
                "frozen_plan_sha256": plan_hash, "statistics": {}}
    presence = Counter()
    for name, frames in selection.items():
        arrays, timestamps = {}, {}
        for key in ("rgb", "left_hand", "right_hand"):
            arrays[key], timestamps[key] = select_frames(streams[name][key], frames, 30., 640, 480, gray=key != "rgb")
        for local, frame in enumerate(frames):
            pts = {key: value[local] for key, value in timestamps.items()}
            if max(pts.values()) - min(pts.values()) > .0012:
                raise ValueError("Cross-stream PTS mismatch")
            image_id = len(coco["images"]) + 1
            directory = output / "images" / name / f"frame-{frame:06d}"
            directory.mkdir(parents=True)
            files = {}
            for key in ("rgb", "left_hand", "right_hand"):
                array = arrays[key][local]
                path = directory / ("rgb.png" if key == "rgb" else f"{key}_raw.png")
                Image.fromarray(array).save(path)
                with Image.open(path) as decoded:
                    if not np.array_equal(np.asarray(decoded), array):
                        raise ValueError("Lossless PNG round-trip mismatch")
                files[key] = {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
            mapping = {"rgb_video": str(streams[name]["rgb"]), "mask_videos": {
                side: str(streams[name][side]) for side in ("left_hand", "right_hand")},
                "source_frame_index": frame, "frame_numbering": "zero_based_exported_video_not_bag_message",
                "pts_seconds": pts}
            image = {"id": image_id, "file_name": files["rgb"]["path"], "width": 640, "height": 480,
                     "source_dataset": "realsense", "recording_id": name, "frame_index": frame,
                     "source_frame_index": frame, "source_mapping": mapping, "primary_test": True,
                     "reference_provided": {"left_hand": True, "right_hand": True},
                     "view_type": "unverified", "render_preselected": frame in plan["render_selection"][name]}
            coco["images"].append(image)
            flags = []
            for category, side in enumerate(("left_hand", "right_hand"), 1):
                raw = arrays[side][local]
                flags.append(bool((raw > 0).any()))
                annotation = side_annotation(raw, image_id, category, len(coco["annotations"]) + 1, mapping)
                if annotation is not None:
                    annotation["label_source"] = REFERENCE_DESCRIPTION
                    coco["annotations"].append(annotation)
            presence["both" if all(flags) else "single" if any(flags) else "empty"] += 1
            overlap = int(((arrays["left_hand"][local] > 0) & (arrays["right_hand"][local] > 0)).sum())
            manifest["image_outputs"].append({"image_id": image_id, "recording_id": name,
                "source_frame_index": frame, "files": files, "left_right_overlap_pixels": overlap,
                "source_mapping": mapping})
        print(f"exported {name}: {len(frames)} fixed test frames", flush=True)
    verify(sources + evidence)
    manifest.update(status="complete", sources_unchanged=True)
    manifest["statistics"] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
        "presence": dict(presence), "images_with_left_right_overlap": sum(
            row["left_right_overlap_pixels"] > 0 for row in manifest["image_outputs"])}
    snapshot = output / "code-snapshot"
    snapshot.mkdir()
    helpers = (Path(__file__), Path(__file__).with_name("prepare_nakehand_test.py"),
               Path(__file__).with_name("audit_nakehand_dataset.py"),
               Path(__file__).with_name("prepare_realsense_manual_review.py"))
    manifest["code"] = []
    for path in helpers:
        shutil.copy2(path, snapshot / path.name)
        manifest["code"].append({"source": str(path.resolve()), "sha256": sha256(snapshot / path.name)})
    write_json(output / "annotations.json", coco)
    write_json(output / "manifest.json", manifest)
    write_json(output / "READY.json", {"format": FORMAT, "status": "complete",
        "dataset_role": "external_test_only", "frozen_plan_sha256": plan_hash,
        "annotations_sha256": sha256(output / "annotations.json"), "manifest_sha256": sha256(output / "manifest.json")})
    return manifest["statistics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.audit_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export all nakehand frames under a frozen recording-level development split.

CPU-only, lossless decoded RGB and both original side masks. These are existing
SAM3-assisted references, not independent manual pixel GT. No source is edited.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

try:
    from scripts.prepare_nakehand_test import (
        CATEGORIES, atomic_json, frame_chunk, load_recording, sha256,
        side_annotation, source_record, verify_sources,
    )
except ModuleNotFoundError:
    from prepare_nakehand_test import (
        CATEGORIES, atomic_json, frame_chunk, load_recording, sha256,
        side_annotation, source_record, verify_sources,
    )


FRAME_COUNTS = {
    "nakehandego/20260907_134035": 629,
    "nakehandego/20260907_140713": 4084,
    "nakehandego/20260907_142020": 3449,
    "nakehandego/20260907_144324": 2362,
    "nakehandexo/20260907_123926": 4379,
    "nakehandexo/20260907_131154": 3595,
}
SPLIT_RECORDINGS = {
    "train": ("nakehandego/20260907_134035", "nakehandego/20260907_140713", "nakehandexo/20260907_123926"),
    "val": ("nakehandego/20260907_142020",),
    "development_holdout": ("nakehandego/20260907_144324", "nakehandexo/20260907_131154"),
}
COCO_SPLITS = {"train": "train", "val": "val", "development_holdout": "test"}
LABEL_SOURCE = "Existing SAM3 prompted/propagated pseudo-label reference; not independent human pixel ground truth"
HELPERS = ("prepare_nakehand_training.py", "prepare_nakehand_test.py", "audit_nakehand_dataset.py")


def split_plan(counts):
    if counts != FRAME_COUNTS:
        raise ValueError("Frozen six-recording inventory/frame counts changed")
    assigned = [name for names in SPLIT_RECORDINGS.values() for name in names]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(counts):
        raise ValueError("Recording split overlaps or is incomplete")
    offset, recordings = 0, {}
    for name, count in sorted(counts.items()):
        recordings[name] = {"frame_count": count, "global_image_id_offset": offset,
                            "frames": {"start": 0, "stop_exclusive": count, "step": 1}}
        offset += count
    return {"recordings": recordings,
            "splits": {split: {"coco_split": COCO_SPLITS[split], "recordings": list(names),
                               "images": sum(counts[name] for name in names)}
                       for split, names in SPLIT_RECORDINGS.items()},
            "total_images": offset}


def parse_pts(log, frame_count, fps):
    """Validate every actual decoded index/PTS; capture timestamps stay separate."""
    pairs = re.findall(r"\bn:\s*(\d+)\s+pts:\s*[-+\d]+\s+pts_time:([-+\d.eE]+)", log)
    if len(pairs) != frame_count:
        raise ValueError(f"Decoded PTS count mismatch: {len(pairs)} != {frame_count}")
    result = []
    for expected, (index, value) in enumerate(pairs):
        pts = float(value)
        if int(index) != expected or not np.isfinite(pts) or abs(pts - expected / fps) > 0.0012:
            raise ValueError(f"Decoded frame index/PTS mismatch at {expected}: {index}/{pts}")
        result.append(pts)
    return result


class VideoFrames:
    """Bounded-memory CPU decode; stderr goes to a file so pipes cannot deadlock."""

    def __init__(self, path, frame_count, fps, width, height, gray=False):
        self.path, self.frame_count, self.fps = Path(path), frame_count, fps
        if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError("Invalid stream frame count/FPS")
        self.shape = (height, width) if gray else (height, width, 3)
        self.frame_bytes = int(np.prod(self.shape))
        self.read_count = 0
        self.log = tempfile.TemporaryFile(mode="w+b")
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-hwaccel", "none",
                   "-threads", "1", "-filter_threads", "1", "-copyts", "-i", str(path),
                   "-map", "0:v:0", "-vf", "showinfo", "-fps_mode", "passthrough",
                   "-threads", "1", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        except BaseException:
            self.log.close()
            raise

    def __enter__(self):
        return self

    def _read(self, length):
        result = bytearray()
        deadline = time.monotonic() + 60
        while len(result) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"CPU decoder stalled: {self.path}")
            chunk = os.read(self.process.stdout.fileno(), length - len(result))
            if not chunk:
                break
            result.extend(chunk)
        return result

    def read(self):
        if self.read_count >= self.frame_count:
            raise ValueError("Attempt to read past the declared stream")
        raw = self._read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            raise ValueError(f"Truncated decoded stream: {self.path} frame {self.read_count}")
        self.read_count += 1
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.shape)

    def finish(self):
        if self.read_count != self.frame_count or self._read(1):
            raise ValueError(f"Stream contains fewer/more frames than metadata: {self.path}")
        code = self.process.wait(timeout=60)
        self.log.seek(0)
        log = self.log.read().decode("utf-8", errors="replace")
        if code:
            raise ValueError(f"ffmpeg failed for {self.path}: {log[-2000:]}")
        return parse_pts(log, self.frame_count, self.fps)

    def __exit__(self, *unused):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.log.close()


def recording_identity(recording):
    metadata = recording["metadata"]
    count = metadata["frame_count"]
    for index, frame in enumerate(metadata["frames"]):
        if frame["video_frame_index"] != index or frame["source_frame_index"] != index:
            raise ValueError(f"Source/video frame identity changed: {recording['recording_id']}/{index}")
    if len(metadata["frames"]) != count:
        raise ValueError("Incomplete source frame map")


def freeze_plan(root, output, prior_test):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the read-only source dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    recordings = [load_recording(root, path) for path in sorted(root.glob("*/*/metadata.json"))]
    counts = {recording["recording_id"]: recording["metadata"]["frame_count"] for recording in recordings}
    plan = split_plan(counts)
    for recording in recordings:
        recording_identity(recording)
    sources = [item for recording in recordings for item in recording["sources"]]
    if len(sources) != 60 or len({item["path"] for item in sources}) != 60:
        raise ValueError("Expected exactly 60 distinct original/provenance source files")
    prior_test = Path(prior_test).resolve()
    prior = json.loads(prior_test.read_text())
    exposed = [{"recording_id": image["recording_id"], "frame_index": image["frame_index"],
                "was_primary_test": image["primary_test"]} for image in prior["images"]]
    if len(exposed) != 602 or sum(row["was_primary_test"] for row in exposed) != 600:
        raise ValueError("Expected existing external test: 600 primary + 2 additional diagnostics")
    if len({(row["recording_id"], row["frame_index"]) for row in exposed}) != len(exposed):
        raise ValueError("Duplicate earlier exposure frames")
    if any(row["recording_id"] not in counts or not 0 <= row["frame_index"] < counts[row["recording_id"]]
           for row in exposed):
        raise ValueError("Earlier external sample outside frozen source inventory")
    plan.update(format="nakehand-development-split-plan-v1", created_at_utc=datetime.now(timezone.utc).isoformat(),
                root=str(root), output=str(output), sources=sources,
                label_source=LABEL_SOURCE, png_compress_level=3,
                source_scope="RGB, left/right mask videos, recording metadata and both SAM3 metadata/interactions/upstream masks; no depth/MANO",
                person_session_camera_relationship="unknown; user confirmed unknown on 2026-09-10; recording-disjoint does not imply person/session-disjoint",
                prior_exposure={"source": source_record(prior_test), "frames": exposed,
                                "scope": "All six recordings already inspected through 600 primary + 2 diagnostic frames; not a pristine final test"},
                human_review="13 individual frames accepted, not a certification of all frames; no pixel editing or acceptance propagation",
                development_holdout_policy="Previously exposed recordings; freeze allocation now. Do not tune or select this experimental round on development_holdout model results.",
                selection="All original frames; no mask/quality/presence/model filtering; whole recording belongs to exactly one split",
                implementation_sources=[source_record(Path(__file__).resolve().with_name(name)) for name in HELPERS])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "frozen-plan.json", plan)
    snapshot_dir = output / "implementation-snapshot"
    snapshot_dir.mkdir()
    for item in plan["implementation_sources"]:
        shutil.copyfile(item["path"], snapshot_dir / Path(item["path"]).name)
    print(json.dumps({"plan": str(output / "frozen-plan.json"), "sha256": sha256(output / "frozen-plan.json"),
                      "splits": plan["splits"], "source_files": len(sources)}, indent=2), flush=True)
    return output / "frozen-plan.json"


def save_png(path, array, compress_level):
    Image.fromarray(array).save(path, compress_level=compress_level)
    with Image.open(path) as saved:
        if saved.format != "PNG" or not np.array_equal(np.asarray(saved), array):
            raise ValueError(f"Lossless PNG round-trip failed: {path}")
    return {"path": path.name, "sha256": sha256(path)}


def validate_export(split_dir, plan_sha256=None):
    """Read-only full PNG/hash/RLE/area/bbox validation; does not require READY."""
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if sha256(split_dir / "annotations.json") != manifest["annotations_sha256"]:
        raise ValueError("Published annotations hash mismatch")
    if plan_sha256 is not None and manifest["frozen_plan_sha256"] != plan_sha256:
        raise ValueError("Frozen plan hash mismatch")
    coco = json.loads((split_dir / "annotations.json").read_text())
    if coco["categories"] != CATEGORIES:
        raise ValueError("Unexpected side vocabulary")
    images = {image["id"]: image for image in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate image IDs")
    annotations = {}
    for annotation in coco["annotations"]:
        key = annotation["image_id"], annotation["category_id"]
        if key in annotations or key[0] not in images or key[1] not in (1, 2):
            raise ValueError("Duplicate/orphan/unrecognized side annotation")
        if annotation["id"] != key[0] * 2 + key[1] - 1:
            raise ValueError("Unstable annotation ID")
        annotations[key] = annotation
    if {row["image_id"] for row in manifest["image_outputs"]} != set(images) or len(manifest["image_outputs"]) != len(images):
        raise ValueError("Incomplete output validation manifest")
    for row in manifest["image_outputs"]:
        image = images[row["image_id"]]
        arrays = {}
        for name, item in row["files"].items():
            path = split_dir / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"PNG hash mismatch: {path}")
            with Image.open(path) as saved:
                arrays[name] = np.asarray(saved).copy()
                if saved.format != "PNG":
                    raise ValueError("Output is not PNG")
        shape = image["height"], image["width"]
        if arrays["rgb"].shape != (*shape, 3):
            raise ValueError("RGB shape mismatch")
        if image["file_name"] != row["files"]["rgb"]["path"]:
            raise ValueError("RGB file mapping mismatch")
        for side, category in (("left", 1), ("right", 2)):
            raw, binary = arrays[f"{side}_instance_raw"], arrays[f"{side}_binary"]
            if raw.shape != shape or binary.shape != shape or not np.array_equal(binary, (raw > 0).astype(np.uint8) * 255):
                raise ValueError("Source/binary mask identity mismatch")
            expected = side_annotation(raw, image["id"], category, image["id"] * 2 + category - 1)
            actual = annotations.get((image["id"], category))
            if (expected is None) != (actual is None):
                raise ValueError("Empty/nonempty side annotation mismatch")
            if actual is not None:
                for key in ("segmentation", "area", "bbox", "source_instance_values"):
                    if actual[key] != expected[key]:
                        raise ValueError(f"Full RLE/area/bbox validation mismatch: {key}")
                if not np.array_equal(mask_utils.decode(actual["segmentation"]), raw > 0):
                    raise ValueError("RLE pixel mismatch")
            provenance = image["source_masks"][side]
            if provenance["video_pts_seconds"] != image["video_pts_seconds"]:
                raise ValueError("RGB/side PTS mismatch in published provenance")
    return {"images": len(images), "annotations": len(annotations), "pngs": len(images) * 5,
            "all_png_sha256_and_pixels_checked": True, "all_rle_area_bbox_checked": True}


def export_split(output, split, plan, plan_sha, recordings):
    directory = output / split
    directory.mkdir(exist_ok=False)
    for folder in ("images", "source-masks", "reference-masks"):
        (directory / folder).mkdir()
    info = {"description": "nakehand full-frame recording-disjoint development data", "split": COCO_SPLITS[split],
            "dataset_role": "validation" if split == "val" else split,
            "label_provenance": LABEL_SOURCE, "mask_semantics": "one annotation per nonempty side, union of all source values > 0; no area filtering",
            "person_session_camera_relationship": plan["person_session_camera_relationship"],
            "prior_exposure": "All six recordings were used in the earlier 600-frame external development comparison; not a pristine final test",
            "primary_test_field": "Legacy inclusion flag is true for every exported frame, including train; it does NOT identify an independent test split. Use info.split and dataset_role.",
            "frozen_plan_sha256": plan_sha, "geometry_inputs": "none", "full_frame_export": True,
            "human_review": plan["human_review"], "development_holdout_policy": plan["development_holdout_policy"]}
    coco = {"info": info, "categories": CATEGORIES, "images": [], "annotations": []}
    manifest = {"format": "nakehand-development-split-v1", "split": split, "frozen_plan_sha256": plan_sha,
                "sources": plan["sources"], "recordings": [], "image_outputs": [], "status": "validating"}
    for name in plan["splits"][split]["recordings"]:
        recording = recordings[name]
        metadata, count = recording["metadata"], recording["metadata"]["frame_count"]
        first = len(coco["images"])
        with ExitStack() as stack:
            streams = {key: stack.enter_context(VideoFrames(recording["paths"][key], count, recording["fps"],
                                                           recording["width"], recording["height"], key != "rgb"))
                       for key in ("rgb", "left", "right")}
            for frame in range(count):
                decoded = {key: stream.read() for key, stream in streams.items()}
                image_id = plan["recordings"][name]["global_image_id_offset"] + frame
                view, sequence = name.split("/")
                basename = f"{view}__{sequence}__frame-{frame:06d}"
                rgb_relative = f"images/{basename}.png"
                image = {"id": image_id, "file_name": rgb_relative, "width": recording["width"], "height": recording["height"],
                         "source": "nakehand", "sequence": sequence, "view": view, "view_type": view.removeprefix("nakehand"),
                         "recording_id": name, "frame_index": frame, "source_frame_index": metadata["frames"][frame]["source_frame_index"],
                         "source_timestamp_seconds": metadata["frames"][frame]["timestamp"], "video_pts_seconds": None,
                         "source_rgb_path": str(recording["paths"]["rgb"]), "source_metadata_path": str(recording["paths"]["metadata"]),
                         "source_mask_paths": {side: str(recording["paths"][side]) for side in ("left", "right")},
                         "primary_test": True, "diagnostic_id": None, "diagnostic_ids": [], "dataset_role": info["dataset_role"],
                         "human_review": {"status": "not_certified_by_this_export", "scope": "No acceptance extrapolated; separate 13-frame review record remains authoritative"},
                         "source_masks": {}, "source_left_right_overlap_pixels": int(((decoded["left"] > 0) & (decoded["right"] > 0)).sum())}
                files = {"rgb": {**save_png(directory / rgb_relative, decoded["rgb"], plan["png_compress_level"]), "path": rgb_relative}}
                for side, category in (("left", 1), ("right", 2)):
                    raw, sidecar = decoded[side], recording["sidecars"][side]
                    chunk = frame_chunk(sidecar["metadata"], frame)
                    values = [int(value) for value in np.unique(raw)]
                    if not set(values) <= {0} | {int(value) for value in chunk["object_id_to_label"].values()}:
                        raise ValueError(f"Undeclared chunk label: {name}/{frame}/{side}")
                    raw_relative = f"source-masks/{basename}__{side}_instance_raw.png"
                    binary_relative = f"reference-masks/{basename}__{side}.png"
                    for key, relative, array in ((f"{side}_instance_raw", raw_relative, raw),
                                                 (f"{side}_binary", binary_relative, (raw > 0).astype(np.uint8) * 255)):
                        files[key] = {**save_png(directory / relative, array, plan["png_compress_level"]), "path": relative}
                    provenance = {"path": str(recording["paths"][side]), "raw_instance_png": raw_relative,
                                  "binary_reference_png": binary_relative, "source_values": values,
                                  "video_pts_seconds": None, "chunk": chunk, "metadata_path": sidecar["metadata_path"],
                                  "interactions_path": sidecar["interactions_path"],
                                  "label_scope": "Source chunk/video label mapping, not an independently verified physical track ID"}
                    image["source_masks"][side] = provenance
                    annotation = side_annotation(raw, image_id, category, image_id * 2 + category - 1, provenance)
                    if annotation is not None:
                        coco["annotations"].append(annotation)
                coco["images"].append(image)
                manifest["image_outputs"].append({"image_id": image_id, "files": files})
                if (frame + 1) % 250 == 0 or frame + 1 == count:
                    print(f"{split}: {name} {frame + 1}/{count}", flush=True)
            pts = {key: stream.finish() for key, stream in streams.items()}
        if not pts["rgb"] == pts["left"] == pts["right"]:
            raise ValueError(f"All-frame RGB/left/right PTS mismatch: {name}")
        for frame, image in enumerate(coco["images"][first:]):
            image["video_pts_seconds"] = pts["rgb"][frame]
            for side in ("left", "right"):
                image["source_masks"][side]["video_pts_seconds"] = pts[side][frame]
        manifest["recordings"].append({"recording_id": name, "frame_count": count, "all_frame_indices_exported": True,
                                       "streams": recording["stream_info"], "video_pts_seconds": pts,
                                       "source_mask_metadata": recording["sidecars"], "identity_frame_map_verified": True})
    per_image = Counter(annotation["image_id"] for annotation in coco["annotations"])
    side_counts = Counter(annotation["category_id"] for annotation in coco["annotations"])
    counts = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
              "left_annotations": side_counts[1], "right_annotations": side_counts[2],
              "empty_images": sum(per_image[image["id"]] == 0 for image in coco["images"]),
              "one_hand_images": sum(per_image[image["id"]] == 1 for image in coco["images"]),
              "two_hand_images": sum(per_image[image["id"]] == 2 for image in coco["images"])}
    if counts["images"] != plan["splits"][split]["images"]:
        raise ValueError("Full-frame export count mismatch")
    atomic_json(directory / "annotations.json", coco)
    manifest.update(counts=counts, annotations_sha256=sha256(directory / "annotations.json"))
    atomic_json(directory / "manifest.json", manifest)
    validation = validate_export(directory, plan_sha)
    verify_sources(plan["sources"])
    manifest.update(status="complete", sources_unchanged=True, validation=validation)
    atomic_json(directory / "manifest.json", manifest)
    receipt = {"status": "complete", "annotations_sha256": manifest["annotations_sha256"],
               "manifest_sha256": sha256(directory / "manifest.json"), "frozen_plan_sha256": plan_sha,
               "counts": counts}
    atomic_json(directory / "READY.json", receipt)
    return {**receipt, "ready_sha256": sha256(directory / "READY.json"), "directory": split}


def export_plan(plan_path):
    plan_path = Path(plan_path).resolve()
    output, plan_sha = plan_path.parent, sha256(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan["format"] != "nakehand-development-split-plan-v1" or Path(plan["output"]).resolve() != output:
        raise ValueError("Frozen plan location/schema mismatch")
    if any((output / name).exists() for name in (*SPLIT_RECORDINGS, "READY.json", "manifest.json", "export-in-progress.json")):
        raise FileExistsError("Refusing to overwrite or silently resume an existing export")
    expected = split_plan({name: item["frame_count"] for name, item in plan["recordings"].items()})
    if any(plan[key] != expected[key] for key in ("recordings", "splits", "total_images")):
        raise ValueError("Frozen allocation differs from declared protocol")
    verify_sources(plan["sources"])
    verify_sources(plan["implementation_sources"])
    verify_sources([plan["prior_exposure"]["source"]])
    root = Path(plan["root"])
    if root == output or root in output.parents:
        raise ValueError("Output must be outside read-only source")
    recordings = {name: load_recording(root, root / name / "metadata.json") for name in plan["recordings"]}
    for recording in recordings.values():
        recording_identity(recording)
    sources = [item for recording in recordings.values() for item in recording["sources"]]
    if sources != plan["sources"]:
        raise ValueError("Source inventory/provenance changed since plan freeze")
    atomic_json(output / "export-in-progress.json", {"status": "in_progress", "frozen_plan_sha256": plan_sha})
    receipts = {split: export_split(output, split, plan, plan_sha, recordings) for split in SPLIT_RECORDINGS}
    verify_sources(plan["sources"])
    verify_sources(plan["implementation_sources"])
    verify_sources([plan["prior_exposure"]["source"]])
    if sha256(plan_path) != plan_sha:
        raise ValueError("Frozen plan changed during export")
    manifest = {"format": "nakehand-development-dataset-v1", "status": "complete", "frozen_plan_sha256": plan_sha,
                "source": plan["root"], "sources": plan["sources"], "sources_unchanged": True,
                "splits": receipts, "total_images": plan["total_images"], "label_source": LABEL_SOURCE,
                "person_session_camera_relationship": plan["person_session_camera_relationship"],
                "prior_exposure": plan["prior_exposure"], "development_holdout_policy": plan["development_holdout_policy"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "export-in-progress.json", {"status": "complete", "frozen_plan_sha256": plan_sha})
    atomic_json(output / "READY.json", {"status": "complete", "frozen_plan_sha256": plan_sha,
                                        "manifest_sha256": sha256(output / "manifest.json"), "splits": receipts})
    print(json.dumps({"output": str(output), "total_images": plan["total_images"], "splits": receipts}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/xuzhefeng/Datasets/wanqing_datasets/nakehand"))
    parser.add_argument("--prior-test", type=Path, default=Path("/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/external-test/nakehand-systematic600-20260910-v2/annotations.json"))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--output-dir", type=Path)
    modes.add_argument("--export-plan", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only and not args.output_dir:
        parser.error("--plan-only requires --output-dir")
    if args.export_plan:
        export_plan(args.export_plan)
    else:
        path = freeze_plan(args.root, args.output_dir, args.prior_test)
        if not args.plan_only:
            export_plan(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare CPU clip manifests for future memory training from bilateral COCO.

This is an alignment manifest, NOT a video training dataset or training loop.
Group by source/sequence/view and split at every missing source frame. Keep
negative frames and both class prompts. Physical identity comes exclusively
from sequence.json.extra.mano_sides, including completely hand-empty views.
Only the current DexYCB zero-based identity frame mapping is supported.
Inputs remain read-only; the output directory must not already exist.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


FORMAT = "sam3-dexycb-temporal-manifest-v1"
SPLITS = ("train", "val", "test")
PROMPTS = ("left_hand", "right_hand")
CATEGORIES = {1: "left_hand", 2: "right_hand"}
DEFAULT_DATA_ROOT = Path("/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1")
DEFAULT_UNIFIED_ROOT = Path("/data/xuzhefeng/Datasets/uni-hoi-dataset")


def integer(value, field):
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def component(value, field):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
        or value in (".", "..")
    ):
        raise ValueError(f"Unsafe/missing {field}: {value!r}")
    return value


class JsonSnapshots:
    def __init__(self):
        self.documents = {}
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        if path not in self.documents:
            raw = path.read_bytes()
            self.documents[path] = json.loads(raw)
            self.hashes[path] = hashlib.sha256(raw).hexdigest()
        return self.documents[path]

    def verify_unchanged(self):
        for path, expected in self.hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Input changed while building manifest: {path}")


def contiguous_segments(frames):
    """Return sorted contiguous runs and explicit gaps; never reindex frames."""
    ordered = sorted(frames, key=lambda frame: frame["frame_index"])
    segments, gaps = [], []
    for frame in ordered:
        if not segments:
            segments.append([frame])
            continue
        previous = segments[-1][-1]["frame_index"]
        current = frame["frame_index"]
        if current <= previous:
            raise ValueError(f"Duplicate source frame_index: {current}")
        if current != previous + 1:
            gaps.append({
                "before_frame_index": previous,
                "after_frame_index": current,
                "missing_frame_count": current - previous - 1,
            })
            segments.append([])
        segments[-1].append(frame)
    return segments, gaps


def clip_windows(segment, clip_length, stride):
    """Keep partial final windows of at least two frames, without padding."""
    windows = []
    for start in range(0, len(segment), stride):
        window = segment[start:start + clip_length]
        if len(window) >= 2:
            windows.append(window)
        if start + clip_length >= len(segment):
            break
    return windows


def _sequence_identity(sequence, source, sequence_id, view):
    if sequence.get("source") != source or sequence.get("seq_id") != sequence_id:
        raise ValueError(f"Sequence provenance mismatch: {source}/{sequence_id}")
    if view not in sequence.get("views", []):
        raise ValueError(f"Unknown view {view} in sequence {sequence_id}")
    side = sequence.get("extra", {}).get("mano_sides")
    if not isinstance(side, list) or len(side) != 1 or side[0] not in ("left", "right"):
        raise ValueError(f"Expected one authoritative mano_sides in {sequence_id}")
    if sequence.get("frame_map") or sequence.get("extra", {}).get("frame_map"):
        raise ValueError(f"Non-identity frame_map requires another adapter: {sequence_id}")
    subject = component(sequence.get("subject"), "subject")
    num_frames = integer(sequence.get("num_frames"), "num_frames")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive: {sequence_id}")
    return side[0], subject, num_frames


def _group_split(
    document, split, unified_root, snapshots, seen_frames, subject_splits
):
    categories = document.get("categories", [])
    if len(categories) != 2 or {
        category["id"]: category["name"] for category in categories
    } != CATEGORIES:
        raise ValueError(f"Expected bilateral categories {CATEGORIES}: {split}")
    annotations_by_image = defaultdict(list)
    annotation_ids = set()
    for annotation in document["annotations"]:
        annotation_id = integer(annotation["id"], "annotation_id")
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation_id in {split}: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = integer(annotation["image_id"], "annotation image_id")
        annotations_by_image[image_id].append(annotation)

    image_ids, filenames, groups = set(), set(), {}
    for image in document["images"]:
        image_id = integer(image["id"], "image_id")
        if image_id in image_ids:
            raise ValueError(f"Duplicate image_id in {split}: {image_id}")
        image_ids.add(image_id)
        source = component(image.get("source"), "source")
        sequence_id = component(image.get("sequence"), "sequence")
        view = component(image.get("view"), "view")
        if source != "dexycb":
            raise ValueError(f"Only DexYCB identity frame mapping is supported: {source}")
        frame = integer(image.get("frame_index"), "frame_index")
        frame_key = (source, sequence_id, view, frame)
        if frame_key in seen_frames:
            raise ValueError(f"Duplicate source frame across/within splits: {frame_key}")
        seen_frames.add(frame_key)
        filename = image.get("file_name")
        expected = f"images/{source}__{sequence_id}__{view}__{frame:08d}.jpg"
        if filename != expected:
            raise ValueError(f"COCO filename/provenance mismatch: {filename!r}")
        if filename in filenames:
            raise ValueError(f"Duplicate file_name in {split}: {filename}")
        filenames.add(filename)

        sequence_path = (
            unified_root / "sequences" / source / sequence_id / "sequence.json"
        )
        sequence = snapshots.read(sequence_path)
        side, subject, num_frames = _sequence_identity(sequence, source, sequence_id, view)
        if not 0 <= frame < num_frames:
            raise ValueError(f"Frame outside sequence bounds: {frame_key}")
        subject_key = (source, subject)
        if subject_key in subject_splits and subject_splits[subject_key] != split:
            raise ValueError(
                f"Subject leaks across splits: {subject_key} in "
                f"{subject_splits[subject_key]} and {split}"
            )
        subject_splits[subject_key] = split

        annotations = annotations_by_image.get(image_id, [])
        category_id = 1 if side == "left" else 2
        if len(annotations) > 1 or any(
            annotation.get("category_id") != category_id for annotation in annotations
        ):
            raise ValueError(f"COCO annotation contradicts authoritative side: {frame_key}")
        ids = sorted(annotation["id"] for annotation in annotations)
        group_key = (source, sequence_id, view)
        if group_key not in groups:
            groups[group_key] = {
                "source": source, "sequence": sequence_id, "view": view,
                "subject": subject, "side": side,
                "physical_hand_id": f"{sequence_id}:{side}",
                "side_authority": "sequence.extra.mano_sides[0]",
                "sequence_path": str(sequence_path), "num_source_frames": num_frames,
                "frames": [],
            }
        groups[group_key]["frames"].append({
            "image_id": image_id, "file_name": filename, "frame_index": frame,
            "source_frame": frame, "annotation_ids": ids,
            "segmentation_target_present": bool(ids),
            "prompt_annotation_ids": {
                prompt: ids if prompt == f"{side}_hand" else [] for prompt in PROMPTS
            },
        })
    if set(annotations_by_image) - image_ids:
        raise ValueError(f"Annotations reference missing image IDs in {split}")
    return groups


def build_temporal_manifests(
    data_root, unified_root, output_dir, *, clip_length=8, stride=8
):
    data_root, unified_root, output_dir = (
        Path(path).resolve() for path in (data_root, unified_root, output_dir)
    )
    if type(clip_length) is not int or clip_length < 2:
        raise ValueError("clip_length must be an integer >= 2")
    if type(stride) is not int or not 1 <= stride <= clip_length:
        raise ValueError("stride must be an integer between 1 and clip_length")
    if output_dir == unified_root or unified_root in output_dir.parents:
        raise ValueError("Output must not be inside the shared unified dataset")
    if output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {output_dir}")

    snapshots = JsonSnapshots()
    grouped_splits, seen_frames, subject_splits = {}, set(), {}
    for split in SPLITS:
        annotation_path = data_root / split / "annotations.json"
        grouped_splits[split] = _group_split(
            snapshots.read(annotation_path), split, unified_root, snapshots,
            seen_frames, subject_splits,
        )

    summary = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU temporal alignment manifest; not a video training dataset",
        "data_root": str(data_root), "unified_root": str(unified_root),
        "output_dir": str(output_dir), "clip_length": clip_length, "stride": stride,
        "minimum_clip_frames": 2, "prompt_names": list(PROMPTS),
        "frame_mapping": "dexycb_contiguous_zero_based_identity",
        "tail_policy": "Keep >=2 frames without padding; record uncovered singleton tails",
        "subject_split_check": "Disjoint source/subject across all three input COCO splits",
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary_files = []
    try:
        for split in SPLITS:
            groups = grouped_splits[split]
            counts = Counter({key: 0 for key in (
                "input_frames", "input_negative_frames", "groups", "contiguous_segments",
                "frame_gaps", "missing_frames_inside_gaps", "source_frames_not_in_coco",
                "clips", "full_clips", "tail_clips", "clip_frame_occurrences",
                "covered_unique_frames", "covered_negative_frames", "uncovered_frames",
                "short_contiguous_segments", "short_tail_windows",
            )})
            gaps, omitted = [], []
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{split}-", suffix=".tmp", dir=output_dir
            )
            temporary_path = Path(temporary_name)
            destination = output_dir / f"{split}.clips.jsonl"
            temporary_files.append((temporary_path, destination))
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for key in sorted(groups):
                    group = groups[key]
                    frames = group["frames"]
                    metadata = {name: value for name, value in group.items() if name != "frames"}
                    counts["groups"] += 1
                    counts["input_frames"] += len(frames)
                    counts["input_negative_frames"] += sum(not frame["annotation_ids"] for frame in frames)
                    counts["source_frames_not_in_coco"] += group["num_source_frames"] - len(frames)
                    segments, group_gaps = contiguous_segments(frames)
                    counts["contiguous_segments"] += len(segments)
                    counts["frame_gaps"] += len(group_gaps)
                    counts["missing_frames_inside_gaps"] += sum(gap["missing_frame_count"] for gap in group_gaps)
                    gaps.extend({"source": key[0], "sequence": key[1], "view": key[2], **gap} for gap in group_gaps)
                    covered_ids = set()
                    for segment in segments:
                        counts["short_contiguous_segments"] += int(len(segment) < 2)
                        windows = clip_windows(segment, clip_length, stride)
                        for window in windows:
                            indices = [frame["frame_index"] for frame in window]
                            clip = {
                                "format": FORMAT, "split": split, **metadata,
                                "clip_id": f"{split}:{':'.join(key)}:{indices[0]}-{indices[-1]}",
                                "num_frames": len(window),
                                "is_tail": len(window) < clip_length,
                                "original_frame_indices": indices,
                                "prompt_names": list(PROMPTS), "frames": window,
                            }
                            stream.write(json.dumps(clip, ensure_ascii=False, allow_nan=False) + "\n")
                            counts["clips"] += 1
                            counts["tail_clips" if clip["is_tail"] else "full_clips"] += 1
                            counts["clip_frame_occurrences"] += len(window)
                            covered_ids.update(frame["image_id"] for frame in window)
                        uncovered = [frame for frame in segment if frame["image_id"] not in covered_ids]
                        if len(segment) >= 2 and uncovered:
                            counts["short_tail_windows"] += 1
                        reason = "segment_shorter_than_two" if len(segment) < 2 else "tail_shorter_than_two"
                        omitted.extend({
                            "source": key[0], "sequence": key[1], "view": key[2],
                            "image_id": frame["image_id"], "frame_index": frame["frame_index"],
                            "reason": reason,
                        } for frame in uncovered)
                    counts["covered_unique_frames"] += len(covered_ids)
                    counts["covered_negative_frames"] += sum(
                        frame["image_id"] in covered_ids and not frame["annotation_ids"]
                        for frame in frames
                    )
                    counts["uncovered_frames"] += len(frames) - len(covered_ids)
            annotation_path = (data_root / split / "annotations.json").resolve()
            summary["splits"][split] = {
                "counts": dict(counts), "frame_gaps": gaps, "omitted_frames": omitted,
                "subjects": sorted(subject for (source, subject), assigned in subject_splits.items() if assigned == split),
                "annotations_sha256": snapshots.hashes[annotation_path],
                "manifest": str(destination),
                "manifest_sha256": hashlib.sha256(temporary_path.read_bytes()).hexdigest(),
            }
        snapshots.verify_unchanged()
        summary["sources_rechecked_unchanged"] = True
        summary["source_files"] = [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(snapshots.hashes.items())
        ]
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".summary-", suffix=".tmp", dir=output_dir
        )
        temporary_files.append((Path(temporary_name), output_dir / "summary.json"))
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        for temporary_path, destination in temporary_files:
            os.link(temporary_path, destination)
        return summary
    finally:
        for temporary_path, _ in temporary_files:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--unified-root", type=Path, default=DEFAULT_UNIFIED_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    summary = build_temporal_manifests(
        args.data_root, args.unified_root, args.output_dir,
        clip_length=args.clip_length, stride=args.stride,
    )
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "splits": {split: data["counts"] for split, data in summary["splits"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()

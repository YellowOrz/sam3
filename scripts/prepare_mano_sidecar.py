#!/usr/bin/env python3
"""Audit/export MANO supervision aligned to the existing bilateral DexYCB COCO.

Contract (sam3-dexycb-mano-sidecar-v1): one JSONL row per exported COCO image,
including negative segmentation images. Join by (split, image_id), never row
number. Each row retains file_name/source/sequence/view/frame_index/source_frame
and one physical hand identified by sequence.extra.mano_sides (exactly one side
for DexYCB). Visible track IDs are optional and are NOT physical identities.
MANO validity is independent of whether the exported image has a visible mask.

Valid hands have finite global_orient[3], hand_pose[45], betas[10], transl[3];
rotations are radians/axis-angle and translation is metres in root_frame=camera.
World-frame parameters require an explicit extrinsic conversion and are invalid
for this camera-frame training adapter. Both
the file's explicit axis-angle declaration and side-specific PCA provenance
(MANO_LEFT/RIGHT.pkl, SHA256, flat_hand_mean=false) are required. That provenance
is checked as a declaration; this tool does not load MANO model pickles or
numerically recompute the upstream PCA expansion. Missing/invalid MANO records
have valid=false, explicit invalid_reasons and ALL parameter arrays null; they
must be masked out of a MANO loss, never converted to a valid zero target.

This adapter supports the current DexYCB converter's contiguous, zero-based
identity frame mapping only. It verifies frame bounds and rejects explicit
nonempty frame maps rather than silently guessing a different temporal mapping.
The source converter enforces source indices == range(num_frames).

Inputs are read-only. JSONL files (<split>.mano.jsonl) and summary.json are
written only under the required --output-dir, refusing existing outputs and
anything inside the shared unified root. SHA256 values refer to the exact bytes
parsed; all source files are rechecked before publishing to detect concurrent
repairs. Summary is published last. Partial temporary files are cleaned up.
This is a metadata/parameter alignment audit, not a mesh reprojection test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


FORMAT = "sam3-dexycb-mano-sidecar-v1"
SIDES = ("left", "right")
CATEGORIES = {1: "left_hand", 2: "right_hand"}
PARAMETER_SIZES = {"global_orient": 3, "hand_pose": 45, "betas": 10, "transl": 3}
DEFAULT_DATA_ROOT = Path("/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1")
DEFAULT_UNIFIED_ROOT = Path("/data/xuzhefeng/Datasets/uni-hoi-dataset")


class SourceSnapshots:
    """Cache parsed JSON and record hashes of the exact bytes we inspected."""

    def __init__(self):
        self.data = {}
        self.files = {}

    def read(self, path: Path, *, optional: bool = False):
        key = str(path.resolve())
        if key in self.data:
            return self.data[key]
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            if not optional:
                raise
            value = None
            descriptor = {"path": key, "exists": False, "sha256": None}
        else:
            value = json.loads(raw)
            descriptor = {
                "path": key, "exists": True,
                "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            }
        self.data[key] = value
        self.files[key] = descriptor
        return value

    def reference(self, path: Path) -> dict:
        return self.files[str(path.resolve())]

    def verify_unchanged(self):
        for path, expected in self.files.items():
            try:
                raw = Path(path).read_bytes()
            except FileNotFoundError:
                if expected["exists"]:
                    raise RuntimeError(f"Source disappeared during audit: {path}")
                continue
            if not expected["exists"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Source changed during audit; rerun on a stable snapshot: {path}")


def integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def component(value, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Unsafe/missing {field}: {value!r}")
    return value


def sequence_side(sequence: dict, source: str, sequence_id: str, view: str) -> str:
    if sequence.get("source") != source or sequence.get("seq_id") != sequence_id:
        raise ValueError(f"Sequence provenance mismatch: {source}/{sequence_id}")
    if view not in sequence.get("views", []):
        raise ValueError(f"Unknown view {view} in sequence {sequence_id}")
    sides = sequence.get("extra", {}).get("mano_sides")
    if not isinstance(sides, list) or len(sides) != 1 or sides[0] not in SIDES:
        raise ValueError(f"Expected one authoritative sequence.extra.mano_sides: {sequence_id}")
    if sequence.get("frame_map") or sequence.get("extra", {}).get("frame_map"):
        raise ValueError(f"Explicit frame_map needs another adapter: {sequence_id}")
    if integer(sequence.get("num_frames"), "num_frames") <= 0:
        raise ValueError("num_frames must be positive")
    return sides[0]


def index_mano(document: dict | None, sequence: dict, view: str) -> dict[int, dict]:
    if document is None:
        return {}
    if not isinstance(document, dict) or document.get("view_id") != view:
        raise ValueError(f"MANO document/view mismatch: {view}")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"MANO records must be a list: {view}")
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid MANO record: {view}")
        frame = integer(record.get("frame"), "MANO frame")
        if not 0 <= frame < sequence["num_frames"]:
            raise ValueError(f"MANO frame out of bounds: {view}/{frame}")
        if frame in indexed:
            raise ValueError(f"Duplicate physical-hand MANO record: {view}/{frame}")
        indexed[frame] = record
    return indexed


def finite_vector(value, size: int) -> bool:
    return (
        isinstance(value, list) and len(value) == size
        and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(item) for item in value)
    )


def hand_target(document: dict | None, record: dict | None, side: str, sequence: dict) -> dict:
    reasons = []
    pca = document.get("pca_conversion") if document else None
    representation = document.get("pose_representation") if document else None
    root_frame = document.get("root_frame") if document else None
    if document is None:
        reasons.append("missing_mano_file")
    else:
        if representation != "axis-angle":
            reasons.append("pose_representation_not_axis_angle")
        if root_frame != "camera":
            reasons.append("root_frame_requires_camera_conversion")
        if not isinstance(pca, dict):
            reasons.append("missing_pca_provenance")
        else:
            if pca.get("file") != f"MANO_{side.upper()}.pkl":
                reasons.append("pca_model_side_mismatch")
            if not isinstance(pca.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", pca["sha256"]):
                reasons.append("invalid_pca_model_sha256")
            if pca.get("flat_hand_mean") is not False:
                reasons.append("pca_flat_hand_mean_must_be_false")
    if record is None:
        reasons.append("missing_mano_record")
    else:
        if record.get("side") != side:
            reasons.append("mano_side_differs_from_sequence")
        if record.get("root_frame") != root_frame:
            reasons.append("record_root_frame_mismatch")
        if record.get("pose_representation", representation) != "axis-angle":
            reasons.append("record_pose_representation_not_axis_angle")
        if record.get("valid", True) is not True:
            reasons.append("source_marked_invalid")
        for name, size in PARAMETER_SIZES.items():
            if not finite_vector(record.get(name), size):
                reasons.append(f"invalid_{name}")
    valid = not reasons
    physical_hand_id = f"{sequence['seq_id']}:{side}"
    return {
        "physical_hand_id": physical_hand_id,
        "side": side,
        "side_authority": "sequence.extra.mano_sides[0]",
        "hand_id": record.get("hand_id") if record else None,
        "track_id": record.get("track_id") if record else None,
        "source": record.get("source") if record else None,
        "valid": valid,
        "invalid_reasons": reasons,
        "root_frame": root_frame,
        "rotation_units": "radians" if representation == "axis-angle" else None,
        "translation_units": "metres",
        "translation_units_basis": "unified DexYCB contract section 4.2",
        "pose_representation": representation,
        "pca_conversion": pca,
        "pca_verification": "metadata_declaration_only",
        **{name: list(record[name]) if valid else None for name in PARAMETER_SIZES},
    }


def load_coco(document: dict) -> tuple[list[dict], dict[int, list[dict]]]:
    if {category["id"]: category["name"] for category in document["categories"]} != CATEGORIES:
        raise ValueError(f"Expected bilateral categories {CATEGORIES}")
    images = sorted(document["images"], key=lambda image: integer(image["id"], "image_id"))
    annotations = {image["id"]: [] for image in images}
    if len(annotations) != len(images):
        raise ValueError("Duplicate COCO image_id")
    for annotation in document["annotations"]:
        image_id = integer(annotation["image_id"], "annotation image_id")
        if image_id not in annotations:
            raise ValueError(f"Annotation references missing image {image_id}")
        annotations[image_id].append(annotation)
    return images, annotations


def prepare_sidecars(
    data_root: Path, unified_root: Path, output_dir: Path,
    splits=("train", "val", "test"), limit: int = 0,
) -> dict:
    data_root, unified_root, output_dir = (Path(path).resolve() for path in (data_root, unified_root, output_dir))
    if output_dir == unified_root or unified_root in output_dir.parents:
        raise ValueError("Output must not be inside the shared unified dataset")
    if limit < 0 or not splits or len(splits) != len(set(splits)):
        raise ValueError("Expected nonnegative limit and unique nonempty splits")
    if any(split not in ("train", "val", "test") for split in splits):
        raise ValueError("Splits must be train, val or test")
    destinations = [output_dir / f"{split}.mano.jsonl" for split in splits]
    destinations.append(output_dir / "summary.json")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = SourceSnapshots()
    mano_indices = {}
    summary = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "unified_root": str(unified_root),
        "output_dir": str(output_dir), "limit_per_split": limit,
        "pca_verification": "metadata_declaration_only; no MANO model or mesh loaded",
        "validity_note": "MANO validity is independent of visible segmentation; invalid arrays are null",
        "splits": {},
    }
    temporary_files = []
    try:
        for split in splits:
            annotation_path = data_root / split / "annotations.json"
            images, annotations = load_coco(snapshots.read(annotation_path))
            selected = images[:limit] if limit else images
            counts = Counter({name: 0 for name in (
                "images", "valid_mano", "invalid_mano", "left_images", "right_images",
                "visible_mask_images", "no_mask_images", "visible_mask_valid_mano",
                "visible_mask_invalid_mano", "no_mask_valid_mano", "no_mask_invalid_mano",
            )})
            invalid_reasons = Counter()
            frames_seen = set()
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{split}.mano-", suffix=".tmp", dir=output_dir)
            temporary_path = Path(temporary_name)
            temporary_files.append((temporary_path, output_dir / f"{split}.mano.jsonl"))
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for image in selected:
                    source = component(image.get("source"), "source")
                    sequence_id = component(image.get("sequence"), "sequence")
                    view = component(image.get("view"), "view")
                    if source != "dexycb":
                        raise ValueError(f"Only DexYCB identity frame mapping is supported: {source}")
                    frame = integer(image.get("frame_index"), "frame_index")
                    filename = image.get("file_name")
                    expected_filename = f"images/{source}__{sequence_id}__{view}__{frame:08d}.jpg"
                    if filename != expected_filename:
                        raise ValueError(f"COCO filename/provenance mismatch: {filename!r}")
                    frame_key = (source, sequence_id, view, frame)
                    if frame_key in frames_seen:
                        raise ValueError(f"Duplicate COCO source frame: {frame_key}")
                    frames_seen.add(frame_key)
                    sequence_path = unified_root / "sequences" / source / sequence_id / "sequence.json"
                    mano_path = sequence_path.parent / view / "mano.json"
                    sequence = snapshots.read(sequence_path)
                    side = sequence_side(sequence, source, sequence_id, view)
                    if not 0 <= frame < sequence["num_frames"]:
                        raise ValueError(f"COCO frame outside sequence bounds: {frame_key}")
                    image_annotations = annotations[image["id"]]
                    if len(image_annotations) > 1 or any(
                        annotation.get("category_id") != SIDES.index(side) + 1
                        for annotation in image_annotations
                    ):
                        raise ValueError(f"COCO hand annotation contradicts sequence side: {frame_key}")
                    document = snapshots.read(mano_path, optional=True)
                    if mano_path not in mano_indices:
                        mano_indices[mano_path] = index_mano(document, sequence, view)
                    record = mano_indices[mano_path].get(frame)
                    target = hand_target(document, record, side, sequence)
                    visible = bool(image_annotations)
                    row = {
                        "format": FORMAT, "split": split, "image_id": image["id"],
                        "file_name": filename, "source": source, "sequence": sequence_id,
                        "view": view, "frame_index": frame, "source_frame": frame,
                        "source_frame_mapping": "dexycb_contiguous_zero_based_identity",
                        "source_paths": sequence.get("source_paths"),
                        "segmentation_target_present": visible,
                        "segmentation_annotation_ids": [annotation["id"] for annotation in image_annotations],
                        "segmentation_track_ids": [annotation.get("track_id") for annotation in image_annotations],
                        "hands": [target],
                        "source_files": {
                            "annotations": snapshots.reference(annotation_path),
                            "sequence": snapshots.reference(sequence_path),
                            "mano": snapshots.reference(mano_path),
                        },
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["images"] += 1
                    counts["valid_mano" if target["valid"] else "invalid_mano"] += 1
                    counts[f"{side}_images"] += 1
                    counts["visible_mask_images" if visible else "no_mask_images"] += 1
                    counts[f"{'visible_mask' if visible else 'no_mask'}_{'valid' if target['valid'] else 'invalid'}_mano"] += 1
                    invalid_reasons.update(target["invalid_reasons"])
            summary["splits"][split] = {
                "available_images": len(images), "counts": dict(counts),
                "invalid_reasons": dict(invalid_reasons),
                "annotations_sha256": snapshots.reference(annotation_path)["sha256"],
                "sidecar": str(output_dir / f"{split}.mano.jsonl"),
                "sidecar_sha256": hashlib.sha256(temporary_path.read_bytes()).hexdigest(),
            }
        snapshots.verify_unchanged()
        summary["source_files"] = list(snapshots.files.values())
        summary["sources_rechecked_unchanged"] = True
        descriptor, temporary_name = tempfile.mkstemp(prefix=".summary-", suffix=".tmp", dir=output_dir)
        temporary_files.append((Path(temporary_name), output_dir / "summary.json"))
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        for temporary_path, destination in temporary_files:
            # Atomic no-clobber publication; no unrelated output may be overwritten.
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
    parser.add_argument("--split", choices=("train", "val", "test"), action="append")
    parser.add_argument("--limit", type=int, default=0, help="First N image IDs per split; 0 audits all")
    args = parser.parse_args()
    summary = prepare_sidecars(args.data_root, args.unified_root, args.output_dir,
                               splits=args.split or ("train", "val", "test"), limit=args.limit)
    print(json.dumps({"output_dir": summary["output_dir"], "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()

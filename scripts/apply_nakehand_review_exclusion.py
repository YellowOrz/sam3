#!/usr/bin/env python3
"""Create a hash-pinned, frame-0-only review exclusion; never edit source data.

This emits annotation/record derivatives, NOT a standalone dataset root and NOT
recomputed metrics. Historical summaries and checkpoints remain untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

FORMAT = "sam3-nakehand-review-exclusion-v1"
TARGET = {
    "image_id": 4713,
    "recording_id": "nakehandego/20260907_142020",
    "source_frame_index": 0,
    "file_name": "images/nakehandego__20260907_142020__frame-000000.png",
}
SIDES = {"left_hand", "right_hand"}
SPLITS = ("train", "val", "development_holdout")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, expected=None):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return json.loads(data), digest


def validate_manifest(manifest):
    if manifest.get("format") != FORMAT or manifest.get("exclude") != [TARGET]:
        raise ValueError("Manifest must authorize exactly the confirmed frame-0 identity")
    if type(manifest["exclude"][0]["image_id"]) is not int or type(
        manifest["exclude"][0]["source_frame_index"]
    ) is not int:
        raise ValueError("Identity integers must not be booleans")
    hashes = manifest.get("source_annotations_sha256", {})
    if not isinstance(hashes, dict) or not set(SPLITS).issubset(hashes):
        raise ValueError("Manifest must pin train, val and development_holdout SHA256")
    if set(hashes) - set(SPLITS) - {"test"}:
        raise ValueError("Unexpected split in source hashes")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
           for v in hashes.values()):
        raise ValueError("Invalid source annotation SHA256")


def image_index(coco):
    images = coco.get("images")
    if not isinstance(images, list) or not isinstance(coco.get("annotations"), list):
        raise ValueError("Require COCO images and annotations lists")
    indexed = {}
    for image in images:
        key = image.get("id")
        if type(key) is not int or key in indexed:
            raise ValueError("Duplicate or invalid COCO image identity")
        if type(image.get("source_frame_index")) is not int:
            raise ValueError("Missing source frame identity")
        if image.get("frame_index") != image["source_frame_index"]:
            raise ValueError("Frame indices disagree")
        indexed[key] = image
    annotation_ids = set()
    for annotation in coco["annotations"]:
        if (type(annotation.get("image_id")) is not int
                or annotation["image_id"] not in indexed
                or type(annotation.get("id")) is not int
                or annotation["id"] in annotation_ids):
            raise ValueError("Invalid or duplicate annotation linkage")
        annotation_ids.add(annotation["id"])
    return indexed


def target_match(image):
    return (image.get("id") == TARGET["image_id"]
            and all(image.get(key) == value for key, value in TARGET.items()
                    if key != "image_id"))


def filter_coco(coco):
    indexed = image_index(coco)
    image = indexed.get(TARGET["image_id"])
    same_frame = [item for item in indexed.values()
                  if item.get("recording_id") == TARGET["recording_id"]
                  and item["source_frame_index"] == TARGET["source_frame_index"]]
    if image is None or not target_match(image) or len(same_frame) != 1:
        raise ValueError("Exact unique frame-0 COCO identity not found")
    filtered = copy.deepcopy(coco)
    filtered["images"] = [item for item in filtered["images"]
                          if item["id"] != TARGET["image_id"]]
    filtered["annotations"] = [item for item in filtered["annotations"]
                               if item["image_id"] != TARGET["image_id"]]
    return filtered


def filter_records(rows, coco, model):
    """Validate every paired identity against COCO before excluding two rows."""
    images = image_index(coco)
    target = images.get(TARGET["image_id"])
    if target is None or not target_match(target):
        raise ValueError("Record filtering requires the exact reviewed COCO identity")
    indices = {image["id"]: index for index, image in enumerate(coco["images"])}
    seen = set()
    if not isinstance(rows, list):
        raise ValueError("Records must be a list")
    for row in rows:
        image_id, side = row.get("image_id"), row.get("prompt_key")
        image = images.get(image_id)
        if (type(image_id) is not int or image is None or side not in SIDES
                or (image_id, side) in seen or row.get("model") != model):
            raise ValueError("Invalid, duplicate or mismatched record identity")
        if (type(row.get("dataset_index")) is not int
                or row["dataset_index"] != indices[image_id]
                or row.get("file_name") != image["file_name"]
                or row.get("recording_id") != image["recording_id"]
                or row.get("observed_coco_image_id") != image_id
                or row.get("identity_verified") is not True):
            raise ValueError("Record provenance disagrees with source COCO")
        if "source_frame_index" in row and row["source_frame_index"] != image["source_frame_index"]:
            raise ValueError("Record source frame disagrees with source COCO")
        seen.add((image_id, side))
    expected = {(image_id, side) for image_id in images for side in SIDES}
    if seen != expected:
        raise ValueError("Require exactly both sides for every source validation image")
    removed = [row for row in rows if row["image_id"] == TARGET["image_id"]]
    if len(removed) != 2 or {row["prompt_key"] for row in removed} != SIDES:
        raise ValueError("Must remove exactly the two frame-0 query rows")
    return [row for row in rows if row["image_id"] != TARGET["image_id"]]


def apply_exclusion(source_root, output_dir, manifest_path, evaluation_dirs=()):
    if Path(output_dir).exists() or Path(output_dir).is_symlink():
        raise ValueError("Output must not already exist, including a dangling symlink")
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_relative_to(source_root):
        raise ValueError("Output must be a new directory outside the source dataset")
    request, request_hash = read_json(manifest_path)
    validate_manifest(request)
    pinned = request["source_annotations_sha256"]
    if (source_root / "test" / "annotations.json").exists() and "test" not in pinned:
        raise ValueError("Existing test split must also be pinned and checked")
    sources = {str(Path(manifest_path).resolve()): request_hash}
    datasets = {}
    for split, digest in pinned.items():
        path = source_root / split / "annotations.json"
        coco, _ = read_json(path, digest)
        image_index(coco)
        if split != "val" and any(
            image.get("recording_id") == TARGET["recording_id"]
            and image["source_frame_index"] == TARGET["source_frame_index"]
            for image in coco["images"]
        ):
            raise ValueError(f"Target source frame unexpectedly appears in {split}")
        datasets[split] = coco
        sources[str(path)] = digest
    val = datasets["val"]
    outputs = {"val.filtered.json": filter_coco(val)}
    record_sets = []
    for number, directory in enumerate(evaluation_dirs):
        directory = Path(directory).resolve()
        summary_path = directory / "summary.json"
        summary, summary_hash = read_json(summary_path)
        sources[str(summary_path)] = summary_hash
        input_hash = summary.get("input_sha256", {}).get(str(source_root / "val/annotations.json"))
        annotation_hash = summary.get("annotations_sha256", input_hash)
        if (summary.get("status") not in {"complete", "completed"}
                or summary.get("full_val_evaluated") is not True
                or annotation_hash != pinned["val"]
                or (input_hash is not None and input_hash != pinned["val"])
                or summary.get("evaluated_image_ids") != [i["id"] for i in val["images"]]
                or summary.get("evaluated_dataset_indices") != list(range(len(val["images"])))):
            raise ValueError("Evaluation must be complete and match the pinned full validation set")
        entries = summary.get("record_files")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Summary must enumerate record_files")
        for model, entry in entries.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
                raise ValueError("Unsafe model label")
            path = Path(entry["path"])
            path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Record file outside its evaluation directory")
            rows, digest = read_json(path, entry["sha256"])
            sources[str(path)] = digest
            filtered = filter_records(rows, val, model)
            relative = f"evaluations/eval-{number:02d}/{model}.filtered.json"
            outputs[relative] = filtered
            record_sets.append({"evaluation": str(directory), "model": model,
                                "source": str(path), "source_sha256": digest,
                                "output": relative, "source_queries": len(rows),
                                "filtered_queries": len(filtered), "removed_queries": 2})
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError("A source changed during validation")
    output_dir.mkdir(parents=True, exist_ok=False)

    def write(relative, data):
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}

    files = [write(relative, data) for relative, data in outputs.items()]
    manifest = {
        "format": FORMAT, "request": request, "request_sha256": request_hash,
        "excluded": [TARGET], "source_root": str(source_root),
        "source_image_root": str(source_root / "val"),
        "standalone_dataset_root": False, "metrics_recomputed": False,
        "warning": "Annotation and record derivatives only. Image file_name remains relative "
                   "to source_image_root. Do not use this output directory as a data root. "
                   "Old summaries describe the unfiltered run; no metrics were recomputed. "
                   "Original dataset_index values and IDs are preserved, not renumbered.",
        "source_images": len(val["images"]), "filtered_images": len(outputs["val.filtered.json"]["images"]),
        "source_annotations": len(val["annotations"]),
        "filtered_annotations": len(outputs["val.filtered.json"]["annotations"]),
        "checked_splits": list(pinned), "record_sets": record_sets,
        "sources": sources, "files": files,
    }
    manifest_file = write("manifest.json", manifest)
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError("A source changed; output has no successful receipt")
    receipt = {"format": FORMAT, "status": "complete", "manifest": manifest_file,
               "source_files_verified_unchanged": True, "metrics_recomputed": False,
               "standalone_dataset_root": False, "source_image_root": str(source_root / "val")}
    write("receipt.json", receipt)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = apply_exclusion(args.source_root, args.output_dir, args.manifest, args.evaluation_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "filtered_images": result["filtered_images"],
                      "record_sets": len(result["record_sets"]), "metrics_recomputed": False}))


if __name__ == "__main__":
    main()

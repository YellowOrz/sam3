#!/usr/bin/env python3
"""Read-only train-reference presence audit; no video decode or label correction.

For every recording/reference-absence mode, select first/middle/last by source
frame order before inspecting images (at most 24). Check published raw-instance
PNG > 0, binary-reference PNG, and COCO RLE agree; references are not human GT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
import torch

from scripts.report_nakehand_semantic_ablation import compare_training_contracts, file_hash
from scripts.train_nakehand_semantic_tokens import validate_training_annotations

MODES = ("reference_empty", "left_reference_only", "right_reference_only", "both_references")
MODE_BY_CATEGORIES = {(): MODES[0], (1,): MODES[1], (2,): MODES[2], (1, 2): MODES[3]}


def presence_mode(annotations):
    categories = tuple(sorted(annotations))
    if categories not in MODE_BY_CATEGORIES:
        raise ValueError("Require zero/left/right/both independent reference categories")
    return MODE_BY_CATEGORIES[categories]


def runs(images, consumed):
    result, current = [], []
    for image in sorted(images, key=lambda row: row["source_frame_index"]):
        if current and image["source_frame_index"] != current[-1]["source_frame_index"] + 1:
            result.append(current)
            current = []
        current.append(image)
    if current:
        result.append(current)
    return [{"first_frame": group[0]["source_frame_index"], "last_frame": group[-1]["source_frame_index"],
             "frames": len(group), "consumed_training_samples": sum(consumed[row["id"]] for row in group)}
            for group in result]


def summarize_presence(images, by_id, observed_ids):
    if len(set(row["id"] for row in images)) != len(images):
        raise ValueError("Duplicate image identities")
    consumed = Counter(observed_ids)
    if not set(consumed).issubset({row["id"] for row in images}):
        raise ValueError("Consumed sample does not belong to train")
    groups = defaultdict(lambda: defaultdict(list))
    for image in images:
        groups[image["recording_id"]][presence_mode(by_id[image["id"]])].append(image)
    report = {}
    for recording, modes in sorted(groups.items()):
        report[recording] = {}
        for mode in MODES:
            selected = modes[mode]
            report[recording][mode] = {"frames": len(selected),
                                       "consumed_training_samples": sum(consumed[row["id"]] for row in selected),
                                       "spans": runs(selected, consumed)}
    totals = {mode: {key: sum(row[mode][key] for row in report.values())
                     for key in ("frames", "consumed_training_samples")} for mode in MODES}
    return {"per_recording": report, "total": totals,
            "reference_absent_training_queries": 2 * totals[MODES[0]]["consumed_training_samples"]
            + totals[MODES[1]]["consumed_training_samples"] + totals[MODES[2]]["consumed_training_samples"],
            "consumed_training_samples": len(observed_ids), "consumed_unique_images": len(consumed)}


def select_samples(images, by_id, maximum=24):
    groups = defaultdict(list)
    for image in images:
        mode = presence_mode(by_id[image["id"]])
        if mode != "both_references":
            groups[(image["recording_id"], mode)].append(image)
    samples = []
    for (recording, mode), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["source_frame_index"])
        ranks = sorted({0, len(ordered) // 2, len(ordered) - 1})
        for rank in ranks:
            image = ordered[rank]
            samples.append({"image_id": image["id"], "recording_id": recording,
                            "source_frame_index": image["source_frame_index"], "reference_mode": mode,
                            "within_group_zero_based_rank": rank, "group_size": len(ordered),
                            "selection_roles": [role for role, value in (("first", 0), ("middle", len(ordered) // 2),
                                                                         ("last", len(ordered) - 1)) if value == rank]})
    if len(samples) > maximum:
        raise ValueError("Predefined selection exceeds 24 frames; do not silently truncate")
    for index, sample in enumerate(samples, 1):
        sample["review_id"] = f"T{index:02d}"
    return samples


def inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Published path must be relative inside train split")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("Published path escapes train split")
    return resolved


def compare_reference(raw, binary, annotation, shape):
    if raw.shape != shape or binary.shape != shape or raw.ndim != 2 or binary.ndim != 2:
        raise ValueError("Reference image shape differs from RGB/COCO")
    if not set(np.unique(binary)).issubset({0, 255}):
        raise ValueError("Published binary reference must contain only 0/255")
    if annotation is None:
        decoded = np.zeros(shape, bool)
    else:
        segmentation = dict(annotation["segmentation"])
        if not isinstance(segmentation.get("counts"), str) or tuple(segmentation["size"]) != shape:
            raise ValueError("Require frozen compressed RLE at original image dimensions")
        segmentation["counts"] = segmentation["counts"].encode("ascii")
        decoded = mask_utils.decode(segmentation).astype(bool)
        if int(decoded.sum()) != annotation["area"]:
            raise ValueError("RLE decoded pixels disagree with annotation area")
    mismatch_raw = int(np.count_nonzero((raw > 0) != (binary > 0)))
    mismatch_rle = int(np.count_nonzero(decoded != (binary > 0)))
    if mismatch_raw or mismatch_rle:
        raise ValueError("Published raw-instance/binary/RLE masks differ")
    return {"raw_instance_values": [int(value) for value in np.unique(raw)],
            "reference_pixels": int(decoded.sum()), "raw_vs_binary_mismatch_pixels": mismatch_raw,
            "binary_vs_rle_mismatch_pixels": mismatch_rle}


def validate_consumption(first, second, images, provenance, annotation_hash):
    comparison = compare_training_contracts(first, second)
    order = list(range(len(images)))
    random.Random(123).shuffle(order)
    order = order[:2000]
    if len(order) != 2000:
        raise ValueError("Require complete 2000-sample train prefix")
    for state in (first, second):
        config = state["training_config"]
        if (config.get("seed") != 123 or config.get("annotations_sha256") != annotation_hash
                or config.get("data_provenance") != provenance
                or state["planned_dataset_indices"] != order
                or state["observed_image_ids"] != [images[index]["id"] for index in order]):
            raise ValueError("Actual training prefix/provenance differs from frozen seed-123 train data")
    return {**comparison, "seed_123_order_regenerated": True,
            "actual_image_ids_match_indices": True}, first["observed_image_ids"]


def contact_sheets(samples, directory):
    pages = []
    width, height, caption = 320, 240, 50
    for page_number, start in enumerate(range(0, len(samples), 6), 1):
        page = samples[start:start + 6]
        canvas = Image.new("RGB", (width * 3, (height + caption) * len(page)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, sample in enumerate(page):
            y = row_index * (height + caption)
            heading = (f"{sample['review_id']} {sample['recording_id'].split('/')[-1]} frame={sample['source_frame_index']} "
                       f"id={sample['image_id']} | {sample['reference_mode']}")
            draw.text((5, y + 3), heading, fill="black")
            draw.text((5, y + 25), "RGB (published frame)", fill="black")
            draw.text((width + 5, y + 25), "LEFT REFERENCE (not human GT)", fill="black")
            draw.text((2 * width + 5, y + 25), "RIGHT REFERENCE (not human GT)", fill="black")
            for column, key in enumerate(("rgb", "left_reference", "right_reference")):
                with Image.open(sample["artifacts"][key]) as image:
                    panel = image.convert("RGB").resize((width, height),
                                                       Image.Resampling.LANCZOS if column == 0 else Image.Resampling.NEAREST)
                canvas.paste(panel, (column * width, y + caption))
        path = directory / f"contact-{page_number:02d}.png"
        canvas.save(path)
        pages.append(str(path.resolve()))
    return pages


def audit(data_root, checkpoints, output_dir):
    data_root, output_dir = data_root.resolve(), output_dir.resolve()
    if output_dir.exists() or data_root == output_dir or data_root in output_dir.parents:
        raise ValueError("Require a new output directory outside the immutable dataset")
    fingerprints = {}

    def remember(path, expected=None):
        path = Path(path).resolve()
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        if str(path) in fingerprints and fingerprints[str(path)] != digest:
            raise RuntimeError(f"Source changed during audit: {path}")
        fingerprints[str(path)] = digest
        return digest

    def read(path):
        raw = path.read_bytes()
        fingerprints[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    root = data_root.parent
    source_paths = {"annotations_sha256": data_root / "annotations.json",
                    "root_ready_sha256": root / "READY.json", "split_ready_sha256": data_root / "READY.json",
                    "root_manifest_sha256": root / "manifest.json", "split_manifest_sha256": data_root / "manifest.json",
                    "frozen_plan_sha256": root / "frozen-plan.json"}
    documents = {key: read(path) for key, path in source_paths.items()}
    hashes = {key: fingerprints[str(path.resolve())] for key, path in source_paths.items()}
    annotation_data = documents["annotations_sha256"]
    images, annotations, counts = validate_training_annotations(annotation_data)
    manifest = documents["split_manifest_sha256"]
    root_ready, split_ready = documents["root_ready_sha256"], documents["split_ready_sha256"]
    root_manifest, plan = documents["root_manifest_sha256"], documents["frozen_plan_sha256"]
    if (root_ready.get("status") != "complete" or split_ready.get("status") != "complete"
            or root_ready.get("manifest_sha256") != hashes["root_manifest_sha256"]
            or split_ready.get("manifest_sha256") != hashes["split_manifest_sha256"]
            or split_ready.get("annotations_sha256") != hashes["annotations_sha256"]
            or manifest.get("annotations_sha256") != hashes["annotations_sha256"]
            or root_ready.get("frozen_plan_sha256") != hashes["frozen_plan_sha256"]
            or split_ready.get("frozen_plan_sha256") != hashes["frozen_plan_sha256"]
            or manifest.get("frozen_plan_sha256") != hashes["frozen_plan_sha256"]
            or root_ready.get("splits", {}).get("train", {}).get("ready_sha256") != hashes["split_ready_sha256"]
            or manifest.get("sources") != plan.get("sources") or manifest.get("sources") != root_manifest.get("sources")
            or manifest.get("status") != "complete" or manifest.get("split") != "train"):
        raise ValueError("Published COCO/manifest/READY bindings disagree")
    provenance = {key: hashes[key] for key in hashes if key != "annotations_sha256"}
    provenance.update(recordings=sorted({image["recording_id"] for image in images}), dataset_role="train")
    states = []
    for checkpoint in checkpoints:
        remember(checkpoint)
        states.append(torch.load(checkpoint, map_location="cpu", weights_only=True))
    training_comparison, observed_ids = validate_consumption(*states, images, provenance, hashes["annotations_sha256"])
    statistics = summarize_presence(images, annotations, observed_ids)
    samples = select_samples(images, annotations)
    by_id = {image["id"]: image for image in images}
    outputs = {row["image_id"]: row for row in manifest["image_outputs"]}
    if len(outputs) != len(images) or set(outputs) != set(by_id):
        raise ValueError("Published file manifest lacks exact full train coverage")
    # Freeze selected identities in memory before opening any sample image.
    selection = [{key: value for key, value in sample.items()} for sample in samples]
    output_dir.mkdir(parents=True)
    consumed = Counter(observed_ids)
    for sample in samples:
        image = by_id[sample["image_id"]]
        files = outputs[image["id"]]["files"]
        paths = {key: inside(data_root, item["path"]) for key, item in files.items()}
        if str(paths["rgb"]) != str(inside(data_root, image["file_name"])):
            raise ValueError("Manifest RGB and COCO image filename disagree")
        for key, path in paths.items():
            remember(path, files[key]["sha256"])
        with Image.open(paths["rgb"]) as rgb:
            if rgb.size != (image["width"], image["height"]):
                raise ValueError("RGB dimensions differ from annotations")
        shape = (image["height"], image["width"])
        sample["pixel_checks"] = {}
        binary_sides = []
        for side, category in (("left", 1), ("right", 2)):
            for key, field in ((f"{side}_instance_raw", "raw_instance_png"), (f"{side}_binary", "binary_reference_png")):
                if files[key]["path"] != image["source_masks"][side][field]:
                    raise ValueError("Per-image source mapping and published file manifest disagree")
            with Image.open(paths[f"{side}_instance_raw"]) as raw_image:
                raw = np.asarray(raw_image).copy()
            with Image.open(paths[f"{side}_binary"]) as binary_image:
                binary = np.asarray(binary_image).copy()
            sample["pixel_checks"][side] = compare_reference(raw, binary, annotations[image["id"]].get(category), shape)
            binary_sides.append(binary > 0)
        sample["left_right_reference_overlap_pixels"] = int(np.logical_and(*binary_sides).sum())
        sample["consumed_in_first_2000"] = consumed[image["id"]]
        target_dir = output_dir / sample["review_id"]
        target_dir.mkdir()
        sample["artifacts"] = {}
        for key, source_key in (("rgb", "rgb"), ("left_reference", "left_binary"), ("right_reference", "right_binary")):
            target = target_dir / f"{key}.png"
            shutil.copyfile(paths[source_key], target)  # Byte-preserving original-size published PNG.
            if file_hash(target) != files[source_key]["sha256"]:
                raise RuntimeError("Copied review PNG differs from published source")
            sample["artifacts"][key] = str(target)
        sample["human_review_status"] = "not_yet_reviewed; do_not_infer_physical_side_from_screen_position"
    pages = contact_sheets(samples, output_dir)
    for source in (Path(__file__), Path(__file__).with_name("train_nakehand_semantic_tokens.py"),
                   Path(__file__).with_name("report_nakehand_semantic_ablation.py")):
        remember(source)
    for path, expected in list(fingerprints.items()):
        remember(path, expected)
    result = {
        "format": "nakehand-train-reference-presence-audit-v1", "data_root": str(data_root),
        "statistics": statistics, "full_train_counts": counts, "training_comparison": training_comparison,
        "selection_rule": "For each recording and reference-empty/left-only/right-only mode: first, upper-middle n//2, last by source frame; deduplicate; reject >24",
        "frozen_selection_before_image_inspection": selection,
        "samples": samples, "contact_sheets": pages,
        "raw_video_decoded_this_audit": False, "published_raw_png_vs_binary_vs_coco_checked": True,
        "dataset_or_training_modified": False, "input_files": [
            {"path": path, "sha256": digest} for path, digest in sorted(fingerprints.items())],
        "inputs_rechecked_unchanged": True,
        "limitations": ["Full counts classify existing references, not human-certified actual hand presence.",
                        "Pixel checks cover selected 24 published frames, not a fresh raw-video decode or all 9092 RGB frames.",
                        "Raw-video/source inventory hashes are bound by publication receipts, not rehashed in this audit.",
                        "First 2000 actual samples came from both completed semantic-delta trials; no future job consumption is inferred.",
                        "Frame-0 RIGHT-hand user confirmation applies to a different validation recording, not these train frames."]}
    output_files = []
    for sample in samples:
        output_files += [Path(path) for path in sample["artifacts"].values()]
    output_files += [Path(path) for path in pages]
    result["artifact_files"] = [{"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}
                                for path in output_files]
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--unconstrained-checkpoint", type=Path, required=True)
    parser.add_argument("--anchored-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.data_root, (args.unconstrained_checkpoint, args.anchored_checkpoint), args.output_dir)
    print(json.dumps({"samples": len(result["samples"]), "statistics": result["statistics"]["total"],
                      "manifest": str((args.output_dir / "manifest.json").resolve()),
                      "manifest_sha256": file_hash(args.output_dir / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

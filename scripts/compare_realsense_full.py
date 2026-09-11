"""CPU-only, complete-shard comparison against immutable full RealSense references.

Inputs are explicitly named, preselected evaluations, never checkpoint discovery.
Missing side streams remain unknown; raw provided references and quality-eligible
references are reported separately. No test-dependent sampling or tuning occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_erosion


FORMAT = "sam3-realsense-full-comparison-v1"
DATA_FORMAT = "sam3-realsense-full-reference-test-v1"
EVALUATION_FORMAT = "sam3-realsense-full-evaluation-v1"
SIDES = ("left_hand", "right_hand")
REFERENCE_DESCRIPTION = "SAM3-assisted propagated references; not independent human pixel ground truth"
IDENTITY_FIELDS = ("recording_id", "source_frame_index", "source_mapping")
METRIC_FIELDS = ("candidate_dice", "candidate_iou", "miss_zero_dice", "miss_zero_iou",
                 "candidate_boundary_iou_4px", "miss_zero_boundary_iou_4px")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def loads(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)


def read_json(path):
    raw = Path(path).read_bytes()
    return loads(raw), hashlib.sha256(raw).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def number(value, name, probability=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (probability and not 0 <= value <= 1):
        raise ValueError(f"Invalid {name}")
    return float(value)


def digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256: {name}")
    return value


def same(actual, expected, name):
    # Canonical JSON keeps bool/int and nested sequence/object types distinct.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"Changed or mismatched {name}")


def decode_rle(rle, shape):
    if (not isinstance(rle, dict) or set(rle) != {"size", "counts"}
            or rle["size"] != list(shape) or any(type(x) is not int for x in rle["size"])
            or not isinstance(rle["counts"], str) or not rle["counts"]):
        raise ValueError("Require original-size compressed RLE")
    # Validate the compressed run totals before passing data into the C decoder.
    # COCO uses signed, 5-bit continuation values and a two-run delta after run 2.
    counts, position = [], 0
    while position < len(rle["counts"]):
        value = shift = 0
        while True:
            if position >= len(rle["counts"]) or shift > 60:
                raise ValueError("Malformed compressed RLE")
            code = ord(rle["counts"][position]) - 48
            position += 1
            if not 0 <= code <= 63:
                raise ValueError("Invalid compressed RLE character")
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0 or value > shape[0] * shape[1] or len(counts) > shape[0] * shape[1]:
            raise ValueError("Invalid compressed RLE run")
        counts.append(value)
    if sum(counts) != shape[0] * shape[1]:
        raise ValueError("RLE run total differs from original image dimensions")
    try:
        decoded = mask_utils.decode(rle)
    except Exception as error:
        raise ValueError("Invalid prediction/reference RLE") from error
    if decoded.shape != tuple(shape) or not np.isin(decoded, (0, 1)).all():
        raise ValueError("Decoded RLE has invalid shape or pixels")
    return decoded.astype(bool)


def load_contract(root, *, expected_images=6204, expected_legacy_images=128):
    root = Path(root).resolve(strict=True)
    names = ("READY.json", "annotations.json", "manifest.json", "frozen-plan.json")
    documents, fingerprints = {}, {}
    for name in names:
        documents[name], fingerprints[name] = read_json(root / name)
    ready, data, manifest, plan = (documents[name] for name in names)
    if (ready.get("format") != DATA_FORMAT or ready.get("status") != "complete"
            or ready.get("dataset_role") != "external_test_only"
            or ready.get("evaluation_scope") != "fixed_development_benchmark"
            or data.get("info", {}).get("dataset_role") != "external_test_only"
            or manifest.get("status") != "complete"):
        raise ValueError("Require a complete full RealSense external-test publication")
    for name, key in (("annotations.json", "annotations_sha256"), ("manifest.json", "manifest_sha256"),
                      ("frozen-plan.json", "frozen_plan_sha256")):
        same(ready.get(key), fingerprints[name], key)
    for document in (data["info"], plan):
        same(document.get("evaluation_scope"), "fixed_development_benchmark", "development benchmark scope")
    same(data["info"].get("frozen_plan_sha256"), fingerprints["frozen-plan.json"], "annotation plan binding")
    same(manifest.get("frozen_plan_sha256"), fingerprints["frozen-plan.json"], "manifest plan binding")
    same(plan.get("selection_uses_predictions_or_mask_pixels"), False, "no result-based selection")
    same(data.get("categories"), [{"id": 1, "name": SIDES[0]}, {"id": 2, "name": SIDES[1]}], "side categories")
    images = sorted(data.get("images", []), key=lambda row: integer(row.get("id"), "image ID", 1))
    if len(images) != expected_images or len({row["id"] for row in images}) != len(images):
        raise ValueError("Full image coverage differs from expected immutable dataset")
    expected_identity = [(name, frame) for name, count in sorted(plan.get("frame_counts", {}).items())
                         for frame in range(integer(count, "recording frame count", 1))]
    same([(row["recording_id"], row["source_frame_index"]) for row in images], expected_identity, "all original frame coverage")
    by_id, identities, legacy, references = {}, set(), {}, {}
    outputs = manifest.get("image_outputs", [])
    output_index = {row["image_id"]: row for row in outputs}
    if len(output_index) != len(outputs) or set(output_index) != {row["id"] for row in images}:
        raise ValueError("Manifest image identity coverage differs")
    for image in images:
        image_id = image["id"]
        same(image.get("source_dataset"), "realsense", "original source dataset")
        shape = (integer(image.get("height"), "height", 1), integer(image.get("width"), "width", 1))
        if shape != (480, 640):
            raise ValueError("Full RealSense requires original 480x640 references")
        provided = image.get("reference_provided")
        if not isinstance(provided, dict) or set(provided) != set(SIDES) or any(type(v) is not bool for v in provided.values()):
            raise ValueError("Every side must explicitly declare reference_provided")
        recording = image.get("recording_id")
        if not isinstance(recording, str) or not recording:
            raise ValueError("Missing recording identity")
        identity = (recording, integer(image.get("source_frame_index"), "source frame"))
        if identity in identities or not isinstance(image.get("source_mapping"), dict):
            raise ValueError("Duplicate or incomplete source frame identity")
        identities.add(identity)
        flags = image.get("quality_flags")
        if not isinstance(flags, list) or any(not isinstance(v, str) or not v for v in flags) or len(set(flags)) != len(flags):
            raise ValueError("Quality flags must be explicit unique strings")
        side_flags = image.get("reference_quality_flags")
        if (not isinstance(side_flags, dict) or set(side_flags) != set(SIDES)
                or any(not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
                       or len(set(values)) != len(values) for values in side_flags.values())
                or set(flags) != set(side_flags[SIDES[0]] + side_flags[SIDES[1]])
                or any("previously_displayed_random_review" in values for values in side_flags.values())):
            raise ValueError("Require side-specific quality flags; previous display is not a quality error")
        group = "both" if all(provided.values()) else "left_only" if provided[SIDES[0]] else "right_only" if provided[SIDES[1]] else "none"
        same(image.get("raw_provided_group"), group, "raw provided group")
        same(image.get("has_both_reference"), all(provided.values()), "has both references")
        if type(image.get("legacy_fixed128")) is not bool or type(image.get("render_preselected")) is not bool:
            raise ValueError("Legacy/render flags must be explicit booleans")
        old_id = image.get("legacy_fixed128_image_id")
        if image["legacy_fixed128"]:
            integer(old_id, "legacy image ID", 1)
            if old_id in legacy or not all(provided.values()):
                raise ValueError("Legacy fixed128 identities must be unique with both references")
            legacy[old_id] = image_id
        elif old_id is not None:
            raise ValueError("Nonlegacy image cannot claim a fixed128 ID")
        output = output_index[image_id]
        for field in (*IDENTITY_FIELDS, "reference_provided", "quality_flags", "reference_quality_flags",
                      "raw_provided_group", "legacy_fixed128", "legacy_fixed128_image_id"):
            same(output.get(field), image[field], f"manifest {field}")
        files = output.get("files", {})
        if set(files) != {"rgb", *(side for side in SIDES if provided[side])}:
            raise ValueError("Manifest files must contain exactly RGB and actually provided side streams")
        same(files["rgb"].get("path"), image.get("file_name"), "RGB path")
        for entry in files.values():
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe dataset manifest path")
            digest(entry.get("sha256"), "dataset file")
        references[image_id] = {side: None for side in SIDES}
        by_id[image_id] = image
    if set(legacy) != set(range(1, expected_legacy_images + 1)):
        raise ValueError("Legacy fixed128 mapping is incomplete")
    seen_annotations = set()
    for annotation in data.get("annotations", []):
        ann_id = integer(annotation.get("id"), "annotation ID", 1)
        image_id = integer(annotation.get("image_id"), "annotation image ID", 1)
        category = integer(annotation.get("category_id"), "category", 1)
        if ann_id in seen_annotations or image_id not in by_id or category not in (1, 2):
            raise ValueError("Unknown or duplicate annotation identity")
        seen_annotations.add(ann_id)
        side = SIDES[category - 1]
        if not by_id[image_id]["reference_provided"][side] or references[image_id][side] is not None:
            raise ValueError("Missing reference cannot acquire an annotation; require one semantic union per side")
        mask = decode_rle(annotation.get("segmentation"), (480, 640))
        if not mask.any():
            raise ValueError("Empty annotation is not a positive semantic reference")
        references[image_id][side] = annotation["segmentation"]
    contract = {"root": root, "images": images, "by_id": by_id, "references": references,
                "fingerprints": fingerprints, "manifest": manifest, "plan": plan, "legacy": legacy}
    # Validate only one original image/side at a time: no full-dataset mask allocation.
    for image in images:
        decoded = {}
        for name, entry in output_index[image["id"]]["files"].items():
            path = root / entry["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError("Dataset asset escapes its publication")
            if sha256(path) != entry["sha256"]:
                raise ValueError("Dataset RGB/reference asset changed")
            if name != "rgb":
                with Image.open(path) as mask_image:
                    array = np.asarray(mask_image)
                expected = reference_mask(contract, image["id"], name)
                if array.shape != expected.shape or not np.array_equal(array > 0, expected):
                    raise ValueError("Raw reference PNG differs from provided-side RLE union")
                decoded[name] = expected
        overlap = int((decoded[SIDES[0]] & decoded[SIDES[1]]).sum()) if len(decoded) == 2 else None
        same(output_index[image["id"]].get("left_right_overlap_pixels"), overlap, "reference overlap")
    return contract


def reference_mask(contract, image_id, side):
    image = contract["by_id"][image_id]
    if not image["reference_provided"][side]:
        return None
    rle = contract["references"][image_id][side]
    return np.zeros((image["height"], image["width"]), dtype=bool) if rle is None else decode_rle(rle, (image["height"], image["width"]))


def _boundary(mask):
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=4, border_value=0)


def measure(candidate, reference, other, confidence):
    confidence = number(confidence, "confidence", True)
    detected = confidence >= .5
    result = {"detected": detected, "reference_provided": reference is not None,
              "other_reference_provided": other is not None, "target_present": None,
              "reference_pixels": None, "candidate_pixels": int(candidate.sum()),
              "false_negative": None, "absent_false_positive": None,
              "opposite_overlap_dominant_proxy": None,
              "detected_opposite_overlap_dominant_proxy": None,
              **dict.fromkeys(METRIC_FIELDS)}
    if reference is None:
        return result
    if reference.shape != candidate.shape or (other is not None and other.shape != candidate.shape):
        raise ValueError("Candidate/reference shape mismatch")
    positive = bool(reference.any())
    result.update(target_present=positive, reference_pixels=int(reference.sum()),
                  false_negative=bool(positive and not detected),
                  absent_false_positive=bool(not positive and detected))
    intersection, union = int((candidate & reference).sum()), int((candidate | reference).sum())
    own_iou = intersection / union if union else 1.
    if positive:
        dice = 2 * intersection / (int(candidate.sum()) + int(reference.sum()))
        a, b = _boundary(candidate), _boundary(reference)
        boundary = int((a & b).sum()) / int((a | b).sum())
        result.update(candidate_dice=dice, candidate_iou=own_iou,
                      miss_zero_dice=dice if detected else 0., miss_zero_iou=own_iou if detected else 0.,
                      candidate_boundary_iou_4px=boundary,
                      miss_zero_boundary_iou_4px=boundary if detected else 0.)
    if other is not None:
        overlap = int((candidate & other).sum())
        other_union = int((candidate | other).sum())
        other_iou = overlap / other_union if other_union else 1.
        proxy = bool(other.any() and overlap > 0 and other_iou > own_iou)
        result.update(opposite_overlap_dominant_proxy=proxy,
                      detected_opposite_overlap_dominant_proxy=bool(detected and proxy))
    return result


def aggregate(rows):
    provided = [row for row in rows if row["reference_provided"]]
    positive = [row for row in provided if row["target_present"]]
    absent = [row for row in provided if not row["target_present"]]
    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None
    fp, fn = sum(row["absent_false_positive"] for row in absent), sum(row["false_negative"] for row in positive)
    return {"images": len({row["image_id"] for row in rows}), "queries": len(rows),
            "provided_queries": len(provided), "missing_reference_queries": len(rows) - len(provided),
            "positive_queries": len(positive), "absent_queries": len(absent),
            "false_negative_queries": fn if provided else None,
            "absent_false_positive_queries": fp if provided else None,
            "false_negative_rate": fn / len(positive) if positive else None,
            "absent_false_positive_rate": fp / len(absent) if absent else None,
            **{key: mean(row[key] for row in positive) for key in METRIC_FIELDS},
            "opposite_proxy_eligible_queries": sum(row["detected_opposite_overlap_dominant_proxy"] is not None for row in rows),
            "detected_opposite_overlap_dominant_proxy_rate": mean(row["detected_opposite_overlap_dominant_proxy"] for row in rows)}


def grouped(rows):
    recordings = {name: aggregate([r for r in rows if r["recording_id"] == name])
                  for name in sorted({r["recording_id"] for r in rows})}
    values = [r["miss_zero_dice"] for r in recordings.values() if r["miss_zero_dice"] is not None]
    return {"overall": aggregate(rows), "per_side": {side: aggregate([r for r in rows if r["prompt_key"] == side]) for side in SIDES},
            "per_recording": recordings, "recording_macro_miss_zero_dice": sum(values) / len(values) if values else None}


def scopes(rows):
    primary = []
    for row in rows:
        if row["reference_provided"] and not row["reference_quality_flags"]:
            clean = dict(row)
            if row["pair_quality_flags"]:
                clean["opposite_overlap_dominant_proxy"] = None
                clean["detected_opposite_overlap_dominant_proxy"] = None
            primary.append(clean)
    return {"all_predictions_coverage": grouped(rows),
            "raw_all_provided": grouped([r for r in rows if r["reference_provided"]]),
            "primary_quality_eligible": grouped(primary),
            "knownwrong_or_uncertain_raw": grouped([r for r in rows if r["reference_provided"] and r["reference_quality_flags"]]),
            "missing_reference_predictions_only": grouped([r for r in rows if not r["reference_provided"]]),
            "legacy_fixed128": grouped([r for r in rows if r["legacy_fixed128"]]),
            "per_quality_flag": {flag: grouped([r for r in rows if flag in r["reference_quality_flags"]])
                                 for flag in sorted({flag for r in rows for flag in r["reference_quality_flags"]})},
            "per_raw_provided_group": {name: grouped([r for r in rows if r["raw_provided_group"] == name])
                                       for name in ("both", "left_only", "right_only", "none")}}


def _model_progress(metadata, group, expected_sha, expected_epoch=1):
    if type(expected_epoch) is not int or expected_epoch not in (1, 2):
        raise ValueError("Only explicit completed epoch 1 or 2 is supported")
    steps_per_epoch = {"mixed": 5021, "nake": 3082}[group]
    expected_step = steps_per_epoch * expected_epoch
    same(metadata.get("checkpoint_sha256"), digest(expected_sha, "preselected checkpoint"), "preselected checkpoint SHA")
    progress, config = metadata.get("progress", {}), metadata.get("training_config", {})
    for key, value in {"global_step": expected_step, "completed_epochs": expected_epoch,
                       "next_epoch": expected_epoch, "next_step_in_epoch": 0, "samples_seen": expected_step * 6}.items():
        same(progress.get(key), value, f"actual epoch{expected_epoch} {key}")
    same(config.get("steps_per_epoch"), steps_per_epoch, "training steps per epoch")
    same(config.get("global_batch_size"), 6, "actual training global batch")
    if integer(config.get("dataset_size"), "training dataset size", 1) // 6 != steps_per_epoch:
        raise ValueError("Training dataset size differs from actual per-epoch exposure")
    if integer(config.get("epochs"), "planned training epochs", 1) < expected_epoch:
        raise ValueError("Selected completed epoch exceeds planned training epochs")
    for key in ("base_sha256", "tokenizer_sha256", "initial_cache_file_sha256", "annotations_sha256"):
        digest(config.get(key), key)
    for key in ("initial_cache_verified", "actual_training_identities_verified"):
        same(metadata.get(key), True, key)
    return {"checkpoint_sha256": expected_sha, "actual_global_step": expected_step, "actual_completed_epochs": expected_epoch,
            "actual_image_exposures": expected_step * 6, "planned_epochs": config["epochs"],
            "original_metadata": metadata}


def _inference_contract(summary, contract, indices):
    for key, value in {"format": EVALUATION_FORMAT, "status": "complete", "dataset_role": "external_test_only",
                       "evaluation_scope": "fixed_development_benchmark",
                       "actual_complete_query_coverage_verified": True,
                       "whole_dataset_images": len(contract["images"]), "detection_threshold": .5,
                       "mask_threshold": .5, "boundary_width_original_pixels": 4,
                       "precision": "BF16 autocast; FP32 logits for sigmoid/interpolation; FP32 delta",
                       "prediction_selection": "argmax(sigmoid(class)*sigmoid(presence)); no reference-dependent selection",
                       "shard_rule": "sorted whole-image ID position modulo shard_count"}.items():
        same(summary.get(key), value, key)
    same(summary.get("protocol"), contract["plan"], "full dataset protocol")
    same(summary.get("dataset_fingerprints"), contract["fingerprints"], "full dataset fingerprints")
    inference = summary.get("source_inference_sha256")
    if not isinstance(inference, dict) or not inference:
        raise ValueError("Missing inference source fingerprints")
    required = {"scripts/evaluate_realsense_full.py", "scripts/evaluate_residual_test.py",
                "scripts/evaluate_nakehand_tokens.py", "scripts/evaluate_bilateral_tokens.py",
                "scripts/cached_ve_text_features.py", "scripts/residual_ddp_validation.py"}
    if not required <= inference.keys() or not any(path.startswith("sam3/") for path in inference):
        raise ValueError("Incomplete source inference fingerprint coverage")
    for path, value in inference.items():
        if Path(path).is_absolute() or ".." in Path(path).parts or not path.startswith(("scripts/", "sam3/")):
            raise ValueError("Unsafe inference source logical path")
        digest(value, path)
    implementation = summary.get("implementation_sha256", {})
    normalized = {}
    for name, value in implementation.items():
        parts = Path(name).parts
        positions = [i for i, part in enumerate(parts) if part in ("sam3", "scripts")]
        if not positions:
            raise ValueError("Unrecognized implementation source path")
        logical = "/".join(parts[positions[0]:])
        if logical in normalized:
            raise ValueError("Duplicate implementation logical source")
        normalized[logical] = digest(value, name)
    same(inference, normalized, "declared inference versus actual implementation fingerprints")
    sources = summary.get("source_fingerprints", {})
    if not isinstance(sources, dict):
        raise ValueError("Missing actual input fingerprints")
    for path, value in sources.items():
        digest(value, path)
    roots = [Path(path).parent for path, value in sources.items()
             if Path(path).name == "READY.json" and value == contract["fingerprints"]["READY.json"]]
    if len(roots) != 1:
        raise ValueError("Ambiguous or missing actual dataset input root")
    expected = {str(roots[0] / name): value for name, value in contract["fingerprints"].items()}
    output_index = {row["image_id"]: row for row in contract["manifest"]["image_outputs"]}
    for index in indices:
        for asset in output_index[contract["images"][index]["id"]]["files"].values():
            expected[str(roots[0] / asset["path"])] = asset["sha256"]
    for path, value in expected.items():
        same(sources.get(path), value, "actual source asset SHA")
    return inference, [value for path, value in sources.items() if path not in expected]


def _recompute_record(record, image, index, mode, contract):
    side = record.get("prompt_key")
    if side not in SIDES:
        raise ValueError("Unknown query side")
    other_side = SIDES[1 - SIDES.index(side)]
    own_flags = image["reference_quality_flags"][side]
    pair_flags = sorted(set(own_flags + image["reference_quality_flags"][other_side]))
    expected = {"model": mode, "mode": mode, "dataset_role": "external_test_only", "identity_verified": True,
                "image_id": image["id"], "dataset_index": index, "file_name": image["file_name"],
                "prompt_text": ("left hand", "right hand")[SIDES.index(side)],
                **{key: image[key] for key in IDENTITY_FIELDS},
                "reference_provided": image["reference_provided"][side],
                "other_reference_provided": image["reference_provided"][other_side],
                "has_both_reference": image["has_both_reference"], "raw_provided_group": image["raw_provided_group"],
                "quality_flags": image["quality_flags"], "reference_quality_flags": own_flags,
                "pair_quality_flags": pair_flags, "quality_flag_scope": "side",
                "primary_test": image["reference_provided"][side] and not own_flags,
                "legacy_fixed128": image["legacy_fixed128"], "legacy_fixed128_image_id": image["legacy_fixed128_image_id"]}
    for key, value in expected.items():
        same(record.get(key), value, f"record {image['id']}/{side}/{key}")
    score = number(record.get("top_confidence"), "confidence", True)
    probability = number(record.get("top_class_probability"), "class probability", True)
    presence = number(record.get("presence_probability"), "presence probability", True)
    product = float(np.float32(probability) * np.float32(presence))
    if not math.isclose(score, product, abs_tol=1e-7, rel_tol=1e-7) or (score >= .5) != (product >= .5):
        raise ValueError("Selected confidence differs from class-times-presence")
    integer(record.get("selected_decoder_query"), "selected decoder query")
    detections = integer(record.get("detections_above_threshold"), "detection count")
    if (detections > 0) != (score >= .5):
        raise ValueError("Detection count contradicts maximum score")
    candidate = decode_rle(record.get("prediction_rle"), (image["height"], image["width"]))
    reference = reference_mask(contract, image["id"], side)
    other = reference_mask(contract, image["id"], other_side)
    measured = measure(candidate, reference, other, score)
    aliases = {"candidate_dice": "top_dice", "candidate_iou": "top_iou", "candidate_pixels": "top_mask_pixels"}
    for key in ("detected", "target_present", "reference_pixels", "candidate_pixels", *METRIC_FIELDS,
                "opposite_overlap_dominant_proxy", "detected_opposite_overlap_dominant_proxy"):
        value, supplied = measured[key], record.get(aliases.get(key, key))
        if isinstance(value, float):
            if supplied is None or not math.isclose(number(supplied, key), value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Record metric differs from RLE recomputation: {key}")
        else:
            same(supplied, value, f"record recomputed {key}")
    return {**expected, "prompt_key": side, "top_confidence": score, **measured}


def load_shard(path, contract):
    path = Path(path).resolve(strict=True)
    summary, summary_sha = read_json(path)
    mode = summary.get("mode")
    emitted = (SIDES[0],) if mode == "ve-left" else (SIDES[1],) if mode == "ve-right" else SIDES if mode in ("ve-both", "residual") else ()
    if not emitted:
        raise ValueError("Unsupported inference mode")
    count = integer(summary.get("shard_count"), "shard count", 1)
    index = integer(summary.get("shard_index"), "shard index")
    if index >= count:
        raise ValueError("Shard index must be smaller than shard count")
    indices = list(range(index, len(contract["images"]), count))
    expected_ids = [contract["images"][i]["id"] for i in indices]
    same(summary.get("image_ids"), expected_ids, "exact modulo-shard image IDs")
    same(summary.get("shard_images"), len(indices), "shard image count")
    same(summary.get("emitted_sides"), list(emitted), "emitted query sides")
    same(summary.get("expected_emitted_queries"), len(indices) * len(emitted), "emitted query count")
    same(summary.get("actual_inference_queries_per_image"), 2, "actual query computation count")
    same(summary.get("expected_actual_inference_queries"), len(indices) * 2, "actual inference query count")
    inference, inputs = _inference_contract(summary, contract, indices)
    if summary.get("records_file") != "records.jsonl":
        raise ValueError("Require explicit local records.jsonl")
    records_path = path.parent / "records.jsonl"
    if records_path.is_symlink():
        raise ValueError("Records may not redirect to another run")
    record_sha = sha256(records_path)
    same(summary.get("records_sha256"), record_sha, "record file SHA")
    rows, seen = [], set()
    positions = {contract["images"][i]["id"]: i for i in indices}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("Blank or unfinished record in completed JSONL")
            record = loads(line)
            if not isinstance(record, dict):
                raise ValueError("Query record must be an object")
            image_id = integer(record.get("image_id"), "record image ID", 1)
            key = image_id, record.get("prompt_key")
            if image_id not in positions or key[1] not in emitted or key in seen:
                raise ValueError("Wrong shard, side, unknown or duplicate query record")
            seen.add(key)
            rows.append(_recompute_record(record, contract["by_id"][image_id], positions[image_id], mode, contract))
    if seen != {(image_id, side) for image_id in expected_ids for side in emitted}:
        raise ValueError("Incomplete shard query coverage")
    same(sha256(records_path), record_sha, "records unchanged during aggregation")
    return {"path": path, "summary": summary, "rows": rows, "inference": inference, "model_inputs": inputs,
            "input_sha256": {str(path): summary_sha, str(records_path): record_sha}}


def compare(data_root, *, ve_left_summaries, ve_right_summaries, mixed_summaries, nake_summaries,
            mixed_checkpoint_sha256, nake_checkpoint_sha256, expected_images=6204, expected_legacy_images=128,
            expected_nake_epoch=1):
    if type(expected_nake_epoch) is not int or expected_nake_epoch not in (1, 2):
        raise ValueError("Explicit nake epoch must be 1 or 2")
    digest(mixed_checkpoint_sha256, "explicit mixed epoch1 checkpoint")
    digest(nake_checkpoint_sha256, f"explicit nake epoch{expected_nake_epoch} checkpoint")
    contract = load_contract(data_root, expected_images=expected_images, expected_legacy_images=expected_legacy_images)
    specifications = {"ve-left": ve_left_summaries, "ve-right": ve_right_summaries,
                      "mixed": mixed_summaries, "nake": nake_summaries}
    cache, all_rows, metadata, input_hashes, common_inference, common_base = {}, {}, {}, {}, None, None
    for group, paths in specifications.items():
        if not paths:
            raise ValueError(f"Missing all shards for {group}; incomplete comparisons are not complete results")
        shards, indices, count, model_metadata = [], set(), None, None
        for path in paths:
            path = Path(path).resolve(strict=True)
            if path not in cache:
                cache[path] = load_shard(path, contract)
            shard = cache[path]
            summary = shard["summary"]
            allowed = (group, "ve-both") if group.startswith("ve-") else ("residual",)
            if summary["mode"] not in allowed:
                raise ValueError("Wrong model mode supplied to comparison group")
            current_count = summary["shard_count"]
            if count is not None and count != current_count or summary["shard_index"] in indices:
                raise ValueError("Duplicate shard index or changed shard-count configuration")
            count = current_count
            indices.add(summary["shard_index"])
            model = summary.get("model_metadata")
            if not isinstance(model, dict):
                raise ValueError("Missing model provenance")
            logical_model = {key: value for key, value in model.items() if key != "checkpoint_path"}
            if model_metadata is not None:
                same(logical_model, model_metadata, "model/checkpoint metadata across shards")
            model_metadata = logical_model
            if common_inference is not None:
                same(shard["inference"], common_inference, "inference source across models/shards")
            common_inference = shard["inference"]
            if group.startswith("ve-"):
                same(sorted(model), sorted(("source", "base_checkpoint_sha256", "tokenizer_sha256")), "VE provenance fields")
                same(model["source"], "actual original VE natural prompts; no cached replacement", "original VE metadata")
                if len(shard["model_inputs"]) != 2:
                    raise ValueError("VE must bind only its original base weights and tokenizer outside dataset assets")
                current_base = sorted(shard["model_inputs"])
                same(current_base, sorted([digest(model["base_checkpoint_sha256"], "VE base"),
                                           digest(model["tokenizer_sha256"], "VE tokenizer")]), "VE actual model fingerprints")
            else:
                expected_sha = mixed_checkpoint_sha256 if group == "mixed" else nake_checkpoint_sha256
                metadata[group] = _model_progress(model, group, expected_sha,
                                                  expected_nake_epoch if group == "nake" else 1)
                note = summary.get("checkpoint_selection_note")
                if not isinstance(note, str) or not note.strip():
                    raise ValueError("Missing predeclared checkpoint selection note")
                if shards:
                    same(note, shards[0]["summary"].get("checkpoint_selection_note"), "selection note across shards")
                metadata[group]["checkpoint_selection_note"] = note
                metadata[group]["best_note"] = model.get("best_note", summary.get("best_note"))
                config = model["training_config"]
                same(model.get("base_checkpoint_sha256"), config["base_sha256"], "residual base fingerprint")
                same(model.get("tokenizer_sha256"), config["tokenizer_sha256"], "residual tokenizer fingerprint")
                expected_inputs = [expected_sha] + [config[key] for key in (
                    "base_sha256", "tokenizer_sha256", "initial_cache_file_sha256", "annotations_sha256")]
                same(sorted(shard["model_inputs"]), sorted(expected_inputs), "actual checkpoint/base/cache/training input SHA")
                current_base = sorted([config["base_sha256"], config["tokenizer_sha256"]])
            if common_base is not None:
                same(current_base, common_base, "base weights/tokenizer across model conditions")
            common_base = current_base
            input_hashes.update(shard["input_sha256"])
            shards.append(shard)
        if indices != set(range(count)):
            raise ValueError(f"Missing shards for {group}; refuse partial accuracy report")
        side_filter = SIDES[:1] if group == "ve-left" else SIDES[1:] if group == "ve-right" else SIDES
        rows = [row for shard in shards for row in shard["rows"] if row["prompt_key"] in side_filter]
        keys = [(row["image_id"], row["prompt_key"]) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != {(image["id"], side) for image in contract["images"] for side in side_filter}:
            raise ValueError("Model-wide image/side coverage is incomplete or duplicated")
        all_rows[group] = sorted(rows, key=lambda row: (row["image_id"], row["prompt_key"]))
    metrics = {group: scopes(rows) for group, rows in all_rows.items()}
    differences = {}
    for group in ("mixed", "nake"):
        differences[group] = {}
        for side, baseline in zip(SIDES, ("ve-left", "ve-right")):
            differences[group][side] = {}
            for scope in ("raw_all_provided", "primary_quality_eligible", "legacy_fixed128"):
                original = metrics[baseline][scope]["per_side"][side]
                residual = metrics[group][scope]["per_side"][side]
                for key in ("queries", "provided_queries", "positive_queries", "absent_queries"):
                    same(residual[key], original[key], "paired comparison reference denominator")
                differences[group][side][scope] = {key: None if residual[key] is None or original[key] is None else residual[key] - original[key]
                    for key in (*METRIC_FIELDS, "false_negative_queries", "absent_false_positive_queries",
                                "false_negative_rate", "absent_false_positive_rate", "detected_opposite_overlap_dominant_proxy_rate")}
                ve_rows = [r for r in all_rows[baseline] if (scope != "legacy_fixed128" or r["legacy_fixed128"])]
                residual_rows = [r for r in all_rows[group] if r["prompt_key"] == side
                                 and (scope != "legacy_fixed128" or r["legacy_fixed128"])]
                if scope == "primary_quality_eligible":
                    ve_rows = [r for r in ve_rows if r["reference_provided"] and not r["reference_quality_flags"]]
                    residual_rows = [r for r in residual_rows if r["reference_provided"] and not r["reference_quality_flags"]]
                else:
                    ve_rows = [r for r in ve_rows if r["reference_provided"]]
                    residual_rows = [r for r in residual_rows if r["reference_provided"]]
                by_recording = {}
                for recording in sorted({r["recording_id"] for r in ve_rows}):
                    a = aggregate([r for r in ve_rows if r["recording_id"] == recording])
                    b = aggregate([r for r in residual_rows if r["recording_id"] == recording])
                    same([a[k] for k in ("provided_queries", "positive_queries", "absent_queries")],
                         [b[k] for k in ("provided_queries", "positive_queries", "absent_queries")], "per-recording paired denominators")
                    by_recording[recording] = {key: None if a[key] is None or b[key] is None else b[key]-a[key]
                                              for key in (*METRIC_FIELDS, "false_negative_queries", "absent_false_positive_queries")}
                differences[group][side][scope]["per_recording"] = by_recording
    for name, value in contract["fingerprints"].items():
        input_hashes[str(contract["root"] / name)] = value
    for path, value in input_hashes.items():
        same(sha256(path), value, "comparison inputs unchanged")
    return {"format": FORMAT, "status": "complete", "dataset_role": "external_test_only",
            "evaluation_scope": "fixed_development_benchmark", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "images": len(contract["images"]), "legacy_fixed128_images": len(contract["legacy"]),
            "reference_overlap_images": sum(bool(row["left_right_overlap_pixels"]) for row in contract["manifest"]["image_outputs"]),
            "record_coverage": {key: len(value) for key, value in all_rows.items()},
            "reference_description": REFERENCE_DESCRIPTION, "model_progress": metadata,
            "metrics_recomputed_from_saved_rle": metrics, "difference_residual_minus_corresponding_ve_side": differences,
            "input_sha256": input_hashes, "source_inference_sha256": common_inference,
            "thresholds": {"detection": .5, "mask": .5, "boundary_original_pixels": 4},
            "comparison_source_sha256": sha256(Path(__file__)),
            "limitations": ["全量包含曾查看帧、已知错侧和不确定参考，是固定开发基准，不是新盲测或全人工精标准确率。",
                "raw保留每个已提供参考；质量敏感主指标只排本侧已记录问题，错侧代理额外要求两侧参考均已提供且无问题。",
                "缺失参考的Dice/IoU/边界/FN/FP均不适用，不得当作全黑负例；原始提供但全黑的参考才进入absent分母。",
                "旧128帧按原映射单列重聚合，没有重新挑选或改变阈值；已看过的开发帧不包装成独立新增测试样本。",
                f"mixed epoch1=5021更新/30126次图像曝光；nake epoch{expected_nake_epoch}="
                f"{3082 * expected_nake_epoch}更新/{18492 * expected_nake_epoch}次曝光，不是等数据量或等更新数对照。",
                "临近帧相关，不作独立样本的假精确置信区间；不根据本结果挑checkpoint、调参或挑展示图。",
                "RLE证明所存候选与参考的一致性；没有完整logits，不能单靠RLE重新证明argmax或像素阈值执行。"]}


def markdown_report(result):
    def fmt(value):
        return "不适用" if value is None else str(value) if type(value) is int else f"{value:.5f}"
    lines = ["# 全量 RealSense 固定开发基准对比", "", f"完成覆盖 {result['images']} 图；四组输出查询数：{result['record_coverage']}。",
             "指标由所有明确指定分片的候选RLE重算；没有用summary的均值替代逐条核验。", "",
             "| 模型 | 实际完成epoch | 更新数 | 图像曝光次数 |", "|---|---:|---:|---:|"]
    for group, progress in result["model_progress"].items():
        lines += [f"| {group} | {progress['actual_completed_epochs']} | {progress['actual_global_step']} | {progress['actual_image_exposures']} |"]
    lines += ["", "两种训练完成点不等规模；选定checkpoint和原始选点说明保存在JSON，未用best替代。", ""]
    for scope, title in (("raw_all_provided", "所有实际提供的原始参考"),
                         ("primary_quality_eligible", "本侧质量敏感主指标"), ("legacy_fixed128", "旧固定128单列")):
        lines += [f"## {title}", "", "| 条件/侧 | 正参考数 | 候选Dice | 漏检置零Dice | 候选IoU | Boundary4px | FN | 空侧FP/空侧数 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for group, scopes_value in result["metrics_recomputed_from_saved_rle"].items():
            for side, values in scopes_value[scope]["per_side"].items():
                if not values["queries"]:
                    continue
                lines.append(f"| {group}/{side} | {values['positive_queries']} | {fmt(values['candidate_dice'])} | "
                    f"{fmt(values['miss_zero_dice'])} | {fmt(values['candidate_iou'])} | {fmt(values['candidate_boundary_iou_4px'])} | "
                    f"{fmt(values['false_negative_queries'])} | {fmt(values['absent_false_positive_queries'])}/{values['absent_queries']} |")
        lines.append("")
    lines += ["完整按录像、左右侧、原始参考可用性和问题flag的统计，以及对应VE侧差值均在JSON。", "", *result["limitations"], ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for label in ("ve-left", "ve-right", "mixed", "nake"):
        parser.add_argument(f"--{label}-summary", type=Path, action="append", required=True)
    for label in ("mixed", "nake"):
        parser.add_argument(f"--{label}-checkpoint-sha256", required=True)
    parser.add_argument("--nake-expected-epoch", type=int, choices=(1, 2), default=1,
                        help="Explicit user-selected completed epoch; never inferred from best or test scores")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("Comparison output must be new; existing results cannot be overwritten")
    output = args.output_dir.resolve()
    input_roots = [args.data_root.resolve(), *(path.resolve().parent for paths in
        (args.ve_left_summary, args.ve_right_summary, args.mixed_summary, args.nake_summary) for path in paths)]
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in input_roots):
        raise ValueError("Comparison output must be separate from immutable datasets and inference runs")
    result = compare(args.data_root, ve_left_summaries=args.ve_left_summary, ve_right_summaries=args.ve_right_summary,
                     mixed_summaries=args.mixed_summary, nake_summaries=args.nake_summary,
                     mixed_checkpoint_sha256=args.mixed_checkpoint_sha256, nake_checkpoint_sha256=args.nake_checkpoint_sha256,
                     expected_nake_epoch=args.nake_expected_epoch)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output_dir / "comparison.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown_report(result))
    print(json.dumps({"status": "complete", "output": str(args.output_dir.resolve()), "images": result["images"]}))


if __name__ == "__main__":
    main()

"""Deterministic independent-GPU shards of the complete RealSense reference test.

Missing side files are UNKNOWN references, never negative labels. VE side jobs
retain the established two-query inference path and emit only the requested
side. No threshold, decoder, frame or checkpoint is selected from test overlap.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
import torch
import torch.nn.functional as F

from scripts import evaluate_residual_test as fixed

shared, bilateral, semantic = fixed.shared, fixed.bilateral, fixed.semantic
cached, checkpoint, initializer = fixed.cached, fixed.checkpoint, fixed.initializer
FORMAT = "sam3-realsense-full-evaluation-v1"
DATA_FORMAT = "sam3-realsense-full-reference-test-v1"
SIDES = tuple(shared.CLASS_NAMES)


def selected_sides(mode):
    if mode == "ve-left":
        return (SIDES[0],)
    if mode == "ve-right":
        return (SIDES[1],)
    if mode in ("residual", "ve-both"):
        return SIDES
    raise ValueError("Unknown evaluation mode")


def shard_indices(images, index, count):
    if type(index) is not int or type(count) is not int or count < 1 or not 0 <= index < count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    ids = [row["id"] for row in images]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Whole-image identities must be unique and sorted before sharding")
    return list(range(index, len(images), count))


def local_file(root, relative):
    path = Path(relative)
    target = root / path
    if (path.is_absolute() or ".." in path.parts or target.is_symlink()
            or root not in target.resolve().parents):
        raise ValueError("Published asset must be a regular in-root file")
    return target


def quality_flags(image, side):
    mapping = image.get("reference_quality_flags")
    if mapping is None:
        return list(image["quality_flags"])
    if set(mapping) != set(SIDES) or any(not isinstance(mapping[key], list) for key in SIDES):
        raise ValueError("Invalid per-side reference quality flags")
    return list(mapping[side])


def load_publication(root):
    """Read only compressed metadata here; do not allocate all reference masks."""
    root = Path(root).resolve()
    names = ("READY.json", "manifest.json", "annotations.json", "frozen-plan.json")
    docs = {name: json.loads((root / name).read_text()) for name in names}
    fingerprints = {str(root / name): shared.sha256(root / name) for name in names}
    ready, manifest, data, plan = (docs[name] for name in names)
    if (ready.get("format") != DATA_FORMAT or ready.get("status") != "complete"
            or ready.get("dataset_role") != "external_test_only"
            or ready.get("evaluation_scope") != "fixed_development_benchmark"
            or manifest.get("status") != "complete"
            or manifest.get("sources_unchanged") is not True):
        raise ValueError("Require completed full RealSense external-test publication")
    for key, name in (("manifest_sha256", "manifest.json"), ("annotations_sha256", "annotations.json"),
                      ("frozen_plan_sha256", "frozen-plan.json")):
        if ready.get(key) != fingerprints[str(root / name)]:
            raise ValueError(f"Publication SHA mismatch: {name}")
    if (data.get("info", {}).get("dataset_role") != "external_test_only"
            or data["info"].get("evaluation_scope") != "fixed_development_benchmark"
            or data["info"].get("frozen_plan_sha256") != ready["frozen_plan_sha256"]
            or manifest.get("frozen_plan_sha256") != ready["frozen_plan_sha256"]
            or plan.get("selection_uses_predictions_or_mask_pixels") is not False
            or data.get("categories") != fixed.preparation.CATEGORIES):
        raise ValueError("Full-test identity/policy/category binding differs")
    images = sorted(data["images"], key=lambda row: row["id"])
    shard_indices(images, 0, 1)
    expected = [(name, frame) for name, count in sorted(plan["frame_counts"].items()) for frame in range(count)]
    if [(row["recording_id"], row["source_frame_index"]) for row in images] != expected:
        raise ValueError("Require every source frame exactly once, including missing/flagged references")
    if [row["id"] for row in images] != list(range(1, len(images) + 1)):
        raise ValueError("Published image IDs must be contiguous from one")
    outputs = {row["image_id"]: row for row in manifest["image_outputs"]}
    if len(outputs) != len(manifest["image_outputs"]) or set(outputs) != {row["id"] for row in images}:
        raise ValueError("Published file identity coverage mismatch")
    annotations = defaultdict(dict)
    seen = set()
    for row in data["annotations"]:
        if (row["id"] in seen or row["image_id"] not in outputs or row["category_id"] not in (1, 2)
                or row["category_id"] in annotations[row["image_id"]]):
            raise ValueError("Require unique side-semantic union annotations")
        seen.add(row["id"])
        annotations[row["image_id"]][row["category_id"]] = row
    for image in images:
        provided = image.get("reference_provided", {})
        if set(provided) != set(SIDES) or any(type(provided[side]) is not bool for side in SIDES):
            raise ValueError("Missing explicit reference availability")
        if (image.get("source_dataset") != "realsense" or image["width"] < 1 or image["height"] < 1
                or not isinstance(image.get("quality_flags"), list)
                or type(image.get("legacy_fixed128")) is not bool
                or type(image.get("render_preselected")) is not bool
                or image.get("has_both_reference") is not all(provided.values())):
            raise ValueError("Invalid source size/quality/group/visualization metadata")
        expected_group = "both" if all(provided.values()) else (
            "left_only" if provided[SIDES[0]] else "right_only" if provided[SIDES[1]] else "none")
        if image.get("raw_provided_group") != expected_group:
            raise ValueError("Reference completeness group differs")
        published = outputs[image["id"]]
        if (published["recording_id"], published["source_frame_index"]) != (
                image["recording_id"], image["source_frame_index"]):
            raise ValueError("Asset manifest/source identity differs")
        files = published["files"]
        if set(files) != {"rgb", *(side for side in SIDES if provided[side])}:
            raise ValueError("Missing streams must not be represented by synthetic empty files")
        if files["rgb"]["path"] != image["file_name"]:
            raise ValueError("RGB path mismatch")
        for asset in files.values():
            local_file(root, asset["path"])
            if not isinstance(asset.get("sha256"), str) or len(asset["sha256"]) != 64:
                raise ValueError("Missing asset SHA256")
        for category, side in enumerate(SIDES, 1):
            quality_flags(image, side)
            if not provided[side] and category in annotations[image["id"]]:
                raise ValueError("Unknown side may not have an annotation")
    return images, annotations, outputs, plan, fingerprints


def batch_references(root, images, indices, annotations, outputs, fingerprints):
    refs = {}
    for index in indices:
        image = images[index]
        shape = image["height"], image["width"]
        refs[image["id"]] = {side: None for side in SIDES}
        for name, asset in outputs[image["id"]]["files"].items():
            path = local_file(root, asset["path"])
            if shared.sha256(path) != asset["sha256"]:
                raise ValueError(f"Asset changed before evaluation: {path}")
            fingerprints[str(path)] = asset["sha256"]
            with Image.open(path) as decoded:
                array = np.asarray(decoded)
            if array.shape != (shape + (3,) if name == "rgb" else shape):
                raise ValueError("Invalid exported image dimensions/channels")
            if name != "rgb":
                reference = shared.decode_gt_mask(annotations[image["id"]].get(SIDES.index(name) + 1), *shape)
                if reference.shape != shape or not np.array_equal(reference, array > 0):
                    raise ValueError("RLE must equal the original positive-ID side PNG union")
                refs[image["id"]][name] = reference
    return refs


def measure_partial(candidate, reference, other, score):
    """Reuse established formulas, then remove every unknown-reference metric."""
    empty = np.zeros_like(candidate, dtype=bool)
    row = bilateral.measure_query(candidate, empty if reference is None else reference,
                                   empty if other is None else other, score)
    fixed.add_boundary(row, candidate, empty if reference is None else reference)
    if reference is None:
        for key in ("target_present", "reference_pixels", "top_dice", "top_iou", "miss_zero_dice",
                    "miss_zero_iou", "top_dice_with_own_reference", "top_iou_with_own_reference",
                    "top_dice_with_other_reference", "top_iou_with_other_reference",
                    "top_other_reference_intersection_pixels"):
            row[key] = None
    if other is None:
        for key in ("other_side_present", "other_reference_pixels", "top_dice_with_other_reference",
                    "top_iou_with_other_reference", "top_other_reference_intersection_pixels"):
            row[key] = None
    if reference is None or other is None:
        for key in ("any_hand_present", "both_hands_present", "opposite_overlap_dominant_proxy",
                    "detected_opposite_overlap_dominant_proxy"):
            row[key] = None
    return row


def group_metrics(rows, *, include_flagged_pair=False):
    present = [row for row in rows if row["target_present"] is True]
    absent = [row for row in rows if row["target_present"] is False]
    pairs = [row for row in rows if row["has_both_reference"] and
             (include_flagged_pair or not row["pair_quality_flags"])]
    other_visible = [row for row in pairs if row["other_side_present"] is True]
    result = {"images": len({row["image_id"] for row in rows}), "queries": len(rows),
        "provided_queries": len(present) + len(absent), "unknown_reference_queries": len(rows)-len(present)-len(absent),
        "present_queries": len(present), "absent_queries": len(absent),
        "detections_all_queries": sum(row["detected"] for row in rows),
        "true_positive_queries": sum(row["detected"] for row in present),
        "false_negative_queries": sum(not row["detected"] for row in present),
        "false_positive_queries": sum(row["detected"] for row in absent),
        "correct_side_detection_rate": shared.mean(row["detected"] for row in present),
        "absent_side_false_positive_rate": shared.mean(row["detected"] for row in absent),
        "known_both_reference_queries_for_side_proxy": len(pairs),
        "other_visible_queries": len(other_visible),
        "other_visible_detected_opposite_dominant_rate_per_all_queries": shared.mean(
            row["detected_opposite_overlap_dominant_proxy"] for row in other_visible),
        "other_visible_detected_opposite_dominant_queries": sum(
            row["detected_opposite_overlap_dominant_proxy"] for row in other_visible),
        "mean_top_confidence": shared.mean(row["top_confidence"] for row in rows)}
    for source, target in (("top_dice", "present_mean_candidate_dice"), ("top_iou", "present_mean_candidate_iou"),
                           ("miss_zero_dice", "present_mean_miss_zero_dice"), ("miss_zero_iou", "present_mean_miss_zero_iou"),
                           ("candidate_boundary_iou_4px", "candidate_boundary_iou_4px"),
                           ("miss_zero_boundary_iou_4px", "miss_zero_boundary_iou_4px")):
        result[target] = shared.mean(row[source] for row in present)
    return result


def summarize(records):
    main = [row for row in records if row["primary_test"]]
    groups = {"primary_provided_nonflagged": main,
        "raw_all_provided": [row for row in records if row["reference_provided"]],
        "primary_both_reference": [row for row in main if row["has_both_reference"]],
        "primary_partial_reference": [row for row in main if not row["has_both_reference"]],
        "primary_legacy_fixed128": [row for row in main if row["legacy_fixed128"]],
        "known_issue_provided": [row for row in records if row["reference_provided"] and row["reference_quality_flags"]],
        "unknown_reference_predictions_only": [row for row in records if not row["reference_provided"]]}
    result = {}
    for name, rows in groups.items():
        def metrics(subset):
            return group_metrics(subset, include_flagged_pair=name in ("raw_all_provided", "known_issue_provided"))
        result[name] = {"overall": metrics(rows),
            "per_side": {side: metrics([row for row in rows if row["prompt_key"] == side]) for side in SIDES},
            "per_recording": {rec: metrics([row for row in rows if row["recording_id"] == rec])
                              for rec in sorted({row["recording_id"] for row in records})}}
    return result


def render_one(root, output, image, side, reference, candidate, record):
    directory = output / "visuals" / f"image-{image['id']:06d}" / side
    directory.mkdir(parents=True, exist_ok=False)
    with Image.open(root / image["file_name"]) as source:
        rgb = source.convert("RGB")
    ref = (Image.new("RGB", rgb.size, (128, 128, 128)) if reference is None else
           Image.fromarray(reference.astype(np.uint8)*255).convert("RGB"))
    pred = Image.fromarray(candidate.astype(np.uint8)*255).convert("RGB")
    detected = pred if record["detected"] else Image.new("RGB", rgb.size, "black")
    entries = (("rgb", rgb, "RGB"), ("reference", ref, "REFERENCE UNKNOWN" if reference is None else
               "REFERENCE FLAGGED" if record["reference_quality_flags"] else "REFERENCE"),
               ("candidate", pred, "CANDIDATE"), ("detected", detected, f"DETECTED score={record['top_confidence']:.4f}"))
    canvas = Image.new("RGB", (rgb.width*4, rgb.height+44), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, panel, title) in enumerate(entries):
        panel.save(directory / f"{name}.png")
        canvas.paste(panel, (index*rgb.width, 44))
        draw.text((index*rgb.width+8, 8), title, fill="black")
    canvas.save(directory / "comparison.png")
    return {"image_id": image["id"], "recording_id": image["recording_id"], "source_frame_index": image["source_frame_index"],
            "side": side, "reference_provided": reference is not None, "comparison": str(directory / "comparison.png")}


def append_batch(handle, records):
    for row in records:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    handle.flush()  # Completed batches survive a later process-level exception/termination.


def evaluate(model, mode, dataset, images, indices, annotations, outputs, root, out, fingerprints, batch_size):
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api
    bilateral.validate_frozen_noninteractive_model(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full test forbids trainable parameters")
    prompts = SIDES if mode == "residual" else cached.NATURAL_PROMPTS
    emitted = selected_sides(mode)
    records, visuals = [], []
    with (out / "records.jsonl").open("x", encoding="utf-8") as handle:
        for number, batch_indices in enumerate(shared.batches(indices, batch_size), 1):
            refs = batch_references(root, images, batch_indices, annotations, outputs, fingerprints)
            samples = [dataset[index] for index in batch_indices]
            batch = collate_fn_api(samples, dict_key="test", with_seg_masks=True)["test"]
            bilateral.validate_batch_identity(batch, batch_indices, images)
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            batch.find_text_batch = list(prompts)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)[0]
            for key in ("pred_logits", "presence_logit_dec", "pred_masks"):
                if not bool(torch.isfinite(output[key]).all()):
                    raise RuntimeError("Nonfinite inference output")
            classes = output["pred_logits"].float().sigmoid().squeeze(-1)
            presence = output["presence_logit_dec"].float().sigmoid().reshape(len(classes), -1)[:, 0]
            confidence = classes * presence[:, None]
            best = confidence.argmax(dim=1)
            stage = batch.find_inputs[0]
            batch_records = []
            for row in range(len(best)):
                index = batch_indices[int(stage.img_ids[row])]
                image = images[index]
                prompt = int(stage.text_ids[row])
                side, other = SIDES[prompt], SIDES[1-prompt]
                if side not in emitted:
                    continue
                logits = F.interpolate(output["pred_masks"][row, best[row]][None, None].float(),
                    size=(image["height"], image["width"]), mode="bilinear", align_corners=False)[0, 0]
                candidate = logits.sigmoid().cpu().numpy() >= fixed.THRESHOLD
                reference = refs[image["id"]][side]
                own_flags, other_flags = quality_flags(image, side), quality_flags(image, other)
                rle = mask_utils.encode(np.asfortranarray(candidate.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                record = {"model": mode, "mode": mode, "dataset_role": "external_test_only", "dataset_index": index,
                    "image_id": image["id"], "identity_verified": True, "file_name": image["file_name"],
                    "recording_id": image["recording_id"], "source_frame_index": image["source_frame_index"],
                    "source_mapping": image["source_mapping"], "prompt_key": side,
                    "prompt_text": cached.NATURAL_PROMPTS[prompt], "reference_provided": reference is not None,
                    "other_reference_provided": refs[image["id"]][other] is not None,
                    "has_both_reference": image["has_both_reference"], "raw_provided_group": image["raw_provided_group"],
                    "quality_flags": image["quality_flags"], "reference_quality_flags": own_flags,
                    "pair_quality_flags": sorted(set(own_flags+other_flags)),
                    "quality_flag_scope": "side" if "reference_quality_flags" in image else "image-conservative",
                    "primary_test": reference is not None and not own_flags, "legacy_fixed128": image["legacy_fixed128"],
                    "legacy_fixed128_image_id": image.get("legacy_fixed128_image_id"),
                    "top_class_probability": float(classes[row, best[row]]), "presence_probability": float(presence[row]),
                    "selected_decoder_query": int(best[row]), "prediction_rle": rle,
                    "detections_above_threshold": int((confidence[row] >= fixed.THRESHOLD).sum()),
                    **measure_partial(candidate, reference, refs[image["id"]][other], float(confidence[row, best[row]]))}
                batch_records.append(record)
                if image["render_preselected"]:
                    visuals.append(render_one(root, out, image, side, reference, candidate, record))
            append_batch(handle, batch_records)
            records.extend({key: value for key, value in record.items() if key != "prediction_rle"} for record in batch_records)
            if number == 1 or number % 20 == 0:
                print(f"{mode}: completed {min(number*batch_size, len(indices))}/{len(indices)} shard images", flush=True)
            del batch, output, samples, classes, presence, confidence, best, refs, logits
    observed = [(row["image_id"], row["prompt_key"]) for row in records]
    expected = {(images[index]["id"], side) for index in indices for side in emitted}
    if len(observed) != len(expected) or set(observed) != expected or dataset.observed_indices != indices:
        raise RuntimeError("Actual image/query shard coverage mismatch")
    return records, visuals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path(__file__).resolve().parents[1]/"sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--mode", choices=("ve-left", "ve-right", "ve-both", "residual"), required=True)
    for name in ("delta-checkpoint", "initial-cache", "training-data-root"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--checkpoint-selection-note")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.28)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Require positive batch size and allocator fraction in (0,1]")
    if args.mode == "residual" and not all((args.delta_checkpoint, args.initial_cache, args.training_data_root, args.checkpoint_selection_note)):
        parser.error("Residual requires checkpoint, original cache, actual training root and selection note")
    if args.output_dir.exists():
        raise ValueError("Output must be a new directory; no overwrite or implicit resume")
    args.data_root = args.data_root.resolve()
    images, annotations, outputs, plan, fingerprints = load_publication(args.data_root)
    if (len(images) != 6204 or len(plan["frame_counts"]) != 10
            or sum(image["legacy_fixed128"] for image in images) != 128):
        raise ValueError("Full evaluation requires all 6204 frames/10 recordings and the 128 legacy identities")
    indices = shard_indices(images, args.shard_index, args.shard_count)
    if not indices:
        raise ValueError("Requested shard is empty")
    base_hash, tokenizer_hash = shared.sha256(args.base_checkpoint), shared.sha256(args.tokenizer_path)
    fingerprints.update({str(args.base_checkpoint): base_hash, str(args.tokenizer_path): tokenizer_hash})
    encoder, metadata = None, None
    if args.mode == "residual":
        for path in (args.delta_checkpoint, args.initial_cache, args.training_data_root/"annotations.json"):
            fingerprints[str(path)] = shared.sha256(path)
        encoder, metadata = fixed.load_residual(args.delta_checkpoint, args.initial_cache, args.training_data_root,
            base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    dependencies = {Path(__file__).resolve(), Path(inspect.getfile(fixed._boundary)).resolve(),
        *(Path(inspect.getfile(module)).resolve() for module in
          (fixed, cached, shared, bilateral, semantic, fixed.preparation, checkpoint, initializer))}
    dependencies.update((Path(__file__).resolve().parents[1]/"sam3").rglob("*.py"))
    dependencies.update(Path(__file__).resolve().parent.glob("residual_*.py"))
    implementation = {str(path): shared.sha256(path) for path in sorted(dependencies)}
    source_root = Path(__file__).resolve().parents[1]
    model_metadata = metadata if encoder is not None else {
        "source": "actual original VE natural prompts; no cached replacement"}
    model_metadata.update(base_checkpoint_sha256=base_hash, tokenizer_sha256=tokenizer_hash)
    if encoder is not None:
        model_metadata["checkpoint_path"] = str(args.delta_checkpoint.resolve())
    summary = {"format": FORMAT, "status": "preflight_only" if args.preflight_only else "running",
        "dataset_role": "external_test_only", "evaluation_scope": "fixed_development_benchmark",
        "mode": args.mode, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "whole_dataset_images": len(images), "shard_images": len(indices), "shard_index": args.shard_index,
        "shard_count": args.shard_count, "shard_rule": "sorted whole-image ID position modulo shard_count",
        "image_ids": [images[index]["id"] for index in indices], "emitted_sides": list(selected_sides(args.mode)),
        "expected_emitted_queries": len(indices)*len(selected_sides(args.mode)), "actual_inference_queries_per_image": 2,
        "expected_actual_inference_queries": len(indices)*2, "batch_size": args.batch_size,
        "detection_threshold": .5, "mask_threshold": .5, "boundary_width_original_pixels": 4,
        "precision": "BF16 autocast; FP32 logits for sigmoid/interpolation; FP32 delta",
        "prediction_selection": "argmax(sigmoid(class)*sigmoid(presence)); no reference-dependent selection",
        "protocol": plan, "source_fingerprints": fingerprints, "implementation_sha256": implementation,
        "dataset_fingerprints": {Path(path).name: value for path, value in fingerprints.items()
                                 if Path(path).parent == args.data_root},
        "source_inference_sha256": {str(Path(path).relative_to(source_root)): value for path, value in implementation.items()},
        "checkpoint_selection_note": args.checkpoint_selection_note,
        "model_metadata": model_metadata, "reference_description": plan.get("reference_description"),
        "limitations": ["Auxiliary-reference agreement, not independent manual ground truth accuracy",
            "Missing reference metrics are null, never synthetic negatives", "Known-issue references excluded from primary, retained in raw",
            "Side-confusion proxy requires both references; primary proxy excludes either-side quality flags",
            "VE single-side jobs internally infer BOTH prompts; no compute-halving claim",
            "Previously viewed development benchmark, not a blind independent final test",
            "Predeclared completed-epoch test only; do not tune LR/epoch/threshold using these results"]}
    if args.preflight_only:
        print(json.dumps({"status": summary["status"], "whole_images": len(images), "shard_images": len(indices), "model": metadata}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; --preflight-only is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir/"code-snapshot"
    snapshot.mkdir()
    for path in dependencies:
        if path.parent.name == "scripts":
            shutil.copy2(path, snapshot/path.name)
    shared.atomic_write_json(args.output_dir/"progress.json", summary)
    started = time.monotonic()
    try:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(123)
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint), bpe_path=str(args.tokenizer_path),
            load_from_HF=False, device="cuda", eval_mode=True, enable_segmentation=True,
            enable_inst_interactivity=False, text_encoder_type="ve")
        model.eval().requires_grad_(False)
        model.register_forward_hook(semantic.assert_finite_model_outputs)
        versions = [(parameter, parameter._version) for parameter in model.parameters()]
        initial_encoder_hash = initializer.cache_fingerprint(encoder.state_dict()) if encoder is not None else None
        if encoder is not None:
            cached.install_cached_ve_text_encoder(model, encoder.to("cuda"))
            cached.set_cached_ve_training_mode(model, train_delta=False)
        raw = shared.make_dataset(args.data_root)
        if len(raw) != len(images):
            raise RuntimeError("Actual dataset count differs from full publication")
        dataset = semantic.IdentityCheckedDataset(raw, images)
        records, visuals = evaluate(model, args.mode, dataset, images, indices, annotations, outputs,
            args.data_root, args.output_dir, fingerprints, args.batch_size)
        if any(parameter._version != version for parameter, version in versions):
            raise RuntimeError("Original model parameters changed during inference")
        if encoder is not None and initializer.cache_fingerprint(encoder.state_dict()) != initial_encoder_hash:
            raise RuntimeError("Residual cache changed during inference")
        for path, digest in {**fingerprints, **implementation}.items():
            if shared.sha256(Path(path)) != digest:
                raise RuntimeError(f"Input/code changed during evaluation: {path}")
        summary.update(status="complete", elapsed_seconds=time.monotonic()-started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), actual_complete_query_coverage_verified=True,
            records_file="records.jsonl", records_sha256=shared.sha256(args.output_dir/"records.jsonl"),
            metrics=summarize(records), visualizations=visuals)
        shared.atomic_write_json(args.output_dir/"summary.json", summary)
        print(json.dumps({"status": "complete", "output": str(args.output_dir), "queries": len(records)}, indent=2))
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started)
        shared.atomic_write_json(args.output_dir/"failure.json", summary)
        raise


if __name__ == "__main__":
    main()

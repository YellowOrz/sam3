#!/usr/bin/env python3
"""Frozen, zero-shot bilateral testing against nakehand SAM3-assisted references.

Both prompts are always evaluated independently: a two-hand frame has two
positive queries, not one positive and an automatically negative opposite side.
No threshold fitting, training, or ground-truth-based candidate selection occurs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

if __package__:
    from . import evaluate_bilateral_tokens as shared
else:
    import evaluate_bilateral_tokens as shared


CLASS_NAMES = shared.CLASS_NAMES
SIDE_BY_CATEGORY = shared.SIDE_BY_CATEGORY
DETECTION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
REFERENCE_DESCRIPTION = "SAM3-assisted propagated masks; only explicitly marked diagnostics human-accepted"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--include-ve", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--indices", help="Explicit dataset indices for a diagnostic-only smoke run")
    parser.add_argument("--render-per-recording", type=int, default=2)
    parser.add_argument("--minimum-completed-epochs", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.learned_checkpoint and not args.include_ve:
        parser.error("provide --learned-checkpoint and/or --include-ve")
    if args.batch_size < 1 or args.render_per_recording < 0 or args.minimum_completed_epochs < 0:
        parser.error("batch-size must be positive and counts nonnegative")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        parser.error("--gpu-memory-fraction must be within (0, 1]")
    labels = [shared.parse_checkpoint_spec(item)[0] for item in args.learned_checkpoint]
    if args.include_ve:
        labels += ["ve-underscore", "ve-natural"]
    if len(labels) != len(set(labels)):
        parser.error("model labels must be unique, including reserved VE labels")
    return args


def view_type(image: dict) -> str:
    explicit = image.get("view_type", image.get("camera_type"))
    if explicit in ("ego", "exo"):
        return explicit
    recording = str(image.get("recording_id", "")).lower()
    for value in ("ego", "exo"):
        if value in recording:
            return value
    raise ValueError(f"Cannot determine ego/exo for image {image.get('id')}")


def diagnostic_ids(image: dict) -> list[str]:
    value = image.get("diagnostic_ids", image.get("diagnostic_id"))
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    return sorted({str(item) for item in values if str(item)})


def load_coco_index(root: Path) -> tuple[list[dict], dict[int, dict[str, np.ndarray]], dict]:
    data = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    if {int(row["id"]): row["name"] for row in data["categories"]} != SIDE_BY_CATEGORY:
        raise ValueError("Expected category 1=left_hand and category 2=right_hand")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    ids = [int(row["id"]) for row in images]
    if not images or len(ids) != len(set(ids)):
        raise ValueError("Images must be nonempty and image IDs unique")
    references = {}
    for image in images:
        if type(image.get("primary_test")) is not bool:
            raise ValueError(f"Image {image['id']} lacks explicit boolean primary_test")
        if not image.get("recording_id"):
            raise ValueError(f"Image {image['id']} lacks recording_id")
        view_type(image)
        height, width = int(image["height"]), int(image["width"])
        if height < 1 or width < 1:
            raise ValueError("Image dimensions must be positive")
        path = Path(image["file_name"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("file_name must remain inside data-root")
        references[int(image["id"])] = {
            side: np.zeros((height, width), dtype=bool) for side in CLASS_NAMES
        }
    annotation_ids = set()
    for annotation in data["annotations"]:
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation ID {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = int(annotation["image_id"])
        if image_id not in references:
            raise ValueError(f"Annotation references unknown image {image_id}")
        category = int(annotation["category_id"])
        if category not in SIDE_BY_CATEGORY:
            raise ValueError(f"Unknown annotation category {category}")
        side = SIDE_BY_CATEGORY[category]
        target = references[image_id][side]
        decoded = shared.decode_gt_mask(annotation, *target.shape)
        if decoded.shape != target.shape:
            raise ValueError(f"RLE size mismatch for image {image_id}")
        target |= decoded  # Multiple IDs/components of one physical side are a union.
    return images, references, data.get("info", {})


def select_indices(images: Sequence[dict], explicit: str | None) -> list[int]:
    if explicit is None:
        return list(range(len(images)))
    indices = sorted({int(value.strip()) for value in explicit.split(",") if value.strip()})
    if not indices or any(index < 0 or index >= len(images) for index in indices):
        raise ValueError("--indices must specify nonempty valid dataset indices")
    return indices


def choose_render_indices(images: Sequence[dict], indices: Sequence[int], per_recording: int) -> list[int]:
    groups = defaultdict(list)
    selected = set()
    for index in indices:
        image = images[index]
        if diagnostic_ids(image):
            selected.add(index)
        if image["primary_test"]:
            groups[image["recording_id"]].append(index)
    for group in groups.values():
        selected.update(shared.evenly_spaced(group, per_recording))
    return sorted(selected)


def validate_batch_identity(batch, dataset_indices: Sequence[int], images: Sequence[dict]) -> None:
    if len(batch.find_inputs) != 1:
        raise RuntimeError("External testing permits exactly one noninteractive query stage")
    shared.validate_batch_identity(batch, dataset_indices, images)
    stage = batch.find_inputs[0]
    observed = [(int(i), int(t)) for i, t in zip(stage.img_ids.tolist(), stage.text_ids.tolist())]
    expected = {(index, prompt) for index in range(len(dataset_indices)) for prompt in range(2)}
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Expected exactly one left query and one right query per actual image")
    if tuple(batch.find_text_batch) != CLASS_NAMES:
        raise RuntimeError(f"Unexpected query text order: {batch.find_text_batch!r}")
    for field in ("input_boxes", "input_points", "input_boxes_before_embed", "input_points_before_embed"):
        value = getattr(stage, field, None)
        if value is not None and (not isinstance(value, torch.Tensor) or value.numel()):
            raise RuntimeError(f"External zero-shot testing forbids geometry prompts: {field}")


def validate_frozen_noninteractive_model(model) -> None:
    if model.training or getattr(model, "num_interactive_steps_val", None) != 0:
        raise RuntimeError("Require eval mode and num_interactive_steps_val=0 to prevent reference-derived prompts")
    if any(module.training for module in model.modules()):
        raise RuntimeError("All submodules must be in eval mode")


def measure_query(prediction: np.ndarray, own_reference: np.ndarray, other_reference: np.ndarray,
                  score: float) -> dict:
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model confidence must be finite within [0, 1]")
    prediction = np.asarray(prediction, dtype=bool)
    own_reference = np.asarray(own_reference, dtype=bool)
    other_reference = np.asarray(other_reference, dtype=bool)
    if prediction.shape != own_reference.shape or prediction.shape != other_reference.shape:
        raise ValueError("Prediction and references must have identical original-image shape")
    target_present = bool(own_reference.any())
    other_present = bool(other_reference.any())
    detected = score >= DETECTION_THRESHOLD
    own_dice, own_iou = shared.dice_iou(prediction, own_reference)
    other_dice, other_iou = shared.dice_iou(prediction, other_reference)
    other_intersection = int(np.logical_and(prediction, other_reference).sum())
    wrong_side_proxy = other_present and other_intersection > 0 and other_iou > own_iou
    return {
        "target_present": target_present,
        "other_side_present": other_present,
        "any_hand_present": target_present or other_present,
        "both_hands_present": target_present and other_present,
        "top_confidence": score,
        "detected": detected,
        "reference_pixels": int(own_reference.sum()),
        "other_reference_pixels": int(other_reference.sum()),
        "top_mask_pixels": int(prediction.sum()),
        "detected_mask_pixels": int(prediction.sum()) if detected else 0,
        "top_dice": own_dice if target_present else None,
        "top_iou": own_iou if target_present else None,
        "miss_zero_dice": (own_dice if detected else 0.0) if target_present else None,
        "miss_zero_iou": (own_iou if detected else 0.0) if target_present else None,
        "top_dice_with_own_reference": own_dice,
        "top_iou_with_own_reference": own_iou,
        "top_dice_with_other_reference": other_dice,
        "top_iou_with_other_reference": other_iou,
        "top_other_reference_intersection_pixels": other_intersection,
        "opposite_overlap_dominant_proxy": bool(wrong_side_proxy),
        "detected_opposite_overlap_dominant_proxy": bool(detected and wrong_side_proxy),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(records: Sequence[dict]) -> dict:
    present = [row for row in records if row["target_present"]]
    absent = [row for row in records if not row["target_present"]]
    single_hand_absent = [row for row in absent if row["other_side_present"]]
    empty = [row for row in records if not row["any_hand_present"]]
    both = [row for row in records if row["both_hands_present"]]
    both_detected = [row for row in both if row["detected"]]
    other_visible = [row for row in records if row["other_side_present"]]
    other_visible_detected = [row for row in other_visible if row["detected"]]
    tp = sum(row["detected"] for row in present)
    fp = sum(row["detected"] for row in absent)
    empty_fp = sum(row["detected"] for row in empty)
    by_image = defaultdict(list)
    for row in records:
        by_image[int(row["image_id"])].append(row)
    empty_images = [rows for rows in by_image.values() if not rows[0]["any_hand_present"]]
    paired_empty_images = [rows for rows in empty_images if {row["prompt_key"] for row in rows} == set(CLASS_NAMES)]
    complete_both = [rows for rows in by_image.values()
                     if len(rows) == 2 and rows[0]["both_hands_present"]]
    simultaneous_swaps = sum(all(row["detected_opposite_overlap_dominant_proxy"] for row in rows)
                             for rows in complete_both)
    both_wrong = sum(row["detected_opposite_overlap_dominant_proxy"] for row in both)
    return {
        "images": len(by_image), "queries": len(records),
        "present_queries": len(present), "true_positive_queries": tp,
        "false_negative_queries": len(present) - tp,
        "correct_side_detection_rate": ratio(tp, len(present)),
        "present_mean_candidate_dice": shared.mean(row["top_dice"] for row in present),
        "present_mean_candidate_iou": shared.mean(row["top_iou"] for row in present),
        "present_mean_miss_zero_dice": shared.mean(row["miss_zero_dice"] for row in present),
        "present_mean_miss_zero_iou": shared.mean(row["miss_zero_iou"] for row in present),
        "absent_queries": len(absent), "false_positive_queries": fp,
        "absent_side_false_positive_rate": ratio(fp, len(absent)),
        "single_hand_absent_queries": len(single_hand_absent),
        "single_hand_absent_false_positive_queries": sum(row["detected"] for row in single_hand_absent),
        "single_hand_absent_false_positive_rate": shared.mean(row["detected"] for row in single_hand_absent),
        "empty_image_queries": len(empty), "empty_image_false_positive_queries": empty_fp,
        "empty_image_query_false_positive_rate": ratio(empty_fp, len(empty)),
        "empty_images": len(empty_images),
        "empty_images_with_both_query_results": len(paired_empty_images),
        "empty_images_with_any_detection": sum(any(row["detected"] for row in rows) for rows in paired_empty_images),
        "empty_image_any_detection_rate": ratio(
            sum(any(row["detected"] for row in rows) for rows in paired_empty_images), len(paired_empty_images)),
        "both_visible_queries": len(both), "both_visible_detected_queries": len(both_detected),
        "both_visible_candidate_opposite_dominant_queries": sum(row["opposite_overlap_dominant_proxy"] for row in both),
        "both_visible_detected_opposite_dominant_queries": both_wrong,
        "both_visible_candidate_opposite_dominant_rate": shared.mean(row["opposite_overlap_dominant_proxy"] for row in both),
        "both_visible_detected_opposite_dominant_rate_per_all_queries": ratio(both_wrong, len(both)),
        "both_visible_detected_opposite_dominant_rate_per_detected_queries": ratio(both_wrong, len(both_detected)),
        "other_visible_queries": len(other_visible), "other_visible_detected_queries": len(other_visible_detected),
        "other_visible_detected_opposite_dominant_queries": sum(row["detected_opposite_overlap_dominant_proxy"] for row in other_visible),
        "other_visible_detected_opposite_dominant_rate_per_all_queries": shared.mean(row["detected_opposite_overlap_dominant_proxy"] for row in other_visible),
        "other_visible_detected_opposite_dominant_rate_per_detected_queries": shared.mean(row["opposite_overlap_dominant_proxy"] for row in other_visible_detected),
        "both_visible_complete_image_pairs": len(complete_both),
        "simultaneous_two_query_swap_proxy_images": simultaneous_swaps,
        "simultaneous_two_query_swap_proxy_image_rate": ratio(simultaneous_swaps, len(complete_both)),
    }


def grouped_summary(records: Sequence[dict]) -> dict:
    groups = {
        "overall": aggregate(records),
        "per_side": {side: aggregate([row for row in records if row["prompt_key"] == side])
                     for side in CLASS_NAMES},
        "per_recording": {}, "per_view_type": {},
    }
    for field, name in (("recording_id", "per_recording"), ("view_type", "per_view_type")):
        values = sorted({row[field] for row in records})
        for value in values:
            subset = [row for row in records if row[field] == value]
            groups[name][value] = {
                "overall": aggregate(subset),
                "per_side": {side: aggregate([row for row in subset if row["prompt_key"] == side])
                             for side in CLASS_NAMES},
            }
    metrics = ("present_mean_candidate_dice", "present_mean_candidate_iou",
               "present_mean_miss_zero_dice", "correct_side_detection_rate",
               "absent_side_false_positive_rate", "empty_image_query_false_positive_rate")
    groups["recording_macro_means"] = {
        key: {"value": shared.mean(row["overall"][key] for row in groups["per_recording"].values()
                                    if row["overall"][key] is not None),
              "recordings_with_denominator": sum(row["overall"][key] is not None
                                                 for row in groups["per_recording"].values())}
        for key in metrics
    }
    return groups


def summarize(records: Sequence[dict]) -> dict:
    result = {}
    for label in sorted({row["model"] for row in records}):
        selected = [row for row in records if row["model"] == label]
        keys = [(row["image_id"], row["prompt_key"]) for row in selected]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate query records for model {label}")
        by_image = defaultdict(set)
        for row in selected:
            by_image[row["image_id"]].add(row["prompt_key"])
        if any(sides != set(CLASS_NAMES) for sides in by_image.values()):
            raise ValueError(f"Missing side query for model {label}")
        main = [row for row in selected if row["primary_test"]]
        diagnostic = [row for row in selected if row["diagnostic_ids"]]
        result[label] = {
            "primary_test": grouped_summary(main),
            "diagnostic_only_no_population_claim": grouped_summary(diagnostic),
            "diagnostic_examples": {
                name: grouped_summary([row for row in diagnostic if name in row["diagnostic_ids"]])
                for name in sorted({name for row in diagnostic for name in row["diagnostic_ids"]})
            },
        }
    return result


def learned_checkpoint_metadata(path: Path, base_checkpoint: Path, minimum_epochs: int) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state.get("training_config", {})
    count = state.get("annotation_summary", {}).get("images")
    batch_size = config.get("batch_size")
    next_step = state.get("next_step")
    epochs = state.get("epochs", config.get("epochs"))
    completed = None
    if all(type(value) is int and value > 0 for value in (count, batch_size, epochs)) and type(next_step) is int:
        per_epoch = math.ceil(count / batch_size)
        if not 0 <= next_step <= per_epoch * epochs:
            raise ValueError(f"Invalid checkpoint step count: {path}")
        completed = next_step // per_epoch
    if minimum_epochs and (completed is None or completed < minimum_epochs):
        raise ValueError(f"Checkpoint has {completed} complete epochs; require {minimum_epochs}: {path}")
    old_base = state.get("base_checkpoint", config.get("base_checkpoint"))
    if old_base and Path(old_base).resolve() != base_checkpoint.resolve():
        raise ValueError(f"Learned checkpoint base path differs: {old_base} != {base_checkpoint}")
    tokens = state.get("class_tokens")
    if not isinstance(tokens, torch.Tensor) or not torch.isfinite(tokens).all():
        raise ValueError("Learned class_tokens are absent or non-finite")
    return {
        "checkpoint_sha256": shared.sha256(path), "completed_epochs_from_steps": completed,
        "training_config": config, "training_annotation_summary": state.get("annotation_summary"),
        "training_progress": state.get("progress"),
        "legacy_base_hash_limitation": "Original training checkpoint does not establish historical base bytes; current shared base SHA is recorded",
    }


def evaluate_variant(*, model, label, prompts, dataset, images, references, indices,
                     render_indices, batch_size, amp):
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    validate_frozen_noninteractive_model(model)
    records, masks = [], {}
    for batch_number, dataset_indices in enumerate(shared.batches(indices, batch_size), 1):
        samples = [dataset[index] for index in dataset_indices]
        batch = collate_fn_api(samples, dict_key="eval", with_seg_masks=True)["eval"]
        validate_batch_identity(batch, dataset_indices, images)
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        batch.find_text_batch = list(prompts)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            output = model(batch)[0]
        probabilities = output["pred_logits"].float().sigmoid().squeeze(-1)
        presence = output["presence_logit_dec"].float().sigmoid().reshape(len(probabilities), -1)[:, 0]
        combined = probabilities * presence[:, None]
        if not bool(torch.isfinite(combined).all()):
            raise RuntimeError("Non-finite confidence in model output")
        top_indices = combined.argmax(dim=1)  # Never rank masks by reference overlap.
        stage = batch.find_inputs[0]
        for row in range(len(top_indices)):
            local_image = int(stage.img_ids[row])
            dataset_index = dataset_indices[local_image]
            image = images[dataset_index]
            image_id = int(image["id"])
            prompt_index = int(stage.text_ids[row])
            side, other = CLASS_NAMES[prompt_index], CLASS_NAMES[1 - prompt_index]
            mask_logit = output["pred_masks"][row, int(top_indices[row])]
            resized = F.interpolate(mask_logit[None, None].float(),
                                    size=(int(image["height"]), int(image["width"])),
                                    mode="bilinear", align_corners=False)[0, 0]
            if not bool(torch.isfinite(resized).all()):
                raise RuntimeError("Non-finite mask logits in model output")
            prediction = resized.sigmoid().cpu().numpy() >= MASK_THRESHOLD
            score = float(combined[row, top_indices[row]])
            record = {
                "model": label, "dataset_index": dataset_index, "image_id": image_id,
                "observed_coco_image_id": image_id, "identity_verified": True,
                "file_name": image["file_name"], "recording_id": image["recording_id"],
                "view_type": view_type(image), "source_frame_index": image.get("source_frame_index"),
                "video_pts_seconds": image.get("video_pts_seconds"),
                "source_mapping": image.get("source_mapping", {
                    key: image.get(key) for key in (
                        "source_rgb_path", "source_mask_paths", "source_metadata_path", "source_timestamp_seconds")
                }),
                "primary_test": image["primary_test"], "diagnostic_ids": diagnostic_ids(image),
                "human_review": image.get("human_review"),
                "reference_description": REFERENCE_DESCRIPTION,
                "prompt_key": side, "prompt_text": prompts[prompt_index],
                "top_class_probability": float(probabilities[row, top_indices[row]]),
                "presence_probability": float(presence[row]), "selected_decoder_query": int(top_indices[row]),
                "detections_above_threshold": int((combined[row] >= DETECTION_THRESHOLD).sum()),
                **measure_query(prediction, references[image_id][side], references[image_id][other], score),
            }
            records.append(record)
            if dataset_index in render_indices:
                masks[(dataset_index, side)] = prediction
        if batch_number == 1 or batch_number % 25 == 0:
            print(f"{label}: images={min(batch_number * batch_size, len(indices))}/{len(indices)}", flush=True)
        del samples, batch, output, resized, mask_logit, probabilities, presence, combined, top_indices
    expected = {(int(images[index]["id"]), side) for index in indices for side in CLASS_NAMES}
    observed = [(row["image_id"], row["prompt_key"]) for row in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Final observed image/query identity coverage differs from frozen selection")
    return records, masks


def titled_panel(image: Image.Image, title: str) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + 48), "white")
    panel.paste(image.convert("RGB"), (0, 48))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((8, 5), title, fill="black", spacing=3)
    return panel


def render_results(*, data_root, output_dir, images, references, render_indices, records, masks, labels):
    by_key = {(row["model"], row["dataset_index"], row["prompt_key"]): row for row in records}
    manifest = []
    output_dir.mkdir()
    for index in render_indices:
        image = images[index]
        image_id = int(image["id"])
        directory = output_dir / f"image-{image_id:06d}"
        directory.mkdir()
        with Image.open(data_root / image["file_name"]) as source:
            rgb = source.convert("RGB")
        rgb.save(directory / "rgb.png")
        refs = {}
        for side in CLASS_NAMES:
            refs[side] = Image.fromarray(references[image_id][side].astype(np.uint8) * 255)
            refs[side].save(directory / f"{side}__reference.png")
        panels = []
        for label in labels:
            model_dir = directory / shared.safe_label(label)
            model_dir.mkdir()
            row = [titled_panel(rgb, f"Original RGB | {label}"),
                   titled_panel(refs["left_hand"], "LEFT REFERENCE (SAM3-assisted)"),
                   titled_panel(refs["right_hand"], "RIGHT REFERENCE (SAM3-assisted)")]
            for side in CLASS_NAMES:
                record = by_key[(label, index, side)]
                candidate = masks[(label, index, side)]
                detected = candidate if record["detected"] else np.zeros_like(candidate)
                Image.fromarray(candidate.astype(np.uint8) * 255).save(model_dir / f"{side}__candidate.png")
                displayed = Image.fromarray(detected.astype(np.uint8) * 255)
                displayed.save(model_dir / f"{side}__detected.png")
                dice = record["miss_zero_dice"]
                dice_text = "absent reference" if dice is None else f"Dice={dice:.3f}"
                row.append(titled_panel(displayed, f"{record['prompt_text']} | score={record['top_confidence']:.3f}\n"
                                                   f"det={int(record['detected'])} | {dice_text}"))
            panels.append(row)
        canvas = Image.new("RGB", (rgb.width * 5, (rgb.height + 48) * len(labels)), "white")
        for row_index, row in enumerate(panels):
            for column_index, panel in enumerate(row):
                canvas.paste(panel, (column_index * rgb.width, row_index * (rgb.height + 48)))
        canvas.save(directory / "comparison.png")
        manifest.append({"image_id": image_id, "dataset_index": index, "diagnostic_ids": diagnostic_ids(image),
                         "directory": str(directory.resolve()), "comparison": str((directory / "comparison.png").resolve())})
    shared.atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def snapshot_code(output_dir: Path) -> list[dict]:
    directory = output_dir / "code-snapshot"
    directory.mkdir()
    sources = {Path(__file__).resolve(), Path(inspect.getfile(shared)).resolve()}
    records = []
    for source in sorted(sources):
        target = directory / source.name
        shutil.copy2(source, target)
        records.append({"source": str(source), "snapshot": str(target), "sha256": shared.sha256(target)})
    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    annotations_hash = shared.sha256(args.data_root / "annotations.json")
    images, references, dataset_info = load_coco_index(args.data_root)
    if shared.sha256(args.data_root / "annotations.json") != annotations_hash:
        raise RuntimeError("Annotations changed while loading references")
    indices = select_indices(images, args.indices)
    learned_specs = [shared.parse_checkpoint_spec(value) for value in args.learned_checkpoint]
    checked_metadata = {label: learned_checkpoint_metadata(path, args.base_checkpoint, args.minimum_completed_epochs)
                        for label, path in learned_specs}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records_dir = args.output_dir / "records"
    records_dir.mkdir()
    base_hash = shared.sha256(args.base_checkpoint)
    image_hashes = []
    for index in indices:
        image = images[index]
        path = args.data_root / image["file_name"]
        with Image.open(path) as rgb:
            if rgb.size != (int(image["width"]), int(image["height"])):
                raise ValueError(f"RGB size differs from annotations: {path}")
        image_hashes.append({"dataset_index": index, "image_id": int(image["id"]),
                             "file_name": image["file_name"], "sha256": shared.sha256(path)})
    shared.atomic_write_json(args.output_dir / "image-identities.json", image_hashes)
    provenance = {
        "format": "nakehand-frozen-bilateral-evaluation-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv if argv is None else argv),
        "data_root": str(args.data_root.resolve()), "dataset_info": dataset_info,
        "annotations_sha256": annotations_hash,
        "base_checkpoint": str(args.base_checkpoint.resolve()), "base_checkpoint_sha256": base_hash,
        "code_snapshots": snapshot_code(args.output_dir),
        "evaluated_dataset_indices": indices, "evaluated_image_ids": [int(images[index]["id"]) for index in indices],
        "evaluated_images": len(indices), "full_export_evaluated": len(indices) == len(images),
        "primary_test_images": sum(images[index]["primary_test"] for index in indices),
        "diagnostic_images": sum(bool(diagnostic_ids(images[index])) for index in indices),
        "detection_threshold": DETECTION_THRESHOLD, "mask_threshold": MASK_THRESHOLD,
        "thresholds_fitted_on_nakehand": False, "training_performed": False,
        "reference_description": REFERENCE_DESCRIPTION,
        "candidate_selection": "argmax(sigmoid(class_logit) * sigmoid(presence_logit)); never reference overlap",
        "preprocessing": "existing SAM3 dataset: square 1008 resize, RGB normalized with mean/std .5; logits bilinear to original H/W before sigmoid >= .5",
        "metric_notes": {
            "scope": "zero-shot cross-dataset agreement with SAM3-assisted reference, not independent manual pixel ground truth",
            "candidate_overlap": "Dice/IoU averaged only over nonempty same-side references; empty denominators are null",
            "detection": "combined confidence >= .5; TPR is presence detection, not an IoU-matched detection metric",
            "miss_zero": "same-side reference present: candidate Dice/IoU if detected, otherwise zero",
            "absence": "a side is absent iff its union reference is empty; the other hand may still be present",
            "swap_proxy": "strict other-reference IoU > own-reference IoU and positive intersection with other; ties false; not proof of anatomical swap, merged masks may trigger",
            "cohorts": "primary_test and named human-accepted diagnostics reported separately; any overlap explicitly retains both memberships",
            "sampling": "600 primary images: 100 uniformly indexed per recording, recording-balanced selection, not an unbiased estimate over all 18498 frames; overall metrics are query-micro averages, recording macro means separate; correlated frames are not independent subjects",
            "geometry_or_reference_prompts": "forbidden; assert eval mode, all submodules eval, num_interactive_steps_val=0, no point/box inputs or embeddings",
        },
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "device": torch.cuda.get_device_name(), "batch_size": args.batch_size,
                    "amp": args.amp, "amp_dtype": "bfloat16" if args.amp else "float32",
                    "gpu_memory_fraction": args.gpu_memory_fraction},
    }
    shared.atomic_write_json(args.output_dir / "run.json", provenance)
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dataset = shared.make_dataset(args.data_root)
    if len(dataset) != len(images):
        raise RuntimeError("COCO and SAM3 loader length mismatch")
    render_indices = choose_render_indices(images, indices, args.render_per_recording)
    all_records, all_masks, metadata = [], {}, {}

    def run_variant(model, label, prompts):
        records, masks = evaluate_variant(
            model=model, label=label, prompts=prompts, dataset=dataset, images=images,
            references=references, indices=indices, render_indices=set(render_indices),
            batch_size=args.batch_size, amp=args.amp)
        shared.atomic_write_json(records_dir / f"{label}.json", records)
        all_records.extend(records)
        all_masks.update({(label, index, side): mask for (index, side), mask in masks.items()})

    for label, path in learned_specs:
        print(f"Loading {label}: {path}", flush=True)
        model, model_meta = shared.load_learned_model(args.base_checkpoint, path)
        metadata[label] = {**model_meta, **checked_metadata[label], "prompt_texts": list(CLASS_NAMES)}
        run_variant(model, label, CLASS_NAMES)
        del model
        torch.cuda.empty_cache()
    if args.include_ve:
        print("Loading original frozen VE text encoder", flush=True)
        model = shared.load_ve_model(args.base_checkpoint)
        for label, prompts in (("ve-underscore", CLASS_NAMES), ("ve-natural", ("left hand", "right hand"))):
            metadata[label] = {"kind": "ve", "prompt_texts": list(prompts)}
            run_variant(model, label, prompts)
        del model
        torch.cuda.empty_cache()
    if shared.sha256(args.data_root / "annotations.json") != annotations_hash:
        raise RuntimeError("Annotations changed during evaluation")
    if shared.sha256(args.base_checkpoint) != base_hash:
        raise RuntimeError("Base checkpoint changed during evaluation")
    for label, path in learned_specs:
        if shared.sha256(path) != checked_metadata[label]["checkpoint_sha256"]:
            raise RuntimeError(f"Learned checkpoint changed during evaluation: {label}")
    for row in image_hashes:
        if shared.sha256(args.data_root / row["file_name"]) != row["sha256"]:
            raise RuntimeError(f"RGB bytes changed during evaluation: image {row['image_id']}")
    visuals = render_results(data_root=args.data_root, output_dir=args.output_dir / "visuals", images=images,
                             references=references, render_indices=render_indices, records=all_records,
                             masks=all_masks, labels=list(metadata))
    summary = {**provenance, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "completed", "models": metadata, "metrics": summarize(all_records),
               "observed_identity_verified": True, "visuals": visuals,
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
               "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
    shared.atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"),
                      "evaluated_images": summary["evaluated_images"],
                      "primary_test_images": summary["primary_test_images"],
                      "diagnostic_images": summary["diagnostic_images"],
                      "models": list(summary["models"])},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

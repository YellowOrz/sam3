#!/usr/bin/env python3
"""Evaluate bilateral learned hand tokens and render side-confusion examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


CLASS_NAMES = ("left_hand", "right_hand")
SIDE_BY_CATEGORY = {1: "left_hand", 2: "right_hand"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--learned-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Learned-token checkpoint; repeat to compare epoch/checkpoint variants",
    )
    parser.add_argument(
        "--include-ve",
        action="store_true",
        help="Also evaluate the original VE encoder with underscore and space prompts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--visual-style", choices=("separate", "overlay"), default="separate",
        help="Separate RGB/GT/prediction PNGs (default), or legacy colored overlays",
    )
    parser.add_argument(
        "--unified-root",
        type=Path,
        default=Path("/data/xuzhefeng/Datasets/uni-hoi-dataset"),
        help="Used only to report the original rgb.mkv/mask.mkv mapping",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--gpu-memory-fraction", type=float, default=None,
        help="Optional per-process PyTorch allocator limit for shared GPUs (0 < fraction <= 1)",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=0,
        help="Evaluate an evenly spread subset of each left/right/empty group; 0 means all",
    )
    parser.add_argument(
        "--render-count-per-group",
        type=int,
        default=4,
        help="Render this many evaluated examples from each left/right/empty group",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated dataset indices; overrides --samples-per-group",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.learned_checkpoint and not args.include_ve:
        parser.error("provide --learned-checkpoint and/or --include-ve")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.gpu_memory_fraction is not None and not 0 < args.gpu_memory_fraction <= 1:
        parser.error("--gpu-memory-fraction must be within (0, 1]")
    if args.samples_per_group < 0 or args.render_count_per_group < 0:
        parser.error("sample counts cannot be negative")
    for name in ("detection_threshold", "mask_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within [0, 1]")
    return args


def atomic_write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not label:
        raise ValueError(f"Invalid empty model label derived from {value!r}")
    return label


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(value).stem
    return safe_label(label), Path(raw_path)


def load_coco_index(root: Path) -> tuple[list[dict], dict[int, dict | None]]:
    annotation_path = root / "annotations.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    if categories != SIDE_BY_CATEGORY:
        raise ValueError(
            f"Expected categories {SIDE_BY_CATEGORY}, got {categories} in {annotation_path}"
        )

    images = sorted(data["images"], key=lambda item: int(item["id"]))
    annotations_by_image: dict[int, dict | None] = {int(item["id"]): None for item in images}
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        if annotations_by_image.get(image_id) is not None:
            raise ValueError(f"Expected at most one physical hand in image {image_id}")
        annotations_by_image[image_id] = annotation
    return images, annotations_by_image


def actual_side(image: dict, annotations_by_image: dict[int, dict | None]) -> str:
    annotation = annotations_by_image[int(image["id"])]
    return "empty" if annotation is None else SIDE_BY_CATEGORY[int(annotation["category_id"])]


def evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    positions = np.linspace(0, len(values) - 1, num=count)
    return sorted({values[int(round(position))] for position in positions})


def choose_indices(
    images: Sequence[dict],
    annotations_by_image: dict[int, dict | None],
    samples_per_group: int,
    explicit: str | None,
) -> list[int]:
    if explicit:
        indices = sorted({int(value.strip()) for value in explicit.split(",") if value.strip()})
        invalid = [index for index in indices if not 0 <= index < len(images)]
        if invalid:
            raise IndexError(f"Dataset indices outside [0, {len(images) - 1}]: {invalid}")
        return indices
    if samples_per_group == 0:
        return list(range(len(images)))

    groups: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        groups[actual_side(image, annotations_by_image)].append(index)
    selected = []
    for group in ("left_hand", "right_hand", "empty"):
        selected.extend(evenly_spaced(groups[group], samples_per_group))
    return sorted(selected)


def choose_render_indices(
    eval_indices: Iterable[int],
    images: Sequence[dict],
    annotations_by_image: dict[int, dict | None],
    count_per_group: int,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in eval_indices:
        groups[actual_side(images[index], annotations_by_image)].append(index)
    selected = []
    for group in ("left_hand", "right_hand", "empty"):
        selected.extend(evenly_spaced(groups[group], count_per_group))
    return sorted(selected)


def make_dataset(root: Path):
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
    from sam3.train.transforms.basic_for_api import (
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )
    from sam3.train.transforms.segmentation import DecodeRle

    return Sam3ImageDataset(
        img_folder=str(root),
        ann_file=str(root / "annotations.json"),
        transforms=[
            DecodeRle(),
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
        max_ann_per_img=100,
        multiplier=1,
        training=False,
        load_segmentation=True,
    )


def load_learned_model(base_checkpoint: Path, token_checkpoint: Path):
    from sam3.model.learnable_text_encoder import LearnableClassTextEncoder
    from sam3.model_builder import build_sam3_image_model

    state = torch.load(token_checkpoint, map_location="cpu", weights_only=True)
    if state.get("class_names") != list(CLASS_NAMES):
        raise ValueError(
            f"{token_checkpoint} class_names must be {list(CLASS_NAMES)}, "
            f"got {state.get('class_names')!r}"
        )
    class_tokens = state.get("class_tokens")
    if not isinstance(class_tokens, torch.Tensor) or class_tokens.ndim != 3:
        raise ValueError(f"{token_checkpoint} has an invalid class_tokens tensor")
    if tuple(class_tokens.shape[:1]) != (2,) or class_tokens.shape[-1] != 256:
        raise ValueError(f"Unexpected class_tokens shape {tuple(class_tokens.shape)}")

    model = build_sam3_image_model(
        checkpoint_path=str(base_checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        text_encoder_type="learnable_class",
        tokens_per_class=int(class_tokens.shape[1]),
    )
    encoder = next(
        module for module in model.modules() if isinstance(module, LearnableClassTextEncoder)
    )
    with torch.no_grad():
        encoder.class_tokens.copy_(class_tokens.to(encoder.class_tokens))
    model.eval()
    metadata = {
        "kind": "learned_class",
        "checkpoint": str(token_checkpoint.resolve()),
        "checkpoint_sha256": sha256(token_checkpoint),
        "checkpoint_format": state.get("format"),
        "tokens_per_class": int(class_tokens.shape[1]),
        "next_step": state.get("next_step", state.get("steps")),
    }
    return model, metadata


def load_ve_model(base_checkpoint: Path):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(base_checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        text_encoder_type="ve",
    )
    model.eval()
    return model


def decode_gt_mask(annotation: dict | None, height: int, width: int) -> np.ndarray:
    if annotation is None:
        return np.zeros((height, width), dtype=bool)
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation["counts"], list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = dict(segmentation)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def dice_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(prediction, target).sum())
    pred_pixels = int(prediction.sum())
    target_pixels = int(target.sum())
    dice = 2.0 * intersection / max(pred_pixels + target_pixels, 1)
    union = pred_pixels + target_pixels - intersection
    iou = intersection / max(union, 1)
    return dice, iou


def batches(values: Sequence[int], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def source_mapping(image: dict, unified_root: Path) -> dict:
    view_dir = (
        unified_root
        / "sequences"
        / str(image["source"])
        / str(image["sequence"])
        / str(image["view"])
    )
    return {
        "view_dir": str(view_dir),
        "rgb_video": str(view_dir / "rgb.mkv"),
        "mask_video": str(view_dir / "mask.mkv"),
        "frame_index": int(image["frame_index"]),
    }


def validate_batch_identity(batch, dataset_indices: Sequence[int], images: Sequence[dict]):
    """Reject loader fallback or category mismatches before comparing with GT."""
    stage = batch.find_inputs[0]
    metadata = batch.find_metadatas[0]
    for row, local_index in enumerate(stage.img_ids.tolist()):
        expected_id = int(images[dataset_indices[local_index]]["id"])
        if int(metadata.coco_image_id[row]) != expected_id:
            raise RuntimeError(f"Dataset substituted image {expected_id} during evaluation")
        expected_category = int(stage.text_ids[row]) + 1
        if int(metadata.original_category_id[row]) != expected_category:
            raise RuntimeError(f"Prompt/category mismatch on image {expected_id}")


def evaluate_variant(
    *,
    model,
    label: str,
    prompt_texts: tuple[str, str],
    dataset,
    images: Sequence[dict],
    annotations_by_image: dict[int, dict | None],
    eval_indices: Sequence[int],
    render_indices: set[int],
    batch_size: int,
    detection_threshold: float,
    mask_threshold: float,
    amp: bool,
    unified_root: Path,
) -> tuple[list[dict], dict[tuple[int, str], np.ndarray]]:
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    records: list[dict] = []
    rendered_masks: dict[tuple[int, str], np.ndarray] = {}
    gt_cache: dict[int, np.ndarray] = {}
    device = torch.device("cuda")

    for batch_number, dataset_indices in enumerate(batches(eval_indices, batch_size), 1):
        samples = [dataset[index] for index in dataset_indices]
        batch = collate_fn_api(samples, dict_key="eval", with_seg_masks=True)["eval"]
        # The dataset can recover from a read error by returning another image.
        # Refuse that fallback here: otherwise prediction and GT would diverge.
        validate_batch_identity(batch, dataset_indices, images)
        batch = copy_data_to_device(batch, device, non_blocking=True)
        if tuple(batch.find_text_batch) != CLASS_NAMES:
            raise RuntimeError(f"Unexpected dataset prompts: {batch.find_text_batch!r}")
        batch.find_text_batch = list(prompt_texts)

        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=amp
        ):
            output = model(batch)[0]

        class_probabilities = output["pred_logits"].float().sigmoid().squeeze(-1)
        presence_probabilities = output["presence_logit_dec"].float().sigmoid().reshape(
            len(class_probabilities), -1
        )[:, 0]
        combined_scores = class_probabilities * presence_probabilities[:, None]
        top_indices = combined_scores.argmax(dim=1)
        query_rows = torch.arange(len(top_indices), device=device)
        top_class = class_probabilities[query_rows, top_indices]
        top_confidence = combined_scores[query_rows, top_indices]
        detections_per_query = (combined_scores >= detection_threshold).sum(dim=1)
        stage = batch.find_inputs[0]

        for query_row in range(len(top_indices)):
            local_image_index = int(stage.img_ids[query_row].item())
            dataset_index = int(dataset_indices[local_image_index])
            image = images[dataset_index]
            image_id = int(image["id"])
            prompt_index = int(stage.text_ids[query_row].item())
            prompt_key = CLASS_NAMES[prompt_index]
            side = actual_side(image, annotations_by_image)
            height, width = int(image["height"]), int(image["width"])

            if image_id not in gt_cache:
                gt_cache[image_id] = decode_gt_mask(
                    annotations_by_image[image_id], height, width
                )
            gt_mask = gt_cache[image_id]
            mask_logit = output["pred_masks"][
                query_row, int(top_indices[query_row].item())
            ]
            top_mask = (
                F.interpolate(
                    mask_logit[None, None].float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                .sigmoid()
                .cpu()
                .numpy()
                >= mask_threshold
            )
            confidence = float(top_confidence[query_row].cpu())
            detected = confidence >= detection_threshold
            detected_mask = top_mask if detected else np.zeros_like(top_mask)
            top_dice, top_iou = dice_iou(top_mask, gt_mask)
            detected_dice, detected_iou = dice_iou(detected_mask, gt_mask)

            record = {
                "model": label,
                "dataset_index": dataset_index,
                "image_id": image_id,
                "file_name": image["file_name"],
                "source": image.get("source"),
                "sequence": image.get("sequence"),
                "view": image.get("view"),
                "frame_index": int(image.get("frame_index", -1)),
                "original_mapping": source_mapping(image, unified_root),
                "actual_side": side,
                "prompt_key": prompt_key,
                "prompt_text": prompt_texts[prompt_index],
                "target_present": side == prompt_key,
                "physical_hand_present": side != "empty",
                "presence_probability": float(presence_probabilities[query_row].cpu()),
                "top_class_probability": float(top_class[query_row].cpu()),
                "top_confidence": confidence,
                "detected": detected,
                "detections_above_threshold": int(
                    detections_per_query[query_row].cpu()
                ),
                "gt_pixels": int(gt_mask.sum()),
                "top_mask_pixels": int(top_mask.sum()),
                "detected_mask_pixels": int(detected_mask.sum()),
                "top_mask_area_ratio": float(top_mask.mean()),
                "top_dice_with_physical_hand": top_dice,
                "top_iou_with_physical_hand": top_iou,
                "detected_dice_with_physical_hand": detected_dice,
                "detected_iou_with_physical_hand": detected_iou,
            }
            records.append(record)
            if dataset_index in render_indices:
                rendered_masks[(dataset_index, prompt_key)] = top_mask

        if batch_number == 1 or batch_number % 50 == 0:
            print(
                f"{label}: batches={batch_number} images={min(batch_number * batch_size, len(eval_indices))}/{len(eval_indices)}",
                flush=True,
            )
        del samples, batch, output

    return records, rendered_masks


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(sum(values) / len(values))


def binary_counts(records: Sequence[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for record in records:
        positive = bool(record["target_present"])
        predicted = float(record["top_confidence"]) >= threshold
        tp += int(positive and predicted)
        fp += int(not positive and predicted)
        tn += int(not positive and not predicted)
        fn += int(positive and not predicted)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }


def average_precision(records: Sequence[dict]) -> float | None:
    positives = sum(bool(record["target_present"]) for record in records)
    if positives == 0:
        return None
    ranked = sorted(records, key=lambda record: float(record["top_confidence"]), reverse=True)
    true_positives = 0
    precision_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        score = float(ranked[start]["top_confidence"])
        while end < len(ranked) and float(ranked[end]["top_confidence"]) == score:
            end += 1
        group_positives = sum(bool(row["target_present"]) for row in ranked[start:end])
        true_positives += group_positives
        precision_sum += group_positives * true_positives / end
        start = end
    return precision_sum / positives


def summarize(records: Sequence[dict], detection_threshold: float) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)

    summaries = {}
    for label, model_records in by_model.items():
        slices = {}
        for side in ("left_hand", "right_hand", "empty"):
            for prompt in CLASS_NAMES:
                subset = [
                    record
                    for record in model_records
                    if record["actual_side"] == side and record["prompt_key"] == prompt
                ]
                slices[f"actual={side}|prompt={prompt}"] = {
                    "count": len(subset),
                    "detection_rate": mean(record["detected"] for record in subset),
                    "presence_positive_rate": mean(
                        record["presence_probability"] >= detection_threshold
                        for record in subset
                    ),
                    "mean_presence_probability": mean(
                        record["presence_probability"] for record in subset
                    ),
                    "mean_top_class_probability": mean(
                        record["top_class_probability"] for record in subset
                    ),
                    "mean_top_confidence": mean(
                        record["top_confidence"] for record in subset
                    ),
                    "mean_top_mask_area_ratio": mean(
                        record["top_mask_area_ratio"] for record in subset
                    ),
                    "mean_top_dice_with_physical_hand": mean(
                        record["top_dice_with_physical_hand"] for record in subset
                    ),
                    "mean_detected_dice_with_physical_hand": mean(
                        record["detected_dice_with_physical_hand"] for record in subset
                    ),
                }

        per_image: dict[int, dict[str, dict]] = defaultdict(dict)
        for record in model_records:
            per_image[int(record["image_id"])][record["prompt_key"]] = record
        visible = [rows for rows in per_image.values() if next(iter(rows.values()))["actual_side"] != "empty"]
        side_correct = 0
        margins = []
        for rows in visible:
            if set(rows) != set(CLASS_NAMES):
                raise RuntimeError(f"Missing prompt result for image: {rows.keys()}")
            side = rows["left_hand"]["actual_side"]
            opposite = "right_hand" if side == "left_hand" else "left_hand"
            correct_score = float(rows[side]["top_confidence"])
            wrong_score = float(rows[opposite]["top_confidence"])
            side_correct += int(correct_score > wrong_score)
            margins.append(correct_score - wrong_score)

        correct_prompts = [record for record in model_records if record["target_present"]]
        wrong_visible_prompts = [
            record
            for record in model_records
            if record["physical_hand_present"] and not record["target_present"]
        ]
        empty_prompts = [
            record for record in model_records if not record["physical_hand_present"]
        ]
        summaries[label] = {
            "prompt_binary_average_precision": average_precision(model_records),
            "prompt_binary_at_detection_threshold": binary_counts(
                model_records, detection_threshold
            ),
            "threshold_sweep": [
                binary_counts(model_records, round(value / 10, 1))
                for value in range(1, 10)
            ],
            "visible_side_selection_accuracy": side_correct / max(len(visible), 1),
            "visible_mean_correct_minus_wrong_confidence": mean(margins),
            "correct_prompt_detection_rate": mean(
                record["detected"] for record in correct_prompts
            ),
            "correct_prompt_mean_top_dice": mean(
                record["top_dice_with_physical_hand"] for record in correct_prompts
            ),
            "correct_prompt_mean_thresholded_dice": mean(
                record["detected_dice_with_physical_hand"] for record in correct_prompts
            ),
            "opposite_prompt_false_positive_rate": mean(
                record["detected"] for record in wrong_visible_prompts
            ),
            "opposite_prompt_mean_hand_dice": mean(
                record["detected_dice_with_physical_hand"]
                for record in wrong_visible_prompts
            ),
            "empty_image_false_positive_rate": mean(
                record["detected"] for record in empty_prompts
            ),
            "slices": slices,
        }
    return summaries


def overlay_panel(
    rgb: Image.Image,
    gt: np.ndarray,
    pred: np.ndarray | None,
    title: str,
) -> Image.Image:
    rgb_array = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    output = rgb_array.copy()
    if pred is None:
        output[gt] = 0.45 * output[gt] + 0.55 * np.asarray((40, 220, 80))
    else:
        false_positive = pred & ~gt
        false_negative = gt & ~pred
        overlap = pred & gt
        for mask, color in (
            (false_positive, (255, 55, 55)),
            (false_negative, (50, 230, 90)),
            (overlap, (255, 215, 35)),
        ):
            output[mask] = 0.45 * output[mask] + 0.55 * np.asarray(color)
    panel = Image.fromarray(output.astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 34), fill=(0, 0, 0))
    draw.text((8, 10), title, fill=(255, 255, 255))
    return panel


def render_montages(
    *,
    root: Path,
    output_dir: Path,
    render_indices: Sequence[int],
    images: Sequence[dict],
    annotations_by_image: dict[int, dict | None],
    records: Sequence[dict],
    masks: dict[tuple[str, int, str], np.ndarray],
    model_labels: Sequence[str],
) -> list[dict]:
    records_by_key = {
        (record["model"], int(record["dataset_index"]), record["prompt_key"]): record
        for record in records
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_index in render_indices:
        image_info = images[dataset_index]
        image_id = int(image_info["id"])
        rgb = Image.open(root / image_info["file_name"]).convert("RGB")
        gt = decode_gt_mask(
            annotations_by_image[image_id], rgb.height, rgb.width
        )
        side = actual_side(image_info, annotations_by_image)
        rows = []
        for label in model_labels:
            row = [overlay_panel(rgb, gt, None, f"GT | actual={side} | model={label}")]
            for prompt in CLASS_NAMES:
                record = records_by_key[(label, dataset_index, prompt)]
                mask = masks[(label, dataset_index, prompt)]
                if not record["detected"]:
                    mask = np.zeros_like(mask)
                role = "correct" if record["target_present"] else "opposite/negative"
                title = (
                    f"{prompt} ({role}) | conf={record['top_confidence']:.3f} "
                    f"pres={record['presence_probability']:.3f} "
                    f"det={int(record['detected'])} "
                    f"dice={record['detected_dice_with_physical_hand']:.3f}"
                )
                row.append(overlay_panel(rgb, gt, mask, title))
            rows.append(row)

        canvas = Image.new("RGB", (rgb.width * 3, rgb.height * len(rows)))
        for row_index, row in enumerate(rows):
            for column_index, panel in enumerate(row):
                canvas.paste(panel, (column_index * rgb.width, row_index * rgb.height))
        filename = (
            f"{side}__idx-{dataset_index:05d}__{image_info['sequence']}__"
            f"{image_info['view']}__frame-{int(image_info['frame_index']):08d}.png"
        )
        canvas.save(output_dir / filename)
        manifest.append(
            {
                "dataset_index": dataset_index,
                "image_id": image_id,
                "actual_side": side,
                "file_name": image_info["file_name"],
                "output": str((output_dir / filename).resolve()),
            }
        )
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def get_visual_renderer(style: str):
    """Choose the scientific separated output or explicitly requested legacy overlay."""
    if style == "overlay":
        return render_montages
    if style == "separate":
        if __package__:
            from .render_separated_masks import render_montages as renderer
        else:
            from render_separated_masks import render_montages as renderer
        return renderer
    raise ValueError(f"Unsupported visual style: {style!r}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3 evaluation")
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(exist_ok=True)

    images, annotations_by_image = load_coco_index(args.data_root)
    dataset = make_dataset(args.data_root)
    if len(dataset) != len(images):
        raise RuntimeError(f"Dataset/COCO length mismatch: {len(dataset)} != {len(images)}")
    eval_indices = choose_indices(
        images, annotations_by_image, args.samples_per_group, args.indices
    )
    render_indices = choose_render_indices(
        eval_indices,
        images,
        annotations_by_image,
        args.render_count_per_group,
    )
    render_index_set = set(render_indices)

    learned_specs = [parse_checkpoint_spec(value) for value in args.learned_checkpoint]
    labels = [label for label, _ in learned_specs]
    if args.include_ve:
        labels.extend(["ve-underscore", "ve-natural"])
    if len(labels) != len(set(labels)):
        raise ValueError(f"Model labels must be unique: {labels}")

    all_records: list[dict] = []
    all_rendered_masks: dict[tuple[str, int, str], np.ndarray] = {}
    model_metadata = {}

    for label, checkpoint in learned_specs:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(f"Loading {label}: {checkpoint}", flush=True)
        model, metadata = load_learned_model(args.base_checkpoint, checkpoint)
        model_metadata[label] = metadata
        variant_records, variant_masks = evaluate_variant(
            model=model,
            label=label,
            prompt_texts=CLASS_NAMES,
            dataset=dataset,
            images=images,
            annotations_by_image=annotations_by_image,
            eval_indices=eval_indices,
            render_indices=render_index_set,
            batch_size=args.batch_size,
            detection_threshold=args.detection_threshold,
            mask_threshold=args.mask_threshold,
            amp=args.amp,
            unified_root=args.unified_root,
        )
        atomic_write_json(records_dir / f"{label}.json", variant_records)
        all_records.extend(variant_records)
        all_rendered_masks.update(
            {(label, index, prompt): mask for (index, prompt), mask in variant_masks.items()}
        )
        del model
        torch.cuda.empty_cache()

    if args.include_ve:
        print("Loading original VE text encoder", flush=True)
        model = load_ve_model(args.base_checkpoint)
        for label, prompt_texts in (
            ("ve-underscore", ("left_hand", "right_hand")),
            ("ve-natural", ("left hand", "right hand")),
        ):
            model_metadata[label] = {
                "kind": "ve",
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "prompt_texts": list(prompt_texts),
            }
            variant_records, variant_masks = evaluate_variant(
                model=model,
                label=label,
                prompt_texts=prompt_texts,
                dataset=dataset,
                images=images,
                annotations_by_image=annotations_by_image,
                eval_indices=eval_indices,
                render_indices=render_index_set,
                batch_size=args.batch_size,
                detection_threshold=args.detection_threshold,
                mask_threshold=args.mask_threshold,
                amp=args.amp,
                unified_root=args.unified_root,
            )
            atomic_write_json(records_dir / f"{label}.json", variant_records)
            all_records.extend(variant_records)
            all_rendered_masks.update(
                {(label, index, prompt): mask for (index, prompt), mask in variant_masks.items()}
            )
        del model
        torch.cuda.empty_cache()

    metrics = summarize(all_records, args.detection_threshold)
    visual_manifest = get_visual_renderer(args.visual_style)(
        root=args.data_root,
        output_dir=args.output_dir / "visuals",
        render_indices=render_indices,
        images=images,
        annotations_by_image=annotations_by_image,
        records=all_records,
        masks=all_rendered_masks,
        model_labels=labels,
    )
    summary = {
        "data_root": str(args.data_root.resolve()),
        "annotations_sha256": sha256(args.data_root / "annotations.json"),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "evaluated_images": len(eval_indices),
        "evaluated_dataset_indices": eval_indices,
        "detection_threshold": args.detection_threshold,
        "mask_threshold": args.mask_threshold,
        "visual_style": args.visual_style,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "confidence_definition": "sigmoid(pred_logits) * sigmoid(presence_logit_dec)",
        "models": model_metadata,
        "metrics": metrics,
        "visuals": visual_manifest,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

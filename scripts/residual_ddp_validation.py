"""Exhaustive distributed validation for the frozen SAM3 residual objective.

Validation partitions images without padding, uses the raw model (no DDP forward
collectives), and reduces task-loss numerators once every local shard finishes.
Prediction selection uses model confidence only; matching is used only for loss.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_erosion
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from scripts import residual_ddp_data as data
from scripts import residual_ddp_objective as objective
from scripts.residual_ddp_runtime import raise_if_distributed_error


THRESHOLD = .5
BOUNDARY_PIXELS = 4
_SIDE_TOTALS = ("positive_count", "candidate_dice_sum", "miss_zero_dice_sum",
                "false_negative_count", "absent_count", "false_positive_count",
                "candidate_boundary_iou_4px_sum", "miss_zero_boundary_iou_4px_sum")
_TOTAL_KEYS = ("images", "queries", "targets") + tuple(
    f"{side}/{name}" for side in data.CLASS_NAMES for name in _SIDE_TOTALS)


def validation_batches(size, batch_size, rank, world_size):
    """List-valued batches include every rank-strided index, including tails."""
    for name, value, minimum in (("size", size, 0), ("batch_size", batch_size, 1),
                                  ("rank", rank, 0), ("world_size", world_size, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")
    indices = list(range(rank, size, world_size))
    return [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]


def _references(contract, image):
    from scripts.evaluate_bilateral_tokens import decode_gt_mask

    shape = (image["height"], image["width"])
    grouped = {annotation["category_id"]: annotation
               for annotation in contract.annotations_by_image[image["id"]]}
    refs = [decode_gt_mask(grouped.get(category), *shape) for category in (1, 2)]
    for category, mask in enumerate(refs, 1):
        if mask.shape != shape or bool(mask.any()) != (category in grouped):
            raise ValueError("Original-resolution reference shape/presence differs from COCO")
    return refs


def _select_predictions(prediction, images):
    logits, presence, masks = (prediction[key] for key in (
        "pred_logits", "presence_logit_dec", "pred_masks"))
    count = 2 * len(images)
    if (logits.ndim != 3 or logits.shape[0] != count or logits.shape[1] < 1
            or logits.shape[2] != 1 or presence.shape[0] != count or presence.numel() != count
            or masks.ndim != 4 or masks.shape[:2] != logits.shape[:2]):
        raise ValueError("Prediction rows must follow two side queries per RGB image")
    if not all(bool(torch.isfinite(value).all()) for value in (logits, presence, masks)):
        raise ValueError("Nonfinite validation prediction")
    probabilities = logits.float().sigmoid().squeeze(-1)
    presence = presence.float().sigmoid().reshape(count)
    scores = probabilities * presence[:, None]
    selected = []
    for row in range(count):
        decoder = int(scores[row].argmax())
        image = images[row // 2]
        resized = F.interpolate(masks[row, decoder][None, None].float(),
                                size=(image["height"], image["width"]),
                                mode="bilinear", align_corners=False)[0, 0]
        selected.append({
            "mask": resized.sigmoid().cpu().numpy() >= THRESHOLD,
            "selected_decoder_query": decoder,
            "top_confidence": float(scores[row, decoder]),
            "top_class_probability": float(probabilities[row, decoder]),
            "presence_probability": float(presence[row]),
            "detections_above_threshold": int((scores[row] >= THRESHOLD).sum()),
        })
    return selected


def _boundary(mask):
    # Same square-erosion inner band and outside-zero handling as the existing
    # boundary diagnostics, with a fixed four ORIGINAL-image-pixel width.
    interior = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool),
                              iterations=BOUNDARY_PIXELS, border_value=0)
    return mask & ~interior


def _query_record(selected, reference, image, dataset_index, side, rank, provenance, totals):
    selected = dict(selected)
    mask = selected.pop("mask")
    present, detected = bool(reference.any()), selected["top_confidence"] >= THRESHOLD
    candidate_dice = boundary_iou = None
    if present:
        candidate_dice = 2 * int((mask & reference).sum()) / (int(mask.sum()) + int(reference.sum()))
        pred_boundary, gt_boundary = _boundary(mask), _boundary(reference)
        boundary_iou = int((pred_boundary & gt_boundary).sum()) / int((pred_boundary | gt_boundary).sum())
        totals[f"{side}/positive_count"] += 1
        totals[f"{side}/candidate_dice_sum"] += candidate_dice
        totals[f"{side}/miss_zero_dice_sum"] += candidate_dice if detected else 0.
        totals[f"{side}/false_negative_count"] += int(not detected)
        totals[f"{side}/candidate_boundary_iou_4px_sum"] += boundary_iou
        totals[f"{side}/miss_zero_boundary_iou_4px_sum"] += boundary_iou if detected else 0.
    else:
        totals[f"{side}/absent_count"] += 1
        totals[f"{side}/false_positive_count"] += int(detected)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {
        "dataset_role": "validation", "rank": rank, "dataset_index": dataset_index,
        "image_id": image["id"], "file_name": image["file_name"], "prompt_key": side,
        "original_size": [image["height"], image["width"]], "identity_verified": True,
        "provenance": provenance, "reference_present": present, "detected": detected,
        "candidate_dice": candidate_dice,
        "miss_zero_dice": (candidate_dice if detected else 0.) if present else None,
        "candidate_boundary_iou_4px": boundary_iou,
        "prediction_rle": rle, **selected,
    }


def _metrics(loss_sums, totals):
    values = {key: float(loss_sums[index] / max(totals["targets"], 1))
              for index, key in enumerate(objective.TARGET_COMPONENTS)}
    for index, key in enumerate(objective.COMPONENTS[4:], 4):
        values[key] = float(loss_sums[index] / max(totals["queries"], 1))
    values["total_loss"] = sum(values.values())
    for key in ("images", "queries", "targets"):
        values[key] = int(totals[key])
    for side in data.CLASS_NAMES:
        for key in ("positive_count", "false_negative_count", "absent_count", "false_positive_count"):
            values[f"{side}/{key}"] = int(totals[f"{side}/{key}"])
        positives, absent = totals[f"{side}/positive_count"], totals[f"{side}/absent_count"]
        if positives:
            for key in ("candidate_dice", "miss_zero_dice", "candidate_boundary_iou_4px", "miss_zero_boundary_iou_4px"):
                values[f"{side}/{key}"] = totals[f"{side}/{key}_sum"] / positives
            values[f"{side}/false_negative_rate"] = totals[f"{side}/false_negative_count"] / positives
        if absent:
            values[f"{side}/false_positive_rate"] = totals[f"{side}/false_positive_count"] / absent
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite global validation metric")
    return values


def _verify_records(output_dir, contract, world_size):
    expected = {(image["id"], side) for image in contract.images for side in data.CLASS_NAMES}
    observed = set()
    for rank in range(world_size):
        with (output_dir / f"rank-{rank}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = (record["image_id"], record["prompt_key"])
                index = record["dataset_index"]
                if (key not in expected or key in observed or not 0 <= index < len(contract.images)
                        or contract.images[index]["id"] != record["image_id"]
                        or index % world_size != rank or record["rank"] != rank
                        or record["dataset_role"] != "validation" or record["identity_verified"] is not True):
                    raise ValueError("Duplicate, substituted or mispartitioned validation query identity")
                observed.add(key)
    if observed != expected:
        raise ValueError("Validation records do not cover every image and both side queries")
    return len(observed)


def _scope(contract):
    sources = {str(image.get("source", image.get("source_dataset", ""))).lower() for image in contract.images}
    if sources == {"dexycb"}:
        return "dexycb_val"
    return "validation"


def evaluate_validation(model, contract, dataset, *, context, batch_size, num_workers,
                        output_dir, monitor=None, step=0, amp=True):
    """Return finite globally normalized metrics; all ranks must call together.

    ``dataset`` is an IdentityCheckedDataset (or supplies validated IndexedSample
    objects). Output must be a fresh directory on a filesystem shared by ranks.
    Loss is the six-component task objective only; no residual anchor is added.
    """
    output_dir = Path(output_dir)
    error = None
    try:
        if isinstance(model, DistributedDataParallel):
            raise ValueError("Validation requires raw SAM3, not its DDP wrapper")
        if len(dataset) != len(contract.images) or not contract.images:
            raise ValueError("Validation dataset must cover the nonempty approved image index")
        source_names = [str(contract.root), *(str(image.get("source_dataset", image.get("source", "")))
                                             for image in contract.images)]
        if any("realsense" in name.lower() for name in source_names):
            raise ValueError("RealSense is outside this tuning validation scope")
        if context.world_size > 1 and (not dist.is_initialized() or
                (dist.get_rank(), dist.get_world_size()) != (context.rank, context.world_size)):
            raise ValueError("Distributed context does not match the active process group")
        batches = validation_batches(len(dataset), batch_size, context.rank, context.world_size)
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if context.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        error = exc
    raise_if_distributed_error(error, context)
    if context.world_size > 1:
        dist.barrier()

    totals = dict.fromkeys(_TOTAL_KEYS, 0.)
    loss_sums = np.zeros(len(objective.COMPONENTS), dtype=np.float64)
    error = None
    try:
        model.eval()
        functions = [function.to(context.device).eval() for function in objective.shared.build_loss_functions()]
        loader = data.make_loader(dataset, batch_sampler=batches, num_workers=num_workers,
                                  pin_memory=context.device.type == "cuda", persistent_workers=False)
        rendered = 0
        with (output_dir / f"rank-{context.rank}.jsonl").open("x", encoding="utf-8") as record_file:
            with torch.no_grad():
                for cpu_batch in loader:
                    indices = cpu_batch.dataset_indices
                    images = [contract.images[index] for index in indices]
                    if (tuple(image["id"] for image in images) != cpu_batch.image_ids
                            or any(index % context.world_size != context.rank for index in indices)):
                        raise ValueError("Validation loader changed the requested image identity")
                    batch = cpu_batch.to(context.device).datapoint
                    local_images = len(images)
                    count = int(batch.find_targets[0].num_boxes.sum())
                    if count != sum(sum(contract.side_counts[image["id"]]) for image in images):
                        raise ValueError("Validation target count differs from approved annotations")
                    with torch.autocast(device_type=context.device.type, dtype=torch.bfloat16,
                                        enabled=bool(amp) and context.device.type == "cuda"):
                        _, vector, prediction = objective.loss_components(model, batch, functions, denominator=1.)
                    vector = vector.detach().double().cpu().numpy()
                    if vector.shape != loss_sums.shape or not np.isfinite(vector).all():
                        raise ValueError("Nonfinite or malformed validation loss components")
                    vector[4:] *= 2 * local_images
                    loss_sums += vector
                    totals["images"] += local_images
                    totals["queries"] += 2 * local_images
                    totals["targets"] += count
                    selected = _select_predictions(prediction, images)
                    for local_index, (index, image) in enumerate(zip(indices, images)):
                        refs = _references(contract, image)
                        render = context.rank == 0 and monitor is not None and rendered < 2
                        if render:
                            rgb_path = data._approved_image_path(contract.root, image, contract.allowed_image_roots)
                            with Image.open(rgb_path) as source:
                                rgb = np.asarray(source.convert("RGB"))
                            if rgb.shape[:2] != (image["height"], image["width"]):
                                raise ValueError("Preview RGB dimensions differ from approved COCO")
                            previews = {f"preview/image-{image['id']}/rgb": rgb}
                        for side_index, side in enumerate(data.CLASS_NAMES):
                            candidate = selected[local_index * 2 + side_index]
                            record = _query_record(candidate, refs[side_index], image, index, side,
                                                   context.rank, cpu_batch.provenance[local_index], totals)
                            record_file.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                            if render:
                                prefix = f"preview/image-{image['id']}/{side}"
                                previews[prefix + "/gt"] = refs[side_index]
                                previews[prefix + "/pred"] = (candidate["mask"] if record["detected"]
                                                                else np.zeros_like(candidate["mask"]))
                        if render:
                            monitor.log_images(step, previews)
                            rendered += 1
                    record_file.flush()
                    del batch, prediction, selected
    except Exception as exc:
        error = exc
    # Unequal/empty shards cannot call collectives inside their batch loops.
    raise_if_distributed_error(error, context)
    aggregate = torch.tensor([*loss_sums, *(totals[key] for key in _TOTAL_KEYS)],
                             dtype=torch.float64, device=context.device)
    if context.world_size > 1:
        dist.all_reduce(aggregate)
    aggregate = aggregate.cpu().tolist()
    loss_sums = aggregate[:len(objective.COMPONENTS)]
    totals = dict(zip(_TOTAL_KEYS, aggregate[len(objective.COMPONENTS):]))
    metrics = _metrics(loss_sums, totals)
    error = None
    try:
        if metrics["images"] != len(contract.images) or metrics["queries"] != 2 * len(contract.images):
            raise ValueError("Reduced validation counts differ from the approved dataset")
        if context.rank == 0:
            verified = _verify_records(output_dir, contract, context.world_size)
            scope = _scope(contract)
            summary = {"dataset_role": "validation", "scope": scope, "global_step": step,
                       "annotations_sha256": contract.annotations_sha256, "world_size": context.world_size,
                       "verified_query_records": verified, "metrics": metrics,
                       "loss_definition": "six original task terms, each weight 1; no anchor penalty",
                       "loss_normalization": "first four global target sums / max(targets,1); CE and presence query means",
                       "prediction_selection": "argmax sigmoid(class)*sigmoid(presence), never reference overlap",
                       "detection_threshold": THRESHOLD, "mask_threshold": THRESHOLD,
                       "boundary_band_original_pixels": BOUNDARY_PIXELS,
                       "undefined_metric_policy": "omit ratios with zero denominator; preserve counts"}
            (output_dir / "summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
            if monitor is not None:
                monitor.log_validation(step, metrics, scope=scope)
    except Exception as exc:
        error = exc
    raise_if_distributed_error(error, context)
    return metrics

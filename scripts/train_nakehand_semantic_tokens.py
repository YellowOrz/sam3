#!/usr/bin/env python3
"""Recording-disjoint nakehand semantic-delta pilot, with optional fixed VE anchor.

Only the explicit train split is used. The two prespecified trials differ only
in anchor weight (0 or 1); both start from the same verified natural VE cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

if __package__:
    from . import train_ve_initialized_tokens as shared
else:
    import train_ve_initialized_tokens as shared

cached = shared.cached
evaluation = shared.evaluation
legacy = shared.legacy
FORMAT = "sam3-nakehand-semantic-delta-training-v1"
LEARNING_RATE = .001
ANCHOR_EPSILON = 1e-12
TRAIN_RECORDINGS = {
    "nakehandego/20260907_134035": 629,
    "nakehandego/20260907_140713": 4084,
    "nakehandexo/20260907_123926": 4379,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--anchor-weight", type=float, choices=(0., 1.), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= 2000 or args.log_every < 1 or not 0 < args.gpu_memory_fraction <= .25:
        parser.error("Require 1..2000 max-steps, positive log-every, and GPU fraction within (0,.25]")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.tokenizer_path is None:
        args.tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if args.output_dir.exists():
        parser.error("Output must be a new directory, also for resumed processes")
    if args.data_root == args.output_dir or args.data_root in args.output_dir.parents:
        parser.error("Do not put checkpoints inside the immutable train split")
    return args


def validate_training_annotations(data, expected_counts=None):
    expected_counts = TRAIN_RECORDINGS if expected_counts is None else expected_counts
    info = data.get("info", {})
    if info.get("split") != "train" or info.get("dataset_role") != "train":
        raise ValueError("Only dataset_role=train / split=train is allowed")
    if data.get("categories") != [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]:
        raise ValueError("Expected exact left_hand=1/right_hand=2 category mapping")
    images = sorted(data.get("images", []), key=lambda image: int(image["id"]))
    ids = [int(image["id"]) for image in images]
    if len(ids) != sum(expected_counts.values()) or len(set(ids)) != len(ids):
        raise ValueError("Training image count/unique IDs differs from fixed recording plan")
    frames = {name: set() for name in expected_counts}
    for image in images:
        recording = image.get("recording_id")
        frame = image.get("source_frame_index")
        if recording not in expected_counts or type(frame) is not int or not 0 <= frame < expected_counts[recording]:
            raise ValueError("Training image leaks outside the fixed recordings/frame ranges")
        if frame in frames[recording] or image.get("frame_index") != frame:
            raise ValueError("Duplicate or inconsistent original frame index")
        frames[recording].add(frame)
    if any(len(frames[name]) != size for name, size in expected_counts.items()):
        raise ValueError("Train split must contain every frame of the three fixed recordings")
    by_id = {image_id: {} for image_id in ids}
    annotation_ids = set()
    for annotation in data.get("annotations", []):
        image_id, category = int(annotation["image_id"]), int(annotation["category_id"])
        annotation_id = int(annotation["id"])
        if image_id not in by_id or category not in (1, 2) or category in by_id[image_id]:
            raise ValueError("Each image permits at most one union mask per left/right category")
        if annotation_id in annotation_ids or annotation_id != image_id * 2 + category - 1:
            raise ValueError("Annotation ID violates the frozen global ID mapping")
        if annotation.get("area", 0) <= 0 or not annotation.get("segmentation"):
            raise ValueError("Visible-side annotation requires a nonempty reference mask")
        annotation_ids.add(annotation_id)
        by_id[image_id][category] = annotation
    side_counts = Counter(category for annotations in by_id.values() for category in annotations)
    if any(side_counts[category] == 0 for category in (1, 2)):
        raise ValueError("Both hand sides require positive training examples")
    return images, by_id, {
        "images": len(images), "annotations": len(annotation_ids),
        "negative_images": sum(not value for value in by_id.values()),
        "both_hand_images": sum(len(value) == 2 for value in by_id.values()),
        "positive_images_by_class": {"left_hand": side_counts[1], "right_hand": side_counts[2]},
        "recordings": dict(expected_counts),
    }


def _read_hashed(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_ready_dataset(data_root: Path):
    """Verify immutable exporter READY bindings before touching any training model."""
    if data_root.name != "train":
        raise ValueError("data-root must be the published train split directory")
    parent = data_root.parent
    root_ready, root_ready_hash = _read_hashed(parent / "READY.json")
    root_manifest, root_manifest_hash = _read_hashed(parent / "manifest.json")
    split_ready, split_ready_hash = _read_hashed(data_root / "READY.json")
    manifest, manifest_hash = _read_hashed(data_root / "manifest.json")
    plan, plan_hash = _read_hashed(parent / "frozen-plan.json")
    data, annotation_hash = _read_hashed(data_root / "annotations.json")
    if root_ready.get("status") != "complete" or split_ready.get("status") != "complete":
        raise ValueError("Both root and train READY must be complete")
    expected = {"annotations_sha256": annotation_hash, "manifest_sha256": manifest_hash, "frozen_plan_sha256": plan_hash}
    if any(split_ready.get(key) != value for key, value in expected.items()):
        raise ValueError("Train READY does not bind the actual annotations/manifest/frozen plan")
    if manifest.get("annotations_sha256") != annotation_hash or manifest.get("frozen_plan_sha256") != plan_hash:
        raise ValueError("Train manifest provenance differs")
    if manifest.get("sources_unchanged") is not True:
        raise ValueError("Source immutability was not verified by the exporter")
    # Export schema is deliberately strict: no guessing another split or root.
    bound = root_ready.get("splits", {}).get("train", {})
    if (bound.get("ready_sha256") != split_ready_hash or bound.get("manifest_sha256") != manifest_hash
            or bound.get("annotations_sha256") != annotation_hash):
        raise ValueError("Root READY does not bind this exact train split")
    if root_ready.get("frozen_plan_sha256") != plan_hash:
        raise ValueError("Root READY frozen-plan hash differs")
    if (root_ready.get("manifest_sha256") != root_manifest_hash or root_manifest.get("status") != "complete"
            or root_manifest.get("frozen_plan_sha256") != plan_hash
            or root_manifest.get("splits") != root_ready.get("splits")
            or root_manifest.get("sources_unchanged") is not True):
        raise ValueError("Root manifest is incomplete or not bound to READY")
    planned_train = plan.get("splits", {}).get("train", {})
    if (sorted(planned_train.get("recordings", [])) != sorted(TRAIN_RECORDINGS)
            or planned_train.get("images") != sum(TRAIN_RECORDINGS.values())
            or planned_train.get("coco_split") != "train"
            or Path(plan.get("output", "")).resolve() != parent.resolve()):
        raise ValueError("Frozen plan does not authorize the exact train recordings/count/location")
    if data.get("info", {}).get("frozen_plan_sha256") != plan_hash:
        raise ValueError("COCO does not bind the frozen plan")
    sources = manifest.get("sources", [])
    if (len(sources) != 60 or len({item["path"] for item in sources}) != 60
            or sources != plan.get("sources") or sources != root_manifest.get("sources")):
        raise ValueError("Expected the complete unchanged 60-file source fingerprint inventory")
    for item in sources:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or evaluation.sha256(path) != item["sha256"]:
            raise ValueError(f"Original source differs from the frozen export: {path}")
    images, annotations, summary = validate_training_annotations(data)
    for image in images:
        recording = plan.get("recordings", {}).get(image["recording_id"], {})
        if (recording.get("frame_count") != TRAIN_RECORDINGS[image["recording_id"]]
                or type(recording.get("global_image_id_offset")) is not int
                or int(image["id"]) != recording["global_image_id_offset"] + image["source_frame_index"]):
            raise ValueError("Global image identity differs from frozen recording offsets")
    expected_counts = {
        "images": summary["images"], "annotations": summary["annotations"],
        "left_annotations": summary["positive_images_by_class"]["left_hand"],
        "right_annotations": summary["positive_images_by_class"]["right_hand"],
        "empty_images": summary["negative_images"], "two_hand_images": summary["both_hand_images"],
        "one_hand_images": summary["images"] - summary["negative_images"] - summary["both_hand_images"],
    }
    if any(container.get("counts") != expected_counts for container in (manifest, split_ready, bound)):
        raise ValueError("COCO counts differ from publication receipts")
    validation = manifest.get("validation", {})
    if (manifest.get("status") != "complete" or manifest.get("split") != "train"
            or validation.get("all_png_sha256_and_pixels_checked") is not True
            or validation.get("all_rle_area_bbox_checked") is not True):
        raise ValueError("Full-frame PNG/RLE exporter validation is incomplete")
    image_outputs = manifest.get("image_outputs", [])
    image_by_id = {int(image["id"]): image for image in images}
    if len(image_outputs) != len(images) or {item["image_id"] for item in image_outputs} != set(image_by_id):
        raise ValueError("Image output manifest does not exactly cover train images")
    for item in image_outputs:
        rgb = item.get("files", {}).get("rgb", {})
        relative = Path(rgb.get("path", ""))
        if (str(relative) != image_by_id[item["image_id"]]["file_name"] or relative.is_absolute()
                or ".." in relative.parts or not cached._valid_sha(rgb.get("sha256"))):
            raise ValueError("RGB output path/hash differs from COCO")
    summary["sha256"] = annotation_hash
    return images, annotations, summary, manifest, {
        "root_ready_sha256": root_ready_hash, "split_ready_sha256": split_ready_hash,
        "root_manifest_sha256": root_manifest_hash,
        "split_manifest_sha256": manifest_hash, "frozen_plan_sha256": plan_hash,
        "recordings": sorted(TRAIN_RECORDINGS), "dataset_role": "train",
    }


def verify_selected_rgb(data_root, image, output):
    item = output["files"]["rgb"]
    path = data_root / image["file_name"]
    if item["path"] != image["file_name"] or evaluation.sha256(path) != item["sha256"]:
        raise RuntimeError(f"Selected RGB bytes changed: image {image['id']}")


def anchor_penalty(encoder, epsilon=ANCHOR_EPSILON):
    """Equal-side mean relative squared L2 drift over the four valid positions."""
    if encoder.delta is None or tuple(encoder.delta.shape) != (2, 4, 256):
        raise ValueError("Anchor requires the semantic delta[2,4,256]")
    valid = (~encoder.padding_cache).transpose(0, 1).unsqueeze(-1)
    energy = (encoder.resized_cache.float().square() * valid).sum(dim=(0, 2)).detach()
    delta_energy = encoder.delta.float().square().sum(dim=(1, 2))
    relative_squared = delta_energy / (energy + epsilon)
    return relative_squared.mean(), relative_squared


def apply_loss_gradients(task_loss, anchor_loss, delta, anchor_weight):
    """One model backward plus a tiny delta-only regularizer backward.

    Check task gradients independently so a nonzero anchor cannot hide a broken
    model-to-delta path. Connected, finite zero gradients on an individual frame
    are allowed; task-only nonzero coverage is checked across fixed windows.
    Zero anchor gradient at initialization is expected.
    """
    if anchor_weight not in (0., 1.) or not bool(torch.isfinite(task_loss)) or not bool(torch.isfinite(anchor_loss)):
        raise RuntimeError("Invalid task/anchor loss or unregistered anchor weight")
    task_gradient = torch.autograd.grad(task_loss, delta)[0]
    anchor_gradient = torch.autograd.grad(anchor_loss, delta)[0]
    if not bool(torch.isfinite(task_gradient).all()) or not bool(torch.isfinite(anchor_gradient).all()):
        raise RuntimeError("Nonfinite task/anchor delta gradient")
    task_norms = task_gradient.flatten(1).norm(dim=1)
    total_gradient = task_gradient + anchor_weight * anchor_gradient
    if not bool(torch.isfinite(total_gradient).all()):
        raise RuntimeError("Nonfinite combined delta gradient")
    delta.grad = total_gradient.detach()
    return {"task": task_norms.detach().cpu().tolist(),
            "anchor": anchor_gradient.flatten(1).norm(dim=1).detach().cpu().tolist(),
            "total": total_gradient.flatten(1).norm(dim=1).detach().cpu().tolist()}


def validate_task_gradient_history(history, completed_steps):
    """Validate task-only coverage at step 20 and each complete 100-step window.

    A frame with absent/saturated supervision can legitimately contribute zero
    gradients. The anchor is deliberately excluded from both norms and counts.
    """
    if len(history) != completed_steps:
        raise ValueError("Task-gradient history length differs from completed steps")
    if any(not isinstance(row, (list, tuple)) or len(row) != 2
           or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in row)
           for row in history):
        raise ValueError("Task-gradient history requires finite nonnegative left/right norms")
    windows = [(0, 20)] if completed_steps >= 20 else []
    windows.extend((end - 100, end) for end in range(100, completed_steps + 1, 100))
    for start, end in windows:
        if any(not any(history[index][side] > 0 for index in range(start, end)) for side in (0, 1)):
            raise ValueError(f"Missing nonzero task-gradient coverage in completed steps {start + 1}..{end}")
    return [sum(row[side] > 0 for row in history) for side in (0, 1)]


def validate_bilateral_batch(batch, image_id, annotations):
    observed = legacy.validate_training_batch_identity(batch, [image_id], {"left_hand": 1, "right_hand": 2})
    shared.validate_unprompted_batch(batch)
    stage, target = batch.find_inputs[0], batch.find_targets[0]
    counts = target.num_boxes.tolist()
    expected = [int(int(text_id) + 1 in annotations) for text_id in stage.text_ids.tolist()]
    if counts != expected or len(counts) != 2 or not 0 <= sum(counts) <= 2:
        raise RuntimeError("Each side's target count must match its own reference, including two positive queries")
    segments, validity = target.segments, target.is_valid_segment
    if (not isinstance(segments, torch.Tensor) or not isinstance(validity, torch.Tensor)
            or len(segments) != sum(counts) or validity.numel() != sum(counts) or not bool(validity.bool().all())):
        raise RuntimeError("Every visible-side target requires exactly one valid segmentation mask")
    if sum(counts) and (segments.ndim != 3 or not bool(segments.flatten(1).any(dim=1).all())):
        raise RuntimeError("A positive side lost its reference mask during preprocessing")
    return observed


def compute_task_loss(model, batch, functions):
    targets = model.back_convert(batch.find_targets[0])
    counts = targets["num_boxes"]
    if counts.numel() != 2 or not bool(((counts == 0) | (counts == 1)).all()) or int(counts.sum()) > 2:
        raise RuntimeError("nakehand batch1 requires two independent zero/one-side queries")
    prediction = model(batch)[0]
    indices = model.matcher(prediction, targets)
    prediction["indices"] = indices
    count = counts.sum().float().clamp(min=1)
    losses = [function(outputs=prediction, targets=targets, indices=indices, num_boxes=count) for function in functions]
    return sum(value["core_loss"] for value in losses), losses


def configuration(args, annotations, core_hashes, initial_state, provenance):
    config = shared.training_config(args, annotations, core_hashes, shared.cache_fingerprint(initial_state))
    config.update({"experiment": "nakehand_natural_ve_semantic_delta", "learning_rate": LEARNING_RATE,
                   "anchor_weight": args.anchor_weight, "anchor_epsilon": ANCHOR_EPSILON,
                   "anchor_definition": "mean_side(sum(delta_side^2)/(sum(valid_F0_side^2)+epsilon))",
                   "data_provenance": provenance, "gradient_counts_kind": "task_loss_only",
                   "task_gradient_policy": "finite_connected_each_step; nonzero_each_side_in_first20_and_each_complete100",
                   "relative_drift_timing": "post_optimizer_success", "loss_curve_timing": "pre_optimizer"})
    config["implementation_sha256"][Path(__file__).name] = evaluation.sha256(Path(__file__))
    return config


def validate_optimizer(state, completed_steps):
    groups = state.get("param_groups", [])
    if len(groups) != 1 or len(groups[0].get("params", [])) != 1:
        raise ValueError("Optimizer must own only semantic delta")
    expected = {"lr": LEARNING_RATE, "weight_decay": 0., "betas": (.9, .999), "eps": 1e-8,
                "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False,
                "foreach": None, "fused": None}
    if any(groups[0].get(key) != value for key, value in expected.items()):
        raise ValueError("Optimizer settings differ from fixed lr=.001 AdamW")
    moments = state.get("state", {})
    if completed_steps == 0:
        if moments:
            raise ValueError("Zero-step checkpoint has optimizer history")
        return
    if set(moments) != set(groups[0]["params"]):
        raise ValueError("Missing/extra optimizer states")
    moment = next(iter(moments.values()))
    step = moment.get("step")
    if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != completed_steps:
        raise ValueError("Optimizer step count differs from successful sample count")
    for key in ("exp_avg", "exp_avg_sq"):
        value = moment.get(key)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (2, 4, 256)
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError("Invalid optimizer moments")
    if not bool((moment["exp_avg_sq"] >= 0).all()):
        raise ValueError("Negative second moment")


def validate_resume(state, config, order, image_ids, initial_state):
    if state.get("format") != FORMAT:
        raise ValueError("Not a nakehand semantic-delta training checkpoint")
    legacy.validate_resume_training_config(state.get("training_config", {}), config)
    steps = state.get("next_step")
    if type(steps) is not int or not 0 <= steps <= 2000:
        raise ValueError("Invalid successful step count")
    if state.get("planned_dataset_indices") != order or state.get("planned_dataset_indices_sha256") != shared.json_hash(order):
        raise ValueError("Training sample order differs")
    expected_ids = [image_ids[index] for index in order[:steps]]
    if state.get("observed_image_ids") != expected_ids:
        raise ValueError("Observed completed images differ from their actual planned prefix")
    gradient_counts = validate_task_gradient_history(state.get("task_grad_norm_history", []), steps)
    if (state.get("progress") != shared.progress(steps, len(image_ids))
            or state.get("observed_identity") != legacy.observed_identity_provenance(0, steps, expected_ids)
            or state.get("gradient_nonzero_steps") != gradient_counts):
        raise ValueError("Progress/identity/task-gradient counters disagree")
    if state.get("data_provenance") != config["data_provenance"]:
        raise ValueError("Dataset readiness provenance differs")
    if (state.get("annotation_summary", {}).get("sha256") != config["annotations_sha256"]
            or state.get("annotation_summary", {}).get("images") != len(image_ids)
            or shared.json_hash(state.get("core_source_hashes")) != config["core_sources_sha256"]):
        raise ValueError("Annotation/core provenance differs")
    for key in ("base_checkpoint_sha256", "tokenizer_sha256", "initial_cache_sha256"):
        if state.get(key) != config[key]:
            raise ValueError(f"Resume source fingerprint differs: {key}")
    if shared.cache_fingerprint(state.get("initial_cache_state_dict", {})) != shared.cache_fingerprint(initial_state):
        raise ValueError("Semantic initialization changed")
    actual_state = state.get("cache_state_dict", {})
    for key, value in initial_state.items():
        if key == "delta":
            continue
        actual = actual_state.get(key)
        if not (isinstance(actual, torch.Tensor) and actual.dtype == value.dtype and torch.equal(actual, value)
                if isinstance(value, torch.Tensor) else actual == value):
            raise ValueError(f"Frozen VE feature changed: {key}")
    delta = actual_state.get("delta")
    if not isinstance(delta, torch.Tensor) or tuple(delta.shape) != (2, 4, 256) or delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all()):
        raise ValueError("Invalid trained delta")
    for name in ("loss_history", "task_loss_history", "anchor_loss_history"):
        values = state.get(name, [])
        if len(values) != steps or any(not math.isfinite(value) for value in values):
            raise ValueError(f"Loss curve does not match successful steps: {name}")
    for total, task, anchor in zip(state["loss_history"], state["task_loss_history"], state["anchor_loss_history"]):
        if anchor < 0 or not math.isclose(total, task + config["anchor_weight"] * anchor, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Total/task/anchor histories disagree")
    drifts = state.get("relative_drift_history", [])
    if len(drifts) != steps or any(len(row) != 2 or any(not math.isfinite(value) or value < 0 for value in row) for row in drifts):
        raise ValueError("Relative drift history does not match successful steps")
    if "rng" not in state:
        raise ValueError("Missing RNG state")
    validate_optimizer(state.get("optimizer", {}), steps)
    return steps


def make_checkpoint(*, encoder, optimizer, config, initial_state, annotation_summary, order,
                    observed_ids, loss_history, gradient_counts, core_hashes, task_history,
                    anchor_history, drift_history, task_grad_history):
    if validate_task_gradient_history(task_grad_history, len(observed_ids)) != gradient_counts:
        raise ValueError("Task-gradient history and nonzero counters disagree")
    result = shared.make_checkpoint(encoder=encoder, optimizer=optimizer, config=config, initial_state=initial_state,
                                    annotation_summary=annotation_summary, order=order, observed_ids=observed_ids,
                                    loss_history=loss_history, gradient_counts=gradient_counts, core_hashes=core_hashes)
    result.update({"format": FORMAT, "data_provenance": deepcopy(config["data_provenance"]),
                   "task_loss_history": list(task_history), "anchor_loss_history": list(anchor_history),
                   "task_grad_norm_history": deepcopy(task_grad_history),
                   "relative_drift_history": deepcopy(drift_history), "initial_relative_drift": [0., 0.]})
    return result


def snapshot_scripts(output_dir):
    result = shared.snapshot_scripts(output_dir)
    source = Path(__file__).resolve()
    target = output_dir / "code-snapshot" / source.name
    shutil.copy2(source, target)
    result.append({"source": str(source), "snapshot": str(target), "sha256": evaluation.sha256(target)})
    return result


def main(argv=None):
    process_started = time.monotonic()
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training; CPU tests validate contracts separately")
    images, references, annotations, manifest, provenance = validate_ready_dataset(args.data_root)
    image_outputs = {item["image_id"]: item for item in manifest["image_outputs"]}
    image_ids = [int(image["id"]) for image in images]
    order = legacy.build_epoch_order(len(images), 123)[:2000]
    core_hashes = shared.core_source_hashes(args.project_root)
    base_hash, tokenizer_hash = evaluation.sha256(args.base_checkpoint), evaluation.sha256(args.tokenizer_path)
    initial_hash = evaluation.sha256(args.initial_cache)
    encoder = shared.load_initial_cache(args.initial_cache, base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    initial_state = shared.cpu_state(encoder.state_dict())
    config = configuration(args, annotations, core_hashes, initial_state, provenance)
    if config["initial_cache_sha256"] != initial_hash or config["base_checkpoint_sha256"] != base_hash or config["tokenizer_sha256"] != tokenizer_hash:
        raise RuntimeError("Base/tokenizer/initial cache changed during setup")
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume is not None and validate_resume(resume, config, order, image_ids, initial_state) > args.max_steps:
        raise ValueError("Resume progress exceeds the requested process stop step")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code_snapshots = snapshot_scripts(args.output_dir)
    shared.atomic_save(args.output_dir / "initial_cache.pt", initial_state)
    evaluation.atomic_write_json(args.output_dir / "run.json", {
        "format": FORMAT, "config": config, "code_snapshots": code_snapshots, "data_provenance": provenance,
        "core_source_hashes": core_hashes, "planned_dataset_indices": order,
        "planned_image_ids": [image_ids[index] for index in order], "requested_stop_step": args.max_steps,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_scope": "lambda 0 and 1 are preregistered pilot settings, not an optimized choice",
        "reference_scope": "SAM3-assisted labels, recording-disjoint train only, no threshold tuning"})
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dataset = evaluation.make_dataset(args.data_root)
    if len(dataset) != len(image_ids):
        raise RuntimeError("Dataset/COCO count mismatch")
    model = shared.build_model_with_matcher(args)
    encoder.to(device="cuda")
    old_ve = cached.install_cached_ve_text_encoder(model, encoder)
    del old_ve
    torch.cuda.empty_cache()
    if cached.set_cached_ve_training_mode(model, train_delta=True) != {"backbone.language_backbone.delta"}:
        raise RuntimeError("Unexpected trainable parameter names")
    optimizer = torch.optim.AdamW([encoder.delta], lr=LEARNING_RATE, weight_decay=0.)
    functions = shared.build_loss_functions()
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameter_count != 2048:
        raise RuntimeError("Expected exactly 2048 trainable semantic-delta parameters")
    observed, totals, tasks, anchors, drifts, task_grad_history = [], [], [], [], [], []
    gradient_counts = [0, 0]
    if resume is not None:
        encoder.load_state_dict(resume["cache_state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        observed, totals = list(resume["observed_image_ids"]), list(resume["loss_history"])
        tasks, anchors = list(resume["task_loss_history"]), list(resume["anchor_loss_history"])
        drifts, gradient_counts = deepcopy(resume["relative_drift_history"]), list(resume["gradient_nonzero_steps"])
        task_grad_history = deepcopy(resume["task_grad_norm_history"])
        shared.restore_rng(resume["rng"])
    shared.verify_inputs(args, config, core_hashes)

    def checkpoint():
        return make_checkpoint(encoder=encoder, optimizer=optimizer, config=config, initial_state=initial_state,
                               annotation_summary=annotations, order=order, observed_ids=observed, loss_history=totals,
                               gradient_counts=gradient_counts, core_hashes=core_hashes, task_history=tasks,
                               anchor_history=anchors, drift_history=drifts, task_grad_history=task_grad_history)

    latest = args.output_dir / "nakehand_semantic_delta_latest.pt"
    shared.atomic_save(latest, checkpoint())
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    started = time.monotonic()
    try:
        for step in range(len(observed), args.max_steps):
            index = order[step]
            verify_selected_rgb(args.data_root, images[index], image_outputs[image_ids[index]])
            sample = dataset[index]
            batch = collate_fn_api([sample], dict_key="train", with_seg_masks=True)["train"]
            actual_ids = validate_bilateral_batch(batch, image_ids[index], references[image_ids[index]])
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                task_loss, terms = compute_task_loss(model, batch, functions)
            anchor_loss, _ = anchor_penalty(encoder)
            values = {"task": float(task_loss.detach().cpu()), "anchor": float(anchor_loss.detach().cpu())}
            values["total"] = values["task"] + args.anchor_weight * values["anchor"]
            norms = apply_loss_gradients(task_loss, anchor_loss, encoder.delta, args.anchor_weight)
            optimizer.step()
            if not bool(torch.isfinite(encoder.delta).all()):
                raise RuntimeError("Nonfinite updated delta; retain previous atomic recovery")
            with torch.no_grad():
                _, relative_squared = anchor_penalty(encoder)
                drift = relative_squared.sqrt().cpu().tolist()
            if any(not math.isfinite(value) for value in drift):
                raise RuntimeError("Nonfinite relative semantic drift")
            observed.extend(actual_ids)
            totals.append(values["total"])
            tasks.append(values["task"])
            anchors.append(values["anchor"])
            drifts.append(drift)
            task_grad_history.append(norms["task"])
            gradient_counts = [value + int(norm > 0) for value, norm in zip(gradient_counts, norms["task"])]
            if step + 1 == 20 or (step + 1) % 100 == 0:
                validate_task_gradient_history(task_grad_history, step + 1)
            if step == 0 or (step + 1) % args.log_every == 0:
                print(f"successful_steps={step+1}/2000 anchor_weight={args.anchor_weight:g} "
                      f"task={values['task']:.6f} anchor={values['anchor']:.6f} total={values['total']:.6f} "
                      f"drift_left={drift[0]:.6f} drift_right={drift[1]:.6f} task_grad={norms['task']} "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)
            if (step + 1) % 100 == 0:
                if shared.core_source_hashes(args.project_root) != core_hashes:
                    raise RuntimeError("SAM3 core source changed during controlled pilot")
                state = checkpoint()
                shared.atomic_save(args.output_dir / f"nakehand_semantic_step{step+1:05d}_recovery.pt", state)
                shared.atomic_save(latest, state, replace=True)
            del sample, batch, task_loss, anchor_loss, terms
    except BaseException as error:
        evaluation.atomic_write_json(args.output_dir / "failure.json", {
            "status": "failed_or_interrupted", "successful_steps_in_memory": len(observed),
            "latest_recovery_checkpoint": str(latest), "error": f"{type(error).__name__}: {error}",
            "note": "Never save a possibly mid-optimizer state; use last atomic recovery"})
        raise
    shared.verify_inputs(args, config, core_hashes)
    _, _, _, _, final_provenance = validate_ready_dataset(args.data_root)
    if final_provenance != provenance:
        raise RuntimeError("Dataset READY/manifest/frozen-plan changed during training")
    for index in order[:len(observed)]:
        verify_selected_rgb(args.data_root, images[index], image_outputs[image_ids[index]])
    final_state = checkpoint()
    suffix = "pilot_complete" if len(observed) == 2000 else "partial"
    final = args.output_dir / f"nakehand_semantic_step{len(observed):05d}_{suffix}.pt"
    shared.atomic_save(final, final_state)
    shared.atomic_save(latest, final_state, replace=True)
    summary = {"format": FORMAT, "status": "completed_requested_steps", "training_config": config,
               "data_provenance": provenance, "progress": final_state["progress"],
               "final_checkpoint": str(final), "final_checkpoint_sha256": evaluation.sha256(final),
               "observed_identity": final_state["observed_identity"], "gradient_nonzero_steps": gradient_counts,
               "last_task_loss": tasks[-1] if tasks else None, "last_anchor_loss": anchors[-1] if anchors else None,
               "last_total_loss": totals[-1] if totals else None, "last_relative_drift": drifts[-1] if drifts else [0., 0.],
               "elapsed_seconds_this_process": time.monotonic() - process_started,
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
               "trainable_parameter_count": trainable_parameter_count,
               "accuracy_evaluated": False, "note": "Short 2000-sample pilot; neither two epochs nor an accuracy improvement claim"}
    evaluation.atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "progress": summary["progress"]}), flush=True)


if __name__ == "__main__":
    main()

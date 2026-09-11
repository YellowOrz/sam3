#!/usr/bin/env python3
"""One-factor hand-boundary supervision pilot over frozen natural VE features.

Keep the same 2000-image nakehand prefix, zero delta, lr=.001 and frozen SAM3.
Only boundary_weight=0 versus 4 differs. A radius-4 symmetric reference band
is defined on the 1008x1008 training grid, with outside-image pixels zero.
This is whole-hand boundary supervision, not a forearm/wrist-specific label.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

if __package__:
    from . import train_nakehand_semantic_tokens as reference
else:
    import train_nakehand_semantic_tokens as reference

shared, cached, evaluation, legacy = reference.shared, reference.cached, reference.evaluation, reference.legacy
FORMAT = "sam3-nakehand-prompt-ablation-training-v1"
BOUNDARY_RADIUS = 4
TRAINING_SIZE = (1008, 1008)
BOUNDARY_DEFINITION = "dilate(reference,r=4)-erode(reference,r=4); square9; outside_image=0; grid=1008x1008"
BOUNDARY_REDUCTION = "sum_matched_masks(mean_all_spatial_pixels(focal_alpha0.25_gamma2*band))/max(total_gt_masks,1)"
COMPONENT_NAMES = ("loss_mask", "loss_dice", "loss_bbox", "loss_giou", "loss_ce", "presence_loss")
HISTORY_NAMES = ("task_loss_history", "boundary_loss_history", "loss_history", "relative_drift_history",
                 "task_grad_norm_history", "boundary_grad_norm_history", "total_grad_norm_history",
                 "loss_component_history", "boundary_support_history")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--boundary-weight", type=float, choices=(0., 4.), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Runtime stop at N total successful steps; planned budget stays 2000")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= 2000 or args.log_every < 1 or not 0 < args.gpu_memory_fraction <= .25:
        parser.error("Require max-steps 1..2000, positive log interval, GPU fraction (0,.25]")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.anchor_weight = 0.  # Fixed, not another independently varied factor.
    if args.tokenizer_path is None:
        args.tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if args.output_dir.exists():
        parser.error("Use a new output directory, including when resuming")
    if args.output_dir == args.data_root or args.data_root in args.output_dir.parents:
        parser.error("Do not write checkpoints inside immutable train data")
    return args


def boundary_band(reference_masks, radius=BOUNDARY_RADIUS):
    """Return [N,H,W] symmetric boundary band; explicit exterior background."""
    if not isinstance(reference_masks, torch.Tensor) or reference_masks.ndim != 3:
        raise ValueError("Boundary reference requires [N,H,W] masks")
    if type(radius) is not int or radius < 1 or min(reference_masks.shape[-2:]) < 1:
        raise ValueError("Positive integer radius and nonempty spatial dimensions required")
    masks = reference_masks.detach().float()
    if not bool(torch.isfinite(masks).all()) or not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("Boundary references must be finite binary masks")
    if not len(masks):
        return masks.bool()
    padded = F.pad(masks[:, None], (radius,) * 4, mode="constant", value=0.)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(padded, kernel, stride=1)
    eroded = -F.max_pool2d(-padded, kernel, stride=1)
    return (dilated > eroded)[:, 0]


def boundary_focal_loss(outputs, targets, indices, num_boxes, radius=BOUNDARY_RADIUS):
    """Extra matched-mask focal term, normalized by all pixels, never band area."""
    all_predictions = outputs["pred_masks"]
    selected = all_predictions[(indices[0], indices[1])]
    reference_masks = targets["masks"] if indices[2] is None else targets["masks"][indices[2]]
    valid = targets["is_valid_mask"] if indices[2] is None else targets["is_valid_mask"][indices[2]]
    if not len(selected) and reference_masks.numel() == 0 and valid.numel() == 0:
        # The actual empty-image collator stores segments as shape [0], not
        # [0,H,W]; retain original loss behavior without fabricating a mask.
        return selected.float().sum(), {"matched_masks": 0, "boundary_pixels": 0,
                                       "pixels_per_mask": TRAINING_SIZE[0] * TRAINING_SIZE[1]}
    if selected.ndim != 3 or reference_masks.ndim != 3 or len(selected) != len(reference_masks):
        raise ValueError("Matched prediction/reference mask identities differ")
    if valid.ndim != 1 or valid.numel() != len(selected) or valid.dtype != torch.bool:
        raise ValueError("Invalid per-matched-mask validity")
    selected, reference_masks = selected[valid], reference_masks[valid]
    denominator = float(num_boxes.detach()) if isinstance(num_boxes, torch.Tensor) else float(num_boxes)
    if not math.isfinite(denominator) or denominator < 1 or len(selected) > denominator:
        raise ValueError("Invalid total GT mask normalization")
    pixels_per_mask = int(reference_masks.shape[-2] * reference_masks.shape[-1])
    if not len(selected):
        # Connected zero is necessary for legitimate empty-side/empty-image
        # supervision; no target geometry or false negative mask is invented.
        return selected.float().sum(), {"matched_masks": 0, "boundary_pixels": 0,
                                       "pixels_per_mask": pixels_per_mask}
    reference_masks = reference_masks.detach().float()
    band = boundary_band(reference_masks, radius)
    logits = F.interpolate(selected[:, None].float(), size=reference_masks.shape[-2:],
                           mode="bilinear", align_corners=False)[:, 0]
    probabilities = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, reference_masks, reduction="none")
    p_t = probabilities * reference_masks + (1 - probabilities) * (1 - reference_masks)
    alpha_t = .25 * reference_masks + .75 * (1 - reference_masks)
    focal = alpha_t * ce * (1 - p_t).square()
    loss = (focal * band).flatten(1).mean(1).sum() / denominator
    return loss, {"matched_masks": len(selected), "boundary_pixels": int(band.sum()),
                  "pixels_per_mask": pixels_per_mask}


def compute_losses(model, batch, functions):
    """One model forward and one training matcher shared by both objectives."""
    targets = model.back_convert(batch.find_targets[0])
    counts = targets["num_boxes"]
    if counts.numel() != 2 or not bool(((counts == 0) | (counts == 1)).all()):
        raise RuntimeError("Require two independently zero/one-side queries")
    if int(counts.sum()) and tuple(targets["masks"].shape[-2:]) != TRAINING_SIZE:
        raise RuntimeError("The fixed boundary width requires 1008x1008 training references")
    prediction = model(batch)[0]
    indices = model.matcher(prediction, targets)
    prediction["indices"] = indices
    count = counts.sum().float().clamp(min=1)
    terms = [function(outputs=prediction, targets=targets, indices=indices, num_boxes=count)
             for function in functions]
    boundary, support = boundary_focal_loss(prediction, targets, indices, count)
    return sum(item["core_loss"] for item in terms), boundary, terms, support


def apply_loss_gradients(task_loss, boundary_loss, delta, boundary_weight):
    """Two backward traversals of one forward; track independent real gradients."""
    if boundary_weight not in (0., 4.) or not bool(torch.isfinite(task_loss)) or not bool(torch.isfinite(boundary_loss)):
        raise RuntimeError("Invalid task/boundary objective")
    task_gradient = torch.autograd.grad(task_loss, delta, retain_graph=True)[0]
    boundary_gradient = torch.autograd.grad(boundary_loss, delta)[0]
    total = task_gradient + boundary_weight * boundary_gradient
    if not all(bool(torch.isfinite(value).all()) for value in (task_gradient, boundary_gradient, total)):
        raise RuntimeError("Nonfinite task/boundary/combined gradient")
    delta.grad = total.detach()
    return {name: value.flatten(1).norm(dim=1).detach().cpu().tolist()
            for name, value in (("task", task_gradient), ("boundary", boundary_gradient), ("total", total))}


def implementation_sources():
    return sorted({Path(__file__).resolve(), *(Path(inspect.getfile(module)).resolve()
                                             for module in (reference, shared, cached, evaluation, legacy))})


def configuration(args, annotations, core_hashes, initial_state, provenance):
    config = reference.configuration(args, annotations, core_hashes, initial_state, provenance)
    config.update(experiment="nakehand_natural_ve_boundary_focal_ablation", boundary_weight=args.boundary_weight,
                  boundary_radius=BOUNDARY_RADIUS, training_mask_size=list(TRAINING_SIZE),
                  boundary_definition=BOUNDARY_DEFINITION, boundary_reduction=BOUNDARY_REDUCTION,
                  boundary_alpha=.25, boundary_gamma=2., anchor_weight=0.,
                  gradient_counts_kind="task_only; boundary and combined norms recorded separately",
                  implementation_sha256={path.name: evaluation.sha256(path) for path in implementation_sources()})
    return config


def _finite_rows(history, name, steps):
    rows = history[name]
    if len(rows) != steps or any(not isinstance(row, (list, tuple)) or len(row) != 2
                                or any(not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0
                                       for value in row) for row in rows):
        raise ValueError(f"Invalid {name}")


def validate_checkpoint_schema(state, *, minimum_samples=0, base_hash=None, tokenizer_hash=None):
    """Public evaluator contract. Validate this format directly, never relabel it.

    Returns metadata, not a claimed evaluation result. Callers must separately
    verify dataset READY, actual training IDs, core sources and cache artifacts.
    """
    if state.get("format") != FORMAT or "class_tokens" in state:
        raise ValueError("Require the dedicated prompt-boundary ablation checkpoint")
    config = state.get("training_config", {})
    expected = {"experiment": "nakehand_natural_ve_boundary_focal_ablation", "learning_rate": .001,
                "optimizer": "AdamW", "weight_decay": 0., "batch_size": 1, "seed": 123,
                "amp": True, "amp_dtype": "bfloat16", "planned_samples": 2000,
                "float32_matmul_precision": "high", "network_mode": "eval_with_delta_autograd",
                "delta_shape": [2, 4, 256], "context_length": 32, "anchor_weight": 0.,
                "boundary_radius": BOUNDARY_RADIUS, "training_mask_size": list(TRAINING_SIZE),
                "boundary_definition": BOUNDARY_DEFINITION, "boundary_reduction": BOUNDARY_REDUCTION,
                "boundary_alpha": .25, "boundary_gamma": 2., "loss_weights": shared.LOSS_WEIGHTS}
    if any(config.get(name) != value for name, value in expected.items()) or config.get("boundary_weight") not in (0., 4.):
        raise ValueError("Checkpoint differs from the one-factor boundary protocol")
    steps = state.get("next_step")
    if (type(steps) is not int or not 0 <= minimum_samples <= steps <= 2000
            or state.get("progress") != shared.progress(steps, 9092)):
        raise ValueError("Invalid/incomplete actual pilot progress")
    for name, requested in (("base_checkpoint_sha256", base_hash), ("tokenizer_sha256", tokenizer_hash),
                            ("initial_cache_sha256", None)):
        digest = config.get(name)
        if not cached._valid_sha(digest) or state.get(name) != digest or (requested is not None and digest != requested):
            raise ValueError(f"Checkpoint source fingerprint differs: {name}")
    initial, current = state.get("initial_cache_state_dict", {}), state.get("cache_state_dict", {})
    if set(initial) != set(current):
        raise ValueError("Initial/current cache keys differ")
    for name, value in initial.items():
        if name == "delta":
            continue
        actual = current[name]
        equal = (isinstance(actual, torch.Tensor) and actual.dtype == value.dtype and torch.equal(actual, value)
                 if isinstance(value, torch.Tensor) else actual == value)
        if not equal:
            raise ValueError(f"Frozen semantic cache changed: {name}")
    for cache_state, initial_mode in ((initial, True), (current, False)):
        extra = cache_state.get("_extra_state", {})
        if extra.get("mode") != "zero_delta":
            raise ValueError("Require full zero-delta semantic cache state")
        if (not isinstance(cache_state.get("delta"), torch.Tensor)
                or cache_state["delta"].dtype != torch.float32
                or tuple(cache_state["delta"].shape) != (2, 4, 256)):
            raise ValueError("Delta must be stored as FP32 [2,4,256], without implicit casting")
        encoder = cached.CachedVETextEncoder(cache_state["padding_cache"], cache_state["resized_cache"],
                                            cache_state["raw_cache"], metadata=extra["metadata"], mode="zero_delta")
        encoder.load_state_dict(cache_state, strict=True)
        if (encoder.delta.dtype != torch.float32 or encoder.resized_cache.dtype != torch.bfloat16
                or encoder.raw_cache.dtype != torch.float32 or not bool(torch.isfinite(encoder.delta).all())
                or (initial_mode and not bool((encoder.delta == 0).all()))):
            raise ValueError("Invalid semantic initialization, trained delta, or precision")
        for name in ("base_checkpoint_sha256", "tokenizer_sha256"):
            if encoder.cache_metadata[name] != config[name]:
                raise ValueError(f"Semantic cache provenance differs: {name}")
    if shared.cache_fingerprint(initial) != config.get("initial_cache_state_sha256"):
        raise ValueError("Initial semantic cache fingerprint differs")
    planned, observed = state.get("planned_dataset_indices", []), state.get("observed_image_ids", [])
    expected_order = legacy.build_epoch_order(9092, 123)[:2000]
    if (planned != expected_order or state.get("planned_dataset_indices_sha256") != shared.json_hash(planned)
            or len(observed) != steps or len(set(observed)) != steps
            or any(type(image_id) is not int or image_id < 0 for image_id in observed)
            or state.get("observed_identity") != legacy.observed_identity_provenance(0, steps, observed)):
        raise ValueError("Invalid planned/observed actual training prefix")
    if (state.get("annotation_summary", {}).get("images") != 9092
            or state["annotation_summary"].get("sha256") != config.get("annotations_sha256")
            or state.get("data_provenance") != config.get("data_provenance")
            or config.get("data_provenance", {}).get("dataset_role") != "train"
            or shared.json_hash(state.get("core_source_hashes")) != config.get("core_sources_sha256")):
        raise ValueError("Invalid training dataset/core provenance")
    if any(name not in state or len(state[name]) != steps for name in HISTORY_NAMES):
        raise ValueError("Incomplete successful-step histories")
    for name in ("task_loss_history", "boundary_loss_history", "loss_history"):
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in state[name]):
            raise ValueError(f"Nonfinite {name}")
    for total, task, boundary in zip(state["loss_history"], state["task_loss_history"], state["boundary_loss_history"]):
        if boundary < 0 or not math.isclose(total, task + config["boundary_weight"] * boundary, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Task/boundary/total loss histories disagree")
    for name in ("relative_drift_history", "boundary_grad_norm_history", "total_grad_norm_history"):
        _finite_rows(state, name, steps)
    counts = reference.validate_task_gradient_history(state["task_grad_norm_history"], steps)
    if state.get("gradient_nonzero_steps") != counts:
        raise ValueError("Task-only gradient counts disagree")
    for step, components in enumerate(state["loss_component_history"]):
        if (set(components) != set(COMPONENT_NAMES)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in components.values())
                or not math.isclose(sum(components.values()), state["task_loss_history"][step], rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError("Named task loss components disagree with task total")
    for support in state["boundary_support_history"]:
        if (set(support) != {"matched_masks", "boundary_pixels", "pixels_per_mask"}
                or any(type(value) is not int for value in support.values())
                or not 0 <= support["matched_masks"] <= 2 or support["pixels_per_mask"] != 1008 * 1008
                or not 0 <= support["boundary_pixels"] <= support["matched_masks"] * support["pixels_per_mask"]):
            raise ValueError("Invalid boundary support/normalization history")
    if "rng" not in state:
        raise ValueError("Missing optimizer-resume RNG state")
    reference.validate_optimizer(state.get("optimizer", {}), steps)
    return {"format": FORMAT, "variant": f"ve-boundary-weight-{config['boundary_weight']:g}",
            "boundary_weight": config["boundary_weight"], "anchor_weight": 0.,
            "completed_steps": steps, "pilot_complete": steps == 2000,
            "boundary_radius_training_pixels": BOUNDARY_RADIUS,
            "boundary_scope": "whole-hand reference band; not independently labelled forearm/wrist"}


def validate_resume(state, config, order, image_ids, initial_state):
    validate_checkpoint_schema(state, base_hash=config["base_checkpoint_sha256"],
                               tokenizer_hash=config["tokenizer_sha256"])
    legacy.validate_resume_training_config(state["training_config"], config)
    if state["planned_dataset_indices"] != order or state["observed_image_ids"] != [image_ids[index] for index in order[:state["next_step"]]]:
        raise ValueError("Resume actual image identities differ from this published dataset")
    if shared.cache_fingerprint(state["initial_cache_state_dict"]) != shared.cache_fingerprint(initial_state):
        raise ValueError("Resume initial semantic features differ")
    return state["next_step"]


def make_checkpoint(*, encoder, optimizer, config, initial_state, annotation_summary, order,
                    observed_ids, histories, core_hashes):
    gradient_counts = reference.validate_task_gradient_history(histories["task_grad_norm_history"], len(observed_ids))
    state = shared.make_checkpoint(encoder=encoder, optimizer=optimizer, config=config, initial_state=initial_state,
                                   annotation_summary=annotation_summary, order=order, observed_ids=observed_ids,
                                   loss_history=histories["loss_history"], gradient_counts=gradient_counts, core_hashes=core_hashes)
    state.update(format=FORMAT, data_provenance=deepcopy(config["data_provenance"]), initial_relative_drift=[0., 0.])
    state.update(deepcopy(histories))
    validate_checkpoint_schema(state)
    return state


def snapshot_scripts(output_dir):
    directory = output_dir / "code-snapshot"
    directory.mkdir()
    rows = []
    for source in implementation_sources():
        digest, target = evaluation.sha256(source), directory / source.name
        shutil.copy2(source, target)
        if evaluation.sha256(source) != digest or evaluation.sha256(target) != digest:
            raise RuntimeError("Implementation changed during code snapshot")
        rows.append({"source": str(source), "snapshot": str(target), "sha256": digest})
    return rows


def verify_inputs(args, config, core_hashes, snapshots):
    shared.verify_inputs(args, config, core_hashes)
    if {path.name: evaluation.sha256(path) for path in implementation_sources()} != config["implementation_sha256"]:
        raise RuntimeError("Prompt experiment implementation changed during training")
    for row in snapshots:
        if evaluation.sha256(Path(row["snapshot"])) != row["sha256"]:
            raise RuntimeError("Executed training snapshot changed")


def main(argv=None):
    started = time.monotonic()
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; use CPU tests for contracts")
    images, references, annotations, manifest, provenance = reference.validate_ready_dataset(args.data_root)
    image_ids = [int(row["id"]) for row in images]
    outputs = {row["image_id"]: row for row in manifest["image_outputs"]}
    order = legacy.build_epoch_order(len(images), 123)[:2000]
    core = shared.core_source_hashes(args.project_root)
    encoder = shared.load_initial_cache(args.initial_cache, base_hash=evaluation.sha256(args.base_checkpoint),
                                        tokenizer_hash=evaluation.sha256(args.tokenizer_path))
    initial = shared.cpu_state(encoder.state_dict())
    config = configuration(args, annotations, core, initial, provenance)
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume is not None and validate_resume(resume, config, order, image_ids, initial) > args.max_steps:
        raise ValueError("Resume progress exceeds requested stop step")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = snapshot_scripts(args.output_dir)
    shared.atomic_save(args.output_dir / "initial_cache.pt", initial)
    evaluation.atomic_write_json(args.output_dir / "run.json", {
        "format": FORMAT, "config": config, "code_snapshots": snapshots, "core_source_hashes": core,
        "planned_dataset_indices": order, "planned_image_ids": [image_ids[index] for index in order],
        "requested_stop_step": args.max_steps, "data_provenance": provenance,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One factor: all-pixel-normalized boundary focal weight 0 or 4; no geometry/memory training"})
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dataset = evaluation.make_dataset(args.data_root)
    if len(dataset) != len(images):
        raise RuntimeError("Dataset/COCO count differs")
    model = shared.build_model_with_matcher(args)
    encoder.to(device="cuda")
    original_ve = cached.install_cached_ve_text_encoder(model, encoder)
    del original_ve
    torch.cuda.empty_cache()
    if cached.set_cached_ve_training_mode(model, train_delta=True) != {"backbone.language_backbone.delta"}:
        raise RuntimeError("Unexpected trainable weights")
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 2048:
        raise RuntimeError("Expected only 2048 semantic delta parameters")
    optimizer = torch.optim.AdamW([encoder.delta], lr=.001, weight_decay=0.)
    functions = shared.build_loss_functions()
    observed, histories = [], {name: [] for name in HISTORY_NAMES}
    if resume is not None:
        encoder.load_state_dict(resume["cache_state_dict"])
        optimizer.load_state_dict(resume["optimizer"])
        observed = list(resume["observed_image_ids"])
        histories = {name: deepcopy(resume[name]) for name in HISTORY_NAMES}
        shared.restore_rng(resume["rng"])
    verify_inputs(args, config, core, snapshots)

    def checkpoint():
        return make_checkpoint(encoder=encoder, optimizer=optimizer, config=config, initial_state=initial,
                               annotation_summary=annotations, order=order, observed_ids=observed,
                               histories=histories, core_hashes=core)

    latest = args.output_dir / "nakehand_prompt_latest.pt"
    shared.atomic_save(latest, checkpoint())
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    try:
        for step in range(len(observed), args.max_steps):
            index = order[step]
            reference.verify_selected_rgb(args.data_root, images[index], outputs[image_ids[index]])
            sample = dataset[index]
            batch = collate_fn_api([sample], dict_key="train", with_seg_masks=True)["train"]
            actual_ids = reference.validate_bilateral_batch(batch, image_ids[index], references[image_ids[index]])
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                task_loss, boundary_loss, terms, support = compute_losses(model, batch, functions)
            components = {name: float(term[name].detach().cpu()) for term in terms for name in COMPONENT_NAMES if name in term}
            task_value, boundary_value = float(task_loss.detach().cpu()), float(boundary_loss.detach().cpu())
            total_value = task_value + args.boundary_weight * boundary_value
            if (set(components) != set(COMPONENT_NAMES)
                    or not all(math.isfinite(value) for value in components.values())
                    or not math.isclose(sum(components.values()), task_value, rel_tol=1e-5, abs_tol=1e-6)):
                raise RuntimeError("Named task components differ from the fixed six-loss objective")
            norms = apply_loss_gradients(task_loss, boundary_loss, encoder.delta, args.boundary_weight)
            optimizer.step()
            if not bool(torch.isfinite(encoder.delta).all()):
                raise RuntimeError("Nonfinite updated delta; retain previous atomic recovery")
            with torch.no_grad():
                _, relative_squared = reference.anchor_penalty(encoder)
                drift = relative_squared.sqrt().cpu().tolist()
            if len(drift) != 2 or any(not math.isfinite(value) or value < 0 for value in drift):
                raise RuntimeError("Nonfinite updated semantic drift; retain previous atomic recovery")
            observed.extend(actual_ids)
            values = {"task_loss_history": task_value, "boundary_loss_history": boundary_value,
                      "loss_history": total_value, "relative_drift_history": drift,
                      "task_grad_norm_history": norms["task"], "boundary_grad_norm_history": norms["boundary"],
                      "total_grad_norm_history": norms["total"], "loss_component_history": components,
                      "boundary_support_history": support}
            for name, value in values.items():
                histories[name].append(value)
            if step + 1 == 20 or (step + 1) % 100 == 0:
                reference.validate_task_gradient_history(histories["task_grad_norm_history"], step + 1)
            if step == 0 or (step + 1) % args.log_every == 0:
                print(f"successful_steps={step+1}/2000 boundary_weight={args.boundary_weight:g} "
                      f"task={task_value:.6f} boundary={boundary_value:.6f} total={total_value:.6f} "
                      f"task_grad={norms['task']} boundary_grad={norms['boundary']} drift={drift} "
                      f"components={components} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if (step + 1) % 100 == 0:
                verify_inputs(args, config, core, snapshots)
                state = checkpoint()
                shared.atomic_save(args.output_dir / f"nakehand_prompt_step{step+1:05d}_recovery.pt", state)
                shared.atomic_save(latest, state, replace=True)
            del sample, batch, task_loss, boundary_loss, terms
        verify_inputs(args, config, core, snapshots)
        if reference.validate_ready_dataset(args.data_root)[-1] != provenance:
            raise RuntimeError("Dataset provenance changed")
        for index in order[:len(observed)]:
            reference.verify_selected_rgb(args.data_root, images[index], outputs[image_ids[index]])
        final_state = checkpoint()
        suffix = "pilot_complete" if len(observed) == 2000 else "partial"
        final = args.output_dir / f"nakehand_prompt_step{len(observed):05d}_{suffix}.pt"
        shared.atomic_save(final, final_state)
        shared.atomic_save(latest, final_state, replace=True)
        evaluation.atomic_write_json(args.output_dir / "summary.json", {
            "format": FORMAT, "status": "completed_requested_steps", "training_config": config,
            "progress": final_state["progress"], "final_checkpoint": str(final),
            "final_checkpoint_sha256": evaluation.sha256(final), "data_provenance": provenance,
            "observed_identity": final_state["observed_identity"],
            "gradient_nonzero_steps": final_state["gradient_nonzero_steps"],
            "last_task_loss": histories["task_loss_history"][-1] if observed else None,
            "last_boundary_loss": histories["boundary_loss_history"][-1] if observed else None,
            "last_relative_drift": histories["relative_drift_history"][-1] if observed else [0., 0.],
            "trainable_parameter_count": 2048, "accuracy_evaluated": False,
            "elapsed_seconds_this_process": time.monotonic()-started,
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "note": "Short one-factor boundary pilot; whole hand boundary, not forearm GT or a proven improvement"})
        print(f"summary={args.output_dir / 'summary.json'} successful_steps={len(observed)}/2000", flush=True)
    except BaseException as error:
        evaluation.atomic_write_json(args.output_dir / "failure.json", {
            "status": "failed_or_interrupted", "successful_steps_in_memory": len(observed),
            "latest_recovery_checkpoint": str(latest), "error": f"{type(error).__name__}: {error}",
            "note": "Do not save a possibly mid-optimizer state; recover from last atomic checkpoint"})
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled nakehand input-word VE residual pilot, with every base weight frozen.

The two FP32 parameter rows are side-word/hand-word roles shared across hands,
not left/right class tokens. Zero residual recovers the natural original VE.
Only the adaptation location changes versus the unanchored output-delta pilot:
same train split, first 2000 seed-123 images, six task losses, AdamW and BF16.
An explicitly different learning rate is recorded, never called a same-LR trial.
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

if __package__:
    from . import soft_ve_prompt as soft
    from . import train_nakehand_semantic_tokens as reference
else:
    import soft_ve_prompt as soft
    import train_nakehand_semantic_tokens as reference

shared, cached, evaluation, legacy = reference.shared, reference.cached, reference.evaluation, reference.legacy
FORMAT = "sam3-nakehand-input-ve-training-v1"
ROLE_NAMES = ("side_word", "hand_word")
SIDE_NAMES = ("left_hand", "right_hand")
COMPONENT_NAMES = ("loss_mask", "loss_dice", "loss_bbox", "loss_giou", "loss_ce", "presence_loss")
HISTORY_NAMES = ("task_loss_history", "loss_history", "task_grad_norm_history",
                 "left_right_feature_grad_norm_history", "input_delta_norm_history", "loss_component_history")
FIXED_CONFIG = {
    "experiment": "nakehand_shared_input_ve_residual", "optimizer": "AdamW", "weight_decay": 0.,
    "batch_size": 1, "seed": 123, "amp": True, "amp_dtype": "bfloat16", "planned_samples": 2000,
    "float32_matmul_precision": "high", "network_mode": "eval_with_input_residual_autograd",
    "delta_shape": [2, 1024], "context_length": 32, "anchor_weight": 0., "boundary_weight": 0.,
    "loss_weights": shared.LOSS_WEIGHTS, "trainable_parameter_count": 2048,
    "input_positions": [1, 2], "shared_across_sides": True, "parameter_row_semantics": list(ROLE_NAMES),
    "feature_gradient_row_semantics": list(SIDE_NAMES), "initialization": "exact_zero_input_embedding_residual",
    "gradient_counts_kind": "task_loss_only_word_roles_not_hand_sides",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", "--max-samples", dest="max_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, choices=(123,), default=123)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.35)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= 2000 or not 1 <= args.checkpoint_every <= 2000 or args.log_every < 1:
        parser.error("Require 1..2000 total max-steps/checkpoint-every, and positive log-every")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= .1:
        parser.error("learning-rate must be finite and within (0,.1]")
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= .35:
        parser.error("gpu-memory-fraction must be finite and within (0,.35]")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.tokenizer_path is None:
        args.tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if args.output_dir.exists() or args.output_dir.is_relative_to(args.data_root):
        parser.error("Use a new output directory outside the immutable train split, including on resume")
    return args


def implementation_sources():
    return sorted({Path(__file__).resolve(), *(Path(inspect.getfile(module)).resolve()
        for module in (soft, reference, shared, cached, evaluation, legacy))})


def initial_residual_state(tokenizer_path):
    from sam3.model.tokenizer_ve import SimpleTokenizer
    tokenizer = SimpleTokenizer(bpe_path=str(tokenizer_path))
    state = {"format": soft.FORMAT, "input_delta": torch.zeros(2, 1024),
             "natural_token_ids": tokenizer(list(soft.PROMPTS), context_length=32),
             "width": 1024, "positions": list(soft.POSITIONS), "prompts": list(soft.PROMPTS),
             "shared_across_sides": True, "context_length": 32}
    validate_residual_state(state, require_zero=True)
    return state


def validate_residual_state(state, *, require_zero=False, expected_ids=None):
    if not isinstance(state, dict) or set(state) != {
        "format", "input_delta", "natural_token_ids", "width", "positions", "prompts",
        "shared_across_sides", "context_length"}:
        raise ValueError("Require only the small dedicated input-residual state, no base weights")
    expected = {"format": soft.FORMAT, "width": 1024, "positions": [1, 2],
                "prompts": list(soft.PROMPTS), "shared_across_sides": True, "context_length": 32}
    if any(state.get(name) != value for name, value in expected.items()):
        raise ValueError("Input residual architecture metadata differs")
    delta, ids = state.get("input_delta"), state.get("natural_token_ids")
    if (not isinstance(delta, torch.Tensor) or tuple(delta.shape) != (2, 1024)
            or delta.dtype != torch.float32 or not bool(torch.isfinite(delta).all())
            or (require_zero and not bool((delta == 0).all()))):
        raise ValueError("Require finite FP32 [2,1024] residual and exact zero initialization")
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or tuple(ids.shape) != (2, 32)
            or not torch.equal(ids.ne(0).cpu(), torch.arange(32).lt(4).expand(2, 32))
            or bool((ids < 0).any()) or ids[0, 0] != ids[1, 0] or ids[0, 3] != ids[1, 3]
            or ids[0, 1] == ids[1, 1] or ids[0, 2] != ids[1, 2]
            or (expected_ids is not None and not torch.equal(ids.cpu(), expected_ids.cpu()))):
        raise ValueError("Invalid natural tokenizer identity or side/hand word positions")
    return state


def configuration(args, annotations, core_hashes, initial_cache_state, initial_input_state, provenance):
    config = shared.training_config(args, annotations, core_hashes, shared.cache_fingerprint(initial_cache_state))
    config.update(deepcopy(FIXED_CONFIG))
    config.update(learning_rate=args.learning_rate, data_provenance=deepcopy(provenance),
                  initial_input_state_sha256=shared.cache_fingerprint(initial_input_state),
                  implementation_sha256={p.name: evaluation.sha256(p) for p in implementation_sources()},
                  task_gradient_policy="finite_connected_each_step; both word roles and both feature sides covered in first20 and each completed100",
                  loss_curve_timing="pre_optimizer", input_delta_norm_timing="post_optimizer_success",
                  initial_cache_role="independent natural VE semantic baseline identity; not the trainable encoder")
    return config


def validate_optimizer(state, completed_steps, learning_rate):
    groups, moments = state.get("param_groups", []), state.get("state", {})
    if len(groups) != 1 or len(groups[0].get("params", [])) != 1:
        raise ValueError("AdamW must own only the input residual parameter")
    expected = {"lr": learning_rate, "weight_decay": 0., "betas": (.9, .999), "eps": 1e-8,
                "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False,
                "foreach": None, "fused": None}
    if any(groups[0].get(name) != value for name, value in expected.items()):
        raise ValueError("AdamW settings differ from the checkpoint configuration")
    if completed_steps == 0:
        if moments:
            raise ValueError("Zero-step checkpoint has optimizer history")
        return
    if set(moments) != set(groups[0]["params"]):
        raise ValueError("Missing or extra optimizer state")
    moment = next(iter(moments.values()))
    step = moment.get("step")
    if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != completed_steps:
        raise ValueError("AdamW step differs from successful sample count")
    for name in ("exp_avg", "exp_avg_sq"):
        value = moment.get(name)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or tuple(value.shape) != (2, 1024) or not bool(torch.isfinite(value).all())):
            raise ValueError("Invalid input-residual optimizer moments")
    if bool((moment["exp_avg_sq"] < 0).any()):
        raise ValueError("Negative second moments")


def gradient_coverage(history, steps, label):
    # Reuse the numerical window check, not its old left/right parameter meaning.
    try:
        return reference.validate_task_gradient_history(history, steps)
    except ValueError as error:
        raise ValueError(f"Invalid {label} gradient coverage: {error}") from error


def apply_task_gradients(loss, delta, features):
    """One backward traversal; separately observe shared parameter and side paths."""
    if (not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss))
            or tuple(delta.shape) != (2, 1024) or not isinstance(features, torch.Tensor)
            or tuple(features.shape) != (32, 2, 256)):
        raise RuntimeError("Invalid task loss/input residual/natural VE feature contract")
    role_gradient, feature_gradient = torch.autograd.grad(loss, (delta, features))
    if not all(bool(torch.isfinite(value).all()) for value in (role_gradient, feature_gradient)):
        raise RuntimeError("Nonfinite task gradient; no optimizer update is allowed")
    delta.grad = role_gradient.detach()
    return {"word_roles": role_gradient.float().norm(dim=1).detach().cpu().tolist(),
            "hand_sides": feature_gradient.float().permute(1, 0, 2).flatten(1).norm(dim=1).detach().cpu().tolist()}


def named_loss_values(terms, loss):
    values = {name: float(term[name].detach().cpu()) for term in terms for name in COMPONENT_NAMES if name in term}
    total = float(loss.detach().cpu())
    if (set(values) != set(COMPONENT_NAMES) or not math.isfinite(total)
            or any(not math.isfinite(value) for value in values.values())
            or not math.isclose(sum(values.values()), total, rel_tol=1e-5, abs_tol=1e-6)):
        raise RuntimeError("Six named task losses must be finite and sum to the unchanged objective")
    return total, values


def validate_checkpoint_schema(state, *, minimum_samples=0, base_hash=None, tokenizer_hash=None):
    """Public CPU schema gate. Live data, code and tokenizer checks are separate.

    Returns metadata; never relabel this input-residual format as an output cache.
    `verify_training_identity` binds its sample prefix to the actual READY train.
    """
    if (not isinstance(state, dict) or state.get("format") != FORMAT
            or "class_tokens" in state or "cache_state_dict" in state):
        raise ValueError("Require the dedicated nakehand input VE training checkpoint")
    config = state.get("training_config", {})
    lr = config.get("learning_rate")
    if (any(config.get(name) != value for name, value in FIXED_CONFIG.items())
            or not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 < lr <= .1):
        raise ValueError("Checkpoint differs from the input VE controlled configuration")
    steps = state.get("next_step")
    if (type(steps) is not int or type(minimum_samples) is not int or not 0 <= minimum_samples <= steps <= 2000
            or state.get("progress") != shared.progress(steps, 9092)):
        raise ValueError("Invalid or insufficient successful pilot progress")
    for name, expected in (("base_checkpoint_sha256", base_hash), ("tokenizer_sha256", tokenizer_hash),
                           ("initial_cache_sha256", None)):
        digest = config.get(name)
        if not cached._valid_sha(digest) or state.get(name) != digest or (expected is not None and digest != expected):
            raise ValueError(f"Checkpoint source fingerprint differs: {name}")
    initial = validate_residual_state(state.get("initial_input_residual_state"), require_zero=True)
    current = validate_residual_state(state.get("input_residual_state"), expected_ids=initial["natural_token_ids"])
    if shared.cache_fingerprint(initial) != config.get("initial_input_state_sha256"):
        raise ValueError("Initial input residual identity differs")
    if steps == 0 and not bool((current["input_delta"] == 0).all()):
        raise ValueError("Zero-step checkpoint already contains an updated input residual")
    initial_cache = state.get("initial_cache_state_dict", {})
    extra = initial_cache.get("_extra_state", {})
    if extra.get("mode") != "zero_delta" or shared.cache_fingerprint(initial_cache) != config.get("initial_cache_state_sha256"):
        raise ValueError("Initial natural VE baseline cache identity differs")
    cache = cached.CachedVETextEncoder(initial_cache["padding_cache"], initial_cache["resized_cache"],
        initial_cache["raw_cache"], metadata=extra["metadata"], mode="zero_delta")
    cache.load_state_dict(initial_cache, strict=True)
    if (cache.resized_cache.dtype != torch.bfloat16 or cache.raw_cache.dtype != torch.float32
            or cache.delta.dtype != torch.float32 or not bool((cache.delta == 0).all())):
        raise ValueError("Invalid reference VE precision/zero delta")
    for name in ("base_checkpoint_sha256", "tokenizer_sha256"):
        if cache.cache_metadata.get(name) != config[name]:
            raise ValueError("Reference cache provenance differs from the original VE")
    if not torch.equal(cache.padding_cache.cpu(), initial["natural_token_ids"].eq(0).cpu()):
        raise ValueError("Natural cache/input tokenizer padding identity differs")
    planned, observed = state.get("planned_dataset_indices"), state.get("observed_image_ids", [])
    if (planned != legacy.build_epoch_order(9092, 123)[:2000]
            or state.get("planned_dataset_indices_sha256") != shared.json_hash(planned)
            or len(observed) != steps or len(set(observed)) != steps
            or any(type(image_id) is not int or image_id < 0 for image_id in observed)
            or state.get("observed_identity") != legacy.observed_identity_provenance(0, steps, observed)):
        raise ValueError("Invalid planned or observed successful training prefix")
    if (state.get("annotation_summary", {}).get("images") != 9092
            or state["annotation_summary"].get("sha256") != config.get("annotations_sha256")
            or not cached._valid_sha(config.get("annotations_sha256"))
            or state.get("data_provenance") != config.get("data_provenance")
            or config.get("data_provenance", {}).get("dataset_role") != "train"
            or shared.json_hash(state.get("core_source_hashes")) != config.get("core_sources_sha256")):
        raise ValueError("Invalid dataset/core provenance")
    if any(name not in state or len(state[name]) != steps for name in HISTORY_NAMES):
        raise ValueError("Successful-step histories are incomplete")
    for index, (task, total, components) in enumerate(zip(state["task_loss_history"], state["loss_history"], state["loss_component_history"])):
        if (not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (task, total))
                or task != total or set(components) != set(COMPONENT_NAMES)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in components.values())
                or not math.isclose(sum(components.values()), task, rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError(f"Six-loss history differs at successful step {index + 1}")
    for row in state["input_delta_norm_history"]:
        if len(row) != 2 or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in row):
            raise ValueError("Invalid post-update word-role residual norm")
    if steps and not torch.allclose(torch.tensor(state["input_delta_norm_history"][-1]), current["input_delta"].norm(dim=1).cpu(), rtol=1e-5, atol=1e-7):
        raise ValueError("Last norm does not describe the stored successful input residual")
    if state.get("gradient_nonzero_steps") != gradient_coverage(state["task_grad_norm_history"], steps, "word-role"):
        raise ValueError("Word-role gradient counters disagree")
    if state.get("left_right_feature_gradient_nonzero_steps") != gradient_coverage(state["left_right_feature_grad_norm_history"], steps, "left/right feature"):
        raise ValueError("Left/right feature-gradient counters disagree")
    rng = state.get("rng", {})
    if (set(rng) != {"python", "torch_cpu", "torch_cuda", "numpy"}
            or not isinstance(rng["torch_cpu"], torch.Tensor) or rng["torch_cpu"].dtype != torch.uint8):
        raise ValueError("Missing/invalid optimizer-resume RNG state")
    validate_optimizer(state.get("optimizer", {}), steps, lr)
    return {"format": FORMAT, "variant": "ve-input-shared", "completed_steps": steps,
            "pilot_complete": steps == 2000, "learning_rate": lr, "trainable_parameters": 2048,
            "parameter_row_semantics": list(ROLE_NAMES), "shared_across_sides": True,
            "anchor_weight": 0., "boundary_weight": 0.}


def verify_training_identity(state):
    # This utility is dataset/seed/progress-only and does not reinterpret formats.
    if __package__:
        from .evaluate_nakehand_semantic_tokens import verify_training_identity as verify
    else:
        from evaluate_nakehand_semantic_tokens import verify_training_identity as verify
    return verify(state)


def validate_resume(state, config, order, image_ids, initial_input_state, initial_cache_state):
    validate_checkpoint_schema(state, base_hash=config["base_checkpoint_sha256"], tokenizer_hash=config["tokenizer_sha256"])
    legacy.validate_resume_training_config(state["training_config"], config)
    if (state["planned_dataset_indices"] != order
            or state["observed_image_ids"] != [image_ids[index] for index in order[:state["next_step"]]]
            or shared.cache_fingerprint(state["initial_input_residual_state"]) != shared.cache_fingerprint(initial_input_state)
            or shared.cache_fingerprint(state["initial_cache_state_dict"]) != shared.cache_fingerprint(initial_cache_state)):
        raise ValueError("Resume differs from actual image prefix or zero natural VE initialization")
    return state["next_step"]


def make_checkpoint(*, encoder, optimizer, config, initial_input_state, initial_cache_state,
                    annotation_summary, order, observed_ids, histories, core_hashes):
    steps = len(observed_ids)
    state = {"format": FORMAT, "input_residual_state": encoder.residual_state(),
             "initial_input_residual_state": shared.cpu_state(initial_input_state),
             "initial_cache_state_dict": shared.cpu_state(initial_cache_state), "training_config": deepcopy(config),
             "annotation_summary": deepcopy(annotation_summary), "optimizer": deepcopy(optimizer.state_dict()),
             "next_step": steps, "progress": shared.progress(steps, annotation_summary["images"]),
             "planned_dataset_indices": list(order), "planned_dataset_indices_sha256": shared.json_hash(order),
             "observed_image_ids": list(observed_ids), "observed_identity": legacy.observed_identity_provenance(0, steps, observed_ids),
             "gradient_nonzero_steps": gradient_coverage(histories["task_grad_norm_history"], steps, "word-role"),
             "left_right_feature_gradient_nonzero_steps": gradient_coverage(histories["left_right_feature_grad_norm_history"], steps, "left/right feature"),
             "core_source_hashes": deepcopy(core_hashes), "data_provenance": deepcopy(config["data_provenance"]),
             "rng": shared.rng_state(), **deepcopy(histories),
             **{key: config[key] for key in ("base_checkpoint_sha256", "tokenizer_sha256", "initial_cache_sha256")}}
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
            raise RuntimeError("Implementation changed during snapshot")
        rows.append({"source": str(source), "snapshot": str(target), "sha256": digest})
    return rows


def verify_inputs(args, config, core_hashes, snapshots):
    shared.verify_inputs(args, config, core_hashes)
    if {path.name: evaluation.sha256(path) for path in implementation_sources()} != config["implementation_sha256"]:
        raise RuntimeError("Input VE implementation changed during the controlled run")
    for row in snapshots:
        if evaluation.sha256(Path(row["snapshot"])) != row["sha256"]:
            raise RuntimeError("Frozen script snapshot changed")


def frozen_parameter_versions(model, delta):
    parameters = [(name, parameter, parameter._version) for name, parameter in model.named_parameters() if parameter is not delta]
    if any(parameter.requires_grad for _, parameter, _ in parameters):
        raise RuntimeError("A base parameter was unfrozen")
    return parameters


def verify_frozen_parameters(versions):
    if any(parameter.requires_grad or parameter.grad is not None or parameter._version != version for _, parameter, version in versions):
        raise RuntimeError("A supposedly frozen base parameter changed or received gradients")


def main(argv=None):
    process_started = time.monotonic()
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training; CPU tests validate contracts separately")
    images, references, annotations, manifest, provenance = reference.validate_ready_dataset(args.data_root)
    outputs = {row["image_id"]: row for row in manifest["image_outputs"]}
    image_ids = [int(image["id"]) for image in images]
    order = legacy.build_epoch_order(len(images), args.seed)[:2000]
    core = shared.core_source_hashes(args.project_root)
    base_hash, tokenizer_hash = evaluation.sha256(args.base_checkpoint), evaluation.sha256(args.tokenizer_path)
    cache_encoder = shared.load_initial_cache(args.initial_cache, base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    initial_cache = shared.cpu_state(cache_encoder.state_dict())
    initial_input = initial_residual_state(args.tokenizer_path)
    config = configuration(args, annotations, core, initial_cache, initial_input, provenance)
    resume = None
    if args.resume:
        resume_hash = evaluation.sha256(args.resume)
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if evaluation.sha256(args.resume) != resume_hash:
            raise RuntimeError("Resume checkpoint changed while reading")
        if validate_resume(resume, config, order, image_ids, initial_input, initial_cache) > args.max_steps:
            raise ValueError("Resume progress exceeds this process's total stop step")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = snapshot_scripts(args.output_dir)
    evaluation.atomic_write_json(args.output_dir / "run.json", {
        "format": FORMAT, "config": config, "core_source_hashes": core, "code_snapshots": snapshots,
        "requested_stop_step": args.max_steps, "checkpoint_every": args.checkpoint_every,
        "gpu_memory_fraction": args.gpu_memory_fraction, "planned_dataset_indices": order,
        "planned_image_ids": [image_ids[index] for index in order], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "resume_checkpoint_sha256": resume_hash if args.resume else None,
        "reference_scope": "SAM3-assisted train references; no val/test optimization or threshold fitting",
        "method_scope": "input word embedding residual, not standard CoOp; two shared word-role rows, not two class rows"})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    latest = args.output_dir / "nakehand_input_ve_latest.pt"
    observed, histories = [], {name: [] for name in HISTORY_NAMES}
    try:
        dataset = evaluation.make_dataset(args.data_root)
        if len(dataset) != len(image_ids):
            raise RuntimeError("COCO/loader image count differs")
        model = shared.build_model_with_matcher(args)
        encoder = soft.install_shared_input_ve(model)
        if shared.cache_fingerprint(encoder.residual_state()) != shared.cache_fingerprint(initial_input):
            raise RuntimeError("Loaded model's zero residual/tokenization differs from the declared original VE")
        names = soft.set_input_ve_training_mode(model, train_residual=True)
        if names != {"backbone.language_backbone.input_delta"} or sum(p.numel() for p in model.parameters() if p.requires_grad) != 2048:
            raise RuntimeError("Only 2048 shared input residual parameters may train")
        frozen_versions = frozen_parameter_versions(model, encoder.input_delta)
        optimizer = torch.optim.AdamW([encoder.input_delta], lr=args.learning_rate, weight_decay=0.)
        functions = shared.build_loss_functions()
        if resume is not None:
            encoder.load_residual_state(resume["input_residual_state"])
            optimizer.load_state_dict(resume["optimizer"])
            observed = list(resume["observed_image_ids"])
            histories = {name: deepcopy(resume[name]) for name in HISTORY_NAMES}
            shared.restore_rng(resume["rng"])
        verify_inputs(args, config, core, snapshots)

        def checkpoint():
            verify_frozen_parameters(frozen_versions)
            return make_checkpoint(encoder=encoder, optimizer=optimizer, config=config,
                initial_input_state=initial_input, initial_cache_state=initial_cache,
                annotation_summary=annotations, order=order, observed_ids=observed, histories=histories, core_hashes=core)

        shared.atomic_save(latest, checkpoint())
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.data.collator import collate_fn_api
        features_seen = []
        hook = encoder.register_forward_hook(lambda _module, _inputs, output: features_seen.append(output[1]))
        try:
            for step in range(len(observed), args.max_steps):
                index = order[step]
                reference.verify_selected_rgb(args.data_root, images[index], outputs[image_ids[index]])
                sample = dataset[index]
                batch = collate_fn_api([sample], dict_key="train", with_seg_masks=True)["train"]
                actual_ids = reference.validate_bilateral_batch(batch, image_ids[index], references[image_ids[index]])
                batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                features_seen.clear()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    task_loss, terms = reference.compute_task_loss(model, batch, functions)
                if len(features_seen) != 1:
                    raise RuntimeError("Expected one natural VE forward for both hand prompts")
                task_value, components = named_loss_values(terms, task_loss)
                norms = apply_task_gradients(task_loss, encoder.input_delta, features_seen[0])
                features_seen.clear()
                optimizer.step()
                if not bool(torch.isfinite(encoder.input_delta).all()):
                    raise RuntimeError("Nonfinite updated input residual; retain last atomic recovery")
                delta_norms = encoder.input_delta.detach().norm(dim=1).cpu().tolist()
                observed.extend(actual_ids)
                values = {"task_loss_history": task_value, "loss_history": task_value,
                          "loss_component_history": components, "task_grad_norm_history": norms["word_roles"],
                          "left_right_feature_grad_norm_history": norms["hand_sides"], "input_delta_norm_history": delta_norms}
                for name, value in values.items():
                    histories[name].append(value)
                if step + 1 == 20 or (step + 1) % 100 == 0:
                    gradient_coverage(histories["task_grad_norm_history"], step + 1, "word-role")
                    gradient_coverage(histories["left_right_feature_grad_norm_history"], step + 1, "left/right feature")
                    verify_frozen_parameters(frozen_versions)
                if step == 0 or (step + 1) % args.log_every == 0:
                    print(f"successful_steps={step+1}/2000 input_ve lr={args.learning_rate:g} task={task_value:.6f} "
                          f"word_role_grad={norms['word_roles']} left_right_feature_grad={norms['hand_sides']} "
                          f"input_delta_norm={delta_norms} components={components} elapsed={time.monotonic()-process_started:.1f}s", flush=True)
                if (step + 1) % args.checkpoint_every == 0:
                    verify_inputs(args, config, core, snapshots)
                    state = checkpoint()
                    shared.atomic_save(args.output_dir / f"nakehand_input_ve_step{step+1:05d}_recovery.pt", state)
                    shared.atomic_save(latest, state, replace=True)
                del sample, batch, task_loss, terms
        finally:
            hook.remove()
            features_seen.clear()
        verify_inputs(args, config, core, snapshots)
        if reference.validate_ready_dataset(args.data_root)[-1] != provenance:
            raise RuntimeError("Train READY/source provenance changed")
        for index in order[:len(observed)]:
            reference.verify_selected_rgb(args.data_root, images[index], outputs[image_ids[index]])
        state = checkpoint()
        suffix = "pilot_complete" if len(observed) == 2000 else "partial"
        final = args.output_dir / f"nakehand_input_ve_step{len(observed):05d}_{suffix}.pt"
        shared.atomic_save(final, state)
        shared.atomic_save(latest, state, replace=True)
        evaluation.atomic_write_json(args.output_dir / "summary.json", {
            "format": FORMAT, "status": "completed_requested_steps", "training_config": config,
            "progress": state["progress"], "data_provenance": provenance, "observed_identity": state["observed_identity"],
            "final_checkpoint": str(final), "final_checkpoint_sha256": evaluation.sha256(final),
            "gradient_nonzero_steps": state["gradient_nonzero_steps"], "parameter_row_semantics": list(ROLE_NAMES),
            "left_right_feature_gradient_nonzero_steps": state["left_right_feature_gradient_nonzero_steps"],
            "last_task_loss": histories["task_loss_history"][-1] if observed else None,
            "last_input_delta_norm": histories["input_delta_norm_history"][-1] if observed else [0., 0.],
            "trainable_parameter_count": 2048, "original_parameters_unchanged_by_version_counter": True,
            "accuracy_evaluated": False, "elapsed_seconds_this_process": time.monotonic()-process_started,
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
            "note": "2000-sample pilot, not a full epoch or evidence of accuracy gains; input/output LR magnitudes need not have equal functional effect"})
        print(f"summary={args.output_dir / 'summary.json'} successful_steps={len(observed)}/2000", flush=True)
    except BaseException as error:
        evaluation.atomic_write_json(args.output_dir / "failure.json", {
            "status": "failed_or_interrupted", "successful_steps_in_memory": len(observed),
            "latest_recovery_checkpoint": str(latest) if latest.exists() else None,
            "error": f"{type(error).__name__}: {error}",
            "note": "Never save a possibly mid-optimizer state; use previous atomic recovery"})
        raise


if __name__ == "__main__":
    main()

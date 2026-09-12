#!/usr/bin/env python3
"""Train zero-initialized output residuals with torchrun, frozen SAM3 and TensorBoard.

Run as a module: torchrun --standalone --nproc_per_node=3 --module scripts.train_residual_ddp ...
The hash-bound selection approval is mandatory. RealSense test data is never read
by this entry point. A new output directory is required even when resuming.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts import cached_ve_text_features as cached
from scripts import train_ve_initialized_tokens as shared
from scripts.residual_ddp_checkpoint import (
    atomic_checkpoint, canonical_hash, make_checkpoint, progress_at, validate_resume,
    validate_rank_cache_consistency, pending_selection_validation_step,
    validate_validation_state,
)
from scripts.residual_ddp_data import load_coco_contract, make_identity_dataset, make_loader
from scripts.residual_ddp_objective import (
    COMPONENTS, NORMALIZATION, ResidualObjective, assert_frozen_versions,
    frozen_versions, mean_across_ranks, require_finite_everywhere,
)
from scripts.residual_ddp_runtime import (
    DeterministicGlobalBatchSampler, capture_rng_state, cleanup_distributed,
    initialize_distributed, raise_if_distributed_error, restore_rng_state, torchrun_ranks,
)
from scripts.residual_training_monitor import TrainingMonitor
from scripts.residual_selection import METRIC_NAME, new_selection_state, update_selection


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Relocated copy of the approved train split")
    parser.add_argument("--val-root", type=Path, help="Relocated copy of the approved validation split")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-cache", type=Path, required=True)
    parser.add_argument("--residual-positions", choices=("all", "content"), default="all",
                        help="Update all four valid positions, or only the two body-token positions")
    parser.add_argument("--tokenizer-path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size-per-rank", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--anchor-weight", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.6)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--stop-after-step", type=int,
                        help="Optional bounded smoke/resume stop; does not alter planned epochs")
    parser.add_argument("--validation-every-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop after this many evaluated epochs without a >min-delta gain")
    parser.add_argument("--early-stopping-min-delta", type=float, default=.001,
                        help="Absolute macro miss-zero Dice gain required to reset patience; best saves any gain")
    parser.add_argument("--initial-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true", help="CPU-only validation, no training or file writes")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size_per_rank", "prefetch_factor", "log_every", "checkpoint_every",
                 "validation_every_epochs", "early_stopping_patience"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.num_workers < 0 or args.seed < 0:
        parser.error("num-workers and seed must be nonnegative")
    for name in ("learning_rate", "anchor_weight", "gpu_memory_fraction", "early_stopping_min_delta"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"{name} must be finite")
    if args.learning_rate <= 0 or args.anchor_weight < 0 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Invalid learning rate, anchor weight or GPU memory limit")
    if args.stop_after_step is not None and args.stop_after_step < 1:
        parser.error("stop-after-step must be positive")
    if args.early_stopping_min_delta < 0:
        parser.error("early-stopping-min-delta must be nonnegative")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    return args


def approval_contracts(args):
    raw = args.approval.read_bytes()
    approval = json.loads(raw)
    if approval.get("format") != "sam3-residual-training-selection-v1" or approval.get("approved_by") != "user":
        raise ValueError("An explicit user-selected train/validation approval is required")
    contracts = {}
    for role in ("train", "val"):
        if role == "val" and not args.validation:
            continue
        declaration = approval.get(role, {})
        root = (args.data_root if role == "train" else args.val_root) or Path(declaration["root"])
        contract = load_coco_contract(root, approved_exhaustive=declaration.get("exhaustive_hand_labels") is True,
                                      allowed_image_roots=declaration.get("allowed_image_roots", []))
        if contract.annotations_sha256 != declaration.get("annotations_sha256"):
            raise ValueError(f"{role} annotation fingerprint differs from the approved selection")
        info = json.loads((contract.root / "annotations.json").read_bytes()).get("info", {})
        stated_role = info.get("dataset_role", info.get("split"))
        if stated_role is not None and stated_role not in (("train",) if role == "train" else ("val", "validation")):
            raise ValueError(f"Refusing to use {stated_role!r} data as {role}")
        if ("realsense" in str(contract.root).lower()
                or any("realsense" in str(image.get("source_dataset", image.get("source", ""))).lower()
                       for image in contract.images)):
            raise ValueError("RealSense is reserved for testing, not training or validation selection")
        contracts[role] = contract
        if role == "val" and any(str(row.get("source_dataset", row.get("source", ""))).lower()
                                 != "dexycb" for row in contract.images):
            raise ValueError("Best/early-stop selection requires the fixed approved DexYCB validation split")
    if "val" in contracts:
        train_paths = {(contracts["train"].root / row["file_name"]).resolve() for row in contracts["train"].images}
        val_paths = {(contracts["val"].root / row["file_name"]).resolve() for row in contracts["val"].images}
        if train_paths & val_paths or contracts["train"].annotations_sha256 == contracts["val"].annotations_sha256:
            raise ValueError("Train/validation images overlap")
    return approval, hashlib.sha256(raw).hexdigest(), contracts


def implementation_hashes():
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / "sam3").rglob("*.py"))
    paths += sorted((project / "scripts").glob("residual_*.py"))
    paths += [Path(__file__).resolve(), project / "scripts/cached_ve_text_features.py",
              project / "scripts/train_ve_initialized_tokens.py", project / "scripts/train_nakehand_semantic_tokens.py",
              project / "scripts/evaluate_bilateral_tokens.py"]
    return {str(path.relative_to(project)): shared.evaluation.sha256(path) for path in paths}


def training_configuration(args, contract, approval_hash, world_size, initial_state, fingerprints,
                           validation_contract=None):
    global_batch = world_size * args.batch_size_per_rank
    steps = len(contract.images) // global_batch
    if not steps:
        raise ValueError("The selected dataset is smaller than one complete global batch")
    validation_config = {"enabled": False}
    if args.validation:
        if validation_contract is None:
            raise ValueError("Validation configuration requires its approved annotation contract")
        validation_config = {
            "enabled": True, "initial_validation": args.initial_validation,
            "every_epochs": args.validation_every_epochs,
            "annotations_sha256": validation_contract.annotations_sha256,
            "selection_policy": {"patience": args.early_stopping_patience,
                                 "min_delta": args.early_stopping_min_delta},
        }
    positions = getattr(args, "residual_positions", "all")
    mode = "zero_delta" if positions == "all" else "content_delta"
    if positions not in ("all", "content"):
        raise ValueError("Unknown residual position policy")
    delta_shape = cached.expected_delta_shape(mode)
    if (cached.validate_delta_state(initial_state) != delta_shape
            or initial_state["_extra_state"]["mode"] != mode
            or not bool((initial_state["delta"] == 0).all())):
        raise ValueError("Initial cache must match the residual policy and contain exactly zero delta")
    config = {"method": "original_natural_VE_plus_zero_initialized_output_delta",
            "dataset_size": len(contract.images), "annotations_sha256": contract.annotations_sha256,
            "image_order_sha256": canonical_hash([row["id"] for row in contract.images]),
            "approval_sha256": approval_hash, "world_size": world_size,
            "batch_size_per_rank": args.batch_size_per_rank, "global_batch_size": global_batch,
            "gradient_accumulation_steps": 1, "epochs": args.epochs, "steps_per_epoch": steps,
            "seed": args.seed, "learning_rate": args.learning_rate, "optimizer": "AdamW", "weight_decay": 0.,
            "anchor_weight": args.anchor_weight, "loss_weights": {name: 1. for name in COMPONENTS},
            "loss_normalization": NORMALIZATION, "drop_last": True,
            "amp_dtype": "bfloat16", "delta_dtype": "float32", "delta_shape": list(delta_shape),
            "network_mode": "frozen_eval_with_delta_autograd", "torch_version": str(torch.__version__),
            "initial_state_sha256": shared.cache_fingerprint(initial_state),
            "base_sha256": fingerprints["base"], "tokenizer_sha256": fingerprints["tokenizer"],
            "initial_cache_file_sha256": fingerprints["cache"],
            "validation": validation_config,
            "implementation_sha256": canonical_hash(fingerprints["implementation"])}
    # Leave the legacy all-position configuration byte-for-byte unchanged.
    # The opt-in layout is bound explicitly, not inferred from a small tensor.
    if positions == "content":
        config.update(residual_positions=positions, residual_mode=mode,
                      trainable_parameter_count=math.prod(delta_shape))
    return config


def require_rank_validation_consistency(states):
    """Every rank supplies its own selection/progress state before save/stop."""
    if not states or any(canonical_hash(value) != canonical_hash(states[0]) for value in states):
        raise ValueError("Validation selection/early-stop state differs across ranks")


def resume_best_checkpoint(resume, resume_path, config, image_ids, initial_state):
    """Load the immutable historical best; never substitute the latest weights.

    Relocating a resume file also requires its step-XXXXXXXX-best.pt sibling
    unless the resume checkpoint itself is the selected best.
    """
    if not config["validation"]["enabled"]:
        return None
    selected = resume["validation_state"]["selection"]
    best_step = selected["best_step"]
    if best_step is None:
        return None
    if best_step == resume["progress"]["global_step"]:
        best = resume
    else:
        path = resume_path.parent / f"step-{best_step:08d}-best.pt"
        best = torch.load(path, map_location="cpu", weights_only=True)
    validate_resume(best, config, image_ids, initial_state)
    best_selection = best["validation_state"]["selection"]
    prefix = [record for record in selected["history"] if record["step"] <= best_step]
    if (best["progress"]["global_step"] != best_step
            or best_selection["best_step"] != best_step
            or best_selection["best_metric"] != selected["best_metric"]
            or best_selection["history"] != prefix):
        raise ValueError("Historical best checkpoint does not match the resumed selection history")
    return best


def rank_stage(context, function):
    error, result = None, None
    try:
        result = function()
    except Exception as exc:
        error = exc
    raise_if_distributed_error(error, context)
    return result


def gather_objects(value, context):
    if not context.distributed:
        return [value]
    values = [None for _ in range(context.world_size)]
    dist.all_gather_object(values, value)
    return values


def verify_frozen_inputs(args, contracts, fingerprints, approval_hash, *, full_hash=False):
    if shared.evaluation.sha256(args.approval) != approval_hash:
        raise RuntimeError("Dataset selection approval changed during the run")
    for contract in contracts.values():
        if shared.evaluation.sha256(contract.root / "annotations.json") != contract.annotations_sha256:
            raise RuntimeError("Approved annotations changed during the run")
    if implementation_hashes() != fingerprints["implementation"]:
        raise RuntimeError("Training implementation changed during the run")
    if full_hash:
        for name, path in (("base", args.base_checkpoint), ("tokenizer", args.tokenizer_path),
                           ("cache", args.initial_cache)):
            if shared.evaluation.sha256(path) != fingerprints[name]:
                raise RuntimeError(f"Frozen {name} input changed during the run")


def build_training_model(args, initial_encoder, context):
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint), bpe_path=str(args.tokenizer_path),
                                   load_from_HF=False, device=str(context.device), eval_mode=False,
                                   enable_segmentation=True, enable_inst_interactivity=False, text_encoder_type="ve")
    old = cached.install_cached_ve_text_encoder(model, initial_encoder.to(context.device))
    del old
    torch.cuda.empty_cache()
    cached.set_cached_ve_training_mode(model, train_delta=True)
    if not callable(getattr(model, "matcher", None)) or model.num_interactive_steps_val != 0:
        raise RuntimeError("Need a matcher without reference-derived interactive prompts")
    return ResidualObjective(model, anchor_weight=args.anchor_weight).to(context.device)


def run(args):
    started = time.monotonic()
    rank, _, world = torchrun_ranks()
    def startup(message):
        if not args.preflight_only:
            print(f"rank={rank}/{world} startup={message}", flush=True)
    startup("validating_approved_data")
    # Preflight does not initialize CUDA or distributed communications.
    approval, approval_hash, contracts = approval_contracts(args)
    startup("hashing_frozen_inputs")
    fingerprints = {"base": shared.evaluation.sha256(args.base_checkpoint),
                    "tokenizer": shared.evaluation.sha256(args.tokenizer_path),
                    "cache": shared.evaluation.sha256(args.initial_cache),
                    "implementation": implementation_hashes()}
    encoder = shared.load_initial_cache(args.initial_cache, base_hash=fingerprints["base"],
                                        tokenizer_hash=fingerprints["tokenizer"],
                                        residual_positions=args.residual_positions)
    initial_state = shared.cpu_state(encoder.state_dict())
    config = training_configuration(args, contracts["train"], approval_hash, world, initial_state, fingerprints,
                                    contracts.get("val"))
    if args.preflight_only:
        print(json.dumps({"status": "preflight_only_no_training", "config": config,
                          "train": contracts["train"].summary,
                          "validation": contracts["val"].summary if "val" in contracts else None}, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Full SAM3 training requires CUDA; use CPU unit tests for DDP correctness")
    startup("initializing_distributed")
    context = initialize_distributed("cuda", timeout_seconds=300)
    startup("distributed_ready")
    monitor = TrainingMonitor(args.output_dir, rank=context.rank, enabled=args.tensorboard)
    completed, observed, epoch_loader = 0, [], None
    output_created = False
    try:
        if context.is_main:
            # Failure propagates before other ranks proceed into model collectives.
            create = lambda: args.output_dir.mkdir(parents=True, exist_ok=False)
        else:
            create = lambda: None
        rank_stage(context, create)
        output_created = True
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, context.device)
        torch.cuda.reset_peak_memory_stats(context.device)
        random.seed(args.seed + context.rank)
        np.random.seed(args.seed + context.rank)
        torch.manual_seed(args.seed + context.rank)
        torch.set_float32_matmul_precision("high")
        objective = rank_stage(context, lambda: build_training_model(args, encoder, context))
        optimizer = torch.optim.AdamW([objective.encoder.delta], lr=args.learning_rate, weight_decay=0.)
        images = contracts["train"].images
        image_ids = [row["id"] for row in images]
        resume, inherited_best = None, None
        validation_state = ({"last_validation_step": None,
                             "selection": new_selection_state(contracts["val"].annotations_sha256,
                                 **config["validation"]["selection_policy"])} if args.validation else None)
        if args.resume:
            def load_resume():
                value = torch.load(args.resume, map_location="cpu", weights_only=True)
                step = validate_resume(value, config, image_ids, initial_state)
                return value, step
            resume, completed = rank_stage(context, load_resume)
            objective.encoder.load_state_dict(resume["cache_state_dict"])
            optimizer.load_state_dict(resume["optimizer"])
            observed = list(resume["rank_states"][context.rank]["image_ids"])
            validation_state = rank_stage(context, lambda: validate_validation_state(
                resume["validation_state"], config, completed))
            # Only rank zero performs artifact I/O; failure propagates to all ranks.
            inherited_best = rank_stage(context, lambda: resume_best_checkpoint(
                resume, args.resume, config, image_ids, initial_state) if context.is_main else None)
        wrapped = (DistributedDataParallel(objective, device_ids=[context.local_rank],
                                          broadcast_buffers=False, find_unused_parameters=False)
                   if context.distributed else objective)
        versions = frozen_versions(objective.model, objective.encoder.delta)
        dataset = rank_stage(context, lambda: make_identity_dataset(contracts["train"].root, contracts["train"]))
        val_dataset = (rank_stage(context, lambda: make_identity_dataset(contracts["val"].root, contracts["val"]))
                       if args.validation else None)
        limit = min(args.stop_after_step or config["steps_per_epoch"] * args.epochs,
                    config["steps_per_epoch"] * args.epochs)
        if limit < completed:
            raise ValueError("Requested stop step precedes resumed progress")
        def write_run():
            if not context.is_main:
                return
            shared.evaluation.atomic_write_json(args.output_dir / "run.json", {
                "format": "sam3-output-residual-ddp-run-v1", "training_config": config,
                "approval": approval, "inputs": {name: str(getattr(args, name)) for name in
                    ("approval", "base_checkpoint", "initial_cache", "tokenizer_path", "resume")},
                "implementation": fingerprints["implementation"],
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_names": gather_gpu_names_local(), "requested_stop_step": limit,
                "loader": {"workers_per_rank": args.num_workers, "prefetch_factor": args.prefetch_factor,
                           "pin_memory": True, "persistent_workers": args.num_workers > 0},
                "note": "Train loss is not accuracy. RealSense is reserved for held-out testing."})
        rank_stage(context, write_run)
        def publish_inherited_best():
            if context.is_main and inherited_best is not None:
                best_step = inherited_best["progress"]["global_step"]
                atomic_checkpoint(args.output_dir / f"step-{best_step:08d}-best.pt", inherited_best)
                atomic_checkpoint(args.output_dir / "best.pt", inherited_best)
        rank_stage(context, publish_inherited_best)

        def save_checkpoint(final=False, *, stage="recovery", is_best=False):
            rank_stage(context, lambda: assert_frozen_versions(versions))
            rank_stage(context, lambda: verify_frozen_inputs(args, contracts, fingerprints, approval_hash, full_hash=final))
            rank_stage(context, lambda: validate_validation_state(validation_state, config, completed))
            require_rank_validation_consistency(gather_objects(validation_state, context))
            cache_hash = rank_stage(context, lambda: shared.cache_fingerprint(
                shared.cpu_state(objective.encoder.state_dict())))
            cache_audit = validate_rank_cache_consistency(
                gather_objects(cache_hash, context), world_size=context.world_size)
            rank_states = gather_objects({"image_ids": observed, "rng": capture_rng_state(context.device)}, context)
            def write():
                if not context.is_main:
                    return
                state = make_checkpoint(objective.encoder, optimizer, config, completed, initial_state,
                                        rank_states, rank_cache_audit=cache_audit, validation_state=validation_state)
                validate_resume(state, config, image_ids, initial_state)
                name = f"step-{completed:08d}-{'final' if final else stage}.pt"
                atomic_checkpoint(args.output_dir / name, state)
                if is_best:
                    atomic_checkpoint(args.output_dir / f"step-{completed:08d}-best.pt", state)
                    atomic_checkpoint(args.output_dir / "best.pt", state, replace=True)
                atomic_checkpoint(args.output_dir / "latest.pt", state, replace=True)
            rank_stage(context, write)

        def validate():
            from scripts.residual_ddp_validation import evaluate_validation
            # Validation must not perturb future training RNG, including after resume.
            rng = capture_rng_state(context.device)
            try:
                return evaluate_validation(objective.model, contracts["val"], val_dataset, context=context,
                    batch_size=args.batch_size_per_rank, num_workers=args.num_workers,
                    output_dir=args.output_dir / f"validation-step-{completed:08d}",
                    monitor=monitor, step=completed, amp=True)
            finally:
                restore_rng_state(rng, context.device)

        # Model construction, DDP initialization, and initial val must not consume resumed RNG.
        if resume:
            restore_rng_state(resume["rank_states"][context.rank]["rng"], context.device)
        def selection_stopped():
            return validation_state is not None and validation_state["selection"]["stopped"]

        def complete_pending_validation():
            nonlocal validation_state
            require_rank_validation_consistency(gather_objects(validation_state, context))
            pending = rank_stage(context, lambda: pending_selection_validation_step(validation_state, config, completed))
            if pending is None:
                return
            metrics = validate()
            def select():
                selected, decision = update_selection(validation_state["selection"], metrics,
                    step=completed, epoch=completed // config["steps_per_epoch"],
                    validation_sha256=contracts["val"].annotations_sha256)
                return {"last_validation_step": completed, "selection": selected}, decision
            validation_state, decision = rank_stage(context, select)
            require_rank_validation_consistency(gather_objects(validation_state, context))
            def log_selection():
                monitor.log_validation(completed, {
                    "macro_miss_zero_dice": decision["metric"], "best_macro_miss_zero_dice": decision["best_metric"],
                    "best_step": decision["best_step"], "bad_epochs": decision["bad_epochs"],
                    "early_stop": int(decision["should_stop"]),
                }, scope="dexycb_val_selection")
                if context.is_main:
                    print(f"validation step={completed} metric={METRIC_NAME} value={decision['metric']:.6f} "
                          f"best={decision['best_metric']:.6f} bad_epochs={decision['bad_epochs']} "
                          f"early_stop={decision['should_stop']}", flush=True)
            rank_stage(context, log_selection)
            save_checkpoint(stage="validated", is_best=decision["is_best"])

        # Includes recovery from the exact epoch-checkpoint-before-validation boundary.
        complete_pending_validation()
        window_started = time.monotonic()
        window_images, window_data, window_h2d, window_compute = 0, 0., 0., 0.
        while completed < limit and not selection_stopped():
            epoch, offset = divmod(completed, config["steps_per_epoch"])
            sampler = DeterministicGlobalBatchSampler(len(images), args.batch_size_per_rank,
                rank=context.rank, world_size=context.world_size, seed=args.seed, epoch=epoch, start_step=offset)
            epoch_loader = make_loader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=args.num_workers > 0,
                prefetch_factor=args.prefetch_factor, seed=args.seed + epoch * context.world_size + context.rank)
            iterator = iter(epoch_loader)
            while completed < limit and completed // config["steps_per_epoch"] == epoch and not selection_stopped():
                data_started = time.monotonic()
                batch = rank_stage(context, lambda: next(iterator))
                window_data += time.monotonic() - data_started
                def check_pinned():
                    if not batch.datapoint.img_batch.is_pinned():
                        raise RuntimeError("Image batch was not pinned; refusing a silently ineffective loader configuration")
                rank_stage(context, check_pinned)
                transfer_start, transfer_end, compute_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                transfer_start.record()
                batch = rank_stage(context, lambda: batch.to(context.device, non_blocking=True))
                transfer_end.record()
                optimizer.zero_grad(set_to_none=True)
                def forward():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        return wrapped(batch.datapoint)
                total, components, anchor = rank_stage(context, forward)
                require_finite_everywhere(total, components, anchor)
                total.backward()
                delta = objective.encoder.delta
                rank_stage(context, lambda: objective.check_frozen_contract())
                rank_stage(context, lambda: require_connected_gradient(delta))
                require_finite_everywhere(delta.grad)
                grad_norms = delta.grad.detach().flatten(1).norm(dim=1)
                rank_stage(context, optimizer.step)
                require_finite_everywhere(delta)
                compute_end.record()
                # Safety gates already synchronize; one wait gives honest event timings.
                compute_end.synchronize()
                window_h2d += transfer_start.elapsed_time(transfer_end)
                window_compute += transfer_end.elapsed_time(compute_end)
                completed += 1
                observed.extend(batch.image_ids)
                window_images += config["global_batch_size"]
                if completed == 1 or completed % args.log_every == 0 or completed == limit:
                    elapsed = time.monotonic() - window_started
                    stats = mean_across_ranks(torch.cat((total.detach().reshape(1), components.float(),
                        anchor.float().reshape(1), grad_norms.float(),
                        torch.tensor([window_data * 1000, window_h2d, window_compute], device=context.device))))
                    values = stats.cpu().tolist()
                    logs = {"train/total_loss": values[0], "train/task_loss": sum(values[1:7]),
                            **{f"train/{name}": value for name, value in zip(COMPONENTS, values[1:7])},
                            "train/anchor_loss": values[7], "train/learning_rate": args.learning_rate,
                            "train/left_gradient_norm": values[8], "train/right_gradient_norm": values[9],
                            "perf/images_per_second": window_images / max(elapsed, 1e-9),
                            "perf/data_wait_ms_per_step": values[10] / (window_images / config["global_batch_size"]),
                            "perf/h2d_ms_per_step": values[11] / (window_images / config["global_batch_size"]),
                            "perf/compute_ms_per_step": values[12] / (window_images / config["global_batch_size"]),
                            "perf/peak_allocated_mib": torch.cuda.max_memory_allocated(context.device) / 2**20,
                            "train/global_batch_size": config["global_batch_size"]}
                    rank_stage(context, lambda: monitor.log_scalars(completed, logs,
                        samples_seen=completed * config["global_batch_size"], wall_seconds=time.monotonic()-started))
                    if context.is_main:
                        print(f"step={completed}/{config['steps_per_epoch']*args.epochs} "
                              f"samples={completed*config['global_batch_size']} loss={values[0]:.6f} "
                              f"images/s={logs['perf/images_per_second']:.2f}", flush=True)
                    window_started = time.monotonic()
                    window_images, window_data, window_h2d, window_compute = 0, 0., 0., 0.
                epoch_done = completed % config["steps_per_epoch"] == 0
                if completed % args.checkpoint_every == 0 or epoch_done:
                    save_checkpoint()
                if epoch_done:
                    complete_pending_validation()
                del batch, total, components, anchor
            del iterator, epoch_loader
            epoch_loader = None
        # A partial smoke is not a complete epoch; final validation remains explicitly optional.
        if args.validation and validation_state["last_validation_step"] != completed:
            validate()
            # Bounded partial-epoch diagnostics do not consume an epoch of patience.
            validation_state = {**validation_state, "last_validation_step": completed}
        rank_stage(context, lambda: assert_frozen_versions(versions))
        save_checkpoint(final=True)
        if context.is_main:
            shared.evaluation.atomic_write_json(args.output_dir / "summary.json", {
                "status": "early_stopped" if selection_stopped() else "completed_requested_steps", "training_config": config,
                "progress": progress_at(completed, config), "final_checkpoint": str(args.output_dir / f"step-{completed:08d}-final.pt"),
                "tensorboard": str(args.output_dir / "tensorboard"), "validation_ran": bool(args.validation),
                "selection": validation_state["selection"] if validation_state is not None else None,
                "best_checkpoint": str(args.output_dir / "best.pt") if validation_state is not None
                    and validation_state["selection"]["best_step"] is not None else None,
                "wall_seconds": time.monotonic()-started,
                "trainable_parameter_count": objective.encoder.delta.numel(),
                "frozen_parameter_versions_unchanged": True,
                "all_rank_encoder_states_identical_at_checkpoint": True,
                "note": "Large global batch changes the optimization budget. This is not proof of accuracy gains."})
    except BaseException as error:
        if context.is_main and output_created and args.output_dir.is_dir():
            shared.evaluation.atomic_write_json(args.output_dir / "failure.json", {
                "status": "failed", "successful_steps_in_memory": completed,
                "error": f"{type(error).__name__}: {error}",
                "recovery": "Use the last fully committed checkpoint, never a partial optimizer update."})
        raise
    finally:
        monitor.close()
        cleanup_distributed(context)


def gather_gpu_names_local():
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def require_connected_gradient(delta):
    if delta.grad is None:
        raise RuntimeError("Residual gradient disconnected")


if __name__ == "__main__":
    run(parse_args())

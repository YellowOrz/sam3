#!/usr/bin/env python3
"""Independent spatial-mask DDP ablation; original VE and SAM3 remain frozen.

Two epochs by default. Fixed Dex validation selects boundary-only best; no
test-driven tuning. Resume preserves world/batch/data/code/optimizer/RNG.
"""
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from scripts import train_residual_ddp as infrastructure
from scripts.train_residual_ddp import shared, cached, rank_stage, gather_objects
from scripts.residual_ddp_runtime import (initialize_distributed, cleanup_distributed,
    capture_rng_state, restore_rng_state, DeterministicGlobalBatchSampler, torchrun_ranks)
from scripts.residual_ddp_data import make_identity_dataset, make_loader
from scripts.residual_ddp_objective import (loss_components, global_target_denominator,
    COMPONENTS, require_finite_everywhere, mean_across_ranks)
from scripts.residual_ddp_checkpoint import atomic_checkpoint, validate_rank_cache_consistency
from scripts.residual_ddp_validation import evaluate_validation
from scripts.residual_training_monitor import TrainingMonitor
from scripts.spatial_training_state import FORMAT, validate_state
from sam3.model.spatial_mask_adapter import attach_spatial_mask_adapter


class SpatialObjective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.functions = nn.ModuleList(shared.build_loss_functions())
        self.eval()

    def forward(self, batch):
        denominator = global_target_denominator(batch.find_targets[0].num_boxes.sum())
        _, components, _ = loss_components(self.model, batch, self.functions, denominator)
        # All-empty batches must retain the same DDP parameter graph.
        connected_zero = sum(p.sum() * 0 for p in self.model.parameters() if p.requires_grad)
        return components[0] + components[1] + connected_zero, components.detach()


def run(args):
    if args.anchor_weight != 0:
        raise ValueError("Spatial ablation has no text-anchor objective")
    approval, approval_hash, contracts = infrastructure.approval_contracts(args)
    base_hash = shared.evaluation.sha256(args.base_checkpoint)
    token_hash = shared.evaluation.sha256(args.tokenizer_path)
    encoder = shared.load_initial_cache(args.initial_cache, base_hash=base_hash, tokenizer_hash=token_hash)
    project = Path(__file__).resolve().parents[1]
    paths = sorted((project / "sam3").rglob("*.py")) + sorted((project / "scripts").glob("*.py"))
    code_hashes = {str(p.relative_to(project)): shared.evaluation.sha256(p) for p in paths}
    _, _, world = torchrun_ranks()
    global_batch = world * args.batch_size_per_rank
    steps_per_epoch = len(contracts["train"].images) // global_batch
    if not steps_per_epoch:
        raise ValueError("Dataset smaller than global batch")
    config = dict(method=FORMAT, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        world_size=world, batch_size_per_rank=args.batch_size_per_rank, seed=args.seed,
        learning_rate=args.learning_rate, optimizer="AdamW", weight_decay=0,
        amp="bfloat16", bottleneck=32, approval_sha256=approval_hash,
        base_sha256=base_hash, tokenizer_sha256=token_hash,
        initial_cache_sha256=shared.evaluation.sha256(args.initial_cache),
        annotation_hashes={k: v.annotations_sha256 for k, v in contracts.items()},
        validation=args.validation, initial_validation=args.initial_validation,
        selection="macro_candidate_boundary_iou_4px", implementation=code_hashes,
        loss="original_mask_focal_plus_dice", drop_last=True)
    if args.preflight_only:
        print(json.dumps(config), flush=True)
        return
    context = initialize_distributed("cuda", timeout_seconds=600)
    monitor = TrainingMonitor(args.output_dir, rank=context.rank, enabled=args.tensorboard)
    try:
        rank_stage(context, lambda: args.output_dir.mkdir(parents=True, exist_ok=False) if context.is_main else None)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, context.device)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        def build():
            from sam3.model_builder import build_sam3_image_model
            model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint),
                bpe_path=str(args.tokenizer_path), load_from_HF=False, device=str(context.device),
                eval_mode=False, enable_segmentation=True, enable_inst_interactivity=False, text_encoder_type="ve")
            cached.install_cached_ve_text_encoder(model, encoder.to(context.device))
            adapter = attach_spatial_mask_adapter(model)
            if model.num_interactive_steps_val != 0:
                raise RuntimeError("Reference interactive prompts forbidden")
            return model, adapter
        model, adapter = rank_stage(context, build)
        objective = SpatialObjective(model).to(context.device)
        params = list(adapter.parameters())
        ids = {id(p) for p in params}
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0)
        template = shared.cpu_state(adapter.state_dict())
        step, last_validation, best = 0, None, None
        resume = None
        if args.resume:
            def read_resume():
                state = torch.load(args.resume, map_location="cpu", weights_only=True)
                validate_state(state, config, template)
                return state
            resume = rank_stage(context, read_resume)
            adapter.load_state_dict(resume["adapter"])
            optimizer.load_state_dict(resume["optimizer"])
            step, last_validation, best = resume["step"], resume["last_validation_step"], resume["best_boundary"]
        wrapped = DistributedDataParallel(objective, device_ids=[context.local_rank], broadcast_buffers=False) if context.distributed else objective
        # DDP initialization broadcasts even frozen parameters and increments
        # their version counters; audit optimizer changes AFTER that sync.
        frozen = [(p, p._version) for p in model.parameters() if id(p) not in ids]
        datasets = rank_stage(context, lambda: {k: make_identity_dataset(v.root, v) for k, v in contracts.items()})
        limit = min(args.stop_after_step or args.epochs * steps_per_epoch, args.epochs * steps_per_epoch)
        if limit < step:
            raise ValueError("Stop step precedes resume")
        def write_run():
            if context.is_main:
                shared.evaluation.atomic_write_json(args.output_dir / "run.json", {
                    "config": config, "approval": approval, "requested_stop": limit,
                    "resume": str(args.resume), "parameter_count": sum(p.numel() for p in params)})
        rank_stage(context, write_run)
        def audit():
            if any(p.requires_grad or p.grad is not None or p._version != v for p, v in frozen):
                raise RuntimeError("Frozen base changed")
            for role, contract in contracts.items():
                if shared.evaluation.sha256(contract.root / "annotations.json") != config["annotation_hashes"][role]:
                    raise RuntimeError("Annotations changed")
            if shared.evaluation.sha256(args.approval) != approval_hash:
                raise RuntimeError("Approval changed")
            if {str(p.relative_to(project)): shared.evaluation.sha256(p) for p in paths} != code_hashes:
                raise RuntimeError("Training source changed; use a frozen code snapshot")
        def save(stage, is_best=False):
            rank_stage(context, audit)
            fingerprint = shared.cache_fingerprint(shared.cpu_state(adapter.state_dict()))
            validate_rank_cache_consistency(gather_objects(fingerprint, context), world_size=world)
            rngs = gather_objects(capture_rng_state(context.device), context)
            def write():
                if context.is_main:
                    state = dict(format=FORMAT, config=config, step=step,
                        adapter=shared.cpu_state(adapter.state_dict()), optimizer=optimizer.state_dict(),
                        rank_rng=rngs, last_validation_step=last_validation, best_boundary=best)
                    validate_state(state, config, template)
                    atomic_checkpoint(args.output_dir / f"step-{step:08d}-{stage}.pt", state)
                    atomic_checkpoint(args.output_dir / "latest.pt", state, replace=True)
                    if is_best: atomic_checkpoint(args.output_dir / "best.pt", state, replace=True)
            rank_stage(context, write)
        def validate_if_due():
            nonlocal last_validation, best
            if not args.validation or step % steps_per_epoch or last_validation == step:
                return
            if step == 0 and not args.initial_validation:
                return
            rng = capture_rng_state(context.device)
            try:
                metrics = evaluate_validation(model, contracts["val"], datasets["val"], context=context,
                    batch_size=args.batch_size_per_rank, num_workers=args.num_workers,
                    output_dir=args.output_dir / f"validation-step-{step:08d}", monitor=monitor, step=step)
            finally:
                restore_rng_state(rng, context.device)
            value = sum(metrics[f"{s}/candidate_boundary_iou_4px"] for s in ("left_hand", "right_hand")) / 2
            improved = best is None or value > best
            if improved: best = value
            last_validation = step
            save("validated", is_best=improved)
            if context.is_main: print(f"validation step={step} boundary={value} best={best}", flush=True)
        if resume:
            restore_rng_state(resume["rank_rng"][context.rank], context.device)
        validate_if_due()
        started = time.monotonic()
        while step < limit:
            epoch, offset = divmod(step, steps_per_epoch)
            sampler = DeterministicGlobalBatchSampler(len(contracts["train"].images), args.batch_size_per_rank,
                rank=context.rank, world_size=world, seed=args.seed, epoch=epoch, start_step=offset)
            loader = make_loader(datasets["train"], batch_sampler=sampler, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=False, prefetch_factor=args.prefetch_factor,
                seed=args.seed + epoch * world + context.rank)
            iterator = iter(loader)
            while step < limit and step // steps_per_epoch == epoch:
                batch = rank_stage(context, lambda: next(iterator).to(context.device, non_blocking=True).datapoint)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, components = wrapped(batch)
                require_finite_everywhere(loss, components)
                loss.backward()
                require_finite_everywhere(*(p.grad for p in params))
                optimizer.step()
                step += 1
                values = mean_across_ranks(torch.cat([loss.detach().reshape(1), components]))
                if step % args.log_every == 0 or step == 1:
                    row = {"loss/mask_objective": float(values[0]), **{f"loss/{k}": float(v) for k, v in zip(COMPONENTS, values[1:])}}
                    monitor.log_scalars(step, row, step * global_batch, time.monotonic() - started)
                    if context.is_main: print(f"step={step}/{args.epochs * steps_per_epoch} loss={float(values[0]):.6f}", flush=True)
                if step % args.checkpoint_every == 0 or step % steps_per_epoch == 0:
                    save("recovery")
                validate_if_due()
                del loss, components, batch
        save("final")
        if context.is_main:
            shared.evaluation.atomic_write_json(args.output_dir / "summary.json", {
                "status": "complete" if step == args.epochs * steps_per_epoch else "bounded_stop",
                "step": step, "completed_epochs": step // steps_per_epoch,
                "samples_seen": step * global_batch, "best_boundary": best,
                "note": "No RealSense-driven selection; see separate validation records"})
    finally:
        monitor.close()
        cleanup_distributed(context)


if __name__ == "__main__":
    run(infrastructure.parse_args())

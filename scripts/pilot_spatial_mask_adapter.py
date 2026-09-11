#!/usr/bin/env python3
"""Bounded single-GPU engineering pilot, NOT a full training/evaluation run.

Uses approved training images only. Checks full-model zero-init equivalence,
finite gradients and frozen base. Writes distinct adapter checkpoints and TB.
No test-set selection, implicit resume, or original checkpoint overwrite.
"""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from scripts.train_residual_ddp import parse_args, approval_contracts, shared, cached
from scripts.residual_ddp_data import make_identity_dataset, make_loader
from scripts.residual_ddp_runtime import DeterministicGlobalBatchSampler
from scripts.residual_ddp_objective import loss_components, COMPONENTS
from scripts.residual_ddp_checkpoint import atomic_checkpoint
from sam3.model.spatial_mask_adapter import attach_spatial_mask_adapter


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This engineering pilot is single-process, not a DDP training entry point")
    if args.resume or args.preflight_only or args.stop_after_step is None:
        raise ValueError("Pilot requires --stop-after-step and does not implement resume/preflight")
    if not 1 <= args.stop_after_step <= 100:
        raise ValueError("Engineering pilot is limited to 100 updates")
    _, approval_hash, contracts = approval_contracts(args)
    base_hash = shared.evaluation.sha256(args.base_checkpoint)
    tokenizer_hash = shared.evaluation.sha256(args.tokenizer_path)
    encoder = shared.load_initial_cache(args.initial_cache, base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    from sam3.model_builder import build_sam3_image_model
    from torch.utils.tensorboard import SummaryWriter
    model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint),
        bpe_path=str(args.tokenizer_path), load_from_HF=False, device="cuda",
        eval_mode=False, enable_segmentation=True, enable_inst_interactivity=False,
        text_encoder_type="ve")
    cached.install_cached_ve_text_encoder(model, encoder.cuda())
    model.requires_grad_(False)
    model.eval()
    if model.num_interactive_steps_val != 0:
        raise RuntimeError("Reference-derived interactive prompts are forbidden")
    contract = contracts["train"]
    sampler = DeterministicGlobalBatchSampler(len(contract.images), args.batch_size_per_rank,
        rank=0, world_size=1, seed=args.seed, epoch=0)
    loader = make_loader(make_identity_dataset(contract.root, contract), batch_sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=args.prefetch_factor, seed=args.seed)
    iterator = iter(loader)
    first = next(iterator).to("cuda", non_blocking=True).datapoint
    def predict(batch):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(batch)[0]
        return {key: result[key].detach().cpu() for key in
                ("pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec") if key in result}
    baseline = predict(first)
    adapter = attach_spatial_mask_adapter(model)
    initialized = predict(first)
    equal = {key: torch.equal(value, initialized[key]) for key, value in baseline.items()}
    if not equal or not all(equal.values()):
        raise RuntimeError(f"Zero-init full SAM3 differs: {equal}")
    parameters = list(adapter.parameters())
    train_ids = {id(p) for p in parameters}
    frozen = [(name, p, p._version) for name, p in model.named_parameters() if id(p) not in train_ids]
    functions = torch.nn.ModuleList(shared.build_loss_functions()).cuda()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    config = {"method": "frozen_natural_VE_instance_mask_spatial_adapter_pilot",
        "steps": args.stop_after_step, "batch_size": args.batch_size_per_rank,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "base_sha256": base_hash, "tokenizer_sha256": tokenizer_hash,
        "initial_cache_sha256": shared.evaluation.sha256(args.initial_cache),
        "annotations_sha256": contract.annotations_sha256, "approval_sha256": approval_hash,
        "zero_init_exact": equal, "loss": "original mask focal + dice only",
        "scope": "engineering pilot; no validation or generalization claim"}
    config["implementation_sha256"] = {
        str(path): shared.evaluation.sha256(path) for path in (
            Path(__file__), Path(__file__).resolve().parents[1] / "sam3/model/spatial_mask_adapter.py")}
    shared.evaluation.atomic_write_json(args.output_dir / "run.json", config)
    writer = SummaryWriter(str(args.output_dir / "tensorboard"))
    history = []
    try:
        for step in range(1, args.stop_after_step + 1):
            batch = first if step == 1 else next(iterator).to("cuda", non_blocking=True).datapoint
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                denominator = batch.find_targets[0].num_boxes.sum().float().clamp(min=1)
                _, components, prediction = loss_components(model, batch, functions, denominator)
                loss = components[0] + components[1]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Missing or nonfinite adapter gradients")
            grad_norm = sum(g.float().square().sum() for g in grads).sqrt().item()
            optimizer.step()
            if any(p.requires_grad or p.grad is not None or p._version != v for _, p, v in frozen):
                raise RuntimeError("Frozen base changed")
            row = {"step": step, "loss": loss.item(), "gradient_norm": grad_norm,
                   **dict(zip(COMPONENTS, components.detach().float().cpu().tolist()))}
            history.append(row)
            for key, value in row.items():
                if key != "step": writer.add_scalar(f"pilot/{key}", value, step)
            print(json.dumps(row), flush=True)
            if step % 10 == 0 or step == args.stop_after_step:
                atomic_checkpoint(args.output_dir / f"adapter-step-{step:04d}.pt",
                    {"format": "sam3-spatial-mask-pilot-v1", "config": config, "step": step,
                     "adapter": shared.cpu_state(adapter.state_dict()), "optimizer": optimizer.state_dict()})
            del prediction, components, loss
        after = predict(first)
        unchanged = {key: torch.equal(value, after[key]) for key, value in baseline.items() if key != "pred_masks"}
        if not all(unchanged.values()):
            raise RuntimeError(f"Detector outputs changed: {unchanged}")
        atomic_checkpoint(args.output_dir / "first-training-batch-before-after.pt",
            {"baseline": baseline, "after": after, "note": "training sample, not test"})
        shared.evaluation.atomic_write_json(args.output_dir / "summary.json",
            {"status": "engineering_pilot_complete", "history": history,
             "zero_init_exact": equal, "detector_unchanged": unchanged,
             "mask_changed": not torch.equal(baseline["pred_masks"], after["pred_masks"]),
             "peak_memory_bytes": torch.cuda.max_memory_allocated(), "config": config})
    finally:
        writer.close()


if __name__ == "__main__":
    run(parse_args())

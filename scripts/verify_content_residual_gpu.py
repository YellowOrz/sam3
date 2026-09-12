"""Bounded content-delta GPU engineering check, never a formal training run.

Uses only approved training images. The same frozen SAM3 must produce identical
full prediction tensors under all-position zero delta and body-only zero delta.
One to four body-only optimizer steps then test connectivity and preservation;
the resulting artifact is explicitly not a DDP recovery point or initializer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from collections.abc import Mapping

import numpy as np
import torch

from scripts import cached_ve_text_features as cached
from scripts import train_residual_ddp as trainer
from scripts import train_ve_initialized_tokens as shared
from scripts.residual_ddp_data import collate_indexed_samples, make_identity_dataset
from scripts.residual_ddp_objective import assert_frozen_versions, frozen_versions, require_finite_everywhere
from scripts.residual_ddp_runtime import DeterministicGlobalBatchSampler, initialize_distributed, torchrun_ranks


PREDICTION_KEYS = ("pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec")
FORMAT = "sam3-content-residual-engineering-only-v1"


def validate_probe_args(args, ranks=None):
    if (ranks if ranks is not None else torchrun_ranks()) != (0, 0, 1):
        raise ValueError("This bounded engineering probe requires one process: RANK=LOCAL_RANK=0, WORLD_SIZE=1")
    if args.residual_positions != "content" or args.resume is not None or args.preflight_only:
        raise ValueError("Probe requires content positions, without resume or preflight-only")
    if type(args.stop_after_step) is not int or not 1 <= args.stop_after_step <= 4:
        raise ValueError("An explicit engineering limit of 1..4 steps is required")
    if args.batch_size_per_rank != 3 or args.learning_rate != 1e-4 or args.anchor_weight != 0:
        raise ValueError("Probe fixes batch=3, learning-rate=1e-4 and anchor-weight=0")


def tensor_signature(value):
    if not isinstance(value, torch.Tensor) or not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError("Prediction/features must be nonempty finite tensors")
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{cpu.dtype}|{tuple(cpu.shape)}|".encode())
    digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "sha256": digest.hexdigest()}


def prediction_signatures(prediction):
    if not isinstance(prediction, Mapping) or any(key not in prediction for key in PREDICTION_KEYS):
        raise ValueError("All four unthresholded SAM3 prediction outputs are required")
    return {key: tensor_signature(prediction[key]) for key in PREDICTION_KEYS}


def assert_same_predictions(reference, candidate):
    if set(reference) != set(PREDICTION_KEYS) or reference != candidate:
        raise RuntimeError("All-zero and content-zero SAM3 outputs differ; refusing the engineering steps")


def verify_content_features(reference, candidate, valid_positions, *, require_change):
    if len(reference) != 3 or len(candidate) != 3:
        raise ValueError("The complete padding/resized/raw triple is required")
    for old, new in zip(reference, candidate):
        if old.shape != new.shape or old.dtype != new.dtype:
            raise RuntimeError("Feature shape or dtype changed")
    padding, original, raw = reference
    if tuple(padding.shape) != (2, 32) or padding.dtype != torch.bool:
        raise ValueError("Probe requires both complete 32-position hand caches")
    if tuple(original.shape) != (32, 2, 256) or tuple(raw.shape) != (32, 2, 1024):
        raise ValueError("Probe requires the complete original resized/raw feature lengths and widths")
    expected = torch.stack([(~row).nonzero().flatten() for row in padding])
    if tuple(expected.shape) != (2, 4) or not torch.equal(expected, valid_positions):
        raise ValueError("Exactly four original valid positions must be retained per hand")
    if not torch.equal(padding, candidate[0]) or not torch.equal(raw, candidate[2]):
        raise RuntimeError("Padding or raw embeddings changed")
    changed = []
    for side in range(2):
        selected = valid_positions[side, 1:3]
        untouched = torch.ones(32, device=original.device, dtype=torch.bool)
        untouched[selected] = False
        if not torch.equal(original[untouched, side], candidate[1][untouched, side]):
            raise RuntimeError("BOS/EOS or padded output features changed")
        body_changes = (original[selected, side] != candidate[1][selected, side]).sum(dim=1).tolist()
        if require_change and sum(body_changes) == 0:
            raise RuntimeError("One hand's body output features did not change after engineering updates")
        if not require_change and sum(body_changes) != 0:
            raise RuntimeError("Zero-initialized body output features differ from the original")
        changed.append(body_changes)
    return {"context_length": 32, "valid_positions": valid_positions.cpu().tolist(),
            "body_changed_elements_per_hand_position": changed, "bos_eos_padding_raw_unchanged": True}


def run(args):
    validate_probe_args(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.monotonic()
    approval, approval_hash, contracts = trainer.approval_contracts(args)
    fingerprints = {"base": shared.evaluation.sha256(args.base_checkpoint),
        "tokenizer": shared.evaluation.sha256(args.tokenizer_path),
        "cache": shared.evaluation.sha256(args.initial_cache), "implementation": trainer.implementation_hashes()}
    own_source = shared.evaluation.sha256(Path(__file__))
    all_encoder = shared.load_initial_cache(args.initial_cache, base_hash=fingerprints["base"],
        tokenizer_hash=fingerprints["tokenizer"], residual_positions="all")
    content = shared.load_initial_cache(args.initial_cache, base_hash=fingerprints["base"],
        tokenizer_hash=fingerprints["tokenizer"], residual_positions="content")
    cached.validate_delta_state(content.state_dict())
    original_features = tuple(value.detach().clone() for value in all_encoder(list(cached.CLASS_NAMES)))
    verify_content_features(original_features, content(list(cached.CLASS_NAMES)), content.valid_positions,
                            require_change=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Actual full-model equivalence requires CUDA; CPU helpers are not that evidence")
    context = initialize_distributed("cuda")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, context.device)
    torch.cuda.reset_peak_memory_stats(context.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    journal = {"format": FORMAT, "status": "running", "formal_training": False,
        "resume_allowed": False, "real_accuracy_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "requested_steps": args.stop_after_step,
        "batch_size": 3, "world_size": 1, "learning_rate": 1e-4, "seed": args.seed,
        "approval_sha256": approval_hash, "approved_train": contracts["train"].summary,
        "source_fingerprints": fingerprints, "probe_source_sha256": own_source,
        "gpu_name": torch.cuda.get_device_name(context.device),
        "data_loading": "bounded direct CPU collation using the unchanged identity-checked dataset",
        "formal_training_must_restart_from_clean_initializer": True}
    shared.evaluation.atomic_write_json(args.output_dir / "run.json", journal)
    image_files, batch_records, updates = {}, [], []
    try:
        print("engineering: loading unchanged SAM3 base with all-position zero delta", flush=True)
        objective = trainer.build_training_model(args, all_encoder, context)
        versions = frozen_versions(objective.model, objective.encoder.delta)
        dataset = make_identity_dataset(contracts["train"].root, contracts["train"])
        sampler = DeterministicGlobalBatchSampler(len(dataset), 3, rank=0, world_size=1, seed=args.seed, epoch=0)
        batches = []
        for indices in itertools.islice(iter(sampler), args.stop_after_step):
            for index in indices:
                image = contracts["train"].images[index]
                path = (contracts["train"].root / image["file_name"]).resolve()
                image_files[str(path)] = shared.evaluation.sha256(path)
            batch = collate_indexed_samples([dataset[index] for index in indices])
            batches.append(batch)
            batch_records.append({"dataset_indices": list(batch.dataset_indices), "image_ids": list(batch.image_ids)})
        if len(batches) != args.stop_after_step:
            raise RuntimeError("Approved dataset cannot cover the bounded engineering steps")
        first = batches[0].to(context.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            all_predictions = prediction_signatures(objective.model(first.datapoint)[0])
        objective.model.backbone.language_backbone = content.to(context.device)
        cached.set_cached_ve_training_mode(objective.model, train_delta=True)
        objective.check_frozen_contract()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            content_predictions = prediction_signatures(objective.model(first.datapoint)[0])
        assert_same_predictions(all_predictions, content_predictions)
        assert_frozen_versions(versions)
        if sum(value.numel() for value in objective.model.parameters() if value.requires_grad) != 1024:
            raise RuntimeError("The content probe must have exactly 1024 trainable parameters")
        print("engineering: full zero-delta SAM3 output equivalence passed", flush=True)
        optimizer = torch.optim.AdamW([content.delta], lr=1e-4, weight_decay=0.)
        del first
        for step, cpu_batch in enumerate(batches, 1):
            batch = cpu_batch.to(context.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, components, anchor = objective(batch.datapoint)
            require_finite_everywhere(loss, components, anchor)
            loss.backward()
            trainer.require_connected_gradient(content.delta)
            require_finite_everywhere(content.delta.grad)
            gradients = content.delta.grad.detach().flatten(1).norm(dim=1)
            if not bool((gradients > 0).all()):
                raise RuntimeError("Both hands must receive nonzero finite body-delta gradients")
            optimizer.step()
            require_finite_everywhere(content.delta)
            objective.check_frozen_contract()
            assert_frozen_versions(versions)
            updates.append({"step": step, "image_ids": list(cpu_batch.image_ids), "loss": float(loss.detach()),
                "components": dict(zip(trainer.COMPONENTS, components.cpu().tolist())),
                "gradient_l2_per_side": gradients.cpu().tolist(),
                "delta_l2_per_side": content.delta.detach().flatten(1).norm(dim=1).cpu().tolist()})
            print(f"engineering: step={step}/{args.stop_after_step} finite and frozen-state checks passed", flush=True)
        features_after = tuple(value.detach().cpu() for value in content(list(cached.CLASS_NAMES)))
        feature_check = verify_content_features(original_features, features_after,
                                               content.valid_positions.cpu(), require_change=True)
        cached_state = shared.cpu_state(content.state_dict())
        cached.validate_delta_state(cached_state)
        trainer.verify_frozen_inputs(args, contracts, fingerprints, approval_hash, full_hash=True)
        if shared.evaluation.sha256(Path(__file__)) != own_source:
            raise RuntimeError("Probe implementation changed during execution")
        if any(shared.evaluation.sha256(Path(path)) != digest for path, digest in image_files.items()):
            raise RuntimeError("An engineering source RGB changed during execution")
        assert_frozen_versions(versions)
        artifact = {"format": FORMAT, "formal_training": False, "resume_allowed": False,
            "initialization_allowed": False, "completed_engineering_steps": len(updates),
            "actual_image_ids": [image_id for row in batch_records for image_id in row["image_ids"]],
            "cache_state_dict": cached_state}
        with (args.output_dir / "engineering_cache.pt").open("xb") as handle:
            torch.save(artifact, handle)
        summary = {**journal, "status": "complete", "completed_steps": len(updates),
            "full_zero_predictions_exactly_equal": True, "all_zero_predictions": all_predictions,
            "content_zero_predictions": content_predictions, "features": feature_check,
            "batches": batch_records, "updates": updates, "source_rgb_sha256": image_files,
            "frozen_parameter_versions": {name: version for name, _, version in versions},
            "frozen_parameters_unchanged": True, "trainable_parameter_count": content.delta.numel(),
            "engineering_artifact_sha256": shared.evaluation.sha256(args.output_dir / "engineering_cache.pt"),
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(context.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(context.device)}
        shared.evaluation.atomic_write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({"status": "complete", "engineering_steps": len(updates),
                          "formal_training": False, "full_zero_predictions_exactly_equal": True}), flush=True)
    except BaseException as error:
        shared.evaluation.atomic_write_json(args.output_dir / "summary.json", {
            **journal, "status": "failed", "completed_steps": len(updates), "updates": updates,
            "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    run(trainer.parse_args())

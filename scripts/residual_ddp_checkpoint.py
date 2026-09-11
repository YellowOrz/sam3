"""Versioned, token-only DDP checkpoints; no silent world-size/batch changes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from scripts import train_ve_initialized_tokens as shared
from scripts.residual_ddp_runtime import DeterministicGlobalBatchSampler

FORMAT = "sam3-output-residual-ddp-v1"
RANK_CACHE_AUDIT_FORMAT = "sam3-ddp-rank-cache-consistency-v1"
OPTIMIZER_OPTIONS = {"betas": (.9, .999), "eps": 1e-8, "amsgrad": False, "maximize": False,
                     "foreach": None, "capturable": False, "differentiable": False, "fused": None,
                     "decoupled_weight_decay": True}


def validate_rng_state(state):
    """Validate using private CPU generators without perturbing any training RNG."""
    try:
        if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
            raise ValueError("Missing RNG fields")
        random.Random().setstate(state["python"])
        np_state = state["numpy"]
        keys = np_state["keys"]
        if (not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64 or tuple(keys.shape) != (624,)
                or not bool(((keys >= 0) & (keys <= 2**32-1)).all())
                or np_state["algorithm"] != "MT19937" or type(np_state["position"]) is not int
                or not 0 <= np_state["position"] <= 624 or np_state["has_gauss"] not in (0, 1)
                or not math.isfinite(np_state["cached_gaussian"])):
            raise ValueError("Invalid NumPy RNG fields")
        np.random.RandomState().set_state((np_state["algorithm"], keys.cpu().numpy().astype(np.uint32),
            np_state["position"], np_state["has_gauss"], np_state["cached_gaussian"]))
        for key in ("torch_cpu", "torch_cuda"):
            value = state[key]
            if key == "torch_cuda" and value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
                raise ValueError("Invalid Torch RNG tensor")
        torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid rank RNG state: {error}") from error


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_rank_cache_consistency(rank_states_or_fingerprints, *, world_size=None):
    """Check actual rank-ordered encoder states/hashes collected at a save boundary.

    This function performs no communication, mutation or RNG operations. Each
    rank must fingerprint its own current encoder.state_dict() before the caller
    gathers those values; repeating rank zero's hash is not a synchronization
    check. Passing complete state dictionaries is also supported for CPU tests.
    The audit concerns the entire encoder state, including delta and frozen cache
    buffers/metadata, rather than an approximate norm or gradient comparison.
    """
    if not isinstance(rank_states_or_fingerprints, (list, tuple)) or not rank_states_or_fingerprints:
        raise ValueError("Rank cache audit needs a nonempty rank-ordered list")
    if world_size is None:
        world_size = len(rank_states_or_fingerprints)
    if type(world_size) is not int or world_size < 1 or len(rank_states_or_fingerprints) != world_size:
        raise ValueError("Rank cache audit count does not match world_size")
    fingerprints = []
    for rank, value in enumerate(rank_states_or_fingerprints):
        if isinstance(value, Mapping):
            delta = value.get("delta")
            if (not isinstance(delta, torch.Tensor) or delta.dtype != torch.float32
                    or tuple(delta.shape) != (2, 4, 256) or not bool(torch.isfinite(delta).all())):
                raise ValueError(f"Rank {rank} cache audit has an invalid output residual")
            value = shared.cache_fingerprint(value)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError(f"Rank {rank} cache fingerprint must be a lowercase SHA256 digest")
        fingerprints.append(value)
    different = [rank for rank, value in enumerate(fingerprints) if value != fingerprints[0]]
    if different:
        raise ValueError(f"Encoder cache differs from rank 0 on ranks {different}; refusing a synchronized checkpoint")
    return {"format": RANK_CACHE_AUDIT_FORMAT, "world_size": world_size,
            "scope": "full_encoder_state_dict_including_output_delta",
            "all_ranks_identical": True, "cache_sha256": fingerprints[0],
            "rank_cache_sha256": fingerprints}


def _validate_rank_cache_audit(audit, cached_state, world_size):
    if not isinstance(audit, dict):
        raise ValueError("Rank cache audit must be an object")
    expected = validate_rank_cache_consistency(audit.get("rank_cache_sha256"), world_size=world_size)
    if audit != expected or audit["cache_sha256"] != shared.cache_fingerprint(cached_state):
        raise ValueError("Rank cache audit does not bind this checkpoint's actual encoder state")


def progress_at(step, config):
    per_epoch = config["steps_per_epoch"]
    if type(step) is not int or not 0 <= step <= per_epoch * config["epochs"]:
        raise ValueError("Invalid completed optimizer-step count")
    epoch, offset = divmod(step, per_epoch)
    return {"global_step": step, "next_epoch": epoch, "next_step_in_epoch": offset,
            "samples_seen": step * config["global_batch_size"],
            "planned_steps": per_epoch * config["epochs"],
            "planned_samples": per_epoch * config["epochs"] * config["global_batch_size"],
            "completed_epochs": epoch, "training_complete": epoch == config["epochs"],
            "dropped_images_per_epoch": config["dataset_size"] % config["global_batch_size"]}


def expected_rank_ids(image_ids, config, step, rank):
    result = []
    full_epochs, partial = divmod(step, config["steps_per_epoch"])
    for epoch in range(full_epochs + (partial > 0)):
        sampler = DeterministicGlobalBatchSampler(
            len(image_ids), config["batch_size_per_rank"], rank=rank,
            world_size=config["world_size"], seed=config["seed"], epoch=epoch)
        count = config["steps_per_epoch"] if epoch < full_epochs else partial
        for index, batch in enumerate(sampler):
            if index >= count:
                break
            result.extend(image_ids[i] for i in batch)
    return result


def validate_resume(state, config, image_ids, initial_state):
    if state.get("format") != FORMAT or state.get("training_config") != config:
        raise ValueError("Checkpoint/config mismatch: data, code, batch, ranks and optimization must stay fixed")
    if state.get("config_sha256") != canonical_hash(config):
        raise ValueError("Checkpoint configuration fingerprint mismatch")
    step = state.get("progress", {}).get("global_step")
    if state.get("progress") != progress_at(step, config):
        raise ValueError("Checkpoint progress is inconsistent")
    if state.get("initial_cache_sha256") != shared.cache_fingerprint(initial_state):
        raise ValueError("Initial semantic cache changed")
    if shared.cache_fingerprint(state.get("initial_cache_state_dict", {})) != shared.cache_fingerprint(initial_state):
        raise ValueError("Saved initial semantic cache differs from the declared initializer")
    cached_state = state.get("cache_state_dict", {})
    if set(cached_state) != set(initial_state):
        raise ValueError("Cache schema changed")
    for name, original in initial_state.items():
        current = cached_state[name]
        if name == "delta":
            if (not isinstance(current, torch.Tensor) or current.shape != original.shape
                    or current.dtype != torch.float32 or not bool(torch.isfinite(current).all())):
                raise ValueError("Invalid output residual")
        elif isinstance(original, torch.Tensor):
            if (not isinstance(current, torch.Tensor) or current.dtype != original.dtype
                    or current.shape != original.shape or not torch.equal(current.cpu(), original.cpu())):
                raise ValueError(f"Frozen cache changed: {name}")
        elif current != original:
            raise ValueError(f"Cache metadata changed: {name}")
    # Old v1 recovery files predate this optional audit. Once present, it must
    # bind all expected ranks to the exact encoder state carried by this file.
    if "rank_cache_audit" in state:
        _validate_rank_cache_audit(state["rank_cache_audit"], cached_state, config["world_size"])
    ranks = state.get("rank_states", [])
    if len(ranks) != config["world_size"]:
        raise ValueError("Per-rank recovery states missing")
    for rank, item in enumerate(ranks):
        if item.get("image_ids") != expected_rank_ids(image_ids, config, step, rank) or not item.get("rng"):
            raise ValueError(f"Rank {rank} actual sample identities or RNG state changed")
        validate_rng_state(item["rng"])
    opt = state.get("optimizer", {})
    groups = opt.get("param_groups", [])
    if (len(groups) != 1 or groups[0].get("params") != [0]
            or groups[0].get("lr") != config["learning_rate"] or groups[0].get("weight_decay") != 0):
        raise ValueError("Unexpected optimizer parameter group")
    if any(groups[0].get(key) != value for key, value in OPTIMIZER_OPTIONS.items()):
        raise ValueError("AdamW options changed")
    moments = opt.get("state", {})
    if step == 0 and moments:
        raise ValueError("Initial optimizer must be empty")
    if step:
        if set(moments) != {0} or float(moments[0].get("step", -1)) != step:
            raise ValueError("Optimizer step does not match completed updates")
        for name in ("exp_avg", "exp_avg_sq"):
            value = moments[0].get(name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != (2, 4, 256)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid optimizer moments")
    return step


def make_checkpoint(encoder, optimizer, config, step, initial_state, rank_states, *, rank_cache_audit=None):
    result = {"format": FORMAT, "training_config": deepcopy(config), "config_sha256": canonical_hash(config),
            "progress": progress_at(step, config), "cache_state_dict": shared.cpu_state(encoder.state_dict()),
            "initial_cache_state_dict": shared.cpu_state(initial_state),
            "initial_cache_sha256": shared.cache_fingerprint(initial_state),
            "optimizer": deepcopy(optimizer.state_dict()), "rank_states": deepcopy(rank_states),
            "trainable_parameter_count": 2048, "accuracy_evaluated": False}
    if rank_cache_audit is not None:
        _validate_rank_cache_audit(rank_cache_audit, result["cache_state_dict"], config["world_size"])
        result["rank_cache_audit"] = deepcopy(rank_cache_audit)
    return result


def atomic_checkpoint(path: Path, value, *, replace=False):
    shared.atomic_save(path, value, replace=replace)

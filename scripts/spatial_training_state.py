"""Separate checkpoint contract for spatial adapters, never token checkpoints."""
import math
import torch
from scripts.residual_ddp_checkpoint import validate_rng_state

FORMAT = "sam3-spatial-mask-ddp-v1"


def validate_state(state, config, template):
    if state.get("format") != FORMAT or state.get("config") != config:
        raise ValueError("Spatial checkpoint format/configuration mismatch")
    step = state.get("step")
    if type(step) is not int or not 0 <= step <= config["epochs"] * config["steps_per_epoch"]:
        raise ValueError("Invalid spatial training step")
    tensors = state.get("adapter", {})
    if set(tensors) != set(template):
        raise ValueError("Adapter parameter names mismatch")
    for name, expected in template.items():
        value = tensors[name]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                or value.dtype != expected.dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid adapter tensor: {name}")
    ranks = state.get("rank_rng", [])
    if len(ranks) != config["world_size"]:
        raise ValueError("Wrong rank RNG count")
    for rng in ranks:
        validate_rng_state(rng)
    last = state.get("last_validation_step")
    if last is not None and (type(last) is not int or not 0 <= last <= step
                             or last % config["steps_per_epoch"]):
        raise ValueError("Invalid validation boundary")
    best = state.get("best_boundary")
    if best is not None and (type(best) not in (int, float) or not math.isfinite(best) or not 0 <= best <= 1):
        raise ValueError("Invalid best boundary metric")
    optimizer = state.get("optimizer", {})
    offset = config.get("optimizer_step_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid optimizer step offset")
    groups = optimizer.get("param_groups", [])
    if (len(groups) != 1 or groups[0].get("lr") != config["learning_rate"]
            or groups[0].get("weight_decay") != 0 or len(groups[0].get("params", [])) != len(template)):
        raise ValueError("Optimizer group configuration mismatch")
    states = optimizer.get("state", {})
    if step + offset > 0:
        if set(states) != set(groups[0]["params"]):
            raise ValueError("Missing optimizer states")
        for parameter_id, parameter in zip(groups[0]["params"], template.values()):
            entry = states[parameter_id]
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid optimizer moment")
            if float(entry.get("step", -1)) != step + offset:
                raise ValueError("Optimizer step mismatch")
    return step


def transfer_stage(source, config, template, source_hash):
    """Explicit dataset-stage transition, not an ordinary resume.

    New sampler starts at stage epoch zero; AdamW counters/moments and RNG
    survive. Initial validation is rerun; source best remains in its archive.
    """
    old = source['config']
    validate_state(source, old, template)
    if (source['step'] != old['epochs'] * old['steps_per_epoch']
            or source['last_validation_step'] != source['step']):
        raise ValueError('Stage source must be complete and validated')
    allowed = {'epochs', 'steps_per_epoch', 'approval_sha256', 'annotation_hashes',
               'implementation', 'stage_source_sha256', 'optimizer_step_offset'}
    for key in set(old) | set(config):
        if key not in allowed and old.get(key) != config.get(key):
            raise ValueError(f'Forbidden stage configuration change: {key}')
    if old['annotation_hashes']['val'] != config['annotation_hashes']['val']:
        raise ValueError('Validation set must remain fixed')
    if (config.get('stage_source_sha256') != source_hash
            or config.get('optimizer_step_offset') != source['step'] + old.get('optimizer_step_offset', 0)):
        raise ValueError('Stage provenance/optimizer offset mismatch')
    result = dict(source, config=config, step=0, last_validation_step=None, best_boundary=None)
    validate_state(result, config, template)
    return result

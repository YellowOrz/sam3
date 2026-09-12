"""Explicit, non-destructive epoch-budget extension of a completed residual run.

Run this helper outside the frozen training snapshot, with that snapshot on
PYTHONPATH. The ordinary trainer/resume contract remains unchanged. This is a
budget-only continuation, not a reset of optimizer, sample history or patience.
For now the completed source must also be the selected best checkpoint. A
separate historical-best migration is deliberately not supported by this helper.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import torch

from scripts import residual_ddp_checkpoint as checkpoint
from scripts import train_ve_initialized_tokens as shared


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def extend_budget(state, epochs, image_ids, initial_state, source_sha256, reason, *,
                  annotations_sha256, initial_cache_file_sha256):
    """Validate the old/new recovery contracts and return an independent copy.

Identity/cache bindings must come from the actual approved training annotation
file and initializer, not merely be copied from the checkpoint. The CLI below
establishes these bindings; approval and implementation matching remain subject
to the unchanged trainer's preflight and resume checks.
    """
    config = state["training_config"]
    _digest(source_sha256, "source_sha256")
    _digest(annotations_sha256, "annotations_sha256")
    _digest(initial_cache_file_sha256, "initial_cache_file_sha256")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A nonempty budget-extension reason is required")
    if (not isinstance(image_ids, (list, tuple))
            or any(type(value) is not int for value in image_ids)
            or len(image_ids) != len(set(image_ids))
            or list(image_ids) != sorted(image_ids)
            or len(image_ids) != config["dataset_size"]
            or checkpoint.canonical_hash(list(image_ids)) != config.get("image_order_sha256")):
        raise ValueError("Actual training image identities/order differ from the checkpoint")
    if annotations_sha256 != config["annotations_sha256"]:
        raise ValueError("Actual training annotation SHA256 changed")
    if initial_cache_file_sha256 != config["initial_cache_file_sha256"]:
        raise ValueError("Actual initial-cache file SHA256 changed")
    if shared.cache_fingerprint(initial_state) != config.get("initial_state_sha256"):
        raise ValueError("Actual initial semantic state differs from the training configuration")
    step = checkpoint.validate_resume(state, config, list(image_ids), initial_state)
    if type(epochs) is not int or epochs <= config["epochs"]:
        raise ValueError("New epoch budget must be an integer strictly greater than the old budget")
    if step != config["epochs"] * config["steps_per_epoch"]:
        raise ValueError("Only completed training budgets may be extended")
    validation = state.get("validation_state")
    if not config.get("validation", {}).get("enabled") or not isinstance(validation, dict):
        raise ValueError("A completed, fully validated source checkpoint is required")
    selection = validation["selection"]
    if selection["stopped"]:
        raise ValueError("A terminal early-stop decision cannot be cleared by extending the budget")
    if (validation["last_validation_step"] != step
            or selection["last_validation_step"] != step
            or selection["last_validation_epoch"] != config["epochs"]):
        raise ValueError("The completed final epoch must have full selector validation")
    if selection["best_step"] != step:
        raise ValueError("Source must be the selected best; separate historical-best migration is not supported")

    result = deepcopy(state)
    result["training_config"]["epochs"] = epochs
    result["config_sha256"] = checkpoint.canonical_hash(result["training_config"])
    result["progress"] = checkpoint.progress_at(step, result["training_config"])
    extension = {"format": "sam3-residual-completed-budget-extension-v1",
                 "source_sha256": source_sha256, "source_config_sha256": state["config_sha256"],
                 "old_epochs": config["epochs"], "new_epochs": epochs,
                 "completed_step": step, "reason": reason}
    if "budget_extension" in state:
        extension["parent_extension"] = deepcopy(state["budget_extension"])
    result["budget_extension"] = extension
    checkpoint.validate_resume(result, result["training_config"], list(image_ids), initial_state)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "training-data-root", "initial-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    source_bytes = args.source.read_bytes()
    state = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=True)
    config = state["training_config"]
    annotation_bytes = (args.training_data_root / "annotations.json").read_bytes()
    rows = json.loads(annotation_bytes)["images"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Training annotations must contain image records")
    ids = [row["id"] for row in rows]
    if any(type(value) is not int for value in ids):
        raise ValueError("Training image IDs must be integers")
    image_ids = sorted(ids)  # Same order as residual_ddp_data.load_coco_contract.
    cache_hash = shared.evaluation.sha256(args.initial_cache)
    initial_state = shared.load_initial_cache(args.initial_cache,
        base_hash=config["base_sha256"], tokenizer_hash=config["tokenizer_sha256"]).state_dict()
    if shared.evaluation.sha256(args.initial_cache) != cache_hash:
        raise ValueError("Initial-cache file changed during validation")
    result = extend_budget(state, args.epochs, image_ids, initial_state,
        hashlib.sha256(source_bytes).hexdigest(), args.reason,
        annotations_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
        initial_cache_file_sha256=cache_hash)
    if args.source.read_bytes() != source_bytes:
        raise ValueError("Source checkpoint changed during validation")
    if (args.training_data_root / "annotations.json").read_bytes() != annotation_bytes:
        raise ValueError("Training annotations changed during validation")
    if shared.evaluation.sha256(args.initial_cache) != cache_hash:
        raise ValueError("Initial-cache file changed during validation")
    # Exclusive creation also refuses a target that appeared after the initial
    # check. Retain an incomplete file as evidence if serialization fails; it
    # can never be accepted by the trainer's strict checkpoint validation.
    with args.output.open("xb") as handle:
        torch.save(result, handle)
    restored = torch.load(args.output, map_location="cpu", weights_only=True)
    checkpoint.validate_resume(restored, result["training_config"], image_ids, initial_state)
    print(json.dumps({"old_epochs": config["epochs"], "new_epochs": args.epochs,
        "source_sha256": result["budget_extension"]["source_sha256"],
        "output_sha256": shared.evaluation.sha256(args.output),
        "global_step": restored["progress"]["global_step"],
        "planned_steps": restored["progress"]["planned_steps"],
        "selector_best_step": restored["validation_state"]["selection"]["best_step"],
        "selector_bad_epochs": restored["validation_state"]["selection"]["bad_epochs"]}, sort_keys=True))


if __name__ == "__main__":
    main()

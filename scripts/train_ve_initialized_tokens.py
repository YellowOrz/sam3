#!/usr/bin/env python3
"""Short DexYCB-only pilot: full natural-VE features plus 2048 zero-delta parameters.

Never updates the frozen SAM3 base, never trains on nakehand, and never calls
2000 samples two epochs. A separate evaluator measures any accuracy change.
"""

from __future__ import annotations

import argparse
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
    from . import cached_ve_text_features as cached
    from . import evaluate_bilateral_tokens as evaluation
    from . import train_learnable_tokens as legacy
else:
    import cached_ve_text_features as cached
    import evaluate_bilateral_tokens as evaluation
    import train_learnable_tokens as legacy


FORMAT = "sam3-ve-initialized-delta-training-v1"
TRAIN_SHA256 = "4064ffed925873df2629c3043050711c1b1925f22b943b1d085bf5e113fc2dbf"
PLANNED_SAMPLES = 2000
LOSS_WEIGHTS = {"mask": 1., "dice": 1., "bbox": 1., "giou": 1., "classification": 1., "presence": 1.}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "initial-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=PLANNED_SAMPLES,
                        help="Stop this process after N successful total steps; does not change the planned 2000-sample pilot")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args(argv)
    if not 1 <= args.max_steps <= PLANNED_SAMPLES or args.log_every < 1:
        parser.error("max-steps must be in [1,2000], log-every positive")
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("GPU memory fraction must lie in (0,1]")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.tokenizer_path is None:
        args.tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if args.output_dir.exists():
        parser.error("Output directory must be new, including for resumed processes")
    return args


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}|{tuple(value.shape)}|".encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_fingerprint(state: dict) -> str:
    return json_hash({name: tensor_hash(value) if isinstance(value, torch.Tensor) else value
                      for name, value in sorted(state.items())})


def cpu_state(state: dict) -> dict:
    return {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for name, value in state.items()}


def core_source_hashes(project_root: Path) -> dict:
    sources = sorted((project_root / "sam3").rglob("*.py"))
    if not sources:
        raise ValueError("No SAM3 core sources found at explicit project-root")
    return {str(path.relative_to(project_root)): evaluation.sha256(path) for path in sources}


def load_initial_cache(path: Path, *, base_hash: str, tokenizer_hash: str) -> cached.CachedVETextEncoder:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("_extra_state", {}).get("mode") not in ("frozen", "zero_delta"):
        raise ValueError("Initial cache must be a frozen/zero-delta CachedVETextEncoder.state_dict")
    metadata = state["_extra_state"]["metadata"]
    if metadata.get("base_checkpoint_sha256") != base_hash or metadata.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Initial VE cache base/tokenizer hashes differ from actual files")
    if "delta" in state and (not isinstance(state["delta"], torch.Tensor)
                             or tuple(state["delta"].shape) != (2, 4, 256)
                             or state["delta"].dtype != torch.float32
                             or not bool((state["delta"] == 0).all())):
        raise ValueError("Semantic pilot must start from exactly zero delta")
    encoder = cached.CachedVETextEncoder(state["padding_cache"], state["resized_cache"], state["raw_cache"],
                                        metadata=metadata, mode="zero_delta")
    if not torch.equal(encoder.valid_positions, state["valid_positions"]):
        raise ValueError("Initial cache valid positions differ from padding mask")
    if encoder.resized_cache.dtype != torch.bfloat16 or encoder.raw_cache.dtype != torch.float32:
        raise ValueError("This controlled pilot requires verified BF16 resized / FP32 raw VE features")
    return encoder


def training_config(args, annotation_summary, core_hashes, initial_fingerprint) -> dict:
    return {
        "experiment": "natural_ve_zero_delta_only", "batch_size": 1, "seed": 123,
        "learning_rate": .01, "optimizer": "AdamW", "weight_decay": 0.,
        "amp": True, "amp_dtype": "bfloat16", "loss_weights": dict(LOSS_WEIGHTS),
        "planned_samples": PLANNED_SAMPLES, "delta_shape": [2, 4, 256], "context_length": 32,
        "data_root": str(args.data_root), "base_checkpoint": str(args.base_checkpoint),
        "tokenizer_path": str(args.tokenizer_path), "initial_cache": str(args.initial_cache),
        "annotations_sha256": annotation_summary["sha256"],
        "base_checkpoint_sha256": evaluation.sha256(args.base_checkpoint),
        "tokenizer_sha256": evaluation.sha256(args.tokenizer_path),
        "initial_cache_sha256": evaluation.sha256(args.initial_cache),
        "initial_cache_state_sha256": initial_fingerprint,
        "core_sources_sha256": json_hash(core_hashes),
        "implementation_sha256": {path.name: evaluation.sha256(path) for path in (
            Path(__file__).resolve(), Path(inspect.getfile(cached)).resolve(),
            Path(inspect.getfile(evaluation)).resolve(), Path(inspect.getfile(legacy)).resolve())},
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "float32_matmul_precision": "high", "network_mode": "eval_with_delta_autograd",
    }


def progress(steps: int, dataset_size: int) -> dict:
    if type(steps) is not int or not 0 <= steps <= PLANNED_SAMPLES or dataset_size < PLANNED_SAMPLES:
        raise ValueError("Invalid pilot progress")
    return {"completed_steps": steps, "samples_seen": steps, "planned_steps": PLANNED_SAMPLES,
            "planned_samples": PLANNED_SAMPLES, "full_epoch_completed": steps == dataset_size,
            "pilot_complete": steps == PLANNED_SAMPLES}


def validate_resume(state, config, order, expected_ids, initial_state):
    if state.get("format") != FORMAT:
        raise ValueError("Resume checkpoint is not the independent VE-initialized delta format")
    legacy.validate_resume_training_config(state.get("training_config", {}), config)
    if state.get("planned_dataset_indices") != order:
        raise ValueError("Resume sample order differs")
    if state.get("planned_dataset_indices_sha256") != json_hash(order):
        raise ValueError("Resume sample-order fingerprint differs")
    step = state.get("next_step")
    if type(step) is not int or not 0 <= step <= len(order):
        raise ValueError("Resume next_step outside planned sample range")
    if state.get("observed_image_ids") != [expected_ids[index] for index in order[:step]]:
        raise ValueError("Resume successful observed identities do not match its completed prefix")
    if state.get("progress") != progress(step, len(expected_ids)):
        raise ValueError("Resume progress does not match successful steps")
    if state.get("observed_identity") != legacy.observed_identity_provenance(0, step, state["observed_image_ids"]):
        raise ValueError("Resume observed identity fingerprint/counts differ")
    if state.get("gradient_nonzero_steps") != [step, step]:
        raise ValueError("Resume gradient counters do not cover both sides at every successful step")
    if (state.get("annotation_summary", {}).get("sha256") != config["annotations_sha256"]
            or state.get("annotation_summary", {}).get("images") != len(expected_ids)):
        raise ValueError("Resume annotation provenance differs")
    if json_hash(state.get("core_source_hashes")) != config["core_sources_sha256"]:
        raise ValueError("Resume core-source map differs from its fingerprint")
    history = state.get("loss_history", [])
    if len(history) != step or any(not math.isfinite(value) for value in history):
        raise ValueError("Resume loss history must cover exactly successful steps")
    if cache_fingerprint(state.get("initial_cache_state_dict", {})) != cache_fingerprint(initial_state):
        raise ValueError("Resume semantic initialization differs")
    saved = state.get("cache_state_dict", {})
    for name, value in initial_state.items():
        if name == "delta":
            continue
        current = saved.get(name)
        same = (isinstance(current, torch.Tensor) and current.dtype == value.dtype and torch.equal(current, value)
                if isinstance(value, torch.Tensor) else current == value)
        if not same:
            raise ValueError(f"Frozen VE cache changed in resume: {name}")
    delta = saved.get("delta")
    if not isinstance(delta, torch.Tensor) or tuple(delta.shape) != (2, 4, 256) or delta.dtype != torch.float32 or not torch.isfinite(delta).all():
        raise ValueError("Resume delta is invalid")
    for name in ("base_checkpoint_sha256", "tokenizer_sha256", "initial_cache_sha256"):
        if state.get(name) != config[name]:
            raise ValueError(f"Resume top-level provenance differs: {name}")
    if "optimizer" not in state or "rng" not in state:
        raise ValueError("Resume optimizer/RNG state missing")
    validate_optimizer_state(state["optimizer"], step)
    return step


def validate_optimizer_state(state: dict, completed_steps: int):
    groups = state.get("param_groups", [])
    if len(groups) != 1 or len(groups[0].get("params", [])) != 1:
        raise ValueError("Resume AdamW must own only the single delta parameter")
    expected = {"lr": .01, "weight_decay": 0., "betas": (.9, .999), "eps": 1e-8,
                "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}
    if any(groups[0].get(name) != value for name, value in expected.items()):
        raise ValueError("Resume optimizer settings differ from the fixed AdamW control")
    values = state.get("state", {})
    if completed_steps == 0:
        if values:
            raise ValueError("Zero-step checkpoint unexpectedly contains optimizer moments")
        return
    if set(values) != set(groups[0]["params"]):
        raise ValueError("Resume optimizer moments missing or have extra parameters")
    moment = next(iter(values.values()))
    step = moment.get("step")
    if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all() or float(step) != completed_steps:
        raise ValueError("Resume AdamW step differs from actual successful steps")
    for name in ("exp_avg", "exp_avg_sq"):
        value = moment.get(name)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (2, 4, 256)
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid resume AdamW moment: {name}")
    if not bool((moment["exp_avg_sq"] >= 0).all()):
        raise ValueError("Negative AdamW squared moments")


def rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])]}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), values[2], values[3], values[4]))
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume visible CUDA device count differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_save(path: Path, state: dict, *, replace=False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
        temporary.replace(path)
    except BaseException:
        # Retain incomplete temporary evidence; never overwrite the previous recovery file.
        raise


def make_checkpoint(*, encoder, optimizer, config, initial_state, annotation_summary,
                    order, observed_ids, loss_history, gradient_counts, core_hashes):
    step = len(observed_ids)
    return {"format": FORMAT, "cache_state_dict": cpu_state(encoder.state_dict()),
            "initial_cache_state_dict": cpu_state(initial_state), "training_config": config,
            "annotation_summary": annotation_summary, "optimizer": optimizer.state_dict(),
            "next_step": step, "progress": progress(step, annotation_summary["images"]),
            "planned_dataset_indices": order, "planned_dataset_indices_sha256": json_hash(order),
            "observed_image_ids": list(observed_ids), "observed_identity": legacy.observed_identity_provenance(0, step, observed_ids),
            "loss_history": list(loss_history), "gradient_nonzero_steps": list(gradient_counts),
            "core_source_hashes": core_hashes, "rng": rng_state(),
            **{key: config[key] for key in ("base_checkpoint_sha256", "tokenizer_sha256", "initial_cache_sha256")}}


def build_model_with_matcher(args):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint), bpe_path=str(args.tokenizer_path),
                                   load_from_HF=False, device="cuda", eval_mode=False,
                                   enable_segmentation=True, enable_inst_interactivity=False, text_encoder_type="ve")
    model.eval()
    if not callable(getattr(model, "matcher", None)):
        raise RuntimeError("Training requires builder eval_mode=False to construct the matcher")
    if getattr(model, "num_interactive_steps_val", None) != 0:
        raise RuntimeError("Reference-derived interactive prompting is forbidden")
    return model


def build_loss_functions():
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks

    return (
        Masks(weight_dict={"loss_mask": 1., "loss_dice": 1.}, compute_aux=False, focal_alpha=.25, focal_gamma=2.),
        Boxes(weight_dict={"loss_bbox": 1., "loss_giou": 1.}, compute_aux=False),
        IABCEMdetr(weight_dict={"loss_ce": 1., "presence_loss": 1.}, compute_aux=False,
                  pos_weight=5., alpha=.25, gamma=2., weak_loss=False, use_presence=True,
                  presence_alpha=.5, presence_gamma=0., pos_focal=False),
    )


def compute_loss(model, batch, functions):
    targets = model.back_convert(batch.find_targets[0])
    if not 0 <= int((targets["num_boxes"] > 0).sum()) <= 1:
        raise RuntimeError("DexYCB batch1 requires zero or one physical-hand positive query")
    prediction = model(batch)[0]
    indices = model.matcher(prediction, targets)
    prediction["indices"] = indices
    count = targets["num_boxes"].sum().float().clamp(min=1)
    terms = [function(outputs=prediction, targets=targets, indices=indices, num_boxes=count) for function in functions]
    return sum(value["core_loss"] for value in terms), terms


def validate_unprompted_batch(batch):
    for field in ("input_boxes", "input_points", "input_boxes_before_embed", "input_points_before_embed"):
        value = getattr(batch.find_inputs[0], field, None)
        if value is not None and (not isinstance(value, torch.Tensor) or value.numel()):
            raise RuntimeError(f"Semantic token-only pilot forbids geometry prompts: {field}")


def verify_inputs(args, config, expected_core):
    for path, name in ((args.data_root / "annotations.json", "annotations_sha256"),
                       (args.base_checkpoint, "base_checkpoint_sha256"),
                       (args.tokenizer_path, "tokenizer_sha256"), (args.initial_cache, "initial_cache_sha256")):
        if evaluation.sha256(path) != config[name]:
            raise RuntimeError(f"Training input changed: {name}")
    if core_source_hashes(args.project_root) != expected_core:
        raise RuntimeError("SAM3 core sources changed during pilot")


def snapshot_scripts(output_dir: Path):
    directory = output_dir / "code-snapshot"
    directory.mkdir()
    sources = {Path(__file__).resolve(), Path(inspect.getfile(cached)).resolve(),
               Path(inspect.getfile(evaluation)).resolve(), Path(inspect.getfile(legacy)).resolve()}
    records = []
    for source in sorted(sources):
        target = directory / source.name
        shutil.copy2(source, target)
        if evaluation.sha256(source) != evaluation.sha256(target):
            raise RuntimeError("Script changed while snapshotting")
        records.append({"source": str(source), "snapshot": str(target), "sha256": evaluation.sha256(target)})
    return records


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU unit tests exercise contracts separately")
    annotation_path = args.data_root / "annotations.json"
    annotation_summary = legacy.inspect_training_annotations(annotation_path)
    annotation_data = json.loads(annotation_path.read_text())
    if annotation_summary["sha256"] != TRAIN_SHA256 or annotation_data.get("info", {}).get("split") != "train":
        raise ValueError("Only the fixed audited DexYCB train split is authorized; never nakehand/val/test")
    if evaluation.sha256(annotation_path) != annotation_summary["sha256"]:
        raise RuntimeError("Training annotations changed while indexing")
    image_ids = sorted(int(image["id"]) for image in annotation_data["images"])
    category_ids = {item["name"]: int(item["id"]) for item in annotation_data["categories"]}
    if len(image_ids) < PLANNED_SAMPLES:
        raise ValueError("Training set smaller than fixed pilot budget")
    order = legacy.build_epoch_order(len(image_ids), 123)[:PLANNED_SAMPLES]
    expected_core = core_source_hashes(args.project_root)
    base_hash, tokenizer_hash = evaluation.sha256(args.base_checkpoint), evaluation.sha256(args.tokenizer_path)
    initial_file_hash = evaluation.sha256(args.initial_cache)
    encoder = load_initial_cache(args.initial_cache, base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    if evaluation.sha256(args.initial_cache) != initial_file_hash:
        raise RuntimeError("Initial verified cache changed during loading")
    initial_state = cpu_state(encoder.state_dict())
    config = training_config(args, annotation_summary, expected_core, cache_fingerprint(initial_state))
    if (config["base_checkpoint_sha256"] != base_hash or config["tokenizer_sha256"] != tokenizer_hash
            or config["initial_cache_sha256"] != initial_file_hash):
        raise RuntimeError("Base/tokenizer/cache changed during setup")
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_step = validate_resume(resume_state, config, order, image_ids, initial_state)
        if start_step > args.max_steps:
            raise ValueError("Resume step exceeds this invocation stop step")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    code = snapshot_scripts(args.output_dir)
    atomic_save(args.output_dir / "initial_cache.pt", initial_state)
    evaluation.atomic_write_json(args.output_dir / "run.json", {
        "format": FORMAT, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "code_snapshots": code, "core_source_hashes": expected_core,
        "planned_dataset_indices": order, "planned_image_ids": [image_ids[index] for index in order],
        "initial_cache_state_sha256": cache_fingerprint(initial_state),
        "requested_stop_step": args.max_steps, "resume": str(args.resume) if args.resume else None,
        "scope": "short training-only semantic-delta pilot, no accuracy claim; frozen VE/cache evaluation is separate"})
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dataset = evaluation.make_dataset(args.data_root)
    if len(dataset) != len(image_ids):
        raise RuntimeError("Dataset/COCO identity count mismatch")
    model = build_model_with_matcher(args)
    encoder.to(device="cuda")
    previous = cached.install_cached_ve_text_encoder(model, encoder)
    del previous  # Original text weights are no longer needed for delta-only training.
    torch.cuda.empty_cache()
    names = cached.set_cached_ve_training_mode(model, train_delta=True)
    if names != {"backbone.language_backbone.delta"} or sum(p.numel() for p in model.parameters() if p.requires_grad) != 2048:
        raise RuntimeError(f"Unexpected trainable parameters: {names}")
    optimizer = torch.optim.AdamW([encoder.delta], lr=.01, weight_decay=0.)
    functions = build_loss_functions()
    observed_ids, history, gradient_counts = [], [], [0, 0]
    if resume_state is not None:
        encoder.load_state_dict(resume_state["cache_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer"])
        observed_ids = list(resume_state["observed_image_ids"])
        history = list(resume_state["loss_history"])
        gradient_counts = list(resume_state["gradient_nonzero_steps"])
        restore_rng(resume_state["rng"])
    verify_inputs(args, config, expected_core)

    def checkpoint():
        return make_checkpoint(encoder=encoder, optimizer=optimizer, config=config, initial_state=initial_state,
                               annotation_summary=annotation_summary, order=order, observed_ids=observed_ids,
                               loss_history=history, gradient_counts=gradient_counts, core_hashes=expected_core)

    latest = args.output_dir / "semantic_delta_latest.pt"
    atomic_save(latest, checkpoint())
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    started = time.monotonic()
    try:
        for step in range(len(observed_ids), args.max_steps):
            index = order[step]
            sample = dataset[index]
            batch = collate_fn_api([sample], dict_key="train", with_seg_masks=True)["train"]
            actual_ids = legacy.validate_training_batch_identity(batch, [image_ids[index]], category_ids)
            validate_unprompted_batch(batch)
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, terms = compute_loss(model, batch, functions)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss at attempted step {step + 1}")
            loss.backward()
            gradient = encoder.delta.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Missing/nonfinite semantic delta gradients")
            norms = gradient.flatten(1).norm(dim=1).detach().cpu()
            if not bool((norms > 0).all()):
                raise RuntimeError(f"Both class deltas must get nonzero gradient: {norms.tolist()}")
            optimizer.step()
            if not bool(torch.isfinite(encoder.delta).all()):
                raise RuntimeError("Optimizer produced nonfinite delta; retain previous recovery checkpoint")
            observed_ids.extend(actual_ids)
            history.append(float(loss.detach().cpu()))
            gradient_counts = [value + 1 for value in gradient_counts]
            if step == 0 or (step + 1) % args.log_every == 0:
                print(f"successful_steps={step+1}/{PLANNED_SAMPLES} stop_at={args.max_steps} image_id={actual_ids[0]} "
                      f"loss={history[-1]:.6f} left_grad={norms[0]:.6e} right_grad={norms[1]:.6e} "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)
            if (step + 1) % 100 == 0:
                if core_source_hashes(args.project_root) != expected_core:
                    raise RuntimeError("SAM3 core sources changed; stopping controlled pilot")
                current = checkpoint()
                atomic_save(args.output_dir / f"semantic_delta_step{step+1:05d}_recovery.pt", current)
                atomic_save(latest, current, replace=True)
            del sample, batch, loss, terms
    except BaseException as error:
        evaluation.atomic_write_json(args.output_dir / "failure.json", {
            "status": "failed_or_interrupted", "successful_steps_in_memory": len(observed_ids),
            "latest_recovery_checkpoint": str(latest), "error": f"{type(error).__name__}: {error}",
            "note": "No possibly mid-optimizer state is saved; recover from the last atomic checkpoint"})
        raise
    verify_inputs(args, config, expected_core)
    final_state = checkpoint()
    suffix = "pilot_complete" if len(observed_ids) == PLANNED_SAMPLES else "partial"
    final = args.output_dir / f"semantic_delta_step{len(observed_ids):05d}_{suffix}.pt"
    atomic_save(final, final_state)
    atomic_save(latest, final_state, replace=True)
    summary = {"format": FORMAT, "status": "completed_requested_steps", "training_config": config,
               "progress": final_state["progress"], "final_checkpoint": str(final),
               "final_checkpoint_sha256": evaluation.sha256(final),
               "observed_identity": final_state["observed_identity"], "gradient_nonzero_steps": gradient_counts,
               "delta_max_abs": float(encoder.delta.detach().abs().max().cpu()),
               "last_loss": history[-1] if history else None,
               "elapsed_seconds_this_process": time.monotonic() - started,
               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
               "accuracy_evaluated": False, "note": "2000 samples is a short pilot, not two epochs; no quality improvement claim"}
    evaluation.atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "progress": summary["progress"]}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate semantic VE delta and its unchanged cached baseline on DexYCB val.

Reuses the existing bilateral scoring, masks and renderer. No training, geometry,
threshold fitting, or nakehand selection. Run with caller-owned GPU scheduling
and an external timeout that ends no later than the recorded deadline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import random
import shutil

import torch

try:
    from scripts import cached_ve_text_features as cached
    from scripts import evaluate_bilateral_tokens as shared
    from scripts.evaluate_nakehand_tokens import validate_frozen_noninteractive_model
    from scripts.run_token_lr_pilot import core_source_hashes, object_hash
except ModuleNotFoundError:
    import cached_ve_text_features as cached
    import evaluate_bilateral_tokens as shared
    from evaluate_nakehand_tokens import validate_frozen_noninteractive_model
    from run_token_lr_pilot import core_source_hashes, object_hash


FORMAT = "sam3-ve-initialized-delta-training-v1"
DEADLINE = datetime.fromisoformat("2026-09-10T21:40:00+08:00")
IMMUTABLE_CACHE_KEYS = ("padding_cache", "resized_cache", "raw_cache", "valid_positions")


def cache_from_state(state, *, frozen=False):
    cached.validate_delta_state(state)
    mode = state["_extra_state"]["mode"]
    encoder = cached.CachedVETextEncoder(
        state["padding_cache"], state["resized_cache"], state["raw_cache"],
        metadata=state["_extra_state"]["metadata"], mode=mode)
    encoder.load_state_dict(state, strict=True)
    if not torch.isfinite(encoder.delta).all():
        raise ValueError("Trained delta must be finite")
    if frozen:
        return cached.CachedVETextEncoder(encoder.padding_cache, encoder.resized_cache,
                                         encoder.raw_cache, metadata=encoder.cache_metadata,
                                         mode="frozen")
    return encoder


def validate_checkpoint(state, *, minimum_samples, base_hash, tokenizer_hash):
    if state.get("format") != FORMAT:
        raise ValueError("Expected semantic VE delta checkpoint, not random class_tokens")
    if "class_tokens" in state:
        raise ValueError("Semantic delta checkpoint must not masquerade as class_tokens")
    config = state.get("training_config", {})
    for name, digest in (("base_checkpoint_sha256", base_hash), ("tokenizer_sha256", tokenizer_hash)):
        if state.get(name) != digest or config.get(name) != digest:
            raise ValueError(f"Checkpoint differs from current source: {name}")
    if config.get("batch_size") != 1 or config.get("amp") is not True:
        raise ValueError("Expected the fixed batch1 BF16 pilot configuration")
    current = state.get("cache_state_dict")
    initial = state.get("initial_cache_state_dict")
    # The reusable cache loader also serves newer DDP experiments. That must
    # not silently broaden this historical four-position pilot's protocol.
    if any(not isinstance(value, dict) or value.get("_extra_state", {}).get("mode") != "zero_delta"
           for value in (current, initial)):
        raise ValueError("Legacy semantic pilot requires the original zero_delta mode")
    current_encoder, initial_encoder = cache_from_state(current), cache_from_state(initial)
    if not torch.equal(initial_encoder.delta, torch.zeros_like(initial_encoder.delta)):
        raise ValueError("Initial semantic delta must be exactly zero")
    for key in IMMUTABLE_CACHE_KEYS:
        if not torch.equal(current[key], initial[key]):
            raise ValueError(f"Cached original VE features changed during training: {key}")
    if current_encoder.cache_metadata != initial_encoder.cache_metadata:
        raise ValueError("Cache metadata changed during training")
    for name, digest in (("base_checkpoint_sha256", base_hash), ("tokenizer_sha256", tokenizer_hash)):
        if current_encoder.cache_metadata[name] != digest:
            raise ValueError(f"Cache provenance differs: {name}")
    progress = state.get("progress", {})
    steps, samples = progress.get("completed_steps"), progress.get("samples_seen")
    if (type(steps) is not int or type(samples) is not int or steps < 0 or samples != steps
            or state.get("next_step") != steps or samples < minimum_samples
            or progress.get("planned_steps") != 2000 or progress.get("planned_samples") != 2000
            or steps > 2000 or progress.get("pilot_complete") is not (steps == 2000)):
        raise ValueError("Invalid/incomplete actual short-pilot progress")
    planned, observed = state.get("planned_dataset_indices"), state.get("observed_image_ids")
    if (not isinstance(planned, list) or len(planned) != 2000 or len(set(planned)) != 2000
            or any(type(index) is not int or index < 0 for index in planned)
            or not isinstance(observed, list) or len(observed) != samples):
        raise ValueError("Missing/invalid planned and observed training identities")
    return current_encoder, initial_encoder


def verify_training_identities(state):
    config = state["training_config"]
    root = Path(config["data_root"])
    path = root / "annotations.json"
    raw = path.read_bytes()
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    if (digest != config.get("annotations_sha256")
            or digest != state.get("annotation_summary", {}).get("sha256")):
        raise ValueError("Original training annotations changed or have inconsistent provenance")
    data = json.loads(raw)
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    if state.get("annotation_summary", {}).get("images") != len(images):
        raise ValueError("Training annotation image count differs from checkpoint")
    order = list(range(len(images)))
    random.Random(config["seed"]).shuffle(order)
    if state["planned_dataset_indices"] != order[:2000]:
        raise ValueError("Planned images differ from recorded seed/annotation order")
    observed = [int(images[index]["id"]) for index in order[:state["progress"]["samples_seen"]]]
    if observed != state["observed_image_ids"]:
        raise ValueError("Observed successful training images differ from the planned prefix")
    return {"annotation_path": str(path.resolve()), "annotations_sha256": digest,
            "observed_samples": len(observed), "actual_prefix_verified": True}


def verify_initial_cache_artifact(state):
    config = state["training_config"]
    path = Path(config["initial_cache"])
    digest = shared.sha256(path)
    if digest != config.get("initial_cache_sha256") or digest != state.get("initial_cache_sha256"):
        raise ValueError("Verified initial VE cache artifact differs from training provenance")
    original = torch.load(path, map_location="cpu", weights_only=True)
    initial = state["initial_cache_state_dict"]
    for name in IMMUTABLE_CACHE_KEYS:
        if original[name].dtype != initial[name].dtype or not torch.equal(original[name], initial[name]):
            raise ValueError(f"Initial features differ from verified cache artifact: {name}")
    if original["_extra_state"]["metadata"] != initial["_extra_state"]["metadata"]:
        raise ValueError("Initial cache metadata differs from original artifact")
    if shared.sha256(path) != digest:
        raise RuntimeError("Initial cache changed during reading")
    return {"path": str(path.resolve()), "sha256": digest, "verified_against_initial_state": True}


def assert_finite_model_outputs(_model, _inputs, outputs):
    prediction = outputs[0]
    for name in ("pred_logits", "presence_logit_dec", "pred_boxes", "pred_masks"):
        if not isinstance(prediction.get(name), torch.Tensor) or not bool(torch.isfinite(prediction[name]).all()):
            raise RuntimeError(f"Nonfinite or missing model output: {name}")


class IdentityCheckedDataset:
    """Reject loader fallback before the unchanged shared evaluator collates."""
    def __init__(self, dataset, images):
        self.dataset, self.images, self.observed_indices = dataset, images, []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        expected = int(self.images[index]["id"])
        queries = sample.find_queries
        if len(queries) != 2 or {query.query_text for query in queries} != set(shared.CLASS_NAMES):
            raise RuntimeError("Expected exactly two distinct hand queries per image")
        for query in queries:
            metadata = query.inference_metadata
            if (query.image_id != 0 or metadata.coco_image_id != expected
                    or metadata.original_category_id != shared.CLASS_NAMES.index(query.query_text) + 1):
                raise RuntimeError("Loader substituted an image or prompt category")
            if any(getattr(query, name, None) is not None and getattr(query, name).numel()
                   for name in ("input_bbox", "input_points")):
                raise RuntimeError("Reference-derived geometry prompts are forbidden")
        self.observed_indices.append(index)
        return sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "delta-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--unified-root", type=Path, default=Path("/data/xuzhefeng/Datasets/uni-hoi-dataset"))
    parser.add_argument("--indices", default=None, help="Optional explicitly labelled diagnostic smoke subset")
    parser.add_argument("--minimum-samples-seen", type=int, default=2000)
    parser.add_argument("--render-count-per-group", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    parser.add_argument("--deadline", default=DEADLINE.isoformat())
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline)
    if args.deadline.tzinfo is None or args.deadline > DEADLINE:
        parser.error("Deadline must not exceed 2026-09-10 21:40 +08:00")
    if not 0 < args.gpu_memory_fraction <= .25 or not 0 <= args.minimum_samples_seen <= 2000:
        parser.error("Invalid memory fraction or minimum sample count")
    if args.render_count_per_group < 0:
        parser.error("Render count must be nonnegative")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists():
        parser.error("Output directory must not exist")
    if args.output_dir == args.data_root or args.data_root in args.output_dir.parents:
        parser.error("Output must not be inside source data")
    return args


def main(argv=None):
    args = parse_args(argv)
    if datetime.now(timezone.utc) >= args.deadline:
        raise RuntimeError("Authorized deadline has passed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    annotation = args.data_root / "annotations.json"
    annotation_hash = shared.sha256(annotation)
    annotation_info = json.loads(annotation.read_text()).get("info", {})
    if annotation_info.get("split") != "val":
        raise ValueError("This pilot evaluator permits only the independent DexYCB val split")
    images, annotations_by_image = shared.load_coco_index(args.data_root)
    if shared.sha256(annotation) != annotation_hash:
        raise RuntimeError("Validation annotations changed during loading")
    indices = shared.choose_indices(images, annotations_by_image, 0, args.indices)
    if not indices:
        raise ValueError("No evaluation images selected")
    base_hash = shared.sha256(args.base_checkpoint)
    tokenizer = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    tokenizer_hash = shared.sha256(tokenizer)
    checkpoint_hash = shared.sha256(args.delta_checkpoint)
    state = torch.load(args.delta_checkpoint, map_location="cpu", weights_only=True)
    if shared.sha256(args.delta_checkpoint) != checkpoint_hash:
        raise RuntimeError("Delta checkpoint changed during loading")
    delta, initial = validate_checkpoint(state, minimum_samples=args.minimum_samples_seen,
                                         base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    precision = state["training_config"].get("float32_matmul_precision", "high")
    if precision != "high":
        raise ValueError("Unexpected float32 matmul policy for the controlled pilot")
    torch.set_float32_matmul_precision(precision)
    training_identity = verify_training_identities(state)
    initial_cache_artifact = verify_initial_cache_artifact(state)
    core = core_source_hashes(args.project_root)
    configured_core = state["training_config"].get("core_sources_sha256")
    if configured_core != object_hash(core):
        raise ValueError("Current SAM3 core differs from the training core source fingerprint")
    rgb_sources = [{"index": index, "image_id": images[index]["id"],
                    "path": str(args.data_root / images[index]["file_name"]),
                    "sha256": shared.sha256(args.data_root / images[index]["file_name"])} for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_dir = args.output_dir / "code-snapshot"
    snapshot_dir.mkdir()
    snapshots = []
    for source in {Path(__file__).resolve(), Path(inspect.getfile(shared)).resolve(),
                   Path(inspect.getfile(cached)).resolve(),
                   Path(inspect.getfile(validate_frozen_noninteractive_model)).resolve(),
                   args.project_root / "scripts/render_separated_masks.py",
                   args.project_root / "scripts/run_token_lr_pilot.py"}:
        target = snapshot_dir / source.name
        shutil.copy2(source, target)
        snapshots.append({"source": str(source), "snapshot": str(target), "sha256": shared.sha256(target)})
    shared.atomic_write_json(args.output_dir / "rgb-identities.json", rgb_sources)
    (args.output_dir / "records").mkdir()
    summary = {"format": "sam3-ve-delta-bilateral-evaluation-v1", "status": "running",
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "deadline": args.deadline.isoformat(),
               "data_root": str(args.data_root), "annotations_sha256": annotation_hash,
               "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": base_hash,
               "tokenizer_sha256": tokenizer_hash, "delta_checkpoint": str(args.delta_checkpoint),
               "delta_checkpoint_sha256": checkpoint_hash, "training_progress": state["progress"],
               "training_identity": training_identity, "initial_cache_sha256": state.get("initial_cache_sha256"),
               "initial_cache_artifact": initial_cache_artifact,
               "core_source_sha256": object_hash(core), "core_sources": core, "code_snapshots": snapshots,
               "evaluated_images": len(indices), "evaluated_dataset_indices": indices,
               "full_val_evaluated": len(indices) == len(images), "diagnostic_subset": args.indices is not None,
               "diagnostic_training_checkpoint": not state["progress"]["pilot_complete"],
               "batch_size": 1, "amp": True, "amp_dtype": "bfloat16",
               "float32_matmul_precision": precision,
               "gpu_memory_fraction": args.gpu_memory_fraction,
               "detection_threshold": .5, "mask_threshold": .5,
               "confidence_definition": "sigmoid(pred_logits) * sigmoid(presence_logit_dec)",
               "training_performed": False, "thresholds_fitted": False, "visual_style": "separate",
               "models": {}, "metrics": {}}
    shared.atomic_write_json(args.output_dir / "progress.json", summary)
    try:
        raw_dataset = shared.make_dataset(args.data_root)
        if len(raw_dataset) != len(images):
            raise RuntimeError("SAM3 loader/COCO length mismatch")
        dataset = IdentityCheckedDataset(raw_dataset, images)
        render_indices = shared.choose_render_indices(indices, images, annotations_by_image, args.render_count_per_group)
        model = shared.load_ve_model(args.base_checkpoint)
        model.register_forward_hook(assert_finite_model_outputs)
        original_ve = model.backbone.language_backbone
        original_versions = [(name, parameter, parameter._version) for name, parameter in model.named_parameters()]
        all_records, all_masks = [], {}
        frozen = cached.CachedVETextEncoder(initial.padding_cache, initial.resized_cache, initial.raw_cache,
                                            metadata=initial.cache_metadata, mode="frozen")
        for label, encoder in (("ve-frozen-cache", frozen), ("ve-delta", delta)):
            if datetime.now(timezone.utc) >= args.deadline:
                raise RuntimeError("Deadline reached before the next model variant")
            encoder.to(device="cuda")
            cached.install_cached_ve_text_encoder(model, encoder)
            cached.set_cached_ve_training_mode(model, train_delta=False)
            validate_frozen_noninteractive_model(model)
            cache_versions = [(parameter, parameter._version) for parameter in encoder.parameters()]
            before = len(dataset.observed_indices)
            records, masks = shared.evaluate_variant(
                model=model, label=label, prompt_texts=shared.CLASS_NAMES, dataset=dataset, images=images,
                annotations_by_image=annotations_by_image, eval_indices=indices,
                render_indices=set(render_indices), batch_size=1, detection_threshold=.5,
                mask_threshold=.5, amp=True, unified_root=args.unified_root)
            if dataset.observed_indices[before:] != indices:
                raise RuntimeError("Actual evaluation image order differs from requested indices")
            if any(parameter._version != version for parameter, version in cache_versions):
                raise RuntimeError("Cached delta parameters changed during evaluation")
            expected = {(int(images[index]["id"]), side) for index in indices for side in shared.CLASS_NAMES}
            if len(records) != len(expected) or {(row["image_id"], row["prompt_key"]) for row in records} != expected:
                raise RuntimeError("Incomplete or duplicate bilateral output identities")
            for row in records:
                for name in ("top_confidence", "presence_probability", "top_class_probability"):
                    if not 0 <= row[name] <= 1:
                        raise RuntimeError(f"Nonfinite/invalid model probability: {name}")
            shared.atomic_write_json(args.output_dir / "records" / f"{label}.json", records)
            all_records.extend(records)
            all_masks.update({(label, index, side): mask for (index, side), mask in masks.items()})
            summary["models"][label] = {"kind": "cached_ve" if encoder.delta is None else "ve_semantic_delta",
                                         "prompt_texts": list(cached.NATURAL_PROMPTS),
                                         "internal_query_keys": list(shared.CLASS_NAMES),
                                         "checkpoint_sha256": checkpoint_hash,
                                         "cache_metadata": encoder.cache_metadata,
                                         "training_progress": None if encoder.delta is None else state["progress"]}
            summary["metrics"] = shared.summarize(all_records, .5)
            shared.atomic_write_json(args.output_dir / "progress.json", summary)
            cached.restore_original_ve_text_encoder(model, original_ve)
            encoder.to(device="cpu")
            torch.cuda.empty_cache()
        if any(parameter._version != version for _, parameter, version in original_versions):
            raise RuntimeError("Frozen original model parameters changed during evaluation")
        if core_source_hashes(args.project_root) != core:
            raise RuntimeError("Core source changed during evaluation")
        for path, digest in [(args.base_checkpoint, base_hash), (annotation, annotation_hash),
                             (tokenizer, tokenizer_hash), (args.delta_checkpoint, checkpoint_hash),
                             (Path(initial_cache_artifact["path"]), initial_cache_artifact["sha256"])]:
            if shared.sha256(path) != digest:
                raise RuntimeError(f"Source changed during evaluation: {path}")
        for row in rgb_sources + snapshots:
            path = Path(row.get("path", row.get("source")))
            if shared.sha256(path) != row["sha256"]:
                raise RuntimeError(f"RGB/script changed during evaluation: {path}")
        summary["visuals"] = shared.get_visual_renderer("separate")(
            root=args.data_root, output_dir=args.output_dir / "visuals", render_indices=render_indices,
            images=images, annotations_by_image=annotations_by_image, records=all_records,
            masks=all_masks, model_labels=list(summary["models"]))
        summary.update(status="completed", observed_identity_verified=True,
                       original_parameters_unchanged_by_version_counter=True)
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        summary["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        shared.atomic_write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

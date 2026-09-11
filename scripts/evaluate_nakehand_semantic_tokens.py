#!/usr/bin/env python3
"""Controlled nakehand validation: original natural VE cache and semantic deltas.

Only the frozen recording-level validation split is permitted. References are
SAM3-assisted masks, not independently annotated human ground truth. Two hand
queries are evaluated independently using the existing nakehand mathematics.
The caller owns GPU scheduling and an external per-run timeout. The user has
removed the former fixed 21:40 cutoff; an explicit cutoff remains optional.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch

if __package__:
    from . import cached_ve_text_features as cached
    from . import evaluate_bilateral_tokens as shared
    from . import evaluate_nakehand_tokens as bilateral
    from . import evaluate_ve_initialized_tokens as semantic
    from .run_token_lr_pilot import core_source_hashes, object_hash
else:
    import cached_ve_text_features as cached
    import evaluate_bilateral_tokens as shared
    import evaluate_nakehand_tokens as bilateral
    import evaluate_ve_initialized_tokens as semantic
    from run_token_lr_pilot import core_source_hashes, object_hash


FORMAT = "sam3-nakehand-semantic-delta-training-v1"
LABELS = ("ve-frozen-cache", "ve-delta-unconstrained", "ve-delta-anchored")
TRAIN_RECORDINGS = ("nakehandego/20260907_134035", "nakehandego/20260907_140713",
                    "nakehandexo/20260907_123926")
VAL_RECORDINGS = ("nakehandego/20260907_142020",)
REFERENCE_DESCRIPTION = "SAM3-assisted propagated reference masks; not full independent human pixel ground truth"


def read_hashed_json(path):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if shared.sha256(Path(path)) != digest:
        raise RuntimeError(f"JSON changed while reading: {path}")
    return data, digest


def verify_split(root, expected_split):
    """Cheap immutable-publication gate; actual RGB bytes are checked separately."""
    root = Path(root).resolve()
    if expected_split not in ("train", "val"):
        raise ValueError("Only train provenance and validation evaluation are authorized")
    ready, ready_hash = read_hashed_json(root / "READY.json")
    manifest, manifest_hash = read_hashed_json(root / "manifest.json")
    data, annotation_hash = read_hashed_json(root / "annotations.json")
    global_ready, global_ready_hash = read_hashed_json(root.parent / "READY.json")
    global_manifest, global_manifest_hash = read_hashed_json(root.parent / "manifest.json")
    plan, plan_hash = read_hashed_json(root.parent / "frozen-plan.json")
    if ready.get("status") != "complete" or global_ready.get("status") != "complete":
        raise ValueError("Dataset export is not READY/complete")
    if (global_ready.get("manifest_sha256") != global_manifest_hash
            or global_ready.get("frozen_plan_sha256") != plan_hash
            or global_manifest.get("sources_unchanged") is not True
            or manifest.get("sources_unchanged") is not True
            or manifest.get("status") != "complete"):
        raise ValueError("Dataset manifest/source verification is incomplete or changed")
    if (ready.get("annotations_sha256") != annotation_hash
            or ready.get("manifest_sha256") != manifest_hash
            or ready.get("frozen_plan_sha256") != plan_hash):
        raise ValueError("Split READY hashes differ from actual annotations/manifest/frozen plan")
    published = global_ready.get("splits", {}).get(root.name, {})
    for name, value in (("ready_sha256", ready_hash), ("annotations_sha256", annotation_hash),
                        ("manifest_sha256", manifest_hash), ("frozen_plan_sha256", plan_hash)):
        if published.get(name) != value:
            raise ValueError(f"Root READY does not bind split artifact: {name}")
    expected_role = "train" if expected_split == "train" else "validation"
    if data.get("info", {}).get("split") != expected_split or data["info"].get("dataset_role") != expected_role:
        raise ValueError("Wrong split/dataset role; development holdout is not a validation substitute")
    expected_count = 9092 if expected_split == "train" else 3449
    expected_recordings = TRAIN_RECORDINGS if expected_split == "train" else VAL_RECORDINGS
    allocation = plan.get("splits", {}).get(root.name, {})
    if (allocation.get("coco_split") != expected_split or allocation.get("images") != expected_count
            or sorted(allocation.get("recordings", [])) != sorted(expected_recordings)
            or data["info"].get("frozen_plan_sha256") != plan_hash):
        raise ValueError("Frozen plan allocation differs from the controlled protocol")
    recordings = sorted({row["recording_id"] for row in data["images"]})
    if len(data["images"]) != expected_count or recordings != sorted(expected_recordings):
        raise ValueError("Dataset count/recording membership differs from the frozen protocol")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    if len({int(row["id"]) for row in images}) != len(images):
        raise ValueError("Duplicate image IDs")
    seen_frames = defaultdict(set)
    for row in images:
        recording = row["recording_id"]
        declared = plan.get("recordings", {}).get(recording, {})
        frame = row.get("source_frame_index")
        if (type(frame) is not int or frame < 0 or frame >= declared.get("frame_count", 0)
                or row.get("frame_index") != frame
                or int(row["id"]) != declared.get("global_image_id_offset", -1) + frame
                or frame in seen_frames[recording]):
            raise ValueError("Source frame/global image identity differs from the frozen plan")
        seen_frames[recording].add(frame)
    if any(len(seen_frames[name]) != plan["recordings"][name]["frame_count"] for name in expected_recordings):
        raise ValueError("Frozen split does not cover every source frame exactly once")
    fingerprints = {str(path): digest for path, digest in (
        (root / "READY.json", ready_hash), (root / "manifest.json", manifest_hash),
        (root / "annotations.json", annotation_hash), (root.parent / "READY.json", global_ready_hash),
        (root.parent / "manifest.json", global_manifest_hash), (root.parent / "frozen-plan.json", plan_hash))}
    output_rows = manifest.get("image_outputs", [])
    published_rgb = {int(row["image_id"]): row["files"]["rgb"] for row in output_rows}
    if len(output_rows) != len(images) or set(published_rgb) != {int(row["id"]) for row in images}:
        raise ValueError("Manifest RGB identity coverage differs from COCO")
    for row in images:
        if published_rgb[int(row["id"])]["path"] != row["file_name"]:
            raise ValueError("Manifest RGB path differs from COCO")
    return data, {"root_ready_sha256": global_ready_hash, "split_ready_sha256": ready_hash,
                  "split_manifest_sha256": manifest_hash, "frozen_plan_sha256": plan_hash,
                  "annotations_sha256": annotation_hash, "recordings": recordings,
                  "dataset_role": expected_role, "files": fingerprints, "rgb_files": published_rgb}


class LazyReferences:
    """Decode only the current image's two masks, avoiding all-val full-size RAM."""
    def __init__(self, data, cache_size=2):
        if cache_size < 1:
            raise ValueError("Reference cache must hold at least one image")
        self.images = {int(row["id"]): row for row in data["images"]}
        self.annotations = defaultdict(list)
        self.cache, self.cache_size = OrderedDict(), cache_size
        if {int(row["id"]): row["name"] for row in data["categories"]} != shared.SIDE_BY_CATEGORY:
            raise ValueError("Expected fixed left_hand/right_hand category IDs")
        seen = set()
        for row in data["annotations"]:
            identifier, image_id = int(row["id"]), int(row["image_id"])
            if identifier in seen or image_id not in self.images or int(row["category_id"]) not in (1, 2):
                raise ValueError("Duplicate/unknown annotation or image/category reference")
            seen.add(identifier)
            self.annotations[image_id].append(row)
        for row in data["images"]:
            path = Path(row["file_name"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("RGB file_name must be a safe relative path")
            if type(row.get("primary_test")) is not bool or not row["primary_test"]:
                raise ValueError("All frozen split images require the legacy inclusion flag")
            if int(row["height"]) <= 0 or int(row["width"]) <= 0:
                raise ValueError("Invalid image size")
            bilateral.view_type(row)

    def __getitem__(self, image_id):
        image_id = int(image_id)
        if image_id not in self.cache:
            image = self.images[image_id]
            shape = int(image["height"]), int(image["width"])
            masks = {side: np.zeros(shape, dtype=bool) for side in shared.CLASS_NAMES}
            for annotation in self.annotations[image_id]:
                mask = shared.decode_gt_mask(annotation, *shape)
                if mask.shape != shape:
                    raise ValueError(f"Reference dimensions differ for image {image_id}")
                masks[shared.SIDE_BY_CATEGORY[int(annotation["category_id"])]] |= mask
            self.cache[image_id] = masks
        self.cache.move_to_end(image_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[image_id]


def validate_checkpoint(state, *, minimum_samples, base_hash, tokenizer_hash, expected_anchor=None):
    if state.get("format") != FORMAT or "class_tokens" in state:
        raise ValueError("Require dedicated nakehand semantic delta checkpoint, not DexYCB/random tokens")
    config = state.get("training_config", {})
    expected = {"batch_size": 1, "amp": True, "amp_dtype": "bfloat16", "seed": 123,
                "learning_rate": .001, "planned_samples": 2000, "float32_matmul_precision": "high"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint differs from the fixed controlled nakehand configuration")
    anchor = config.get("anchor_weight")
    if anchor not in (0., 1.) or (expected_anchor is not None and anchor != expected_anchor):
        raise ValueError("Anchor weight does not match the requested model label")
    # Reuse the tested semantic cache/progress validator without modifying any
    # frozen dependency. Only its dataset-specific format discriminator changes.
    semantic_state = dict(state, format=semantic.FORMAT)
    current, initial = semantic.validate_checkpoint(semantic_state, minimum_samples=minimum_samples,
                                                    base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    if current.resized_cache.dtype != torch.bfloat16 or current.raw_cache.dtype != torch.float32:
        raise ValueError("Require the verified BF16-resized / FP32-raw original VE cache")
    if state["progress"].get("full_epoch_completed") is not False:
        raise ValueError("The 2000-sample pilot must not claim a complete 9092-image epoch")
    for key in semantic.IMMUTABLE_CACHE_KEYS:
        if state["cache_state_dict"][key].dtype != state["initial_cache_state_dict"][key].dtype:
            raise ValueError("Frozen original cache dtype changed")
    return current, initial


def verify_training_identity(state):
    config = state["training_config"]
    data, provenance = verify_split(Path(config["data_root"]), "train")
    if (config.get("annotations_sha256") != provenance["annotations_sha256"]
            or state.get("annotation_summary", {}).get("sha256") != provenance["annotations_sha256"]
            or state["annotation_summary"].get("images") != 9092):
        raise ValueError("Training annotation identities do not match the checkpoint")
    recorded = config.get("data_provenance", {})
    for key in ("root_ready_sha256", "split_ready_sha256", "split_manifest_sha256",
                "frozen_plan_sha256", "recordings", "dataset_role"):
        if recorded.get(key) != provenance[key]:
            raise ValueError(f"Training split provenance changed: {key}")
    if state.get("data_provenance") != recorded:
        raise ValueError("Top-level training data provenance differs from config")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    order = list(range(len(images)))
    random.Random(123).shuffle(order)
    planned = order[:2000]
    observed = [int(images[index]["id"]) for index in planned[:state["next_step"]]]
    if (state["planned_dataset_indices"] != planned
            or state.get("planned_dataset_indices_sha256") != object_hash(planned)
            or state["observed_image_ids"] != observed):
        raise ValueError("Actual training prefix differs from frozen seed/COCO identities")
    return {"actual_prefix_verified": True, "observed_samples": len(observed),
            "observed_image_ids": observed, "planned_dataset_indices_sha256": object_hash(planned),
            "provenance": provenance}


def verify_comparable(states):
    """Catch mismatched same-label parallel workers before allowing comparison."""
    if not states:
        raise ValueError("No checkpoints provided")
    reference = states[0]
    keys = ("base_checkpoint_sha256", "tokenizer_sha256", "initial_cache_sha256",
            "annotations_sha256", "seed", "learning_rate", "batch_size", "amp", "amp_dtype",
            "float32_matmul_precision", "core_sources_sha256", "data_provenance", "loss_weights")
    for state in states[1:]:
        for key in keys:
            if state["training_config"].get(key) != reference["training_config"].get(key):
                raise ValueError(f"Incomparable training inputs/numerics: {key}")
        for key in ("planned_dataset_indices", "observed_image_ids", "progress"):
            if state[key] != reference[key]:
                raise ValueError(f"Incomparable actual training prefix: {key}")
        for key, value in reference["initial_cache_state_dict"].items():
            other = state["initial_cache_state_dict"].get(key)
            if isinstance(value, torch.Tensor):
                equal = isinstance(other, torch.Tensor) and value.dtype == other.dtype and torch.equal(value, other)
            else:
                equal = value == other
            if not equal:
                raise ValueError(f"Initial natural VE feature states differ: {key}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("baseline-checkpoint", "unconstrained-checkpoint", "constrained-checkpoint"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--variant", choices=("all", *LABELS), default="all")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--indices", help="Only explicit smoke diagnostics, never a formal validation subset")
    parser.add_argument("--minimum-samples-seen", type=int, choices=(20, 2000), default=2000)
    parser.add_argument("--render-per-recording", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    parser.add_argument("--deadline", help="Optional timezone-aware cutoff; caller still enforces finite run timeout")
    args = parser.parse_args(argv)
    args.deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    if args.deadline is not None and args.deadline.tzinfo is None:
        parser.error("An explicit deadline must include its timezone")
    if not 0 < args.gpu_memory_fraction <= .25 or args.render_per_recording < 0:
        parser.error("Invalid memory fraction/render count")
    if args.minimum_samples_seen == 20 and args.indices is None:
        parser.error("20-step checkpoint evaluation requires explicit diagnostic --indices")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir.exists():
        parser.error("Refusing to overwrite output directory")
    if args.output_dir == args.data_root or args.data_root in args.output_dir.parents:
        parser.error("Output must not be inside source split")
    args.labels = list(LABELS) if args.variant == "all" else [args.variant]
    supplied = {LABELS[0]: args.baseline_checkpoint or args.unconstrained_checkpoint or args.constrained_checkpoint,
                LABELS[1]: args.unconstrained_checkpoint, LABELS[2]: args.constrained_checkpoint}
    if any(supplied[label] is None for label in args.labels):
        parser.error("Supply the checkpoint corresponding to each requested variant")
    args.checkpoints = {label: supplied[label] for label in args.labels}
    return args


def summarize_validation(records):
    metrics = bilateral.summarize(records)
    for label, value in metrics.items():
        value["validation"] = value["primary_test"]
    return metrics


def main(argv=None):
    args = parse_args(argv)
    if args.deadline is not None and datetime.now(timezone.utc) >= args.deadline:
        raise RuntimeError("Authorized deadline passed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    data, provenance = verify_split(args.data_root, "val")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    references = LazyReferences(data)
    indices = bilateral.select_indices(images, args.indices)
    if args.minimum_samples_seen == 20 and len(indices) > 20:
        raise ValueError("20-step diagnostic evaluation is limited to at most 20 images")
    tokenizer = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    base_hash, tokenizer_hash = shared.sha256(args.base_checkpoint), shared.sha256(tokenizer)
    core = core_source_hashes(args.project_root)
    if not core:
        raise ValueError("No SAM3 core source fingerprint found")
    artifacts = dict(provenance["files"])
    artifacts.update({str(args.base_checkpoint): base_hash, str(tokenizer): tokenizer_hash})
    states, encoders, model_metadata = {}, {}, {}
    for label, checkpoint in args.checkpoints.items():
        digest = shared.sha256(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if shared.sha256(checkpoint) != digest:
            raise RuntimeError("Checkpoint changed while reading")
        anchor = None if label == LABELS[0] else float(label == LABELS[2])
        trained, initial = validate_checkpoint(state, minimum_samples=args.minimum_samples_seen,
                                               base_hash=base_hash, tokenizer_hash=tokenizer_hash,
                                               expected_anchor=anchor)
        training_identity = verify_training_identity(state)
        if (training_identity["provenance"]["root_ready_sha256"] != provenance["root_ready_sha256"]
                or training_identity["provenance"]["frozen_plan_sha256"] != provenance["frozen_plan_sha256"]):
            raise ValueError("Train and validation were not published under the same frozen split")
        if state["training_config"].get("core_sources_sha256") != object_hash(core):
            raise ValueError("SAM3 core changed since training")
        cache_artifact = semantic.verify_initial_cache_artifact(state)
        artifacts.update(training_identity["provenance"]["files"])
        artifacts[str(checkpoint)] = digest
        artifacts[cache_artifact["path"]] = cache_artifact["sha256"]
        states[label] = state
        encoders[label] = semantic.cache_from_state(state["initial_cache_state_dict"], frozen=True) if label == LABELS[0] else trained
        model_metadata[label] = {"kind": "frozen_natural_ve_cache" if label == LABELS[0] else "semantic_delta",
                                 "anchor_weight": anchor, "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                                 "initial_cache_artifact": cache_artifact, "cache_metadata": initial.cache_metadata,
                                 "training_progress": state["progress"], "training_identity": training_identity,
                                 "training_config": state["training_config"],
                                 "training_applied_to_this_variant": label != LABELS[0],
                                 "prompt_texts": list(cached.NATURAL_PROMPTS),
                                 "internal_query_keys": list(shared.CLASS_NAMES)}
    verify_comparable(list(states.values()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir / "code-snapshot"
    snapshot.mkdir()
    sources = {Path(__file__).resolve(), Path(inspect.getfile(shared)).resolve(),
               Path(inspect.getfile(bilateral)).resolve(), Path(inspect.getfile(cached)).resolve(),
               Path(inspect.getfile(semantic)).resolve(), args.project_root / "scripts/run_token_lr_pilot.py"}
    snapshots = []
    for source in sorted(sources):
        digest = shared.sha256(source)
        target = snapshot / source.name
        shutil.copy2(source, target)
        if shared.sha256(target) != digest:
            raise RuntimeError("Script changed during snapshot")
        artifacts[str(source)] = digest
        artifacts[str(target)] = digest
        snapshots.append({"source": str(source), "snapshot": str(target), "sha256": digest})
    rgb_identities = []
    for index in indices:
        image = images[index]
        path = args.data_root / image["file_name"]
        with Image.open(path) as rgb:
            if rgb.size != (int(image["width"]), int(image["height"])):
                raise ValueError("RGB dimensions differ from annotations")
        digest = shared.sha256(path)
        if digest != provenance["rgb_files"][int(image["id"])]["sha256"]:
            raise RuntimeError("RGB bytes differ from the published dataset manifest")
        artifacts[str(path)] = digest
        rgb_identities.append({"dataset_index": index, "image_id": int(image["id"]),
                               "file_name": image["file_name"], "sha256": digest})
    shared.atomic_write_json(args.output_dir / "image-identities.json", rgb_identities)
    (args.output_dir / "records").mkdir()
    summary = {"format": "sam3-nakehand-semantic-evaluation-v1", "status": "running",
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "deadline": args.deadline.isoformat() if args.deadline else None,
               "data_root": str(args.data_root), "dataset_role": "validation", "dataset_info": data["info"],
               "dataset_provenance": provenance, "annotations_sha256": provenance["annotations_sha256"],
               "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": base_hash,
               "tokenizer_sha256": tokenizer_hash, "core_source_sha256": object_hash(core), "core_sources": core,
               "initial_cache_sha256": next(iter(states.values()))["initial_cache_sha256"],
               "evaluated_images": len(indices), "evaluated_dataset_indices": indices,
               "evaluated_image_ids": [int(images[index]["id"]) for index in indices],
               "full_val_evaluated": len(indices) == 3449, "diagnostic_subset": args.indices is not None,
               "diagnostic_training_checkpoint": any(not state["progress"]["pilot_complete"] for state in states.values()),
               "reference_description": REFERENCE_DESCRIPTION, "training_performed": False,
               "thresholds_fitted_on_nakehand": False, "detection_threshold": .5, "mask_threshold": .5,
               "confidence_definition": "sigmoid(pred_logits) * sigmoid(presence_logit_dec)",
               "candidate_selection": "highest combined model confidence, never reference overlap",
               "metric_notes": {"detection": "Confidence >= .5 only; does not guarantee correct-side mask/IoU",
                                "miss_zero": "Present same-side reference: candidate Dice if detected, otherwise zero",
                                "absence": "Independent side union empty; other side may be present",
                                "swap_proxy": "Other-side IoU strictly greater than own and other intersection > 0; not anatomical proof",
                                "cohorts": "Legacy primary_test is only an inclusion flag; validation alias is this frozen val split",
                                "scope": "Previously explored recording-level validation, correlated video frames; not untouched test or full manual GT"},
               "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                           "device": torch.cuda.get_device_name(), "batch_size": 1, "amp": True,
                           "amp_dtype": "bfloat16", "float32_matmul_precision": "high",
                           "gpu_memory_fraction": args.gpu_memory_fraction},
               "models": {}, "metrics": {}, "code_snapshots": snapshots, "source_fingerprints": artifacts}
    shared.atomic_write_json(args.output_dir / "progress.json", summary)
    try:
        raw_dataset = shared.make_dataset(args.data_root)
        if len(raw_dataset) != len(images):
            raise RuntimeError("COCO/loader image count mismatch")
        dataset = semantic.IdentityCheckedDataset(raw_dataset, images)
        render_indices = bilateral.choose_render_indices(images, indices, args.render_per_recording)
        model = shared.load_ve_model(args.base_checkpoint)
        model.register_forward_hook(semantic.assert_finite_model_outputs)
        original_ve = model.backbone.language_backbone
        original_versions = [(parameter, parameter._version) for parameter in model.parameters()]
        records, masks = [], {}
        for label in args.labels:
            if args.deadline is not None and datetime.now(timezone.utc) >= args.deadline:
                raise RuntimeError("Deadline reached before model variant")
            encoder = encoders[label].to(device="cuda")
            cached.install_cached_ve_text_encoder(model, encoder)
            cached.set_cached_ve_training_mode(model, train_delta=False)
            bilateral.validate_frozen_noninteractive_model(model)
            versions = [(parameter, parameter._version) for parameter in encoder.parameters()]
            start = len(dataset.observed_indices)
            rows, predictions = bilateral.evaluate_variant(
                model=model, label=label, prompts=shared.CLASS_NAMES, dataset=dataset, images=images,
                references=references, indices=indices, render_indices=set(render_indices), batch_size=1, amp=True)
            if dataset.observed_indices[start:] != indices:
                raise RuntimeError("Actual loader identities/order differ from request")
            if any(parameter._version != version for parameter, version in versions):
                raise RuntimeError("Delta mutated during evaluation")
            for row in rows:
                row.update(split="val", dataset_role="validation", reference_description=REFERENCE_DESCRIPTION,
                           prompt_text=cached.NATURAL_PROMPTS[shared.CLASS_NAMES.index(row["prompt_key"])])
            shared.atomic_write_json(args.output_dir / "records" / f"{label}.json", rows)
            records.extend(rows)
            masks.update({(label, index, side): mask for (index, side), mask in predictions.items()})
            summary["models"][label] = model_metadata[label]
            summary["metrics"] = summarize_validation(records)
            shared.atomic_write_json(args.output_dir / "progress.json", summary)
            cached.restore_original_ve_text_encoder(model, original_ve)
            encoder.to(device="cpu")
            torch.cuda.empty_cache()
        if any(parameter._version != version for parameter, version in original_versions):
            raise RuntimeError("Original model parameters mutated during evaluation")
        summary["visuals"] = bilateral.render_results(
            data_root=args.data_root, output_dir=args.output_dir / "visuals", images=images,
            references=references, render_indices=render_indices, records=records, masks=masks, labels=args.labels)
        if core_source_hashes(args.project_root) != core:
            raise RuntimeError("SAM3 core source changed during evaluation")
        for path, digest in artifacts.items():
            if shared.sha256(Path(path)) != digest:
                raise RuntimeError(f"Input/script/RGB changed during evaluation: {path}")
        summary.update(status="completed", observed_identity_verified=True,
                       original_parameters_unchanged_by_version_counter=True,
                       all_sources_unchanged=True, actual_training_prefix_verified=True)
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

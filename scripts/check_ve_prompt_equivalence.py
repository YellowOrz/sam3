#!/usr/bin/env python3
"""Strict forward-only equivalence checks for VE, absent MANO and cached VE.

No optimizer, synthetic MANO, reference-derived prompts or threshold fitting.
GPU/deadline allocation is owned by the caller (use an external timeout).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shutil

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from scripts import evaluate_nakehand_tokens as evaluation
    from scripts import cached_ve_text_features as cached
    from scripts.run_token_lr_pilot import core_source_hashes, object_hash
except ModuleNotFoundError:
    import evaluate_nakehand_tokens as evaluation
    import cached_ve_text_features as cached
    from run_token_lr_pilot import core_source_hashes, object_hash


OUTPUT_KEYS = ("pred_logits", "presence_logit_dec", "pred_boxes", "pred_masks")
NATURAL_PROMPTS = ("left hand", "right hand")


def compare_outputs(expected, actual):
    """Strict dtype/shape/finite/value and native-resolution binary mask checks."""
    results = {}
    for name in OUTPUT_KEYS:
        left, right = expected[name], actual[name]
        same_shape = tuple(left.shape) == tuple(right.shape)
        same_dtype = left.dtype == right.dtype
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        equal = same_shape and same_dtype and finite and torch.equal(left, right)
        difference = float((left.float() - right.float()).abs().max()) if same_shape and finite else None
        results[name] = {"expected_shape": list(left.shape), "actual_shape": list(right.shape),
                         "expected_dtype": str(left.dtype), "actual_dtype": str(right.dtype),
                         "finite": finite, "torch_equal": equal, "max_abs_difference": difference}
        if name == "pred_masks":
            results[name]["binary_masks_equal"] = same_shape and finite and torch.equal(
                left.float().sigmoid() >= .5, right.float().sigmoid() >= .5)
    return {"passed": all(value["torch_equal"] and value.get("binary_masks_equal", True)
                           for value in results.values()), "outputs": results}


def parameter_versions(model):
    return [(name, parameter, int(parameter._version)) for name, parameter in model.named_parameters()]


def verify_parameter_versions(snapshot):
    changed = [name for name, parameter, version in snapshot if int(parameter._version) != version]
    if changed:
        raise RuntimeError(f"Parameters were modified during a forward-only check: {changed[:8]}")


def snapshot_scripts(output, paths):
    directory = output / "code-snapshot"
    directory.mkdir()
    records = []
    for source in sorted({Path(path).resolve() for path in paths}):
        target = directory / source.name
        shutil.copy2(source, target)
        digest = evaluation.shared.sha256(target)
        if evaluation.shared.sha256(source) != digest:
            raise RuntimeError(f"Script changed during snapshot: {source}")
        records.append({"source": str(source), "snapshot": str(target), "sha256": digest})
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--indices", default="0,217,435")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    args = parser.parse_args(argv)
    if not 0 < args.gpu_memory_fraction <= .25:
        parser.error("memory fraction must be within (0,.25]")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists():
        parser.error("output directory must not exist")
    if args.output_dir == args.data_root or args.data_root in args.output_dir.parents:
        parser.error("output must not be inside source data")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU tests cover helpers only")
    from sam3.model.mano_prompt_adapter import attach_mano_geometry_encoder
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    annotation_path = args.data_root / "annotations.json"
    annotation_hash = evaluation.shared.sha256(annotation_path)
    images, _, _ = evaluation.load_coco_index(args.data_root)
    if evaluation.shared.sha256(annotation_path) != annotation_hash:
        raise RuntimeError("Annotations changed during loading")
    indices = evaluation.select_indices(images, args.indices)
    base_hash = evaluation.shared.sha256(args.base_checkpoint)
    tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    tokenizer_hash = evaluation.shared.sha256(tokenizer_path)
    core = core_source_hashes(args.project_root)
    rgb_sources = [{"index": index, "image_id": images[index]["id"],
                    "path": str(args.data_root / images[index]["file_name"]),
                    "sha256": evaluation.shared.sha256(args.data_root / images[index]["file_name"])}
                   for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"format": "sam3-ve-prompt-equivalence-v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_performed": False, "mano_data_used": False,
              "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": base_hash,
              "data_root": str(args.data_root), "annotations_sha256": annotation_hash,
              "tokenizer_sha256": tokenizer_hash, "core_source_sha256": object_hash(core),
              "core_sources": core, "rgb_sources": rgb_sources,
              "indices": indices, "comparison": "strict torch.equal; no tolerance relaxation",
              "binary_mask_comparison": "all decoder masks at native output resolution, sigmoid >= .5",
              "amp": True, "amp_dtype": "bfloat16", "gpu_memory_fraction": args.gpu_memory_fraction,
              "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                          "gpu_name": torch.cuda.get_device_name(),
                          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "checks": []}
    result["code_snapshots"] = snapshot_scripts(args.output_dir, [
        __file__, inspect.getfile(evaluation), inspect.getfile(evaluation.shared),
        inspect.getfile(cached), args.project_root / "scripts/run_token_lr_pilot.py"])
    evaluation.shared.atomic_write_json(args.output_dir / "progress.json", result)
    try:
        dataset = evaluation.shared.make_dataset(args.data_root)
        if len(dataset) != len(images):
            raise RuntimeError("Loader/image count mismatch")
        model = evaluation.shared.load_ve_model(args.base_checkpoint)
        evaluation.validate_frozen_noninteractive_model(model)
        original_parameters = parameter_versions(model)
        original_ve = model.backbone.language_backbone
        original_geometry = model.geometry_encoder

        def forward(index, *, use_cache=False):
            batch = collate_fn_api([dataset[index]], dict_key="eval", with_seg_masks=True)["eval"]
            evaluation.validate_batch_identity(batch, [index], images)
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            if not use_cache:
                batch.find_text_batch = list(NATURAL_PROMPTS)
            evaluation.validate_frozen_noninteractive_model(model)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)[0]
            copied = {name: output[name].detach().cpu().clone() for name in OUTPUT_KEYS}
            del batch, output
            return copied

        baseline = {index: forward(index) for index in indices}
        for index, output in baseline.items():
            if not compare_outputs(output, output)["passed"]:
                raise RuntimeError(f"Baseline has nonfinite outputs: {index}")

        def compare_variant(label, *, use_cache=False):
            current_parameters = parameter_versions(model)
            for index in indices:
                comparison = compare_outputs(baseline[index], forward(index, use_cache=use_cache))
                result["checks"].append({"variant": label, "index": index,
                                         "image_id": images[index]["id"], **comparison})
                evaluation.shared.atomic_write_json(args.output_dir / "progress.json", result)
                print(json.dumps(result["checks"][-1]), flush=True)
            verify_parameter_versions(current_parameters)
            verify_parameter_versions(original_parameters)

        attach_mano_geometry_encoder(model)
        compare_variant("mano_wrapper_no_mano")
        model.geometry_encoder = original_geometry
        compare_variant("restored_original_ve")
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
        ):
            cache = cached.capture_ve_text_cache(
                original_ve, base_checkpoint_sha256=base_hash, tokenizer_sha256=tokenizer_hash,
                device="cuda", mode="frozen", metadata={"purpose": "forward_only_equivalence"})
        torch.save(cache.state_dict(), args.output_dir / "ve_frozen_cache.pt")
        cached.install_cached_ve_text_encoder(model, cache)
        compare_variant("frozen_ve_cache", use_cache=True)
        cached.restore_original_ve_text_encoder(model, original_ve)
        zero_delta = cached.CachedVETextEncoder(
            cache.padding_cache, cache.resized_cache, cache.raw_cache,
            metadata=cache.cache_metadata, mode="zero_delta")
        torch.save(zero_delta.state_dict(), args.output_dir / "ve_zero_delta_cache.pt")
        cached.install_cached_ve_text_encoder(model, zero_delta)
        compare_variant("zero_delta_ve_cache", use_cache=True)
        verify_parameter_versions(original_parameters)
        result["parameters_unchanged_by_version_counter"] = True
        result["parameter_version_check_scope"] = "original parameters and each variant; not a full in-memory tensor byte hash"
        result["cache_metadata"] = cache.cache_metadata
        result["cache_artifacts"] = [{"path": str(args.output_dir / name),
                                       "sha256": evaluation.shared.sha256(args.output_dir / name)}
                                      for name in ("ve_frozen_cache.pt", "ve_zero_delta_cache.pt")]
        if core_source_hashes(args.project_root) != core:
            raise RuntimeError("Core source changed during equivalence check")
        for row in result["code_snapshots"]:
            if evaluation.shared.sha256(Path(row["source"])) != row["sha256"]:
                raise RuntimeError(f"Script changed during equivalence check: {row['source']}")
        for path, digest in [(args.base_checkpoint, base_hash), (annotation_path, annotation_hash),
                             (tokenizer_path, tokenizer_hash)]:
            if evaluation.shared.sha256(path) != digest:
                raise RuntimeError(f"Source changed: {path}")
        for row in rgb_sources:
            if evaluation.shared.sha256(Path(row["path"])) != row["sha256"]:
                raise RuntimeError(f"RGB changed: {row['path']}")
        result["status"] = "passed" if all(row["passed"] for row in result["checks"]) else "failed_equivalence"
        if result["status"] != "passed":
            raise RuntimeError("Strict output equivalence failed; see saved per-output differences")
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        evaluation.shared.atomic_write_json(args.output_dir / "summary.json", result)


if __name__ == "__main__":
    main()

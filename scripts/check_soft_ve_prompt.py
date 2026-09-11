#!/usr/bin/env python3
"""Forward equivalence and train-split gradient probe for input-word residuals.

No optimizer exists in this check. Original VE, object vocabulary and all visual
weights stay frozen. Only a bounded real-train probe tests the autograd path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path

import torch

from scripts import soft_ve_prompt as soft
from scripts import train_nakehand_semantic_tokens as training
from scripts import check_ve_prompt_equivalence as equivalence


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25,
                        help="Explicit per-process allocator fraction; finite value in (0, .35].")
    args = parser.parse_args(argv)
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= .35:
        parser.error("gpu-memory-fraction must be finite and within (0, .35]")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if args.output_dir.exists() or args.output_dir.is_relative_to(args.data_root):
        parser.error("Use a new output directory outside the immutable training data")
    return args


def check_gradient(gradient, rows, label):
    if (not isinstance(gradient, torch.Tensor) or gradient.ndim != 2
            or gradient.shape[0] != rows or not bool(torch.isfinite(gradient).all())):
        raise RuntimeError(f"Invalid {label} gradient")
    norms = gradient.norm(dim=1)
    if not bool((norms > 0).all()):
        raise RuntimeError(f"Missing {label} gradient signal")
    return norms.detach().cpu().tolist()


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the real model probe")
    images, references, annotations, manifest, provenance = training.validate_ready_dataset(args.data_root)
    first = {}
    for index, image in enumerate(images):
        first.setdefault(tuple(sorted(references[image["id"]])), index)
    if not all(kind in first for kind in ((), (2,), (1, 2))):
        raise ValueError("Require fixed empty/right-only/both training examples")
    indices = [first[kind] for kind in ((), (2,), (1, 2))]
    outputs_by_id = {row["image_id"]: row for row in manifest["image_outputs"]}
    core = training.shared.core_source_hashes(args.project_root)
    sources = {str(path): training.evaluation.sha256(path) for path in (
        args.base_checkpoint, args.tokenizer_path, args.data_root / "annotations.json",
        args.data_root.parent / "READY.json")}
    for index in indices:
        training.verify_selected_rgb(args.data_root, images[index], outputs_by_id[images[index]["id"]])
        path = args.data_root / images[index]["file_name"]
        sources[str(path)] = training.evaluation.sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshots = equivalence.snapshot_scripts(args.output_dir, [__file__, inspect.getfile(soft),
        inspect.getfile(training), inspect.getfile(training.shared), inspect.getfile(training.evaluation),
        inspect.getfile(training.legacy), inspect.getfile(equivalence)])
    for row in snapshots:
        sources[row["source"]] = row["sha256"]
        sources[row["snapshot"]] = row["sha256"]
    result = {"format": "sam3-soft-input-ve-probe-v1", "status": "running",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "optimizer_steps": 0,
              "indices": indices, "image_ids": [images[index]["id"] for index in indices],
              "selection": "first empty/right-only/both-visible train images, before any model predictions",
              "data_provenance": provenance, "sources_sha256": sources,
              "core_sources": core, "code_snapshots": snapshots,
              "amp": "bfloat16", "float32_matmul_precision": "high", "gpu_memory_fraction": args.gpu_memory_fraction,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "checks": []}
    write = lambda: training.evaluation.atomic_write_json(args.output_dir / "summary.json", result)
    write()
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api
    try:
        dataset = training.evaluation.make_dataset(args.data_root)
        model = training.shared.build_model_with_matcher(args)
        model.requires_grad_(False)
        original = model.backbone.language_backbone
        original_versions = equivalence.parameter_versions(model)

        def batch_for(index, natural=False):
            sample = dataset[index]
            batch = collate_fn_api([sample], dict_key="train", with_seg_masks=True)["train"]
            training.validate_bilateral_batch(batch, images[index]["id"], references[images[index]["id"]])
            if natural:
                batch.find_text_batch = list(soft.PROMPTS)
            return copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

        def forward(index, natural):
            batch = batch_for(index, natural)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)[0]
            return {key: output[key].detach().cpu().clone() for key in equivalence.OUTPUT_KEYS}

        baseline = {index: forward(index, True) for index in indices}
        adapter = soft.install_shared_input_ve(model)
        if adapter.input_delta.numel() != 2048:
            raise RuntimeError("Production shared input residual must have 2048 parameters")
        for index in indices:
            comparison = equivalence.compare_outputs(baseline[index], forward(index, False))
            result["checks"].append({"index": index, **comparison})
            write()
            if not comparison["passed"]:
                raise RuntimeError("Zero input residual failed strict full-model equivalence")
        del baseline

        # Test untouched open vocabulary even after a deterministic residual
        # perturbation. This is not a trained state; restore exact zero afterward.
        texts = ["left hand", "cup", "right hand", "knife"]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            expected = original(texts, device="cuda")
        with torch.no_grad():
            adapter.input_delta.fill_(.01)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            current = adapter(texts, device="cuda")
        result["nonhand_features_unchanged"] = all(torch.equal(a[:, [1, 3]], b[:, [1, 3]])
            for a, b in zip(expected[1:], current[1:])) and torch.equal(expected[0], current[0])
        if not result["nonhand_features_unchanged"]:
            raise RuntimeError("Nonhand vocabulary changed after hand-only input residual")
        with torch.no_grad():
            adapter.input_delta.zero_()
        soft.set_input_ve_training_mode(model, train_residual=True)
        captured = {}
        def capture(module, inputs, output):
            output[1].retain_grad()
            captured["features"] = output[1]
        handle = adapter.register_forward_hook(capture)
        try:
            batch = batch_for(first[(1, 2)])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = training.compute_task_loss(model, batch, training.shared.build_loss_functions())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite real segmentation task loss")
            loss.backward()
            role_norms = check_gradient(adapter.input_delta.grad, 2, "word-role parameter")
            feature_grad = captured["features"].grad.float().permute(1, 0, 2).flatten(1)
            side_norms = check_gradient(feature_grad, 2, "left/right output-feature")
            result["gradient_probe"] = {"train_image_id": images[first[(1, 2)]]["id"],
                "task_loss": float(loss.detach().cpu()), "word_role_parameter_norms": role_norms,
                "left_right_feature_norms": side_norms,
                "role_note": "parameter rows denote side-word/hand-word, not left/right classes"}
        finally:
            handle.remove()
        if not bool((adapter.input_delta == 0).all()):
            raise RuntimeError("Gradient-only probe must not update residual parameters")
        equivalence.verify_parameter_versions(original_versions)
        if any(parameter.grad is not None for parameter in adapter.original_ve.parameters()):
            raise RuntimeError("Original text weights received gradients")
        if training.shared.core_source_hashes(args.project_root) != core:
            raise RuntimeError("Core changed during probe")
        for path, expected_sha in sources.items():
            if training.evaluation.sha256(Path(path)) != expected_sha:
                raise RuntimeError(f"Source changed during probe: {path}")
        training.shared.atomic_save(args.output_dir / "zero_input_residual.pt", adapter.residual_state())
        result.update(status="passed", source_hashes_unchanged=True, original_parameters_unchanged=True,
                      trainable_parameters=2048, accuracy_evaluated=False)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        write()
    print(json.dumps({"status": result["status"], "output": str(args.output_dir), "gradient_probe": result["gradient_probe"]}))


if __name__ == "__main__":
    main()

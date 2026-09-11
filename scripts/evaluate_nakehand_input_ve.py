#!/usr/bin/env python3
"""Frozen validation of original VE, output delta and shared input VE residuals.

Reuse the spatial evaluator's unchanged score-selected masks and metric schema.
Reference-empty outputs are disagreements with SAM3-assisted annotations, not
automatically true false detections: reference completeness is not established.
No reference-derived candidate selection, threshold fitting or training occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import math
from pathlib import Path
import shutil

from PIL import Image
import torch
from torch import nn

if __package__:
    from . import train_nakehand_input_ve as training
    from . import evaluate_hand_boundary_diagnostics as spatial
else:
    import train_nakehand_input_ve as training
    import evaluate_hand_boundary_diagnostics as spatial

previous, shared, bilateral, semantic, cached = spatial.previous, spatial.shared, spatial.bilateral, spatial.semantic, spatial.cached
soft = training.soft
FORMAT = "sam3-input-ve-boundary-validation-v1"
LABELS = ("baseline", "output-delta", "input-ve")
REFERENCE_ABSENCE_WARNING = (
    "Absent-side FP is relative to the provided reference (reference-empty output rate), "
    "not verified true false detection. The user confirmed validation image 4713/frame 0 contains a RIGHT hand "
    "while both reference masks are empty; the inspected output-delta LEFT prompt detected that right hand. "
    "Missing reference and model side confusion coexist: neither all true false detections nor all good "
    "predictions unfairly penalized. Do not tune thresholds or relabel the frozen data from this one case."
)


class NaturalVEAliases(nn.Module):
    """Only map internal aliases to natural strings; all encoding is original VE."""
    def __init__(self, original_ve):
        super().__init__()
        self.original_ve = original_ve
        self.requires_grad_(False)
        self.eval()

    def forward(self, text, input_boxes=None, device=None):
        if isinstance(text, (list, tuple)) and text and isinstance(text[0], str):
            text = [soft.ALIASES.get(value, value) for value in text]
        return self.original_ve(text, input_boxes=input_boxes, device=device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "input-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-delta-checkpoint", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--variant", choices=("all", *LABELS), default="all")
    parser.add_argument("--indices", help="Explicit smoke diagnostics only; omit for full validation")
    parser.add_argument("--minimum-samples-seen", type=int, choices=(20, 2000), default=2000)
    parser.add_argument("--render-count", type=int, default=8)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.35)
    parser.add_argument("--deadline", help="Optional timezone-aware cutoff; external finite timeout is still recommended")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= .35
            or not 0 <= args.render_count <= 12):
        parser.error("Require finite GPU fraction in (0,.35] and render-count in 0..12")
    if args.minimum_samples_seen < 2000 and args.indices is None:
        parser.error("Partial training checkpoints require explicit --indices diagnostic mode")
    try:
        args.deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    except ValueError:
        parser.error("Invalid ISO deadline")
    if args.deadline is not None and args.deadline.tzinfo is None:
        parser.error("Deadline must contain a timezone")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.tokenizer_path is None:
        args.tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if args.output_dir.exists() or args.output_dir.is_relative_to(args.data_root):
        parser.error("Output must be a new directory outside the immutable split")
    if args.variant == "output-delta" and args.output_delta_checkpoint is None:
        parser.error("The output-delta variant needs --output-delta-checkpoint")
    args.labels = ([name for name in LABELS if name != "output-delta" or args.output_delta_checkpoint is not None]
                   if args.variant == "all" else [args.variant])
    return args


def compare_training_inputs(input_state, output_state=None):
    """Reject identity differences; explicitly expose LR/budget differences."""
    if output_state is None:
        return {"reference_label": "input-ve", "matched_actual_training_budget": True,
                "matched_training_numerics": None, "training_comparison_available": False,
                "interpretation": "Original VE versus one trained adapter; no output-delta checkpoint supplied"}
    config0, config1 = input_state["training_config"], output_state["training_config"]
    identity_keys = ("base_checkpoint_sha256", "tokenizer_sha256", "initial_cache_sha256",
                     "annotations_sha256", "data_provenance", "core_sources_sha256")
    for key in identity_keys:
        if config0.get(key) != config1.get(key):
            raise ValueError(f"Different original model/data provenance: {key}")
    if training.shared.cache_fingerprint(input_state["initial_cache_state_dict"]) != training.shared.cache_fingerprint(output_state["initial_cache_state_dict"]):
        raise ValueError("Natural VE baseline features differ")
    if input_state["planned_dataset_indices"] != output_state["planned_dataset_indices"]:
        raise ValueError("Different planned training order")
    numeric_keys = ("learning_rate", "seed", "batch_size", "amp", "amp_dtype", "optimizer", "weight_decay",
                    "float32_matmul_precision", "loss_weights", "anchor_weight")
    differences = [name for name in numeric_keys if config0.get(name) != config1.get(name)]
    matched = (input_state["progress"] == output_state["progress"]
               and input_state["observed_image_ids"] == output_state["observed_image_ids"])
    return {"reference_label": "output-delta", "matched_actual_training_budget": matched,
            "matched_training_numerics": not differences, "numeric_configuration_differences": differences,
            "training_comparison_available": True,
            "architecture_difference": "input shared word-role residual through frozen original Transformer vs per-side output feature residual",
            "interpretation": "Same parameter count is not equal functional capacity; equal LR is not equal feature displacement. Unequal LR or completed prefixes are descriptive comparisons, not a one-factor architecture ablation."}


def read_checkpoint(path, *, input_format, minimum_samples, base_hash, tokenizer_hash):
    digest = shared.sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if shared.sha256(path) != digest:
        raise RuntimeError("Checkpoint changed while loading")
    if input_format:
        metadata = training.validate_checkpoint_schema(state, minimum_samples=minimum_samples,
            base_hash=base_hash, tokenizer_hash=tokenizer_hash)
        implementations = {source.name: shared.sha256(source) for source in training.implementation_sources()}
        if state["training_config"].get("implementation_sha256") != implementations:
            raise ValueError("Input VE training implementation differs from checkpoint source fingerprints")
    else:
        previous.validate_checkpoint(state, minimum_samples=minimum_samples, base_hash=base_hash,
                                     tokenizer_hash=tokenizer_hash, expected_anchor=0.)
        metadata = {"format": state["format"], "variant": "output-delta", "anchor_weight": 0.}
    identity = training.verify_training_identity(state)
    cache_artifact = semantic.verify_initial_cache_artifact(state)
    return state, {**metadata, "checkpoint": str(path), "checkpoint_sha256": digest,
                   "training_config": state["training_config"], "training_progress": state["progress"],
                   "training_identity": identity, "initial_cache_artifact": cache_artifact,
                   "training_applied_to_this_variant": True}


def load_original_model(args):
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint), bpe_path=str(args.tokenizer_path),
        load_from_HF=False, device="cuda", eval_mode=True, enable_segmentation=True,
        enable_inst_interactivity=False, text_encoder_type="ve")
    model.requires_grad_(False)
    model.eval()
    bilateral.validate_frozen_noninteractive_model(model)
    return model


def render_report(summary):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    lines = ["# 输入词向量残差：原 VE / 输出增量 / 输入增量对照", "",
             f"实际评估 {summary['evaluated_images']} 张；完整 val：{summary['full_val_evaluated']}。"
             "参考为 SAM3 辅助标注，不是全量独立人工轮廓真值。", "",
             "重要：空参考下输出率（schema 保留 absent-side FP 字段）只是相对参考的误报。"
             "用户已确认 image4713/frame0 为右手，但两侧参考均空，且被检查的输出增量模型以 left_hand 检出该右手。"
             "这说明参考漏标与模型错侧同时存在；不能直接当作真实误检率，也不能把所有此类输出当作好预测被冤枉。", "",
             "baseline 实际调用原始自然文本 VE，仅把内部 left_hand/right_hand 名称映射成 left hand/right hand。"
             "input-ve 只训练 side-word/hand-word 两行共享输入残差（2,048 参数），保留原始文本 Transformer；"
             "output-delta 在原 VE 输出特征上训练左右独立残差。其余网络均冻结。", "",
             "| 模型 | 训练样本 | LR | 阶段 | Dice | 漏掉参考比例 | Boundary IoU 4px | 8px | 16px |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for label, metrics in summary["spatial_metrics"].items():
        model = summary["models"][label]
        count = 0 if label == "baseline" else model["training_progress"]["samples_seen"]
        lr = "—" if label == "baseline" else str(model["training_config"]["learning_rate"])
        for stage in ("candidate", "detected"):
            values = metrics[stage]["overall"]
            numbers = [values["present_macro"]["own_dice"], values["present_macro"]["own_missed_fraction"],
                       *(values["boundary_present_macro"][key] for key in spatial.RATIO_KEYS)]
            lines.append(f"| {label} | {count} | {lr} | {stage} | " + " | ".join(map(fmt, numbers)) + " |")
    lines += ["", "candidate 由 class×presence 最大分数选取，从不按参考重叠挑选；detected 在分数<0.5时置空。"
              "精确阈值、左右相交、参考外像素和空参考输出指标保留在 summary.json/records。"
              "边界在原生640×480尺度对应4/8/16px，其他分辨率按对角线比例调整。", "",
              "参考外像素不是前臂真值，另一侧重叠优势也不是经人工确认的解剖学错手。"
              "既往检查过的录像只称 validation，不冒称未见独立 test。", "",
              f"同实际训练预算：{summary['comparison']['matched_actual_training_budget']}；"
              f"数值配置完全相同：{summary['comparison']['matched_training_numerics']}。"
              "LR或实际样本进度不同时，不作为单因素架构消融结论。", ""]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for model evaluation; CPU tests use helper contracts only")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    data, provenance = previous.verify_split(args.data_root, "val")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    references = previous.LazyReferences(data, cache_size=2)
    indices = bilateral.select_indices(images, args.indices)
    render_indices = shared.evenly_spaced(indices, args.render_count) if args.render_count else []
    base_hash, tokenizer_hash = shared.sha256(args.base_checkpoint), shared.sha256(args.tokenizer_path)
    core = previous.core_source_hashes(args.project_root)
    fingerprints = {**provenance["files"], str(args.base_checkpoint): base_hash, str(args.tokenizer_path): tokenizer_hash}
    states, metadata = {}, {}
    specs = {"input-ve": args.input_checkpoint}
    if args.output_delta_checkpoint:
        specs["output-delta"] = args.output_delta_checkpoint
    for label, path in specs.items():
        state, meta = read_checkpoint(path, input_format=label == "input-ve", minimum_samples=args.minimum_samples_seen,
                                      base_hash=base_hash, tokenizer_hash=tokenizer_hash)
        identity = meta["training_identity"]["provenance"]
        if (identity["root_ready_sha256"] != provenance["root_ready_sha256"]
                or identity["frozen_plan_sha256"] != provenance["frozen_plan_sha256"]
                or state["training_config"].get("core_sources_sha256") != previous.object_hash(core)):
            raise ValueError("Training checkpoint and evaluation model/data publication differ")
        fingerprints.update(identity["files"])
        fingerprints[str(path)] = meta["checkpoint_sha256"]
        fingerprints[meta["initial_cache_artifact"]["path"]] = meta["initial_cache_artifact"]["sha256"]
        states[label], metadata[label] = state, meta
    expected_zero = training.initial_residual_state(args.tokenizer_path)
    if training.shared.cache_fingerprint(expected_zero) != training.shared.cache_fingerprint(states["input-ve"]["initial_input_residual_state"]):
        raise ValueError("Checkpoint input tokenizer state differs from the current verified tokenizer")
    comparison = compare_training_inputs(states["input-ve"], states.get("output-delta"))
    metadata["baseline"] = {"variant": "baseline", "training_applied_to_this_variant": False,
        "base_checkpoint_sha256": base_hash, "tokenizer_sha256": tokenizer_hash,
        "encoder": "actual frozen original VE with natural-prompt alias mapping only", "training_samples_applied": 0}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "records").mkdir()
    (args.output_dir / "code-snapshot").mkdir()
    sources = set(training.implementation_sources()) | {Path(__file__).resolve(),
        *(Path(inspect.getfile(module)).resolve() for module in (spatial, previous, bilateral, semantic, spatial.errors)),
        args.project_root / "scripts/run_token_lr_pilot.py"}
    snapshots = []
    for source in sorted(sources):
        digest, target = shared.sha256(source), args.output_dir / "code-snapshot" / source.name
        shutil.copy2(source, target)
        if shared.sha256(source) != digest or shared.sha256(target) != digest:
            raise RuntimeError("Evaluation code changed during snapshot")
        fingerprints[str(source)] = fingerprints[str(target)] = digest
        snapshots.append({"source": str(source), "snapshot": str(target), "sha256": digest})
    for index in indices:
        image, path = images[index], args.data_root / images[index]["file_name"]
        with Image.open(path) as rgb:
            if rgb.size != (int(image["width"]), int(image["height"])):
                raise ValueError("RGB dimensions differ from annotations")
        digest = shared.sha256(path)
        if digest != provenance["rgb_files"][int(image["id"])]["sha256"]:
            raise ValueError("RGB bytes differ from the published manifest")
        fingerprints[str(path)] = digest
    summary = {"format": FORMAT, "metric_schema": "sam3-hand-boundary-validation-v1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False,
        "data_root": str(args.data_root), "dataset_role": "validation", "dataset_provenance": provenance,
        "annotations_sha256": provenance["annotations_sha256"], "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": base_hash, "tokenizer_sha256": tokenizer_hash,
        "core_sources": core, "core_source_sha256": previous.object_hash(core),
        "evaluated_images": 0, "planned_images": len(indices), "evaluated_dataset_indices": indices,
        "evaluated_image_ids": [int(images[index]["id"]) for index in indices], "full_val_evaluated": False,
        "planned_full_val": len(indices) == len(images) == 3449, "diagnostic_subset": args.indices is not None,
        "diagnostic_training_checkpoint": any(state["next_step"] < 2000 for state in states.values()),
        "render_indices": render_indices, "render_selection": "At most 12 evenly spaced indices fixed before any model outputs",
        "boundary_ratios": list(spatial.RATIOS), "boundary_pixels_at_640x480": [4, 8, 16],
        "metric_definitions": spatial.errors.metric_definitions(), "reference_description": previous.REFERENCE_DESCRIPTION,
        "reference_absence_warning": REFERENCE_ABSENCE_WARNING,
        "score_and_selection": "sigmoid(class)*sigmoid(presence), own score argmax, detection>=.5; never reference overlap",
        "detection_threshold": .5, "mask_threshold": .5, "thresholds_fitted": False,
        "mano_used": False, "geometry_or_reference_prompts_used": False,
        "preprocessing": "Unchanged 1008-square RGB; BF16 inference; FP32 mask logits bilinear to original H/W then sigmoid>=.5",
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "gpu_memory_fraction": args.gpu_memory_fraction,
                    "amp": True, "amp_dtype": "bfloat16", "batch_size": 1, "float32_matmul_precision": "high"},
        "comparison": comparison, "models": {name: metadata[name] for name in args.labels},
        "metrics": {}, "spatial_metrics": {}, "completed_images_per_model": {},
        "code_snapshots": snapshots, "input_sha256": fingerprints}
    shared.atomic_write_json(args.output_dir / "progress.json", summary)
    try:
        dataset = semantic.IdentityCheckedDataset(shared.make_dataset(args.data_root), images)
        if len(dataset) != len(images):
            raise RuntimeError("Loader count differs from frozen validation")
        model = load_original_model(args)
        model.register_forward_hook(semantic.assert_finite_model_outputs)
        original_ve = model.backbone.language_backbone
        base_versions = [(parameter, parameter._version) for parameter in model.parameters()]
        rows_all, masks_all = [], {}
        for label in args.labels:
            if args.deadline is not None and datetime.now(timezone.utc) >= args.deadline:
                raise RuntimeError("Explicit evaluation deadline reached")
            model.backbone.language_backbone = original_ve
            if label == "baseline":
                encoder = NaturalVEAliases(original_ve)
                model.backbone.language_backbone = encoder
            elif label == "input-ve":
                encoder = soft.install_shared_input_ve(model)
                encoder.load_residual_state(states[label]["input_residual_state"])
                soft.set_input_ve_training_mode(model, train_residual=False)
            else:
                encoder = semantic.cache_from_state(states[label]["cache_state_dict"]).to(device="cuda")
                cached.install_cached_ve_text_encoder(model, encoder)
                cached.set_cached_ve_training_mode(model, train_delta=False)
            model.eval()
            model.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("Evaluation must not enable parameter gradients")
            versions = [(parameter, parameter._version) for parameter in encoder.parameters()]
            start = len(dataset.observed_indices)

            def progress(completed):
                summary["completed_images_per_model"][label] = completed
                shared.atomic_write_json(args.output_dir / "progress.json", summary)
                print(f"model={label} completed_images={completed}/{len(indices)}", flush=True)
                if args.deadline is not None and datetime.now(timezone.utc) >= args.deadline:
                    raise RuntimeError("Explicit evaluation deadline reached")

            with (args.output_dir / "records" / f"{label}.partial.jsonl").open("x") as stream:
                rows, masks = spatial.evaluate_model(model, dataset, images, references, indices, label,
                    stream, set(render_indices), progress)
            if dataset.observed_indices[start:] != indices:
                raise RuntimeError("Actual loader identities differ from the fixed validation order")
            if any(parameter._version != version for parameter, version in versions):
                raise RuntimeError("Encoder parameters changed during evaluation")
            for row in rows:
                row["reference_absence_interpretation"] = "reference-relative output rate; not independently adjudicated true false detection"
            path = args.output_dir / "records" / f"{label}.json"
            shared.atomic_write_json(path, rows)
            summary.setdefault("record_files", {})[label] = {"path": str(path), "sha256": shared.sha256(path)}
            rows_all.extend(rows)
            masks_all.update(masks)
            summary["metrics"].update(previous.summarize_validation(rows))
            summary["spatial_metrics"][label] = spatial.grouped_spatial_summary(rows)
            shared.atomic_write_json(args.output_dir / "progress.json", summary)
            # Never move an input wrapper's original VE to CPU accidentally.
            model.backbone.language_backbone = original_ve
            if label == "output-delta":
                encoder.to(device="cpu")
            del encoder
            torch.cuda.empty_cache()
        if any(parameter._version != version for parameter, version in base_versions):
            raise RuntimeError("Original base parameters changed during evaluation")
        summary["visuals"] = bilateral.render_results(data_root=args.data_root, output_dir=args.output_dir / "visuals",
            images=images, references=references, render_indices=render_indices, records=rows_all, masks=masks_all, labels=args.labels)
        if previous.core_source_hashes(args.project_root) != core:
            raise RuntimeError("Core sources changed during evaluation")
        for path, digest in fingerprints.items():
            if shared.sha256(Path(path)) != digest:
                raise RuntimeError(f"Input/implementation changed during evaluation: {path}")
        summary.update(status="completed", all_sources_unchanged=True, observed_identity_verified=True,
                       parameters_unchanged_by_version_counter=True, evaluated_images=len(indices),
                       full_val_evaluated=len(indices) == len(images) == 3449)
        with (args.output_dir / "REPORT.md").open("x") as stream:
            stream.write(render_report(summary))
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
        summary["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved()/2**20
        shared.atomic_write_json(args.output_dir / "summary.json", summary)
        shared.atomic_write_json(args.output_dir / "progress.json", summary)


if __name__ == "__main__":
    main()

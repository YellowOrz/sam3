#!/usr/bin/env python3
"""Full frozen nakehand validation with side/area and multiscale boundary errors.

Compute all metrics while each score-selected mask exists in memory. Save at
most twelve predetermined image groups. No reconstruction of full masks from
legacy scalar records, no GT candidate ranking, and no training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_cdt
import torch
import torch.nn.functional as F

try:
    from scripts import evaluate_nakehand_semantic_tokens as previous
    from scripts import analyze_hand_segmentation_failures as errors
except ModuleNotFoundError:
    import evaluate_nakehand_semantic_tokens as previous
    import analyze_hand_segmentation_failures as errors

shared, bilateral, semantic, cached = previous.shared, previous.bilateral, previous.semantic, previous.cached
RATIOS = (.005, .01, .02)
RATIO_KEYS = ("r0.005", "r0.010", "r0.020")
LABELS = ("baseline", "CP0", "CP1")
BOUNDARY_TRIAL_FORMAT = "sam3-nakehand-prompt-ablation-training-v1"


def boundary_trial_module():
    # Lazy: legacy pilot evaluation does not depend on the optional new trainer.
    if __package__:
        from . import train_nakehand_prompt_ablation as module
    else:
        import train_nakehand_prompt_ablation as module
    if module.FORMAT != BOUNDARY_TRIAL_FORMAT:
        raise ValueError("Boundary trial module exposes an unexpected format")
    return module


def boundary_bands(mask):
    """Three exact square-erosion inner bands from one chessboard transform.

    A surrounding zero ring includes boundaries truncated by the image edge.
    The distance <= d equals mask minus d iterations of 3x3 binary erosion.
    """
    binary = errors.boolean_mask(mask)
    widths = {key: errors.boundary_width(binary.shape, ratio) for key, ratio in zip(RATIO_KEYS, RATIOS)}
    if not binary.any():
        return {key: np.zeros_like(binary) for key in RATIO_KEYS}, widths
    distances = distance_transform_cdt(np.pad(binary, 1, constant_values=False), metric="chessboard")[1:-1, 1:-1]
    return {key: binary & (distances <= width) for key, width in widths.items()}, widths


def area_metrics(prediction, own, other):
    prediction, own, other = map(errors.boolean_mask, (prediction, own, other))
    if prediction.shape != own.shape or own.shape != other.shape:
        raise ValueError("Mask shapes must be identical at original image resolution")
    p, g, o = (int(mask.sum()) for mask in (prediction, own, other))
    tp = int((prediction & own).sum())
    opposite_only = int((prediction & other & ~own).sum())
    outside = int((prediction & ~(own | other)).sum())
    other_intersection = int((prediction & other).sum())
    own_iou, other_iou = errors.ratio(tp, p + g - tp), errors.ratio(other_intersection, p + o - other_intersection)
    if tp + opposite_only + outside != p:
        raise RuntimeError("Prediction partition is not exhaustive and disjoint")
    return {
        "prediction_pixels": p, "own_reference_pixels": g, "other_reference_pixels": o,
        "reference_overlap_pixels": int((own & other).sum()),
        "own_intersection_pixels": tp, "own_missed_pixels": g - tp,
        "other_only_intersection_pixels": opposite_only, "other_intersection_pixels": other_intersection,
        "outside_both_references_pixels": outside,
        "own_dice": errors.ratio(2 * tp, p + g), "own_iou": own_iou,
        "own_precision": errors.ratio(tp, p), "own_recall": errors.ratio(tp, g),
        "own_missed_fraction": errors.ratio(g - tp, g),
        "other_only_prediction_fraction": errors.ratio(opposite_only, p),
        "outside_both_prediction_fraction": errors.ratio(outside, p),
        "opposite_overlap_dominant_proxy": bool(other_intersection and own_iou is not None and
                                                other_iou is not None and other_iou > own_iou),
    }


def multiscale_metrics(prediction, own, other, *, own_bands=None, prediction_bands=None):
    value = area_metrics(prediction, own, other)
    own_bands = boundary_bands(own)[0] if own_bands is None else own_bands
    prediction_bands, widths = boundary_bands(prediction) if prediction_bands is None else prediction_bands
    value["boundary"] = {}
    for key in RATIO_KEYS:
        intersection = int((prediction_bands[key] & own_bands[key]).sum())
        union = int((prediction_bands[key] | own_bands[key]).sum())
        value["boundary"][key] = {"band_pixels": widths[key], "intersection_pixels": intersection,
                                   "union_pixels": union, "iou": errors.ratio(intersection, union)}
    return value


def candidate_and_detected_metrics(prediction, own, other, *, detected, own_bands=None):
    """Do not repeat distance transforms for the threshold-kept candidate."""
    bands = boundary_bands(prediction)
    own_bands = boundary_bands(own)[0] if own_bands is None else own_bands
    candidate = multiscale_metrics(prediction, own, other, own_bands=own_bands, prediction_bands=bands)
    if detected:
        return {"candidate": candidate, "detected": candidate}
    empty = np.zeros_like(prediction, dtype=bool)
    empty_bands = ({key: empty for key in RATIO_KEYS}, bands[1])
    return {"candidate": candidate, "detected": multiscale_metrics(
        empty, own, other, own_bands=own_bands, prediction_bands=empty_bands)}


def select_predictions(output, shape):
    """No reference input: each side is selected only by its own combined score."""
    logits, presence_logits, masks = (output[key] for key in ("pred_logits", "presence_logit_dec", "pred_masks"))
    if logits.ndim != 3 or logits.shape[0] != 2 or logits.shape[-1] != 1 or logits.shape[1] < 1:
        raise ValueError("Expected exactly two side rows with decoder class logits")
    if presence_logits.shape[0] != 2 or presence_logits.numel() != 2:
        raise ValueError("Expected one presence value for each side")
    if masks.ndim != 4 or masks.shape[:2] != logits.shape[:2]:
        raise ValueError("Mask rows/decoder indices differ from class logits")
    if not all(bool(torch.isfinite(value).all()) for value in (logits, presence_logits, masks)):
        raise ValueError("Nonfinite inference output")
    classes = logits.float().sigmoid().squeeze(-1)
    presence = presence_logits.float().sigmoid().reshape(2)
    scores = classes * presence[:, None]
    result = []
    for row in range(2):
        top = int(scores[row].argmax())
        resized = F.interpolate(masks[row, top][None, None].float(), size=shape,
                                mode="bilinear", align_corners=False)[0, 0]
        result.append({"prediction": resized.sigmoid().cpu().numpy() >= .5,
                       "top_class_probability": float(classes[row, top]),
                       "presence_probability": float(presence[row]), "top_confidence": float(scores[row, top]),
                       "selected_decoder_query": top, "detections_above_threshold": int((scores[row] >= .5).sum())})
    return result


def validate_checkpoint_dispatch(state, *, minimum_samples, base_hash, tokenizer_hash):
    """Explicit format gate. Future trial formats require their own validator."""
    if state.get("format") == previous.FORMAT:
        return previous.validate_checkpoint(state, minimum_samples=minimum_samples, base_hash=base_hash,
                                             tokenizer_hash=tokenizer_hash, expected_anchor=None)
    if state.get("format") == BOUNDARY_TRIAL_FORMAT:
        boundary_trial_module().validate_checkpoint_schema(state, minimum_samples=minimum_samples,
            base_hash=base_hash, tokenizer_hash=tokenizer_hash)
        return (semantic.cache_from_state(state["cache_state_dict"]),
                semantic.cache_from_state(state["initial_cache_state_dict"]))
    raise ValueError(f"Unsupported checkpoint format {state.get('format')!r}; do not relabel a new trial or DexYCB checkpoint as the old 2000 pilot")


def compare_checkpoint_inputs(states):
    """Allow different recovery steps but explicitly report unequal budgets/configs."""
    labels = list(states)
    if not labels:
        raise ValueError("No checkpoint states")
    reference = states[labels[0]]
    differences = {}
    for label in labels[1:]:
        state = states[label]
        for key, value in reference["initial_cache_state_dict"].items():
            other = state["initial_cache_state_dict"].get(key)
            equal = (isinstance(other, torch.Tensor) and value.dtype == other.dtype and torch.equal(value, other)) if isinstance(value, torch.Tensor) else value == other
            if not equal:
                raise ValueError("Models do not share exactly the same initial VE cache")
        config0, config1 = reference["training_config"], state["training_config"]
        for key in ("base_checkpoint_sha256", "tokenizer_sha256", "annotations_sha256", "data_provenance"):
            if config0.get(key) != config1.get(key):
                raise ValueError(f"Models do not share source/data identity: {key}")
        ignored = {"anchor_weight", "initial_cache", "base_checkpoint", "tokenizer_path", "data_root"}
        different = [key for key in sorted(set(config0) | set(config1)) if key not in ignored and config0.get(key) != config1.get(key)]
        if state["progress"] != reference["progress"]:
            different.append("actual_training_progress")
        if state["observed_image_ids"] != reference["observed_image_ids"]:
            different.append("actual_observed_training_prefix")
        differences[label] = different
    return {"reference_label": labels[0], "differences_excluding_anchor_weight_and_file_paths": differences,
            "matched_actual_training_budget": len({state["progress"]["samples_seen"] for state in states.values()}) == 1,
            "interpretation": "Unequal checkpoint steps/configurations are descriptive comparisons, not a matched-budget ablation; baseline uses no trained delta"}


def grouped_spatial_summary(rows):
    def summarize(subset, stage):
        values = [row["spatial"][stage] for row in subset]
        present = [value for value in values if value["own_reference_pixels"]]
        totals = {key: sum(value[key] for value in values) for key in (
            "prediction_pixels", "own_reference_pixels", "own_intersection_pixels", "own_missed_pixels",
            "other_only_intersection_pixels", "outside_both_references_pixels")}
        return {
            "queries": len(values), "present_queries": len(present),
            "empty_predictions": sum(not value["prediction_pixels"] for value in values),
            "present_macro": {key: errors.mean_defined(value[key] for value in present) for key in (
                "own_dice", "own_iou", "own_precision", "own_recall", "own_missed_fraction")},
            "precision_defined_present_queries": sum(value["own_precision"] is not None for value in present),
            "boundary_present_macro": {key: errors.mean_defined(value["boundary"][key]["iou"] for value in present) for key in RATIO_KEYS},
            "totals": totals,
            "all_query_prediction_pixel_fractions": {
                "own_correct": errors.ratio(totals["own_intersection_pixels"], totals["prediction_pixels"]),
                "other_only": errors.ratio(totals["other_only_intersection_pixels"], totals["prediction_pixels"]),
                "outside_both": errors.ratio(totals["outside_both_references_pixels"], totals["prediction_pixels"])},
            "opposite_overlap_dominant_queries": sum(value["opposite_overlap_dominant_proxy"] for value in values),
        }
    result = {}
    for stage in ("candidate", "detected"):
        groups = {"overall": list(rows)}
        for field in ("prompt_key", "view_type", "recording_id"):
            for value in sorted({row[field] for row in rows}):
                groups[f"{field}:{value}"] = [row for row in rows if row[field] == value]
        for present in (True, False):
            groups[f"own_reference_present:{present}"] = [row for row in rows if row["target_present"] == present]
        result[stage] = {key: summarize(value, stage) for key, value in groups.items()}
    return result


def render_report(result):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    lines = ["# 手部分割：全量验证边界与错误分解", "",
             f"实际评估 {result['evaluated_images']} 张；完整 val：{result['full_val_evaluated']}。"
             "所有边界指标来自本次逐图前向时的真实 mask，不是从旧 scalar records 恢复。"
             f"仅预先固定的 {len(result['render_indices'])} 张保存可视化；指标覆盖范围不受 PNG 保存数量限制。", "",
             "参考是 SAM3 辅助 mask，不是全量独立人工真值。参考外像素不等同于前臂泄漏；同侧缺失像素也不能直接断言为漏手指。"
             "检测仅指 class×presence 分数≥0.5，不保证识别正确手。", "",
             "| 模型 | 实际训练样本 | 阶段 | Dice | 同侧漏掉比例 | Boundary IoU 4px等价 | 8px等价 | 16px等价 | 预测像素另一手独占 | 预测像素参考外 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, metrics in result["spatial_metrics"].items():
        samples = 0 if label == "baseline" else result["models"][label]["training_progress"]["samples_seen"]
        for stage in ("candidate", "detected"):
            group = metrics[stage]["overall"]
            macro, boundary, fractions = group["present_macro"], group["boundary_present_macro"], group["all_query_prediction_pixel_fractions"]
            lines.append(f"| {label} | {samples} | {stage} | " + " | ".join(fmt(value) for value in (
                macro["own_dice"], macro["own_missed_fraction"], *(boundary[key] for key in RATIO_KEYS),
                fractions["other_only"], fractions["outside_both"])) + " |")
    lines += ["", "Boundary IoU 内边带宽为 round(原图对角线×0.005/0.01/0.02)，最少1px；640×480时正好4/8/16px。"
              "这不是在所有分辨率固定4/8/16px，也不是 Boundary AP。零填充 chessboard distance transform 与重复3×3方形腐蚀等价。"
              f"定义：[作者实现]({errors.BOUNDARY_SOURCE})。", "",
              "Dice、漏分比例、Boundary IoU 对非空同侧参考查询取平均；最后两列汇总全部查询预测像素。"
              "预测为空时错误像素可能减少，因此必须联合看漏分与空预测数。所有零分母记null，不记满分。", "",
              "baseline 使用同一原始 VE 特征、无训练增量。CP0/CP1 是传入的恢复点别名，不隐含数据集、损失或完成轮数；"
              "实际 checkpoint format、训练配置、样本进度及是否同预算比较均记录于 summary.json。"
              "训练未完成2000样本也可诊断，但不能称2000样本结果或完整epoch。", ""]
    if not result["comparison"]["matched_actual_training_budget"]:
        lines += ["注意：恢复点的实际训练预算不同，不能把此表当成同预算消融实验。", ""]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("baseline-checkpoint", "cp0-checkpoint", "cp1-checkpoint"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--variant", choices=("all", *LABELS), default="all")
    parser.add_argument("--minimum-samples-seen", type=int, default=1)
    parser.add_argument("--indices", help="Optional explicit diagnostic subset; omitted means all 3449 val images")
    parser.add_argument("--render-count", type=int, default=12)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if not 0 <= args.render_count <= 12 or not 0 < args.gpu_memory_fraction <= .25 or args.minimum_samples_seen < 0:
        parser.error("Require render-count 0..12, memory fraction (0,.25], nonnegative minimum-samples")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.output_dir.exists() or args.output_dir.is_relative_to(args.data_root):
        parser.error("Output must be new and outside source split")
    args.labels = list(LABELS) if args.variant == "all" else [args.variant]
    supplied = {"baseline": args.baseline_checkpoint or args.cp0_checkpoint or args.cp1_checkpoint,
                "CP0": args.cp0_checkpoint, "CP1": args.cp1_checkpoint}
    if any(supplied[label] is None for label in args.labels):
        parser.error("Supply each requested checkpoint; baseline may derive from either provided checkpoint")
    args.checkpoints = {label: supplied[label] for label in args.labels}
    return args


def evaluate_model(model, dataset, images, references, indices, label, record_log, render_indices, progress_callback):
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    rows, rendered = [], {}
    for completed, index in enumerate(indices, 1):
        image = images[index]
        image_id = int(image["id"])
        batch = collate_fn_api([dataset[index]], dict_key="eval", with_seg_masks=True)["eval"]
        bilateral.validate_batch_identity(batch, [index], images)
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        bilateral.validate_frozen_noninteractive_model(model)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)[0]
        selected = select_predictions(output, (int(image["height"]), int(image["width"])))
        refs = references[image_id]
        ref_bands = {side: boundary_bands(refs[side])[0] for side in shared.CLASS_NAMES}
        stage = batch.find_inputs[0]
        for output_row, prediction in enumerate(selected):
            text_index = int(stage.text_ids[output_row])
            side, other = shared.CLASS_NAMES[text_index], shared.CLASS_NAMES[1 - text_index]
            mask = prediction.pop("prediction")
            measured = bilateral.measure_query(mask, refs[side], refs[other], prediction["top_confidence"])
            row = {
                "model": label, "dataset_index": index, "image_id": image_id,
                "observed_coco_image_id": image_id, "identity_verified": True,
                "file_name": image["file_name"], "recording_id": image["recording_id"],
                "view_type": bilateral.view_type(image), "primary_test": True,
                "diagnostic_ids": bilateral.diagnostic_ids(image), "prompt_key": side,
                "prompt_text": cached.NATURAL_PROMPTS[text_index],
                "split": "val", "dataset_role": "validation", "reference_description": previous.REFERENCE_DESCRIPTION,
                **prediction, **measured,
                "spatial": candidate_and_detected_metrics(mask, refs[side], refs[other],
                    detected=measured["detected"], own_bands=ref_bands[side]),
            }
            rows.append(row)
            record_log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if index in render_indices:
                rendered[(label, index, side)] = mask
        if completed == 1 or completed % 25 == 0 or completed == len(indices):
            record_log.flush()
            progress_callback(completed)
        del batch, output, selected, refs, ref_bands
    expected = {(int(images[index]["id"]), side) for index in indices for side in shared.CLASS_NAMES}
    observed = [(row["image_id"], row["prompt_key"]) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Actual image/query identity coverage differs from requested validation")
    return rows, rendered


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no GPU is used by the CPU helper tests")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")
    data, provenance = previous.verify_split(args.data_root, "val")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    references = previous.LazyReferences(data, cache_size=2)
    indices = bilateral.select_indices(images, args.indices)
    render_indices = shared.evenly_spaced(indices, args.render_count) if args.render_count else []
    tokenizer = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    base_hash, tokenizer_hash = shared.sha256(args.base_checkpoint), shared.sha256(tokenizer)
    core = previous.core_source_hashes(args.project_root)
    if not core:
        raise ValueError("No core source fingerprints")
    fingerprints = dict(provenance["files"])
    fingerprints.update({str(args.base_checkpoint): base_hash, str(tokenizer): tokenizer_hash})
    states, encoders, metadata = {}, {}, {}
    for label, path in args.checkpoints.items():
        digest = shared.sha256(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if shared.sha256(path) != digest:
            raise RuntimeError("Checkpoint is not a frozen snapshot; it changed while loading")
        trained, initial = validate_checkpoint_dispatch(state, minimum_samples=args.minimum_samples_seen,
                                                       base_hash=base_hash, tokenizer_hash=tokenizer_hash)
        if state["format"] == BOUNDARY_TRIAL_FORMAT:
            implementations = {path.name: shared.sha256(path) for path in boundary_trial_module().implementation_sources()}
            if state["training_config"].get("implementation_sha256") != implementations:
                raise ValueError("Boundary trial implementation fingerprints differ from checkpoint provenance")
        identity = previous.verify_training_identity(state)
        if (identity["provenance"]["root_ready_sha256"] != provenance["root_ready_sha256"]
                or identity["provenance"]["frozen_plan_sha256"] != provenance["frozen_plan_sha256"]):
            raise ValueError("Checkpoint train and validation publication differ")
        if state["training_config"].get("core_sources_sha256") != previous.object_hash(core):
            raise ValueError("Core sources differ from the training checkpoint")
        cache_artifact = semantic.verify_initial_cache_artifact(state)
        fingerprints.update(identity["provenance"]["files"])
        fingerprints[str(path)] = digest
        fingerprints[cache_artifact["path"]] = cache_artifact["sha256"]
        states[label] = state
        encoders[label] = semantic.cache_from_state(state["initial_cache_state_dict"], frozen=True) if label == "baseline" else trained
        metadata[label] = {
            "checkpoint": str(path), "checkpoint_sha256": digest, "checkpoint_format": state["format"],
            "training_config": state["training_config"], "training_progress": state["progress"],
            "training_applied_to_this_variant": label != "baseline", "training_identity": identity,
            "initial_cache_artifact": cache_artifact, "cache_metadata": initial.cache_metadata,
            "label_semantics": "Explicit checkpoint alias, not a claim of completed epochs or a fixed anchor coefficient",
        }
    comparison = compare_checkpoint_inputs(states)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "records").mkdir()
    (args.output_dir / "code-snapshot").mkdir()
    sources = {Path(__file__).resolve(), Path(inspect.getfile(previous)).resolve(), Path(inspect.getfile(shared)).resolve(),
               Path(inspect.getfile(bilateral)).resolve(), Path(inspect.getfile(semantic)).resolve(),
               Path(inspect.getfile(cached)).resolve(), Path(inspect.getfile(errors)).resolve(),
               args.project_root / "scripts/run_token_lr_pilot.py"}
    if any(state["format"] == BOUNDARY_TRIAL_FORMAT for state in states.values()):
        sources.update(boundary_trial_module().implementation_sources())
    snapshots = []
    for source in sorted(sources):
        digest = shared.sha256(source)
        target = args.output_dir / "code-snapshot" / source.name
        shutil.copy2(source, target)
        if shared.sha256(target) != digest or shared.sha256(source) != digest:
            raise RuntimeError("Code changed while making snapshot")
        fingerprints[str(source)] = digest
        fingerprints[str(target)] = digest
        snapshots.append({"source": str(source), "snapshot": str(target), "sha256": digest})
    for index in indices:
        image = images[index]
        path = args.data_root / image["file_name"]
        with Image.open(path) as rgb:
            if rgb.size != (int(image["width"]), int(image["height"])):
                raise ValueError("RGB dimensions differ from annotation")
        digest = shared.sha256(path)
        if digest != provenance["rgb_files"][int(image["id"])]["sha256"]:
            raise ValueError("Actual RGB bytes differ from published manifest")
        fingerprints[str(path)] = digest
    summary = {
        "format": "sam3-hand-boundary-validation-v1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False,
        "data_root": str(args.data_root), "dataset_role": "validation", "dataset_provenance": provenance,
        "base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": base_hash,
        "tokenizer_sha256": tokenizer_hash, "core_sources": core, "core_source_sha256": previous.object_hash(core),
        "evaluated_images": 0, "planned_images": len(indices), "evaluated_dataset_indices": indices,
        "evaluated_image_ids": [int(images[index]["id"]) for index in indices],
        "full_val_evaluated": False, "planned_full_val": len(indices) == len(images) == 3449,
        "diagnostic_subset": args.indices is not None, "render_indices": render_indices,
        "render_selection": "At most 12 evenly spaced source-frame-ordered selected indices; fixed before any model output",
        "boundary_scope": "Every evaluated candidate and thresholded mask, not only saved visualization images",
        "boundary_ratios": list(RATIOS), "boundary_pixels_at_640x480": [4, 8, 16],
        "boundary_implementation": "One zero-padded chessboard distance transform per candidate/reference; three width thresholds; detected reuses candidate or exact empty mask",
        "metric_definitions": errors.metric_definitions(), "reference_description": previous.REFERENCE_DESCRIPTION,
        "score_and_selection": "sigmoid(class)*sigmoid(presence), own score argmax, detection>=.5, no GT candidate choice",
        "preprocessing": "Unchanged existing 1008-square RGB preprocessing; BF16 forward; FP32 logits bilinear to original H/W then sigmoid>=.5",
        "thresholds_fitted": False, "mano_used": False, "geometry_or_reference_prompts_used": False,
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "gpu_memory_fraction": args.gpu_memory_fraction,
                    "amp": True, "amp_dtype": "bfloat16", "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "batch_size": 1},
        "comparison": comparison, "models": metadata, "metrics": {}, "spatial_metrics": {},
        "completed_images_per_model": {}, "code_snapshots": snapshots, "input_sha256": fingerprints,
    }
    shared.atomic_write_json(args.output_dir / "progress.json", summary)
    try:
        raw_dataset = shared.make_dataset(args.data_root)
        if len(raw_dataset) != len(images):
            raise RuntimeError("Loader length differs from COCO")
        dataset = semantic.IdentityCheckedDataset(raw_dataset, images)
        model = shared.load_ve_model(args.base_checkpoint)
        model.register_forward_hook(semantic.assert_finite_model_outputs)
        original_ve = model.backbone.language_backbone
        versions = [(parameter, int(parameter._version)) for parameter in model.parameters()]
        all_rows, all_masks = [], {}
        for label in args.labels:
            encoder = encoders[label].to(device="cuda")
            cached.install_cached_ve_text_encoder(model, encoder)
            cached.set_cached_ve_training_mode(model, train_delta=False)
            delta_versions = [(parameter, int(parameter._version)) for parameter in encoder.parameters()]
            start = len(dataset.observed_indices)

            def progress(completed):
                summary["completed_images_per_model"][label] = completed
                shared.atomic_write_json(args.output_dir / "progress.json", summary)
                print(json.dumps({"model": label, "completed_images": completed, "planned_images": len(indices)}), flush=True)

            with (args.output_dir / "records" / f"{label}.partial.jsonl").open("x", encoding="utf-8") as record_log:
                rows, masks = evaluate_model(model, dataset, images, references, indices, label,
                                             record_log, set(render_indices), progress)
            if dataset.observed_indices[start:] != indices:
                raise RuntimeError("Actual loader access differs from frozen requested order")
            if any(int(parameter._version) != version for parameter, version in delta_versions):
                raise RuntimeError("Delta changed during evaluation")
            record_path = args.output_dir / "records" / f"{label}.json"
            shared.atomic_write_json(record_path, rows)
            summary.setdefault("record_files", {})[label] = {"path": str(record_path), "sha256": shared.sha256(record_path)}
            all_rows.extend(rows)
            all_masks.update(masks)
            summary["metrics"].update(previous.summarize_validation(rows))
            summary["spatial_metrics"][label] = grouped_spatial_summary(rows)
            shared.atomic_write_json(args.output_dir / "progress.json", summary)
            cached.restore_original_ve_text_encoder(model, original_ve)
            encoder.to(device="cpu")
            torch.cuda.empty_cache()
        if any(int(parameter._version) != version for parameter, version in versions):
            raise RuntimeError("Original parameters changed during evaluation")
        summary["visuals"] = bilateral.render_results(data_root=args.data_root, output_dir=args.output_dir / "visuals",
            images=images, references=references, render_indices=render_indices,
            records=all_rows, masks=all_masks, labels=args.labels)
        if previous.core_source_hashes(args.project_root) != core:
            raise RuntimeError("Core sources changed during evaluation")
        for path, digest in fingerprints.items():
            if shared.sha256(Path(path)) != digest:
                raise RuntimeError(f"Source changed during evaluation: {path}")
        summary.update(status="completed", all_sources_unchanged=True, observed_identity_verified=True,
                       parameters_unchanged_by_version_counter=True, evaluated_images=len(indices),
                       full_val_evaluated=len(indices) == len(images) == 3449)
        with (args.output_dir / "REPORT.md").open("x", encoding="utf-8") as handle:
            handle.write(render_report(summary))
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        summary["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        shared.atomic_write_json(args.output_dir / "summary.json", summary)
        shared.atomic_write_json(args.output_dir / "progress.json", summary)


if __name__ == "__main__":
    main()

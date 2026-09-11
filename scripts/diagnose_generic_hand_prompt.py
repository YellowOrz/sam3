#!/usr/bin/env python3
"""Small frozen-VE diagnostic: generic hand versus anatomical side prompts.

Selection is fixed from the previous visualization manifest before inference:
A/B/C, then the earliest remaining both-visible images, up to eight images.
No training, GT prompts, candidate oracle, threshold fitting, or source writes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import inspect
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from scripts import evaluate_nakehand_tokens as evaluation
    from scripts import analyze_hand_segmentation_failures as errors
    from scripts.check_ve_prompt_equivalence import parameter_versions, verify_parameter_versions, snapshot_scripts
    from scripts.run_token_lr_pilot import core_source_hashes, object_hash
except ModuleNotFoundError:
    import evaluate_nakehand_tokens as evaluation
    import analyze_hand_segmentation_failures as errors
    from check_ve_prompt_equivalence import parameter_versions, verify_parameter_versions, snapshot_scripts
    from run_token_lr_pilot import core_source_hashes, object_hash


PROMPTS = ("hand", "left hand", "right hand")
KEYS = ("hand", "left_hand", "right_hand")
THRESHOLD = .5


def select_diagnostic_indices(images, references, manifest):
    """Reference presence is used only for a frozen diagnostic sample, not inference."""
    by_id = {int(image["id"]): index for index, image in enumerate(images)}
    seen = set()
    available = []
    diagnostics = {}
    for entry in manifest:
        image_id = int(entry["image_id"])
        if image_id in seen or image_id not in by_id or entry["dataset_index"] != by_id[image_id]:
            raise ValueError("Invalid or duplicate previous visualization identity")
        seen.add(image_id)
        index = by_id[image_id]
        names = evaluation.diagnostic_ids(images[index])
        if sorted(entry["diagnostic_ids"]) != sorted(names):
            raise ValueError("Diagnostic identity differs from annotations")
        for name in names:
            if name in diagnostics:
                raise ValueError("Duplicate named diagnostic image")
            diagnostics[name] = index
        if all(references[image_id][side].any() for side in evaluation.CLASS_NAMES):
            available.append(index)
    if not all(name in diagnostics for name in ("A", "B", "C")):
        raise ValueError("Previous visuals must include the fixed A/B/C diagnostics")
    selected = [diagnostics[name] for name in ("A", "B", "C")]
    if len(set(selected)) != 3:
        raise ValueError("A/B/C must identify three different images")
    selected += [index for index in sorted(available) if index not in selected][:5]
    return selected


def make_prompt_only_sample(sample):
    """Retain transformed RGB and identity; clear all object and geometry targets."""
    if len(sample.images) != 1 or [query.query_text for query in sample.find_queries] != list(evaluation.CLASS_NAMES):
        raise ValueError("Require one image and the original left/right query order")
    queries = []
    for key, prompt, template in zip(KEYS, PROMPTS, (sample.find_queries[0], *sample.find_queries)):
        if template.image_id != 0 or template.query_processing_order != 0 or template.inference_metadata is None:
            raise ValueError("Require one noninteractive image stage with identity metadata")
        for field in ("input_bbox", "input_bbox_label", "input_points"):
            value = getattr(template, field)
            if value is not None and (not isinstance(value, torch.Tensor) or value.numel()):
                raise ValueError("Original evaluation sample contains geometry prompts")
        metadata = replace(template.inference_metadata, object_id=-1, is_conditioning_only=False,
                           original_category_id=-1 if key == "hand" else template.inference_metadata.original_category_id)
        queries.append(replace(template, query_text=prompt, object_ids_output=[],
                               input_bbox=None, input_bbox_label=None, input_points=None,
                               semantic_target=None, is_exhaustive=False, is_pixel_exhaustive=False,
                               inference_metadata=metadata))
    return replace(sample, find_queries=queries, images=[replace(sample.images[0], objects=[])])


def validate_prompt_batch(batch, image):
    if len(batch.find_inputs) != 1 or len(batch.find_metadatas) != 1 or tuple(batch.find_text_batch) != PROMPTS:
        raise RuntimeError("Expected exactly one stage with hand/left hand/right hand")
    stage, metadata = batch.find_inputs[0], batch.find_metadatas[0]
    if stage.img_ids.tolist() != [0, 0, 0] or stage.text_ids.tolist() != [0, 1, 2]:
        raise RuntimeError("Unexpected actual prompt-to-image mapping")
    if metadata.coco_image_id.tolist() != [int(image["id"])] * 3:
        raise RuntimeError("Observed COCO image identity mismatch")
    if metadata.original_size.tolist() != [[int(image["height"]), int(image["width"])]] * 3:
        raise RuntimeError("Observed original image size mismatch")
    for field in ("input_boxes", "input_points", "input_boxes_before_embed", "input_points_before_embed"):
        value = getattr(stage, field, None)
        if value is not None and (not isinstance(value, torch.Tensor) or value.numel()):
            raise RuntimeError("Geometry prompts are forbidden")
    target = batch.find_targets[0]
    if target.num_boxes.tolist() != [0, 0, 0] or target.boxes.numel() or any(stage.object_ids):
        raise RuntimeError("Reference objects must not be passed to model inference")


def postprocess_outputs(output, shape):
    """Score selection precedes any reference measurement; no reference arguments."""
    logits, presence_logits, boxes, masks = (output[name] for name in (
        "pred_logits", "presence_logit_dec", "pred_boxes", "pred_masks"))
    if logits.ndim != 3 or logits.shape[0] != 3 or logits.shape[-1] != 1 or logits.shape[1] < 1:
        raise ValueError("Expected logits [3, decoder_queries, 1]")
    if presence_logits.shape[0] != 3 or presence_logits.numel() != 3:
        raise ValueError("Expected one presence logit per prompt")
    if boxes.shape != (*logits.shape[:2], 4) or masks.ndim != 4 or masks.shape[:2] != logits.shape[:2]:
        raise ValueError("Box/mask shape differs from prompt/decoder query count")
    if any(not bool(torch.isfinite(value).all()) for value in (logits, presence_logits, boxes, masks)):
        raise ValueError("Non-finite model output")
    probabilities = logits.float().sigmoid().squeeze(-1)
    presence = presence_logits.float().sigmoid().reshape(3)
    scores = probabilities * presence[:, None]
    result, predictions = {}, {}
    for row, key in enumerate(KEYS):
        top = int(scores[row].argmax())
        selected = torch.nonzero(scores[row] >= THRESHOLD, as_tuple=False).flatten().tolist()
        # Each resize exactly matches the existing evaluator: float32 logits,
        # bilinear original H/W, align_corners=False, then sigmoid >= .5.
        needed = sorted(set(selected + [top])) if key == "hand" else [top]
        decoded = {}
        for query in needed:
            resized = F.interpolate(masks[row, query][None, None].float(), size=shape,
                                    mode="bilinear", align_corners=False)[0, 0]
            decoded[query] = (resized.sigmoid() >= THRESHOLD).cpu().numpy()
        top_mask = decoded[top]
        union = np.zeros(shape, dtype=bool)
        if key == "hand":
            for query in selected:
                union |= decoded[query]
        else:
            union = top_mask.copy() if top in selected else union
        result[key] = {
            "prompt_text": PROMPTS[row], "row": row,
            "presence_probability": float(presence[row]),
            "top_decoder_query": top, "top_class_probability": float(probabilities[row, top]),
            "top_score": float(scores[row, top]), "top_detected": top in selected,
            "all_decoder_scores": [float(value) for value in scores[row]],
            "all_score_passing_decoder_queries": selected,
            "all_score_passing_count": len(selected),
            "retained_decoder_queries": selected if key == "hand" else ([top] if top in selected else []),
            "retention_rule": "all score>=0.5; no NMS or GT matching" if key == "hand" else "own top-score candidate; empty if score<0.5",
            "retained_boxes_cxcywh_normalized": [boxes[row, query].float().cpu().tolist()
                                                 for query in (selected if key == "hand" else ([top] if top in selected else []))],
        }
        predictions[key] = {"top_candidate": top_mask, "detected_union": union,
                            "instances": {query: decoded[query] for query in selected} if key == "hand" else {}}
    return result, predictions


def measure_diagnostic(predictions, references):
    """Only post-hoc measurements. Does not choose, modify, or rank predictions."""
    left, right = (references[side] for side in evaluation.CLASS_NAMES)
    reference_union = left | right
    empty = np.zeros_like(reference_union)
    result = {
        "generic_all_detected_union_vs_reference_union": errors.measure_mask(
            predictions["hand"]["detected_union"], reference_union, empty),
        "generic_top_candidate_vs_reference_union": errors.measure_mask(
            predictions["hand"]["top_candidate"], reference_union, empty),
        "side_top_detected_union_vs_reference_union": errors.measure_mask(
            predictions["left_hand"]["detected_union"] | predictions["right_hand"]["detected_union"], reference_union, empty),
        "per_side": {},
    }
    for side, own, other in (("left_hand", left, right), ("right_hand", right, left)):
        result["per_side"][side] = {
            stage: errors.measure_mask(predictions[side][stage], own, other)
            for stage in ("top_candidate", "detected_union")}
        result["per_side"][side]["generic_union_reference_coverage"] = errors.ratio(
            int((predictions["hand"]["detected_union"] & own).sum()), int(own.sum()))
    return result


def save_visuals(directory, source_rgb, references, predictions):
    directory.mkdir(parents=True, exist_ok=False)
    with Image.open(source_rgb) as image:
        image.convert("RGB").save(directory / "rgb.png")
    artifacts = [directory / "rgb.png"]
    masks = {"left_hand__reference": references["left_hand"], "right_hand__reference": references["right_hand"],
             "both_hands__reference": references["left_hand"] | references["right_hand"]}
    for key in KEYS:
        masks[f"{key}__top_candidate"] = predictions[key]["top_candidate"]
        masks[f"{key}__detected_union"] = predictions[key]["detected_union"]
    masks["side_prompts__detected_union"] = predictions["left_hand"]["detected_union"] | predictions["right_hand"]["detected_union"]
    for query, mask in predictions["hand"]["instances"].items():
        masks[f"hand__instance-query-{query:03d}"] = mask
    for name, mask in masks.items():
        path = directory / f"{name}.png"
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)
        with Image.open(path) as decoded:
            if not np.array_equal(np.asarray(decoded) > 0, mask):
                raise RuntimeError("PNG round-trip changed a prediction or reference")
        artifacts.append(path)
    return [{"path": str(path.resolve()), "sha256": evaluation.shared.sha256(path)} for path in artifacts]


def markdown_report(result):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    lines = ["# Generic hand 与左右手提示：固定小样本诊断", "",
             "同一冻结原 VE、同一张预处理图片、同一次 BF16 前向中的 hand / left hand / right hand 三查询。"
             "不训练、不拟合阈值、不以参考挑选实例。此为诊断子集，不是完整测试集结论。", "",
             "当前三个查询在同一次前向内比较；旧外测每张只有两个查询，batch 形状不同可能产生数值差异，"
             "本工具没有证明当前左右 mask 与旧外测逐位等价。", "",
             "generic hand 保留所有 class×presence 分数≥0.5 的实例并取并集；左右查询各保留自身最高分候选，低于0.5则为空。"
             "两种规则保留的实例数量不同，union Dice 不能单独证明“提示更好”；须结合每侧覆盖率、实例数和分离 PNG。", "",
             "| image ID | 诊断名 | hand 过阈值实例数 | hand union Dice | 左右 top union Dice | 左手检测后 Dice | 右手检测后 Dice | hand union 覆盖左参考 | hand union 覆盖右参考 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["records"]:
        metrics = row["metrics"]
        lines.append(f"| {row['image_id']} | {','.join(row['diagnostic_ids']) or '-'} | {row['prompts']['hand']['all_score_passing_count']} | " + " | ".join(fmt(value) for value in (
            metrics["generic_all_detected_union_vs_reference_union"]["own_dice"],
            metrics["side_top_detected_union_vs_reference_union"]["own_dice"],
            metrics["per_side"]["left_hand"]["detected_union"]["own_dice"],
            metrics["per_side"]["right_hand"]["detected_union"]["own_dice"],
            metrics["per_side"]["left_hand"]["generic_union_reference_coverage"],
            metrics["per_side"]["right_hand"]["generic_union_reference_coverage"])) + " |")
    lines += ["", "如何解读：hand 的实例并集覆盖两只参考手，而某一侧 query 漏掉或分到另一只手，支持优先调查左右身份绑定／提示语义；"
              "若 generic 与 side 都存在相同轮廓误差，则仅改左右语义未必足够。"
              "这是定位线索，不是因果证明；所有参考外像素可能是背景、物体、手臂或参考误差，不能直接称为前臂泄漏。", "",
              "本工具沿用既有外测 ≥0.5 分数／mask 约定。仓库 Sam3Processor 的边界比较为严格 >0.5；"
              "除这个明确记录的边界约定外，使用相同 class×presence 评分和原尺寸双线性插值规则。"
              "不执行额外 NMS、重复实例合并或 GT 匹配；过阈值 decoder 实例数量不等同于解剖学手数量。", "",
              "参考为 SAM3 辅助标注；A/B/C 已获用户认可，不代表所有像素均为独立人工真值。"
              "仅按事前固定的 A/B/C＋旧可视化中最早的其他双手帧选择，最多8张，没有按本次结果挑图。", "",
              "RGB、左右参考、参考并集、各提示候选／检测后 mask、generic 逐实例 mask 均分别保存于 visuals；没有混色叠加。", ""]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-summary", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu-memory-fraction", type=float, default=.25)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if not 0 < args.gpu_memory_fraction <= .25:
        parser.error("memory fraction must be in (0,.25]")
    if args.output_dir.exists():
        parser.error("output directory must not exist")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for inference; CPU tests cover helpers")
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    source_raw = args.source_summary.read_bytes()
    source = json.loads(source_raw)
    source_sha = hashlib.sha256(source_raw).hexdigest()
    if source.get("format") != "nakehand-frozen-bilateral-evaluation-v1" or source.get("status") != "completed":
        raise ValueError("Require the completed old nakehand evaluation")
    data_root = Path(source["data_root"]).resolve()
    if args.output_dir.is_relative_to(data_root) or args.output_dir.is_relative_to(args.source_summary.parent):
        raise ValueError("Output must be outside source data and old evaluation results")
    annotation_path = data_root / "annotations.json"
    annotation_sha = evaluation.shared.sha256(annotation_path)
    if annotation_sha != source["annotations_sha256"]:
        raise ValueError("Source annotations differ from old evaluation")
    images, references, _ = evaluation.load_coco_index(data_root)
    if evaluation.shared.sha256(annotation_path) != annotation_sha:
        raise ValueError("Source annotations changed while decoding references")
    manifest_path = args.source_summary.parent / "visuals/manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest != source["visuals"]:
        raise ValueError("Source visual manifest differs from completed evaluation")
    indices = select_diagnostic_indices(images, references, manifest)
    base_sha = evaluation.shared.sha256(args.base_checkpoint)
    if base_sha != source["base_checkpoint_sha256"]:
        raise ValueError("Require the same original VE checkpoint bytes as the old external test")
    tokenizer_path = args.project_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    tokenizer_sha = evaluation.shared.sha256(tokenizer_path)
    core = core_source_hashes(args.project_root)
    input_hashes = {str(args.source_summary): source_sha, str(manifest_path): manifest_sha,
                    str(annotation_path): annotation_sha, str(args.base_checkpoint): base_sha,
                    str(tokenizer_path): tokenizer_sha}
    for index in indices:
        image = images[index]
        rgb_path = errors.checked_child(data_root, image["file_name"])
        with Image.open(rgb_path) as rgb:
            if rgb.size != (image["width"], image["height"]):
                raise ValueError("Source RGB size mismatch")
        input_hashes[str(rgb_path)] = evaluation.shared.sha256(rgb_path)
        for side in evaluation.CLASS_NAMES:
            path = errors.checked_child(data_root, image["source_masks"][side.removesuffix("_hand")]["binary_reference_png"])
            input_hashes[str(path)] = evaluation.shared.sha256(path)
            with Image.open(path) as png:
                if not np.array_equal(errors.boolean_mask(np.asarray(png)), references[image["id"]][side]):
                    raise ValueError("Source reference PNG differs from COCO RLE")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "format": "sam3-generic-hand-prompt-diagnostic-v1", "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False,
        "scope": "fixed_diagnostic_subset_not_full_external_test", "prompts": list(PROMPTS),
        "indices": indices, "image_ids": [images[index]["id"] for index in indices],
        "selection_rule": "A/B/C in that order, then earliest remaining both-reference-visible old visualization indices; up to 8; frozen before inference",
        "source_summary": str(args.source_summary), "data_root": str(data_root),
        "base_checkpoint": str(args.base_checkpoint), "input_sha256": input_hashes,
        "core_sources": core, "core_source_sha256": object_hash(core),
        "preprocessing": "existing 1008 square RGB resize, normalize mean/std=.5; shared transformed RGB for all three prompts",
        "postprocessing": "float32 sigmoid(class)*sigmoid(presence); score>=.5; float32 logits bilinear to original H/W align_corners=False then sigmoid>=.5",
        "processor_boundary_difference": "Sam3Processor uses strict >.5; this diagnostic follows prior evaluation >=.5",
        "legacy_output_equivalence_limitation": "Three prompt rows rather than old bilateral two; same preprocessing/postprocessing rules, not a demonstrated bitwise match with historical side outputs",
        "reference_prompts_used": False, "reference_candidate_selection_used": False,
        "gt_matching_used": False, "thresholds_fitted": False, "mano_used": False,
        "model_input_targets": "all query object outputs and image object lists cleared; no points/boxes/semantic target",
        "metric_definitions": errors.metric_definitions(),
        "runtime": {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "amp": True, "amp_dtype": "bfloat16", "gpu_memory_fraction": args.gpu_memory_fraction,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "images_per_batch": 1, "prompt_rows_per_batch": 3},
        "records": [],
    }
    result["code_snapshots"] = snapshot_scripts(args.output_dir, [__file__, inspect.getfile(evaluation),
        inspect.getfile(evaluation.shared), inspect.getfile(errors),
        args.project_root / "scripts/check_ve_prompt_equivalence.py", args.project_root / "scripts/run_token_lr_pilot.py"])
    evaluation.shared.atomic_write_json(args.output_dir / "progress.json", result)
    try:
        dataset = evaluation.shared.make_dataset(data_root)
        if len(dataset) != len(images):
            raise RuntimeError("Loader image count differs from COCO")
        model = evaluation.shared.load_ve_model(args.base_checkpoint)
        model.requires_grad_(False)
        evaluation.validate_frozen_noninteractive_model(model)
        versions = parameter_versions(model)
        for index in indices:
            image = images[index]
            sample = dataset[index]
            original_batch = collate_fn_api([sample], dict_key="eval", with_seg_masks=True)["eval"]
            evaluation.validate_batch_identity(original_batch, [index], images)
            del original_batch
            prompt_sample = make_prompt_only_sample(sample)
            batch = collate_fn_api([prompt_sample], dict_key="eval", with_seg_masks=True)["eval"]
            validate_prompt_batch(batch, image)
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            evaluation.validate_frozen_noninteractive_model(model)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)[0]
            prompts, predictions = postprocess_outputs(output, (int(image["height"]), int(image["width"])))
            row = {"dataset_index": index, "image_id": image["id"], "file_name": image["file_name"],
                   "identity_verified": True, "diagnostic_ids": evaluation.diagnostic_ids(image),
                   "recording_id": image["recording_id"], "view_type": evaluation.view_type(image),
                   "prompts": prompts, "metrics": measure_diagnostic(predictions, references[image["id"]])}
            row["artifacts"] = save_visuals(args.output_dir / "visuals" / f"image-{image['id']:06d}",
                data_root / image["file_name"], references[image["id"]], predictions)
            result["records"].append(row)
            evaluation.shared.atomic_write_json(args.output_dir / "progress.json", result)
            print(json.dumps({"image_id": image["id"], "hand_instances": prompts["hand"]["all_score_passing_count"],
                              "completed_images": len(result["records"]), "planned_images": len(indices)}), flush=True)
            del sample, prompt_sample, batch, output, predictions
        verify_parameter_versions(versions)
        if core_source_hashes(args.project_root) != core:
            raise RuntimeError("Core sources changed during inference")
        for row in result["code_snapshots"]:
            input_hashes[row["source"]] = row["sha256"]
        for path, expected in input_hashes.items():
            if evaluation.shared.sha256(Path(path)) != expected:
                raise RuntimeError(f"Source changed during inference: {path}")
        result["parameters_unchanged_by_version_counter"] = True
        result["parameter_version_check_scope"] = "parameter version counters, not full tensor byte hashing"
        result["source_hashes_before_after_verified"] = True
        result["status"] = "completed"
        with (args.output_dir / "REPORT.md").open("x", encoding="utf-8") as handle:
            handle.write(markdown_report(result))
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        evaluation.shared.atomic_write_json(args.output_dir / "summary.json", result)
        evaluation.shared.atomic_write_json(args.output_dir / "progress.json", result)
    print(json.dumps({"status": result["status"], "images": len(result["records"]), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

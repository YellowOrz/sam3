"""CPU-only comparison of two preselected, complete fixed RealSense test runs.

Recompute metrics from saved candidate RLEs and the frozen auxiliary references.
Never discover checkpoints, rank checkpoints, alter thresholds, select samples,
or generate new favorable visualizations from test outcomes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from scripts import evaluate_residual_test as evaluation
from scripts import evaluate_nakehand_tokens as bilateral
from scripts import prepare_realsense_test as preparation


FORMAT = "sam3-fixed-residual-test-comparison-v1"
LABELS = ("ve-natural", "residual-preselected")
SIDES = ("left_hand", "right_hand")
PREDICTION_SELECTION = "argmax(sigmoid(class)*sigmoid(presence)); no reference-dependent selection"
PRECISION = "BF16 autocast; FP32 logits for sigmoid/interpolation; FP32 delta"
NON_INFERENCE_HELPERS = {"scripts/residual_ddp_checkpoint.py", "scripts/prepare_realsense_test.py",
                         "scripts/train_ve_initialized_tokens.py"}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(raw, object_pairs_hook=_object, parse_constant=invalid), hashlib.sha256(raw).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, probability=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (probability and not 0 <= value <= 1):
        raise ValueError(f"Invalid numeric value: {name}")
    return float(value)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256: {name}")
    return value


def _same(actual, expected, name):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Different object fields: {name}")
        for key, value in expected.items():
            _same(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Different list: {name}")
        for index, value in enumerate(expected):
            _same(actual[index], value, f"{name}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(_number(actual, name), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Different recomputed value: {name}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Different value or type: {name}")


def load_contract(root):
    root = Path(root).resolve()
    # This checks publication hashes, all 384 PNG hashes, reference PNG/RLE
    # agreement, seeded selection and complete source image/query identities.
    images, refs, plan, fingerprints = evaluation.load_test(root)
    documents = {}
    for name in ("READY.json", "manifest.json", "annotations.json", "frozen-plan.json"):
        documents[name], digest = read_json(root / name)
        if digest != fingerprints[str(root / name)]:
            raise ValueError("Test publication changed during comparison")
    manifest = documents["manifest.json"]
    if plan.get("format") != preparation.FORMAT or plan.get("dataset_role") != "external_test_only":
        raise ValueError("Unexpected frozen test plan")
    expected_render = {name: [frames[0], frames[len(frames) // 2]] for name, frames in plan["selection"].items()}
    _same(plan["render_selection"], expected_render, "fixed render selection")
    outputs = {row["image_id"]: row for row in manifest["image_outputs"]}
    overlap_ids = []
    for image in images:
        image_id = _integer(image["id"], "image ID", 1)
        _integer(image["source_frame_index"], "source frame")
        _same(outputs[image_id]["source_mapping"], image["source_mapping"], "manifest source mapping")
        overlap = int((refs[image_id][SIDES[0]] & refs[image_id][SIDES[1]]).sum())
        _same(outputs[image_id]["left_right_overlap_pixels"], overlap, "reference overlap pixels")
        if overlap:
            overlap_ids.append(image_id)
    _same(manifest["statistics"]["images_with_left_right_overlap"], len(overlap_ids), "reference overlap count")
    return {"root": root, "images": images, "references": refs, "plan": plan,
            "fingerprints": fingerprints, "overlap_image_ids": overlap_ids}


def _code_hashes(summary):
    result = {}
    for path, digest in summary.get("implementation_sha256", {}).items():
        parts = Path(path).parts
        positions = [index for index, part in enumerate(parts) if part in ("scripts", "sam3")]
        if not positions:
            raise ValueError("Unrecognized implementation path")
        logical = "/".join(parts[positions[0]:])
        if logical in result:
            raise ValueError("Duplicate logical implementation source")
        result[logical] = _digest(digest, logical)
    required = {"scripts/evaluate_residual_test.py", "scripts/evaluate_nakehand_tokens.py",
                "scripts/evaluate_bilateral_tokens.py", "scripts/cached_ve_text_features.py",
                "scripts/evaluate_ve_initialized_tokens.py", "scripts/residual_ddp_validation.py"}
    if not required <= result.keys() or not any(name.startswith("sam3/") for name in result):
        raise ValueError("Incomplete inference implementation fingerprints")
    return result


def _check_summary(summary, label, contract):
    for key, expected in {"format": evaluation.FORMAT, "status": "complete",
            "dataset_role": "external_test_only", "images": 128, "queries_per_model": 256,
            "actual_complete_query_coverage_verified": True,
            "reference_description": preparation.REFERENCE_DESCRIPTION,
            "detection_threshold": .5, "mask_threshold": .5, "boundary_width_original_pixels": 4,
            "prediction_selection": PREDICTION_SELECTION, "precision": PRECISION}.items():
        _same(summary.get(key), expected, key)
    _same(summary.get("protocol"), contract["plan"], "frozen protocol")
    models = summary.get("models")
    if not isinstance(models, dict) or label not in models or not set(models) <= set(LABELS):
        raise ValueError("Expected the explicit preselected model label")
    if LABELS[0] in models:
        _same(models[LABELS[0]], {"source": "actual original text encoder, natural prompts, no cached replacement"},
              "original VE encoder metadata")
    fingerprints = summary.get("source_fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("Missing source fingerprints")
    for path, digest in fingerprints.items():
        _digest(digest, path)
    plan_hash = contract["fingerprints"][str(contract["root"] / "frozen-plan.json")]
    roots = [Path(path).parent for path, digest in fingerprints.items()
             if Path(path).name == "frozen-plan.json" and digest == plan_hash]
    if len(roots) != 1:
        raise ValueError("Missing or ambiguous fixed test fingerprint root")
    expected = {str(roots[0] / Path(path).relative_to(contract["root"])): digest
                for path, digest in contract["fingerprints"].items()}
    for path, digest in expected.items():
        if fingerprints.get(path) != digest:
            raise ValueError(f"Changed or missing fixed test input fingerprint: {path}")
    return _code_hashes(summary), {path: digest for path, digest in fingerprints.items() if path not in expected}


def recompute_records(records, label, contract):
    if not isinstance(records, list) or len(records) != 256:
        raise ValueError("Require all 256 records for each model")
    images = {image["id"]: image for image in contract["images"]}
    seen, result = set(), []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid query record")
        image_id = _integer(record.get("image_id"), "record image ID", 1)
        side = record.get("prompt_key")
        key = (image_id, side)
        if image_id not in images or side not in SIDES or key in seen:
            raise ValueError("Unknown or duplicate image/side identity")
        seen.add(key)
        image = images[image_id]
        expected = {"model": label, "dataset_role": "external_test_only", "identity_verified": True,
            "dataset_index": image_id - 1, "image_id": image_id, "primary_test": True,
            "diagnostic_ids": [], "reference_description": preparation.REFERENCE_DESCRIPTION,
            "prompt_text": evaluation.cached.NATURAL_PROMPTS[SIDES.index(side)],
            **{name: image[name] for name in ("file_name", "recording_id", "source_frame_index", "source_mapping")}}
        for name, value in expected.items():
            _same(record.get(name), value, name)
        confidence = _number(record.get("top_confidence"), "confidence", probability=True)
        probability = _number(record.get("top_class_probability"), "class probability", probability=True)
        presence = _number(record.get("presence_probability"), "presence probability", probability=True)
        product = float(np.float32(probability) * np.float32(presence))
        if (not math.isclose(confidence, product, abs_tol=1e-7, rel_tol=1e-7)
                or (confidence >= .5) != (product >= .5)):
            raise ValueError("Confidence differs from class * presence")
        _integer(record.get("selected_decoder_query"), "selected decoder query")
        detections = _integer(record.get("detections_above_threshold"), "detection count")
        if (detections > 0) != (confidence >= .5):
            raise ValueError("Detection count contradicts selected maximum confidence")
        rle = record.get("prediction_rle")
        if (not isinstance(rle, dict) or set(rle) != {"size", "counts"}
                or rle.get("size") != [480, 640] or not isinstance(rle.get("counts"), str)):
            raise ValueError("Require original-size compressed candidate RLE")
        _same(rle["size"], [480, 640], "RLE dimensions")
        try:
            candidate = mask_utils.decode(rle)
        except Exception as error:
            raise ValueError("Invalid prediction RLE") from error
        if candidate.shape != (480, 640) or not np.isin(candidate, (0, 1)).all():
            raise ValueError("Invalid decoded candidate dimensions or mask values")
        reference = contract["references"][image_id][side]
        other = contract["references"][image_id][SIDES[1 - SIDES.index(side)]]
        computed = bilateral.measure_query(candidate.astype(bool), reference, other, confidence)
        evaluation.add_boundary(computed, candidate.astype(bool), reference)
        for name, value in computed.items():
            if name not in record:
                raise ValueError(f"Missing recomputed record field: {name}")
            _same(record.get(name), value, f"record {image_id}/{side}/{name}")
        result.append({**expected, "prompt_key": side, **computed})
    if seen != {(image_id, side) for image_id in images for side in SIDES}:
        raise ValueError("Incomplete fixed image/side coverage")
    return sorted(result, key=lambda row: (row["image_id"], SIDES.index(row["prompt_key"])))


def _progress(summary):
    metadata = summary["models"][LABELS[1]]
    progress, config = metadata.get("progress"), metadata.get("training_config")
    if not isinstance(progress, dict) or not isinstance(config, dict):
        raise ValueError("Missing actual residual checkpoint progress")
    step = _integer(progress.get("global_step"), "actual checkpoint step", 1)
    per_epoch = _integer(config.get("steps_per_epoch"), "steps per epoch", 1)
    epochs = _integer(config.get("epochs"), "planned epochs", 1)
    batch = _integer(config.get("global_batch_size"), "global batch", 1)
    size = _integer(config.get("dataset_size"), "training dataset size", 1)
    if size // batch != per_epoch or step > per_epoch * epochs:
        raise ValueError("Checkpoint progress contradicts training configuration")
    completed, offset = divmod(step, per_epoch)
    expected = {"global_step": step, "next_epoch": completed, "next_step_in_epoch": offset,
        "samples_seen": step * batch, "planned_steps": per_epoch * epochs,
        "planned_samples": per_epoch * epochs * batch, "completed_epochs": completed,
        "training_complete": completed == epochs, "dropped_images_per_epoch": size % batch}
    _same(progress, expected, "actual checkpoint progress")
    for key in ("initial_cache_verified", "actual_training_identities_verified"):
        _same(metadata.get(key), True, key)
    rank_audit = metadata.get("rank_cache_consistency_audit_present_and_verified")
    if type(rank_audit) is not bool:
        raise ValueError("Rank cache consistency audit status must be boolean")
    norms = metadata.get("delta_l2_per_side")
    if not isinstance(norms, list) or len(norms) != 2 or any(_number(value, "delta norm") < 0 for value in norms):
        raise ValueError("Invalid residual norm metadata")
    note = summary.get("checkpoint_selection_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("Missing unchanged pre-test checkpoint selection note")
    return {"checkpoint_sha256": _digest(metadata.get("checkpoint_sha256"), "residual checkpoint"),
            "global_step": step, "actual_epochs": step / per_epoch,
            "completed_epochs": completed, "steps_into_next_epoch": offset, "samples_seen": step * batch,
            "planned_epochs": epochs, "checkpoint_selection_note": note,
            "rank_cache_consistency_audit_present_and_verified": rank_audit,
            "best_note": deepcopy(metadata.get("best_note", summary.get("best_note", note))),
            "original_metadata": deepcopy(metadata)}


def _visualizations(summary, summary_path, contract):
    expected = [image for image in contract["images"] if image["render_preselected"]]
    rows = summary.get("visualizations")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Incomplete pre-fixed visualizations")
    by_id = {}
    for row in rows:
        image_id = _integer(row.get("image_id"), "visualization image ID", 1)
        if image_id in by_id:
            raise ValueError("Duplicate visualization identity")
        by_id[image_id] = row
    if set(by_id) != {image["id"] for image in expected}:
        raise ValueError("Visualization selection differs from the frozen plan")
    result = []
    for image in expected:
        row = by_id[image["id"]]
        _same(row.get("dataset_index"), image["id"] - 1, "visualization index")
        _same(row.get("diagnostic_ids"), [], "visualization diagnostics")
        name = f"image-{image['id']:06d}"
        original = Path(row.get("comparison", ""))
        if original.name != "comparison.png" or original.parent.name != name or Path(row.get("directory", "")) != original.parent:
            raise ValueError("Visualization path contradicts fixed image identity")
        local = summary_path.parent / "visuals" / name / "comparison.png"
        result.append({"image_id": image["id"], "recording_id": image["recording_id"],
            "source_frame_index": image["source_frame_index"], "comparison": str(local),
            "available_locally": local.is_file(), "original_comparison": str(original),
            "comparison_sha256": sha256(local) if local.is_file() else None,
            "rgb": str(local.parent / "rgb.png")})
    return result


def _difference(left, right):
    if isinstance(left, dict):
        return {key: _difference(value, right[key]) for key, value in left.items()}
    return None if left is None or right is None else right - left


def compare(ve_summary, residual_summary, data_root, *, ve_records=None, residual_records=None):
    contract = load_contract(data_root)
    inputs, summaries, recomputed, visuals, codes, model_inputs = {}, {}, {}, {}, {}, {}
    for label, summary_path, record_path in zip(LABELS, (ve_summary, residual_summary), (ve_records, residual_records)):
        summary_path = Path(summary_path).resolve()
        record_path = Path(record_path).resolve() if record_path is not None else summary_path.parent / f"records-{label}.json"
        summary, inputs[str(summary_path)] = read_json(summary_path)
        records, inputs[str(record_path)] = read_json(record_path)
        codes[label], model_inputs[label] = _check_summary(summary, label, contract)
        rows = recompute_records(records, label, contract)
        recomputed[label] = evaluation.summarize(rows)
        _same(summary.get("metrics", {}).get(label), recomputed[label], f"summary metrics for {label}")
        visuals[label] = _visualizations(summary, summary_path, contract)
        summaries[label] = summary
    inference = [{name: digest for name, digest in codes[label].items() if name not in NON_INFERENCE_HELPERS} for label in LABELS]
    if inference[0] != inference[1]:
        raise ValueError("Inference implementation changed between model evaluations")
    metadata = _progress(summaries[LABELS[1]])
    config = metadata["original_metadata"]["training_config"]
    for label in LABELS:
        expected = [_digest(config.get(key), key) for key in ("base_sha256", "tokenizer_sha256")]
        if LABELS[1] in summaries[label]["models"]:
            current = _progress(summaries[label])
            if current["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
                raise ValueError("Only the supplied preselected residual checkpoint may be compared")
            current_config = current["original_metadata"]["training_config"]
            expected += [current["checkpoint_sha256"],
                         _digest(current_config.get("initial_cache_file_sha256"), "initial cache"),
                         _digest(current_config.get("annotations_sha256"), "training annotations")]
        if sorted(model_inputs[label].values()) != sorted(expected):
            raise ValueError("Different or unbound model/tokenizer/residual input fingerprints")
    visual_hashes = {row["comparison"]: row["comparison_sha256"] for rows in visuals.values()
                     for row in rows if row["comparison_sha256"] is not None}
    input_hashes = {**inputs, **contract["fingerprints"], **visual_hashes}
    for path, expected_hash in input_hashes.items():
        if sha256(path) != expected_hash:
            raise ValueError("Comparison input changed while reading")
    overlap_count = len(contract["overlap_image_ids"])
    return {"format": FORMAT, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_role": "external_test_only", "images": 128, "queries_per_model": 256,
        "reference_description": preparation.REFERENCE_DESCRIPTION,
        "reference_overlap": {"images": overlap_count, "image_ids": contract["overlap_image_ids"],
            "interpretation": "左右辅助参考正像素有交集，原样保留；不等于确证标注错误数或预测错侧数。"},
        "thresholds": {"detection": .5, "mask": .5, "boundary_original_pixels": 4},
        "residual_checkpoint": metadata, "metrics_recomputed_from_candidate_rle_and_frozen_references": recomputed,
        "difference_residual_minus_ve": _difference(recomputed[LABELS[0]], recomputed[LABELS[1]]),
        "visualizations": visuals, "input_sha256": input_hashes,
        "evaluation_implementation_sha256": codes,
        "comparison_implementation_sha256": {str(path): sha256(path) for path in
            (Path(__file__).resolve(), Path(evaluation.__file__).resolve(), Path(bilateral.__file__).resolve(),
             Path(preparation.__file__).resolve(), Path(evaluation.shared.__file__).resolve(),
             Path(__file__).resolve().with_name("residual_ddp_validation.py"))},
        "non_inference_helper_differences": sorted(name for name in NON_INFERENCE_HELPERS
            if codes[LABELS[0]].get(name) != codes[LABELS[1]].get(name)),
        "selection_policy": "Exactly the supplied residual checkpoint; no checkpoint or sample selection using test results",
        "limitations": ["这是SAM3辅助传播参考的一致性，不是独立全人工像素GT准确率。",
            "128帧来自8段录像，帧间相关；不据此给出独立样本假设下的精确置信区间或显著性结论。",
            "错侧指标仅为另一侧参考重叠占优代理，不是解剖左右身份真值。",
            "RLE可重算已选候选与参考的一致性；没有全部decoder/logits，不能重新证明argmax或像素阈值执行。",
            "已验展示选择身份并记录本地比较图哈希；PNG内容未与RLE逐像素核验。",
            "只列冻结计划的16张可视化；未按本次效果筛选、生成或替换样本。"]}


def markdown_report(result):
    metrics = result["metrics_recomputed_from_candidate_rle_and_frozen_references"]
    metadata = result["residual_checkpoint"]
    def value(number):
        return "不适用" if number is None else str(number) if type(number) is int else f"{number:.4f}"
    lines = ["固定 RealSense 测试对比", "",
        f"128 图、8 段录像、每模型 256 条左右查询。残差 checkpoint 实际完成 step {metadata['global_step']}，"
        f"{metadata['actual_epochs']:.4f} epoch（完整 {metadata['completed_epochs']} epoch + {metadata['steps_into_next_epoch']} 步；"
        f"计划 {metadata['planned_epochs']} epoch）。",
        f"原始选点说明：{metadata['checkpoint_selection_note']}", "",
        f"{result['reference_overlap']['images']} 张图的左右辅助参考正像素有交集，已原样保留；不代表相同数量的确证错误或模型错侧。",
        "指标均由保存的候选 RLE 与冻结参考重新计算，并核验原汇总；差值为残差减原 VE。", "",
        "| 范围／指标 | 原 VE | 残差 | 差值 |", "|---|---:|---:|---:|"]
    if metadata["best_note"] != metadata["checkpoint_selection_note"]:
        lines[4:4] = [f"原始 best_note：{metadata['best_note']}", ""]
    fields = {"present_mean_candidate_dice": "候选 Dice", "present_mean_miss_zero_dice": "漏检置零 Dice",
              "candidate_boundary_iou_4px": "候选 BoundaryIoU 4px", "miss_zero_boundary_iou_4px": "漏检置零 BoundaryIoU 4px",
              "false_negative_queries": "FN 数", "absent_side_false_positive_rate": "无本侧参考 FP 率",
              "false_positive_queries": "无本侧参考 FP 数",
              "both_visible_detected_opposite_dominant_rate_per_all_queries": "双手可见错侧代理率"}
    for scope, path in (("整体", ("overall",)), ("左手", ("per_side", "left_hand")), ("右手", ("per_side", "right_hand"))):
        groups = [metrics[label] for label in LABELS]
        for component in path:
            groups = [group[component] for group in groups]
        for key, title in fields.items():
            a, b = (group[key] for group in groups)
            lines.append(f"| {scope}／{title} | {value(a)} | {value(b)} | {value(None if a is None or b is None else b-a)} |")
    lines += ["", "按录像列出漏检置零 Dice；完整左右、边界、FN/FP 和代理差异保存在 JSON。", "",
              "| 录像 | 原 VE | 残差 | 差值 |", "|---|---:|---:|---:|"]
    for recording in sorted(metrics[LABELS[0]]["per_recording"]):
        a, b = (metrics[label]["per_recording"][recording]["present_mean_miss_zero_dice"] for label in LABELS)
        lines.append(f"| {recording} | {value(a)} | {value(b)} | {value(None if a is None or b is None else b-a)} |")
    a, b = (metrics[label]["recording_macro_miss_zero_dice"] for label in LABELS)
    lines.append(f"| 录像等权平均 | {value(a)} | {value(b)} | {value(None if a is None or b is None else b-a)} |")
    lines += ["", "冻结的可视化入口（每录像预选第一张和中间一张；未按结果改选）：", "",
              "| 录像／源帧／image ID | 原 VE | 残差 |", "|---|---|---|"]
    for ve, residual in zip(result["visualizations"][LABELS[0]], result["visualizations"][LABELS[1]]):
        links = [f"[{'查看' if row['available_locally'] else '图像待汇入'}](<{row['comparison']}>)" for row in (ve, residual)]
        lines.append(f"| {ve['recording_id']}／{ve['source_frame_index']}／{ve['image_id']} | {links[0]} | {links[1]} |")
    lines += ["", *result["limitations"], ""]
    if result["non_inference_helper_differences"]:
        lines += ["两次评估的非推理辅助文件哈希不同（已逐项保留）："
                  + "、".join(result["non_inference_helper_differences"])
                  + "。推理实现指纹一致性已单独核验。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ve-summary", "residual-summary", "data-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--ve-records", type=Path)
    parser.add_argument("--residual-records", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Comparison output must be a new directory")
    result = compare(args.ve_summary, args.residual_summary, args.data_root,
                     ve_records=args.ve_records, residual_records=args.residual_records)
    report = markdown_report(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output_dir / "comparison.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": "complete", "output": str(args.output_dir.resolve()),
                      "residual_step": result["residual_checkpoint"]["global_step"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

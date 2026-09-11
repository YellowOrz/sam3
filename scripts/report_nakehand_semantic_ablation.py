#!/usr/bin/env python3
"""Checked CPU report for the three-mode nakehand development ablation.

No threshold tuning, external writes or GPU work. Reject partial comparisons.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from scripts import evaluate_nakehand_tokens as metrics

LABELS = ("ve-frozen-cache", "ve-delta-unconstrained", "ve-delta-anchored")
FORMAT = "sam3-nakehand-semantic-evaluation-v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_training_contracts(first, second):
    """Compare actual initial conditions and successfully consumed samples."""
    import torch

    configs = [deepcopy(state["training_config"]) for state in (first, second)]
    weights = [config.pop("anchor_weight") for config in configs]
    if weights != [0., 1.] or configs[0] != configs[1]:
        raise ValueError("Trials must differ only by anchor_weight=0 versus 1")
    for state in (first, second):
        if state.get("format") != "sam3-nakehand-semantic-delta-training-v1":
            raise ValueError("Wrong training format")
        progress = state.get("progress", {})
        if (progress.get("samples_seen") != 2000 or progress.get("completed_steps") != 2000
                or progress.get("pilot_complete") is not True or state.get("next_step") != 2000):
            raise ValueError("Not two complete 2000-sample short pilots")
    for name in ("planned_dataset_indices", "observed_image_ids"):
        if first.get(name) != second.get(name) or len(first.get(name, [])) != 2000:
            raise ValueError(f"Trial sample identities differ: {name}")
    a, b = first["initial_cache_state_dict"], second["initial_cache_state_dict"]
    if set(a) != set(b):
        raise ValueError("Initial cache keys differ")
    for name, value in a.items():
        other = b[name]
        same = (isinstance(other, torch.Tensor) and value.dtype == other.dtype
                and torch.equal(value, other)) if isinstance(value, torch.Tensor) else value == other
        if not same:
            raise ValueError(f"Initial semantic cache differs: {name}")
    return {"same_initial_cache": True, "same_training_sample_prefix": True,
            "only_anchor_weight_differs": True, "actual_samples_per_trial": 2000}


def validate_summary(summary, records):
    if summary.get("format") != FORMAT or summary.get("status") != "completed":
        raise ValueError("Require a completed semantic evaluation summary")
    if (summary.get("full_val_evaluated") is not True or summary.get("evaluated_images") != 3449
            or summary.get("diagnostic_training_checkpoint") is not False
            or set(summary.get("models", {})) != set(LABELS)):
        raise ValueError("Require all three variants on all 3449 validation frames")
    if summary.get("detection_threshold") != .5 or summary.get("mask_threshold") != .5:
        raise ValueError("Fixed 0.5 thresholds required")
    if (summary.get("dataset_role") != "validation" or summary.get("training_performed") is not False
            or summary.get("thresholds_fitted_on_nakehand") is not False
            or any(summary.get(key) is not True for key in (
                "observed_identity_verified", "all_sources_unchanged", "actual_training_prefix_verified"))):
        raise ValueError("Validation/source/identity or no-calibration contract is not satisfied")
    if set(records) != set(LABELS):
        raise ValueError("Missing model records")
    ids = summary.get("evaluated_image_ids", [])
    if len(ids) != 3449 or len(set(ids)) != 3449:
        raise ValueError("Invalid validation image identity set")
    expected = {(image_id, side) for image_id in ids for side in metrics.CLASS_NAMES}
    output = {}
    for label in LABELS:
        model = summary["models"][label]
        expected_kind = "frozen_natural_ve_cache" if label == LABELS[0] else "semantic_delta"
        expected_anchor = None if label == LABELS[0] else float(label == LABELS[2])
        if (model.get("kind") != expected_kind or model.get("anchor_weight") != expected_anchor
                or model.get("training_applied_to_this_variant") is not (label != LABELS[0])):
            raise ValueError("Model role does not match its comparison label")
        rows = records[label]
        if len(rows) != len(expected) or {(row["image_id"], row["prompt_key"]) for row in rows} != expected:
            raise ValueError(f"Incomplete or duplicate actual query identities: {label}")
        for row in rows:
            if (row.get("model") != label or row.get("identity_verified") is not True
                    or row.get("observed_coco_image_id") != row["image_id"]
                    or row.get("split") != "val" or row.get("dataset_role") != "validation"
                    or row.get("primary_test") is not True):
                raise ValueError("Wrong record model or unchecked actual image identity")
            for key in ("top_confidence", "presence_probability", "top_class_probability"):
                value = row[key]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {key}")
            if (row.get("detected") != (row["top_confidence"] >= .5)
                    or not math.isclose(row["top_confidence"], row["presence_probability"] * row["top_class_probability"],
                                        rel_tol=1e-6, abs_tol=1e-7)):
                raise ValueError("Inconsistent confidence or threshold")
        recalculated = metrics.grouped_summary(rows)
        saved = summary["metrics"][label].get("validation")
        if recalculated != saved:
            raise ValueError("Saved validation metrics differ from actual records")
        output[label] = recalculated
    return output


def number(value):
    return "N/A" if value is None else f"{value:.4f}"


def fraction(count, denominator):
    return f"{count}/{denominator}（{count / denominator:.2%}）" if denominator else "0/0（N/A）"


def render_report(summary, measured, source, comparison):
    lines = ["# nakehand：原VE与语义增量的开发对照", "",
             "固定录像级val共3449帧（ego142020）。两训练组各实际完成2000样本，并非两轮或完整epoch；原VE不训练。",
             "参考来自SAM3辅助传播，13帧曾获人眼认可，不是全量独立人工GT。val已用于开发，不能当未见最终benchmark。",
             "本轮不使用development_holdout，不调整0.5检测/mask阈值，不按参考重叠挑预测。检出只代表达阈，不保证物理手别正确。", "",
             "## 完整验证结果", "",
             "| 模型 | 候选Dice | 漏检计零Dice | 达阈检出/正查询 | 缺席侧误检/负查询 | 双手候选错侧代理 |",
             "|---|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        row = measured[label]["overall"]
        lines.append(f"| {label} | {number(row['present_mean_candidate_dice'])} | "
                     f"{number(row['present_mean_miss_zero_dice'])} | "
                     f"{fraction(row['true_positive_queries'], row['present_queries'])} | "
                     f"{fraction(row['false_positive_queries'], row['absent_queries'])} | "
                     f"{fraction(row['both_visible_candidate_opposite_dominant_queries'], row['both_visible_queries'])} |")
    lines += ["", "## 对照条件和解释边界", "",
              "两训练组相同语义缓存、基座、实际2000样本顺序、AdamW、lr=0.001、batch1/BF16和task loss；唯一语义配置差异为归一化增量正则权重0/1。该LR和正则强度为预先固定试探值，不是已选出的最优值。",
              "候选错侧是另一侧参考IoU更高且相交的重叠代理，不等价于经过人工确认的解剖学左右互换；完整双侧/空图细分保留在JSON。",
              "必须同时看mask质量、漏检和错侧，不能仅凭损失下降或分数提高切换主线。原VE的错误也不作为强制模仿目标。", "",
              f"可比性核验：`{json.dumps(comparison, ensure_ascii=False, sort_keys=True)}`", "",
              f"[完整摘要]({source})", "", "## 分离可视化", ""]
    for item in summary.get("visuals", []):
        lines.append(f"- [图像 {item['image_id']}]({item['comparison']})")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--unconstrained-checkpoint", type=Path, required=True)
    parser.add_argument("--constrained-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.summary.resolve()
    summary_hash = file_hash(source)
    summary = json.loads(source.read_text())
    records = {label: json.loads((source.parent / "records" / f"{label}.json").read_text()) for label in LABELS}
    measured = validate_summary(summary, records)
    import torch

    states = [torch.load(path, map_location="cpu", weights_only=True)
              for path in (args.unconstrained_checkpoint, args.constrained_checkpoint)]
    for label, path in zip(LABELS[1:], (args.unconstrained_checkpoint, args.constrained_checkpoint)):
        if file_hash(path) != summary["models"][label]["checkpoint_sha256"]:
            raise ValueError("Reported model differs from supplied training checkpoint")
    comparison = compare_training_contracts(*states)
    for path, key in ((Path(summary["base_checkpoint"]), "base_checkpoint_sha256"),
                      (Path(summary["data_root"]) / "annotations.json", "annotations_sha256")):
        if file_hash(path) != summary[key]:
            raise ValueError(f"Report provenance changed: {key}")
    if file_hash(source) != summary_hash:
        raise RuntimeError("Summary changed during reporting")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(render_report(summary, measured, source, comparison))
    print(args.output, flush=True)


if __name__ == "__main__":
    main()

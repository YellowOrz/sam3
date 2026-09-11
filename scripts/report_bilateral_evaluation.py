#!/usr/bin/env python3
"""Build a CPU-only Markdown report from completed bilateral evaluations.

Inputs must share the annotation hash, evaluated images, thresholds, confidence
definition, and base checkpoint path. No threshold/model is selected on test.
This report describes measured effects; it does not equate training loss with
generalization or prompt-presence AP with COCO mask AP.
"""

import argparse
import hashlib
import json
from pathlib import Path


COMPARISON_KEYS = (
    "annotations_sha256", "base_checkpoint", "evaluated_images",
    "detection_threshold", "mask_threshold", "confidence_definition",
)


def merge_summaries(summaries):
    if not summaries:
        raise ValueError("At least one completed evaluation is required")
    reference = summaries[0]
    for key in (*COMPARISON_KEYS, "evaluated_dataset_indices", "metrics", "models"):
        if key not in reference:
            raise ValueError(f"Incomplete evaluation: missing {key}")
    indices = reference["evaluated_dataset_indices"]
    if len(set(indices)) != len(indices) or len(indices) != reference["evaluated_images"]:
        raise ValueError("Invalid evaluated image indices/count")
    merged = dict(reference, metrics={}, models={})
    for summary in summaries:
        for key in COMPARISON_KEYS:
            if key not in summary or summary[key] != reference[key]:
                raise ValueError(f"Incompatible evaluations: {key}")
        if sorted(summary.get("evaluated_dataset_indices", [])) != sorted(indices):
            raise ValueError("Incompatible evaluations: evaluated_dataset_indices")
        if not summary.get("metrics") or set(summary["metrics"]) != set(summary.get("models", {})):
            raise ValueError("Incomplete evaluation: model/metric labels differ")
        for label, metrics in summary["metrics"].items():
            if label in merged["metrics"]:
                if metrics != merged["metrics"][label] or summary["models"][label] != merged["models"][label]:
                    raise ValueError(f"Conflicting model label: {label}; use unique labels")
            merged["metrics"][label] = metrics
            merged["models"][label] = summary["models"][label]
    return merged


def number(value):
    return "N/A" if value is None else f"{value:.4f}"


def escaped(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(summary, sources):
    lines = [
        "# 双侧 token 分割评估", "",
        f"- 数据：`{summary.get('data_root', 'unspecified')}`。",
        f"- 评估图片：{summary['evaluated_images']}；标注 SHA256：`{summary['annotations_sha256']}`。",
        f"- 检出阈值：{summary['detection_threshold']}；mask 阈值：{summary['mask_threshold']}。",
        f"- 分数定义：`{summary['confidence_definition']}`。", "",
        "## 汇总", "",
        "| 模型 | 正确侧 top-mask Dice ↑ | 阈值后 Dice ↑ | 正确侧检出率 ↑ | 对侧误检率 ↓ | 无手 prompt 误检率 ↓ | 左右分数择侧准确率 ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = (
        "correct_prompt_mean_top_dice", "correct_prompt_mean_thresholded_dice",
        "correct_prompt_detection_rate", "opposite_prompt_false_positive_rate",
        "empty_image_false_positive_rate", "visible_side_selection_accuracy",
    )
    for label, metrics in summary["metrics"].items():
        values = " | ".join(number(metrics.get(key)) for key in keys)
        lines.append(f"| {escaped(label)} | {values} |")
    lines += [
        "", "## 按实际手别", "",
        "| 模型 | 实际侧 | 正确侧样本数 | 正确侧 top-mask Dice | 阈值后 Dice | 对侧检出率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, metrics in summary["metrics"].items():
        slices = metrics.get("slices", {})
        for side, opposite in (("left_hand", "right_hand"), ("right_hand", "left_hand")):
            correct = slices.get(f"actual={side}|prompt={side}", {})
            wrong = slices.get(f"actual={side}|prompt={opposite}", {})
            lines.append(
                f"| {escaped(label)} | {side} | {correct.get('count', 'N/A')} | "
                f"{number(correct.get('mean_top_dice_with_physical_hand'))} | "
                f"{number(correct.get('mean_detected_dice_with_physical_hand'))} | "
                f"{number(wrong.get('detection_rate'))} |"
            )
    lines += [
        "", "## 解读边界", "",
        "- top-mask Dice 不考虑是否通过检出阈值；阈值后 Dice 将漏检计为 0。两者应一起看。",
        "- 无手误检率按左右 prompt 分别统计，不是“任一侧检出”的每图误检率。",
        "- 左右分数择侧准确率比较两个分数，不要求通过阈值；并列不记为正确。",
        "- 分割改善不能抵消未说明的对侧误检恶化；这里不自动宣称某模型全面优于基线。",
        "- 视频相邻帧相关、左右样本量不均衡；这些逐帧均值不是独立样本显著性检验。",
        "- 在 val 上确定模型与阈值；test 只作最终评估。prompt 二分类 AP 不是 COCO mask AP。",
        "- GT mask 仅用于计分；候选 mask 由模型自身分数选择。", "",
        "- 可比性校验覆盖基座路径，但旧评估摘要未记录基座文件 SHA256 和 AMP 配置；路径相同不能独立证明权重字节及数值配置完全一致，必要时结合运行快照核验。", "",
        "## 来源", "",
    ]
    for path, digest in sources:
        lines.append(f"- `{path}`，SHA256 `{digest}`。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, sources = [], []
    for path in args.summary:
        raw = path.read_bytes()
        summaries.append(json.loads(raw))
        sources.append((str(path.resolve()), hashlib.sha256(raw).hexdigest()))
    report = render_report(merge_summaries(summaries), sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects previous reports; choose a new name to rerun.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

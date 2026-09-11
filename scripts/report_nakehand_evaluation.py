#!/usr/bin/env python3
"""Create a checked Chinese Markdown report from one completed nakehand run.

CPU only. This does not combine runs, fit thresholds, rank candidates, or train.
Existing reports are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

if __package__:
    from . import evaluate_nakehand_tokens as evaluation
else:
    import evaluate_nakehand_tokens as evaluation


def read_json_with_hash(path: Path) -> tuple[object, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_run(summary_path: Path) -> tuple[dict, list[dict], dict]:
    summary, summary_hash = read_json_with_hash(summary_path)
    if summary.get("format") != "nakehand-frozen-bilateral-evaluation-v1" or summary.get("status") != "completed":
        raise ValueError("Require a completed nakehand frozen bilateral summary")
    for key in ("annotations_sha256", "base_checkpoint_sha256"):
        if not valid_sha(summary.get(key)):
            raise ValueError(f"Missing/invalid provenance hash: {key}")
    if summary.get("detection_threshold") != .5 or summary.get("mask_threshold") != .5:
        raise ValueError("Expected preregistered fixed detection/mask thresholds .5")
    if summary.get("thresholds_fitted_on_nakehand") is not False or summary.get("training_performed") is not False:
        raise ValueError("This report requires a no-training, no-test-calibration run")
    if summary.get("observed_identity_verified") is not True:
        raise ValueError("Observed loader identity was not verified")
    base = Path(summary["base_checkpoint"])
    if evaluation.shared.sha256(base) != summary["base_checkpoint_sha256"]:
        raise ValueError("Current base checkpoint SHA differs from evaluation")
    root = Path(summary["data_root"])
    annotation_path = root / "annotations.json"
    if evaluation.shared.sha256(annotation_path) != summary["annotations_sha256"]:
        raise ValueError("Current annotation SHA differs from evaluation")
    images, references, _ = evaluation.load_coco_index(root)
    if evaluation.shared.sha256(annotation_path) != summary["annotations_sha256"]:
        raise ValueError("Annotations changed during report validation")
    indices = summary.get("evaluated_dataset_indices")
    ids = summary.get("evaluated_image_ids")
    if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 or index >= len(images) for index in indices)):
        raise ValueError("Invalid evaluated dataset index selection")
    if ids != [int(images[index]["id"]) for index in indices] or len(indices) != summary.get("evaluated_images"):
        raise ValueError("Evaluated image IDs/count do not match annotation selection")
    selected = {int(images[index]["id"]): (index, images[index]) for index in indices}
    primary_count = sum(image["primary_test"] for _, image in selected.values())
    diagnostic_count = sum(bool(evaluation.diagnostic_ids(image)) for _, image in selected.values())
    if primary_count != summary.get("primary_test_images") or diagnostic_count != summary.get("diagnostic_images"):
        raise ValueError("Primary/diagnostic image counts differ from actual annotations")
    models = summary.get("models")
    if not isinstance(models, dict) or not models or set(models) != set(summary.get("metrics", {})):
        raise ValueError("Model metadata and metric labels differ")
    records = []
    record_sources = []
    for label, model in models.items():
        if evaluation.shared.safe_label(label) != label:
            raise ValueError("Unsafe model label")
        if model.get("kind") == "learned_class":
            if not valid_sha(model.get("checkpoint_sha256")):
                raise ValueError("Missing learned checkpoint SHA")
            if not isinstance(model.get("completed_epochs_from_steps"), int) or model["completed_epochs_from_steps"] < 2:
                raise ValueError("Learning token comparison must use at least two actually completed epochs")
            if evaluation.shared.sha256(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
                raise ValueError("Current token checkpoint SHA differs from evaluation")
        elif model.get("kind") != "ve":
            raise ValueError("Unknown model kind")
        path = summary_path.parent / "records" / f"{label}.json"
        rows, digest = read_json_with_hash(path)
        if not isinstance(rows, list) or len(rows) != 2 * len(indices):
            raise ValueError(f"Record image/query count mismatch: {label}")
        keys = set()
        for row in rows:
            image_id, side = row.get("image_id"), row.get("prompt_key")
            key = image_id, side
            if key in keys or image_id not in selected or side not in evaluation.CLASS_NAMES:
                raise ValueError(f"Duplicate/unknown image or query side: {label}")
            keys.add(key)
            index, image = selected[image_id]
            other = evaluation.CLASS_NAMES[1 - evaluation.CLASS_NAMES.index(side)]
            if row.get("model") != label or row.get("dataset_index") != index:
                raise ValueError("Record model/index differs from frozen selection")
            expected = {
                "observed_coco_image_id": image_id, "identity_verified": True,
                "primary_test": image["primary_test"], "diagnostic_ids": evaluation.diagnostic_ids(image),
                "recording_id": image["recording_id"], "view_type": evaluation.view_type(image),
                "file_name": image["file_name"],
                "target_present": bool(references[image_id][side].any()),
                "other_side_present": bool(references[image_id][other].any()),
                "reference_pixels": int(references[image_id][side].sum()),
                "other_reference_pixels": int(references[image_id][other].sum()),
            }
            for name, value in expected.items():
                if row.get(name) != value:
                    raise ValueError(f"Record/annotation mismatch: {name}, image {image_id}, {label}")
            for name in ("top_confidence", "top_class_probability", "presence_probability"):
                value = row.get(name)
                if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid probability: {name}")
            if not math.isclose(row["top_confidence"], row["top_class_probability"] * row["presence_probability"],
                                rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("Confidence differs from class probability times presence probability")
            if row.get("detected") != (row["top_confidence"] >= .5):
                raise ValueError("Record detection differs from fixed threshold")
        if keys != {(image_id, side) for image_id in ids for side in evaluation.CLASS_NAMES}:
            raise ValueError("Records do not contain exactly both side queries for every image")
        records.extend(rows)
        record_sources.append({"path": str(path.resolve()), "sha256": digest})
    recomputed = evaluation.summarize(records)
    if recomputed != summary["metrics"]:
        raise ValueError("Summary metrics differ from records; refusing a misleading report")
    sources = {"summary_path": str(summary_path.resolve()), "summary_sha256": summary_hash,
               "records": record_sources}
    return summary, records, sources


def number(value: float | None) -> str:
    return "N/A（无分母）" if value is None else f"{value:.4f}"


def proportion(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}（{numerator / denominator:.2%}）" if denominator else "0/0（N/A）"


def escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def link(label: str, path: str | Path) -> str:
    return f"[{label}](<{Path(path).resolve()}>)"


def segmentation_table(rows: Sequence[tuple[str, str, dict]]) -> list[str]:
    lines = ["| 模型 | 分组 | 可见侧 query 数 | 候选 Dice | 候选 IoU | 漏检计零 Dice | 检出数/可见数 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, group, value in rows:
        lines.append(f"| {escape(label)} | {escape(group)} | {value['present_queries']} | "
                     f"{number(value['present_mean_candidate_dice'])} | {number(value['present_mean_candidate_iou'])} | "
                     f"{number(value['present_mean_miss_zero_dice'])} | "
                     f"{proportion(value['true_positive_queries'], value['present_queries'])} |")
    return lines


def wrong_side_table(rows: Sequence[tuple[str, str, dict]]) -> list[str]:
    lines = [
        "| 模型 | 分组 | 候选错侧/全部双手查询 | 高分错侧/全部双手查询 | 高分错侧/已检出双手查询 | 两查询同时高分反向偏重/双手图片 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label, group, value in rows:
        candidate = value["both_visible_candidate_opposite_dominant_queries"]
        detected = value["both_visible_detected_opposite_dominant_queries"]
        lines.append(
            f"| {escape(label)} | {escape(group)} | {proportion(candidate, value['both_visible_queries'])} | "
            f"{proportion(detected, value['both_visible_queries'])} | "
            f"{proportion(detected, value['both_visible_detected_queries'])} | "
            f"{proportion(value['simultaneous_two_query_swap_proxy_images'], value['both_visible_complete_image_pairs'])} |"
        )
    return lines


def render_report(summary: dict, records: Sequence[dict], sources: dict) -> str:
    primary = summary["primary_test_images"]
    models = summary["models"]
    labels = [label for label in ("epoch2", "ve-underscore", "ve-natural") if label in models]
    labels += [label for label in models if label not in labels]
    metrics = summary["metrics"]
    lines = ["# nakehand：第二轮 token 与原始 text encoder 外部测试", "",
             f"已完成 {summary['evaluated_images']} 张不同图片、每图左右两个查询、{len(models)} 个模型/提示变体。",
             f"主测试样本 {primary} 张；人工认可诊断样本 {summary['diagnostic_images']} 张（可与主样本重合，不重复加入主平均）。", "",
             "这里衡量的是与现有 **SAM3 辅助生成参考 mask** 的一致性，不是全部经人工逐像素修订的独立真值。"
             "用户仅认可了 A/B/C 三张的可接受性；这不代表其他图片都已人工核验。", "",
             "本次不训练、不调整阈值。检测阈值和 mask 阈值均固定为 0.5；"
             "分数为 `sigmoid(class_logit) × sigmoid(presence_logit)`，候选 mask 由模型最高分决定，不按参考重叠程度挑选。", "",
             "下表的检出率只表示查询分数达到阈值，不保证 mask 对应正确的物理手；"
             "即使左右都“检出”，也可能分到了同一只手。请同时看同侧 Dice 和未筛候选的错侧重叠代理。", ""]
    if primary:
        lines += ["## 主测试结果", "",
                  "计划为每段录像均匀取 100 帧，共 6 段、600 帧；这是录像均衡抽样，"
                  "不是对全部 18,498 帧原始分布的无偏估计。表中总体为可见侧 query 微平均；录像宏平均另列。", ""]
        if primary != 600 or not summary.get("full_export_evaluated"):
            lines += [f"注意：本文件实际只有 {primary} 张主样本，不能称为完整 600 帧主测试结果。", ""]
        lines += segmentation_table([(label, "主样本总体", metrics[label]["primary_test"]["overall"]) for label in labels])
        lines += ["", "### 左右手与 ego/exo", ""]
        group_rows = []
        for label in labels:
            groups = metrics[label]["primary_test"]
            group_rows.extend((label, "左手" if side == "left_hand" else "右手", groups["per_side"][side])
                              for side in evaluation.CLASS_NAMES)
            group_rows.extend((label, name, value["overall"]) for name, value in groups["per_view_type"].items())
        lines += segmentation_table(group_rows)
        lines += ["", "### 缺席侧与无手误检", "",
                  "双手都在时，左右都是正查询。缺席侧指该侧参考为空，不会把另一只实际存在的手一律当成负样本。"
                  "“无手图”仅指左右参考都为空，并不保证经人工逐像素确认实际无人手；参考漏标也可能影响误检计数。", "",
                  "| 模型 | 全部缺席侧 FP/query | 单手图缺席侧 FP/query | 无手图 FP/query | 无手图任一侧检出/图片 |",
                  "|---|---:|---:|---:|---:|"]
        for label in labels:
            value = metrics[label]["primary_test"]["overall"]
            pairs = (("false_positive_queries", "absent_queries"),
                     ("single_hand_absent_false_positive_queries", "single_hand_absent_queries"),
                     ("empty_image_false_positive_queries", "empty_image_queries"),
                     ("empty_images_with_any_detection", "empty_images_with_both_query_results"))
            lines.append(f"| {escape(label)} | " + " | ".join(proportion(value[a], value[b]) for a, b in pairs) + " |")
        lines += ["", "### 双手图错侧重叠代理", "",
                  "此处只标记：候选与另一侧参考相交，且其 IoU 严格大于与自身参考的 IoU；并列不算。"
                  "它不是确定的解剖学左右互换，合并两只手的 mask 也可能触发。"
                  "候选列不经过置信度筛选，因此低分错侧不会被高分列的零值掩盖。", ""]
        lines += wrong_side_table([(label, "主样本双手图", metrics[label]["primary_test"]["overall"]) for label in labels])
        lines += ["", "### 录像宏平均", "",
                  "先在每段录像内求可见侧均值，再对有分母的录像等权平均；与上面的总体 query 微平均不是同一个量。", "",
                  "| 模型 | 候选 Dice | 漏检计零 Dice | 参与录像数 |", "|---|---:|---:|---:|"]
        for label in labels:
            macro = metrics[label]["primary_test"]["recording_macro_means"]
            candidate, detected = macro["present_mean_candidate_dice"], macro["present_mean_miss_zero_dice"]
            lines.append(f"| {escape(label)} | {number(candidate['value'])} | {number(detected['value'])} | "
                         f"{candidate['recordings_with_denominator']} |")
    else:
        lines += ["## 仅诊断冒烟，不是主测试结论", "",
                  "本文件没有主测试样本；以下 A/B/C 的表现只用于检查推理与可视化，不能推断整个数据集的效果。"]
    lines += ["", "## A/B/C 人工认可样例（独立列出）", ""]
    diagnostic_rows = []
    for name in sorted({name for label in labels for name in metrics[label]["diagnostic_examples"]}):
        for label in labels:
            if name in metrics[label]["diagnostic_examples"]:
                diagnostic_rows.append((label, name, metrics[label]["diagnostic_examples"][name]["overall"]))
    lines += segmentation_table(diagnostic_rows) if diagnostic_rows else ["本次评估未包含 A/B/C 样例。"]
    if diagnostic_rows:
        lines += ["", "### A/B/C 候选与高分错侧重叠代理", "",
                  "仅统计双手参考都可见的查询；定义为另一侧 IoU 严格更大且存在相交，并列不算。"
                  "候选列包含低分输出，高分列额外要求 score ≥ 0.5；这是重叠代理，不是确定的解剖学错手。", ""]
        lines += wrong_side_table(diagnostic_rows)
    visuals = summary.get("visuals", [])
    for name in sorted({name for row in visuals for name in row.get("diagnostic_ids", [])}):
        entries = [row for row in visuals if name in row.get("diagnostic_ids", [])]
        for row in entries:
            directory = Path(row["directory"])
            lines += ["", f"### 样例 {escape(name)}", "",
                      " · ".join((link("分离对比图", row["comparison"]), link("原图", directory / "rgb.png"),
                                  link("左手参考", directory / "left_hand__reference.png"),
                                  link("右手参考", directory / "right_hand__reference.png")))]
            lines.append("")
            for label in labels:
                model_dir = directory / label
                lines.append(f"- {escape(label)}：" + " · ".join((
                    link("左手预测", model_dir / "left_hand__detected.png"),
                    link("右手预测", model_dir / "right_hand__detected.png"),
                    link("左手未筛候选", model_dir / "left_hand__candidate.png"),
                    link("右手未筛候选", model_dir / "right_hand__candidate.png"))))
    lines += ["", "## 如何解释", "",
              "- 候选 Dice/IoU 只在该侧参考存在时平均；低于检测阈值的候选仍可有较好形状。漏检计零 Dice 同时惩罚漏检。",
              "- 检出率仅表示该侧查询的置信度达到阈值，不等同于 IoU 匹配后的目标检出率；空参考不参与正样本 Dice 平均。",
              "- 阈值固定有利于复现，但不同 text encoder 的分数可能校准不同；本测试不据此重新调阈值。",
              "- 参考来自 SAM3 辅助传播，对比存在模型家族相关性；与参考更一致不自动证明真实边界更准确。可用人工复核检查分歧。",
              "- 相邻视频帧相关，6 段录像不等于 6 个不同受试者；这里没有跨人泛化、独立样本显著性或完整像素人工真值的保证。",
              "- 未使用 MANO 或 geometry 微调；模型处于 eval 模式且禁用参考诱导的交互点/框。", "",
              "## 核验与来源", "",
              f"- {link('完整结果 JSON', sources['summary_path'])}；SHA256 `{sources['summary_sha256']}`。",
              f"- annotation SHA256：`{summary['annotations_sha256']}`。",
              f"- 当前共用 base checkpoint SHA256：`{summary['base_checkpoint_sha256']}`。",
              "- 报告生成时复核原始参考、实际图片 ID/双查询覆盖、主/诊断分组、当前 checkpoint 哈希，并从逐查询 records 重算摘要一致。",
              "- 当前基座哈希可以核验本次各模型的共同权重；旧训练 checkpoint 未记录历史基座字节哈希，因此不能仅凭路径证明训练时字节完全相同。"]
    for source in sources["records"]:
        lines.append(f"- {link(Path(source['path']).stem + ' 逐查询记录', source['path'])}；SHA256 `{source['sha256']}`。")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    summary, records, sources = validate_run(args.summary)
    report = render_report(summary, records, sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

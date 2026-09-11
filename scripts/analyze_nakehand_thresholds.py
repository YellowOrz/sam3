#!/usr/bin/env python3
"""CPU-only, same-validation threshold diagnostics for bilateral nakehand.

Reads the completed three-variant evaluation without changing its candidates,
mask threshold, model, or deployment threshold. Unlike the DexYCB diagnostic,
each frame may have both hands, one hand, or neither annotated hand.
"Absent" means reference-absent, NOT a verified absence of a real visible hand.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from scripts import evaluate_nakehand_tokens as metrics
from scripts import report_nakehand_semantic_ablation as checked_report

SIDES = ("left_hand", "right_hand")
LABELS = checked_report.LABELS
BUDGETS = ("0", "0.01", "0.05")
IDENTITY_FIELDS = (
    "dataset_index", "image_id", "file_name", "recording_id", "view_type",
    "source_frame_index", "video_pts_seconds", "source_mapping", "prompt_key",
    "reference_pixels", "other_reference_pixels", "target_present",
    "other_side_present", "any_hand_present", "both_hands_present",
)


def probability(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid probability: {name}")
    return float(value)


def validate_bilateral_rows(rows):
    """Check independent left/right references, metric consistency, and pairing."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Require nonempty bilateral records")
    pairs, keys, labels = defaultdict(dict), set(), set()
    for row in rows:
        image_id, side = row["image_id"], row["prompt_key"]
        if type(image_id) is not int or side not in SIDES:
            raise ValueError("Invalid image/prompt identity")
        key = (image_id, side)
        if key in keys:
            raise ValueError("Duplicate image/prompt identity")
        keys.add(key)
        labels.add(row["model"])
        pairs[image_id][side] = row
        for field in ("dataset_index", "source_frame_index", "reference_pixels",
                      "other_reference_pixels", "top_mask_pixels", "selected_decoder_query",
                      "top_other_reference_intersection_pixels"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"Invalid nonnegative integer: {field}")
        if row["selected_decoder_query"] >= 200:
            raise ValueError("Candidate index exceeds this evaluation's 200-query decoder")
        for field in ("target_present", "other_side_present", "any_hand_present", "both_hands_present"):
            if type(row[field]) is not bool:
                raise ValueError(f"Presence must be boolean: {field}")
        present, other = row["target_present"], row["other_side_present"]
        if (present != (row["reference_pixels"] > 0)
                or other != (row["other_reference_pixels"] > 0)
                or row["any_hand_present"] != (present or other)
                or row["both_hands_present"] != (present and other)):
            raise ValueError("Inconsistent bilateral presence labels")
        score = probability(row["top_confidence"], "top_confidence")
        for prefix in ("own", "other"):
            dice = probability(row[f"top_dice_with_{prefix}_reference"], "candidate Dice")
            iou = probability(row[f"top_iou_with_{prefix}_reference"], "candidate IoU")
            area = row["reference_pixels" if prefix == "own" else "other_reference_pixels"]
            intersection = dice * (area + row["top_mask_pixels"]) / 2
            if (not math.isclose(dice, 2 * iou / (1 + iou), abs_tol=1e-10)
                    or not math.isclose(intersection, round(intersection), abs_tol=1e-6)
                    or not 0 <= round(intersection) <= min(area, row["top_mask_pixels"])):
                raise ValueError("Inconsistent candidate overlap metrics")
            if prefix == "other" and round(intersection) != row["top_other_reference_intersection_pixels"]:
                raise ValueError("Other-reference intersection disagrees with Dice")
        for suffix in ("dice", "iou"):
            candidate = row[f"top_{suffix}_with_own_reference"] if present else None
            thresholded = (candidate if score >= .5 else 0.) if present else None
            if row[f"top_{suffix}"] != candidate or row[f"miss_zero_{suffix}"] != thresholded:
                raise ValueError("Saved positive/miss-zero metric inconsistent with fixed candidate")
        proxy = (other and row["top_other_reference_intersection_pixels"] > 0
                 and row["top_iou_with_other_reference"] > row["top_iou_with_own_reference"])
        if (row["opposite_overlap_dominant_proxy"] is not proxy
                or row["detected_opposite_overlap_dominant_proxy"] is not (proxy and score >= .5)
                or row["detected"] is not (score >= .5)
                or row["detected_mask_pixels"] != (row["top_mask_pixels"] if score >= .5 else 0)):
            raise ValueError("Saved detection or wrong-side proxy inconsistent")
    if len(labels) != 1:
        raise ValueError("Each record set must contain exactly one model")
    image_fields = ("dataset_index", "file_name", "recording_id", "view_type", "source_frame_index",
                    "video_pts_seconds", "source_mapping", "any_hand_present", "both_hands_present")
    for pair in pairs.values():
        if set(pair) != set(SIDES):
            raise ValueError("Every image requires exactly two independent hand queries")
        left, right = (pair[side] for side in SIDES)
        if (any(left[field] != right[field] for field in image_fields)
                or left["reference_pixels"] != right["other_reference_pixels"]
                or right["reference_pixels"] != left["other_reference_pixels"]):
            raise ValueError("Paired prompts disagree on image or left/right references")
    indices = [pair[SIDES[0]]["dataset_index"] for pair in pairs.values()]
    if len(set(indices)) != len(indices):
        raise ValueError("Repeated dataset index across image identities")
    return {key: tuple(row[field] for field in IDENTITY_FIELDS)
            for key in keys for row in (pairs[key[0]][key[1]],)}


def select_threshold(rows, budget):
    """Max TPR under all-absent-query FP budget, without consulting mask Dice."""
    cap = Fraction(str(budget))
    if not 0 <= cap <= 1:
        raise ValueError("FPR budget must be between zero and one")
    absent = sum(not row["target_present"] for row in rows)
    positive = len(rows) - absent
    if not absent or not positive:
        raise ValueError("Threshold fitting requires positive and absent-side queries")
    allowed = cap.numerator * absent // cap.denominator  # Exact integer floor, no float tolerance.
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        groups[probability(row["top_confidence"], "score")][not row["target_present"]] += 1
    best = (math.nextafter(max(groups), math.inf), 0, 0)
    tp = fp = 0
    for score in sorted(groups, reverse=True):
        add_tp, add_fp = groups[score]
        tp += add_tp
        fp += add_fp
        if fp > allowed:
            break
        if (tp, -fp, score) > (best[1], -best[2], best[0]):
            best = (score, tp, fp)
    return {"budget": str(budget), "allowed_false_positive_queries": allowed,
            "absent_query_denominator": absent, "threshold": best[0],
            "reject_all_above_one": best[0] > 1,
            "fit_true_positive_queries": best[1], "fit_false_positive_queries": best[2]}


def rethreshold(rows, threshold):
    """Only gate the existing score-selected candidate; no reranking or mask edit."""
    result = []
    for row in rows:
        detected = row["top_confidence"] >= threshold
        revised = dict(row, detected=detected,
                       detected_mask_pixels=row["top_mask_pixels"] if detected else 0,
                       detected_opposite_overlap_dominant_proxy=bool(
                           detected and row["opposite_overlap_dominant_proxy"]))
        for suffix in ("dice", "iou"):
            revised[f"miss_zero_{suffix}"] = ((row[f"top_{suffix}"] if detected else 0.)
                                               if row["target_present"] else None)
        result.append(revised)
    return result


def contiguous_ranges(rows):
    """Deduplicate frames, then report temporal concentration, not independent N."""
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["recording_id"]].add(row["source_frame_index"])
    ranges = []
    for recording, indices in sorted(grouped.items()):
        run = []
        for index in sorted(indices):
            if run and index != run[-1] + 1:
                ranges.append({"recording_id": recording, "first_frame": run[0],
                               "last_frame": run[-1], "frames": len(run)})
                run = []
            run.append(index)
        if run:
            ranges.append({"recording_id": recording, "first_frame": run[0],
                           "last_frame": run[-1], "frames": len(run)})
    total = sum(item["frames"] for item in ranges)
    ranked = sorted(ranges, key=lambda item: (-item["frames"], item["recording_id"], item["first_frame"]))
    return {"unique_frames": total, "recordings": len(grouped), "contiguous_runs": len(ranges),
            "largest_run_fraction": ranked[0]["frames"] / total if total else None,
            "top_10_runs": ranked[:10], "all_runs_shown": len(ranges) <= 10}


def negative_concentration(rows, detected_only=False):
    selected = [row for row in rows if not row["target_present"]
                and (not detected_only or row["detected"])]
    result = {}
    subsets = {"all_absent_queries": selected,
               "empty_frame_queries": [row for row in selected if not row["any_hand_present"]],
               "single_hand_opposite_queries": [row for row in selected if row["other_side_present"]]}
    for name, subset in subsets.items():
        result[name] = {"queries": len(subset), **contiguous_ranges(subset),
                        "per_prompt": {side: {"queries": sum(row["prompt_key"] == side for row in subset),
                                               **contiguous_ranges([row for row in subset if row["prompt_key"] == side])}
                                       for side in SIDES}}
    return result


def analyze_rows(rows):
    validate_bilateral_rows(rows)
    points = []
    specifications = [{"name": "default_0.5", "threshold": .5, "threshold_fitted": False}]
    specifications += [{"name": f"absent_fpr_{budget}", "threshold_fitted": True,
                        **select_threshold(rows, budget)} for budget in BUDGETS]
    for spec in specifications:
        revised = rethreshold(rows, spec["threshold"])
        overall = metrics.aggregate(revised)
        if spec["threshold_fitted"] and (
                overall["false_positive_queries"] != spec["fit_false_positive_queries"]
                or overall["true_positive_queries"] != spec["fit_true_positive_queries"]):
            raise RuntimeError("Threshold sweep and independent aggregate disagree")
        points.append({**spec, "overall": overall,
                       "per_prompt": {side: metrics.aggregate([row for row in revised if row["prompt_key"] == side])
                                      for side in SIDES},
                       "false_positive_concentration": negative_concentration(revised, detected_only=True)})
    return {"queries": len(rows), "unique_scores": len({row["top_confidence"] for row in rows}),
            "operating_points": points}


def analyze_summary(path):
    fingerprints = {}

    def fingerprint(source, expected=None):
        source = Path(source).resolve()
        digest = checked_report.file_hash(source)
        if expected is not None and digest != expected:
            raise ValueError(f"Input SHA256 mismatch: {source}")
        if str(source) in fingerprints and fingerprints[str(source)] != digest:
            raise RuntimeError(f"Input changed: {source}")
        fingerprints[str(source)] = digest
        return digest

    def read(source):
        source = Path(source).resolve()
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(source) in fingerprints and fingerprints[str(source)] != digest:
            raise RuntimeError(f"Input changed: {source}")
        fingerprints[str(source)] = digest
        return json.loads(raw)

    source = Path(path).resolve()
    summary = read(source)
    if summary.get("candidate_selection") != "highest combined model confidence, never reference overlap":
        raise ValueError("Require original score-argmax candidate selection")
    records = {label: read(source.parent / "records" / f"{label}.json") for label in LABELS}
    checked_report.validate_summary(summary, records)  # All 3449 x 2 x 3, .5 metrics recomputed.
    annotation_path = Path(summary["data_root"]) / "annotations.json"
    annotations = read(annotation_path)
    if (fingerprints[str(annotation_path.resolve())] != summary["annotations_sha256"]
            or annotations.get("info", {}).get("split") != "val"
            or annotations["info"].get("dataset_role") != "validation"
            or {row["id"]: row["name"] for row in annotations["categories"]} != {1: SIDES[0], 2: SIDES[1]}):
        raise ValueError("Require unchanged bilateral validation annotations")
    images = sorted(annotations["images"], key=lambda row: row["id"])
    image_lookup = {row["id"]: row for row in images}
    if (len(images) != 3449 or len(image_lookup) != 3449
            or summary["evaluated_image_ids"] != [row["id"] for row in images]
            or summary["evaluated_dataset_indices"] != list(range(3449))):
        raise ValueError("Full validation image identities do not match annotations")
    areas = defaultdict(dict)
    annotation_ids = set()
    for annotation in annotations["annotations"]:
        image_id, category, area = annotation["image_id"], annotation["category_id"], annotation["area"]
        if (annotation["id"] in annotation_ids or image_id not in image_lookup or category not in (1, 2)
                or category in areas[image_id] or type(area) not in (int, float) or not math.isfinite(area)
                or area <= 0):
            raise ValueError("Expected this frozen export's one nonempty reference per side")
        annotation_ids.add(annotation["id"])
        areas[image_id][category] = area
    identities = None
    for label, rows in records.items():
        actual = validate_bilateral_rows(rows)
        if identities is not None and actual != identities:
            raise ValueError("Models disagree on image/prompt/reference identities")
        identities = actual
        for row in rows:
            image = images[row["dataset_index"]]
            category = SIDES.index(row["prompt_key"]) + 1
            if (row["image_id"] != image["id"]
                    or any(row[field] != image[field] for field in
                           ("file_name", "recording_id", "view_type", "source_frame_index", "video_pts_seconds"))
                    or row["reference_pixels"] != areas[image["id"]].get(category, 0)
                    or row["other_reference_pixels"] != areas[image["id"]].get(3 - category, 0)):
                raise ValueError("Record identity/reference does not match frozen annotations")
    # Check the small historical inference code snapshots, not a new forward pass.
    for item in summary["code_snapshots"]:
        fingerprint(item["snapshot"], item["sha256"])
    fingerprint(Path(__file__))
    fingerprint(Path(metrics.__file__))
    fingerprint(Path(checked_report.__file__))
    result = {
        "format": "sam3-nakehand-same-val-threshold-diagnostics-v1",
        "dataset_role": "validation", "evaluated_images": 3449,
        "queries_per_model": 6898, "models_compared": len(LABELS),
        "annotations_sha256": summary["annotations_sha256"],
        "base_checkpoint_sha256_as_recorded_by_evaluation": summary["base_checkpoint_sha256"],
        "model_checkpoint_sha256_as_recorded_by_evaluation": {
            label: summary["models"][label]["checkpoint_sha256"] for label in LABELS},
        "candidate_selection": summary["candidate_selection"],
        "candidate_reselection_performed": False, "mask_threshold": .5,
        "negative_label_semantics": "Reference-absent output rate / relative-to-label FP, not verified real false positive",
        "thresholds_fitted_on_and_measured_on_same_validation": True,
        "production_threshold_changed": False, "training_performed": False,
        "selection_objective": "Maximize present-query detection under ALL absent-query FPR budget; not optimize mask Dice",
        "tie_policy": "score >= threshold; equal scores inseparable; equal TP chooses fewer FP then higher threshold",
        "budget_policy": "floor(exact decimal rational * absent query count); 0%, 1%, 5%",
        "reference_frame_concentration": negative_concentration(records[LABELS[0]]),
        "models": {label: analyze_rows(records[label]) for label in LABELS},
        "limitations": [
            "Same-val fitting is optimistic development analysis, not independent generalization or a deployment guarantee.",
            "All 3449 frames are one ego recording; adjacent frames and absent-query episodes are correlated.",
            "User confirmed frame 0 / image 4713 is the physical RIGHT hand despite two empty references; both delta variants select it for LEFT, a wrong-side error plus missing labels, not a vindicated true positive.",
            "SAM3-assisted references and 13 accepted examples do not establish full independent manual ground truth.",
            "Detection means crossing a score threshold, not independently verified physical-side correctness.",
            "All-absent FPR constrains the combined denominator; neither prompt nor negative subgroup has its own guarantee.",
            "Mask overlaps are checked algebraically and aggregated from saved candidate metrics, not decoded again from pixels.",
            "Only the argmax candidate was saved; selection is inherited from hashed evaluator code, not re-proven from all 200 logits.",
            "Base/training checkpoints were not loaded or freshly rehashed by this CPU diagnostic; recorded evaluation hashes identify them.",
        ],
    }
    for input_path, expected in list(fingerprints.items()):
        fingerprint(input_path, expected)
    result["input_files"] = [{"path": path, "sha256": sha} for path, sha in sorted(fingerprints.items())]
    result["inputs_rechecked_unchanged"] = True
    return result


def fraction(n, d):
    return f"{n}/{d} ({n / d:.2%})" if d else "0/0 (N/A)"


def render_markdown(result, json_path):
    names = dict(zip(LABELS, ("原始 VE 缓存", "语义 delta（无 anchor）", "语义 delta（anchor=1）")))
    lines = ["# nakehand 阈值诊断：参考缺席输出率与检出率的关系", "",
             "日期：2026-09-11。只读 CPU 分析已完成的三模型全量验证 records；没有重新训练、重新选 mask 或修改默认阈值。", "",
             "**重要缺标警示：`image_id=4713 / source_frame_index=0` 的 RGB 左下角可见一只操作鼠标的手，但左右参考 mask 均空。"
             "用户已明确确认这是人体右手；两种 delta 却用 `left_hand` 分割它，属于错侧，同时原双空参考也有漏标。"
             "因此不能把该预测解释成仅因漏标被冤枉的正确预测。以下 FP 仍一律是“相对现有冻结参考标签的 FP / reference-absent 输出”，"
             "不等同全部样本已逐帧人工确认的真实误检。不能只用提高阈值替代左右手纠错，也不将第 0 帧人审外推给其余帧。**", "",
             "[第 0 帧右手人审与冻结指标边界](/home/zhengyuxi/projects/sam3-yelloworz/docs/nakehand-frame0-right-hand-review-2026-09-11.md)", "",
             "[该帧 RGB、左右参考与三模型分离对比](/home/zhengyuxi/datasets/sam3-nakehand-experiments/semantic-anchor-validation-recovery-20260910-2249/validation/visuals/image-004713/comparison.png)", "",
             "## 数据与判定方法", "",
             "同一开发验证录像 ego/20260907_142020 的 3449 帧，每帧独立查询左右手，共每模型 6898 条、三模型 20694 条。"
             "按参考标签，双手图是两个正查询，单手图才有一个对侧负查询；双侧参考均空有两个负查询，但不等于真实无手。所有标注/图像身份与跨模型配对均校验，默认指标从 records 重算并与原摘要逐项一致。", "",
             "阈值依据 `score = sigmoid(class_logit) × sigmoid(presence_logit)`；使用历史推理按 score argmax 选出的固定候选，mask 阈值始终为 0.5。"
             "同分候选查询整组纳入，`score ≥ threshold`；在全部缺席侧 FPR 预算内优先最大化正查询检出数，同检出数时选更少 FP、再选更高阈值，不以 GT Dice 选择预测或阈值。", "",
             "这里拟合阈值与测量结果使用**同一个 val**，只能用来诊断，不是独立测试、不代表部署保证，未把任何新阈值设为默认。"
             "参考 mask 来自 SAM3 辅助传播；13 张人工认可不能推广成全量独立人工 GT。", "",
             "## 全部参考缺席侧输出率预算下的结果", "",
             "候选 Dice 不依赖检测阈值；漏检计零 Dice 是正查询中低于阈值的候选计 0 后再平均。检出不是解剖学左右手正确性的人工认证。", "",
             "| 模型 | 标签 FP 预算 | 阈值 | 正查询检出 | 候选 Dice | 漏检计零 Dice | 参考缺席输出 | 单侧参考图的对侧输出 | 双侧参考空图的输出 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        for point in result["models"][label]["operating_points"]:
            row = point["overall"]
            condition = "默认 0.5" if not point["threshold_fitted"] else f"FPR≤{float(point['budget']):.0%}"
            lines.append(f"| {names[label]} | {condition} | {point['threshold']:.9g} | "
                         f"{fraction(row['true_positive_queries'], row['present_queries'])} | "
                         f"{row['present_mean_candidate_dice']:.6f} | {row['present_mean_miss_zero_dice']:.6f} | "
                         f"{fraction(row['false_positive_queries'], row['absent_queries'])} | "
                         f"{fraction(row['single_hand_absent_false_positive_queries'], row['single_hand_absent_queries'])} | "
                         f"{fraction(row['empty_image_false_positive_queries'], row['empty_image_queries'])} |")
    lines += ["", "预算分母是全部 496 个缺席侧查询：0%、1%、5% 分别最多允许 0、4、24 次 FP，采用精确有理数计算向下取整。"
              "该合并预算不单独保证左 prompt、右 prompt、单手对侧或空图群体各自满足相同 FPR。", "",
              "## 同一阈值按 prompt 细分（不另拟合左右阈值）", "",
              "| 模型 | 标签 FP 预算 | prompt | 正检出 | 漏检计零 Dice | 参考缺席输出 |", "|---|---|---|---:|---:|---:|"]
    for label in LABELS:
        for point in result["models"][label]["operating_points"]:
            condition = "默认" if not point["threshold_fitted"] else f"≤{float(point['budget']):.0%}"
            for side in SIDES:
                row = point["per_prompt"][side]
                lines.append(f"| {names[label]} | {condition} | {side} | "
                             f"{fraction(row['true_positive_queries'], row['present_queries'])} | "
                             f"{row['present_mean_miss_zero_dice']:.6f} | "
                             f"{fraction(row['false_positive_queries'], row['absent_queries'])} |")
    lines += ["", "## 负样本不是 496 个独立场景", ""]
    for name, title in (("all_absent_queries", "全部缺席侧"), ("empty_frame_queries", "双侧参考均空的图"),
                        ("single_hand_opposite_queries", "单手图中的对侧查询")):
        group = result["reference_frame_concentration"][name]
        lines.append(f"- {title}：{group['queries']} 个查询，来自 {group['unique_frames']} 帧 / "
                     f"{group['contiguous_runs']} 段连续帧；最大连续段占这些帧的 {group['largest_run_fraction']:.2%}。")
        spans = [f"{item['first_frame']}–{item['last_frame']} ({item['frames']} 帧)" for item in group["top_10_runs"]]
        lines.append("  按长度排列的前十段（源视频帧号从 0 开始）：" + "；".join(spans) + "。")
    lines += ["", "JSON 还分别保存各模型/阈值下相对标签 FP 的双侧参考空图、单侧参考图对侧、左右 prompt 时间聚集情况，以及双侧参考空图按帧“任一手报出”的计数。"
              "因此不能把这些相邻帧当独立样本给出统计显著性，0/496 也不表示以后永不误检；未复核缺标前更不能把降低这个值本身当作目标。", "",
              "## 可解释范围和复现", "",
              "这次从记录中的候选 Dice/IoU、像素面积及检测分数重新计算聚合指标，检查重叠代数、双侧存在关系、COCO 面积和源帧身份。"
              "没有重新解码全部预测 mask，也没有重新计算 200 个候选 logits；固定 argmax 规则由已哈希的历史推理代码和摘要建立。"
              "输入摘要、三份 records、COCO 标注、六个历史代码快照和分析依赖均做 SHA256 及前后不变检查；基座/训练 checkpoint 的 SHA 引用已完成评估的记录，本 CPU 脚本不重新加载它们。", "",
              f"[机器可读结果与全部输入 SHA256]({json_path})", "",
              "复现（必须写到尚不存在的新输出路径）：", "", "```bash",
              "OMP_NUM_THREADS=2 /home/zhengyuxi/.conda/envs/sam3-tokens/bin/python -m scripts.analyze_nakehand_thresholds \\",
              "  --summary /home/zhengyuxi/datasets/sam3-nakehand-experiments/semantic-anchor-validation-recovery-20260910-2249/validation/summary.json \\",
              "  --output-json /tmp/nakehand-threshold-recheck.json \\",
              "  --output-md /tmp/nakehand-threshold-recheck.md", "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args(argv)
    outputs = (args.output_json.resolve(), args.output_md.resolve())
    if outputs[0] == outputs[1] or any(path.exists() for path in outputs):
        raise FileExistsError("Require distinct new output files; never overwrite an input or previous result")
    result = analyze_summary(args.summary)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with outputs[1].open("x", encoding="utf-8") as stream:
        stream.write(render_markdown(result, outputs[0]))
    print(json.dumps({"outputs": [str(path) for path in outputs],
                      "images": result["evaluated_images"], "queries_per_model": result["queries_per_model"],
                      "inputs_rechecked_unchanged": result["inputs_rechecked_unchanged"]}))


if __name__ == "__main__":
    main()

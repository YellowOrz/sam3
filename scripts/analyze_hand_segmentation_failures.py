#!/usr/bin/env python3
"""CPU error decomposition of already-selected nakehand prediction PNGs.

No inference, candidate selection, threshold fitting, or source writes. References
are checked against source COCO RLE. This analyzes the saved visualization subset,
not all evaluated images. Outside-reference pixels are NOT anatomical forearm GT.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_erosion


SIDES = ("left_hand", "right_hand")
BOUNDARY_SOURCE = "https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py"
BOUNDARY_PAPER = "https://arxiv.org/abs/2103.16562"


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def mean_defined(values):
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def boolean_mask(value):
    array = np.asarray(value)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("Require a nonzero-size, two-dimensional binary mask")
    if not np.all(np.isin(array, (0, 1, 255))):
        raise ValueError("Mask must contain only binary values 0/1/255")
    return array != 0


def boundary_width(shape, boundary_ratio=.02):
    if not math.isfinite(boundary_ratio) or not 0 < boundary_ratio <= 1:
        raise ValueError("boundary_ratio must be finite and in (0, 1]")
    return max(1, int(round(boundary_ratio * math.hypot(*shape))))


def inner_boundary(mask, boundary_ratio=.02):
    """Inner band using the authors' 3x3 erosion and image-diagonal rule.

    SciPy binary erosion with outside value zero includes image-edge boundaries.
    This implements the mask-band operation, not the COCO Boundary AP protocol.
    See BOUNDARY_SOURCE; empty-mask conventions below are explicitly ours.
    """
    binary = boolean_mask(mask)
    width = boundary_width(binary.shape, boundary_ratio)
    interior = binary_erosion(binary, structure=np.ones((3, 3), dtype=bool),
                              iterations=width, border_value=0)
    return binary & ~interior


def measure_mask(prediction, own_reference, other_reference, boundary_ratio=.02):
    prediction, own, other = map(boolean_mask, (prediction, own_reference, other_reference))
    if prediction.shape != own.shape or own.shape != other.shape:
        raise ValueError("Prediction and both references must have identical shape")
    predicted, own_area, other_area = (int(value.sum()) for value in (prediction, own, other))
    intersection = int((prediction & own).sum())
    other_only = other & ~own
    other_only_area = int(other_only.sum())
    opposite = int((prediction & other_only).sum())
    other_intersection = int((prediction & other).sum())
    outside = int((prediction & ~(own | other)).sum())
    missed = own_area - intersection
    assert intersection + opposite + outside == predicted
    own_iou = ratio(intersection, predicted + own_area - intersection)
    other_iou = ratio(other_intersection, predicted + other_area - other_intersection)
    pred_boundary = inner_boundary(prediction, boundary_ratio)
    own_boundary = inner_boundary(own, boundary_ratio)
    boundary_intersection = int((pred_boundary & own_boundary).sum())
    boundary_union = int((pred_boundary | own_boundary).sum())
    return {
        "prediction_pixels": predicted, "own_reference_pixels": own_area,
        "other_reference_pixels": other_area, "other_only_reference_pixels": other_only_area,
        "reference_overlap_pixels": int((own & other).sum()),
        "own_intersection_pixels": intersection, "own_missed_pixels": missed,
        "other_only_intersection_pixels": opposite,
        "other_intersection_pixels": other_intersection,
        "outside_both_references_pixels": outside,
        "own_false_positive_pixels": predicted - intersection,
        "own_dice": ratio(2 * intersection, predicted + own_area), "own_iou": own_iou,
        "own_precision": ratio(intersection, predicted), "own_recall": ratio(intersection, own_area),
        "own_missed_fraction": ratio(missed, own_area),
        "other_only_reference_coverage": ratio(opposite, other_only_area),
        "other_only_prediction_fraction": ratio(opposite, predicted),
        "outside_both_prediction_fraction": ratio(outside, predicted),
        "outside_both_pixels_per_own_reference_pixel": ratio(outside, own_area),
        "opposite_overlap_dominant_proxy": bool(other_intersection and
            other_iou is not None and own_iou is not None and other_iou > own_iou),
        "boundary_band_pixels": boundary_width(prediction.shape, boundary_ratio),
        "boundary_intersection_pixels": boundary_intersection,
        "boundary_union_pixels": boundary_union,
        "own_boundary_iou": ratio(boundary_intersection, boundary_union),
    }


class TrackedInputs:
    """Hash exactly the input bytes decoded, then recheck before publishing."""

    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f"Source changed during analysis: {path}")
        self.hashes[str(path)] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def png(self, path, *, binary=True):
        with Image.open(BytesIO(self.read(path))) as image:
            if image.format != "PNG":
                raise ValueError(f"Require a PNG: {path}")
            array = np.asarray(image if binary else image.convert("RGB"))
        return boolean_mask(array) if binary else array

    def verify(self):
        for filename, expected in self.hashes.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source changed during analysis: {filename}")


def checked_child(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input path escapes its declared root")
    return path


def decode_rle(segmentation, shape):
    if not isinstance(segmentation, dict) or list(segmentation.get("size", [])) != list(shape):
        raise ValueError("Require an RLE with the original image shape")
    rle = dict(segmentation)
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, *shape)
    elif isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.shape != shape:
        raise ValueError("Decoded RLE shape mismatch")
    return boolean_mask(decoded)


def validate_candidate_record(row, prediction, own, other):
    """Check saved candidate against existing score-selected record; never reselect."""
    metrics = measure_mask(prediction, own, other)
    expected = {
        "top_mask_pixels": metrics["prediction_pixels"],
        "reference_pixels": metrics["own_reference_pixels"],
        "other_reference_pixels": metrics["other_reference_pixels"],
        "top_other_reference_intersection_pixels": metrics["other_intersection_pixels"],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Saved candidate disagrees with evaluation record: {key}")
    # Existing evaluator assigns empty/empty Dice and IoU zero; analysis uses null.
    for key, value in (("top_dice_with_own_reference", metrics["own_dice"]),
                       ("top_iou_with_own_reference", metrics["own_iou"])):
        if not math.isclose(row.get(key, math.nan), value if value is not None else 0.,
                            rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Saved candidate disagrees with evaluation record: {key}")


def analyze_saved_visuals(summary_path, boundary_ratio=.02):
    boundary_width((1, 1), boundary_ratio)
    summary_path = Path(summary_path).resolve()
    tracked = TrackedInputs()
    summary = tracked.json(summary_path)
    if summary.get("format") != "nakehand-frozen-bilateral-evaluation-v1" or summary.get("status") != "completed":
        raise ValueError("Require a completed frozen nakehand evaluation summary")
    if summary.get("observed_identity_verified") is not True:
        raise ValueError("Source evaluation did not verify image/query identity")
    if (summary.get("detection_threshold") != .5 or summary.get("mask_threshold") != .5
            or summary.get("thresholds_fitted_on_nakehand") is not False):
        raise ValueError("Require original fixed thresholds .5 and no nakehand calibration")
    if "argmax" not in summary.get("candidate_selection", "") or "never reference overlap" not in summary["candidate_selection"]:
        raise ValueError("Source must declare score-only candidate selection")
    data_root = Path(summary["data_root"]).resolve()
    annotation_path = data_root / "annotations.json"
    coco = tracked.json(annotation_path)
    if tracked.hashes[str(annotation_path)] != summary.get("annotations_sha256"):
        raise ValueError("COCO annotation hash differs from evaluation")
    if {item["id"]: item["name"] for item in coco["categories"]} != {1: SIDES[0], 2: SIDES[1]}:
        raise ValueError("Require explicit left/right COCO categories 1/2")
    images = {image["id"]: image for image in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate COCO image IDs")
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in images or annotation["category_id"] not in (1, 2):
            raise ValueError("Unknown annotation image/category")
        by_image[annotation["image_id"]].append(annotation)
    visuals_root = summary_path.parent / "visuals"
    manifest = tracked.json(visuals_root / "manifest.json")
    if not manifest or manifest != summary.get("visuals"):
        raise ValueError("Visualization manifest differs from frozen summary")
    selected_ids = [item["image_id"] for item in manifest]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Duplicate visualization image IDs")
    evaluated_ids, evaluated_indices = summary["evaluated_image_ids"], summary["evaluated_dataset_indices"]
    if (len(evaluated_ids) != summary["evaluated_images"] or len(evaluated_ids) != len(evaluated_indices)
            or len(set(evaluated_ids)) != len(evaluated_ids) or len(set(evaluated_indices)) != len(evaluated_indices)):
        raise ValueError("Invalid source evaluation image selection")
    evaluated_mapping = dict(zip(evaluated_ids, evaluated_indices))
    labels = list(summary["models"])
    if not labels or any(not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels):
        raise ValueError("Invalid model labels")
    records = {}
    for label in labels:
        rows = tracked.json(summary_path.parent / "records" / f"{label}.json")
        for row in rows:
            key = (label, row["image_id"], row["prompt_key"])
            if key in records or row["model"] != label or row["prompt_key"] not in SIDES:
                raise ValueError("Duplicate or mismatched evaluation records")
            records[key] = row
    output_rows = []
    for entry in manifest:
        image_id = entry["image_id"]
        if evaluated_mapping.get(image_id) != entry["dataset_index"]:
            raise ValueError("Visualization not in source evaluated image selection")
        image = images[image_id]
        shape = (image["height"], image["width"])
        directory = checked_child(visuals_root, f"image-{image_id:06d}")
        if directory != Path(entry["directory"]).resolve():
            raise ValueError("Manifest image directory mismatch")
        source_rgb = tracked.png(checked_child(data_root, image["file_name"]), binary=False)
        rendered_rgb = tracked.png(directory / "rgb.png", binary=False)
        if source_rgb.shape != (*shape, 3) or not np.array_equal(source_rgb, rendered_rgb):
            raise ValueError("Visualization RGB does not match source image")
        references = {side: np.zeros(shape, dtype=bool) for side in SIDES}
        for annotation in by_image[image_id]:
            references[SIDES[annotation["category_id"] - 1]] |= decode_rle(annotation["segmentation"], shape)
        for side in SIDES:
            if not np.array_equal(tracked.png(directory / f"{side}__reference.png"), references[side]):
                raise ValueError("Visualization reference differs from source COCO RLE")
            source_mask = image["source_masks"][side.removesuffix("_hand")]["binary_reference_png"]
            if not np.array_equal(tracked.png(checked_child(data_root, source_mask)), references[side]):
                raise ValueError("Source reference PNG differs from source COCO RLE")
        for label in labels:
            for side in SIDES:
                record = records[(label, image_id, side)]
                if (record.get("dataset_index") != entry["dataset_index"]
                        or record.get("observed_coco_image_id") != image_id
                        or record.get("identity_verified") is not True
                        or record.get("file_name") != image["file_name"]
                        or record.get("recording_id") != image["recording_id"]
                        or record.get("view_type") != image["view_type"]
                        or record.get("primary_test") != image["primary_test"]):
                    raise ValueError("Record/visualization image identity mismatch")
                score = record["top_confidence"]
                if (not math.isfinite(score) or not 0 <= score <= 1
                        or not math.isclose(score, record["top_class_probability"] * record["presence_probability"],
                                            rel_tol=1e-6, abs_tol=1e-7)
                        or type(record["detected"]) is not bool or record["detected"] != (score >= .5)):
                    raise ValueError("Detection score or fixed-threshold record is invalid")
                candidate_path = directory / label / f"{side}__candidate.png"
                detected_path = directory / label / f"{side}__detected.png"
                candidate = tracked.png(candidate_path)
                detected = tracked.png(detected_path)
                expected_detected = candidate if record["detected"] else np.zeros_like(candidate)
                if not np.array_equal(detected, expected_detected):
                    raise ValueError("Detected PNG differs from fixed-threshold candidate")
                other = SIDES[1 - SIDES.index(side)]
                validate_candidate_record(record, candidate, references[side], references[other])
                for stage, prediction, path in (("candidate", candidate, candidate_path),
                                                ("detected", detected, detected_path)):
                    output_rows.append({
                        "model": label, "stage": stage, "image_id": image_id,
                        "dataset_index": entry["dataset_index"], "prompt_key": side,
                        "recording_id": image["recording_id"], "view_type": image["view_type"],
                        "primary_test": bool(image["primary_test"]), "diagnostic_ids": entry["diagnostic_ids"],
                        "score": score, "score_detected": record["detected"],
                        "selected_decoder_query": record.get("selected_decoder_query"),
                        "prediction_png": str(path),
                        **measure_mask(prediction, references[side], references[other], boundary_ratio),
                    })
    tracked.read(Path(__file__))
    tracked.verify()
    return {
        "format": "sam3-hand-mask-error-decomposition-v1", "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "saved_visualization_subset_not_full_external_test",
        "source_evaluated_images": summary["evaluated_images"], "analyzed_images": len(manifest),
        "image_ids": selected_ids, "models": labels, "boundary_ratio": boundary_ratio,
        "source_summary": str(summary_path), "source_hashes_before_after_verified": True,
        "inputs_sha256": tracked.hashes, "definitions": metric_definitions(),
        "records": output_rows, "aggregates": aggregate_rows(output_rows),
    }


def aggregate_group(rows):
    present = [row for row in rows if row["own_reference_pixels"] > 0]
    totals = {name: sum(row[name] for row in rows) for name in (
        "prediction_pixels", "own_reference_pixels", "own_intersection_pixels", "own_missed_pixels",
        "other_only_intersection_pixels", "outside_both_references_pixels")}
    return {
        "queries": len(rows), "present_queries": len(present), "absent_queries": len(rows) - len(present),
        "empty_predictions": sum(row["prediction_pixels"] == 0 for row in rows),
        "score_detected_queries": sum(row["score_detected"] for row in rows),
        "totals": totals,
        "present_macro": {name: mean_defined(row[name] for row in present) for name in (
            "own_dice", "own_iou", "own_precision", "own_recall", "own_missed_fraction", "own_boundary_iou")},
        "present_precision_defined_queries": sum(row["own_precision"] is not None for row in present),
        "all_query_pixel_micro": {
            "own_precision": ratio(totals["own_intersection_pixels"], totals["prediction_pixels"]),
            "own_recall": ratio(totals["own_intersection_pixels"], totals["own_reference_pixels"]),
            "other_only_prediction_fraction": ratio(totals["other_only_intersection_pixels"], totals["prediction_pixels"]),
            "outside_both_prediction_fraction": ratio(totals["outside_both_references_pixels"], totals["prediction_pixels"]),
        },
        "opposite_overlap_dominant_queries": sum(row["opposite_overlap_dominant_proxy"] for row in rows),
    }


def aggregate_rows(rows):
    result = {}
    for label in sorted({row["model"] for row in rows}):
        result[label] = {}
        for stage in ("candidate", "detected"):
            selected = [row for row in rows if row["model"] == label and row["stage"] == stage]
            result[label][stage] = {
                "overall": aggregate_group(selected),
                "per_side": {side: aggregate_group([row for row in selected if row["prompt_key"] == side]) for side in SIDES},
                "per_view_type": {view: aggregate_group([row for row in selected if row["view_type"] == view])
                                  for view in sorted({row["view_type"] for row in selected})},
            }
    return result


def metric_definitions():
    return {
        "notation": "P prediction; G same-side reference; O opposite-side reference; pixel counts at original resolution",
        "prediction_partition": "P intersect G; P intersect (O minus G); P minus (G union O) form a disjoint partition of P",
        "overlapping_references": "G/O overlap is reported; own reference has priority in the disjoint partition; inclusive opposite overlap also reported",
        "dice_iou": "Dice=2|P intersect G|/(|P|+|G|); IoU=|P intersect G|/|P union G|",
        "precision_recall": "precision=|P intersect G|/|P|; recall=|P intersect G|/|G|; missed fraction=|G minus P|/|G|",
        "empty_denominators": "All zero-denominator metrics are null, not perfect scores; nonempty G plus empty P gives Dice/IoU/recall/boundary IoU 0, precision null",
        "boundary_iou": "IoU between inner bands M minus erode(M), 3x3 square, zero outside image; iterations=max(1,round(ratio*hypot(H,W))); ratio is recorded; not Boundary AP",
        "boundary_sources": [BOUNDARY_PAPER, BOUNDARY_SOURCE],
        "aggregation": "present_macro averages only nonempty own references; undefined values excluded with precision count reported; pixel_micro sums areas over all queries including absent references",
        "candidate_selection": "Use existing score-argmax candidate PNG, verify against evaluation record; no reference-driven candidate selection",
        "detected": "Existing candidate retained iff original class*presence score>=0.5; otherwise empty mask; no threshold refitting",
        "outside_reference_limitation": "Outside both reference masks is not a forearm label: may contain arm, object, background, or reference error; no anatomical leakage rate is claimed",
        "reference_limitation": "SAM3-assisted references, not fully independent manual pixel GT; visualization subset was not randomly sampled for population estimation",
        "missing_region_limitation": "Missing own-reference pixels do not identify anatomical fingers without additional region labels",
    }


def render_markdown(result):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"

    lines = ["# 手部分割错误分解（已保存可视化子集）", "",
             f"核验并分析 {result['analyzed_images']} 张可视化图片；原实验共 {result['source_evaluated_images']} 张。"
             "本表不是完整外测结果，也不能外推整个数据集。", "",
             "每张 RGB 与源图片逐像素核对，左右参考 PNG 与 COCO RLE 逐像素核对；"
             "候选 PNG 与原评估记录核对，阈值后 PNG 与固定 0.5 规则核对。所有读取文件 SHA256 前后不变。", "",
             "“两手参考外”只能说明预测超出参考并集，不能直接称为“手臂混入”：还可能是背景、物体或参考误差。"
             "同侧漏掉像素也不能在无局部标签时直接称为漏手指。参考是 SAM3 辅助标注。", "",
             "## 可见侧查询平均与全部查询像素分解", "",
             "| 模型 | 阶段 | 可见侧数 | Dice | IoU | Boundary IoU | 同侧漏掉比例 | 预测像素中另一手独占占比 | 预测像素中两手参考外占比 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, stages in result["aggregates"].items():
        for stage, groups in stages.items():
            value = groups["overall"]
            macro, micro = value["present_macro"], value["all_query_pixel_micro"]
            lines.append(f"| {label} | {stage} | {value['present_queries']} | " + " | ".join(fmt(item) for item in (
                macro["own_dice"], macro["own_iou"], macro["own_boundary_iou"], macro["own_missed_fraction"],
                micro["other_only_prediction_fraction"], micro["outside_both_prediction_fraction"])) + " |")
    lines += ["", "candidate：原评估按模型得分选定的候选，不按参考挑选。detected：低于固定 0.5 分数即输出空 mask。"
              "前三种重叠指标和漏掉比例对有同侧参考的查询取平均；最后两列先汇总所有查询的像素再计算比例。"
              "预测为空会导致检出后错误占比下降，因此必须同时看 Dice、漏掉比例和 JSON 中的空预测数。", "",
              "## 精确定义与边界约定", "",
              "令 P 为预测、G 为同侧参考、O 为另一侧参考：P∩G、P∩(O\\G)、P\\(G∪O) 是互不重叠的三个预测部分。"
              "如果参考重叠，交叠像素优先计为同侧正确，并单独记录参考交叠面积。", "",
              "Dice=2|P∩G|/(|P|+|G|)，IoU=|P∩G|/|P∪G|，precision=|P∩G|/|P|，"
              "recall=|P∩G|/|G|，漏掉比例=|G\\P|/|G|。所有分母为零的数值记 null；"
              "有参考而预测为空时 Dice/IoU/recall/Boundary IoU 为 0，precision 为 null。", "",
              f"Boundary IoU 使用内边带：原 mask 减去 3×3 方形结构腐蚀后的 mask，"
              f"腐蚀次数=max(1, round({result['boundary_ratio']}×图像对角线像素数))；图像外视为 0。"
              "计算两条内边带的 IoU；本脚本不是 Boundary AP。"
              f"定义依据[论文]({BOUNDARY_PAPER})和[作者实现]({BOUNDARY_SOURCE})，采用 SciPy CPU 实现，无需 OpenCV。", "",
              "左右手、ego/exo 分组，逐查询 precision/recall、原始像素计数、图片路径、源 SHA256 和空分母约定均保存在 analysis.json。", "",
              f"源评估：[summary.json](<{result['source_summary']}>)", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--boundary-ratio", type=float, default=.02)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    result = analyze_saved_visuals(args.summary, args.boundary_ratio)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in (("analysis.json", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                          ("REPORT.md", render_markdown(result))):
        with (args.output_dir / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
    print(json.dumps({"status": "completed", "images": result["analyzed_images"],
                      "rows": len(result["records"]), "output_dir": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

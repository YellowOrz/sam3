#!/usr/bin/env python3
"""CPU-only, validation-only threshold analysis under fixed opposite-hand FPR budgets.

Thresholds are fitted and measured on the SAME val records, not independent test
guarantees. This tool does not select a checkpoint or change training/inference.
The mask candidate and mask threshold remain fixed; missed correct prompts get
zero Dice. Threshold ties are indivisible because detection is score >= cutoff.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


BUDGETS = (0.01, 0.05, 0.10)
PROMPTS = ("left_hand", "right_hand")
COMPARISON_KEYS = (
    "annotations_sha256", "base_checkpoint", "evaluated_images",
    "mask_threshold", "confidence_definition",
)


def _probability(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number within [0,1]")
    return float(value)


def validate_records(records):
    """Require exactly one left and right prompt per aligned image."""
    if not isinstance(records, list) or not records:
        raise ValueError("Expected nonempty records")
    images, seen, model_labels = {}, set(), set()
    for row in records:
        index = row["dataset_index"]
        if type(index) is not int or index < 0:
            raise ValueError("Invalid dataset_index")
        side, prompt = row["actual_side"], row["prompt_key"]
        if side not in (*PROMPTS, "empty") or prompt not in PROMPTS:
            raise ValueError("Invalid actual_side or prompt_key")
        key = (index, prompt)
        if key in seen:
            raise ValueError(f"Duplicate image/prompt record: {key}")
        seen.add(key)
        model_labels.add(row["model"])
        if (row["target_present"] is not (side == prompt)
                or row["physical_hand_present"] is not (side != "empty")):
            raise ValueError("Inconsistent target/physical presence labels")
        _probability(row["top_confidence"], "top_confidence")
        _probability(row["top_dice_with_physical_hand"], "top Dice")
        identity = tuple(row[field] for field in (
            "image_id", "file_name", "source", "sequence", "view", "frame_index", "actual_side"
        ))
        if index in images and images[index] != identity:
            raise ValueError("Prompt rows disagree on image provenance")
        images[index] = identity
    if len(model_labels) != 1 or len(seen) != 2 * len(images):
        raise ValueError("Require one model and both prompts for every image")
    if len({identity[0] for identity in images.values()}) != len(images):
        raise ValueError("Duplicate image_id across dataset indices")
    return images


def calibrate_records(records, budgets=BUDGETS):
    validate_records(records)
    groups = defaultdict(list)
    totals = [0, 0, 0]  # correct prompt, visible opposite prompt, empty prompt
    correct_dice = []
    for row in records:
        category = 0 if row["target_present"] else (1 if row["physical_hand_present"] else 2)
        totals[category] += 1
        dice = float(row["top_dice_with_physical_hand"])
        if category == 0:
            correct_dice.append(dice)
        groups[float(row["top_confidence"])].append((category, dice))
    if not totals[0] or not totals[1]:
        raise ValueError("Visible correct and opposite prompts are required to calibrate")

    # Include the valid reject-all cutoff, even for saturated score==1 ties.
    states = [(math.nextafter(max(groups), math.inf), 0, 0, 0, 0.0)]
    counts, accepted_dice = [0, 0, 0], 0.0
    for score in sorted(groups, reverse=True):
        for category, dice in groups[score]:
            counts[category] += 1
            if category == 0:
                accepted_dice += dice
        states.append((score, *counts, accepted_dice))

    points = []
    for budget in budgets:
        _probability(budget, "FPR budget")
        fraction = Fraction(str(budget))
        max_false_positives = fraction.numerator * totals[1] // fraction.denominator
        feasible = [state for state in states if state[2] <= max_false_positives]
        # Maximize TPR. Flat TPR regions choose fewer errors, then the higher cutoff.
        threshold, true_positive, false_positive, empty_positive, dice_sum = max(
            feasible, key=lambda state: (state[1], -state[2], -state[3], state[0])
        )
        points.append({
            "opposite_fpr_budget": budget,
            "selected_threshold": threshold,
            "threshold_above_one_requires_reject_all": threshold > 1,
            "opposite_max_false_positives": max_false_positives,
            "correct_true_positives": true_positive,
            "opposite_false_positives": false_positive,
            "empty_false_positives": empty_positive,
            "correct_prompt_tpr": true_positive / totals[0],
            "actual_opposite_fpr": false_positive / totals[1],
            "correct_prompt_mean_thresholded_dice": dice_sum / totals[0],
            "empty_prompt_fpr": empty_positive / totals[2] if totals[2] else None,
        })
    return {
        "counts": dict(zip(("correct_prompts", "visible_opposite_prompts", "empty_prompts"), totals)),
        "correct_prompt_mean_top_dice": math.fsum(correct_dice) / totals[0],
        "unique_scores": len(groups), "operating_points": points,
    }


def analyze_summaries(paths):
    sources, reference, reference_identity, results = {}, None, None, {}

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in sources and sources[str(path)] != digest:
            raise RuntimeError(f"Source changed during analysis: {path}")
        sources[str(path)] = digest
        return json.loads(raw)

    for path in paths:
        path = Path(path).resolve()
        summary = read(path)
        data_root = Path(summary["data_root"]).resolve()
        if data_root.name != "val":
            raise ValueError("Threshold fitting is restricted to a val data directory")
        annotation_path = data_root / "annotations.json"
        annotations = read(annotation_path)
        if annotations.get("info", {}).get("split") != "val":
            raise ValueError("COCO info.split must explicitly be val; never calibrate on test")
        if sources[str(annotation_path)] != summary["annotations_sha256"]:
            raise ValueError("COCO annotation hash differs from completed evaluation")
        indices = summary["evaluated_dataset_indices"]
        if (len(set(indices)) != len(indices) or len(indices) != summary["evaluated_images"]
                or any(type(index) is not int or index < 0 for index in indices)):
            raise ValueError("Invalid evaluated indices/count")
        if reference is None:
            reference = summary
        else:
            for key in COMPARISON_KEYS:
                if summary[key] != reference[key]:
                    raise ValueError(f"Incompatible evaluations: {key}")
            if sorted(indices) != sorted(reference["evaluated_dataset_indices"]):
                raise ValueError("Incompatible evaluated_dataset_indices")
        if set(summary["models"]) != set(summary["metrics"]) or not summary["models"]:
            raise ValueError("Incomplete model/metric summary")
        for label in summary["models"]:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in (".", ".."):
                raise ValueError(f"Invalid model label: {label!r}")
            if label in results:
                raise ValueError(f"Duplicate model label: {label}")
            records = read(path.parent / "records" / f"{label}.json")
            if any(row["model"] != label for row in records):
                raise ValueError("Record model label differs from summary")
            identity = validate_records(records)
            if sorted(identity) != sorted(indices):
                raise ValueError("Record image coverage differs from summary")
            if reference_identity is None:
                reference_identity = identity
            elif identity != reference_identity:
                raise ValueError("Models were evaluated on different images or physical sides")
            results[label] = {
                "model_metadata": summary["models"][label], **calibrate_records(records)
            }
    if reference is None:
        raise ValueError("At least one completed validation summary is required")
    for path, expected in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed during analysis: {path}")
    return {
        "format": "sam3-bilateral-val-fpr-calibration-v1", "split": "val",
        "data_root": reference["data_root"],
        "annotations_sha256": reference["annotations_sha256"],
        "evaluated_images": reference["evaluated_images"],
        "mask_threshold": reference["mask_threshold"],
        "confidence_definition": reference["confidence_definition"],
        "fpr_budgets": list(BUDGETS),
        "selection_objective": "Maximize correct-prompt TPR subject to visible opposite-prompt FPR budget",
        "tie_policy": "score >= threshold; equal scores enter together; flat TPR prefers lower opposite/empty FP then higher threshold",
        "dice_definition": "Use unchanged top mask Dice when correct prompt is detected; count misses as zero",
        "limitations": [
            "Thresholds fitted and measured on this same val set; not independent test guarantees.",
            "Empty-prompt FPR is reported separately and is not constrained by the visible-opposite budget.",
            "A higher TPR under some budgets does not demonstrate uniformly better token quality; compare top-mask Dice too.",
            "Correlated video frames are not independent observations or a significance test.",
            "Legacy summaries lack base-checkpoint SHA256/AMP; identical paths do not prove numerical configuration equality.",
        ],
        "models": results, "source_files": [{"path": path, "sha256": digest} for path, digest in sources.items()],
        "sources_rechecked_unchanged": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, help="New JSON path; omitted means stdout only")
    args = parser.parse_args()
    result = analyze_summaries(args.summary)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        print(args.output.resolve())


if __name__ == "__main__":
    main()

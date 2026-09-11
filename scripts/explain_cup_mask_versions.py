"""Read-only three-layer cup diagnosis; no causal claim or label replacement."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.compare_source_mask_versions import MaskDecoder, compare_frame, probe, source_receipt


ROOT = Path("/data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh/cup")
PRIOR = Path("/home/xuzhefeng/Projects/sam3/outputs/realsense_hand_object_with_seg/interactive/cup/right_hand")
ONLY_LABEL = Path("/data/xuzhefeng/Datasets/realsense_hand_object_with_seg_only_label/cup/masks_sam3/right_hand.mkv")


def frame_ranges(indices):
    result = []
    for index in sorted(indices):
        if result and result[-1][1] + 1 == index:
            result[-1][1] = index
        else:
            result.append([index, index])
    return result


def geometry(mask):
    ys, xs = np.nonzero(mask > 0)
    return {"pixels": len(xs),
            "bbox_xyxy_exclusive": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None,
            "centroid_xy": [float(xs.mean()), float(ys.mean())] if len(xs) else None}


def diagnose(output):
    if output.exists():
        raise FileExistsError(output)
    streams = {"prior_right": PRIOR / "masks.mkv", "current_right": ROOT / "masks_sam3/right_hand.mkv",
               "current_left": ROOT / "masks_sam3/left_hand.mkv"}
    metadata_paths = [PRIOR / "metadata.json", PRIOR / "interactions.json"]
    # These exact companion locations were inspected; no repository-wide history
    # search, execution of source scripts, or inference from file mtime is made.
    history_dirs = [ROOT / "masks_sam3", ONLY_LABEL.parent, PRIOR,
                    Path("/home/xuzhefeng/Datasets/realsense_hand_object_with_seg/cup")]
    directory_inventory = {str(path): sorted(item.name for item in path.iterdir()) if path.is_dir() else None
                           for path in history_dirs}
    sources = [source_receipt(path) for path in [*streams.values(), *metadata_paths, ONLY_LABEL]]
    metadata = {name: probe(path) for name, path in streams.items()}
    reference = metadata["prior_right"]
    for name, item in metadata.items():
        if item["pts"] != reference["pts"] or any(item["stream"][key] != reference["stream"][key]
                                                   for key in ("width", "height", "time_base")):
            raise ValueError(f"Three-video frame/PTS/shape mismatch: {name}")
    width, height = reference["stream"]["width"], reference["stream"]["height"]
    prior_metadata = json.loads(metadata_paths[0].read_text())
    events = json.loads(metadata_paths[1].read_text())
    confirmed = events["confirmed_points"]
    points_by_frame = {}
    for point in confirmed:
        points_by_frame.setdefault(point["frame_index"], []).append(point)
    rows, point_results = [], []
    previous = None
    with ExitStack() as cleanup:
        decoders = {}
        for name, path in streams.items():
            decoder = MaskDecoder(path, width, height)
            cleanup.callback(decoder.close)
            decoders[name] = decoder
        for index, pts in enumerate(reference["pts"]):
            arrays = {name: decoder.frame() for name, decoder in decoders.items()}
            if any(value is None for value in arrays.values()):
                raise ValueError("Decoder ended before exact PTS count")
            old, right, left = [arrays[name] for name in ("prior_right", "current_right", "current_left")]
            o, r, l = old > 0, right > 0, left > 0
            old_count = int(o.sum())
            vs_right = compare_frame(old, right)
            vs_left = compare_frame(old, left)
            row = {"frame_index": index, "pts": pts,
                   "prior_right": geometry(old), "current_right": geometry(right), "current_left": geometry(left),
                   "prior_vs_current_right_dice": vs_right["binary_dice"],
                   "prior_vs_current_right_iou": vs_right["binary_iou"],
                   "prior_vs_current_left_dice": vs_left["binary_dice"],
                   "prior_vs_current_left_iou": vs_left["binary_iou"],
                   "prior_pixels_in_current_left": int((o & l).sum()),
                   "prior_pixels_in_current_right": int((o & r).sum()),
                   "prior_pixels_in_left_only": int((o & l & ~r).sum()),
                   "prior_pixels_in_right_only": int((o & r & ~l).sum()),
                   "prior_pixels_in_both_current_layers": int((o & r & l).sum()),
                   "prior_pixels_outside_both_current_layers": int((o & ~(r | l)).sum()),
                   "fraction_prior_inside_left": float((o & l).sum() / old_count) if old_count else None,
                   "fraction_prior_inside_right": float((o & r).sum() / old_count) if old_count else None,
                   "current_left_right_overlap_pixels": int((l & r).sum()),
                   "current_sides_are_independent_ground_truth": False}
            if old_count == 0:
                row["prior_spatial_preference"] = "prior_empty"
            elif row["prior_vs_current_left_iou"] > row["prior_vs_current_right_iou"]:
                row["prior_spatial_preference"] = "current_left"
            elif row["prior_vs_current_left_iou"] < row["prior_vs_current_right_iou"]:
                row["prior_spatial_preference"] = "current_right"
            else:
                row["prior_spatial_preference"] = "tie"
            if previous is not None:
                row["adjacent_frame_iou_not_motion_compensated"] = {
                    name: compare_frame(previous[name], value)["binary_iou"] for name, value in arrays.items()}
            rows.append(row)
            previous = arrays
            for point in points_by_frame.get(index, []):
                x, y = point["x"], point["y"]
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError("Confirmed point outside frame")
                point_results.append({**point, "values_in_final_export_layers": {
                    name: int(value[y, x]) for name, value in arrays.items()},
                    "warning": "A final export need not equal the intermediate mask displayed when this point was confirmed."})
        if any(decoder.frame() is not None for decoder in decoders.values()):
            raise ValueError("Unexpected extra decoded frames")
        for decoder in decoders.values():
            decoder.finish()
    after = [source_receipt(Path(item["path"])) for item in sources]
    if sources != after:
        raise RuntimeError("Source changed; refuse a stable explanation")
    after_inventory = {str(path): sorted(item.name for item in path.iterdir()) if path.is_dir() else None
                       for path in history_dirs}
    if directory_inventory != after_inventory:
        raise RuntimeError("Companion source directory changed during history check")
    starts_and_confirms = [item for item in events["events"] if item["type"] in (
        "initial_prompt", "confirm", "propagation_start", "propagation_end", "finalize", "cancel_edit")]
    source_by_path = {item["path"]: item for item in sources}
    report = {"format": "realsense-cup-d3-three-mask-diagnosis-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "completed", "sources": sources, "sources_after": after, "sources_unchanged": True,
              "stream_metadata": metadata, "decoded_frames": len(rows), "exact_pts_match": True,
              "frame_185": rows[185], "frame_170_through_200": rows[170:201],
              "spatial_preference_counts": dict(Counter(row["prior_spatial_preference"] for row in rows)),
              "prior_more_like_current_left_ranges_inclusive": frame_ranges([
                  row["frame_index"] for row in rows if row["prior_spatial_preference"] == "current_left"]),
              "interaction_evidence": {"metadata": prior_metadata,
                                       "top_level_propagation_direction": events["propagation_direction"],
                                       "confirmed_points": confirmed, "confirmed_points_sampled_final_masks": point_results,
                                       "chronological_confirm_and_propagation_events": starts_and_confirms,
                                       "applies_to": "prior export only, not the current right mask",
                                       "intermediate_generation_masks_available": False,
                                       "causal_event_producing_frame185_proven": False},
              "current_version_history": {"checked_companion_directory_entries": directory_inventory,
                                          "same_bytes_as_only_label": source_by_path[str(streams["current_right"])]["sha256"] == source_by_path[str(ONLY_LABEL)]["sha256"],
                                          "current_revision_generation_history_found": False,
                                          "limits": "Only the explicitly listed cup source/export companion locations were checked. No claim that a history cannot exist elsewhere; copying identity is not generation history."},
              "interpretation": {"anatomical_identity_independently_verified_here": False,
                                 "current_left_and_right_may_themselves_contain_label_errors": True,
                                 "pixel_overlap_is_spatial_evidence_not_causal_proof": True,
                                 "temporal_iou_is_not_motion_compensated_and_not_a_tracking_accuracy": True,
                                 "model_inference_or_training": False, "source_edits_or_label_merging": False},
              "frames": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("decoded_frames", "frame_185", "spatial_preference_counts", "prior_more_like_current_left_ranges_inclusive")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    diagnose(parser.parse_args().output)

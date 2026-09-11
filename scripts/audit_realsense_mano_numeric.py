#!/usr/bin/env python3
"""Read-only full NPZ numeric audit and current-mask/NPZ temporal contract check.

No pickle, ROS payload deserialization, mesh fitting, projection, conversion or
training. Independent mask layers remain independent, including their overlaps.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

try:
    from scripts.audit_realsense_hand_object import inventory, differences
except ModuleNotFoundError:
    from audit_realsense_hand_object import inventory, differences


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tight_box(binary):
    y, x = np.nonzero(binary)
    return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=np.float32) if len(x) else np.full(4, np.nan, dtype=np.float32)


def rotation_statistics(value, valid, tolerance=1e-4):
    if value.shape[-2:] != (3, 3):
        return {"error": "Not a rotation matrix array"}
    arrays = value.reshape(len(valid), -1, 3, 3)
    finite = np.isfinite(arrays).all(axis=(2, 3))
    selected = arrays[valid].astype(np.float64)
    selected_finite = np.isfinite(selected).all(axis=(2, 3))
    matrices = selected[selected_finite]
    result = {"valid_frame_count": int(valid.sum()), "finite_rotation_matrices_on_valid_frames": int(selected_finite.sum()),
              "nonfinite_rotation_matrices_on_valid_frames": int(selected_finite.size - selected_finite.sum()),
              "orthogonality_and_det_tolerance": tolerance}
    if matrices.size:
        orthogonality = np.max(np.abs(np.swapaxes(matrices, -1, -2) @ matrices - np.eye(3)), axis=(1, 2))
        determinants = np.linalg.det(matrices)
        bad = (orthogonality > tolerance) | (np.abs(determinants - 1) > tolerance)
        result.update(max_abs_rt_r_minus_i=float(orthogonality.max()), determinant_min=float(determinants.min()),
                      determinant_max=float(determinants.max()), max_abs_det_minus_one=float(np.abs(determinants - 1).max()),
                      matrices_outside_tolerance=int(bad.sum()))
    return result


def inspect_npz(path, frame_count):
    digest = sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}  # Full array bodies and ZIP CRCs, never object pickle.
    if sha256(path) != digest:
        raise ValueError("NPZ changed while reading")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays are forbidden")
    boxes = arrays["bbox_xyxy"]
    if boxes.shape != (frame_count, 4):
        raise ValueError("NPZ bbox frame shape differs from current RGB")
    flags = arrays.get("has_hand")
    if flags is not None and (flags.shape != (frame_count,) or flags.dtype != np.bool_):
        raise ValueError("has_hand must be boolean and match RGB frame count")
    validity = flags if flags is not None else np.isfinite(boxes).all(axis=1)
    instance = arrays.get("instance_label")
    label = int(instance.item()) if instance is not None else None
    if label is not None and not 1 <= label <= 255:
        raise ValueError("Invalid declared instance label")
    schema = "instance_label_and_has_hand" if label is not None and flags is not None else (
        "legacy_no_instance_label_with_has_hand" if flags is not None else "legacy_no_instance_label_no_has_hand")
    report = {"path": str(path.resolve()), "sha256": digest, "schema": schema,
              "mask_selector": f"native_label=={label}" if label is not None else "native_label>0 (explicit legacy union assumption, not proven producer contract)",
              "validity_source": "declared has_hand" if flags is not None else "finite bbox inferred for audit only; no has_hand declaration",
              "valid_frames": int(validity.sum()), "invalid_frames": int((~validity).sum()),
              "complete_array_bodies_read_allow_pickle_false": True, "arrays": {},
              "hand": str(arrays["hand"].item()), "pose_format": str(arrays["pose_format"].item()),
              "frame_indices_equal_complete_current_rgb": bool(np.array_equal(arrays["frame_indices"], np.arange(frame_count))),
              "total_frames_equal_current_rgb": int(arrays["total_frames"].item()) == frame_count,
              "width": int(arrays["width"].item()), "height": int(arrays["height"].item()),
              "instance_label": label, "has_hand_present": flags is not None, "rotations": {}}
    for key, value in arrays.items():
        row = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if np.issubdtype(value.dtype, np.inexact):
            finite = np.isfinite(value)
            row["nonfinite_total"] = int(value.size - finite.sum())
            if finite.any():
                row.update(finite_min=float(value[finite].min()), finite_max=float(value[finite].max()))
            if value.ndim and value.shape[0] == frame_count:
                per_frame = finite.reshape(frame_count, -1).all(axis=1)
                row.update(nonfinite_valid_frame_indices=np.flatnonzero(validity & ~per_frame).tolist(),
                           invalid_frame_all_nan_count=int(np.isnan(value[~validity]).reshape(int((~validity).sum()), -1).all(axis=1).sum()) if (~validity).any() else 0,
                           finite_invalid_frame_count=int((~validity & per_frame).sum()))
        elif value.ndim == 0:
            row["value"] = value.item()
        report["arrays"][key] = row
    for key in ("global_orient", "hand_pose"):
        report["rotations"][key] = rotation_statistics(arrays[key], validity)
    confidence = arrays.get("confidence")
    if confidence is not None:
        selected = confidence[validity]
        report["confidence_valid_outside_0_1"] = int(((selected < 0) | (selected > 1)).sum())
    return report, arrays, label


def stream_mask_contracts(directory, video_metadata, selectors):
    """One simultaneous CPU streaming pass over independent current mask layers."""
    paths = {Path(relative).stem: directory / relative for relative in video_metadata if relative.startswith("masks_sam3/")}
    metadata = {Path(relative).stem: value for relative, value in video_metadata.items() if relative.startswith("masks_sam3/")}
    expected = video_metadata["color.mp4"]["decoded_frames"]
    h, w = video_metadata["color.mp4"]["height"], video_metadata["color.mp4"]["width"]
    boxes = {key: [] for key in selectors}
    overlaps = {key: {"pixels": 0, "frames": 0, "max_pixels_in_one_frame": 0} for key in ("left_right", "left_object", "right_object")}
    source_hashes = {str(path): sha256(path) for path in paths.values()}
    with ExitStack() as stack:
        processes = {}
        for role, path in paths.items():
            if metadata[role]["pix_fmt"] != "gray" or (metadata[role]["height"], metadata[role]["width"]) != (h, w):
                raise ValueError("Mask must be native gray at RGB size")
            stderr = stack.enter_context(tempfile.TemporaryFile())
            process = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "gray", "-threads", "1", "pipe:1"],
                stdout=subprocess.PIPE, stderr=stderr)
            processes[role] = process, stderr
        try:
            for frame in range(expected):
                layers = {}
                for role, (process, _) in processes.items():
                    raw = process.stdout.read(h * w)
                    if len(raw) != h * w:
                        raise ValueError(f"Truncated mask stream at frame {frame}: {role}")
                    layers[role] = np.frombuffer(raw, np.uint8).reshape(h, w)
                for key, (role, label) in selectors.items():
                    pixels = layers[role]
                    boxes[key].append(tight_box(pixels == label if label is not None else pixels > 0))
                for key, roles in (("left_right", ("left_hand", "right_hand")),
                                   ("left_object", ("left_hand", "object")), ("right_object", ("right_hand", "object"))):
                    if all(role in layers for role in roles):
                        count = int(((layers[roles[0]] > 0) & (layers[roles[1]] > 0)).sum())
                        overlaps[key]["pixels"] += count
                        overlaps[key]["frames"] += int(count > 0)
                        overlaps[key]["max_pixels_in_one_frame"] = max(overlaps[key]["max_pixels_in_one_frame"], count)
            for role, (process, stderr) in processes.items():
                if process.stdout.read(1):
                    raise ValueError(f"Mask has extra frames: {role}")
                code = process.wait(timeout=30)
                stderr.seek(0)
                error = stderr.read()
                if code or error:
                    raise ValueError(f"Mask decode error: {role}: {error[-1000:]}")
        finally:
            for process, _ in processes.values():
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
    for path, digest in source_hashes.items():
        if sha256(path) != digest:
            raise ValueError("Source masks changed during decoding")
    return {key: np.stack(value) for key, value in boxes.items()}, {
        "frames": expected, "roles_present": sorted(paths), "mask_sha256": source_hashes,
        "overlap_definition": "independent layer unions native_value>0; no forced exclusivity and no proof overlap is an error",
        "overlap_pairs_with_missing_roles_are_not_evaluated": True,
        "overlaps": {key: value for key, value in overlaps.items() if all(role in paths for role in
            {"left_right": ("left_hand", "right_hand"), "left_object": ("left_hand", "object"), "right_object": ("right_hand", "object")}[key])}}


def compare_mask_contract(current_boxes, arrays):
    observed = arrays["bbox_xyxy"]
    current_present = np.isfinite(current_boxes).all(axis=1)
    observed_finite = np.isfinite(observed).all(axis=1)
    comparable = current_present & observed_finite
    mismatch = comparable & np.any(current_boxes != observed, axis=1)
    flags = arrays.get("has_hand")
    indices = np.flatnonzero(mismatch)
    return {"bbox_convention": "xmin,ymin,xmax+1,ymax+1; exact pixel equality, no crop expansion or tolerance",
        "current_mask_present_frames": int(current_present.sum()), "stored_bbox_finite_frames": int(observed_finite.sum()),
        "comparable_present_frames": int(comparable.sum()), "bbox_mismatch_count": len(indices),
        "bbox_mismatch_frame_indices": indices.tolist(),
        "bbox_mismatch_examples": [{"frame_index": int(i), "current_mask_bbox": current_boxes[i].tolist(), "stored_bbox": observed[i].tolist()} for i in indices[:20]],
        "mask_present_but_bbox_nonfinite": np.flatnonzero(current_present & ~observed_finite).tolist(),
        "mask_empty_but_bbox_finite": np.flatnonzero(~current_present & observed_finite).tolist(),
        "has_hand_disagrees_current_mask": np.flatnonzero(flags != current_present).tolist() if flags is not None else None,
        "no_has_hand_warning": "No declared validity; finite-bbox comparison cannot substitute an authoritative has_hand flag" if flags is None else None}


def run(root, structural):
    before = inventory(root)
    result = {"format": "realsense-mano-full-numeric-audit-v1", "root": str(root), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "full_numeric_read_no_pickle": True, "sequences": {}, "errors": [],
              "geometry_ready": False, "geometry_readiness_limit": "Numeric validity and mask bbox agreement alone do not verify MANO handed coordinate transforms, calibrated camera projection, depth alignment, or real-scene accuracy"}
    for name, source in structural["sequences"].items():
        row = {"npz": {}}
        result["sequences"][name] = row
        loaded, selectors = {}, {}
        for relative in source["npz"]:
            try:
                report, arrays, label = inspect_npz(root / name / relative, source["videos"]["color.mp4"]["decoded_frames"])
                row["npz"][relative] = report
                loaded[relative] = arrays
                role = str(arrays.get("mask_source", arrays["hand"]).item())
                if role not in ("left_hand", "right_hand"):
                    raise ValueError("Unsupported declared hand/mask source")
                selectors[relative] = role, label
            except Exception as error:
                result["errors"].append({"file": str(root / name / relative), "error": f"{type(error).__name__}: {error}"})
        try:
            boxes, layer_summary = stream_mask_contracts(root / name, source["videos"], selectors)
            row["mask_layers"] = layer_summary
            for relative, values in loaded.items():
                row["npz"][relative]["current_mask_contract"] = compare_mask_contract(boxes[relative], values)
        except Exception as error:
            result["errors"].append({"sequence": name, "phase": "mask_comparison", "error": f"{type(error).__name__}: {error}"})
        print(json.dumps({"sequence": name, "npz": len(row["npz"]), "errors": len(result["errors"])}), flush=True)
    after = inventory(root)
    result.update(finished_at_utc=datetime.now(timezone.utc).isoformat(), stat_changes=differences(before, after),
                  changes_since_structural_audit=differences(structural["inventory_after"], after))
    result["status"] = "completed" if not result["errors"] and not any(result["stat_changes"].values()) else "completed_with_findings"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Use a new output outside source data")
    source = json.loads(args.structural_audit.read_text())
    if Path(source["root"]).resolve() != args.root.resolve():
        parser.error("Structural audit names a different root")
    result = run(args.root.resolve(), source)
    result["structural_audit_sha256"] = sha256(args.structural_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "errors": len(result["errors"])}))


if __name__ == "__main__":
    main()

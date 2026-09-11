#!/usr/bin/env python3
"""Read-only local nakehand inventory and bounded exact-timestamp mask audit.

Writes diagnostics only to an explicit new output directory, never the source.
RGB videos are probed and seek-sampled, not fully decoded. Small FFV1 mask
streams are packet-counted; sampled RGB/mask PTS must match the requested
zero-based frame at nominal FPS within the Matroska millisecond timebase.
NPZ is loaded with allow_pickle=False. SAM3 outputs are called annotations/
pseudo-labels, not ground-truth. No model, network or GPU is used.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path, count_packets=False):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_packets:
        command.append("-count_packets")
    command += ["-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_packets,start_time:stream_tags:format=duration,size", "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command))


def extract(path, frame, fps, width, height, gray=False):
    seek = max(0, frame / fps - 0.0006)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1",
               "-copyts", "-ss", f"{seek:.6f}", "-i", str(path), "-map", "0:v:0",
               "-vf", "showinfo", "-frames:v", "1", "-threads", "1", "-f", "rawvideo",
               "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
    text = result.stderr.decode("utf-8", errors="replace")
    timestamps = re.findall(r"\bpts_time:([0-9.+-]+)", text)
    if not timestamps or abs(float(timestamps[0]) - frame / fps) > 0.0012:
        raise ValueError(f"Seek did not return expected frame {frame}: {path}; PTS={timestamps[:2]}")
    channels = 1 if gray else 3
    if len(result.stdout) != width * height * channels:
        raise ValueError(f"Unexpected decoded byte length: {path}/{frame}")
    shape = (height, width) if gray else (height, width, 3)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(shape).copy(), float(timestamps[0])


def npz_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {"path": str(path), "sha256": sha256(path), "arrays": {}}
        valid = archive["has_hand"].astype(bool)
        indices = archive["frame_indices"]
        result.update(total_rows=len(valid), valid_rows=int(valid.sum()),
                      invalid_rows=int((~valid).sum()),
                      frame_indices_range=[int(indices.min()), int(indices.max())],
                      unique_frame_indices=len(np.unique(indices)),
                      frame_indices_identity=bool(np.array_equal(indices, np.arange(len(indices)))))
        for key in archive.files:
            values = archive[key]
            entry = {"shape": list(values.shape), "dtype": str(values.dtype)}
            if values.ndim == 0:
                entry["value"] = values.item()
            elif values.dtype.kind in "fiu" and values.shape[0] == len(valid):
                entry["finite_valid_values"] = bool(np.isfinite(values[valid]).all())
                entry["finite_invalid_values"] = bool(np.isfinite(values[~valid]).all())
                if key in ("confidence", "camera_translation", "betas") and valid.any():
                    entry["valid_min"] = float(np.nanmin(values[valid]))
                    entry["valid_max"] = float(np.nanmax(values[valid]))
                if key == "camera_translation" and valid.any() and np.isfinite(values[valid]).all():
                    z = values[valid, 2]
                    row = int(np.flatnonzero(valid)[np.argmax(z)])
                    entry["z_quantile_probabilities"] = [0, 0.5, 0.95, 0.99, 1]
                    entry["z_quantiles_source_units"] = np.quantile(z, entry["z_quantile_probabilities"]).tolist()
                    entry["max_z_row"] = row
                    entry["max_z_source_frame"] = int(indices[row])
                    entry["max_z_confidence"] = float(archive["confidence"][row])
                    entry["max_z_bbox_xyxy"] = archive["bbox_xyxy"][row].tolist()
                    entry["unit_status"] = "not inferred; inspect source coordinate/focal-length conventions before use"
                if key == "betas" and valid.any() and "wilor_right_canonical_beta10" in archive:
                    entry["max_abs_delta_fixed_shape"] = float(np.max(np.abs(values[valid] - archive["wilor_right_canonical_beta10"])))
                if key in ("global_orient", "hand_pose") and valid.any():
                    rotations = values[valid].reshape(-1, 3, 3).astype(np.float64)
                    if np.isfinite(rotations).all():
                        entry["max_RtR_error"] = float(np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3))))
                        determinants = np.linalg.det(rotations)
                        entry["determinant_min_max"] = [float(determinants.min()), float(determinants.max())]
            result["arrays"][key] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/xuzhefeng/Datasets/wanqing_datasets/nakehand"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-recording", type=int, default=7)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    if root == output or root in output.parents or args.samples_per_recording < 2:
        raise ValueError("Need external new output and at least two sampled frames")
    output.mkdir(parents=True, exist_ok=False)
    all_files = sorted(path for path in root.rglob("*") if path.is_file())
    inventory = [{"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size}
                 for path in all_files]
    suffix_counts = Counter(path.suffix.lower() for path in all_files)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
               "file_count": len(all_files), "file_bytes": sum(item["bytes"] for item in inventory),
               "suffix_counts": dict(suffix_counts), "inventory": inventory,
               "recordings": [], "mask_provenance": [], "npz": [], "samples": [],
               "method": "header probes; full packet count of small masks only; exact-PTS seek samples of RGB/masks; no full RGB decode"}
    sidecars = root.parent / "nakehand_masks_sam3"
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        recording_dir = metadata_path.parent
        relative = recording_dir.relative_to(root)
        metadata = json.loads(metadata_path.read_text())
        frames = metadata["frames"]
        count = metadata["frame_count"]
        fps = float(metadata["timestamps"]["video_nominal_fps"])
        timestamp_values = np.asarray([frame["timestamp"] for frame in frames])
        deltas = np.diff(timestamp_values)
        recording = {"recording": str(relative), "metadata_sha256": sha256(metadata_path),
                     "metadata": {key: value for key, value in metadata.items() if key != "frames"},
                     "frame_mapping": {"entries": len(frames),
                                       "video_identity": [frame["video_frame_index"] for frame in frames] == list(range(count)),
                                       "source_identity": [frame["source_frame_index"] for frame in frames] == list(range(count)),
                                       "timestamp_strictly_increasing": bool((deltas > 0).all()),
                                       "timestamp_delta_min_median_max": [float(deltas.min()), float(np.median(deltas)), float(deltas.max())],
                                       "depth_missing_frames": sum(bool(frame["depth_missing"]) for frame in frames)},
                     "streams": {}}
        video_paths = {"rgb": recording_dir / "rgb.mkv", "depth": recording_dir / "depth.mkv",
                       "left": recording_dir / "masks_sam3/left_hand.mkv", "right": recording_dir / "masks_sam3/right_hand.mkv"}
        for name, path in video_paths.items():
            recording["streams"][name] = probe(path, count_packets=name in ("left", "right"))
        summary["recordings"].append(recording)
        for side in ("left", "right"):
            parent = sidecars / relative / "masks_sam3" / f"{side}_hand"
            meta = parent / "metadata.json"
            if meta.is_file():
                current = video_paths[side]
                upstream = parent / "masks.mkv"
                evidence = {"recording": str(relative), "side": side, "metadata_path": str(meta),
                            "metadata_sha256": sha256(meta), "metadata": json.loads(meta.read_text()),
                            "current_mask_sha256": sha256(current),
                            "sibling_mask_sha256": sha256(upstream) if upstream.is_file() else None}
                evidence["same_mask_bytes"] = evidence["current_mask_sha256"] == evidence["sibling_mask_sha256"]
                interaction = parent / "interactions.json"
                if interaction.is_file():
                    interactions = json.loads(interaction.read_text())
                    evidence["interactions"] = {"path": str(interaction), "sha256": sha256(interaction),
                                                "confirmed_points_count": len(interactions.get("confirmed_points", [])),
                                                "event_types": dict(Counter(event.get("type") for event in interactions.get("events", []))),
                                                "text_prompt": interactions.get("text_prompt")}
                summary["mask_provenance"].append(evidence)
        for npz in sorted(recording_dir.glob("MANO_wilor/*/*.npz")):
            summary["npz"].append(npz_summary(npz))
        width, height = metadata["intrinsics"]["width"], metadata["intrinsics"]["height"]
        sampled = sorted(set(int(round(value)) for value in np.linspace(0, count - 1, args.samples_per_recording)))
        for frame in sampled:
            rgb, rgb_pts = extract(video_paths["rgb"], frame, fps, width, height)
            left, left_pts = extract(video_paths["left"], frame, fps, width, height, gray=True)
            right, right_pts = extract(video_paths["right"], frame, fps, width, height, gray=True)
            if len({rgb_pts, left_pts, right_pts}) != 1:
                raise ValueError(f"RGB/mask PTS mismatch: {relative}/{frame}")
            sample_dir = output / "samples" / str(relative).replace("/", "__") / f"frame-{frame:06d}"
            sample_dir.mkdir(parents=True)
            Image.fromarray(rgb).save(sample_dir / "rgb.png")
            Image.fromarray(left).save(sample_dir / "left_instance_raw.png")
            Image.fromarray(right).save(sample_dir / "right_instance_raw.png")
            Image.fromarray((left > 0).astype(np.uint8) * 255).save(sample_dir / "left_binary.png")
            Image.fromarray((right > 0).astype(np.uint8) * 255).save(sample_dir / "right_binary.png")
            panels = [Image.fromarray(rgb), Image.fromarray((left > 0).astype(np.uint8) * 255).convert("RGB"),
                      Image.fromarray((right > 0).astype(np.uint8) * 255).convert("RGB")]
            sheet = Image.new("RGB", (width * 3, height + 46), (28, 28, 28))
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            for index, (panel, label) in enumerate(zip(panels, ["Original RGB", "SAM3 left annotation (NOT independent GT)", "SAM3 right annotation (NOT independent GT)"])):
                sheet.paste(panel, (index * width, 46))
                draw.text((index * width + 5, 4), label, font=font, fill="white")
                draw.text((index * width + 5, 23), f"{relative} | frame={frame} PTS={rgb_pts:.3f}", font=font, fill="white")
            sheet.save(sample_dir / "comparison.png")
            summary["samples"].append({"recording": str(relative), "frame": frame,
                                       "source_frame": frames[frame]["source_frame_index"], "rgb_mask_pts": rgb_pts,
                                       "left_labels": np.unique(left).tolist(), "right_labels": np.unique(right).tolist(),
                                       "left_pixels": int((left > 0).sum()), "right_pixels": int((right > 0).sum()),
                                       "left_right_overlap_pixels": int(((left > 0) & (right > 0)).sum()),
                                       "comparison": str(sample_dir / "comparison.png"), "sample_dir": str(sample_dir)})
        (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"audited {relative}: {count} declared frames; {len(sampled)} exact-PTS samples", flush=True)
    print(json.dumps({"file_count": summary["file_count"], "file_bytes": summary["file_bytes"],
                      "recordings": len(summary["recordings"]), "samples": len(summary["samples"]),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

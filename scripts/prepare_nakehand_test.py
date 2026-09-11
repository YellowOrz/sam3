#!/usr/bin/env python3
"""Export a fixed, read-only nakehand external *pseudo-label* test sample.

No model, training, geometry, network, or GPU is used. Each recording contributes
the same number of uniformly selected original frame indices, including both
endpoints. User-reviewed A/B/C are included as separately flagged diagnostics;
their acceptance is NOT extrapolated to the remaining annotations.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

try:
    from scripts.audit_nakehand_dataset import extract, probe, sha256
except ModuleNotFoundError:
    from audit_nakehand_dataset import extract, probe, sha256


DIAGNOSTICS = {
    ("nakehandego/20260907_142020", 575): "A",
    ("nakehandexo/20260907_123926", 1459): "B",
    ("nakehandego/20260907_142020", 1724): "C",
}
CATEGORIES = [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]


def uniform_indices(frame_count, sample_count):
    """Integer-only round-half-up linspace: reproducible, unique, both endpoints."""
    if not isinstance(frame_count, int) or not isinstance(sample_count, int):
        raise ValueError("frame/sample counts must be integers")
    if sample_count < 2 or frame_count < sample_count:
        raise ValueError("Need at least two samples and at least sample_count frames")
    denominator = 2 * (sample_count - 1)
    result = [(2 * index * (frame_count - 1) + sample_count - 1) // denominator
              for index in range(sample_count)]
    if len(set(result)) != sample_count or result[0] != 0 or result[-1] != frame_count - 1:
        raise ValueError("Invalid systematic sampling")
    return result


def select_frames(path, indices, fps, width, height, gray=False):
    """Decode selected original indices with one CPU thread; check every PTS.

    communicate() inside subprocess.run consumes stdout and stderr concurrently.
    At 100 RGB samples of 640x480, raw output is about 92 MB, not a full video.
    """
    if not indices or indices != sorted(set(indices)) or indices[0] < 0:
        raise ValueError("Frame indices must be nonempty, sorted, unique, nonnegative")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    if len(indices) <= 3:
        arrays, timestamps = zip(*(extract(path, index, fps, width, height, gray)
                                   for index in indices))
        return np.stack(arrays), list(timestamps)
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "1",
               "-copyts", "-i", str(path), "-map", "0:v:0", "-vf", f"select={selection},showinfo",
               "-fps_mode", "passthrough", "-frames:v", str(len(indices)), "-threads", "1",
               "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True, timeout=1200)
    pts = [float(value) for value in re.findall(r"\bpts_time:([0-9.+-]+)",
                                               result.stderr.decode("utf-8", errors="replace"))]
    channels = 1 if gray else 3
    expected = len(indices) * width * height * channels
    if len(result.stdout) != expected or len(pts) != len(indices):
        raise ValueError(f"Decoded size/count mismatch: {path}: bytes={len(result.stdout)}/{expected}, PTS={len(pts)}")
    for index, timestamp in zip(indices, pts):
        if not np.isfinite(timestamp) or abs(timestamp - index / fps) > 0.0012:
            raise ValueError(f"PTS mismatch: {path} frame={index} timestamp={timestamp}")
    shape = (len(indices), height, width) if gray else (len(indices), height, width, 3)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(shape), pts


def side_annotation(raw_mask, image_id, category_id, annotation_id, provenance=None):
    """Union ALL positive chunk labels on one physical side; never area-filter."""
    if raw_mask.ndim != 2 or raw_mask.dtype != np.uint8:
        raise ValueError("Expected uint8 2D source instance mask")
    binary = np.asfortranarray(raw_mask > 0, dtype=np.uint8)
    if not binary.any():
        return None
    rle = mask_utils.encode(binary)
    area = int(mask_utils.area(rle))
    bbox = [float(value) for value in mask_utils.toBbox(rle)]
    if area != int(binary.sum()) or not np.array_equal(mask_utils.decode(rle), binary):
        raise ValueError("RLE mask round-trip mismatch")
    rle["counts"] = rle["counts"].decode("ascii")
    result = {"id": annotation_id, "image_id": image_id, "category_id": category_id,
              "segmentation": rle, "area": area, "bbox": bbox, "iscrowd": 0,
              "source_instance_values": [int(value) for value in np.unique(raw_mask) if value > 0],
              "label_source": "SAM3 prompted/propagated pseudo-label; not independent pixel ground truth"}
    if provenance is not None:
        result["source_mask"] = provenance
    return result


def frame_chunk(metadata, frame):
    if "chunks" not in metadata and "object_id_to_label" in metadata:
        count = metadata["source"]["frame_count"]
        if not 0 <= frame < count:
            raise ValueError(f"Frame outside legacy video-level label mapping: {frame}")
        return {"index": None, "start_frame": 0, "end_frame_exclusive": count,
                "object_id_to_label": metadata["object_id_to_label"], "mapping_scope": "video",
                "normalization": "Legacy source declares one video-wide mapping, not a chunk; bounds from source.frame_count"}
    matches = [chunk for chunk in metadata.get("chunks", [])
               if chunk["start_frame"] <= frame < chunk["end_frame_exclusive"]]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source chunk for frame {frame}")
    return {**matches[0], "mapping_scope": "chunk"}


def source_record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path, value):
    """Never expose a partially written JSON file to a waiting evaluator."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def verify_sources(records):
    for record in records:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Source changed during export; dataset not published: {path}")


def load_recording(root, metadata_path):
    directory = metadata_path.parent
    relative = directory.relative_to(root).as_posix()
    paths = {"rgb": directory / "rgb.mkv", "left": directory / "masks_sam3/left_hand.mkv",
             "right": directory / "masks_sam3/right_hand.mkv", "metadata": metadata_path}
    sources = [source_record(path) for path in paths.values()]
    metadata = json.loads(metadata_path.read_text())
    frame_count = metadata["frame_count"]
    frames = metadata["frames"]
    if len(frames) != frame_count or [frame["video_frame_index"] for frame in frames] != list(range(frame_count)):
        raise ValueError(f"Non-identity/incomplete video frame map: {relative}")
    if not all(np.isfinite(frame["timestamp"]) for frame in frames):
        raise ValueError(f"Nonfinite source capture timestamps: {relative}")
    if any(left["timestamp"] >= right["timestamp"] for left, right in zip(frames, frames[1:])):
        raise ValueError(f"Non-increasing capture timestamps: {relative}")
    width, height = metadata["intrinsics"]["width"], metadata["intrinsics"]["height"]
    fps = float(metadata["timestamps"]["video_nominal_fps"])
    stream_info = {}
    sidecars = {}
    for stream in ("rgb", "left", "right"):
        stream_info[stream] = probe(paths[stream], count_packets=stream != "rgb")
        details = stream_info[stream]["streams"][0]
        if details["width"] != width or details["height"] != height:
            raise ValueError(f"Stream dimensions mismatch: {relative}/{stream}")
        if stream != "rgb" and int(details["nb_read_packets"]) != frame_count:
            raise ValueError(f"Mask frame count mismatch: {relative}/{stream}")
    for side in ("left", "right"):
        side_dir = root.parent / "nakehand_masks_sam3" / relative / "masks_sam3" / f"{side}_hand"
        source_metadata = side_dir / "metadata.json"
        interactions = side_dir / "interactions.json"
        upstream_mask = side_dir / "masks.mkv"
        for path in (source_metadata, interactions, upstream_mask):
            sources.append(source_record(path))
        sidecars[side] = {"metadata_path": str(source_metadata), "metadata": json.loads(source_metadata.read_text()),
                          "interactions_path": str(interactions), "upstream_mask_path": str(upstream_mask)}
        if sha256(upstream_mask) != sha256(paths[side]):
            raise ValueError(f"Cannot establish upstream SAM3 mask provenance: {relative}/{side}")
    return {"recording_id": relative, "paths": paths, "metadata": metadata, "sources": sources,
            "sidecars": sidecars, "stream_info": stream_info, "width": width, "height": height, "fps": fps}


def export_dataset(root, output, samples_per_recording=100, diagnostics_only=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the read-only source dataset")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite any output: {output}")
    paths = sorted(root.glob("*/*/metadata.json"))
    if len(paths) != 6:
        raise ValueError(f"Expected frozen six-recording inventory, found {len(paths)}")
    if diagnostics_only:
        paths = [path for path in paths if any(recording == path.parent.relative_to(root).as_posix()
                                              for recording, _ in DIAGNOSTICS)]
    recordings = [load_recording(root, path) for path in paths]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "export-in-progress.json", {"status": "in_progress", "source": str(root)})
    shutil.copyfile(Path(__file__).resolve(), output / "prepare_nakehand_test_snapshot.py")
    shutil.copyfile(Path(__file__).resolve().with_name("audit_nakehand_dataset.py"), output / "audit_nakehand_dataset.py")
    all_sources = [record for recording in recordings for record in recording["sources"]]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
                "sources": all_sources, "recordings": [], "image_outputs": [],
                "source_scope": "used RGB and side-mask videos, recording metadata, SAM3 side metadata/interactions/upstream masks; depth/MANO not used"}
    info = {"description": "nakehand external agreement test against existing SAM3 pseudo-labels",
            "split": "external_test", "label_provenance": "SAM3 text/point prompted video propagation; no independent full-frame manual GT",
            "sampling": {"method": "uniform original frame indices, integer round-half-up linspace, including endpoints",
                         "samples_per_recording": 0 if diagnostics_only else samples_per_recording,
                         "model_independent": True, "diagnostics_only": diagnostics_only,
                         "diagnostics_in_main_average": False},
            "human_review": {"A": "accepted by user", "B": "accepted by user", "C": "accepted by user",
                             "scope": "Only the three named RGB/left-mask/right-mask panels; no edits or full-dataset GT certification"},
            "mask_semantics": "One annotation per nonempty side: union of all source mask values > 0, no area filtering",
            "timestamps": "video_pts_seconds checked against original frame index/nominal FPS; source_timestamp_seconds is separate authoritative capture time",
            "published_only_after_source_rehash_and_full_output_validation": True}
    coco = {"info": info, "categories": CATEGORIES, "images": [], "annotations": []}
    for recording in recordings:
        name, metadata = recording["recording_id"], recording["metadata"]
        primary = set() if diagnostics_only else set(uniform_indices(metadata["frame_count"], samples_per_recording))
        diagnostic = {frame: label for (record, frame), label in DIAGNOSTICS.items() if record == name}
        if any(frame >= metadata["frame_count"] for frame in diagnostic):
            raise ValueError(f"Diagnostic frame outside recording: {name}")
        selected = sorted(primary | set(diagnostic))
        arrays, timestamps = {}, {}
        for stream in ("rgb", "left", "right"):
            arrays[stream], timestamps[stream] = select_frames(recording["paths"][stream], selected, recording["fps"],
                                                             recording["width"], recording["height"], stream != "rgb")
        if not (timestamps["rgb"] == timestamps["left"] == timestamps["right"]):
            raise ValueError(f"RGB and masks have different PTS: {name}")
        view, sequence = name.split("/")
        for row, frame in enumerate(selected):
            image_id = len(coco["images"])
            basename = f"{view}__{sequence}__frame-{frame:06d}"
            rgb_relative = f"images/{basename}.png"
            rgb_path = output / rgb_relative
            rgb_path.parent.mkdir(exist_ok=True)
            Image.fromarray(arrays["rgb"][row]).save(rgb_path)
            frame_meta = metadata["frames"][frame]
            label = diagnostic.get(frame)
            image = {"id": image_id, "file_name": rgb_relative, "width": recording["width"], "height": recording["height"],
                     "source": "nakehand", "sequence": sequence, "view": view,
                     "view_type": {"nakehandego": "ego", "nakehandexo": "exo"}[view],
                     "recording_id": name, "frame_index": frame, "source_frame_index": frame_meta["source_frame_index"],
                     "primary_test": frame in primary, "diagnostic_id": label,
                     "human_review": {"status": "accepted" if label else "not_reviewed", "diagnostic_id": label,
                                      "scope": "RGB/left/right mask panel only; no pixel edits" if label else "none"},
                     "video_pts_seconds": timestamps["rgb"][row], "source_timestamp_seconds": frame_meta["timestamp"],
                     "source_rgb_path": str(recording["paths"]["rgb"]),
                     "source_mask_paths": {side: str(recording["paths"][side]) for side in ("left", "right")},
                     "source_metadata_path": str(recording["paths"]["metadata"]), "source_masks": {}}
            image_files = {"rgb": {"path": rgb_relative, "sha256": sha256(rgb_path)}}
            for side, category_id in (("left", 1), ("right", 2)):
                raw = arrays[side][row]
                sidecar = recording["sidecars"][side]
                chunk = frame_chunk(sidecar["metadata"], frame)
                values = [int(value) for value in np.unique(raw)]
                valid_values = {0} | {int(value) for value in chunk["object_id_to_label"].values()}
                if not set(values) <= valid_values:
                    raise ValueError(f"Source mask values not in declared chunk mapping: {name}/{frame}/{side}")
                raw_relative = f"source-masks/{basename}__{side}_instance_raw.png"
                raw_path = output / raw_relative
                raw_path.parent.mkdir(exist_ok=True)
                Image.fromarray(raw).save(raw_path)
                binary_relative = f"reference-masks/{basename}__{side}.png"
                binary_path = output / binary_relative
                binary_path.parent.mkdir(exist_ok=True)
                Image.fromarray((raw > 0).astype(np.uint8) * 255).save(binary_path)
                provenance = {"path": str(recording["paths"][side]), "raw_instance_png": raw_relative,
                              "binary_reference_png": binary_relative, "source_values": values,
                              "video_pts_seconds": timestamps[side][row], "chunk": chunk,
                              "label_scope": f"{chunk['mapping_scope']}; source object label is not independently verified as a stable physical track identifier",
                              "metadata_path": sidecar["metadata_path"], "interactions_path": sidecar["interactions_path"]}
                image["source_masks"][side] = provenance
                annotation = side_annotation(raw, image_id, category_id, len(coco["annotations"]), provenance)
                if annotation is not None:
                    coco["annotations"].append(annotation)
                image_files[f"{side}_instance_raw"] = {"path": raw_relative, "sha256": sha256(raw_path)}
                image_files[f"{side}_binary"] = {"path": binary_relative, "sha256": sha256(binary_path)}
                if not np.array_equal(np.asarray(Image.open(raw_path)), raw):
                    raise ValueError("Raw instance PNG round-trip mismatch")
                if not np.array_equal(np.asarray(Image.open(binary_path)) > 0, raw > 0):
                    raise ValueError("Binary reference PNG round-trip mismatch")
            if not np.array_equal(np.asarray(Image.open(rgb_path)), arrays["rgb"][row]):
                raise ValueError("RGB PNG round-trip mismatch")
            image["source_left_right_overlap_pixels"] = int(((arrays["left"][row] > 0) & (arrays["right"][row] > 0)).sum())
            coco["images"].append(image)
            manifest["image_outputs"].append({"image_id": image_id, "files": image_files})
        manifest["recordings"].append({"recording_id": name, "frame_count": metadata["frame_count"],
                                       "primary_indices": sorted(primary), "diagnostic_indices": diagnostic,
                                       "selected_indices": selected, "streams": recording["stream_info"],
                                       "source_mask_metadata": recording["sidecars"]})
        del arrays, timestamps
        gc.collect()
        print(f"exported {name}: {len(primary)} primary + {len(set(diagnostic) - primary)} additional diagnostics", flush=True)
    if {image["diagnostic_id"] for image in coco["images"] if image["diagnostic_id"]} != {"A", "B", "C"}:
        raise ValueError("Missing an explicitly reviewed diagnostic")
    annotation_sides = Counter(annotation["category_id"] for annotation in coco["annotations"])
    per_image = Counter(annotation["image_id"] for annotation in coco["annotations"])
    primary_ids = {image["id"] for image in coco["images"] if image["primary_test"]}
    counts = {"images": len(coco["images"]), "primary_images": len(primary_ids),
              "diagnostic_images": 3, "diagnostic_only_images": sum(not image["primary_test"] and bool(image["diagnostic_id"]) for image in coco["images"]),
              "annotations": len(coco["annotations"]), "left_annotations": annotation_sides[1], "right_annotations": annotation_sides[2],
              "primary_empty_images": sum(per_image[image_id] == 0 for image_id in primary_ids),
              "primary_one_hand_images": sum(per_image[image_id] == 1 for image_id in primary_ids),
              "primary_two_hand_images": sum(per_image[image_id] == 2 for image_id in primary_ids)}
    if not diagnostics_only and len(primary_ids) != 6 * samples_per_recording:
        raise ValueError("Main sample count mismatch")
    print(f"rehashing {len(all_sources)} read-only sources before publication", flush=True)
    verify_sources(all_sources)
    manifest.update(status="complete", sources_unchanged=True, counts=counts)
    annotation_path = output / "annotations.json"
    atomic_json(annotation_path, coco)
    manifest["annotations_sha256"] = sha256(annotation_path)
    atomic_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# nakehand external test sample\n\n"
        "This is an external **agreement test against existing SAM3 pseudo-labels**, not independent manually drawn pixel ground truth. "
        "The user accepted only diagnostic panels A/B/C without editing their masks. That review is not extrapolated to other frames.\n\n"
        f"Counts: `{json.dumps(counts)}`.\n\n"
        "Primary frames are selected uniformly by original frame index, independently of model outputs, with equal recording counts and both endpoints. "
        "Additional A/B/C diagnostics have `primary_test=false`; diagnostics already selected systematically retain `primary_test=true`. "
        "Use `primary_test` for the main aggregate and `diagnostic_id` for the separate diagnostic table. "
        "Both physical sides can be present; the other side must not automatically be treated as a negative.\n\n"
        "RGB PNG preserves decoded rgb24 pixels. Each side is the union of *all* positive source mask values; no small-mask filtering. "
        "Raw instance PNG, binary reference PNG, COCO RLE, area/bbox, source chunk mappings, exact video PTS and separate capture timestamps are retained. "
        "Source wrist/forearm boundaries are preserved, not corrected. No depth/MANO inputs, training, or GPU is involved.\n\n"
        "All used RGB/mask videos, recording metadata and upstream mask provenance files were SHA256-hashed before and after export. "
        "Every output RGB/mask PNG and every RLE was checked pixel-exact. See `manifest.json` for provenance, hashes and frame lists.\n\n"
        f"Annotation SHA256: `{manifest['annotations_sha256']}`.\n", encoding="utf-8")
    atomic_json(output / "export-in-progress.json", {"status": "complete", "annotations_sha256": manifest["annotations_sha256"]})
    atomic_json(output / "READY.json", {"status": "complete", "annotations_sha256": manifest["annotations_sha256"],
                                         "manifest_sha256": sha256(output / "manifest.json")})
    print(json.dumps({**counts, "output": str(output), "annotations_sha256": manifest["annotations_sha256"]}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/xuzhefeng/Datasets/wanqing_datasets/nakehand"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-recording", type=int, default=100)
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    export_dataset(args.root, args.output_dir, args.samples_per_recording, args.diagnostics_only)


if __name__ == "__main__":
    main()

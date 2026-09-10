#!/usr/bin/env python3
"""Export a side-aware unified-dataset manifest into a SAM3 COCO dataset."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

from unified_to_sam3 import (
    _RawFrameReader,
    _load_instances,
    _probe_video,
    build_frame_annotations,
)


SIDE_TO_KIND = {"left": "left_hand", "right": "right_hand"}
CATEGORY_IDS = {"left_hand": 1, "right_hand": 2}


def image_file_name(record: Dict, frame_index: int) -> str:
    """Return a collision-free relative JPEG path for one manifest frame."""
    return (
        "images/"
        f"{record['source']}__{record['sequence']}__"
        f"{record['view']}__{frame_index:08d}.jpg"
    )


def export_manifest(
    manifest_path: Path,
    output_dir: Path,
    split: str,
    overwrite: bool = False,
    progress_every: int = 20,
    include_empty: bool = False,
    min_mask_area: int = 1,
) -> Path:
    """Export side-correct hand positives and optional exhaustive negatives."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Manifest must contain a list: {manifest_path}")
    if min_mask_area < 1:
        raise ValueError("min_mask_area must be at least 1")

    annotation_path = output_dir / "annotations.json"
    if annotation_path.exists() and not overwrite:
        raise FileExistsError(annotation_path)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    images: List[Dict] = []
    annotations: List[Dict] = []
    skipped_frames = 0
    skipped_small_masks = 0
    negative_frames = 0
    side_counts = {"left_hand": 0, "right_hand": 0}
    image_id = 0
    annotation_id = 0

    for view_number, record in enumerate(records, start=1):
        side = record.get("hand_side")
        if side not in SIDE_TO_KIND:
            raise ValueError(f"Manifest record has invalid hand_side: {side!r}")
        target_kind = SIDE_TO_KIND[side]
        visible_frames = [int(frame) for frame in record["hand_visible_frames"]]
        if not visible_frames and not include_empty:
            continue
        visible_frame_set = set(visible_frames)
        if len(visible_frames) != len(visible_frame_set):
            raise ValueError(f"Duplicate hand-visible frame in {record['view_dir']}")

        view_dir = Path(record["view_dir"])
        rgb_path = view_dir / "rgb.mkv"
        mask_path = view_dir / "mask.mkv"
        width, height, num_frames = _probe_video(rgb_path)
        mask_width, mask_height, mask_frames = _probe_video(mask_path)
        if (width, height, num_frames) != (
            mask_width,
            mask_height,
            mask_frames,
        ):
            raise ValueError(f"RGB/mask mismatch in {view_dir}")

        start_frame = 0 if include_empty else min(visible_frames)
        end_frame = num_frames - 1 if include_empty else max(visible_frames)
        if start_frame < 0 or end_frame >= num_frames:
            raise ValueError(f"Frame range outside video in {view_dir}")

        instances = _load_instances(Path(record["instances_path"]))
        hand_instances = [
            instance
            for instance in instances
            if str(instance.get("kind", "")).startswith("hand_")
        ]
        if len(hand_instances) != 1 and visible_frames:
            raise ValueError(
                f"Expected exactly one hand instance in {record['instances_path']}"
            )
        # Correct the known legacy unified-export side bug in memory. The mask
        # instance IDs themselves are already side-independent and correct.
        instances = [dict(instance) for instance in instances]
        for instance in instances:
            if str(instance.get("kind", "")).startswith("hand_"):
                instance["kind"] = target_kind
        rgb_reader = _RawFrameReader(
            rgb_path,
            width,
            height,
            start_frame,
            end_frame,
            pixel_format="rgb24",
            channels=3,
            dtype="uint8",
        )
        mask_reader = _RawFrameReader(
            mask_path,
            width,
            height,
            start_frame,
            end_frame,
            pixel_format="gray16le",
            channels=1,
            dtype="<u2",
        )

        try:
            for frame_index in range(start_frame, end_frame + 1):
                rgb = rgb_reader.read()
                instance_mask = mask_reader.read()
                if not include_empty and frame_index not in visible_frame_set:
                    continue

                frame_annotations = build_frame_annotations(
                    instance_mask,
                    frame_index=frame_index,
                    instances=instances,
                    category_ids=CATEGORY_IDS,
                    image_id=image_id,
                )
                if not frame_annotations:
                    if frame_index in visible_frame_set:
                        skipped_frames += 1
                        continue
                    negative_frames += 1
                else:
                    if len(frame_annotations) != 1:
                        raise ValueError(
                            f"Expected one hand annotation in {view_dir} frame {frame_index}"
                        )
                    annotation = frame_annotations[0]
                    if annotation["area"] < min_mask_area:
                        skipped_small_masks += 1
                        continue
                    side_counts[target_kind] += 1

                relative_name = image_file_name(record, frame_index)
                Image.fromarray(rgb, mode="RGB").save(
                    output_dir / relative_name,
                    format="JPEG",
                    quality=95,
                )
                images.append(
                    {
                        "id": image_id,
                        "file_name": relative_name,
                        "width": width,
                        "height": height,
                        "frame_index": frame_index,
                        "source": record["source"],
                        "sequence": record["sequence"],
                        "view": record["view"],
                    }
                )
                for annotation in frame_annotations:
                    annotation["id"] = annotation_id
                    annotation["image_id"] = image_id
                    annotations.append(annotation)
                    annotation_id += 1
                image_id += 1
        finally:
            rgb_reader.close()
            mask_reader.close()

        if view_number == 1 or view_number % progress_every == 0:
            print(
                f"views={view_number}/{len(records)} "
                f"images={len(images)} annotations={len(annotations)}",
                flush=True,
            )

    result = {
        "info": {
            "description": "SAM3 left/right-hand export from the unified dataset",
            "source_manifest": str(manifest_path),
            "split": split,
            "positive_only": not include_empty,
            "negative_frames": negative_frames,
            "min_mask_area": min_mask_area,
            "skipped_frames_below_min_mask_area": skipped_small_masks,
            "skipped_frames_without_mask": skipped_frames,
            "positive_frames_by_kind": side_counts,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": kind, "supercategory": "hand"}
            for kind, category_id in CATEGORY_IDS.items()
        ],
    }
    annotation_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return annotation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include frames with no hand as exhaustive negatives for both sides",
    )
    parser.add_argument("--min-mask-area", type=int, default=1)
    args = parser.parse_args()

    output = export_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
        include_empty=args.include_empty,
        min_mask_area=args.min_mask_area,
    )
    print(f"完成: {output}")


if __name__ == "__main__":
    main()

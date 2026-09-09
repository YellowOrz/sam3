#!/usr/bin/env python3
"""Export a unified-dataset view manifest into one SAM3 COCO dataset."""

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
) -> Path:
    """Export all right-hand-positive frames listed in one manifest."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Manifest must contain a list: {manifest_path}")

    annotation_path = output_dir / "annotations.json"
    if annotation_path.exists() and not overwrite:
        raise FileExistsError(annotation_path)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    images: List[Dict] = []
    annotations: List[Dict] = []
    skipped_frames = 0
    image_id = 0
    annotation_id = 0

    for view_number, record in enumerate(records, start=1):
        visible_frames = [
            int(frame) for frame in record["right_hand_visible_frames"]
        ]
        if not visible_frames:
            continue

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

        start_frame = min(visible_frames)
        end_frame = max(visible_frames)
        if start_frame < 0 or end_frame >= num_frames:
            raise ValueError(f"Frame range outside video in {view_dir}")

        instances = _load_instances(Path(record["instances_path"]))
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
                if frame_index not in visible_frames:
                    continue

                frame_annotations = build_frame_annotations(
                    instance_mask,
                    frame_index=frame_index,
                    instances=instances,
                    category_ids={"right_hand": 1},
                    image_id=image_id,
                )
                if not frame_annotations:
                    skipped_frames += 1
                    continue

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
            "description": "SAM3 right-hand export from the unified dataset",
            "source_manifest": str(manifest_path),
            "split": split,
            "positive_only": True,
            "skipped_frames_without_mask": skipped_frames,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "right_hand", "supercategory": "hand"}
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
    args = parser.parse_args()

    output = export_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        split=args.split,
        overwrite=args.overwrite,
    )
    print(f"完成: {output}")


if __name__ == "__main__":
    main()

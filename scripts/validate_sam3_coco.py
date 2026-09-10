#!/usr/bin/env python3
"""Validate the side-aware SAM3 COCO exports without changing them."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED_CATEGORIES = {1: "left_hand", 2: "right_hand"}


def validate_split(root: Path, split: str) -> dict:
    split_root = root / split
    annotation_path = split_root / "annotations.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(f"{split}: unexpected categories {categories!r}")

    images = data["images"]
    annotations = data["annotations"]
    image_ids = [int(item["id"]) for item in images]
    if image_ids != list(range(len(images))):
        raise ValueError(f"{split}: image ids are not contiguous from zero")
    image_by_id = {int(item["id"]): item for item in images}
    annotation_ids = [int(item["id"]) for item in annotations]
    if annotation_ids != list(range(len(annotations))):
        raise ValueError(f"{split}: annotation ids are not contiguous from zero")

    per_image = Counter()
    per_class = Counter()
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in image_by_id:
            raise ValueError(f"{split}: annotation references image {image_id}")
        if category_id not in EXPECTED_CATEGORIES:
            raise ValueError(f"{split}: unknown category id {category_id}")
        per_image[image_id] += 1
        if per_image[image_id] > 1:
            raise ValueError(f"{split}: more than one hand annotation on image {image_id}")
        segmentation = annotation["segmentation"]
        if not isinstance(segmentation, dict) or "counts" not in segmentation:
            raise ValueError(f"{split}: image {image_id} does not use COCO RLE")
        if int(annotation["area"]) <= 0:
            raise ValueError(f"{split}: non-positive area on annotation {annotation['id']}")
        bbox = annotation["bbox"]
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise ValueError(f"{split}: invalid bbox on annotation {annotation['id']}")
        per_class[EXPECTED_CATEGORIES[category_id]] += 1

    # Decode a deterministic spread of masks. This catches malformed RLE while
    # keeping validation fast enough to run before every training launch.
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        mask_utils = None
    decode_indices = set()
    if annotations:
        decode_indices.update({0, len(annotations) // 2, len(annotations) - 1})
    if mask_utils is not None:
        for index in sorted(decode_indices):
            annotation = annotations[index]
            decoded = mask_utils.decode(annotation["segmentation"])
            if int(decoded.sum()) != int(annotation["area"]):
                raise ValueError(f"{split}: RLE area mismatch on annotation {annotation['id']}")

    missing_files = []
    for item in images:
        image_path = split_root / item["file_name"]
        if not image_path.is_file():
            missing_files.append(str(image_path))
            continue
        with Image.open(image_path) as image:
            if image.size != (int(item["width"]), int(item["height"])):
                raise ValueError(f"{split}: image size mismatch for {image_path}")
    if missing_files:
        raise FileNotFoundError(f"{split}: missing {len(missing_files)} image files")

    negative_images = len(images) - len(per_image)
    info = data.get("info", {})
    if int(info.get("negative_frames", negative_images)) != negative_images:
        raise ValueError(f"{split}: negative frame count disagrees with annotations")
    return {
        "images": len(images),
        "annotations": len(annotations),
        "negative_images": negative_images,
        "positive_images_by_class": dict(per_class),
        "rle_decode_checked": mask_utils is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = {split: validate_split(args.root, split) for split in ("train", "val", "test")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

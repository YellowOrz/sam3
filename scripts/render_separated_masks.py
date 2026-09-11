"""Render scientific RGB/GT/prediction artifacts without colored overlays.

``render_montages`` accepts the evaluator's existing keyword-only interface.
Every sample gets an untouched decoded RGB PNG, a physical-hand GT PNG and,
for every model/prompt, separate candidate and detection-thresholded PNGs.
Masks are grayscale 0/255 with no text drawn over their pixels. Low-confidence
candidates remain inspectable, while their detected masks are entirely black.
Comparison sheets add labels outside the image area and quote existing record
metrics without recomputing or relabeling them. Opposite-prompt sheets explicitly
distinguish the physical GT reference from that prompt's empty target.

Only NumPy/Pillow and the standard library are used; no model, torch or GPU is
loaded. COCO compressed/uncompressed RLE is supported. Polygon annotations are
rejected rather than rasterized with a different polygon convention.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("left_hand", "right_hand")
SIDE_BY_CATEGORY = {1: "left_hand", 2: "right_hand"}


def binary_mask(mask, height: int, width: int, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != (height, width):
        raise ValueError(f"{name}: expected {(height, width)}, got {array.shape}")
    if array.dtype.kind not in "buif" or not bool(np.all((array == 0) | (array == 1))):
        raise ValueError(f"{name}: expected a bool or 0/1 mask")
    return array.astype(bool, copy=True)


def decode_physical_gt(annotation: dict | None, height: int, width: int) -> np.ndarray:
    if annotation is None:
        return np.zeros((height, width), dtype=bool)
    rle = annotation.get("segmentation")
    if not isinstance(rle, dict) or rle.get("size") != [height, width]:
        raise ValueError("Expected COCO RLE segmentation with the original [height,width]")
    encoded = rle.get("counts")
    if isinstance(encoded, bytes):
        encoded = encoded.decode("ascii")
    if isinstance(encoded, str):
        # COCO maskApi's signed 5-bit variable-length/delta run-length format.
        counts = []
        position = 0
        while position < len(encoded):
            number = shift = 0
            while True:
                if position >= len(encoded):
                    raise ValueError("Truncated compressed COCO RLE")
                code = ord(encoded[position]) - 48
                position += 1
                if not 0 <= code <= 63:
                    raise ValueError("Invalid compressed COCO RLE character")
                number |= (code & 31) << shift
                shift += 5
                if shift > 65:
                    raise ValueError("Oversized compressed COCO RLE run")
                if not code & 32:
                    if code & 16:
                        number |= -1 << shift
                    break
            if len(counts) > 2:
                number += counts[-2]
            counts.append(number)
    elif isinstance(encoded, list):
        counts = encoded
    else:
        raise ValueError("Expected COCO RLE counts string or list")
    if any(type(count) is not int or count < 0 for count in counts) or sum(counts) != height * width:
        raise ValueError("Invalid COCO RLE run lengths/total size")
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[offset:offset + count] = True
        offset += count
    return flat.reshape((height, width), order="F")


def _mask_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255)


def _font(size: int):
    for filename in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(filename).is_file():
            return ImageFont.truetype(filename, size), True
    for filename in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(filename, size), False
        except OSError:
            pass
    return ImageFont.load_default(), False


def _metric(record: dict, name: str) -> str:
    value = record.get(name)
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Invalid existing metric {name}: {value!r}")
    return f"{value:.4f}"


def _labelled_tile(pixels: Image.Image, lines: list[str], font, min_width: int) -> Image.Image:
    # All text is in a new top margin; original pixels are not covered/resized.
    line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 7, 24)
    margin = line_height * len(lines) + 12
    tile = Image.new("RGB", (max(pixels.width, min_width), pixels.height + margin), (24, 24, 24))
    tile.paste(pixels.convert("RGB"), (0, margin))
    draw = ImageDraw.Draw(tile)
    for line_index, line in enumerate(lines):
        draw.text((7, 5 + line_index * line_height), line, font=font, fill="white")
    return tile


def _comparison_sheet(rgb, gt, predictions, model_labels, prompt, *, opposite: bool, empty: bool):
    font, chinese = _font(17)
    sheet_rows = []
    for label in model_labels:
        record, candidate, detected = predictions[(label, prompt)]
        if empty:
            gt_note = "该图无手；查询目标为空" if chinese else "No physical hand; prompt target is empty"
        elif opposite:
            gt_note = "物理手参考；本查询目标应为空" if chinese else "Physical GT reference; THIS prompt target is empty"
        else:
            gt_note = "物理手GT = 本查询目标" if chinese else "Physical-hand GT = this prompt's target"
        headings = [
            ["原图 RGB" if chinese else "Original RGB", str(label),
             f"prompt={record.get('prompt_text', prompt)}"],
            ["物理手 GT (白=手)" if chinese else "Physical-hand GT (white=hand)", gt_note, ""],
            ["原始候选 mask" if chinese else "Raw top candidate mask",
             f"score={_metric(record, 'top_confidence')}  detected={int(record['detected'])}",
             f"Dice vs physical GT={_metric(record, 'top_dice_with_physical_hand')}"],
            ["阈值后预测 mask" if chinese else "Detection-thresholded mask",
             f"score={_metric(record, 'top_confidence')}  detected={int(record['detected'])}",
             f"Dice vs physical GT={_metric(record, 'detected_dice_with_physical_hand')}"],
        ]
        pixel_images = [rgb, _mask_image(gt), _mask_image(candidate), _mask_image(detected)]
        minimum_width = max(420, rgb.width)
        tiles = [_labelled_tile(pixels, lines, font, minimum_width) for pixels, lines in zip(pixel_images, headings)]
        width, height = sum(tile.width for tile in tiles), max(tile.height for tile in tiles)
        row_image = Image.new("RGB", (width, height), (24, 24, 24))
        offset = 0
        for tile in tiles:
            row_image.paste(tile, (offset, 0))
            offset += tile.width
        sheet_rows.append(row_image)
    canvas = Image.new("RGB", (max(row.width for row in sheet_rows), sum(row.height for row in sheet_rows)), (24, 24, 24))
    offset = 0
    for row in sheet_rows:
        canvas.paste(row, (0, offset))
        offset += row.height
    return canvas


def render_montages(
    *, root: Path, output_dir: Path, render_indices: Sequence[int],
    images: Sequence[dict], annotations_by_image: dict[int, dict | None],
    records: Sequence[dict], masks: dict[tuple[str, int, str], np.ndarray],
    model_labels: Sequence[str],
) -> list[dict]:
    """Drop-in evaluator renderer; return full per-sample artifact manifest."""
    root, output_dir = Path(root), Path(output_dir)
    if not model_labels or len(model_labels) != len(set(model_labels)):
        raise ValueError("Need unique, nonempty model labels")
    if len(render_indices) != len(set(render_indices)):
        raise ValueError("Duplicate render indices")
    records_by_key = {}
    for record in records:
        key = (record["model"], int(record["dataset_index"]), record["prompt_key"])
        if key in records_by_key:
            raise ValueError(f"Duplicate prediction record: {key}")
        records_by_key[key] = record
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dataset_index in render_indices:
        if not 0 <= dataset_index < len(images):
            raise IndexError(dataset_index)
        image_info = images[dataset_index]
        image_id = int(image_info["id"])
        annotation = annotations_by_image[image_id]
        side = "empty" if annotation is None else SIDE_BY_CATEGORY[annotation["category_id"]]
        rgb_path = root / image_info["file_name"]
        with Image.open(rgb_path) as original:
            rgb = original.convert("RGB")
        height, width = rgb.height, rgb.width
        if image_info.get("height", height) != height or image_info.get("width", width) != width:
            raise ValueError(f"RGB/COCO dimensions differ: {rgb_path}")
        gt = decode_physical_gt(annotation, height, width)
        predictions = {}
        for label in model_labels:
            for prompt in CLASS_NAMES:
                key = (label, dataset_index, prompt)
                record = records_by_key[key]
                if type(record.get("detected")) is not bool:
                    raise ValueError(f"Record needs boolean detected: {key}")
                if int(record.get("image_id", image_id)) != image_id or record.get("actual_side", side) != side:
                    raise ValueError(f"Record/GT identity mismatch: {key}")
                if record.get("target_present", side == prompt) is not (side == prompt):
                    raise ValueError(f"Record target/GT side mismatch: {key}")
                candidate = binary_mask(masks[key], height, width, str(key))
                detected = candidate.copy() if record["detected"] else np.zeros_like(candidate)
                predictions[(label, prompt)] = (record, candidate, detected)
        sample_dir = output_dir / f"idx-{dataset_index:05d}__image-{image_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgb_output, gt_output = sample_dir / "rgb.png", sample_dir / "gt_physical_hand.png"
        rgb.save(rgb_output)
        _mask_image(gt).save(gt_output)
        model_outputs = {}
        for model_index, label in enumerate(model_labels):
            safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-.") or "model"
            model_dir = sample_dir / f"model-{model_index:02d}__{safe_label}"
            model_dir.mkdir(exist_ok=True)
            model_outputs[label] = {}
            for prompt in CLASS_NAMES:
                record, candidate, detected = predictions[(label, prompt)]
                candidate_path = model_dir / f"{prompt}__candidate.png"
                detected_path = model_dir / f"{prompt}__detected.png"
                _mask_image(candidate).save(candidate_path)
                _mask_image(detected).save(detected_path)
                model_outputs[label][prompt] = {
                    "candidate": str(candidate_path.resolve()),
                    "detected": str(detected_path.resolve()),
                    "record": record,
                }
        correct_prompt = side if side != "empty" else "left_hand"
        opposite_prompt = "right_hand" if correct_prompt == "left_hand" else "left_hand"
        sheets = {}
        for role, prompt, opposite in (("correct", correct_prompt, False), ("opposite", opposite_prompt, True)):
            # No correct side exists on an empty image: label both queries negative.
            semantic_role = role if side != "empty" else f"{prompt}_negative"
            sheet_path = sample_dir / f"{semantic_role}_prompt_comparison.png"
            _comparison_sheet(rgb, gt, predictions, model_labels, prompt, opposite=opposite, empty=side == "empty").save(sheet_path)
            sheets[semantic_role] = {
                "path": str(sheet_path.resolve()), "prompt_key": prompt,
                "prompt_target": "physical_hand" if role == "correct" and side != "empty" else "empty",
                "gt_column": "physical_hand_reference",
            }
        first_record = predictions[(model_labels[0], CLASS_NAMES[0])][0]
        manifest.append({
            "dataset_index": dataset_index, "image_id": image_id, "actual_side": side,
            "file_name": image_info["file_name"], "source_rgb": str(rgb_path.resolve()),
            "source": image_info.get("source"), "sequence": image_info.get("sequence"),
            "view": image_info.get("view"), "frame_index": image_info.get("frame_index"),
            "original_mapping": first_record.get("original_mapping"),
            "image_metadata": image_info,
            "rgb": str(rgb_output.resolve()), "gt_physical_hand": str(gt_output.resolve()),
            "models": model_outputs, "comparison_sheets": sheets,
            "output": next(iter(sheets.values()))["path"],
            "mask_encoding": "PNG grayscale: background=0, hand=255; no text/overlay",
            "candidate_semantics": "Raw highest-score candidate, preserved even below detection threshold",
            "detected_semantics": "Candidate if record.detected=true, otherwise an all-black mask",
            "dice_semantics": "Existing record Dice vs physical hand GT; opposite prompt target is empty",
        })
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    return manifest


render_separated_masks = render_montages

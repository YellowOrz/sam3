"""Render explicitly selected audit outliers without changing labels or inputs."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.prepare_realsense_manual_review import MASK_NAMES, panel, stable_record, verify
from scripts.audit_nakehand_dataset import extract, probe


def validate_mask_stream(path, width, height, fps, count):
    stream = probe(path, count_packets=True)["streams"][0]
    actual = (stream["width"], stream["height"], stream["pix_fmt"],
              float(Fraction(stream["avg_frame_rate"])), int(stream["nb_read_packets"]))
    if actual != (width, height, "gray", fps, count):
        raise ValueError(f"Mask native shape/type/rate/count mismatch: {path}: {actual}")


def overlap_or_unknown(masks):
    if "left_hand" not in masks or "right_hand" not in masks:
        return None
    return int(((masks["left_hand"] > 0) & (masks["right_hand"] > 0)).sum())


def save_lossless(path, array):
    Image.fromarray(array).save(path)
    with Image.open(path) as image:
        if not np.array_equal(np.asarray(image), array):
            raise ValueError(f"PNG pixel round trip failed: {path}")


def validate_samples(samples):
    if not 1 <= len(samples) <= 5:
        raise ValueError("Bounded diagnostic review requires one to five selected frames")
    ids = []
    for sample in samples:
        for key in ("review_id", "recording"):
            if not isinstance(sample.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample[key]):
                raise ValueError(f"Simple safe {key} required")
        if type(sample.get("frame_index")) is not int or sample["frame_index"] < 0:
            raise ValueError("Nonnegative integer frame index required")
        if len(sample.get("extra_masks", [])) > 2:
            raise ValueError("At most two extra diagnostic masks")
        ids.append(sample["review_id"].casefold())
    if len(ids) != len(set(ids)):
        raise ValueError("Unique portable review IDs required")


def render(plan_path, output):
    plan_path, output = plan_path.resolve(), output.resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan["source_root"]).resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Output must be new and outside source data")
    validate_samples(plan["samples"])
    sources = {str(plan_path): stable_record(plan_path)}
    for name in ("render_realsense_diagnostics.py", "prepare_realsense_manual_review.py", "audit_nakehand_dataset.py"):
        path = Path(__file__).resolve().with_name(name)
        sources[str(path)] = stable_record(path)
    for sample in plan["samples"]:
        directory = root / sample["recording"]
        for path in [directory / "color.mp4", *[directory / "masks_sam3" / f"{role}.mkv" for role in MASK_NAMES],
                     *[Path(item["path"]) for item in sample.get("extra_masks", [])]]:
            if path.is_file():
                sources[str(path)] = stable_record(path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, output / "selection.json")
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    result = {"source_records": list(sources.values()), "selection_rule": plan["selection_rule"],
              "not_representative_random_sample": True, "human_review_status": "pending", "samples": []}
    for sample in plan["samples"]:
        recording, frame = sample["recording"], sample["frame_index"]
        directory = root / recording
        meta = probe(directory / "color.mp4")["streams"][0]
        w, h = meta["width"], meta["height"]
        fps = float(Fraction(meta["avg_frame_rate"]))
        if not 0 <= frame < int(meta["nb_frames"]):
            raise ValueError("Diagnostic frame outside source")
        rgb, rgb_pts = extract(directory / "color.mp4", frame, fps, w, h)
        masks, pts = {}, {"rgb": rgb_pts}
        for role in MASK_NAMES:
            path = directory / "masks_sam3" / f"{role}.mkv"
            if path.exists():
                validate_mask_stream(path, w, h, fps, int(meta["nb_frames"]))
                masks[role], pts[role] = extract(path, frame, fps, w, h, True)
        extra = []
        for item in sample.get("extra_masks", []):
            validate_mask_stream(Path(item["path"]), w, h, fps, int(meta["nb_frames"]))
            array, timestamp = extract(Path(item["path"]), frame, fps, w, h, True)
            extra.append((item["label"], array))
            pts[item["label"]] = timestamp
        if max(pts.values()) - min(pts.values()) > .0012:
            raise ValueError("Diagnostic streams mismatch PTS")
        base = panel(rgb, masks, sample)
        sheet = Image.new("RGB", ((4 + len(extra)) * w, base.height), "white")
        sheet.paste(base, (0, 0))
        header = base.height - h
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        for i, (label, mask) in enumerate(extra, 4):
            draw.text((i*w+8, 10), label, font=font, fill="black")
            draw.text((i*w+8, 40), "Prior export, NOT a new model prediction", font=font, fill="black")
            sheet.paste(Image.fromarray((mask > 0).astype(np.uint8)*255).convert("RGB"), (i*w, header))
        sub = output / sample["review_id"]
        if not sub.resolve().is_relative_to(output):
            raise ValueError("Diagnostic output escaped target directory")
        sub.mkdir()
        save_lossless(sub / "rgb.png", rgb)
        for role, raw in masks.items():
            save_lossless(sub / f"{role}_raw.png", raw)
            save_lossless(sub / f"{role}_binary.png", (raw > 0).astype(np.uint8)*255)
        for i, (_, raw) in enumerate(extra):
            save_lossless(sub / f"extra_{i}_raw.png", raw)
            save_lossless(sub / f"extra_{i}_binary.png", (raw > 0).astype(np.uint8)*255)
        sheet.save(sub / "comparison.png")
        pngs = [stable_record(path) for path in sorted(sub.glob("*.png"))]
        result["samples"].append({**sample, "pts_seconds": pts, "left_right_overlap_pixels": overlap_or_unknown(masks),
                                  "roles_provided": {role: role in masks for role in MASK_NAMES},
                                  "files": pngs, "comparison": f"{sample['review_id']}/comparison.png"})
    verify(list(sources.values()))
    result["sources_unchanged"] = True
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"images": len(render(args.plan, args.output_dir)["samples"]), "output": str(args.output_dir)}))

"""Read-only stratified source-mask review, not a dataset export or model run.

Choose one frame per recording BEFORE looking at RGB/masks. Keep missing mask
streams visibly distinct from empty masks. Write unmodified RGB and native mask
PNGs plus separate monochrome panels, never blended annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.audit_nakehand_dataset import extract, probe, sha256


MASK_NAMES = ("left_hand", "right_hand", "object")


def sample_frames(counts: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    if not counts or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError("Positive integer recording counts required")
    return [{"review_id": f"{i:02d}", "recording": name,
             "frame_index": rng.randrange(counts[name])}
            for i, name in enumerate(sorted(counts), 1)]


def panel(rgb: np.ndarray, masks: dict, sample: dict) -> Image.Image:
    h, w = rgb.shape[:2]
    header = 82
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    result = Image.new("RGB", (4 * w, h + header), "white")
    draw = ImageDraw.Draw(result)
    title = f"{sample['review_id']} | {sample['recording']} | source frame {sample['frame_index']} (zero-based)"
    draw.text((8, 5), title, font=font, fill="black")
    result.paste(Image.fromarray(rgb), (0, header))
    draw.text((8, 35), "Original RGB", font=font, fill="black")
    for i, name in enumerate(MASK_NAMES, 1):
        x = i * w
        raw = masks.get(name)
        if raw is None:
            result.paste(Image.new("RGB", (w, h), (160, 160, 160)), (x, header))
            draw.text((x + 15, header + 30), "NO SOURCE MASK FILE", font=font, fill="black")
            draw.text((x + 15, header + 60), "Not an empty/negative annotation", font=small, fill="black")
        else:
            result.paste(Image.fromarray((raw > 0).astype(np.uint8) * 255).convert("RGB"), (x, header))
        draw.text((x + 8, 35), f"{name}: source mask (white = >0)", font=font, fill="black")
        draw.text((x + 8, 56), "Anatomical side and annotation quality need review", font=small, fill=(70, 70, 70))
    return result


def stable_record(path: Path) -> dict:
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest}


def verify(records: list[dict]) -> None:
    for record in records:
        if stable_record(Path(record["path"])) != record:
            raise ValueError(f"Source changed during review: {record['path']}")


def render(root: Path, output: Path, seed: int) -> dict:
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents or output.exists():
        raise ValueError("Use a new directory outside the source dataset")
    counts, recordings, sources = {}, {}, []
    for meta in sorted(root.glob("*/meta.json")):
        info = json.loads(meta.read_text())
        name, directory = meta.parent.name, meta.parent
        counts[name] = info["frames"]
        streams = {"rgb": directory / "color.mp4"}
        streams.update({name: directory / "masks_sam3" / f"{name}.mkv" for name in MASK_NAMES})
        existing = {key: path for key, path in streams.items() if path.is_file()}
        if "rgb" not in existing:
            raise ValueError(f"Missing RGB: {directory}")
        video = probe(existing["rgb"])["streams"][0]
        if int(video["nb_frames"]) != counts[name]:
            raise ValueError(f"RGB metadata frame count mismatch: {name}")
        w, h, fps = video["width"], video["height"], float(Fraction(video["avg_frame_rate"]))
        for key, path in existing.items():
            if key != "rgb":
                details = probe(path, count_packets=True)["streams"][0]
                if (details["width"], details["height"], float(Fraction(details["avg_frame_rate"])),
                    int(details["nb_read_packets"]), details["pix_fmt"]) != (w, h, fps, counts[name], "gray"):
                    raise ValueError(f"Mask stream incompatible with RGB: {path}")
            sources.append(stable_record(path))
        sources.append(stable_record(meta))
        recordings[name] = {"streams": existing, "width": w, "height": h, "fps": fps}
    samples = sample_frames(counts, seed)
    output.mkdir(parents=True, exist_ok=False)
    plan = {"format": "realsense-manual-review-plan-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(root), "seed": seed, "frame_counts": counts, "samples": samples,
            "selection": "random.Random(seed).randrange(count), sorted recording names; one per recording; selected before pixel decode",
            "selection_uses_mask_quality_or_presence": False, "sources": sources}
    with (output / "frozen-plan.json").open("x") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    helpers = [Path(__file__).resolve(), Path(__file__).with_name("audit_nakehand_dataset.py")]
    source_code = []
    for helper in helpers:
        shutil.copyfile(helper, output / helper.name)
        source_code.append(stable_record(helper))
    result = {"format": "realsense-manual-review-v1", "plan_sha256": sha256(output / "frozen-plan.json"),
              "sources": sources, "code": source_code, "samples": [], "pages": [],
              "human_review_status": "pending", "model_or_gpu_used": False,
              "notes": "These are source reference masks, not new predictions or independently verified human GT. Missing stream is UNKNOWN, not empty."}
    sheets = []
    for sample in samples:
        rec = recordings[sample["recording"]]
        rgb = None
        masks, timestamps = {}, {}
        for name, path in rec["streams"].items():
            values, pts = extract(path, sample["frame_index"], rec["fps"], rec["width"], rec["height"], gray=name != "rgb")
            timestamps[name] = pts
            if name == "rgb":
                rgb = values
            else:
                masks[name] = values
        if max(timestamps.values()) - min(timestamps.values()) > .0012:
            raise ValueError(f"Cross-stream PTS mismatch: {sample}")
        subdir = output / "samples" / sample["review_id"]
        subdir.mkdir(parents=True)
        arrays = {"rgb.png": rgb}
        statistics = {}
        for name in MASK_NAMES:
            raw = masks.get(name)
            statistics[name] = {"provided": raw is not None}
            if raw is not None:
                arrays[f"{name}_raw.png"] = raw
                arrays[f"{name}_binary.png"] = (raw > 0).astype(np.uint8) * 255
                statistics[name].update(values=np.unique(raw).tolist(), pixels=int((raw > 0).sum()))
        files = {}
        for filename, array in arrays.items():
            path = subdir / filename
            Image.fromarray(array).save(path)
            with Image.open(path) as decoded:
                if not np.array_equal(array, np.asarray(decoded)):
                    raise ValueError(f"PNG pixel round trip failed: {path}")
            files[filename] = {"relative_path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
        sheet = panel(rgb, masks, sample)
        sheet.save(subdir / "comparison.png")
        sheets.append(sheet)
        result["samples"].append({**sample, "pts_seconds": timestamps, "masks": statistics, "files": files,
                                  "comparison": (subdir / "comparison.png").relative_to(output).as_posix()})
        print(f"review {sample['review_id']} {sample['recording']} frame={sample['frame_index']}", flush=True)
    for start in range(0, len(sheets), 5):
        thumbnails = [im.resize((1600, round(im.height * 1600 / im.width)), Image.Resampling.LANCZOS) for im in sheets[start:start + 5]]
        page = Image.new("RGB", (1600, sum(im.height for im in thumbnails)), "white")
        y = 0
        for im in thumbnails:
            page.paste(im, (0, y))
            y += im.height
        path = output / f"overview-{start // 5 + 1:02d}.png"
        page.save(path)
        result["pages"].append({"relative_path": path.name, "sha256": sha256(path),
                                "warning": "Overview resized for display; evaluate exact boundaries on original-size individual PNGs."})
    verify(sources + source_code)
    result["source_hashes_unchanged_after_render"] = True
    with (output / "manifest.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# RealSense 数据人工复核（尚待用户确认）", "", "每行依次为原始 RGB、左手、右手、物体参考 mask。白色为所有正标签的并集；灰色表示没有提供该文件，**不是负样本**。没有运行模型或改变源标注。", "", "随机方案在查看像素前固定：每段一张，共十张。请按人体自身左右判断，而非画面左右；分别看手指/腕部、物体边缘及遮挡。此轮确认不自动涵盖深度、MANO、物理接触或整段录像。", ""]
    for entry in result["pages"]:
        lines.extend([f"![分离复核页]({entry['relative_path']})", ""])
    lines.extend(["## 原尺寸复核与反馈", "", "可回复：`01–10 手 mask 可接受；03 物体漏分；07 左右不确定`。手和物体请分开评价。", ""])
    for sample in result["samples"]:
        lines.append(f"- {sample['review_id']}：{sample['recording']} / 零基帧 {sample['frame_index']}；[原尺寸四列]({sample['comparison']})；[原图]({sample['files']['rgb.png']['relative_path']})。")
    (output / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    result = render(args.root, args.output_dir, args.seed)
    print(json.dumps({"samples": len(result["samples"]), "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

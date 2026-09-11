"""CPU source-video context for a disputed mask; no predictions or label edits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.audit_nakehand_dataset import probe
from scripts.prepare_nakehand_test import select_frames
from scripts.prepare_realsense_manual_review import stable_record, verify
from scripts.render_realsense_diagnostics import save_lossless, validate_mask_stream


def encode_video(path, frames, fps):
    first = frames[0]
    w, h = first.width, first.height
    if w % 2 or h % 2:
        raise ValueError("Even review-video dimensions required")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.asarray(frame.convert("RGB")).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait(timeout=60) != 0:
            raise ValueError(error)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stderr.close()
    metadata = probe(path)["streams"][0]
    if int(metadata["nb_frames"]) != len(frames):
        raise ValueError("Encoded review clip lost frames")


def render(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Use a new directory outside source dataset")
    # Predeclared, bounded source context around basket/609. No quality filtering.
    indices = list(range(570, 646))
    still_indices = [570, 585, 600, 609, 615, 630, 645]
    paths = {"rgb": root / "basket/color.mp4", "right": root / "basket/masks_sam3/right_hand.mkv",
             "object": root / "basket/masks_sam3/object.mkv"}
    sources = [stable_record(path) for path in paths.values()]
    helpers = [Path(__file__).resolve().with_name(name) for name in (
        "render_realsense_temporal_review.py", "prepare_nakehand_test.py", "audit_nakehand_dataset.py",
        "prepare_realsense_manual_review.py", "render_realsense_diagnostics.py")]
    sources += [stable_record(path) for path in helpers]
    rgb_meta = probe(paths["rgb"])["streams"][0]
    if (rgb_meta["width"], rgb_meta["height"], rgb_meta["avg_frame_rate"], int(rgb_meta["nb_frames"])) != (640, 480, "30/1", 733):
        raise ValueError("Frozen basket RGB contract changed")
    arrays, pts = {}, {}
    for role, path in paths.items():
        if role != "rgb":
            validate_mask_stream(path, 640, 480, 30., 733)
        arrays[role], pts[role] = select_frames(path, indices, 30., 640, 480, gray=role != "rgb")
    for role in ("right", "object"):
        if max(abs(a-b) for a,b in zip(pts["rgb"], pts[role])) > .0012:
            raise ValueError("Context RGB-mask PTS mismatch")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    clips = {key: [] for key in ("rgb_slow", "right_slow", "object_slow", "separated_slow", "crop_separated_slow")}
    stills = []
    for offset, index in enumerate(indices):
        tiles = {"rgb": Image.fromarray(arrays["rgb"][offset])}
        for role in ("right", "object"):
            tiles[role] = Image.fromarray((arrays[role][offset] > 0).astype(np.uint8)*255).convert("RGB")
        for role, title in (("rgb", "Source RGB"), ("right", "RIGHT source mask >0"), ("object", "OBJECT source mask >0")):
            frame = Image.new("RGB", (640, 540), "white")
            frame.paste(tiles[role], (0, 60))
            draw = ImageDraw.Draw(frame)
            draw.text((8, 5), f"basket | frame {index} | source time {index/30:.3f}s", font=font, fill="black")
            draw.text((8, 32), f"{title} | slow playback 10 fps (source 30)", font=font, fill="black")
            clips[role + "_slow"].append(frame)
        whole = Image.new("RGB", (1920, 540), "white")
        cropped = Image.new("RGB", (1320, 620), "white")
        crop_draw = ImageDraw.Draw(cropped)
        for column, role in enumerate(("rgb", "right", "object")):
            whole.paste(clips[role + "_slow"][-1], (column*640, 0))
            crop = tiles[role].crop((0, 200, 220, 480)).resize((440, 560), Image.Resampling.LANCZOS if role == "rgb" else Image.Resampling.NEAREST)
            cropped.paste(crop, (column*440, 60))
            crop_draw.text((column*440+8, 5), f"{role.upper()} | frame {index}", font=font, fill="black")
            crop_draw.text((column*440+8, 32), "ROI x[0,220) y[200,480), 2x", font=font, fill="black")
        clips["separated_slow"].append(whole)
        clips["crop_separated_slow"].append(cropped)
        if index in still_indices:
            sub = output / f"frame-{index:06d}"
            sub.mkdir()
            for role in ("rgb", "right", "object"):
                save_lossless(sub / f"{role}_raw.png", arrays[role][offset])
            whole.save(sub / "separated.png")
            cropped.save(sub / "crop_separated.png")
            stills.append({"frame_index": index, "full": f"{sub.name}/separated.png", "crop": f"{sub.name}/crop_separated.png"})
    for name, frames in clips.items():
        encode_video(output / f"{name}.mp4", frames, 10)
    verify(sources)
    result = {"format": "realsense-source-temporal-review-v1", "source_records": sources, "sources_unchanged": True,
              "recording": "basket", "frame_indices": indices, "source_fps": 30, "playback_fps": 10,
              "source_pts_seconds": pts, "roi_xyxy": [0, 200, 220, 480], "stills": stills,
              "no_model_inference_or_label_edit": True, "human_review_status": "pending",
              "display_limit": "MP4 is H264 CRF18/yuv420p display re-encoding, not pixel ground truth; independent raw PNGs preserve decoded source arrays; cropped RGB is Lanczos zoom and masks nearest-neighbor zoom",
              "videos": [stable_record(output / f"{name}.mp4") for name in clips]}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"frames": len(render(args.root, args.output_dir)["frame_indices"]), "output": str(args.output_dir)}))

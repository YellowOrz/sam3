"""Full recording and frame-aligned source-mask review, without label changes.

The source file names left/right are annotation names, not a new anatomical
judgment by this tool. All new videos are display encodings, never new GT.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.audit_nakehand_dataset import probe
from scripts.prepare_nakehand_test import select_frames
from scripts.prepare_realsense_manual_review import stable_record, verify
from scripts.render_realsense_diagnostics import save_lossless

FRAME_COUNT = 3449
STILLS = (0, 5, 10, 17, 18, 19, 20, 30, 60, 90, 120, 180, 299)
RECORDING = "nakehandego/20260907_142020"


def transcode_command(source, destination, *, mask=False, slow=False):
    filters = ["format=gray", "lut=y='if(gt(val,0),255,0)'"] if mask else []
    if slow:
        # First 300 consecutive frames: 10 seconds source, 30 seconds playback.
        filters += ["trim=start_frame=0:end_frame=300", "setpts=3*(PTS-STARTPTS)",
                    "pad=iw:ih+56:0:56:color=white",
                    "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                    "text='SOURCE frame %{n} | 3x slow | side unconfirmed':x=8:y=18:fontsize=18:fontcolor=black"]
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source), "-an"]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += ["-r", "10" if slow else "30", "-fps_mode", "cfr", "-c:v", "libx264",
                "-threads", "2", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(destination)]
    return command


def render(root, data_root, output):
    root, data_root, output = (path.resolve() for path in (root, data_root, output))
    if output.exists() or any(output == source or source in output.parents for source in (root, data_root)):
        raise ValueError("Require a new output directory outside source data")
    directory = root / RECORDING
    paths = {"rgb": directory / "rgb.mkv", "left_reference": directory / "masks_sam3/left_hand.mkv",
             "right_reference": directory / "masks_sam3/right_hand.mkv"}
    sources = [stable_record(path) for path in [*paths.values(), directory / "metadata.json",
               data_root / "val/annotations.json", data_root / "val/manifest.json", Path(__file__).resolve()]]
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["frame_count"] != FRAME_COUNT:
        raise ValueError("Recording frame count differs")
    for role, path in paths.items():
        stream = probe(path)["streams"][0]
        if (stream["width"], stream["height"], stream["avg_frame_rate"]) != (640, 480, "30/1"):
            raise ValueError(f"Source stream dimensions/fps differ: {role}")
    data = json.loads((data_root / "val/annotations.json").read_text())
    images = {row["source_frame_index"]: row for row in data["images"] if row["recording_id"] == RECORDING}
    if set(images) != set(range(FRAME_COUNT)):
        raise ValueError("Source recording and full validation image identities differ")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    shutil.copy2(directory / "metadata.json", output / "source_metadata.json")
    video_rows = []

    def encode(item):
        role, slow = item
        target = output / f"{role}_{'first300_slow' if slow else 'full'}.mp4"
        command = transcode_command(paths[role], target, mask=role != "rgb", slow=slow)
        subprocess.run(command, check=True, timeout=600, capture_output=True)
        stream = probe(target)["streams"][0]
        expected_frames, expected_fps = (300, "10/1") if slow else (FRAME_COUNT, "30/1")
        if int(stream["nb_frames"]) != expected_frames or stream["avg_frame_rate"] != expected_fps:
            raise ValueError(f"Display encoding dropped/repeated frames: {target}")
        return {"role": role, "slow": slow, "frames": expected_frames, "fps": expected_fps,
                "file": target.name, "command": command, **stable_record(target)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        video_rows = list(pool.map(encode, [(role, slow) for slow in (False, True) for role in paths]))
    arrays, pts = {}, {}
    for role, path in paths.items():
        arrays[role], pts[role] = select_frames(path, list(STILLS), 30., 640, 480, gray=role != "rgb")
    if pts["rgb"] != pts["left_reference"] or pts["rgb"] != pts["right_reference"]:
        raise ValueError("Source RGB/mask still PTS differ")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    still_rows = []
    contact = Image.new("RGB", (960, len(STILLS) * 268), "white")
    for position, index in enumerate(STILLS):
        frame_dir = output / f"frame-{index:06d}"
        frame_dir.mkdir()
        sheet = Image.new("RGB", (1920, 536), "white")
        row = images[index]
        for column, role in enumerate(paths):
            raw = arrays[role][position]
            if role == "rgb":
                derived = data_root / "val" / row["file_name"]
                with Image.open(derived) as stored:
                    if not np.array_equal(raw, np.asarray(stored.convert("RGB"))):
                        raise ValueError("Selected raw RGB differs from exported validation frame")
                display = Image.fromarray(raw)
            else:
                side = role.split("_")[0]
                derived = data_root / "val" / row["source_masks"][side]["raw_instance_png"]
                with Image.open(derived) as stored:
                    if not np.array_equal(raw, np.asarray(stored)):
                        raise ValueError("Selected source mask differs from exported raw instance PNG")
                display = Image.fromarray((raw > 0).astype(np.uint8) * 255).convert("RGB")
            sources.append(stable_record(derived))
            save_lossless(frame_dir / f"{role}_raw.png", raw)
            sheet.paste(display, (640 * column, 56))
            draw = ImageDraw.Draw(sheet)
            draw.text((640 * column + 8, 5), f"frame {index} | source {index/30:.3f}s | image {row['id']}", fill="black", font=font)
            draw.text((640 * column + 8, 30), role + " | source label name, side pending", fill="black", font=font)
        sheet.save(frame_dir / "separated.png")
        contact.paste(sheet.resize((960, 268), Image.Resampling.NEAREST), (0, 268 * position))
        still_rows.append({"source_frame_index": index, "coco_image_id": row["id"], "sheet": f"{frame_dir.name}/separated.png",
                           "reference_pixels": {role: int((arrays[role][position] > 0).sum()) for role in paths if role != "rgb"}})
    contact.save(output / "all-stills.png")
    verify(sources)
    result = {"format": "nakehand-full-temporal-review-v1", "recording": RECORDING,
              "source_directory": str(directory), "full_frames": FRAME_COUNT, "source_fps": 30,
              "duration_seconds": FRAME_COUNT / 30, "image_id_offset": 4713,
              "videos": video_rows, "stills": still_rows, "source_still_pts": pts,
              "source_records": sources, "sources_unchanged": True, "selected_derived_pixels_match_source": True,
              "anatomical_side": "pending user review; image position is not an anatomical side label",
              "reference_note": "left/right denote original source mask file names; no correction or side decision was made",
              "display_note": "H264 CRF18/yuv420p previews are display-only; still raw PNG preserves exact decoded pixels",
              "training_or_model_inference": False, "source_or_training_data_modified": False}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    text = ["# nakehand 完整时序复核：有手但双参考为空，手别待确认", "",
            f"源录像 `{RECORDING}`，完整 {FRAME_COUNT} 帧，30 fps，约 {FRAME_COUNT/30:.2f} 秒。"
            "本例是录像第一帧，因此没有更早的帧；完整后续均提供。", "",
            "RGB 画面左下角不等于人体左手。用户认为可能为右手，当前没有确定手别，也未改标签。", ""]
    for role, name in (("rgb", "原始RGB"), ("left_reference", "源left文件参考"), ("right_reference", "源right文件参考")):
        text += [f"- [{name}完整视频]({role}_full.mp4)；[开头300连续帧三倍慢放]({role}_first300_slow.mp4)。"]
    text += ["", "[13个带帧号连续上下文截图](all-stills.png)，每行RGB/left参考/right参考分开。"
             "原尺寸raw PNG分帧保存，mask视频白色仅为源像素>0的显示，不是模型新预测。", "",
             f"[原始源目录]({directory})；[原始逐帧元数据](source_metadata.json)；[SHA和导出校验](manifest.json)。", "",
             "源像素与已导出val抽检13帧一致，所有源文件前后指纹未变。MP4是兼容播放器的显示重编码，不作为像素GT。"]
    (output / "REVIEW.md").write_text("\n".join(text) + "\n")
    # One explicitly bounded full-recording review, not the complete dataset.
    members = sorted(path for path in output.rglob("*") if path.is_file())
    if sum(path.stat().st_size for path in members) > 300 * 1024**2:
        raise ValueError("Review bundle exceeds 300 MiB cap; keep files without making ZIP")
    archive = output / "nakehand-full-recording-review.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in members:
            bundle.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Review archive checksum failed")
    receipt = {"output": str(output), "archive": stable_record(archive), "members": len(members),
               "windows_transfer": "not performed", "no_label_edits": True}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root, args.data_root, args.output_dir), ensure_ascii=False))

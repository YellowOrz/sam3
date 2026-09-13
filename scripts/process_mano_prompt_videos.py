#!/usr/bin/env python3
r"""说明：递归处理 color.mp4，保留输入目录结构；使用文本和指定侧 MANO 信息执行 SAM3
视频分割。点、框直接进入 detector geometry encoder，保留视频检测、跟踪与 memory。
MANO 文件位于视频同级 MANO_wilor/<left|right>_hand/result_mano_*.npz。
默认每帧使用提示；间隔 N 对应原视频第 0、N、2N…帧，与传播方向无关。
点使用有效、在画面内的全部关节，均为正点；NPZ 没有可见性，遮挡关节也可能被使用。
框由投影 mesh（默认）或 joints 包围框生成，每侧默认扩张原宽高的 5%，再裁到画面内。
缺失帧退回文本和已有 memory；整段缺少 NPZ 或匹配多个文件时记录失败并继续批次。
输出：masks.mkv（全部实例、FFV1）、result.mp4（黄点/青框）、metadata.json；
批次汇总写入 batch_summary.json。双向模式分别写 forward/、backward/，均正序播放。

命令行参数：
  --input-root：单视频目录或数据集根目录，默认 ~/Datasets。
  --output-root：输出根目录，默认 ~/sam3_outputs；不得与输入根目录相同。
  --prompt：必填文本提示，例如 "left hand"；--hand-side：必填 left 或 right。
  --prompt-mode：points / box / both，默认 both。
  --box-source：mesh / joints，默认 mesh；--box-padding：每侧扩张比例，默认 0.05。
  --prompt-interval：提示帧间隔，正整数，默认 1。
  --mano-name：指定侧目录中的 NPZ 文件名；默认自动选择唯一 result_mano_*.npz。
  --focal-length：原图像素单位焦距，默认 5000 / 256 * max(width, height)。
      投影为 p_cam = joints/vertices + camera_translation，uv = f*xy/z + [W/2,H/2]；
      左手坐标在 WiLoR 保存前已翻转，此处不再镜像。焦距默认匹配本地 WiLoR 配置。
  --version：仅 sam3（默认）；--device：CUDA 设备，默认 cuda:0。
  --checkpoint：SAM3 权重路径，默认 ~/.cache/modelscope/models/facebook--sam3/
      snapshots/master/sam3.pt。
  --direction：forward / backward / both，默认 forward。
  --max-sequences、--max-frames：可选的序列数和帧数上限。
  --list-only：核对视频、NPZ、投影及缺失统计，不加载模型或写输出。
  --overwrite：重新处理已有结果；--help：显示参数帮助。
  --compare-gt：启用 GT 评测；--gt-mask-name：启用评测时必填，不从手侧推导。
  --gt-dir-name：RGB 同级 GT 目录名，默认 masks_sam3。
  --compare-skip-frames：每次评测后跳过的帧数，默认 0；2 评测第 0、3、6…帧。
      生成 comparison.mp4、gt_metrics.csv/json 和根目录 gt_summary.json。
      完整预测可免 GPU 补评测，仍校验 MANO 和缓存配置；GT 失败保留预测并继续批次。

示例：python scripts/process_mano_prompt_videos.py --input-root DATA --output-root OUT \
  --prompt "left hand" --hand-side left --prompt-mode both --device cuda:0 --list-only

TODO：
  1. 支持 SAM3.1，并验证其 detector geometry 和预计算路径。
  2. 引入关节遮挡/可见性信息，避免把遮挡关节作为正点。
  3. 支持匹配多个 result_mano_*.npz 的实例选择、关联和处理；当前需指定唯一文件。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

if __package__:
    from scripts import process_dataset_videos as dataset
else:
    import process_dataset_videos as dataset


LOGGER = logging.getLogger("sam3_mano_processor")


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return number


def positive_float(value: str) -> float:
    number = nonnegative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def mano_filename(value: str) -> str:
    if Path(value).name != value or not value.endswith(".npz"):
        raise argparse.ArgumentTypeError("mano-name must be a filename ending in .npz")
    return value


def find_mano(video: Path, side: str, name: Optional[str]) -> Path:
    directory = video.parent / "MANO_wilor" / f"{side}_hand"
    matches = (
        [directory / name]
        if name is not None
        else sorted(directory.glob("result_mano_*.npz"))
    )
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"no MANO NPZ in {directory} (name={name!r})")
    if len(matches) != 1:
        raise ValueError(f"multiple MANO NPZ files in {directory}; specify --mano-name")
    return matches[0]


def project_points(
    points: np.ndarray, translation: np.ndarray, width: int, height: int, focal: float
) -> np.ndarray:
    """Project valid positive-depth MANO coordinates; retain offscreen points for boxes."""
    camera = np.asarray(points, dtype=np.float64) + translation
    valid = np.isfinite(camera).all(axis=-1) & (camera[:, 2] > 0)
    camera = camera[valid]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xy = focal * (camera[:, :2] / camera[:, 2:]) + [width / 2, height / 2]
    return xy[np.isfinite(xy).all(axis=-1)]


def load_geometry(
    path: Path, info: dict, args: argparse.Namespace
) -> Tuple[Dict[int, Dict[str, Any]], dict]:
    """Read the WiLoR schema and build normalized, frame-indexed detector prompts."""
    width, height = int(info["width"]), int(info["height"])
    frame_count = dataset.expected_frame_count(info, args.max_frames)
    focal = args.focal_length or 5000 / 256 * max(width, height)
    need_points = args.prompt_mode in ("points", "both")
    need_boxes = args.prompt_mode in ("box", "both")
    sources = set()
    if need_points or (need_boxes and args.box_source == "joints"):
        sources.add("joints")
    if need_boxes and args.box_source == "mesh":
        sources.add("vertices")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "width",
            "height",
            "hand",
            "frame_indices",
            "has_hand",
            "camera_translation",
        } | sources
        if required - set(data.files):
            raise ValueError(
                f"missing NPZ fields: {sorted(required - set(data.files))}"
            )
        if int(data["width"]) != width or int(data["height"]) != height:
            raise ValueError("MANO dimensions differ from color.mp4")
        if str(data["hand"].item()) != f"{args.hand_side}_hand":
            raise ValueError("MANO hand differs from --hand-side")
        if "fps" in data and not np.isclose(float(data["fps"]), info["fps"], atol=0.01):
            raise ValueError(
                "MANO FPS differs from color.mp4; frame alignment is unsafe"
            )
        indices = data["frame_indices"]
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("frame_indices must be a one-dimensional integer array")
        if (
            len(np.unique(indices)) != len(indices)
            or (indices < 0).any()
            or (indices >= int(info["frame_count"])).any()
        ):
            raise ValueError("frame_indices must be unique and inside color.mp4")
        count = len(indices)
        present = data["has_hand"]
        translations = np.asarray(data["camera_translation"], dtype=np.float64)
        if present.shape != (count,) or present.dtype.kind != "b":
            raise ValueError(
                "has_hand must be a boolean array aligned with frame_indices"
            )
        if translations.shape != (count, 3):
            raise ValueError("camera_translation must have shape (T, 3)")
        coordinates = {}
        for source in sources:
            array = np.asarray(data[source], dtype=np.float64)
            expected = (count, 21 if source == "joints" else 778, 3)
            if array.shape != expected:
                raise ValueError(f"{source} must have shape {expected}")
            coordinates[source] = array

    prompts = {}
    available = set()
    for row, raw_index in enumerate(indices):
        frame_index = int(raw_index)
        if frame_index >= frame_count or not present[row]:
            continue
        if not np.isfinite(translations[row]).all():
            continue
        available.add(frame_index)
        if frame_index % args.prompt_interval:
            continue
        projected = {
            source: project_points(array[row], translations[row], width, height, focal)
            for source, array in coordinates.items()
        }
        geometry = {}
        if need_points:
            xy = projected["joints"]
            xy = xy[((xy >= 0) & (xy < [width, height])).all(axis=-1)]
            if len(xy):
                geometry["points"] = (xy / [width, height]).tolist()
        if need_boxes:
            xy = projected["vertices" if args.box_source == "mesh" else "joints"]
            if len(xy):
                lower, upper = xy.min(axis=0), xy.max(axis=0)
                padding = (upper - lower) * args.box_padding
                lower = np.clip(lower - padding, 0, [width, height])
                upper = np.clip(upper + padding, 0, [width, height])
                if (upper > lower).all():
                    geometry["boxes"] = [
                        np.concatenate(
                            (lower / [width, height], (upper - lower) / [width, height])
                        ).tolist()
                    ]
        if geometry:
            prompts[frame_index] = geometry

    stat = path.stat()
    metadata = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "hand_side": args.hand_side,
        "mode": args.prompt_mode,
        "box_source": args.box_source,
        "box_padding": args.box_padding,
        "prompt_interval": args.prompt_interval,
        "focal_length": focal,
        "missing_frames": sorted(set(range(frame_count)) - available),
        "geometry_frames": sorted(prompts),
        "points_frames": sum("points" in value for value in prompts.values()),
        "box_frames": sum("boxes" in value for value in prompts.values()),
        "unusable_prompt_frames": sorted(
            t for t in available if t % args.prompt_interval == 0 and t not in prompts
        ),
    }
    return prompts, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        parents=[dataset.build_parser()],
        add_help=False,
        conflict_handler="resolve",
        description="Batch SAM3 video segmentation with text and MANO detector geometry.",
    )
    parser.add_argument("--version", choices=["sam3"], default="sam3")
    parser.add_argument("--hand-side", choices=["left", "right"], required=True)
    parser.add_argument(
        "--prompt-mode", choices=["points", "box", "both"], default="both"
    )
    parser.add_argument("--box-source", choices=["mesh", "joints"], default="mesh")
    parser.add_argument("--box-padding", type=nonnegative_float, default=0.05)
    parser.add_argument("--prompt-interval", type=dataset.positive_int, default=1)
    parser.add_argument("--mano-name", type=mano_filename)
    parser.add_argument("--focal-length", type=positive_float)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset.validate_gt_arguments(parser, args)
    if not args.prompt.strip() or args.prompt == "visual":
        parser.error("--prompt must be nonempty text, not the reserved value 'visual'")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    input_root = dataset.expand_path(args.input_root)
    output_root = dataset.expand_path(args.output_root)
    if not input_root.is_dir() or input_root.resolve() == output_root.resolve():
        LOGGER.error("Input root must exist and differ from output root")
        return 2
    videos = dataset.discover_color_videos(input_root)
    if args.max_sequences is not None:
        videos = videos[: args.max_sequences]
    if not videos:
        LOGGER.error("No color.mp4 found below %s", input_root)
        return 2
    checkpoint = dataset.expand_path(args.checkpoint) if args.checkpoint else None
    if not args.list_only and checkpoint is not None and not checkpoint.is_file():
        LOGGER.error("Checkpoint does not exist: %s", checkpoint)
        return 2

    comparison = dataset.GTComparison(
        args, dataset.processing_directions(args.direction)
    )
    results = []
    with dataset.lazy_predictor(args) as get_predictor:
        try:
            for video in videos:
                output = dataset.output_dir_for(video, input_root, output_root)
                try:
                    mano = find_mano(video, args.hand_side, args.mano_name)
                    info = dataset.probe_video(video)
                    prompts, metadata = load_geometry(mano, info, args)
                except Exception as exc:
                    LOGGER.exception("MANO validation failed: %s", video)
                    results.append(
                        {"video": str(video), "status": "failed", "error": str(exc)}
                    )
                    if not args.list_only:
                        for direction in dataset.processing_directions(args.direction):
                            destination = dataset.directional_output_dir(
                                output, args.direction, direction
                            )
                            comparison.record_failure(
                                video, destination, direction, exc
                            )
                    continue
                LOGGER.info(
                    "%s: %s, geometry=%d frames, missing=%d, unusable=%d",
                    video,
                    mano.name,
                    len(prompts),
                    len(metadata["missing_frames"]),
                    len(metadata["unusable_prompt_frames"]),
                )
                if args.list_only:
                    print(f"{video.relative_to(input_root)} -> {mano}")
                    continue
                video_stat = video.stat()
                checkpoint_stat = checkpoint.stat() if checkpoint is not None else None
                extra = {
                    "mano": metadata,
                    "processor": "mano_geometry_v1",
                    "input_size": video_stat.st_size,
                    "input_mtime_ns": video_stat.st_mtime_ns,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                    "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns
                    if checkpoint_stat
                    else None,
                }
                for direction in dataset.processing_directions(args.direction):
                    destination = dataset.directional_output_dir(
                        output, args.direction, direction
                    )
                    record = {"video": str(video), "direction": direction}
                    try:
                        record["status"] = dataset.process_video(
                            None,
                            video,
                            destination,
                            input_root,
                            args.prompt,
                            "sam3",
                            args.max_frames,
                            args.overwrite,
                            direction,
                            extra_metadata=extra,
                            geometry_prompts=prompts,
                            predictor_factory=get_predictor,
                        )
                        comparison.compare(video, destination, direction)
                    except Exception as exc:
                        comparison.record_failure(video, destination, direction, exc)
                        LOGGER.exception("Sequence failed: %s (%s)", video, direction)
                        record.update(status="failed", error=str(exc))
                    results.append(record)
        finally:
            if not args.list_only:
                output_root.mkdir(parents=True, exist_ok=True)
                dataset.write_json(
                    output_root / "batch_summary.json", {"results": results}
                )
                comparison.write_summary(output_root)
    return 1 if any(item["status"] == "failed" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())

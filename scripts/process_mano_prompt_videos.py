#!/usr/bin/env python3
r"""说明：递归处理 RGB 视频（默认 color.mp4），保留输入目录结构；使用文本和指定侧 MANO 信息执行 SAM3
视频分割。点、框直接进入 detector geometry encoder，保留视频检测、跟踪与 memory。
MANO 文件位于视频同级 <mano-dir-name>/<left|right>_hand/*.npz，默认使用该目录下全部 NPZ。
默认每帧使用提示；间隔 N 对应原视频第 0、N、2N…帧，与传播方向无关。
点使用有效、在画面内的全部关节，均为正点；NPZ 没有可见性，遮挡关节也可能被使用。
框由投影 mesh（默认）或 joints 包围框生成，每侧默认扩张原宽高的 5%，再裁到画面内。
同一帧多个 NPZ 的框和点全部送入同一检测查询。缺失帧退回文本和已有 memory（禁用文本时仅跟踪已有对象）；
缺少指定手的 NPZ 时跳过视频，任一文件无效时记录失败并继续批次。
输出：masks.mkv（全部实例、FFV1）、result.mp4（预测叠加、MANO 骨架、实际 prompts）、metadata.json；
批次汇总写入 batch_summary.json。双向模式分别写 forward/、backward/，均正序播放。
result.mp4 与 comparison 第一列在任一 NPZ 有手的帧上画全部在屏关节和骨架，不受 prompt-interval 限制。

命令行参数：
  --input-root：单视频目录或数据集根目录，默认 ~/Datasets。
  --rgb-name：递归查找的 RGB 文件名，默认 color.mp4，包含根目录和所有子目录。
  --output-root：输出根目录，默认 ~/sam3_outputs；不得与输入根目录相同。
  --text-prompt：必填文本提示，例如 "left hand"；传 "" 禁用文本；--hand-side：必填 left 或 right。
  --prompt-mode：points / box / both，默认 both。
  --box-source：mesh / joints，默认 mesh；--box-padding：每侧扩张比例，默认 0.05。
  --prompt-interval：提示帧间隔，正整数，默认 1。
  --mano-dir-name：视频同级的 MANO 目录名，默认 MANO_wilor。
  --focal-length：原图像素单位焦距，默认 5000 / 256 * max(width, height)。
      投影为 p_cam = joints/vertices + camera_translation，uv = f*xy/z + [W/2,H/2]；
      左手坐标在 WiLoR 保存前已翻转，此处不再镜像。焦距默认匹配本地 WiLoR 配置。
  --version：仅 sam3（默认）；--device：CUDA 设备，默认 cuda:0。
  --checkpoint：SAM3 权重路径，默认 ~/.cache/modelscope/models/facebook--sam3/
      snapshots/master/sam3.pt。
  --direction：forward / backward / both，默认 forward。
  --max-sequences、--max-frames：可选的序列数和帧数上限。
  --list-only：仅列出发现的输入视频，不加载模型、不执行分割或 GT 评测、不写结果文件。
      本脚本额外校验对应 MANO 文件、投影及缺失统计，并列出视频与 MANO 文件的对应关系。
  --overwrite：重新处理已有结果；--help：显示参数帮助。
  --compare-gt：启用 GT 评测；--gt-mask-name：启用评测时必填，不从手侧推导。
  --gt-dir-name：RGB 同级 GT 目录名，默认 masks_sam3。
  --compare-skip-frames：每次评测后跳过的帧数，默认 0；2 评测第 0、3、6…帧。
      生成 comparison.mp4、gt_metrics.csv/json 和根目录 gt_summary.json
      comparison.mp4 最左侧 RGB 列叠加该帧实际使用的黄色点／青色框，以及 MANO 骨架。
      完整预测可免 GPU 补评测，仍校验 MANO 和缓存配置；GT 失败保留预测并继续批次。
  --save-detector：另存 detector_masks.mkv（阈值后的原始 det_out 并集，FFV1 gray8）
      和 detector_result.mp4（detector 着色叠加与实际 prompts，不含骨架）。
      与 --compare-gt 同时开启时，comparison.mp4 为 RGB／Detector／Prediction／GT，
      并在 gt_metrics 中增加 detector_* 指标。无 GT 时不生成 comparison.mp4。
      缺少完整 detector 视频的已有预测会重新推理。

示例：python scripts/process_mano_prompt_videos.py --input-root DATA --output-root OUT \
  --text-prompt "left hand" --hand-side left --prompt-mode both --device cuda:0 --list-only

TODO：
  1. 支持 SAM3.1，并验证其 detector geometry 和预计算路径。
  2. 引入关节遮挡/可见性信息，避免把遮挡关节作为正点。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__:
    from scripts import process_dataset_videos as dataset
    from scripts.common.compare_gt_masks import file_name
    from scripts.common.video_cli import (
        add_video_arguments,
        discover_rgb_videos,
        validate_input_output,
    )
else:
    import process_dataset_videos as dataset
    from common.compare_gt_masks import file_name
    from common.video_cli import (
        add_video_arguments,
        discover_rgb_videos,
        validate_input_output,
    )


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


def find_mano(video: Path, side: str, dir_name: str = "MANO_wilor") -> List[Path]:
    directory = video.parent / dir_name / f"{side}_hand"
    matches = sorted(path for path in directory.glob("*.npz") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"no MANO NPZ in {directory}")
    return matches


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


def project_hand(
    joints: np.ndarray, translation: np.ndarray, width: int, height: int, focal: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return all valid projections and the in-frame subset, both (21, 2) with NaNs."""
    camera = np.asarray(joints, dtype=np.float64) + translation
    xy = np.full((len(camera), 2), np.nan)
    valid = np.isfinite(camera).all(axis=-1) & (camera[:, 2] > 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = focal * (camera[valid, :2] / camera[valid, 2:]) + [
            width / 2,
            height / 2,
        ]
    xy[valid] = projected
    xy[~np.isfinite(xy).all(axis=-1)] = np.nan
    onscreen = xy.copy()
    in_frame = (onscreen >= 0).all(axis=-1) & (onscreen < [width, height]).all(axis=-1)
    onscreen[~in_frame] = np.nan
    return xy, onscreen


def load_geometry(
    path: Path, info: dict, args: argparse.Namespace
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Any]], dict]:
    """Read one WiLoR NPZ into detector prompts and per-frame preview hands."""
    width, height = int(info["width"]), int(info["height"])
    frame_count = dataset.expected_frame_count(info, args.max_frames)
    focal = args.focal_length or 5000 / 256 * max(width, height)
    need_points = args.prompt_mode in ("points", "both")
    need_boxes = args.prompt_mode in ("box", "both")
    sources = {"joints"}
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
            raise ValueError("MANO dimensions differ from RGB video")
        if str(data["hand"].item()) != f"{args.hand_side}_hand":
            raise ValueError("MANO hand differs from --hand-side")
        if "fps" in data and not np.isclose(float(data["fps"]), info["fps"], atol=0.01):
            raise ValueError(
                "MANO FPS differs from RGB video; frame alignment is unsafe"
            )
        indices = data["frame_indices"]
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("frame_indices must be a one-dimensional integer array")
        if (
            len(np.unique(indices)) != len(indices)
            or (indices < 0).any()
            or (indices >= int(info["frame_count"])).any()
        ):
            raise ValueError("frame_indices must be unique and inside RGB video")
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

    prompts: Dict[int, Dict[str, Any]] = {}
    overlay: Dict[int, List[Any]] = {}
    available = set()
    for row, raw_index in enumerate(indices):
        frame_index = int(raw_index)
        if frame_index >= frame_count or not present[row]:
            continue
        if not np.isfinite(translations[row]).all():
            continue
        available.add(frame_index)
        all_xy, onscreen = project_hand(
            coordinates["joints"][row], translations[row], width, height, focal
        )
        if np.isfinite(onscreen).any():
            overlay.setdefault(frame_index, []).append(
                (onscreen / [width, height]).tolist()
            )
        if frame_index % args.prompt_interval:
            continue
        geometry: Dict[str, Any] = {}
        if need_points:
            xy = onscreen[np.isfinite(onscreen).all(axis=-1)]
            if len(xy):
                geometry["points"] = (xy / [width, height]).tolist()
        if need_boxes:
            if args.box_source == "mesh":
                xy = project_points(
                    coordinates["vertices"][row],
                    translations[row],
                    width,
                    height,
                    focal,
                )
            else:
                xy = all_xy[np.isfinite(all_xy).all(axis=-1)]
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
        "overlay_frames": sorted(overlay),
        "points_frames": sum("points" in value for value in prompts.values()),
        "box_frames": sum("boxes" in value for value in prompts.values()),
        "unusable_prompt_frames": sorted(
            t for t in available if t % args.prompt_interval == 0 and t not in prompts
        ),
    }
    return prompts, overlay, metadata


def load_geometries(
    paths: Sequence[Path], info: dict, args: argparse.Namespace
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Any]], dict]:
    prompts: Dict[int, Dict[str, Any]] = {}
    overlay: Dict[int, List[Any]] = {}
    files = []
    available = set()
    missing = None
    metadata = None
    frame_count = (
        dataset.expected_frame_count(info, args.max_frames)
        if "frame_count" in info
        else None
    )
    for path in paths:
        file_prompts, file_overlay, metadata = load_geometry(path, info, args)
        files.append(
            {
                "path": metadata.get("path", str(path)),
                "size": metadata.get("size"),
                "mtime_ns": metadata.get("mtime_ns"),
            }
        )
        file_missing = set(metadata.get("missing_frames", []))
        missing = file_missing if missing is None else missing & file_missing
        if frame_count is not None:
            available.update(set(range(frame_count)) - file_missing)
        else:
            available.update(set(file_prompts) | set(file_overlay))
        for frame_index, geometry in file_prompts.items():
            destination = prompts.setdefault(frame_index, {})
            for key, values in geometry.items():
                destination.setdefault(key, []).extend(values)
        for frame_index, hands in file_overlay.items():
            overlay.setdefault(frame_index, []).extend(hands)
    if metadata is None:
        raise FileNotFoundError("no MANO NPZ")
    metadata = {
        **metadata,
        "files": files,
        "missing_frames": (
            sorted(set(range(frame_count)) - available)
            if frame_count is not None
            else sorted(missing or [])
        ),
        "geometry_frames": sorted(prompts),
        "overlay_frames": sorted(overlay),
        "points_frames": sum("points" in value for value in prompts.values()),
        "box_frames": sum("boxes" in value for value in prompts.values()),
        "unusable_prompt_frames": sorted(
            t for t in available if t % args.prompt_interval == 0 and t not in prompts
        ),
    }
    metadata.pop("path", None)
    metadata.pop("size", None)
    metadata.pop("mtime_ns", None)
    return prompts, overlay, metadata


def visualization_prompts(
    prompts: Dict[int, Dict[str, Any]], overlay: Dict[int, List[Any]]
) -> Dict[int, Dict[str, Any]]:
    result = {index: {"hands": hands} for index, hands in overlay.items()}
    for index, geometry in prompts.items():
        result.setdefault(index, {}).update(geometry)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch SAM3 video segmentation with text and MANO detector geometry. "
            "With --list-only, also validate MANO files, projections and missing "
            "frames, and list the video-to-MANO file mappings."
        ),
    )
    add_video_arguments(
        parser,
        list_only_extra=(
            "Also validate MANO files, projections and missing frames, "
            "and list video-to-MANO file mappings."
        ),
    )
    parser.add_argument(
        "--text-prompt",
        dest="prompt",
        required=True,
        help='Shared text prompt; use "" to disable text',
    )
    parser.add_argument("--version", choices=["sam3"], default="sam3")
    parser.add_argument("--hand-side", choices=["left", "right"], required=True)
    parser.add_argument(
        "--prompt-mode", choices=["points", "box", "both"], default="both"
    )
    parser.add_argument("--box-source", choices=["mesh", "joints"], default="mesh")
    parser.add_argument("--box-padding", type=nonnegative_float, default=0.05)
    parser.add_argument("--prompt-interval", type=dataset.positive_int, default=1)
    parser.add_argument(
        "--mano-dir-name",
        type=file_name,
        default="MANO_wilor",
        help="MANO directory beside RGB (default: MANO_wilor); missing hand NPZ skips video",
    )
    parser.add_argument("--focal-length", type=positive_float)
    parser.add_argument(
        "--save-detector",
        action="store_true",
        help="Save detector masks and detector_result.mp4",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset.validate_gt_arguments(parser, args)
    args.prompt = args.prompt.strip()
    if args.prompt == "visual":
        parser.error(
            "--text-prompt cannot be 'visual'; use an empty string for no text"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    input_root = dataset.expand_path(args.input_root)
    output_root = dataset.expand_path(args.output_root)
    try:
        validate_input_output(input_root, output_root)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2
    videos = discover_rgb_videos(input_root, args.rgb_name)
    if args.max_sequences is not None:
        videos = videos[: args.max_sequences]
    if not videos:
        LOGGER.error("No files named %s found below %s", args.rgb_name, input_root)
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
                    try:
                        manos = find_mano(video, args.hand_side, args.mano_dir_name)
                    except FileNotFoundError as exc:
                        LOGGER.info("Skipping %s: %s", video, exc)
                        results.append(
                            {
                                "video": str(video),
                                "status": "skipped",
                                "reason": str(exc),
                            }
                        )
                        continue
                    info = dataset.probe_video(video)
                    prompts, overlay, metadata = load_geometries(manos, info, args)
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
                visualization = visualization_prompts(prompts, overlay)
                LOGGER.info(
                    "%s: %s, geometry=%d frames, missing=%d, unusable=%d",
                    video,
                    ",".join(path.name for path in manos),
                    len(prompts),
                    len(metadata["missing_frames"]),
                    len(metadata["unusable_prompt_frames"]),
                )
                if args.list_only:
                    print(
                        f"{video.relative_to(input_root)} -> "
                        + ", ".join(str(path) for path in manos)
                    )
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
                            overlay_geometry=visualization,
                            predictor_factory=get_predictor,
                            save_detector=args.save_detector,
                        )
                        comparison.compare(
                            video, destination, direction, visualization
                        )
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

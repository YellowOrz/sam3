#!/usr/bin/env python3
"""使用 SAM 3 可学习目标特征批量分割视频，并可与 GT 对比。

递归查找指定名称的 RGB 视频，输出保留输入目录结构。加载目标特征文件，
通过 add_learned_prompt 推理，不加载文本编码器；不支持 SAM 3.1。

命令行参数（路径支持 ~）：
    --input-root PATH
        输入根目录，可以是单个序列目录；默认 ~/Datasets。
    --output-root PATH
        输出根目录；默认 ~/sam3_learned_outputs。
    --rgb-name NAME
        递归查找的 RGB 文件名；默认 color.mp4，也可指定 rgb.mkv。
    --learned-prompt PATH
        训练后或初始化的目标特征文件（.pt）；除 --list-only 外必须提供。
    --checkpoint PATH
        基础 SAM 3 权重；推理加载时校验其 SHA-256 与特征文件一致。
        默认 ~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt。
    --target-id ID
        目标标识；默认从特征文件读取，显式指定时必须与文件中的 ID 一致。
    --direction {forward,backward,both}
        传播方向；默认 forward。both 独立执行正反向传播，分别输出到
        forward/ 和 backward/；结果视频与对比视频均按原始时间顺序播放。
    --device cuda:N
        推理使用的 CUDA 设备；默认 cuda:0，cuda 等同于 cuda:0。
    --max-sequences N
        最多处理排序后的前 N 个视频，N 必须为正整数；默认不限制。
    --max-frames N
        每个视频仅推理、评测前 N 帧，N 必须为正整数；默认处理全部帧。
    --overwrite
        重新推理并覆盖已有预测；默认复用目标、方向及帧数匹配的完整预测。
        更换同一目标的特征或权重后，应使用此参数或更换输出目录。
    --list-only
        仅列出发现的视频相对路径，不加载模型，不要求特征文件或权重存在。
    --compare-gt
        启用 GT 对比；默认关闭。完整预测可直接补做对比，无需加载模型或
        使用 CUDA，但仍需提供有效的特征文件与 checkpoint 路径。
    --gt-dir-name NAME
        每个 RGB 视频同级的 GT 文件夹名；默认 masks_sam3。
    --gt-mask-name NAME
        GT 标签视频文件名；启用 --compare-gt 时必填，例如 left_hand.mkv。
    --compare-skip-frames N
        每比较一帧后跳过 N 帧，N 为非负整数；默认 0，即逐帧比较。
        例如 2 取原视频第 0、3、6…帧，同时作用于对比视频和指标计算，
        不影响模型逐帧推理。对比视频帧率为原帧率 / (N + 1)。
    -h, --help
        显示命令行帮助并退出。

RGB 文件名、GT 文件夹名和 GT 文件名仅接受单个名称，不接受路径或通配符。
GT 为逐帧对齐的 uint8 灰度标签视频，0 是背景，所有非零实例取前景并集。
GT 与完整 RGB 的尺寸、帧率及声明帧数必须一致，不自动缩放或截断对齐。
GT 参数仅在 --compare-gt 启用时参与对比；对比每次重新生成。

输出与指标：
    原有 masks.mkv、result.mp4、metadata.json 保存预测结果。
    启用对比后，每个预测目录新增 comparison.mp4（原图／预测叠加／GT 叠加）、
    gt_metrics.csv（逐采样帧指标及原始帧号）、gt_metrics.json（视频汇总）。
    输出根目录的 gt_summary.json 按传播方向分别汇总本次批次的成功序列。
    IoU、Dice 同时报告逐帧均值及累计像素指标；双方都空计 1，仅一方空计 0。
    对比失败时保留预测、清除过期对比视频及 CSV，记录原因并继续其他序列；
    最终返回非零退出码。没有成功评测帧时，汇总指标为 null。

示例：先核对输入目录中的 rgb.mkv：
    python scripts/process_learned_prompt_videos.py \
        --input-root ~/Datasets/wanqing_datasets \
        --rgb-name rgb.mkv --list-only

示例：分割左手，并每隔两帧与 GT 比较一次：
    python scripts/process_learned_prompt_videos.py \
        --input-root ~/Datasets/realsense_hand_object_with_seg_to_wjh \
        --output-root ~/sam3_learned_outputs \
        --rgb-name color.mp4 \
        --learned-prompt outputs/learned_left_hand/best.pt \
        --checkpoint /models/sam3.pt \
        --device cuda:0 \
        --compare-gt --gt-dir-name masks_sam3 \
        --gt-mask-name left_hand.mkv --compare-skip-frames 2
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__:
    from scripts.common.compare_gt_masks import (
        add_gt_arguments,
        file_name,
        GTComparison,
        validate_gt_arguments,
    )
    from scripts.common.video_utils import expand_path, parse_device, positive_int
    from scripts.process_dataset_videos import (
        directional_output_dir,
        expected_frame_count,
        is_complete,
        output_dir_for,
        probe_video,
        process_video,
        processing_directions,
        require_cuda,
    )
else:
    from common.compare_gt_masks import (
        add_gt_arguments,
        file_name,
        GTComparison,
        validate_gt_arguments,
    )
    from common.video_utils import expand_path, parse_device, positive_int
    from process_dataset_videos import (  # type: ignore[no-redef]
        directional_output_dir,
        expected_frame_count,
        is_complete,
        output_dir_for,
        probe_video,
        process_video,
        processing_directions,
        require_cuda,
    )


LOGGER = logging.getLogger("sam3_learned_prompt_processor")


def read_target_id(learned_prompt: Path) -> str:
    import torch

    payload = torch.load(learned_prompt, map_location="cpu", weights_only=True)
    target_id = payload.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError(f"learned prompt file has no target_id: {learned_prompt}")
    return target_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively process RGB videos with a SAM 3 learned "
            "target feature file (no text encoder)."
        )
    )
    parser.add_argument("--input-root", default="~/Datasets")
    parser.add_argument("--rgb-name", type=file_name, default="color.mp4")
    add_gt_arguments(parser)
    parser.add_argument("--output-root", default="~/sam3_learned_outputs")
    parser.add_argument(
        "--checkpoint",
        default="~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt",
        help="Base SAM 3 checkpoint; must match the feature file SHA-256",
    )
    parser.add_argument(
        "--learned-prompt",
        help="Trained or initial target feature file (.pt)",
    )
    parser.add_argument(
        "--target-id",
        help="Must match the feature file; defaults to the ID stored in the file",
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "backward", "both"],
        default="forward",
        help=(
            "Propagation/playback direction. 'both' writes independent results "
            "below forward/ and backward/ (default: forward)"
        ),
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=("cuda:0", 0),
        metavar="cuda:N",
        help="CUDA device to use (default: cuda:0)",
    )
    parser.add_argument("--max-sequences", type=positive_int)
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument(
        "--overwrite", action="store_true", help="Reprocess completed sequences"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List discovered RGB videos without loading the model",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_gt_arguments(parser, args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    input_root = expand_path(args.input_root)
    output_root = expand_path(args.output_root)
    checkpoint = expand_path(args.checkpoint)
    device_name, device_index = args.device

    if not input_root.is_dir():
        LOGGER.error("Input root does not exist or is not a directory: %s", input_root)
        return 2
    videos = sorted(path for path in input_root.rglob(args.rgb_name) if path.is_file())
    if args.max_sequences is not None:
        videos = videos[: args.max_sequences]
    if not videos:
        LOGGER.error("No files named %s found below %s", args.rgb_name, input_root)
        return 2

    LOGGER.info("Discovered %d RGB video(s)", len(videos))
    if args.list_only:
        for video_path in videos:
            print(video_path.relative_to(input_root))
        return 0
    if not args.learned_prompt:
        LOGGER.error("--learned-prompt is required unless --list-only")
        return 2

    learned_prompt = expand_path(args.learned_prompt)
    if not learned_prompt.is_file():
        LOGGER.error("Learned prompt does not exist: %s", learned_prompt)
        return 2
    if not checkpoint.is_file():
        LOGGER.error("Checkpoint does not exist: %s", checkpoint)
        return 2

    try:
        stored_target_id = read_target_id(learned_prompt)
    except Exception as exc:
        LOGGER.error("Cannot read learned prompt: %s", exc)
        return 2
    target_id = args.target_id or stored_target_id
    if args.target_id and args.target_id != stored_target_id:
        LOGGER.error(
            "Requested target %r does not match feature file %r",
            args.target_id,
            stored_target_id,
        )
        return 2
    predictor = None
    counts = {"success": 0, "skipped": 0, "failed": 0}
    comparison = GTComparison(args, processing_directions(args.direction))
    try:
        for video_path in videos:
            for direction in processing_directions(args.direction):
                output_dir = directional_output_dir(
                    output_dir_for(video_path, input_root, output_root),
                    args.direction,
                    direction,
                )
                try:
                    complete = not args.overwrite and is_complete(
                        output_dir,
                        video_path,
                        target_id,
                        "sam3",
                        expected_frame_count(probe_video(video_path), args.max_frames),
                        direction,
                    )
                    if complete:
                        status = "skipped"
                    else:
                        if predictor is None:
                            if not require_cuda(device_name, device_index):
                                return 2
                            from sam3.model_builder import build_sam3_video_predictor

                            LOGGER.info("Loading SAM 3 on %s", device_name)
                            predictor = build_sam3_video_predictor(
                                checkpoint_path=str(checkpoint),
                                learned_prompt_path=str(learned_prompt),
                                gpus_to_use=[device_index],
                                compile=False,
                                async_loading_frames=True,
                            )
                        status = process_video(
                            predictor=predictor,
                            video_path=video_path,
                            output_dir=output_dir,
                            input_root=input_root,
                            prompt=target_id,
                            model_version="sam3",
                            max_frames=args.max_frames,
                            overwrite=args.overwrite,
                            direction=direction,
                            prompt_request_type="learned",
                            extra_metadata={
                                "prompt_mode": "learned",
                                "target_id": target_id,
                                "learned_prompt_path": str(learned_prompt),
                                "checkpoint": str(checkpoint),
                            },
                        )
                    comparison.compare(video_path, output_dir, direction)
                    counts[status] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    LOGGER.exception("Sequence failed (%s): %s", direction, video_path)
                    comparison.record_failure(video_path, output_dir, direction, exc)
    finally:
        if predictor is not None:
            predictor.shutdown()

    comparison.write_summary(output_root)

    LOGGER.info(
        "Finished: %d succeeded, %d skipped, %d failed",
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run SAM 3 learned-prompt video segmentation over color.mp4 sequences.

This is the dataset-tree counterpart of text-prompted
``process_dataset_videos.py``. It loads one target feature file, never a
text encoder, and calls ``add_learned_prompt`` instead of ``add_prompt``.
SAM 3.1 is not supported.

Example:
    python scripts/process_learned_prompt_videos.py \
        --input-root ~/Datasets \
        --output-root ~/sam3_learned_outputs \
        --learned-prompt outputs/learned_left_hand/best.pt \
        --checkpoint /models/sam3.pt \
        --device cuda:0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

if __package__:
    from scripts.process_dataset_videos import (
        discover_color_videos,
        require_cuda,
        run_sequences,
    )
    from scripts.video_utils import expand_path, parse_device, positive_int
else:
    from process_dataset_videos import (  # type: ignore[no-redef]
        discover_color_videos,
        require_cuda,
        run_sequences,
    )
    from video_utils import expand_path, parse_device, positive_int  # type: ignore[no-redef]


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
            "Recursively process only color.mp4 files with a SAM 3 learned "
            "target feature file (no text encoder)."
        )
    )
    parser.add_argument("--input-root", default="~/Datasets")
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
        help="List discovered color videos without loading the model",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
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
    videos = discover_color_videos(input_root)
    if args.max_sequences is not None:
        videos = videos[: args.max_sequences]
    if not videos:
        LOGGER.error("No files named color.mp4 found below %s", input_root)
        return 2

    LOGGER.info("Discovered %d color video(s)", len(videos))
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
    if not require_cuda(device_name, device_index):
        return 2

    LOGGER.info(
        "Loading SAM 3 on %s from %s with target %s",
        device_name,
        checkpoint,
        target_id,
    )
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint),
        learned_prompt_path=str(learned_prompt),
        gpus_to_use=[device_index],
        compile=False,
        async_loading_frames=True,
    )
    try:
        counts = run_sequences(
            predictor=predictor,
            videos=videos,
            input_root=input_root,
            output_root=output_root,
            prompt=target_id,
            model_version="sam3",
            max_frames=args.max_frames,
            overwrite=args.overwrite,
            requested_direction=args.direction,
            prompt_request_type="learned",
            extra_metadata={
                "prompt_mode": "learned",
                "target_id": target_id,
                "learned_prompt_path": str(learned_prompt),
                "checkpoint": str(checkpoint),
            },
        )
    finally:
        predictor.shutdown()

    LOGGER.info(
        "Finished: %d succeeded, %d skipped, %d failed",
        counts["success"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

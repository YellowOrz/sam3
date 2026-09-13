"""Shared arguments and input discovery for batch video processors."""

import argparse
from pathlib import Path
from typing import List

from .compare_gt_masks import add_gt_arguments, file_name
from .video_utils import parse_device, positive_int


def add_video_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_root: str = "~/sam3_outputs",
    checkpoint_help: str = "Checkpoint path (auto-downloads from HuggingFace if omitted)",
    list_only_extra: str = "",
) -> None:
    io = parser.add_argument_group("Data input/output")
    io.add_argument("--input-root", default="~/Datasets")
    io.add_argument(
        "--rgb-name",
        type=file_name,
        default="color.mp4",
        help="RGB filename to find recursively, including the input root",
    )
    io.add_argument(
        "--output-root",
        default=output_root,
        help="Output root; must differ from input root",
    )
    io.add_argument(
        "--overwrite", action="store_true", help="Reprocess completed sequences"
    )

    run = parser.add_argument_group("Batch execution")
    run.add_argument(
        "--direction",
        choices=["forward", "backward", "both"],
        default="forward",
        help="Propagation/playback direction; both writes forward/ and backward/",
    )
    run.add_argument("--max-sequences", type=positive_int)
    run.add_argument("--max-frames", type=positive_int)
    run.add_argument(
        "--list-only",
        action="store_true",
        help=(
            "List discovered input videos; do not load the model, run segmentation "
            "or GT evaluation, or write output files. " + list_only_extra
        ).strip(),
    )

    model = parser.add_argument_group("Model loading")
    model.add_argument(
        "--checkpoint",
        default="~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt",
        help=checkpoint_help,
    )
    model.add_argument(
        "--device",
        type=parse_device,
        default=("cuda:0", 0),
        metavar="cuda:N",
        help="CUDA device to use (default: cuda:0)",
    )
    add_gt_arguments(parser.add_argument_group("GT evaluation"))


def validate_input_output(input_root: Path, output_root: Path) -> None:
    if not input_root.is_dir():
        raise ValueError(
            f"Input root does not exist or is not a directory: {input_root}"
        )
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Input root must differ from output root")


def discover_rgb_videos(input_root: Path, rgb_name: str) -> List[Path]:
    return sorted(
        (path for path in input_root.rglob(rgb_name) if path.is_file()),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )

#!/usr/bin/env python3
"""Build side-by-side comparison videos from SAM 3 dataset outputs.

One comparison video is produced for each ``result.mp4`` relative path shared by
all input roots, with cells ordered exactly as the roots appear on the command
line. Paths missing from any root are skipped.

Example:
    uv run python scripts/compare_dataset_videos.py \
        outputs/test_sam3/the_visible_hands_of_a_person \
        outputs/test_sam3/all_visible_human_hands \
        outputs/test_sam3/visible_human_hands \
        --output-dir outputs/comparisons
"""

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__:
    from scripts.video_utils import expand_path
else:
    from video_utils import expand_path  # type: ignore[no-redef]


LOGGER = logging.getLogger("sam3_video_comparison")
FPS_TOLERANCE = 1e-3
Grid = Tuple[int, int]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class SequenceInputs:
    name: str
    relative_dir: Path
    videos: Tuple[Path, ...]
    info: VideoInfo


def parse_grid(value: str) -> Grid:
    match = re.fullmatch(r"([1-9]\d*)[xX]([1-9]\d*)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("grid must look like ROWSxCOLS, e.g. 2x3")
    return int(match.group(1)), int(match.group(2))


def recommend_grid(cell_count: int) -> Grid:
    """Return the closest-to-square grid, preferring landscape on ties."""
    if cell_count <= 0:
        raise ValueError("cell_count must be positive")
    candidates = []
    for rows in range(1, cell_count + 1):
        columns = math.ceil(cell_count / rows)
        candidates.append((rows, columns))
    return min(
        candidates,
        key=lambda grid: (
            abs(grid[0] - grid[1]),
            grid[0] * grid[1] - cell_count,
            grid[0] > grid[1],
        ),
    )


def discover_results(output_root: Path) -> Dict[Path, Path]:
    """Map each result video by its sequence directory relative to a root."""
    results: Dict[Path, Path] = {}
    for video_path in sorted(output_root.rglob("result.mp4")):
        if not video_path.is_file():
            continue
        relative_dir = video_path.parent.relative_to(output_root)
        if relative_dir == Path("."):
            raise RuntimeError(
                f"result.mp4 must be inside a sequence directory: {video_path}"
            )
        results[relative_dir] = video_path
    return results


def probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        info = VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        cap.release()
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError(f"invalid video dimensions: {video_path}")
    if info.frame_count <= 0:
        raise RuntimeError(f"video contains no frames: {video_path}")
    if not math.isfinite(info.fps) or info.fps <= 0:
        raise RuntimeError(f"invalid video FPS: {video_path}")
    return info


def validate_video_info(
    sequence_dir: Path,
    reference_path: Path,
    reference: VideoInfo,
    current_path: Path,
    current: VideoInfo,
) -> None:
    mismatches = []
    for field in ("width", "height", "frame_count"):
        expected = getattr(reference, field)
        actual = getattr(current, field)
        if actual != expected:
            mismatches.append(f"{field}: {expected} != {actual}")
    if not math.isclose(reference.fps, current.fps, rel_tol=0.0, abs_tol=FPS_TOLERANCE):
        mismatches.append(f"fps: {reference.fps:.6g} != {current.fps:.6g}")
    if mismatches:
        raise RuntimeError(
            f"video parameters differ for sequence {sequence_dir.as_posix()}: "
            f"{reference_path} vs {current_path} ({'; '.join(mismatches)})"
        )


def collect_sequences(output_roots: Sequence[Path]) -> List[SequenceInputs]:
    """Discover and fully preflight all inputs before writing any output."""
    discovered = [discover_results(root) for root in output_roots]
    if not discovered[0]:
        raise RuntimeError(f"no result.mp4 files found below {output_roots[0]}")

    path_sets = [set(result) for result in discovered]
    common_paths = set.intersection(*path_sets)
    skipped_paths = set.union(*path_sets) - common_paths
    if skipped_paths:
        LOGGER.warning(
            "Skipping result.mp4 paths missing from one or more roots: %s",
            ", ".join(path.as_posix() for path in sorted(skipped_paths)),
        )
    if not common_paths:
        raise RuntimeError("no result.mp4 relative paths are shared by all input roots")

    names: Dict[str, Path] = {}
    for relative_dir in sorted(common_paths, key=lambda path: path.as_posix()):
        sequence_name = relative_dir.name
        if sequence_name in names:
            raise RuntimeError(
                "sequence-name collision in flat output: "
                f"{names[sequence_name].as_posix()} and {relative_dir.as_posix()} "
                f"both map to {sequence_name}.mp4"
            )
        names[sequence_name] = relative_dir

    sequences = []
    for relative_dir in sorted(common_paths, key=lambda path: path.as_posix()):
        sequence_name = relative_dir.name

        videos = tuple(result[relative_dir] for result in discovered)
        reference_info = probe_video(videos[0])
        for video_path in videos[1:]:
            current_info = probe_video(video_path)
            validate_video_info(
                relative_dir,
                videos[0],
                reference_info,
                video_path,
                current_info,
            )
        sequences.append(
            SequenceInputs(
                name=sequence_name,
                relative_dir=relative_dir,
                videos=videos,
                info=reference_info,
            )
        )
    return sequences


def fit_label(label: str, max_width: int, scale: float, thickness: int) -> str:
    """Ellipsize a label if it does not fit even at the minimum font scale."""
    if (
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        <= max_width
    ):
        return label
    suffix = "..."
    available = (
        max_width
        - cv2.getTextSize(suffix, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    )
    if available <= 0:
        return suffix
    end = len(label)
    while end > 0:
        candidate = label[:end]
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[
            0
        ][0]
        if width <= available:
            return candidate + suffix
        end -= 1
    return suffix


def make_title_bar(label: str, width: int, height: int) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    thickness = max(1, height // 24)
    max_text_width = max(1, width - 16)
    nominal_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness)
    height_scale = (height * 0.55) / max(1, nominal_size[1])
    width_scale = max_text_width / max(1, nominal_size[0])
    scale = min(1.5, height_scale, width_scale)
    minimum_scale = min(0.3, height_scale)
    scale = max(minimum_scale, scale)
    rendered_label = fit_label(label, max_text_width, scale, thickness)
    text_size, baseline = cv2.getTextSize(
        rendered_label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    origin_x = max(0, (width - text_size[0]) // 2)
    origin_y = max(text_size[1], (height + text_size[1] - baseline) // 2)
    cv2.putText(
        bar,
        rendered_label,
        (origin_x, origin_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return bar


def compose_sequence(
    sequence: SequenceInputs,
    labels: Sequence[str],
    grid: Grid,
    output_path: Path,
) -> None:
    rows, columns = grid
    width = sequence.info.width
    height = sequence.info.height
    title_height = max(32, round(height * 0.08))
    title_height += title_height % 2
    cell_height = title_height + height
    frame_size = (columns * width, rows * cell_height)
    title_bars = [make_title_bar(label, width, title_height) for label in labels]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sequence.info.fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"cannot create output video: {temporary_path}")

    captures = [cv2.VideoCapture(str(path)) for path in sequence.videos]
    try:
        for video_path, capture in zip(sequence.videos, captures):
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {video_path}")

        decoded_frames = 0
        while True:
            decoded = [capture.read() for capture in captures]
            readable = [ok and frame is not None for ok, frame in decoded]
            if not any(readable):
                break
            if not all(readable):
                ended = [
                    str(path)
                    for path, is_readable in zip(sequence.videos, readable)
                    if not is_readable
                ]
                raise RuntimeError(
                    f"videos ended at different frames for {sequence.relative_dir}: "
                    + ", ".join(ended)
                )

            canvas = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            for index, ((_, frame), title_bar) in enumerate(zip(decoded, title_bars)):
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"decoded frame dimensions changed in {sequence.videos[index]}: "
                        f"expected {width}x{height}, got "
                        f"{frame.shape[1]}x{frame.shape[0]}"
                    )
                row, column = divmod(index, columns)
                x = column * width
                y = row * cell_height
                canvas[y : y + title_height, x : x + width] = title_bar
                canvas[y + title_height : y + cell_height, x : x + width] = frame
            writer.write(canvas)
            decoded_frames += 1

        if decoded_frames != sequence.info.frame_count:
            raise RuntimeError(
                f"decoded frame count differs from metadata for "
                f"{sequence.relative_dir}: expected {sequence.info.frame_count}, "
                f"got {decoded_frames}"
            )
    except Exception:
        writer.release()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        for capture in captures:
            capture.release()
        writer.release()

    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one labeled comparison video per matching SAM 3 result sequence."
    )
    parser.add_argument(
        "output_roots",
        nargs="+",
        metavar="OUTPUT_ROOT",
        help="Prompt-specific roots containing result.mp4 files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for flat <sequence-name>.mp4 outputs",
    )
    parser.add_argument(
        "--grid",
        type=parse_grid,
        metavar="ROWSxCOLS",
        help="Skip the interactive grid prompt",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(args.output_roots) < 2:
        parser.error("at least two OUTPUT_ROOT arguments are required")
    output_roots = [expand_path(value) for value in args.output_roots]
    if len(set(output_roots)) != len(output_roots):
        parser.error("OUTPUT_ROOT arguments must be unique")
    for output_root in output_roots:
        if not output_root.is_dir():
            parser.error(f"OUTPUT_ROOT is not a directory: {output_root}")

    grid = args.grid or recommend_grid(len(output_roots))
    if grid[0] * grid[1] < len(output_roots):
        parser.error(
            f"grid {grid[0]}x{grid[1]} has fewer than " f"{len(output_roots)} cells"
        )

    output_dir = expand_path(args.output_dir)
    try:
        sequences = collect_sequences(output_roots)
        labels = [root.name for root in output_roots]
        LOGGER.info(
            "Validated %d sequence(s) across %d roots; using grid %dx%d",
            len(sequences),
            len(output_roots),
            grid[0],
            grid[1],
        )
        for index, sequence in enumerate(sequences, start=1):
            output_path = output_dir / f"{sequence.name}.mp4"
            LOGGER.info("[%d/%d] Writing %s", index, len(sequences), output_path)
            compose_sequence(sequence, labels, grid, output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import compare_dataset_videos as compare


@pytest.mark.parametrize(
    ("cell_count", "expected"),
    [(2, (1, 2)), (3, (2, 2)), (5, (2, 3)), (6, (2, 3)), (9, (3, 3))],
)
def test_recommend_grid(cell_count: int, expected: compare.Grid) -> None:
    assert compare.recommend_grid(cell_count) == expected


@pytest.mark.parametrize("value", ["0x2", "2x0", "2", "2 by 3", "-1x3"])
def test_parse_grid_rejects_invalid_values(value: str) -> None:
    with pytest.raises(Exception, match="ROWSxCOLS"):
        compare.parse_grid(value)


def touch_result(root: Path, relative_dir: str) -> None:
    result_path = root / relative_dir / "result.mp4"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.touch()


def test_collect_sequences_requires_identical_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    touch_result(first, "can")
    touch_result(first, "basket")
    touch_result(second, "can")

    with pytest.raises(RuntimeError, match=r"missing: basket"):
        compare.collect_sequences([first, second])


def test_collect_sequences_rejects_flat_name_collisions(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        touch_result(root, "group_a/can")
        touch_result(root, "group_b/can")

    with pytest.raises(RuntimeError, match="sequence-name collision"):
        compare.collect_sequences([first, second])


def write_video(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (16, 12))
    assert writer.isOpened()
    try:
        for frame_index in range(3):
            frame = np.full((12, 16, 3), color, dtype=np.uint8)
            frame[:, frame_index : frame_index + 2] = (255, 255, 255)
            writer.write(frame)
    finally:
        writer.release()


def test_main_writes_and_overwrites_comparison_video(tmp_path: Path) -> None:
    first = tmp_path / "prompt_one"
    second = tmp_path / "prompt_two"
    write_video(first / "can" / "result.mp4", (0, 0, 255))
    write_video(second / "can" / "result.mp4", (0, 255, 0))
    output_dir = tmp_path / "comparisons"
    arguments = [
        str(first),
        str(second),
        "--output-dir",
        str(output_dir),
        "--grid",
        "1x2",
    ]

    assert compare.main(arguments) == 0
    output_path = output_dir / "can.mp4"
    assert output_path.is_file()

    # A second run replaces the existing target without requiring a flag.
    assert compare.main(arguments) == 0
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 44
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12.0, abs=1e-3)
    finally:
        capture.release()

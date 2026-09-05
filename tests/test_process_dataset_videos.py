from pathlib import Path

import pytest

from scripts import process_dataset_videos as processor


def test_discover_accepts_single_sequence_or_dataset_tree(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    first = dataset / "a" / "color.mp4"
    second = dataset / "b" / "color.mp4"
    ignored = dataset / "b" / "result.mp4"
    for path in (first, second, ignored):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert processor.discover_color_videos(dataset) == [first, second]
    output_root = tmp_path / "output"
    assert processor.output_dir_for(first, dataset, output_root) == output_root / "a"


def test_direct_color_video_uses_output_root_directly(tmp_path: Path) -> None:
    input_root = tmp_path / "milk"
    input_root.mkdir()
    video = input_root / "color.mp4"
    video.touch()
    output_root = tmp_path / "output"

    assert processor.discover_color_videos(input_root) == [video]
    assert processor.output_dir_for(video, input_root, output_root) == output_root


def test_both_directions_get_independent_output_directories(tmp_path: Path) -> None:
    output = tmp_path / "output"

    assert processor.directional_output_dir(output, "forward", "forward") == output
    assert processor.directional_output_dir(output, "both", "forward") == (
        output / "forward"
    )
    assert processor.directional_output_dir(output, "both", "backward") == (
        output / "backward"
    )


def test_direction_choices_and_expansion() -> None:
    parser = processor.build_parser()
    args = parser.parse_args(["--prompt", "hand", "--direction", "both"])

    assert processor.processing_directions(args.direction) == (
        "forward",
        "backward",
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--prompt", "hand", "--direction", "sideways"])


def test_backward_propagation_writes_result_in_forward_playback_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeWriter:
        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

    class FakePredictor:
        request = None

        def handle_stream_request(self, request: dict):
            self.request = request
            for frame_index in (2, 1, 0):
                yield {"frame_index": frame_index, "outputs": processor.empty_outputs()}

    written = []
    monkeypatch.setattr(
        processor.cv2, "VideoWriter", lambda *args, **kwargs: FakeWriter()
    )
    monkeypatch.setattr(
        processor,
        "write_frame_outputs",
        lambda *args, **kwargs: written.append(args[3]),
    )
    predictor = FakePredictor()

    processor.propagate_and_write(
        predictor,
        "session",
        tmp_path,
        tmp_path / "masks.mkv",
        tmp_path / "result.mp4",
        3,
        16,
        12,
        10.0,
        "hand",
        "backward",
    )

    assert predictor.request["propagation_direction"] == "backward"
    assert predictor.request["start_frame_index"] == 2
    assert written == [0, 1, 2]


def test_list_only_accepts_single_sequence_directory(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "milk"
    input_root.mkdir()
    video = input_root / "color.mp4"
    video.touch()

    assert (
        processor.main(
            [
                "--input-root",
                str(input_root),
                "--output-root",
                str(tmp_path / "out"),
                "--prompt",
                "hand",
                "--list-only",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "color.mp4"

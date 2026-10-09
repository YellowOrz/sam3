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


def test_both_directions_share_one_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    assert processor.directional_output_dir(output, "forward", "forward") == output
    assert processor.directional_output_dir(output, "both", "both") == output


def test_clip_overseg_is_opt_in() -> None:
    parser = processor.build_parser()
    assert parser.parse_args(["--text-prompt", "hand"]).clip_overseg is False
    assert parser.parse_args(["--text-prompt", "hand", "--clip-overseg"]).clip_overseg
    with pytest.raises(SystemExit):
        processor.main(
            ["--text-prompt", "hand", "--version", "sam3.1", "--clip-overseg"]
        )


def test_direction_choices_and_expansion() -> None:
    parser = processor.build_parser()
    args = parser.parse_args(["--text-prompt", "hand", "--direction", "both"])

    assert processor.processing_directions(args.direction) == ("both",)
    with pytest.raises(SystemExit):
        parser.parse_args(["--text-prompt", "hand", "--direction", "sideways"])


def test_add_prompt_request_switches_text_and_learned() -> None:
    text = processor.add_prompt_request("s", 3, "hand", "text")
    learned = processor.add_prompt_request("s", 3, "left_hand", "learned")

    assert text == {
        "type": "add_prompt",
        "session_id": "s",
        "frame_index": 3,
        "text": "hand",
    }
    assert learned == {
        "type": "add_learned_prompt",
        "session_id": "s",
        "frame_index": 3,
        "target_id": "left_hand",
    }
    with pytest.raises(ValueError, match="unsupported prompt request type"):
        processor.add_prompt_request("s", 0, "hand", "box")


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
    monkeypatch.setattr(processor, "H264Writer", lambda *args, **kwargs: FakeWriter())
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
                "--text-prompt",
                "hand",
                "--list-only",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "color.mp4"


@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("reverse_anchor", [False, True])
def test_both_propagation_uses_same_session_and_anchor_and_writes_once(
    tmp_path, monkeypatch, start, reverse_anchor
):
    class Writer:
        def isOpened(self):
            return True

        def release(self):
            pass

    class Predictor:
        def __init__(self):
            self.requests = []

        def handle_stream_request(self, request):
            self.requests.append(request)
            if request["propagation_direction"] == "forward":
                indices = range(start, 5)
            else:
                indices = range(start if reverse_anchor else start - 1, -1, -1)
            for index in indices:
                yield {"frame_index": index, "outputs": processor.empty_outputs()}

    written = []
    monkeypatch.setattr(processor.cv2, "VideoWriter", lambda *a, **kw: Writer())
    monkeypatch.setattr(processor, "H264Writer", lambda *a, **kw: Writer())
    monkeypatch.setattr(
        processor, "write_frame_outputs", lambda *a, **kw: written.append(a[3])
    )
    predictor = Predictor()
    processor.propagate_and_write(
        predictor,
        "session",
        tmp_path,
        tmp_path / "masks.mkv",
        tmp_path / "result.mp4",
        5,
        16,
        12,
        10.0,
        "hand",
        "both",
        start_frame_index=start,
        prompt_outputs=processor.empty_outputs(),
    )
    assert written == list(range(5))
    assert [r["propagation_direction"] for r in predictor.requests] == (
        ["forward"] if start == 0 else ["forward", "backward"]
    )
    for request in predictor.requests:
        assert request["session_id"] == "session"
        assert request["start_frame_index"] == start
        assert request["max_frame_num_to_track"] == 5


@pytest.mark.parametrize(
    "direction, expected", [("forward", 0), ("backward", 4), ("both", 2)]
)
def test_default_propagation_start(direction, expected):
    assert processor.propagation_start_frame(5, direction) == expected
    assert processor.propagation_start_frame(1, direction) == 0


def test_both_cache_preserves_side_masks_anchor_and_detector_fallback(
    tmp_path, monkeypatch
):
    import numpy as np

    class Writer:
        def isOpened(self):
            return True

        def release(self):
            pass

    def outputs(value, detector=False):
        result = {
            "out_obj_ids": np.array([7]),
            "out_probs": np.array([0.9]),
            "out_binary_masks": np.full((1, 12, 16), value, dtype=np.uint8),
        }
        if detector:
            result["out_detector_mask"] = np.ones((12, 16), dtype=bool)
        return result

    class Predictor:
        def handle_stream_request(self, request):
            if request["propagation_direction"] == "forward":
                yield {"frame_index": 1, "outputs": outputs(1)}
                yield {"frame_index": 2, "outputs": outputs(2)}
            else:
                yield {"frame_index": 1, "outputs": outputs(9)}
                yield {"frame_index": 0, "outputs": outputs(3)}

    written = {}
    monkeypatch.setattr(processor.cv2, "VideoWriter", lambda *a, **kw: Writer())
    monkeypatch.setattr(processor, "H264Writer", lambda *a, **kw: Writer())
    monkeypatch.setattr(
        processor, "write_frame_outputs", lambda *a, **kw: written.update({a[3]: a[4]})
    )
    processor.propagate_and_write(
        Predictor(),
        "session",
        tmp_path,
        tmp_path / "masks.mkv",
        tmp_path / "result.mp4",
        3,
        16,
        12,
        10.0,
        "hand",
        "both",
        save_detector=True,
        prompt_outputs=outputs(1, detector=True),
    )
    assert list(written) == [0, 1, 2]
    assert [int(written[i]["out_binary_masks"][0, 0, 0]) for i in range(3)] == [3, 1, 2]
    assert written[1]["out_detector_mask"].all()
    assert all(written[i]["out_obj_ids"].tolist() == [7] for i in range(3))


@pytest.mark.parametrize("leg, indices", [("forward", [2, 1]), ("backward", [0, 1])])
def test_both_rejects_out_of_order_frames(tmp_path, monkeypatch, leg, indices):
    class Writer:
        def isOpened(self):
            return True

        def release(self):
            pass

    class Predictor:
        def handle_stream_request(self, request):
            if request["propagation_direction"] == leg:
                for index in indices:
                    yield {"frame_index": index, "outputs": processor.empty_outputs()}

    monkeypatch.setattr(processor.cv2, "VideoWriter", lambda *a, **kw: Writer())
    monkeypatch.setattr(processor, "H264Writer", lambda *a, **kw: Writer())
    with pytest.raises(RuntimeError, match="out-of-order"):
        processor.propagate_and_write(
            Predictor(),
            "session",
            tmp_path,
            tmp_path / "masks.mkv",
            tmp_path / "result.mp4",
            3,
            16,
            12,
            10.0,
            "hand",
            "both",
        )

import json
from pathlib import Path

import cv2
import numpy as np

from scripts import qualitative_test_interactive as qualitative


class FakePredictor:
    def __init__(self) -> None:
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        obj_id = request.get("obj_id", 1)
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:8] = True
        return {
            "frame_index": request.get("frame_index", 0),
            "outputs": {
                "out_obj_ids": np.array([obj_id]),
                "out_binary_masks": mask[None],
            },
        }


def test_parser_has_no_interactive_switch() -> None:
    args = qualitative.build_parser().parse_args(
        ["--video", "input.mp4", "--output-dir", "output"]
    )

    assert not hasattr(args, "interactive")


def make_app(tmp_path: Path) -> qualitative.InteractiveApp:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for frame_index in range(3):
        frame = np.full((8, 10, 3), 30 + frame_index, dtype=np.uint8)
        assert cv2.imwrite(str(frame_dir / f"{frame_index:05d}.jpg"), frame)
    return qualitative.InteractiveApp(
        predictor=FakePredictor(),
        version="sam3.1",
        session_id="session",
        frame_dir=frame_dir,
        video_path=tmp_path / "input.mp4",
        video_info=qualitative.VideoInfo(10, 8, 3, 12.0),
        frame_count=3,
        prompt="hand",
        output_dir=tmp_path / "output",
        window_width=1280,
    )


def test_normalize_masks_copies_and_indexes_masks() -> None:
    masks = np.zeros((2, 1, 4, 5), dtype=np.uint8)
    masks[0, 0, 1, 2] = 1
    masks[1, 0, 3, 4] = 1

    result = qualitative.normalize_masks(
        {"out_obj_ids": np.array([7, 11]), "out_binary_masks": masks}
    )

    assert set(result) == {7, 11}
    assert result[7].dtype == np.bool_
    assert result[7][1, 2]
    masks[:] = 0
    assert result[11][3, 4]
    assert qualitative.normalize_masks(None) == {}


def test_middle_click_cycles_overlapping_objects_smallest_first(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    large = np.zeros((8, 10), dtype=bool)
    small = np.zeros((8, 10), dtype=bool)
    large[1:7, 1:9] = True
    small[3:5, 4:6] = True
    app.cache[0] = {3: large, 9: small}

    app.select_object(4, 3)
    assert app.active_obj == 9
    app.select_object(4, 3)
    assert app.active_obj == 3


def test_left_and_right_click_pause_then_build_global_draft(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 4
    stops = []
    app.stop_propagation = lambda: stops.append(True)

    app.add_point(2, 3, 1)
    app.active_obj = 8
    app.add_point(7, 6, 0)

    assert stops == [True]
    assert app.editing
    assert app.edit_frame == 0
    assert [(point.obj_id, point.label) for point in app.draft_points] == [
        (4, 1),
        (8, 0),
    ]
    app.undo()
    assert app.active_obj == 8
    assert [(point.obj_id, point.label) for point in app.draft_points] == [(4, 1)]


def test_preview_then_enter_commits_once_and_starts_propagation(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    starts = []
    app.start_propagation = lambda frame_index: starts.append(frame_index)
    app.add_point(4, 2, 1)
    point_sequence = app.draft_points[0].sequence

    app.preview()
    add_requests = [
        request
        for request in app.predictor.requests
        if request.get("type") == "add_prompt"
    ]
    assert add_requests[-1]["points"] == [[4, 2]]
    assert add_requests[-1]["point_labels"] == [1]
    assert add_requests[-1]["rel_coordinates"] is False

    app.confirm()
    assert not app.editing
    assert [point.sequence for point in app.confirmed_points] == [point_sequence]
    assert app.commits[0].point_sequences == (point_sequence,)
    assert starts == [0]
    assert len(add_requests) == 1


def test_enter_without_preview_applies_points_before_propagating(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.active_obj = 2
    app.stop_propagation = lambda: None
    starts = []
    app.start_propagation = lambda frame_index: starts.append(frame_index)
    app.add_point(1, 1, 0)

    app.confirm()

    assert app.confirmed_points[0].label == 0
    assert starts == [0]
    assert any(
        request.get("type") == "add_prompt" for request in app.predictor.requests
    )


def test_interactive_outputs_write_only_video_and_json(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    app.cache = {frame_index: {1: mask} for frame_index in range(3)}
    app.confirmed_points = [qualitative.PointEdit(1, 1, 1, 4, 3, 1)]
    app.events = [{"sequence": 1, "type": "confirm"}]

    qualitative.write_interactive_outputs(app)

    result_path = app.output_dir / "result.mp4"
    metadata_path = app.output_dir / "interactions.json"
    assert result_path.is_file()
    assert metadata_path.is_file()
    assert not list(app.output_dir.glob("*.png"))
    capture = cv2.VideoCapture(str(result_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert capture.get(cv2.CAP_PROP_FPS) == 12.0
    finally:
        capture.release()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"]["fps"] == 12.0
    assert metadata["confirmed_points"][0]["frame_index"] == 1


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "result.mp4").touch()

    try:
        qualitative.validate_interactive_outputs(output_dir, overwrite=False)
    except RuntimeError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("existing output should have been rejected")

    qualitative.validate_interactive_outputs(output_dir, overwrite=True)

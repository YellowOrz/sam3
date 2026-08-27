import json
from contextlib import nullcontext
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import torch

from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_video_inference import _frame_progress
from scripts import qualitative_test_interactive as qualitative


class FakePredictor:
    def __init__(self) -> None:
        self.requests = []
        self.checkpoint_frames = {0}

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "save_checkpoint":
            created = request["frame_index"] not in self.checkpoint_frames
            self.checkpoint_frames.add(request["frame_index"])
            return {
                "is_success": True,
                "frame_index": request["frame_index"],
                "created": created,
            }
        if request["type"] == "restore_checkpoint":
            candidates = [
                frame
                for frame in self.checkpoint_frames
                if frame <= request["frame_index"]
            ]
            return {
                "is_success": bool(candidates),
                "frame_index": max(candidates) if candidates else None,
            }
        if request["type"] == "clear_checkpoints":
            self.checkpoint_frames.clear()
            return {"is_success": True}
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

    def handle_stream_request(self, request):
        start = request.get("start_frame_index", 0)
        end = start + request.get("max_frame_num_to_track", 0)
        for frame_index in range(start, end + 1):
            mask = np.zeros((8, 10), dtype=bool)
            mask[2:6, 3:8] = True
            yield {
                "frame_index": frame_index,
                "outputs": {
                    "out_obj_ids": np.array([1]),
                    "out_binary_masks": mask[None],
                },
            }


def test_forward_progress_uses_absolute_frame_numbers() -> None:
    progress = _frame_progress(range(50, 200), 200, reverse=False, file=StringIO())

    assert (progress.n, progress.total) == (50, 200)
    list(progress)
    assert progress.n == 200


def test_parser_has_no_interactive_switch() -> None:
    args = qualitative.build_parser().parse_args(
        ["--video", "input.mp4", "--output-dir", "output"]
    )

    assert not hasattr(args, "interactive")
    assert args.propagation_direction == "both"
    assert args.checkpoint_interval == 20
    assert qualitative.WINDOW_FLAGS & cv2.WINDOW_GUI_NORMAL

    forward = qualitative.build_parser().parse_args(
        [
            "--video",
            "input.mp4",
            "--output-dir",
            "output",
            "--propagation-direction",
            "forward",
        ]
    )
    assert forward.propagation_direction == "forward"


def test_propagation_thread_binds_callers_cuda_device(monkeypatch) -> None:
    calls = []

    class StreamingPredictor:
        def handle_stream_request(self, request):
            calls.append(
                (
                    "predict",
                    request["session_id"],
                    request["propagation_direction"],
                )
            )
            return iter(())

    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qualitative.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        qualitative.torch.cuda,
        "set_device",
        lambda device: calls.append(("set_device", device)),
    )
    monkeypatch.setattr(
        qualitative.torch,
        "autocast",
        lambda **kwargs: nullcontext(),
    )
    runner = qualitative.PropagationRunner(
        StreamingPredictor(), qualitative.queue.Queue(), "both"
    )

    runner.start("session", start_frame_index=0, generation=1)
    runner.join()

    assert calls == [("set_device", 3), ("predict", "session", "both")]


def test_propagation_thread_reports_backward_phase(monkeypatch) -> None:
    class StreamingPredictor:
        def handle_stream_request(self, request):
            for frame_index in (1, 2, 0):
                yield {"frame_index": frame_index, "outputs": {}}

        def handle_request(self, request):
            return {
                "is_success": True,
                "frame_index": request["frame_index"],
                "created": True,
            }

    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qualitative.torch, "autocast", lambda **kwargs: nullcontext())
    events = qualitative.queue.Queue()
    runner = qualitative.PropagationRunner(
        StreamingPredictor(), events, "both", checkpoint_interval=100
    )

    runner.start("session", start_frame_index=1, generation=3)
    runner.join()

    queued = []
    while not events.empty():
        queued.append(events.get_nowait())
    assert ("direction", 3, -1, 1) in queued


def make_app(
    tmp_path: Path, propagation_direction: str = "both"
) -> qualitative.InteractiveApp:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for frame_index in range(3):
        frame = np.full((8, 10, 3), 30 + frame_index, dtype=np.uint8)
        assert cv2.imwrite(str(frame_dir / f"{frame_index:05d}.jpg"), frame)
    app = qualitative.InteractiveApp(
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
        propagation_direction=propagation_direction,
    )
    app.follow_live = False
    app.playing = False
    return app


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

    app.playing = True
    app.add_point(2, 3, 1)
    assert not app.draft_points
    assert app.status == "pause playback before editing"
    app.playing = False

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


def test_point_limit_rejects_seventeenth_point(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 4
    stops = []
    app.stop_propagation = lambda: stops.append(True)

    for index in range(qualitative.MAX_PROMPT_POINTS):
        app.add_point(index % 10, index % 8, index % 2)
    sequence_at_limit = app.sequence
    event_count_at_limit = len(app.events)
    app.add_point(1, 1, 1)

    assert stops == [True]
    assert len(app.draft_points) == qualitative.MAX_PROMPT_POINTS
    assert app.sequence == sequence_at_limit
    assert len(app.events) == event_count_at_limit
    assert app.status == "point limit reached (16) for obj 4"


def test_completed_propagation_stays_open_for_review(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.generation = 2
    app.follow_live = True
    app.playing = True
    app.event_queue.put(("done", 2, False, None))

    app.drain_events()

    assert app.propagation_complete
    assert not app.follow_live
    assert not app.playing
    assert app.status == "complete - ready for review"
    assert app.handle_key(255)


def test_forward_propagation_only_invalidates_later_frames(tmp_path: Path) -> None:
    app = make_app(tmp_path, propagation_direction="forward")
    app.cache = {0: {}, 1: {}, 2: {}}
    starts = []
    app.runner.start = lambda session_id, frame_index, generation: starts.append(
        (session_id, frame_index, generation)
    )

    app.start_propagation(1)

    assert app.stale_frames == {1, 2}
    assert starts == [("session", 1, 1)]


def test_space_pauses_propagation_and_playback(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.playing = True
    point = qualitative.PointEdit(1, 0, 3, 4, 2, 1)
    app.confirmed_points = [point]
    app.runner.thread = type("LiveThread", (), {"is_alive": lambda self: True})()
    stopped = []
    app.stop_propagation = lambda: stopped.append(True)

    assert app.handle_key(ord(" "))

    assert stopped == [True]
    assert not app.follow_live
    assert not app.playing
    assert app.status == "paused"
    assert app.confirmed_points == [point]


def test_playback_waits_for_fresh_frames_then_reverses(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.stale_frames = {0, 2}
    app.start_playback_cycle(1)

    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 1

    app.stale_frames.discard(2)
    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 2

    app.reverse_ready = True
    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 1
    assert app.playback_direction == -1

    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 1
    app.stale_frames.discard(0)
    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 0
    app.last_play_time = 0
    app.advance_playback()
    assert not app.playing
    assert app.playback_origin is None


def test_forward_playback_stops_at_end(tmp_path: Path) -> None:
    app = make_app(tmp_path, propagation_direction="forward")
    app.cache = {0: {}, 1: {}, 2: {}}
    app.stale_frames = set()
    app.start_playback_cycle(1)
    app.display_index = 2
    app.propagation_complete = True

    app.last_play_time = 0
    app.advance_playback()

    assert not app.playing
    assert app.playback_origin is None
    assert app.display_index == 2


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
    assert add_requests[-1]["points"] == [[0.4, 0.25]]
    assert add_requests[-1]["point_labels"] == [1]
    assert add_requests[-1]["rel_coordinates"] is True

    app.confirm()
    assert not app.editing
    assert [point.sequence for point in app.confirmed_points] == [point_sequence]
    assert app.commits[0].point_sequences == (point_sequence,)
    assert starts == [0]
    assert len(add_requests) == 1
    assert app.playing
    assert not app.follow_live
    assert app.playback_origin == 0


def test_points_on_multiple_frames_are_applied_independently(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.confirmed_points = [
        qualitative.PointEdit(1, 0, 5, 2, 3, 1),
        qualitative.PointEdit(2, 2, 5, 7, 6, 0),
    ]

    app.apply_points([(0, 5), (2, 5)])

    requests = [
        request
        for request in app.predictor.requests
        if request.get("type") == "add_prompt" and "points" in request
    ]
    assert [
        (request["frame_index"], request["point_labels"]) for request in requests
    ] == [
        (0, [1]),
        (2, [0]),
    ]


def test_preview_rollback_restores_checkpoint_without_reloading_frames(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    app.add_point(4, 2, 1)
    app.preview()
    app.undo()
    starts = []
    app.start_propagation = lambda frame_index: starts.append(frame_index)

    app.confirm()

    request_types = [request["type"] for request in app.predictor.requests]
    assert "restore_checkpoint" in request_types
    assert "reset_session" not in request_types
    assert "close_session" not in request_types
    assert "start_session" not in request_types
    assert starts == [0]


def test_changed_preview_is_recomputed_from_confirmed_state(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    app.add_point(4, 2, 0)
    app.preview()
    app.add_point(5, 3, 1)

    app.preview()

    request_types = [request["type"] for request in app.predictor.requests]
    assert request_types.count("restore_checkpoint") == 2
    assert "reset_session" not in request_types
    assert "close_session" not in request_types
    add_requests = [
        request
        for request in app.predictor.requests
        if request.get("type") == "add_prompt" and "points" in request
    ]
    assert add_requests[-1]["points"] == [[0.4, 0.25], [0.5, 0.375]]
    assert add_requests[-1]["point_labels"] == [0, 1]


def test_undo_then_preview_restores_the_same_checkpoint(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    app.add_point(4, 2, 0)
    app.preview()
    first_points = [
        request["points"]
        for request in app.predictor.requests
        if request.get("type") == "add_prompt" and "points" in request
    ][-1]

    app.add_point(5, 3, 1)
    app.preview()
    app.undo()
    app.preview()

    restore_requests = [
        request
        for request in app.predictor.requests
        if request["type"] == "restore_checkpoint"
    ]
    final_points = [
        request["points"]
        for request in app.predictor.requests
        if request.get("type") == "add_prompt" and "points" in request
    ][-1]
    assert len(restore_requests) == 3
    assert final_points == first_points


def test_cpu_checkpoints_reuse_unchanged_tensor_snapshots() -> None:
    predictor = Sam3BasePredictor()
    tracked = torch.arange(8, dtype=torch.bfloat16)
    state = {
        "input_batch": torch.ones(100, dtype=torch.float32),
        "constants": {"constant": torch.ones(50)},
        "feature_cache": {"temporary": torch.ones(25)},
        "previous_stages_out": ["done", "done", None],
        "tracker_inference_states": [{"memory": tracked}],
        "action_history": [{"type": "propagation_full", "frame_idx": 0}],
    }
    predictor._all_inference_states["session"] = {
        "state": state,
        "session_id": "session",
        "last_use_time": 0.0,
        "checkpoints": {},
        "checkpoint_tensor_cache": {},
    }

    predictor.save_checkpoint("session", 1)
    second = predictor.save_checkpoint("session", 2)
    duplicate = predictor.save_checkpoint("session", 2)
    checkpoints = predictor._all_inference_states["session"]["checkpoints"]
    first_snapshot = checkpoints[1]["tracker_inference_states"][0]["memory"]
    second_snapshot = checkpoints[2]["tracker_inference_states"][0]["memory"]
    tracked.fill_(99)
    restored = predictor.restore_checkpoint("session", 1)

    assert first_snapshot is second_snapshot
    assert second["created"] and not duplicate["created"]
    assert restored["frame_index"] == 1
    restored_memory = predictor._all_inference_states["session"]["state"][
        "tracker_inference_states"
    ][0]["memory"]
    assert torch.equal(restored_memory, torch.arange(8, dtype=torch.bfloat16))
    assert (
        "temporary"
        not in predictor._all_inference_states["session"]["state"]["feature_cache"]
    )


def test_cpu_checkpoints_support_mutated_inference_tensors() -> None:
    predictor = Sam3BasePredictor()
    with torch.inference_mode():
        tracked = torch.arange(4, dtype=torch.float32)
    state = {
        "previous_stages_out": ["done", None],
        "tracker_inference_states": [{"memory": tracked}],
        "action_history": [],
    }
    predictor._all_inference_states["session"] = {
        "state": state,
        "session_id": "session",
        "last_use_time": 0.0,
        "checkpoints": {},
        "checkpoint_tensor_cache": {},
    }

    predictor.save_checkpoint("session", 0)
    with torch.inference_mode():
        tracked.add_(10)
    state["previous_stages_out"][1] = "done"
    predictor.save_checkpoint("session", 1)

    predictor.restore_checkpoint("session", 0)
    first = predictor._all_inference_states["session"]["state"][
        "tracker_inference_states"
    ][0]["memory"]
    assert torch.equal(first, torch.arange(4, dtype=torch.float32))

    predictor.restore_checkpoint("session", 1)
    second = predictor._all_inference_states["session"]["state"][
        "tracker_inference_states"
    ][0]["memory"]
    assert torch.equal(second, torch.arange(4, dtype=torch.float32) + 10)


def test_unchanged_preview_does_not_run_model_twice(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    app.add_point(4, 2, 1)
    app.preview()
    request_count = len(app.predictor.requests)

    app.preview()

    assert len(app.predictor.requests) == request_count


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


def test_c_clears_all_interactions_and_restarts_text_propagation(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.confirmed_points = [qualitative.PointEdit(1, 0, 3, 4, 2, 1)]
    app.draft_points = [qualitative.PointEdit(2, 0, 3, 5, 3, 0)]
    app.commits = [qualitative.CommitRecord(0, ((0, 3),), (1,))]
    app.editing = True
    app.edit_frame = 0
    starts = []
    app.start_propagation = lambda frame_index: starts.append(frame_index)

    assert app.handle_key(ord("c"))

    assert app.confirmed_points == []
    assert app.draft_points == []
    assert app.commits == []
    assert not app.editing
    assert app.active_obj is None
    assert starts == [0]
    assert any(event["type"] == "clear_all_interactions" for event in app.events)
    request_types = [request["type"] for request in app.predictor.requests]
    assert "reset_session" in request_types
    assert "close_session" not in request_types


def test_mouse_clicks_and_keyboard_keys_are_logged(tmp_path: Path, capsys) -> None:
    app = make_app(tmp_path)
    app.select_object = lambda x, y: None
    app.add_point = lambda x, y, label: None

    app.on_mouse(cv2.EVENT_LBUTTONDOWN, 2, 3, 0, None)
    app.on_mouse(cv2.EVENT_MBUTTONDOWN, 4, 5, 0, None)
    app.on_mouse(cv2.EVENT_RBUTTONDOWN, 6, 7, 0, None)
    assert app.handle_key(ord("x"))

    output = capsys.readouterr().err
    assert output.count("INTERACTION ") == 4
    assert '"button": "left"' in output
    assert '"button": "middle"' in output
    assert '"button": "right"' in output
    assert '"key": "x"' in output


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
    assert metadata["propagation_direction"] == "both"
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

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
        if request["type"] == "remove_object":
            return {
                "is_success": True,
                "frame_index": request.get("frame_index", 0),
            }
        obj_id = request.get("obj_id", 1)
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:8] = True
        return {
            "frame_index": request.get("frame_index", 0),
            "outputs": {
                "out_obj_ids": np.array([obj_id]),
                "out_probs": np.array([0.75]),
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
                    "out_probs": np.array([0.75]),
                    "out_binary_masks": mask[None],
                },
            }


def test_forward_progress_uses_absolute_frame_numbers() -> None:
    progress = _frame_progress(range(50, 200), 200, reverse=False, file=StringIO())

    assert (progress.n, progress.total) == (50, 200)
    list(progress)
    assert progress.n == 200


def test_reverse_progress_counts_down_absolute_frame_numbers() -> None:
    output = StringIO()
    progress = _frame_progress(range(4, -1, -1), 10, reverse=True, file=output)

    assert list(progress) == [4, 3, 2, 1, 0]
    assert "5/10" in output.getvalue()
    assert "0/10" in output.getvalue()


def test_cuda_amp_prefers_bfloat16_when_supported(monkeypatch) -> None:
    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qualitative.torch.cuda, "is_bf16_supported", lambda: True)
    assert qualitative.cuda_amp_dtype() is torch.bfloat16


def test_cuda_amp_falls_back_to_float16_without_bf16(monkeypatch) -> None:
    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qualitative.torch.cuda, "is_bf16_supported", lambda: False)
    assert qualitative.cuda_amp_dtype() is torch.float16


def test_parser_has_no_interactive_switch() -> None:
    args = qualitative.build_parser().parse_args(
        ["--video", "input.mp4", "--output-dir", "output"]
    )

    assert not hasattr(args, "interactive")
    assert not hasattr(args, "propagation_direction")
    assert args.checkpoint_interval == 20
    assert args.chunk_frames == 0
    assert qualitative.WINDOW_FLAGS & cv2.WINDOW_GUI_NORMAL


def test_chunk_frames_parser_and_ranges() -> None:
    args = qualitative.build_parser().parse_args(
        [
            "--video",
            "input.mp4",
            "--output-dir",
            "output",
            "--chunk-frames",
            "500",
        ]
    )

    assert args.chunk_frames == 500
    assert qualitative.frame_ranges(2000, args.chunk_frames) == [
        (0, 500),
        (500, 1000),
        (1000, 1500),
        (1500, 2000),
    ]
    assert qualitative.frame_ranges(1001, 500)[-1] == (1000, 1001)
    assert qualitative.chunk_frames_int("0") == 0
    assert qualitative.chunk_frames_int("100") == 100
    try:
        qualitative.chunk_frames_int("99")
    except qualitative.argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("chunk sizes below 100 should be rejected")


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
        StreamingPredictor(), qualitative.queue.Queue()
    )

    runner.start("session", start_frame_index=0, generation=1, mode="forward")
    runner.join()

    assert calls == [("set_device", 3), ("predict", "session", "forward")]


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
        StreamingPredictor(), events, checkpoint_interval=100
    )

    runner.start("session", start_frame_index=1, generation=3, mode="forward_backward")
    runner.join()

    queued = []
    while not events.empty():
        queued.append(events.get_nowait())
    assert ("direction", 3, "backward", "forward_backward") in queued


def test_backward_propagation_saves_directional_cpu_checkpoints(monkeypatch) -> None:
    saved = []

    class StreamingPredictor:
        def handle_stream_request(self, request):
            for frame_index in range(99, -1, -1):
                yield {"frame_index": frame_index, "outputs": {}}

        def handle_request(self, request):
            saved.append(request)
            return {
                "is_success": True,
                "frame_index": request["frame_index"],
                "propagation_direction": request["propagation_direction"],
                "created": True,
            }

    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qualitative.torch, "autocast", lambda **kwargs: nullcontext())
    runner = qualitative.PropagationRunner(
        StreamingPredictor(), qualitative.queue.Queue(), checkpoint_interval=20
    )

    runner.start("session", start_frame_index=100, generation=1, mode="backward")
    runner.join()

    assert [request["frame_index"] for request in saved] == [100, 80, 60, 40, 20, 0]
    assert all(request["propagation_direction"] == "backward" for request in saved)
    assert all(request["exact_frame"] for request in saved)


def make_app(tmp_path: Path) -> qualitative.InteractiveApp:
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
    )
    app.playing = False
    return app


def test_control_panel_draws_dynamic_timeline_and_all_buttons(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.display_index = 1

    panel = app.render_controls()
    rendered = app.render()

    assert panel.shape == (
        qualitative.CONTROL_HEIGHT,
        app.display_width,
        3,
    )
    assert rendered.shape == (
        app.display_height + qualitative.CONTROL_HEIGHT,
        app.display_width,
        3,
    )
    assert app.timeline_bounds[0] < app.timeline_bounds[1] < app.display_width - 20
    assert set(app.hitboxes) == {
        "play_backward",
        "play_step_backward",
        "play_pause",
        "play_step_forward",
        "play_forward",
        "prop_pause",
        "prop_backward",
        "prop_forward",
        "prop_forward_backward",
        "prop_backward_forward",
    }


def test_playback_and_propagation_controls_are_independent(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.playing = True
    stopped = []
    app.stop_propagation = lambda: stopped.append(True)

    app.handle_control_click("prop_pause")
    assert app.playing
    assert stopped == [True]

    app.propagation_mode = "forward"
    app.handle_control_click("play_pause")
    assert not app.playing
    assert app.propagation_mode == "forward"


def test_play_direction_clears_selection_but_keeps_draft_points(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    draft = qualitative.PointEdit(1, 1, 7, 4, 3, 1)
    app.display_index = 1
    app.active_obj = 7
    app.selection_position = (1, 4, 3)
    app.selection_candidates = [7]
    app.draft_points = [draft]

    app.handle_control_click("play_pause")
    assert app.active_obj == 7

    app.handle_control_click("play_backward")
    assert app.active_obj is None
    assert app.selection_position is None
    assert app.selection_candidates == []
    assert app.draft_points == [draft]

    app.active_obj = 7
    app.handle_control_click("play_forward")
    assert app.active_obj is None
    assert app.draft_points == [draft]


def test_step_buttons_move_one_playable_frame_while_paused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.display_index = 0
    app.playing = False
    app.active_obj = 7
    draft = qualitative.PointEdit(1, 0, 7, 4, 3, 1)
    app.draft_points = [draft]

    app.handle_control_click("play_step_forward")
    assert app.display_index == 1
    assert not app.playing
    assert app.active_obj == 7
    assert app.draft_points == [draft]

    app.handle_control_click("play_step_forward")
    assert app.display_index == 1
    assert app.status == "at playable frontier"

    app.handle_control_click("play_step_backward")
    assert app.display_index == 0
    app.handle_control_click("play_step_backward")
    assert app.display_index == 0
    assert app.status == "at first frame"


def test_step_buttons_are_ignored_while_playing(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.display_index = 1
    app.playing = True
    app.playback_direction = 1

    app.handle_control_click("play_step_forward")
    app.handle_control_click("play_step_backward")

    assert app.display_index == 1
    assert app.playing
    assert app.status == "pause playback before stepping frames"


def test_mouse_hitbox_uses_canvas_coordinates_below_video(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.render_controls()
    left, top, right, bottom = app.hitboxes["play_forward"]

    app.on_mouse(
        cv2.EVENT_LBUTTONDOWN,
        (left + right) // 2,
        app.display_height + (top + bottom) // 2,
        0,
        None,
    )

    assert app.playing
    assert app.playback_direction == 1


def test_enter_and_space_do_not_commit_when_paused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 2
    app.add_point(1, 1, 1)

    assert app.handle_key(13)
    assert app.handle_key(32)

    assert app.editing
    assert app.draft_points
    assert not app.confirmed_points
    assert app.status == "Enter and Space are disabled"


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


def test_render_frame_matches_dataset_annotations(monkeypatch) -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    texts = []
    original_put_text = cv2.putText

    def capture_text(image, text, *args):
        texts.append(text)
        return original_put_text(image, text, *args)

    monkeypatch.setattr(cv2, "putText", capture_text)

    qualitative.render_frame_bgr(
        frame,
        {7: mask},
        probabilities_by_obj={7: 0.876},
        object_to_label={7: 1},
        frame_index=3,
        prompt="hand",
    )

    assert "label=1 id=7 p=0.88" in texts
    assert "frame=3 prompt=hand" in texts
    assert qualitative.EDGE_HALO_THICKNESS == 2
    assert qualitative.EDGE_COLOR_THICKNESS == 1


def test_overlapping_masks_blend_both_instance_colors(monkeypatch) -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    monkeypatch.setattr(cv2, "drawContours", lambda image, *args: image)
    monkeypatch.setattr(cv2, "putText", lambda image, *args: image)

    rendered = qualitative.render_frame_bgr(
        frame,
        {1: mask, 2: mask},
        object_to_label={1: 1, 2: 2},
    )

    first = np.asarray(qualitative.COLORS[0], dtype=np.float32)
    second = np.asarray(qualitative.COLORS[1], dtype=np.float32)
    expected = (
        first * qualitative.MASK_ALPHA * (1.0 - qualitative.MASK_ALPHA)
        + second * qualitative.MASK_ALPHA
    ).astype(np.uint8)
    assert np.array_equal(rendered[10, 10], expected)


def test_active_mask_has_cyan_fill_and_thick_outline(monkeypatch) -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.ones((20, 20), dtype=bool)
    thicknesses = []
    original_draw_contours = cv2.drawContours

    def capture_contours(image, contours, index, color, thickness, line_type):
        thicknesses.append(thickness)
        return original_draw_contours(
            image, contours, index, color, thickness, line_type
        )

    monkeypatch.setattr(cv2, "drawContours", capture_contours)
    monkeypatch.setattr(cv2, "putText", lambda image, *args: image)

    rendered = qualitative.render_frame_bgr(frame, {7: mask}, active_obj=7)

    expected = (
        np.asarray(qualitative.SELECTED_COLOR) * qualitative.SELECTED_MASK_ALPHA
    ).astype(np.uint8)
    assert np.array_equal(rendered[10, 10], expected)
    assert qualitative.SELECTED_HALO_THICKNESS == 4
    assert qualitative.SELECTED_EDGE_THICKNESS == 2
    assert qualitative.SELECTED_HALO_THICKNESS in thicknesses
    assert qualitative.SELECTED_EDGE_THICKNESS in thicknesses


def test_propagation_modes_expand_in_requested_order() -> None:
    assert qualitative.propagation_legs("backward") == ("backward",)
    assert qualitative.propagation_legs("forward") == ("forward",)
    assert qualitative.propagation_legs("forward_backward") == (
        "forward",
        "backward",
    )
    assert qualitative.propagation_legs("backward_forward") == (
        "backward",
        "forward",
    )


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
    app.select_object(4, 3)
    assert app.active_obj is None


def test_points_require_both_modes_paused_then_build_global_draft(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.active_obj = 4

    app.playing = True
    app.add_point(2, 3, 1)
    assert not app.draft_points
    assert app.status == "pause playback and propagation before editing"
    app.playing = False

    app.add_point(2, 3, 1)
    app.active_obj = 8
    app.add_point(7, 6, 0)

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
    for index in range(qualitative.MAX_PROMPT_POINTS):
        app.add_point(index % 10, index % 8, index % 2)
    sequence_at_limit = app.sequence
    event_count_at_limit = len(app.events)
    app.add_point(1, 1, 1)

    assert len(app.draft_points) == qualitative.MAX_PROMPT_POINTS
    assert app.sequence == sequence_at_limit
    assert len(app.events) == event_count_at_limit
    assert app.status == "point limit reached (16) for obj 4"


def test_positive_point_without_selection_creates_non_reused_object(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.cache[0] = {3: np.ones((8, 10), dtype=bool)}
    app.next_obj_id = 4

    app.add_point(2, 3, 0)
    assert not app.draft_points
    assert app.status == "add a positive point to create an object"

    app.add_point(2, 3, 1)
    assert app.active_obj == 4
    assert app.draft_points[0].obj_id == 4
    app.cancel_edit()
    assert app.active_obj is None

    app.add_point(3, 4, 1)
    assert app.active_obj == 5


def test_completed_propagation_does_not_change_playback(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.generation = 2
    app.playing = True
    app.propagation_mode = "forward"
    app.event_queue.put(("done", 2, False, None))

    app.drain_events()

    assert app.propagation_complete
    assert app.playing
    assert app.propagation_mode is None
    assert app.status == "propagation complete (forward)"
    assert app.handle_key(255)


def test_forward_propagation_only_invalidates_later_frames(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    starts = []
    app.runner.start = (
        lambda session_id, frame_index, generation, mode, **kwargs: starts.append(
            (session_id, frame_index, generation, mode)
        )
    )

    app.start_propagation(1, "forward")

    assert app.stale_frames == {2}
    assert starts == [("session", 1, 1, "forward")]


def test_space_is_ignored_while_running(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.playing = True
    point = qualitative.PointEdit(1, 0, 3, 4, 2, 1)
    app.confirmed_points = [point]
    app.runner.thread = type("LiveThread", (), {"is_alive": lambda self: True})()

    assert app.handle_key(ord(" "))

    assert app.playing
    assert app.runner.is_alive
    assert app.status == "pause playback and propagation before editing"
    assert app.confirmed_points == [point]


def test_playback_waits_at_frontier_and_reverse_stops_at_zero(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.stale_frames = set()
    app.display_index = 1
    app.playing = True
    app.playback_direction = 1

    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 1
    assert app.playing

    app.cache[2] = {}
    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 2

    app.handle_control_click("play_backward")
    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 1
    app.last_play_time = 0
    app.advance_playback()
    assert app.display_index == 0
    app.last_play_time = 0
    app.advance_playback()
    assert not app.playing


def test_playback_can_show_processed_frames_marked_stale(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.stale_frames = {1}
    app.display_index = 0
    app.playing = True
    app.playback_direction = 1
    app.last_play_time = 0

    app.advance_playback()

    assert app.display_index == 1


def test_frame_status_overlay_draws_badge_and_border() -> None:
    image = np.zeros((90, 220, 3), dtype=np.uint8)
    qualitative.draw_frame_status_overlay(image, "stale")
    fg = qualitative.FRAME_STATUS_STYLES["stale"]["fg"]

    assert tuple(int(value) for value in image[0, 0]) == fg
    assert tuple(int(value) for value in image[0, -1]) == fg


def test_display_frame_status_current_stale_and_preview(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.stale_frames = {1}
    app.display_index = 0
    assert app.display_frame_status() == "current"

    app.display_index = 1
    assert app.display_frame_status() == "stale"

    app.editing = True
    app.edit_frame = 1
    app.preview_signature = ((1, 1, 1, 2, 3, 1),)
    assert app.display_frame_status() == "preview"


def test_render_marks_stale_and_current_frames(tmp_path: Path, monkeypatch) -> None:
    texts: list[str] = []
    original_put_text = cv2.putText

    def capture_text(image, text, *args):
        texts.append(text)
        return original_put_text(image, text, *args)

    monkeypatch.setattr(cv2, "putText", capture_text)
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.stale_frames = {1}

    app.display_index = 1
    rendered = app.render()
    assert "STALE" in texts
    fg = qualitative.FRAME_STATUS_STYLES["stale"]["fg"]
    assert tuple(int(value) for value in rendered[0, 0]) == fg

    texts.clear()
    app.display_index = 0
    rendered = app.render()
    assert "CURRENT" in texts
    fg = qualitative.FRAME_STATUS_STYLES["current"]["fg"]
    assert tuple(int(value) for value in rendered[0, 0]) == fg


def test_forward_playback_stops_at_end(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.stale_frames = set()
    app.playing = True
    app.playback_direction = 1
    app.display_index = 2
    app.propagation_complete = True

    app.last_play_time = 0
    app.advance_playback()

    assert not app.playing
    assert app.display_index == 2


def test_preview_then_direction_commits_once_and_starts_propagation(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    starts = []
    app.start_propagation = lambda frame_index, mode: starts.append((frame_index, mode))
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

    app.confirm_and_propagate("backward_forward")
    assert not app.editing
    assert [point.sequence for point in app.confirmed_points] == [point_sequence]
    assert app.commits[0].point_sequences == (point_sequence,)
    assert starts == [(0, "backward_forward")]
    assert len(add_requests) == 1
    assert not app.playing


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

    request_types = [request["type"] for request in app.predictor.requests]
    assert "restore_checkpoint" in request_types
    assert "reset_session" not in request_types
    assert "close_session" not in request_types
    assert "start_session" not in request_types
    assert not app.editing


def test_preview_rebuilds_when_editing_before_the_only_checkpoint(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.stale_frames = set()
    app.display_index = 2
    app.active_obj = 1
    app.start_propagation = lambda frame_index, mode: None

    app.add_point(4, 2, 1)
    app.confirm_and_propagate("forward")
    assert app.predictor.checkpoint_frames == {2}

    app.display_index = 1
    app.add_point(5, 3, 0)
    app.preview()

    assert app.status == "preview"
    assert any(event["type"] == "preview_baseline_rebuilt" for event in app.events)
    assert any(request["type"] == "reset_session" for request in app.predictor.requests)


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


def test_cpu_checkpoint_restore_uses_nearest_compatible_direction() -> None:
    predictor = Sam3BasePredictor()
    state = {
        "previous_stages_out": ["done"] * 101,
        "marker": torch.tensor(60),
        "action_history": [],
    }
    predictor._all_inference_states["session"] = {
        "state": state,
        "session_id": "session",
        "last_use_time": 0.0,
        "checkpoints": {},
        "checkpoint_tensor_cache": {},
    }
    predictor.save_checkpoint(
        "session", 60, propagation_direction="forward", exact_frame=True
    )
    state["marker"] = torch.tensor(-80)
    predictor.save_checkpoint(
        "session", 80, propagation_direction="forward", exact_frame=True
    )
    state["marker"] = torch.tensor(100)
    predictor.save_checkpoint(
        "session", 100, propagation_direction="backward", exact_frame=True
    )
    state["marker"] = torch.tensor(80)
    predictor.save_checkpoint(
        "session", 80, propagation_direction="backward", exact_frame=True
    )

    restored = predictor.restore_checkpoint("session", 75)

    assert restored == {
        "is_success": True,
        "frame_index": 80,
        "propagation_direction": "backward",
    }
    assert predictor._all_inference_states["session"]["state"]["marker"].item() == 80

    exact = predictor.restore_checkpoint("session", 80)
    assert exact["propagation_direction"] == "backward"
    assert predictor._all_inference_states["session"]["state"]["marker"].item() == 80


def test_unchanged_preview_does_not_run_model_twice(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.active_obj = 5
    app.stop_propagation = lambda: None
    app.add_point(4, 2, 1)
    app.preview()
    request_count = len(app.predictor.requests)

    app.preview()

    assert len(app.predictor.requests) == request_count


def test_direction_without_preview_applies_points_before_propagating(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.active_obj = 2
    app.stop_propagation = lambda: None
    starts = []
    app.start_propagation = lambda frame_index, mode: starts.append((frame_index, mode))
    app.add_point(1, 1, 0)

    app.confirm_and_propagate("forward")

    assert app.confirmed_points[0].label == 0
    assert starts == [(0, "forward")]
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
    app.start_propagation = lambda frame_index, mode: starts.append((frame_index, mode))

    assert app.handle_key(ord("c"))

    assert app.confirmed_points == []
    assert app.draft_points == []
    assert app.commits == []
    assert not app.editing
    assert app.active_obj is None
    assert starts == [(0, "forward")]
    assert any(event["type"] == "clear_all_interactions" for event in app.events)
    request_types = [request["type"] for request in app.predictor.requests]
    assert "reset_session" in request_types
    assert "close_session" not in request_types


def test_d_removes_selected_object_from_all_cached_frames(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keep = np.ones((8, 10), dtype=bool)
    drop = np.zeros((8, 10), dtype=bool)
    drop[1:3, 1:3] = True
    app.cache = {
        0: {1: keep.copy(), 2: drop.copy()},
        1: {1: keep.copy(), 2: drop.copy()},
    }
    app.probability_cache = {0: {1: 0.9, 2: 0.4}, 1: {1: 0.8, 2: 0.3}}
    app.confirmed_points = [
        qualitative.PointEdit(1, 0, 1, 4, 2, 1),
        qualitative.PointEdit(2, 0, 2, 5, 3, 0),
    ]
    app.commits = [
        qualitative.CommitRecord(0, ((0, 1),), (1,)),
        qualitative.CommitRecord(0, ((0, 2),), (2,)),
    ]
    app.active_obj = 2
    app.display_index = 1

    assert app.handle_key(ord("d"))

    assert 2 not in app.cache[0]
    assert 2 not in app.cache[1]
    assert 1 in app.cache[0] and 1 in app.cache[1]
    assert app.probability_cache[0] == {1: 0.9}
    assert [point.obj_id for point in app.confirmed_points] == [1]
    assert app.commits == [qualitative.CommitRecord(0, ((0, 1),), (1,))]
    assert app.active_obj is None
    assert app.status == "removed obj 2"
    request_types = [request["type"] for request in app.predictor.requests]
    assert request_types.count("remove_object") == 1
    assert "clear_checkpoints" in request_types
    assert "save_checkpoint" in request_types
    assert any(
        event["type"] == "remove_object" and event["obj_id"] == 2
        for event in app.events
    )


def test_d_discards_uncommitted_new_object_without_model_remove(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {1: np.ones((8, 10), dtype=bool)}}
    app.next_obj_id = 2
    app.active_obj = None
    app.add_point(4, 3, 1)
    new_id = app.active_obj
    assert new_id == 2
    assert 2 in app.draft_new_obj_ids
    app.predictor.requests.clear()

    assert app.handle_key(ord("d"))

    assert not app.editing
    assert app.draft_points == []
    assert app.status == f"discarded draft obj {new_id}"
    assert not any(
        request.get("type") == "remove_object" for request in app.predictor.requests
    )
    assert 1 in app.cache[0]


def test_d_requires_selection_and_paused_playback(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {1: np.ones((8, 10), dtype=bool)}}
    app.playing = True
    app.active_obj = 1

    assert app.handle_key(ord("d"))
    assert 1 in app.cache[0]
    assert app.status == "pause playback and propagation before editing"

    app.playing = False
    app.active_obj = None
    assert app.handle_key(ord("d"))
    assert app.status == "select an object before deleting"


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


def test_q_only_exits_after_every_frame_was_processed(tmp_path: Path, capsys) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 2: {}}
    app.active_obj = 1
    app.add_point(4, 3, 1)
    draft = list(app.draft_points)

    assert app.handle_key(ord("q"))
    assert "1 unprocessed frames (1)" in app.status
    assert "还有 1 帧未处理" in capsys.readouterr().err
    assert app.draft_points == draft

    app.cache[1] = {}
    app.stale_frames = {1, 2}
    assert not app.handle_key(ord("q"))
    assert app.draft_points == draft


def test_finalize_neither_commits_draft_nor_propagates(
    tmp_path: Path, monkeypatch
) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.active_obj = 1
    app.add_point(4, 3, 1)
    draft = list(app.draft_points)
    written = []
    app.stop_propagation = lambda: None
    app.commit_draft = lambda: (_ for _ in ()).throw(AssertionError("committed"))
    app.synchronous_propagation = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("propagated")
    )
    monkeypatch.setattr(
        qualitative,
        "write_interactive_outputs",
        lambda current: written.append(current),
    )

    app.finalize()

    assert app.draft_points == draft
    assert written == [app]


def test_interactive_outputs_write_mask_video_and_metadata(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    app.cache = {frame_index: {1: mask} for frame_index in range(3)}
    app.probability_cache = {frame_index: {1: 0.75} for frame_index in range(3)}
    app.confirmed_points = [qualitative.PointEdit(1, 1, 1, 4, 3, 1)]
    app.events = [{"sequence": 1, "type": "confirm"}]

    qualitative.write_interactive_outputs(app)

    result_path = app.output_dir / "result.mp4"
    masks_path = app.output_dir / "masks.mkv"
    metadata_path = app.output_dir / "metadata.json"
    interactions_path = app.output_dir / "interactions.json"
    assert result_path.is_file()
    assert masks_path.is_file()
    assert metadata_path.is_file()
    assert interactions_path.is_file()
    assert not list(app.output_dir.glob("*.png"))
    capture = cv2.VideoCapture(str(result_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert capture.get(cv2.CAP_PROP_FPS) == 12.0
    finally:
        capture.release()
    capture = cv2.VideoCapture(str(masks_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        ok, labels = capture.read()
        assert ok
        assert set(np.unique(labels)) == {0, 1}
    finally:
        capture.release()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"]["fps"] == 12.0
    assert metadata["frames_processed"] == 3
    assert metadata["object_id_to_label"] == {"1": 1}
    assert metadata["outputs"]["instance_masks_codec"] == "FFV1"
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    assert interactions["propagation_direction"] == "forward"
    assert interactions["confirmed_points"][0]["frame_index"] == 1


def test_merge_chunk_outputs_concatenates_videos_and_offsets_interactions(
    tmp_path: Path,
) -> None:
    chunks = []
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:8] = True
    for chunk_index, (start, frame_count) in enumerate(((0, 2), (2, 1)), 1):
        chunk_root = tmp_path / f"chunk-{chunk_index}"
        chunk_root.mkdir()
        app = make_app(chunk_root)
        app.frame_count = frame_count
        app.video_info = qualitative.VideoInfo(10, 8, frame_count, 12.0)
        app.frame_offset = start
        app.cache = {index: {1: mask} for index in range(frame_count)}
        app.probability_cache = {index: {1: 0.75} for index in range(frame_count)}
        app.confirmed_points = [qualitative.PointEdit(1, 0, 1, 4, 3, 1)]
        app.events = [
            {
                "sequence": 1,
                "type": "confirm",
                "frame_index": 0,
                "point_sequences": [1],
            }
        ]
        qualitative.write_interactive_outputs(app)
        chunks.append((start, start + frame_count, app.output_dir))

    output_dir = tmp_path / "merged"
    qualitative.merge_chunk_outputs(
        chunks,
        output_dir,
        tmp_path / "input.mp4",
        qualitative.VideoInfo(10, 8, 3, 12.0),
        "hand",
        "sam3.1",
        2,
    )

    capture = cv2.VideoCapture(str(output_dir / "result.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["frames_processed"] == 3
    assert [
        (chunk["start_frame"], chunk["end_frame_exclusive"])
        for chunk in metadata["chunks"]
    ] == [
        (0, 2),
        (2, 3),
    ]
    interactions = json.loads((output_dir / "interactions.json").read_text())
    assert [point["frame_index"] for point in interactions["confirmed_points"]] == [
        0,
        2,
    ]
    assert [point["sequence"] for point in interactions["confirmed_points"]] == [
        1,
        2,
    ]
    assert [event["point_sequences"] for event in interactions["events"]] == [
        [1],
        [2],
    ]


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


def test_prompt_markers_merge_frames_and_jump_when_playable(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}}
    app.confirmed_points = [qualitative.PointEdit(1, 1, 3, 4, 2, 1)]
    app.draft_points = [qualitative.PointEdit(2, 1, 3, 5, 2, 1)]
    app.display_index = 0
    app.render_controls()

    assert [frame for frame, _rect in app.prompt_marker_hitboxes] == [1]
    _, rect = app.prompt_marker_hitboxes[0]
    app.on_mouse(
        cv2.EVENT_LBUTTONDOWN,
        (rect[0] + rect[2]) // 2,
        app.display_height + (rect[1] + rect[3]) // 2,
        0,
        None,
    )

    assert app.display_index == 1
    assert not app.playing


def test_prompt_marker_skips_unplayable_frames(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}}
    app.display_index = 0
    app.seek_prompt_marker(2)

    assert app.display_index == 0
    assert "not playable" in app.status


def test_propagation_stops_at_range_bounds(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.prop_range_left = 0
    app.prop_range_right = 1
    captured: dict = {}

    def fake_start(
        session_id,
        frame_index,
        generation,
        mode,
        max_forward_track=None,
        max_backward_track=None,
    ):
        del session_id, generation
        captured.update(
            {
                "frame_index": frame_index,
                "mode": mode,
                "max_forward_track": max_forward_track,
                "max_backward_track": max_backward_track,
            }
        )

    app.runner.start = fake_start
    app.start_propagation(0, "forward")

    assert captured["max_forward_track"] == 1
    assert captured["max_backward_track"] == 0
    assert app.stale_frames == {1}
    assert app.propagation_track_limit(2, "forward") == 0
    assert app.propagation_track_limit(2, "backward") == 0


def test_range_brackets_align_bar_to_bound_frames(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.prop_range_left = 0
    app.prop_range_right = 2
    app.render_controls()

    left_x = app.track_x_for_frame(0)
    right_x = app.track_x_for_frame(2)
    left_box = app.range_hitboxes["left"]
    right_box = app.range_hitboxes["right"]

    assert left_box[0] < left_x <= left_box[2]
    assert left_x - left_box[0] > left_box[2] - left_x
    assert right_box[0] <= right_x < right_box[2]
    assert right_box[2] - right_x > right_x - right_box[0]


def test_bracket_keys_set_propagation_range(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.display_index = 1
    app.handle_key(ord("]"))
    assert (app.prop_range_left, app.prop_range_right) == (0, 1)

    app.display_index = 2
    app.handle_key(ord("["))
    assert (app.prop_range_left, app.prop_range_right) == (2, 2)


def test_clear_all_keeps_propagation_range(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.prop_range_left = 1
    app.prop_range_right = 2
    starts = []
    app.start_propagation = lambda frame_index, mode: starts.append((frame_index, mode))
    app.restore_confirmed_state = lambda: None
    app.save_checkpoint = lambda frame_index: None
    app.stop_propagation = lambda: None

    app.clear_all_interactions()

    assert (app.prop_range_left, app.prop_range_right) == (1, 2)
    assert starts == [(0, "forward")]

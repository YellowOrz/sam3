import json
import threading
from contextlib import nullcontext
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pytest
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
    parser = qualitative.build_parser()
    argv = ["--video", "input.mp4", "--output-dir", "output"]
    args = parser.parse_args([*argv, "--text-prompt", "hand"])
    assert args.text_prompt == "hand"
    for old_option in ("--prompt", "--text_prompt"):
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, old_option, "hand"])

    assert not hasattr(args, "interactive")
    assert not hasattr(args, "propagation_direction")
    assert args.checkpoint_interval == 20
    assert args.chunk_frames == 0
    assert args.clip_overseg is False
    assert qualitative.WINDOW_FLAGS & cv2.WINDOW_GUI_NORMAL


def test_clip_overseg_requires_base_sam3() -> None:
    enabled = qualitative.build_parser().parse_args(
        [
            "--video",
            "input.mp4",
            "--output-dir",
            "output",
            "--version",
            "sam3",
            "--clip-overseg",
        ]
    )
    assert enabled.clip_overseg is True
    with pytest.raises(SystemExit):
        qualitative.main(
            ["--video", "input.mp4", "--output-dir", "output", "--clip-overseg"]
        )


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
    monkeypatch.setattr(qualitative, "cuda_amp_dtype", lambda: torch.float16)

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
            if "frame_index" not in request:
                return {"is_success": True, "updated": 0}
            return {
                "is_success": True,
                "frame_index": request["frame_index"],
                "created": True,
                "updated": 0,
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
            if "frame_index" not in request:
                return {"is_success": True, "updated": 0}
            return {
                "is_success": True,
                "frame_index": request["frame_index"],
                "propagation_direction": request.get("propagation_direction"),
                "created": request.get("type") == "save_checkpoint",
                "updated": 0,
            }

    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qualitative.torch, "autocast", lambda **kwargs: nullcontext())
    runner = qualitative.PropagationRunner(
        StreamingPredictor(), qualitative.queue.Queue(), checkpoint_interval=20
    )

    runner.start("session", start_frame_index=100, generation=1, mode="backward")
    runner.join()

    saves = [request for request in saved if request.get("type") == "save_checkpoint"]
    assert [request["frame_index"] for request in saves] == [100, 80, 60, 40, 20, 0]
    assert all(request["propagation_direction"] == "backward" for request in saves)
    assert all(request["exact_frame"] for request in saves)


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
        app.display_width + qualitative.HELP_PANEL_WIDTH,
        3,
    )
    assert app.timeline_bounds[0] < app.timeline_bounds[1] < app.display_width - 20
    help_strip = rendered[:, app.display_width :]
    assert help_strip.shape[1] == qualitative.HELP_PANEL_WIDTH
    assert tuple(int(v) for v in help_strip[8, 8]) == (27, 29, 32)
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


def test_help_panel_ignores_clicks(tmp_path: Path, capsys) -> None:
    app = make_app(tmp_path)
    added: list[tuple[int, int, int]] = []
    app.add_point = lambda x, y, label: added.append((x, y, label))

    app.on_mouse(cv2.EVENT_LBUTTONDOWN, app.display_width + 10, 3, 0, None)

    assert added == []
    assert "INTERACTION " not in capsys.readouterr().err


def test_key_help_copy_switches_language() -> None:
    title, note, rows, mouse_title, mouse_rows = qualitative.key_help_copy(True)
    assert title == "键盘说明"
    assert "暂停" in note
    assert rows[0] == ("P", "刷新当前帧预览")
    assert mouse_title == "鼠标说明"
    assert mouse_rows[0][0] == "左键"

    title, note, rows, mouse_title, mouse_rows = qualitative.key_help_copy(False)
    assert title == "Keyboard"
    assert rows[0][0] == "P"
    assert "preview" in rows[0][1]
    assert mouse_title == "Mouse"
    assert mouse_rows[0][0] == "L-click"


def test_english_help_panel_renders_without_cjk_font() -> None:
    panel = qualitative.render_key_help_panel(200, use_cjk=False)
    assert panel.shape == (200, qualitative.HELP_PANEL_WIDTH, 3)
    assert tuple(int(v) for v in panel[8, 8]) == (27, 29, 32)


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


def test_frame_mutation_rate_uses_max_one_minus_iou() -> None:
    hand = np.zeros((4, 4), dtype=bool)
    hand[:2, :2] = True
    grown = np.ones((4, 4), dtype=bool)
    other = np.zeros((4, 4), dtype=bool)
    other[2:, 2:] = True
    empty = np.zeros((4, 4), dtype=bool)

    assert qualitative.frame_mutation_rate({1: hand}, {1: hand}) == 0.0
    assert qualitative.frame_mutation_rate({1: hand}, {1: grown}) == pytest.approx(0.75)
    assert qualitative.frame_mutation_rate({1: hand}, {2: other}) == 1.0
    assert qualitative.frame_mutation_rate({1: hand}, {}) == 1.0
    assert qualitative.frame_mutation_rate({}, {}) is None
    assert qualitative.frame_mutation_rate({1: empty}, {1: empty}) is None
    assert (
        qualitative.frame_mutation_rate({1: hand, 2: hand}, {1: hand, 2: other}) == 1.0
    )


def test_mutation_series_breaks_when_a_frame_is_missing() -> None:
    mask = np.ones((2, 2), dtype=bool)
    series = qualitative.mutation_series(
        {0: {1: mask}, 1: {1: mask}, 3: {1: mask}, 4: {1: mask}},
        5,
    )

    assert series == [(1, 0.0), (4, 0.0)]


def test_mutation_chart_click_seeks_without_pausing(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[0, 0] = True
    app.cache = {0: {1: mask}, 1: {1: mask}, 2: {1: mask}}
    app.display_index = 0
    app.playing = True
    app.playback_direction = -1
    app.last_play_time = 0.0
    app.active_obj = 7
    draft = qualitative.PointEdit(1, 0, 7, 4, 3, 1)
    app.draft_points = [draft]
    y = app.display_height + qualitative.TIMELINE_HEIGHT + 4

    app.on_mouse(cv2.EVENT_LBUTTONDOWN, app.track_x_for_frame(2), y, 0, None)

    assert app.display_index == 2
    assert app.playing
    assert app.playback_direction == -1
    assert app.active_obj == 7
    assert app.draft_points == [draft]
    assert app.last_play_time > 0

    app.playing = False
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, app.track_x_for_frame(1), y, 0, None)
    assert app.display_index == 1
    assert not app.playing

    app.cache = {0: {1: mask}}
    app.display_index = 0
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, app.track_x_for_frame(2), y, 0, None)
    assert app.display_index == 0


def test_mutation_chart_draws_a_spike(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    small = np.zeros((8, 10), dtype=bool)
    small[0, 0] = True
    app.cache = {0: {1: small}, 1: {1: np.ones((8, 10), dtype=bool)}}

    panel = app.render_controls()
    band = panel[
        qualitative.TIMELINE_HEIGHT : qualitative.TIMELINE_HEIGHT
        + qualitative.MUTATION_HEIGHT
    ]

    assert int(band[:, :, 1].max()) > 150


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


def test_confirm_keeps_checkpoints_outside_the_edit(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    app.stale_frames = set()
    app.display_index = 2
    app.active_obj = 1
    app.start_propagation = lambda frame_index, mode: None

    app.add_point(4, 2, 1)
    app.confirm_and_propagate("forward")

    assert app.predictor.checkpoint_frames == {0, 2}
    assert "clear_checkpoints" not in [
        request["type"] for request in app.predictor.requests
    ]
    app.display_index = 1
    app.add_point(5, 3, 0)
    app.preview()
    assert app.status == "preview"
    assert not any(event["type"] == "preview_baseline_rebuilt" for event in app.events)


def test_preview_rebuilds_when_no_checkpoint_can_reach_the_edit(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    app.predictor.checkpoint_frames = {2}
    app.cache = {0: {}, 1: {}, 2: {}}
    app.stale_frames = set()
    app.display_index = 1
    app.active_obj = 1
    app.stop_propagation = lambda: None

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


def test_updated_checkpoint_keeps_the_previous_snapshot_outside_its_span() -> None:
    predictor = Sam3BasePredictor()
    state = {
        "previous_stages_out": ["done"] * 301,
        "marker": torch.tensor(1),
        "action_history": [],
    }
    predictor._all_inference_states["session"] = {
        "state": state,
        "session_id": "session",
        "last_use_time": 0.0,
        "checkpoints": {},
        "checkpoint_tensor_cache": {},
    }
    for frame in (0, 80, 260, 280):
        predictor.save_checkpoint(
            "session", frame, propagation_direction="forward", exact_frame=True
        )
    predictor.limit_checkpoint_spans("session", [0, 80, 260, 280], (0, 300))
    state["marker"] = torch.tensor(999)
    predictor.update_frame_checkpoints("session", 260)
    predictor.limit_checkpoint_spans("session", [260], (250, 260))
    state["marker"] = torch.tensor(250)
    predictor.save_checkpoint(
        "session", 250, propagation_direction="both", exact_frame=True
    )
    predictor.limit_checkpoint_spans("session", [250], (250, 260))

    assert predictor.restore_checkpoint("session", 90)["frame_index"] == 80
    assert predictor.restore_checkpoint("session", 255)["frame_index"] == 250
    outside = predictor.restore_checkpoint("session", 270)
    assert outside["frame_index"] == 260
    assert predictor._all_inference_states["session"]["state"]["marker"].item() == 1

    state = predictor._all_inference_states["session"]["state"]
    state["marker"] = torch.tensor(7)
    predictor.save_checkpoint(
        "session", 40, propagation_direction="both", exact_frame=True
    )
    predictor.update_frame_checkpoints("session", 40)
    predictor.limit_checkpoint_spans("session", [40], (40, 40))
    assert predictor.restore_checkpoint("session", 30)["frame_index"] == 0


def test_propagation_segments_reprompt_before_the_confirmed_frame() -> None:
    assert qualitative.propagation_segments(90, 260, "forward", [250]) == [
        ("propagate", 90, 159),
        ("reprompt", 250, 0),
        ("propagate", 250, 10),
    ]
    assert qualitative.propagation_segments(250, 0, "backward", [90]) == [
        ("propagate", 250, 159),
        ("reprompt", 90, 0),
        ("propagate", 91, 91),
    ]
    assert qualitative.propagation_segments(0, 20, "forward", []) == [
        ("propagate", 0, 20)
    ]


def test_runner_reprompts_before_crossing_a_confirmed_frame(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(qualitative.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qualitative.torch, "autocast", lambda **kwargs: nullcontext())

    class StreamingPredictor:
        def handle_stream_request(self, request):
            calls.append(request)
            start = request["start_frame_index"]
            limit = request.get("max_frame_num_to_track", 0)
            for frame_index in range(start, start + limit + 1):
                yield {"frame_index": frame_index, "outputs": {}}

        def handle_request(self, request):
            calls.append(request)
            frame = request.get("frame_index")
            return {
                "is_success": True,
                "updated": 1 if frame == 10 else 0,
                "created": request["type"] == "save_checkpoint" and frame == 0,
            }

    prompted = []
    runner = qualitative.PropagationRunner(
        StreamingPredictor(), qualitative.queue.Queue(), checkpoint_interval=100
    )
    runner.start(
        "session",
        start_frame_index=0,
        generation=1,
        mode="forward",
        max_forward_track=10,
        frame_count=30,
        reprompt_frames=(5,),
        reprompt=prompted.append,
    )
    runner.join()

    propagated = [
        (request["start_frame_index"], request.get("max_frame_num_to_track"))
        for request in calls
        if request.get("type") == "propagate_in_video"
    ]
    assert propagated == [(0, 4), (5, 5)]
    assert prompted == [5]
    limits = [
        request for request in calls if request["type"] == "limit_checkpoint_spans"
    ]
    assert limits[-1]["frame_indexes"] == [0, 10]
    assert limits[-1]["valid_span"] == [0, 10]


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
    assert metadata["object_id_to_label"] == {"0": 1}
    assert metadata["original_object_id_to_object_id"] == {"1": 0}
    assert metadata["outputs"]["instance_masks_codec"] == "FFV1"
    assert metadata["clip_overseg"] is False
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    assert interactions["propagation_direction"] == "forward"
    assert interactions["confirmed_points"][0]["frame_index"] == 1
    assert interactions["confirmed_points"][0]["obj_id"] == 0
    assert interactions["confirmed_points"][0]["original_obj_id"] == 1


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
    frame_dir = tmp_path / "all-frames"
    frame_dir.mkdir()
    for index in range(3):
        cv2.imwrite(
            str(frame_dir / f"{index:05d}.jpg"), np.zeros((8, 10, 3), dtype=np.uint8)
        )
    qualitative.merge_chunk_outputs(
        chunks,
        output_dir,
        tmp_path / "input.mp4",
        qualitative.VideoInfo(10, 8, 3, 12.0),
        "hand",
        "sam3.1",
        2,
        frame_dir,
    )

    capture = cv2.VideoCapture(str(output_dir / "result.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["frames_processed"] == 3
    assert metadata["clip_overseg"] is False
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
        **kwargs,
    ):
        del kwargs
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


def test_render_cache_tracks_image_and_control_changes(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    app.cache = {0: {1: np.ones((8, 10), dtype=bool)}}
    loads = []
    original_load = qualitative.load_frame_bgr

    def load(*args):
        loads.append(args)
        return original_load(*args)

    monkeypatch.setattr(qualitative, "load_frame_bgr", load)
    rendered = app.render()
    assert app.render() is rendered
    assert len(loads) == 1

    def replace_mask():
        app.cache[0][1] = np.zeros((8, 10), dtype=bool)

    changes = [
        lambda: setattr(app, "active_obj", 1),
        lambda: app.draft_points.append(qualitative.PointEdit(1, 0, 1, 3, 4, 1)),
        lambda: app.draft_points.clear(),
        lambda: app.stale_frames.add(0),
        lambda: app.probability_cache.update({0: {1: 0.5}}),
        replace_mask,
        lambda: app.cache[0].pop(1),
        lambda: app.cache.update({1: {}}),
        lambda: setattr(app, "playing", True),
        lambda: setattr(app, "playback_direction", -1),
        lambda: setattr(app, "prop_range_right", 1),
        lambda: setattr(app, "status", "propagating backward"),
        lambda: app.confirmed_points.append(qualitative.PointEdit(2, 1, 1, 3, 4, 1)),
        lambda: setattr(app, "display_index", 1),
    ]
    for change in changes:
        change()
        updated = app.render()
        assert updated is not rendered
        assert app.render() is updated
        app._render_key = None
        np.testing.assert_array_equal(updated, app.render())
        rendered = app.render()


def test_cached_window_still_pumps_keyboard_events(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    app.cache = {0: {}, 1: {}, 2: {}}
    shown = []
    waits = []
    for name in ("namedWindow", "setWindowTitle", "setMouseCallback", "destroyWindow"):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: shown.append(args))

    def wait(delay):
        waits.append(delay)
        return ord("q") if len(waits) == 3 else 255

    monkeypatch.setattr(cv2, "waitKey", wait)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 1)
    monkeypatch.setattr(app, "finalize", lambda: None)
    app.run()
    assert len(shown) == 1
    assert waits == [15, 15, 15]


def test_checkpoints_skip_nested_recomputed_features(tmp_path, monkeypatch):
    from sam3.model import sam3_base_predictor as base

    predictor = Sam3BasePredictor()
    feature = torch.ones(25)
    features = {3: feature}
    memory = torch.arange(4)
    state = {
        "feature_cache": features,
        "tracker_inference_states": [
            {"cached_features": features, "memory": memory},
            {"cached_features": features, "memory": memory},
        ],
        "unrelated": {"cached_features": torch.tensor(7)},
    }
    original_clone = base._clone_state_to_cpu

    def clone(value, *args):
        assert value is not feature, "recomputed features must not be copied"
        return original_clone(value, *args)

    monkeypatch.setattr(base, "_clone_state_to_cpu", clone)
    predictor._all_inference_states["session"] = {
        "state": state,
        "session_id": "session",
        "last_use_time": 0.0,
        "checkpoints": {},
        "checkpoint_tensor_cache": {},
    }
    predictor.save_checkpoint("session", 3)
    assert state["feature_cache"][3] is feature
    assert state["tracker_inference_states"][0]["cached_features"] is features
    snapshot = predictor._all_inference_states["session"]["checkpoints"][3]
    assert snapshot["tracker_inference_states"][0]["cached_features"] == {}
    predictor.restore_checkpoint("session", 3)
    first, second = state["tracker_inference_states"]
    assert (
        first["cached_features"] is second["cached_features"] is state["feature_cache"]
    )
    assert first["memory"] is second["memory"]
    assert torch.equal(first["memory"], memory)
    assert state["unrelated"]["cached_features"].item() == 7


@pytest.mark.parametrize("stop", [None, "interrupt", "unconfirmed", "skip"])
def test_batch_reuses_model_and_reports_failures_immediately(
    tmp_path, monkeypatch, capsys, stop
):
    root, output = tmp_path / "input", tmp_path / "output"
    for directory in (root, root / "a", root / "b"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rgb.mkv").touch()
    events = []

    class Predictor:
        def shutdown(self):
            events.append("shutdown")

    predictor = Predictor()

    def load(*_):
        events.append("load")
        return predictor

    def process(model, args, video, destination):
        assert model is predictor
        assert (
            destination
            == output / video.parent.relative_to(root) / "masks_sam3" / "circle"
        )
        error = capsys.readouterr().err
        assert ("broken video" in error) == (
            stop is None and video.parent == root / "b"
        )
        events.append(video.parent.relative_to(root).as_posix())
        if video.parent == root / "a":
            if stop == "interrupt":
                raise KeyboardInterrupt
            if stop == "unconfirmed":
                return False
            if stop == "skip":
                raise qualitative.SkipVideo
            raise RuntimeError("broken video")
        return True

    monkeypatch.setattr(qualitative, "load_predictor", load)
    monkeypatch.setattr(qualitative, "process_interactive_video", process)
    assert qualitative.main(
        [
            "--input-root",
            str(root),
            "--output-root",
            str(output),
            "--rgb-name",
            "rgb.mkv",
        ]
    ) == (0 if stop == "skip" else 1)
    assert events == (
        ["load", "a", "b", ".", "shutdown"]
        if stop in (None, "skip")
        else ["load", "a", "shutdown"]
    )
    report = capsys.readouterr()
    assert ("broken video" in report.err) == (stop is None)
    if stop == "skip":
        assert "2 succeeded, 1 skipped, 0 failed" in report.out


@pytest.mark.parametrize(
    "incomplete", ["unconfirmed", "metadata_only", "corrupt_video"]
)
def test_batch_skips_only_complete_outputs_and_overwrite_reprocesses(
    tmp_path, monkeypatch, incomplete
):
    app = make_app(tmp_path)
    output_root = app.output_dir
    app.output_dir = output_root / "masks_sam3" / app.prompt.replace(" ", "_")
    app.cache = {i: {} for i in range(3)}
    qualitative.write_interactive_outputs(app)
    args = qualitative.build_parser().parse_args(
        [
            "--input-root",
            str(app.video_path.parent),
            "--output-root",
            str(output_root),
            "--text-prompt",
            app.prompt,
            "--version",
            app.version,
        ]
    )
    # The test app uses PNG input frames; provide the matching source probe only.
    probe = qualitative.probe_video
    monkeypatch.setattr(
        qualitative,
        "probe_video",
        lambda p: app.video_info if p == app.video_path else probe(p),
    )
    assert qualitative.completed_interactive_video(app.output_dir, app.video_path, args)
    monkeypatch.setattr(qualitative, "discover_rgb_videos", lambda *_: [app.video_path])
    loads = []

    class Predictor:
        def shutdown(self):
            pass

    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *_: loads.append(1) or Predictor()
    )
    monkeypatch.setattr(qualitative, "process_interactive_video", lambda *_: True)
    argv = [
        "--input-root",
        str(app.video_path.parent),
        "--output-root",
        str(output_root),
        "--text-prompt",
        app.prompt,
        "--version",
        app.version,
    ]
    assert qualitative.main(argv) == 0
    assert loads == []
    metadata_path = app.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Complete outputs with different settings remain protected.
    metadata["prompt"] = "different prompt"
    metadata_path.write_text(json.dumps(metadata))
    assert qualitative.main(argv) == 1
    assert loads == []
    metadata["prompt"] = app.prompt
    if incomplete == "unconfirmed":
        metadata["status"] = "unconfirmed"
    elif incomplete == "metadata_only":
        for name in ("result.mp4", "masks.mkv", "interactions.json"):
            (app.output_dir / name).unlink()
    else:
        (app.output_dir / "masks.mkv").write_bytes(b"broken")
    metadata_path.write_text(json.dumps(metadata))
    assert qualitative.main(argv) == 0
    assert loads == [1]
    assert qualitative.main([*argv, "--overwrite"]) == 0
    assert loads == [1, 1]


def test_batch_reruns_only_when_clip_overseg_changes(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    output_root = app.output_dir
    app.output_dir = output_root / "masks_sam3" / app.prompt.replace(" ", "_")
    app.version = "sam3"
    app.cache = {i: {} for i in range(3)}
    qualitative.write_interactive_outputs(app)
    metadata_path = app.output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("clip_overseg")
    metadata_path.write_text(json.dumps(metadata))
    probe = qualitative.probe_video
    monkeypatch.setattr(
        qualitative,
        "probe_video",
        lambda p: app.video_info if p == app.video_path else probe(p),
    )
    monkeypatch.setattr(qualitative, "discover_rgb_videos", lambda *_: [app.video_path])
    loads = []

    class Predictor:
        def shutdown(self):
            pass

    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *_: loads.append(1) or Predictor()
    )
    monkeypatch.setattr(qualitative, "process_interactive_video", lambda *_: True)
    argv = [
        "--input-root",
        str(app.video_path.parent),
        "--output-root",
        str(output_root),
        "--text-prompt",
        app.prompt,
        "--version",
        "sam3",
    ]
    assert qualitative.main(argv) == 0
    assert loads == []
    assert qualitative.main([*argv, "--clip-overseg"]) == 0
    assert loads == [1]
    metadata["clip_overseg"] = True
    metadata_path.write_text(json.dumps(metadata))
    assert qualitative.main([*argv, "--clip-overseg"]) == 0
    assert loads == [1]
    metadata["prompt"] = "different prompt"
    metadata_path.write_text(json.dumps(metadata))
    assert qualitative.main(argv) == 1
    assert loads == [1]


@pytest.mark.parametrize(
    "argv",
    [
        ["--input-root", "in", "--output-dir", "out"],
        ["--video", "in.mp4", "--output-root", "out"],
        ["--input-root", "in", "--output-root", "out", "--rgb-name", "../color.mp4"],
        ["--input-root", "in", "--output-root", "out", "--max-sequences", "2"],
    ],
)
def test_batch_rejects_invalid_cli(argv):
    with pytest.raises(SystemExit):
        qualitative.main(argv)


def test_batch_rejects_same_roots_and_empty_input(tmp_path):
    for output in (tmp_path, tmp_path / "out"):
        with pytest.raises(SystemExit):
            qualitative.main(
                ["--input-root", str(tmp_path), "--output-root", str(output)]
            )


def test_closed_editor_aborts_without_finalizing(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    for name in (
        "namedWindow",
        "setWindowTitle",
        "setMouseCallback",
        "destroyWindow",
        "imshow",
    ):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *_: 255)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 0)
    monkeypatch.setattr(
        app, "finalize", lambda: pytest.fail("must not finalize on window close")
    )
    with pytest.raises(KeyboardInterrupt):
        app.run()
    assert not app.window_open


def test_interrupted_overwrite_does_not_leave_success_status(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    app.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = app.output_dir / "metadata.json"
    metadata.write_text('{"status": "success"}')
    args = qualitative.build_parser().parse_args(
        [
            "--video",
            str(app.video_path),
            "--output-dir",
            str(app.output_dir),
            "--overwrite",
        ]
    )
    monkeypatch.setattr(qualitative, "probe_video", lambda *_: app.video_info)
    monkeypatch.setattr(qualitative, "extract_frames", lambda *_: 3)

    def interrupt(*_, **__):
        raise KeyboardInterrupt

    monkeypatch.setattr(qualitative, "run_interactive", interrupt)
    with pytest.raises(KeyboardInterrupt):
        qualitative.process_interactive_video(
            None, args, app.video_path, app.output_dir
        )
    assert json.loads(metadata.read_text())["status"] == "processing"


@pytest.mark.parametrize("error", [RuntimeError, qualitative.SkipVideo])
def test_each_video_closes_its_own_session_even_on_failure(
    tmp_path, monkeypatch, error
):
    events = []

    class Predictor:
        def handle_request(self, request):
            events.append(request["type"])
            return {"session_id": str(len(events)), "outputs": {}}

    class App:
        def __init__(self, predictor, version, session_id, *args, **kwargs):
            self.session_id = session_id

        def start(self, outputs):
            pass

        def run(self):
            raise error("failed interaction")

    monkeypatch.setattr(qualitative, "InteractiveApp", App)
    predictor = Predictor()
    for _ in range(2):
        with pytest.raises(error, match="failed interaction"):
            qualitative.run_interactive(
                predictor,
                "sam3",
                tmp_path / "color.mp4",
                qualitative.VideoInfo(10, 8, 3, 12),
                tmp_path,
                3,
                "hand",
                tmp_path / "out",
                1280,
                20,
            )
    assert events == ["start_session", "add_prompt", "close_session"] * 2


@pytest.mark.parametrize("subdir", [None, "custom_masks"])
@pytest.mark.parametrize(
    "prompt", ["left hand", "right hand without arm", "left  hand"]
)
def test_batch_output_subdir_removes_only_trailing_raw(
    tmp_path, monkeypatch, subdir, prompt
):
    root, output = tmp_path / "input", tmp_path / "output"
    paths = ["a/raw/rgb.mkv", "b/rgb.mkv", "raw/c/rgb.mkv"]
    for name in paths:
        video = root / name
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()

    class Predictor:
        def shutdown(self):
            pass

    monkeypatch.setattr(qualitative, "load_predictor", lambda *_: Predictor())
    destinations = []
    monkeypatch.setattr(
        qualitative,
        "process_interactive_video",
        lambda model, args, video, dest: destinations.append(dest) or True,
    )
    argv = [
        "--input-root",
        str(root),
        "--output-root",
        str(output),
        "--rgb-name",
        "rgb.mkv",
    ]
    argv += ["--text-prompt", prompt]
    if subdir:
        argv += ["--output-subdir", subdir]
    assert qualitative.main(argv) == 0
    assert destinations == [
        output / name / (subdir or "masks_sam3") / prompt.replace(" ", "_")
        for name in ["a", "b", "raw/c"]
    ]


@pytest.mark.parametrize("subdir", ["", ".", "../out", "/tmp/out", "masks/../../out"])
def test_output_subdir_rejects_unsafe_paths(subdir):
    with pytest.raises(SystemExit):
        qualitative.build_parser().parse_args(
            ["--input-root", "in", "--output-root", "out", "--output-subdir", subdir]
        )


def test_batch_rejects_output_collisions(tmp_path):
    root = tmp_path / "input"
    (root / "raw").mkdir(parents=True)
    (root / "color.mp4").touch()
    (root / "raw" / "color.mp4").touch()
    with pytest.raises(SystemExit):
        qualitative.main(
            ["--input-root", str(root), "--output-root", str(tmp_path / "out")]
        )


@pytest.mark.parametrize(
    "prompt", ["", ".", "..", "../hand", "left/hand", "left\\hand"]
)
def test_batch_rejects_prompt_directory_traversal(tmp_path, prompt):
    root = tmp_path / "input"
    root.mkdir()
    (root / "color.mp4").touch()
    with pytest.raises(SystemExit):
        qualitative.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(tmp_path / "out"),
                "--text-prompt",
                prompt,
            ]
        )


def make_loaded_app(tmp_path, get_predictor=None, version="sam3"):
    return qualitative.LoadedInteractiveApp(
        predictor=None,
        version=version,
        session_id="loaded",
        frame_dir=tmp_path / "frames",
        video_path=tmp_path / "input.mp4",
        video_info=qualitative.VideoInfo(10, 8, 100, 12.0),
        frame_count=100,
        prompt="hand",
        output_dir=tmp_path / "output",
        window_width=640,
        get_predictor=get_predictor or (lambda: pytest.fail("unexpected model load")),
    )


def test_load_existing_parser_conflicts_with_overwrite():
    argv = ["--video", "in.mp4", "--output-dir", "out", "--load-existing"]
    assert qualitative.build_parser().parse_args(argv).load_existing
    with pytest.raises(SystemExit):
        qualitative.build_parser().parse_args([*argv, "--overwrite"])


def test_load_masks_restores_curve_and_read_only_exit(tmp_path, monkeypatch):
    original = make_app(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:4, 1:4] = True
    original.cache = {0: {7: mask}, 1: {7: mask}, 2: {7: ~mask}}
    original.confirmed_points = [qualitative.PointEdit(1, 1, 7, 2, 2, 1)]
    qualitative.write_interactive_outputs(original)
    before = {path.name: path.read_bytes() for path in original.output_dir.iterdir()}
    app = make_loaded_app(tmp_path, version=original.version)
    app.frame_count = 3
    app.video_info = original.video_info
    qualitative.load_interactive_result(app)
    assert app.playing is False
    assert app.runner.is_alive is False
    assert app.cached_mutation_series() == [(1, 0.0), (2, 1.0)]
    assert app.loaded_anchor_frames == {1}
    assert app.confirmed_points[0].obj_id == 0
    assert app.next_obj_id == 1
    app.render_controls()  # The existing curve renderer receives the loaded cache.
    app.finalize()
    assert before == {
        path.name: path.read_bytes() for path in original.output_dir.iterdir()
    }


def test_loaded_edit_rollback_and_save_preserves_label_ids(tmp_path, monkeypatch):
    original = make_app(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[:, :4] = True
    original.cache = {index: {4: mask, 8: ~mask} for index in range(3)}
    original.confirmed_points = [qualitative.PointEdit(1, 0, 8, 7, 3, 1)]
    original.events = [{"sequence": 2, "type": "select_object", "obj_id": 8}]
    qualitative.write_interactive_outputs(original)
    app = make_loaded_app(tmp_path, version=original.version)
    app.frame_count = 3
    app.video_info = original.video_info
    qualitative.load_interactive_result(app)
    # A deleted ID must not cause renumbering of the untouched surviving object.
    app.active_obj = 0
    app.remove_selected_object()
    app.dirty = True
    app.finalize()
    metadata = json.loads((app.output_dir / "metadata.json").read_text())
    assert metadata["object_id_to_label"]["1"] == 2
    interactions = json.loads((app.output_dir / "interactions.json").read_text())
    assert interactions["confirmed_points"][0]["original_obj_id"] == 8
    assert interactions["events"][0]["original_obj_id"] == 8
    cap = cv2.VideoCapture(str(app.output_dir / "masks.mkv"))
    try:
        ok, image = cap.read()
        assert ok
        assert np.all(image[:, 4:] == 2)
    finally:
        cap.release()


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("mode", ["forward", "forward_backward", "backward_forward"])
def test_loaded_propagation_hard_bounds_and_anchors(
    tmp_path, monkeypatch, mode, serialized
):
    app = make_loaded_app(tmp_path, lambda: None)
    mask = np.ones((8, 10), dtype=bool)
    app.cache = {frame: {0: mask.copy()} for frame in range(100)}
    before = {frame: app.cache[frame][0].copy() for frame in range(100)}
    app.loaded_anchor_frames = {40, 50, 60}
    app.confirmed_points = [
        qualitative.PointEdit(1, 40, 0, 3, 3, 1),
        qualitative.PointEdit(2, 60, 0, 3, 3, 1),
    ]
    app.prop_range_left, app.prop_range_right = (30, 70)
    calls = []
    seeds = []
    backend = app.predictor

    def initialize():
        seeds.append((backend.seed_frame, backend.seed_masks[0].copy()))
        backend.state = {"obj_ids": [0]}

    def track(frame, reverse):
        calls.append((frame, reverse))
        return {
            "out_obj_ids": np.array([0]),
            "out_binary_masks": np.zeros((1, 8, 10), dtype=bool),
        }

    monkeypatch.setattr(backend, "initialize", initialize)
    monkeypatch.setattr(backend, "track_frame", track)
    original_request = backend.handle_request

    def request(data):
        assert data["type"] != "add_prompt", "loaded anchors must not be reprompted"
        return original_request(data)

    monkeypatch.setattr(backend, "handle_request", request)
    if serialized:
        app.predictor = qualitative.SerializedPredictor(backend, threading.RLock(), 0)
        app.runner.predictor = app.predictor
    app.start_propagation(50, mode)
    app.runner.join()
    app.drain_events()
    assert app.fatal_error is None
    forward = list(range(51, 60)) + list(range(61, 71))
    backward = list(range(49, 40, -1)) + list(range(39, 29, -1))
    expected = {
        "forward": [(frame, False) for frame in forward],
        "forward_backward": [(frame, False) for frame in forward]
        + [(frame, True) for frame in backward],
        "backward_forward": [(frame, True) for frame in backward]
        + [(frame, False) for frame in forward],
    }
    assert calls == expected[mode]
    for frame in [*range(30), *range(71, 100), 40, 50, 60]:
        assert np.array_equal(app.cache[frame][0], before[frame])
    assert all(np.array_equal(seed, before[frame]) for frame, seed in seeds)
    assert all(30 <= frame <= 70 for frame, _ in seeds)
    assert app.confirmed_points[0].frame_index == 40
    assert not app.loaded_anchor_frames & app.stale_frames


def test_loaded_default_range_both_directions_and_empty_leg(tmp_path, monkeypatch):
    app = make_loaded_app(tmp_path)
    app.cache = {frame: {0: np.ones((8, 10), dtype=bool)} for frame in range(100)}
    backend = app.predictor
    calls = []
    monkeypatch.setattr(
        backend, "initialize", lambda: setattr(backend, "state", {"obj_ids": [0]})
    )
    monkeypatch.setattr(
        backend, "track_frame", lambda frame, reverse: calls.append(frame) or {}
    )
    for direction in ("forward", "backward"):
        list(
            backend.handle_stream_request(
                {
                    "start_frame_index": 50,
                    "max_frame_num_to_track": app.propagation_track_limit(
                        50, direction
                    ),
                    "propagation_direction": direction,
                }
            )
        )
    assert calls == [*range(51, 100), *range(49, -1, -1)]
    calls.clear()
    app.prop_range_left, app.prop_range_right = 50, 70
    list(
        backend.handle_stream_request(
            {
                "start_frame_index": 50,
                "max_frame_num_to_track": 0,
                "propagation_direction": "backward",
            }
        )
    )
    assert calls == []
    app.start_propagation(49, "forward")
    assert not app.runner.is_alive


@pytest.mark.parametrize("batch", [False, True])
def test_load_existing_routes_before_model_loading(tmp_path, monkeypatch, batch):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    video = input_dir / "color.mp4"
    video.touch()
    root = tmp_path / "out"
    destination = root / "masks_sam3" / "hand" if batch else root
    destination.mkdir(parents=True)
    (destination / "metadata.json").write_text("{}")
    visited = []
    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *args: pytest.fail("model loaded")
    )
    monkeypatch.setattr(
        qualitative,
        "process_loaded_interactive_video",
        lambda get_predictor, args, path, out: visited.append((path, out)) or True,
    )
    inputs = (
        ["--input-root", str(input_dir), "--output-root", str(root)]
        if batch
        else ["--video", str(video), "--output-dir", str(root)]
    )
    assert qualitative.main([*inputs, "--text-prompt", "hand", "--load-existing"]) == 0
    assert visited == [(video.resolve(), destination.resolve())]


def test_load_existing_rejects_invalid_without_writes(tmp_path, monkeypatch):
    app = make_loaded_app(tmp_path)
    app.output_dir.mkdir()
    saved = app.output_dir / "metadata.json"
    saved.write_text('{"status": "success"}')
    monkeypatch.setattr(
        qualitative, "extract_frames", lambda *args: pytest.fail("extracted frames")
    )
    args = qualitative.build_parser().parse_args(
        [
            "--video",
            str(app.video_path),
            "--output-dir",
            str(app.output_dir),
            "--load-existing",
        ]
    )
    with pytest.raises(RuntimeError, match="files left unchanged"):
        qualitative.process_loaded_interactive_video(
            lambda: pytest.fail("model loaded"), args, app.video_path, app.output_dir
        )
    assert saved.read_text() == '{"status": "success"}'


@pytest.mark.parametrize("version", ["sam3", "sam3.1"])
def test_loaded_backend_initializes_only_edit_frame_and_registers_new_id(
    tmp_path, version
):
    from types import SimpleNamespace

    from sam3.model.data_misc import NestedTensor

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    cv2.imwrite(str(frame_dir / "00050.jpg"), np.zeros((8, 10, 3), dtype=np.uint8))
    calls = []

    class Backbone:
        def forward_image(self, image, **kwargs):
            tensor = image if version == "sam3" else image.tensors
            calls.append(("image", tensor.dtype, tuple(tensor.shape)))
            fpn = [tensor.clone(), tensor.clone(), tensor.clone()]
            if version == "sam3":
                return {"sam2_backbone_out": {"backbone_fpn": fpn}}
            assert kwargs == {
                "need_sam3_out": False,
                "need_interactive_out": True,
                "need_propagation_out": True,
            }
            return {
                key: {"backbone_fpn": [NestedTensor(x.clone(), None) for x in fpn]}
                for key in ("interactive", "sam2_backbone_out")
            }

    class Tracker:
        sam_mask_decoder = SimpleNamespace(conv_s0=lambda x: x, conv_s1=lambda x: x)
        interactive_sam_mask_decoder = sam_mask_decoder

        def init_state(self, **kwargs):
            assert set(kwargs["cached_features"]) == {50}
            return {"obj_ids": [], "masks": {}, "locked": False}

        def add_new_mask(self, state, frame, obj, mask):
            assert not state["locked"]
            state["obj_ids"].append(obj)
            state["masks"][obj] = mask
            calls.append(("mask", frame, obj))

        def add_new_masks(self, state, frame, ids, masks):
            for obj, mask in zip(ids, masks):
                self.add_new_mask(state, frame, obj, mask)

        def propagate_in_video_preflight(self, state, **kwargs):
            state["locked"] = True
            calls.append(("memory", 50))

        def propagate_in_video(
            self, state, start_frame_idx, max_frame_num_to_track, **kwargs
        ):
            assert start_frame_idx == 50
            assert max_frame_num_to_track == 0
            masks = torch.stack([state["masks"][obj] for obj in state["obj_ids"]])[
                :, None
            ]
            calls.append(("track", start_frame_idx))
            yield (
                start_frame_idx,
                state["obj_ids"],
                None,
                masks.float() * 2 - 1,
                torch.ones(len(masks)),
            )

        def add_new_points_or_box(self, state, frame, obj, **kwargs):
            assert obj in state["obj_ids"]  # Must register before tracking locks IDs.
            if version == "sam3.1":
                assert state["user_refined_frames_per_obj"][obj] == {50}
            assert kwargs["rel_coordinates"]
            masks = torch.stack([state["masks"][obj] for obj in state["obj_ids"]])[
                :, None
            ]
            calls.append(("points", frame, obj))
            return frame, state["obj_ids"], None, masks.float() * 2 - 1

    model = SimpleNamespace(
        image_size=16,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        device=torch.device("cpu"),
        tracker=Tracker(),
        detector=SimpleNamespace(backbone=Backbone()),
    )
    app = make_loaded_app(tmp_path, lambda: SimpleNamespace(model=model), version)
    app.cache[50] = {0: np.ones((8, 10), dtype=bool)}
    backend = app.predictor
    backend.reset(50, app.cache[50])
    response = backend.handle_request(
        {
            "type": "add_prompt",
            "frame_index": 50,
            "obj_id": 1,
            "points": [[0.5, 0.5]],
            "point_labels": [1],
        }
    )
    assert response["outputs"]["out_obj_ids"].tolist() == [0, 1]
    assert calls.count(("image", torch.float32, (1, 3, 16, 16))) == 1
    assert ("mask", 50, 1) in calls
    assert ("points", 50, 1) in calls
    assert set(backend.features) == {50}


def test_loaded_preview_cancel_never_rebuilds_prefix(tmp_path, monkeypatch):
    app = make_loaded_app(tmp_path)
    mask = np.ones((8, 10), dtype=bool)
    app.cache = {frame: {0: mask.copy()} for frame in range(100)}
    original = app.cache[50][0].copy()
    app.edit_frame, app.editing = 50, True
    app.draft_points = [qualitative.PointEdit(1, 50, 0, 4, 3, 1)]
    resets = []
    monkeypatch.setattr(
        app.predictor, "reset", lambda frame, masks: resets.append(frame)
    )
    monkeypatch.setattr(
        app, "apply_points", lambda keys: app.store_masks(50, {0: ~mask}, {})
    )
    app.preview()
    assert not np.array_equal(app.cache[50][0], original)
    app.cancel_edit()
    assert resets == [50, 50]
    assert np.array_equal(app.cache[50][0], original)
    assert not app.dirty


def test_loaded_result_rejects_unknown_mask_label_without_writes(tmp_path):
    original = make_app(tmp_path)
    original.cache = {frame: {0: np.ones((8, 10), dtype=bool)} for frame in range(3)}
    qualitative.write_interactive_outputs(original)
    path = original.output_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["object_id_to_label"] = {"0": 2}
    path.write_text(json.dumps(metadata))
    before = {p.name: p.read_bytes() for p in original.output_dir.iterdir()}
    app = make_loaded_app(tmp_path)
    app.frame_count = 3
    app.video_info = original.video_info
    with pytest.raises(ValueError, match="unknown instance labels"):
        qualitative.load_interactive_result(app)
    assert not app.cache
    assert before == {p.name: p.read_bytes() for p in original.output_dir.iterdir()}


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("source_inside_raw", [False, True])
@pytest.mark.parametrize("target_exists", [False, True])
def test_load_existing_finds_input_side_results(
    tmp_path, monkeypatch, batch, target_exists, source_inside_raw
):
    root = tmp_path / "data"
    raw = root / "session" / "raw"
    raw.mkdir(parents=True)
    video = raw / "rgb.mkv"
    video.touch()
    source_parent = raw if source_inside_raw else root / "session"
    source = source_parent / "masks_sam3" / "green_object"
    source.mkdir(parents=True)
    (source / "metadata.json").write_text("{}")
    output_root = tmp_path / "reviewed"
    target = (
        output_root / "session" / "masks_sam3" / "green_object"
        if batch
        else output_root
    )
    if target_exists:
        target.mkdir(parents=True)
        (target / "metadata.json").write_text("{}")
    visited = []
    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *a: pytest.fail("unexpected inference")
    )
    monkeypatch.setattr(
        qualitative,
        "process_loaded_interactive_video",
        lambda factory, args, path, out, **kw: visited.append(
            (path, out, kw.get("result_dir", out))
        )
        or True,
    )
    argv = (
        [
            "--input-root",
            str(root),
            "--output-root",
            str(output_root),
            "--rgb-name",
            "rgb.mkv",
        ]
        if batch
        else ["--video", str(video), "--output-dir", str(output_root)]
    )
    assert (
        qualitative.main([*argv, "--text-prompt", "green object", "--load-existing"])
        == 0
    )
    assert visited == [(video, target, target if target_exists else source)]


@pytest.mark.parametrize("action", ["unchanged", "edited", "interrupted", "skipped"])
def test_review_input_results_publishes_only_on_completion(
    tmp_path, monkeypatch, action
):
    import shutil

    original = make_app(tmp_path)
    mask = np.zeros((8, 10), dtype=bool)
    mask[:, :4] = True
    original.cache = {frame: {7: mask} for frame in range(3)}
    qualitative.write_interactive_outputs(original)
    shutil.copy2(original.output_dir / "result.mp4", original.video_path)
    before = {p.name: p.read_bytes() for p in original.output_dir.iterdir()}
    target = tmp_path / "reviewed"
    args = qualitative.build_parser().parse_args(
        [
            "--video",
            str(original.video_path),
            "--output-dir",
            str(target),
            "--version",
            original.version,
            "--text-prompt",
            original.prompt,
            "--load-existing",
        ]
    )

    def review(app):
        assert app.loaded_result_dir == original.output_dir
        assert app.output_dir == target
        assert len(app.cache) == 3
        assert not target.exists()
        if action == "interrupted":
            raise KeyboardInterrupt("window closed")
        if action == "skipped":
            app.store_masks(1, {0: ~mask}, {})
            app.dirty = True
            app.handle_key(ord("s"))
        if action == "edited":
            app.store_masks(1, {0: ~mask}, {})
            app.dirty = True
        app.finalize()

    monkeypatch.setattr(qualitative.LoadedInteractiveApp, "run", review)
    factory = lambda: pytest.fail("unexpected model load")
    if action in ("interrupted", "skipped"):
        error = KeyboardInterrupt if action == "interrupted" else qualitative.SkipVideo
        with pytest.raises(error):
            qualitative.process_loaded_interactive_video(
                factory,
                args,
                original.video_path,
                target,
                result_dir=original.output_dir,
            )
        assert not target.exists()
    else:
        assert qualitative.process_loaded_interactive_video(
            factory, args, original.video_path, target, result_dir=original.output_dir
        )
        assert qualitative.completed_interactive_video(
            target, original.video_path, args
        )
        if action == "unchanged":
            assert before == {p.name: p.read_bytes() for p in target.iterdir()}
        else:
            cap = cv2.VideoCapture(str(target / "masks.mkv"))
            try:
                assert cap.read()[0]
                ok, changed = cap.read()
                assert ok
                assert np.array_equal(changed[:, :, 0], (~mask).astype(np.uint8))
            finally:
                cap.release()
    assert before == {p.name: p.read_bytes() for p in original.output_dir.iterdir()}


@pytest.mark.parametrize("failure", ["copy", "publish"])
def test_review_copy_failure_leaves_no_partial_result(tmp_path, monkeypatch, failure):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    names = ("result.mp4", "masks.mkv", "interactions.json", "metadata.json")
    for name in names:
        (source / name).write_bytes(name.encode())
    copy = qualitative.shutil.copy2
    link = qualitative.os.link

    def fail_copy(src, dest):
        if src.name == "interactions.json":
            raise OSError("copy failed")
        return copy(src, dest)

    def fail_link(src, dest):
        if dest.name == "metadata.json":
            raise OSError("publish failed")
        return link(src, dest)

    monkeypatch.setattr(
        qualitative.shutil, "copy2", fail_copy if failure == "copy" else copy
    )
    monkeypatch.setattr(
        qualitative.os, "link", fail_link if failure == "publish" else link
    )
    with pytest.raises(OSError):
        qualitative.copy_interactive_outputs(source, target)
    assert not list(target.iterdir())
    assert all((source / name).read_bytes() == name.encode() for name in names)


def test_review_copy_protects_existing_destination(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (target / "masks.mkv").write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="output already exists"):
        qualitative.copy_interactive_outputs(source, target)
    assert (target / "masks.mkv").read_bytes() == b"keep"


def test_review_three_file_dataset_result_without_inference(tmp_path, monkeypatch):
    import shutil

    original = make_app(tmp_path)
    mask = np.ones((8, 10), dtype=bool)
    original.cache = {frame: {0: mask.copy()} for frame in range(3)}
    qualitative.write_interactive_outputs(original)
    shutil.copy2(original.output_dir / "result.mp4", original.video_path)
    path = original.output_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.pop("original_object_id_to_object_id")
    metadata["input_relative_path"] = "input.mp4"
    metadata["detector_only"] = False
    path.write_text(json.dumps(metadata))
    (original.output_dir / "interactions.json").unlink()
    before = {p.name: p.read_bytes() for p in original.output_dir.iterdir()}
    target = tmp_path / "reviewed"
    args = qualitative.build_parser().parse_args(
        [
            "--video",
            str(original.video_path),
            "--output-dir",
            str(target),
            "--version",
            original.version,
            "--text-prompt",
            original.prompt,
            "--load-existing",
        ]
    )
    assert qualitative.completed_interactive_video(
        original.output_dir, original.video_path, args
    )

    def review(app):
        assert len(app.cache) == 3
        assert app.confirmed_points == []
        assert app.cached_mutation_series() == [(1, 0.0), (2, 0.0)]
        app.finalize()

    monkeypatch.setattr(qualitative.LoadedInteractiveApp, "run", review)
    assert qualitative.process_loaded_interactive_video(
        lambda: pytest.fail("model initialized"),
        args,
        original.video_path,
        target,
        result_dir=original.output_dir,
    )
    assert before == {p.name: p.read_bytes() for p in original.output_dir.iterdir()}
    assert all((target / name).read_bytes() == value for name, value in before.items())
    interactions = json.loads((target / "interactions.json").read_text())
    assert interactions["status"] == "success"
    assert interactions["confirmed_points"] == interactions["events"] == []
    assert qualitative.completed_interactive_video(target, original.video_path, args)


def test_missing_interactive_log_is_not_treated_as_no_interactions(tmp_path):
    original = make_app(tmp_path)
    original.cache = {frame: {0: np.ones((8, 10), dtype=bool)} for frame in range(3)}
    qualitative.write_interactive_outputs(original)
    (original.output_dir / "interactions.json").unlink()
    metadata = json.loads((original.output_dir / "metadata.json").read_text())
    with pytest.raises(ValueError, match="missing interaction log"):
        qualitative.read_saved_interactions(original.output_dir, metadata)


@pytest.mark.parametrize(
    "directory", ["", ".", "../masks", "/tmp/masks", "masks/../other"]
)
def test_existing_mask_dir_rejects_unsafe_paths(directory):
    with pytest.raises(SystemExit):
        qualitative.build_parser().parse_args(
            [
                "--video",
                "in.mp4",
                "--output-dir",
                "out",
                "--load-existing",
                "--existing-mask-dir",
                directory,
            ]
        )


def test_existing_mask_dir_requires_load_existing():
    with pytest.raises(SystemExit):
        qualitative.main(
            [
                "--video",
                "in.mp4",
                "--output-dir",
                "out",
                "--existing-mask-dir",
                "masks_sam3",
            ]
        )


@pytest.mark.parametrize("batch", [False, True])
def test_explicit_existing_mask_dir_is_relative_to_rgb_not_output(
    tmp_path, monkeypatch, batch
):
    root = tmp_path / "input"
    raw = root / "session" / "raw"
    raw.mkdir(parents=True)
    video = raw / "rgb.mkv"
    video.touch()
    source = raw / "source_masks" / "green_object"
    source.mkdir(parents=True)
    (source / "metadata.json").write_text("{}")
    out_root = tmp_path / "reviewed"
    target = (
        out_root / "session" / "review_masks" / "green_object" if batch else out_root
    )
    target.mkdir(parents=True)
    (target / "metadata.json").write_text("{}")
    visited = []
    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *a: pytest.fail("model initialized")
    )
    monkeypatch.setattr(
        qualitative,
        "process_loaded_interactive_video",
        lambda factory, args, path, out, **kw: visited.append(
            (path, out, kw.get("result_dir", out))
        )
        or True,
    )
    argv = (
        [
            "--input-root",
            str(root),
            "--output-root",
            str(out_root),
            "--rgb-name",
            "rgb.mkv",
        ]
        if batch
        else ["--video", str(video), "--output-dir", str(out_root)]
    )
    assert (
        qualitative.main(
            [
                *argv,
                "--text-prompt",
                "green object",
                "--output-subdir",
                "review_masks",
                "--load-existing",
                "--existing-mask-dir",
                "source_masks",
            ]
        )
        == 0
    )
    assert visited == [(video, target, source)]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_explicit_missing_or_partial_masks_never_start_inference(
    tmp_path, monkeypatch, capsys, batch, partial
):
    root = tmp_path / "input"
    raw = root / "session" / "raw"
    raw.mkdir(parents=True)
    video = raw / "rgb.mkv"
    video.touch()
    source = raw / "source_masks" / "green_object"
    if partial:
        source.mkdir(parents=True)
        (source / "metadata.json").write_text('{"status":"processing"}')
    target = tmp_path / "out"
    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *a: pytest.fail("model initialized")
    )
    monkeypatch.setattr(
        qualitative,
        "process_interactive_video",
        lambda *a: pytest.fail("fresh inference"),
    )
    monkeypatch.setattr(
        qualitative, "extract_frames", lambda *a: pytest.fail("frames extracted")
    )
    argv = (
        [
            "--input-root",
            str(root),
            "--output-root",
            str(target),
            "--rgb-name",
            "rgb.mkv",
        ]
        if batch
        else ["--video", str(video), "--output-dir", str(target)]
    )
    assert (
        qualitative.main(
            [
                *argv,
                "--text-prompt",
                "green object",
                "--load-existing",
                "--existing-mask-dir",
                "source_masks",
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert str(source) in output.err
    assert "starting new segmentation" not in output.out
    assert not target.exists()
    if partial:
        assert (source / "metadata.json").read_text() == '{"status":"processing"}'


def test_explicit_mask_dir_rejects_single_video_prompt_traversal(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    with pytest.raises(SystemExit):
        qualitative.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(tmp_path / "out"),
                "--text-prompt",
                "../green",
                "--load-existing",
                "--existing-mask-dir",
                "masks_sam3",
            ]
        )


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("explicit_masks", [False, True])
def test_load_existing_batch_skips_completed_output_but_single_reopens(
    tmp_path, monkeypatch, capsys, batch, explicit_masks
):
    import shutil

    root = tmp_path / "input"
    raw = root / "raw"
    raw.mkdir(parents=True)
    video = raw / "rgb.mkv"
    out_root = tmp_path / "reviewed"
    target = out_root / "masks_sam3" / "hand"
    original = make_app(tmp_path)
    original.video_path = video
    original.output_dir = target
    original.cache = {frame: {0: np.ones((8, 10), dtype=bool)} for frame in range(3)}
    qualitative.write_interactive_outputs(original)
    shutil.copy2(target / "result.mp4", video)
    source = raw / "masks_sam3" / "hand"
    if not batch:
        shutil.copytree(target, source)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    visited = []
    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *a: pytest.fail("model loaded")
    )
    monkeypatch.setattr(
        qualitative,
        "process_interactive_video",
        lambda *a: pytest.fail("fresh inference"),
    )
    monkeypatch.setattr(
        qualitative,
        "process_loaded_interactive_video",
        lambda factory, args, path, out, **kw: visited.append(kw.get("result_dir", out))
        or True,
    )
    argv = (
        [
            "--input-root",
            str(root),
            "--output-root",
            str(out_root),
            "--rgb-name",
            "rgb.mkv",
        ]
        if batch
        else ["--video", str(video), "--output-dir", str(target)]
    )
    if explicit_masks:
        argv += ["--existing-mask-dir", "masks_sam3"]
    assert (
        qualitative.main(
            [
                *argv,
                "--load-existing",
                "--version",
                original.version,
                "--text-prompt",
                "hand",
            ]
        )
        == 0
    )
    assert visited == ([] if batch else [source if explicit_masks else target])
    log = capsys.readouterr().out
    assert ("Skipping completed video" in log) == batch
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}


@pytest.mark.parametrize("status", ["processing", "unconfirmed"])
def test_load_existing_batch_does_not_skip_incomplete_output(
    tmp_path, monkeypatch, capsys, status
):
    root = tmp_path / "input"
    root.mkdir()
    video = root / "rgb.mkv"
    video.touch()
    out_root = tmp_path / "reviewed"
    target = out_root / "masks_sam3" / "hand"
    target.mkdir(parents=True)
    metadata = json.dumps({"status": status})
    (target / "metadata.json").write_text(metadata)
    monkeypatch.setattr(
        qualitative, "load_predictor", lambda *a: pytest.fail("model loaded")
    )
    assert (
        qualitative.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(out_root),
                "--rgb-name",
                "rgb.mkv",
                "--load-existing",
                "--text-prompt",
                "hand",
            ]
        )
        == 1
    )
    log = capsys.readouterr()
    assert "Skipping completed video" not in log.out
    assert "incomplete" in log.err
    assert (target / "metadata.json").read_text() == metadata


@pytest.mark.parametrize("key", ["s", "S"])
@pytest.mark.parametrize("playing", [False, True])
def test_skip_editor_without_finalizing(tmp_path, monkeypatch, key, playing):
    app = make_app(tmp_path)
    app.playing = playing
    for name in (
        "namedWindow",
        "setWindowTitle",
        "setMouseCallback",
        "destroyWindow",
        "imshow",
    ):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *_: ord(key))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 1)
    monkeypatch.setattr(
        app, "finalize", lambda: pytest.fail("must not finalize on skip")
    )
    with pytest.raises(qualitative.SkipVideo):
        app.run()
    assert not app.window_open
    assert not app.output_dir.exists()


@pytest.mark.parametrize("chunk_frames", [0, 100])
def test_skip_video_does_not_publish_results(tmp_path, monkeypatch, chunk_frames):
    app = make_app(tmp_path)
    args = qualitative.build_parser().parse_args(
        [
            "--video",
            str(app.video_path),
            "--output-dir",
            str(app.output_dir),
            "--chunk-frames",
            str(chunk_frames),
        ]
    )
    monkeypatch.setattr(qualitative, "probe_video", lambda *_: app.video_info)
    monkeypatch.setattr(qualitative, "extract_frames", lambda *_: 203)
    monkeypatch.setattr(qualitative, "make_chunk_frame_dir", lambda *_: None)
    calls = []

    def skip(*args, **kwargs):
        calls.append(args)
        raise qualitative.SkipVideo

    monkeypatch.setattr(qualitative, "run_interactive", skip)
    with pytest.raises(qualitative.SkipVideo):
        qualitative.process_interactive_video(
            None, args, app.video_path, app.output_dir
        )
    assert len(calls) == 1
    assert (
        json.loads((app.output_dir / "metadata.json").read_text())["status"]
        == "processing"
    )
    assert not (app.output_dir / "result.mp4").exists()
    assert not (app.output_dir / "masks.mkv").exists()


def test_skip_stops_active_propagation(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    events = []

    class Runner:
        is_alive = True

        def request_stop(self, cancel_model):
            assert cancel_model is False
            events.append("stop")

        def join(self):
            events.append("join")

    app.runner = Runner()
    for name in (
        "namedWindow",
        "setWindowTitle",
        "setMouseCallback",
        "imshow",
    ):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_: events.append("close"))
    monkeypatch.setattr(cv2, "waitKey", lambda *_: ord("s"))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_: 1)
    with pytest.raises(qualitative.SkipVideo):
        app.run()
    assert events == ["stop", "join", "close"]

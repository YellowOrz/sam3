"""Multi-window review contracts without model weights or a native desktop."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from sam3.model.sam3_base_predictor import Sam3BasePredictor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from scripts import qualitative_test_interactive as ui


def test_registered_text_sessions_keep_memory_frames_and_independent_checkpoints():
    predictor = Sam3VideoPredictor.__new__(Sam3VideoPredictor)
    Sam3BasePredictor.__init__(predictor)
    predictor.model = SimpleNamespace(device=torch.device("cpu"))
    frames = object()
    children = [
        {
            "input_batch": frames,
            "constants": {},
            "memory": torch.tensor([i]),
            "previous_stages_out": [True],
        }
        for i in (1, 2)
    ]
    parent = predictor._register_session({"text_prompt_states": children})["session_id"]
    request = {"type": "register_text_prompt_sessions", "session_id": parent}
    ids = predictor.handle_request(request)["session_ids"]
    assert predictor.handle_request(request)["session_ids"] == ids
    assert len(set(ids)) == 2
    assert predictor._get_session(ids[0])["state"] is children[0]
    predictor.save_checkpoint(ids[0], 0)
    predictor.close_session(parent, run_gc_collect=False)
    children[0]["memory"].add_(20)
    assert predictor.restore_checkpoint(ids[0], 0)["is_success"]
    assert children[0]["memory"].item() == 1
    assert children[1]["memory"].item() == 2
    assert children[0]["input_batch"] is frames
    assert predictor._get_session(ids[1])["checkpoints"] == {}


def output(prompt):
    mask = np.zeros((1, 48, 64), dtype=bool)
    mask[:, 8:24, : 32 if prompt == "left hand" else 64] = True
    return {
        "out_obj_ids": np.array([0]),
        "out_probs": np.array([0.99]),
        "out_binary_masks": mask,
    }


class FakeBatchPredictor:
    def __init__(self):
        self.model = SimpleNamespace()
        self.requests = []
        self.texts = []
        self.count = 4
        self.fail = False
        self.checkpoints = {}

    def handle_request(self, request):
        self.requests.append(dict(request))
        kind = request["type"]
        if kind == "start_session":
            self.count = len(list(Path(request["resource_path"]).glob("*.jpg")))
            return {"session_id": "batch"}
        if kind == "add_text_prompts":
            if self.fail:
                raise RuntimeError("CUDA out of memory")
            self.texts = request["texts"]
            return {"outputs": [output(p) for p in self.texts]}
        if kind == "register_text_prompt_sessions":
            return {"session_ids": [f"child-{i}" for i in range(len(self.texts))]}
        if kind == "limit_checkpoint_spans":
            assert request["frame_indexes"]
        return {
            "created": True,
            "updated": 0,
            "frame_index": request.get("frame_index", 0),
            "is_success": True,
        }

    def handle_stream_request(self, request):
        self.requests.append(dict(request))
        assert request["type"] == "propagate_in_video_batch"
        for index in range(self.count):
            yield {"frame_index": index, "outputs": [output(p) for p in self.texts]}

    def shutdown(self):
        pass


def make_apps(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(4):
        assert cv2.imwrite(
            str(frames / f"{index:05d}.jpg"), np.zeros((48, 64, 3), np.uint8)
        )
    lock = threading.RLock()
    backend = FakeBatchPredictor()
    proxy = ui.SerializedPredictor(backend, lock, 0)
    apps = []
    for prompt in ("left hand", "right hand"):
        app = ui.InteractiveApp(
            proxy,
            "sam3",
            prompt,
            frames,
            tmp_path / "color.mp4",
            ui.VideoInfo(64, 48, 4, 12),
            4,
            prompt,
            tmp_path / prompt,
            640,
        )
        for frame in range(4):
            app.store_frame_outputs(frame, output(prompt))
        app.playing = False
        app.render()
        apps.append(app)
    return apps, lock, backend


def finish(editor):
    if editor.worker:
        editor.worker.join(timeout=5)
        assert not editor.worker.is_alive()
        editor.finish_worker()


def test_windows_have_independent_cursors_and_edit_contexts(tmp_path, monkeypatch):
    apps, lock, backend = make_apps(tmp_path)
    left, right = apps
    editor = ui.MultiWindowEditor(apps, lock)
    left.set_display_index(3)
    assert right.display_index == 0
    left.playing = True
    assert right.playing is False
    left.playing = False
    editor.on_mouse(right, cv2.EVENT_MBUTTONDOWN, 100, 100, 0, None)
    editor.on_mouse(right, cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    assert editor.active is right
    assert len(right.draft_points) == 1 and left.draft_points == []
    monkeypatch.setattr(
        right,
        "preview",
        lambda: backend.requests.append({"session_id": right.session_id}),
    )
    editor.on_key(ord("p"))
    finish(editor)
    assert backend.requests[-1]["session_id"] == right.session_id
    editor.on_key(9)
    assert editor.active is left


def test_gpu_job_freezes_only_its_owner_and_blocks_peer_editing(tmp_path, monkeypatch):
    apps, lock, _ = make_apps(tmp_path)
    left, right = apps
    editor = ui.MultiWindowEditor(apps, lock)
    entered, release = threading.Event(), threading.Event()

    def work():
        entered.set()
        assert release.wait(5)

    editor.launch(left, work)
    assert entered.wait(5)
    editor.on_mouse(right, cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
    assert not right.draft_points
    right.set_display_index(2)
    assert right.display_index == 2 and left.display_index == 0
    editor.launch(right, lambda: pytest.fail("concurrent job"))
    assert editor.worker_app is left
    release.set()
    finish(editor)


def test_q_confirms_one_window_without_finalizing_or_confirming_peer(
    tmp_path, monkeypatch
):
    apps, lock, _ = make_apps(tmp_path)
    editor = ui.MultiWindowEditor(apps, lock)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_: None)
    for app in apps:
        monkeypatch.setattr(
            app, "finalize", lambda: pytest.fail("must defer publication")
        )
    editor.on_key(ord("q"))
    finish(editor)
    assert editor.live == [apps[1]]
    assert editor.active is None
    editor.on_key(ord("q"))  # A held Q must not confirm a second prompt.
    assert editor.worker is None
    editor.on_key(9)
    editor.on_key(ord("q"))
    finish(editor)
    assert editor.live == []


@pytest.mark.parametrize("exit_kind", ["skip", "close", "interrupt"])
def test_multi_window_abort_never_finalizes_any_app(tmp_path, monkeypatch, exit_kind):
    apps, lock, _ = make_apps(tmp_path)
    editor = ui.MultiWindowEditor(apps, lock)
    for name in (
        "namedWindow",
        "setWindowTitle",
        "setMouseCallback",
        "destroyWindow",
        "moveWindow",
        "imshow",
    ):
        monkeypatch.setattr(cv2, name, lambda *args: None)
    for app in apps:
        monkeypatch.setattr(app, "finalize", lambda: pytest.fail("must not save"))
    monkeypatch.setattr(
        cv2, "getWindowProperty", lambda *_: 0 if exit_kind == "close" else 1
    )

    def key(_):
        if exit_kind == "interrupt":
            raise KeyboardInterrupt
        return ord("s") if exit_kind == "skip" else 255

    monkeypatch.setattr(cv2, "waitKey", key)
    with pytest.raises(ui.SkipVideo if exit_kind == "skip" else KeyboardInterrupt):
        editor.run()
    assert not any(app.window_open for app in apps)


def test_publication_rolls_back_all_prompts_and_preserves_unrelated_files(
    tmp_path, monkeypatch
):
    stages = {}
    original = {}
    for prompt in ("left", "right"):
        source, destination = tmp_path / f"stage-{prompt}", tmp_path / prompt
        source.mkdir()
        destination.mkdir()
        for name in ui.INTERACTIVE_FILES:
            (source / name).write_bytes(f"new-{prompt}-{name}".encode())
            (destination / name).write_bytes(f"old-{prompt}-{name}".encode())
        (destination / "notes.txt").write_text("keep")
        stages[destination] = source
        original[destination] = {p.name: p.read_bytes() for p in destination.iterdir()}
    replace = Path.replace

    def fail(path, target):
        if (
            path.name == "result.mp4"
            and ".sam3-publish-" in str(path)
            and "right" in str(path)
        ):
            raise OSError("simulated disk error")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        ui.publish_reviewed_outputs(stages)
    for destination in stages:
        assert {p.name: p.read_bytes() for p in destination.iterdir()} == original[
            destination
        ]
    monkeypatch.setattr(Path, "replace", replace)
    ui.publish_reviewed_outputs(stages)
    for destination, source in stages.items():
        assert (destination / "notes.txt").read_text() == "keep"
        assert all(
            (destination / name).read_bytes() == (source / name).read_bytes()
            for name in ui.INTERACTIVE_FILES
        )


def write_rgb(path, count=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    assert writer.isOpened()
    for index in range(count):
        writer.write(np.full((48, 64, 3), index, np.uint8))
    writer.release()


@pytest.fixture
def cli_model(monkeypatch):
    backend = FakeBatchPredictor()
    builds = []
    monkeypatch.setattr(
        ui, "load_predictor", lambda *args: builds.append(True) or backend
    )
    reviews = []

    def review(editor):
        reviews.append(
            [
                (app.prompt, app.frame_offset, app.frame_count, type(app).__name__)
                for app in editor.apps
            ]
        )
        assert all(
            not app.unprocessed_frames() and not app.playing for app in editor.apps
        )
        for app in editor.apps:
            app.cancel_edit()

    monkeypatch.setattr(ui.MultiWindowEditor, "run", review)
    return SimpleNamespace(backend=backend, builds=builds, reviews=reviews)


@pytest.mark.parametrize("chunk", [0, 100])
def test_cli_batches_per_chunk_reuses_model_and_defers_publication(
    tmp_path, monkeypatch, cli_model, chunk
):
    video, out = tmp_path / "input" / "raw" / "color.mp4", tmp_path / "out"
    write_rgb(video, 104 if chunk else 4)

    def terminal(chunks, output_dir, *args, **kwargs):
        assert not (out / "masks_sam3" / "left_hand" / "metadata.json").exists()
        ui.merge_chunk_outputs(
            chunks,
            output_dir,
            *args[:5],
            args[5],
            [{0: 0} for _ in chunks],
            True,
            clip_overseg=False,
        )
        return True

    if chunk:
        monkeypatch.setattr(ui, "review_chunk_outputs", terminal)
    args = [
        "--input-root",
        str(video.parent),
        "--output-root",
        str(out),
        "--text-prompt",
        "left hand",
        "right hand",
        "left hand",
        "--chunk-frames",
        str(chunk),
    ]
    assert ui.main(args) == 0
    assert len(cli_model.builds) == 1
    starts = [r for r in cli_model.backend.requests if r["type"] == "start_session"]
    assert len(starts) == (2 if chunk else 1)
    assert all(
        [entry[0] for entry in round_] == ["left hand", "right hand"]
        for round_ in cli_model.reviews
    )
    assert all(
        json.loads((out / "masks_sam3" / name / "metadata.json").read_text())["status"]
        == "success"
        for name in ("left_hand", "right_hand")
    )
    requests = len(cli_model.backend.requests)
    assert ui.main(args) == 0
    assert len(cli_model.backend.requests) == requests  # all cached: no model or GUI


@pytest.mark.parametrize("invalid", ["sam3.1", "collision", "empty"])
def test_invalid_multi_cli_does_not_load_model_or_write(tmp_path, cli_model, invalid):
    prompts = (
        ["left hand", "left_hand"]
        if invalid == "collision"
        else ["", "right"]
        if invalid == "empty"
        else ["left", "right"]
    )
    extra = ["--version", "sam3.1"] if invalid == "sam3.1" else []
    with pytest.raises(SystemExit):
        ui.main(
            [
                "--video",
                str(tmp_path / "missing.mp4"),
                "--output-dir",
                str(tmp_path / "out"),
                "--text-prompt",
                *prompts,
                *extra,
            ]
        )
    assert not cli_model.builds and not (tmp_path / "out").exists()


def test_skip_after_first_chunk_publishes_nothing(tmp_path, monkeypatch, cli_model):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video, 104)
    rounds = []

    def review(editor):
        rounds.append(True)
        if len(rounds) == 2:
            raise ui.SkipVideo

    monkeypatch.setattr(ui.MultiWindowEditor, "run", review)
    assert (
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left",
                "right",
                "--chunk-frames",
                "100",
            ]
        )
        == 0
    )
    assert len(rounds) == 2
    assert not out.exists()


def test_explicit_missing_prompt_fails_before_model_and_preserves_output(
    tmp_path, cli_model
):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video)
    assert (
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left",
                "right",
                "--load-existing",
                "--existing-mask-dir",
                "masks_sam3",
            ]
        )
        == 1
    )
    assert not cli_model.builds and not out.exists()


def saved_result(directory, video, prompt, count=4, chunk=0, three_files=False):
    directory.mkdir(parents=True, exist_ok=True)
    write_rgb(directory / "result.mp4", count)
    writer = cv2.VideoWriter(
        str(directory / "masks.mkv"),
        cv2.VideoWriter_fourcc(*"FFV1"),
        12,
        (64, 48),
        False,
    )
    assert writer.isOpened()
    mask = np.zeros((48, 64), np.uint8)
    mask[8:24, :32] = 17
    for _ in range(count):
        writer.write(mask)
    writer.release()
    metadata = {
        "status": "success",
        "input_video": str(video),
        "prompt": prompt,
        "model_version": "sam3",
        "chunk_frames": chunk,
        "frames_processed": count,
        "source": {"width": 64, "height": 48, "frame_count": count, "fps": 12},
        "object_id_to_label": {"7": 17},
        "outputs": {
            "instance_masks_video": "masks.mkv",
            "visualization_video": "result.mp4",
        },
    }
    if three_files:
        metadata["input_relative_path"] = "color.mp4"
    else:
        metadata["original_object_id_to_object_id"] = {"7": 7}
        (directory / "interactions.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "text_prompt": prompt,
                    "confirmed_points": [],
                    "events": [],
                }
            )
        )
    (directory / "metadata.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("three_files", [False, True])
def test_all_input_results_load_without_model_and_copy_only_after_confirmation(
    tmp_path, cli_model, three_files
):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video)
    sources = {}
    for prompt, name in (("left hand", "left_hand"), ("right hand", "right_hand")):
        source = tmp_path / "masks_sam3" / name
        saved_result(source, video, prompt, three_files=three_files)
        sources[source] = {p.name: p.read_bytes() for p in source.iterdir()}
    assert (
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left hand",
                "right hand",
                "--load-existing",
            ]
        )
        == 0
    )
    assert cli_model.builds == []
    assert len(cli_model.reviews) == 1
    assert all(item[-1] == "LoadedInteractiveApp" for item in cli_model.reviews[0])
    for source, before in sources.items():
        assert {p.name: p.read_bytes() for p in source.iterdir()} == before
        for name, data in before.items():
            assert (out / source.name / name).read_bytes() == data
        assert (out / source.name / "interactions.json").is_file()


def test_loaded_output_browsing_does_not_rewrite_files(tmp_path, cli_model):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video)
    for name in ("left", "right"):
        saved_result(out / name, video, name)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.rglob("*") if p.is_file()
    }
    assert (
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left",
                "right",
                "--load-existing",
            ]
        )
        == 0
    )
    assert not cli_model.builds
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == data for p, data in before.items()
    )


def test_mixed_full_video_loaded_and_fresh_chunks_review_loaded_once(
    tmp_path, monkeypatch, cli_model
):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video, 104)
    left = tmp_path / "masks_sam3" / "left_hand"
    saved_result(left, video, "left hand", 104, chunk=100)
    before = {p.name: p.read_bytes() for p in left.iterdir()}

    def terminal(chunks, output_dir, *args):
        assert not out.exists()
        ui.merge_chunk_outputs(
            chunks,
            output_dir,
            *args[:5],
            args[5],
            [{0: 0} for _ in chunks],
            True,
            clip_overseg=False,
        )
        return True

    monkeypatch.setattr(ui, "review_chunk_outputs", terminal)
    assert (
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left hand",
                "right hand",
                "--load-existing",
                "--chunk-frames",
                "100",
            ]
        )
        == 0
    )
    assert cli_model.reviews == [
        [
            ("left hand", 0, 104, "LoadedInteractiveApp"),
            ("right hand", 0, 100, "InteractiveApp"),
        ],
        [("right hand", 100, 4, "InteractiveApp")],
    ]
    assert all(
        r["texts"] == ["right hand"]
        for r in cli_model.backend.requests
        if r["type"] == "add_text_prompts"
    )
    assert all((left / name).read_bytes() == data for name, data in before.items())


def test_bad_loaded_prompt_prevents_fresh_inference_and_preserves_all_sources(
    tmp_path, cli_model
):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video)
    source = tmp_path / "masks_sam3" / "left"
    saved_result(source, video, "left")
    (source / "masks.mkv").write_bytes(b"bad masks")
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert (
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left",
                "right",
                "--load-existing",
            ]
        )
        == 1
    )
    assert not cli_model.builds and not out.exists()
    assert all((source / name).read_bytes() == data for name, data in before.items())


def test_batch_partially_completed_prompt_is_not_reopened(tmp_path, cli_model):
    video, out = tmp_path / "input" / "color.mp4", tmp_path / "out"
    write_rgb(video)
    saved_result(out / "masks_sam3" / "left", video, "left")
    before = {p: p.read_bytes() for p in (out / "masks_sam3" / "left").iterdir()}
    assert (
        ui.main(
            [
                "--input-root",
                str(video.parent),
                "--output-root",
                str(out),
                "--text-prompt",
                "left",
                "right",
            ]
        )
        == 0
    )
    assert cli_model.reviews[0][0][0] == "right" and len(cli_model.reviews[0]) == 1
    assert all(p.read_bytes() == value for p, value in before.items())


def test_oom_marks_video_failed_and_continues_without_serial_fallback(
    tmp_path, cli_model
):
    root, out = tmp_path / "input", tmp_path / "out"
    for name in ("a", "b"):
        write_rgb(root / name / "color.mp4")
    original = cli_model.backend.handle_request

    def request(data):
        if data["type"] == "add_text_prompts" and not getattr(
            cli_model.backend, "failed_once", False
        ):
            cli_model.backend.failed_once = True
            raise RuntimeError("CUDA out of memory")
        return original(data)

    cli_model.backend.handle_request = request
    assert (
        ui.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(out),
                "--text-prompt",
                "left",
                "right",
            ]
        )
        == 1
    )
    assert not (out / "a").exists()
    assert (out / "b" / "masks_sam3" / "right" / "metadata.json").is_file()
    assert not any(r["type"] == "add_prompt" for r in cli_model.backend.requests)


def test_output_directory_alias_rejected_before_model(tmp_path, cli_model):
    video, out = tmp_path / "color.mp4", tmp_path / "out"
    write_rgb(video)
    (out / "left").mkdir(parents=True)
    (out / "right").symlink_to(out / "left", target_is_directory=True)
    with pytest.raises(SystemExit):
        ui.main(
            [
                "--video",
                str(video),
                "--output-dir",
                str(out),
                "--text-prompt",
                "left",
                "right",
            ]
        )
    assert not cli_model.builds


def test_publication_preserves_original_backups_if_rollback_also_fails(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "stage", tmp_path / "output"
    source.mkdir()
    destination.mkdir()
    for name in ui.INTERACTIVE_FILES:
        (source / name).write_bytes(b"new")
        (destination / name).write_bytes(b"old")
    replace = Path.replace

    def fail(path, target):
        if path.name == "result.mp4" and ".sam3-publish-" in str(path):
            raise OSError("publication error")
        if path.name.endswith(".backup"):
            raise OSError("rollback error")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(RuntimeError, match="backups retained"):
        ui.publish_reviewed_outputs({destination: source})
    backups = list(destination.glob(".sam3-publish-*/*.backup"))
    assert backups and all(path.read_bytes() == b"old" for path in backups)


def test_input_side_publication_never_overwrites_concurrent_destination_result(
    tmp_path,
):
    source, destination = tmp_path / "stage", tmp_path / "output"
    source.mkdir()
    destination.mkdir()
    for name in ui.INTERACTIVE_FILES:
        (source / name).write_bytes(b"reviewed")
    (destination / "result.mp4").write_bytes(b"concurrent result")
    with pytest.raises(RuntimeError, match="output already exists"):
        ui.publish_reviewed_outputs({destination: source}, [destination])
    assert (destination / "result.mp4").read_bytes() == b"concurrent result"


def test_parser_defaults_to_sam3_and_preserves_single_prompt_shape():
    base = ["--video", "input.mp4", "--output-dir", "out"]
    parser = ui.build_parser()
    default = parser.parse_args(base)
    assert (default.version, default.text_prompt) == ("sam3", "circle")
    single = parser.parse_args([*base, "--version", "sam3.1", "--text-prompt", "hand"])
    assert (single.version, single.text_prompt) == ("sam3.1", "hand")
    assert parser.parse_args([*base, "--text-prompt", "left", "right"]).text_prompt == [
        "left",
        "right",
    ]


def test_multi_help_documents_confirmation_and_keyboard_routing():
    single = ui.render_key_help_panel(650, False)
    multi = ui.render_key_help_panel(650, False, True)
    assert single.shape == multi.shape
    assert not np.array_equal(single, multi)

"""Public headless boundary; CPU fixtures, no checkpoints or model downloads."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from sam3 import video_backend as backend, video_outputs


class Predictor:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "test"}
        if request["type"] == "close_session":
            return {}
        return {"outputs": self.output()}

    def output(self):
        return {
            "out_obj_ids": np.array([7]),
            "out_binary_masks": np.ones((1, 8, 10), dtype=bool),
            "out_probs": np.array([0.9]),
            "out_tracker_probs": np.array([0.0]),
        }

    def handle_stream_request(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("failed stream")
        start = request["start_frame_index"]
        frames = (
            range(start, 3)
            if request["propagation_direction"] == "forward"
            else range(start - 1, -1, -1)
        )
        for frame in frames:
            yield {"frame_index": frame, "outputs": self.output()}


@pytest.mark.parametrize("render_preview", [False, True])
def test_headless_exports_scores_without_desktop_checkpoints_and_closes(
    tmp_path, monkeypatch, render_preview
):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        Image.new("RGB", (10, 8)).save(frames / f"{i:05d}.jpg")
    if not render_preview:

        def no_preview(*args, **kwargs):
            pytest.fail("mask-only export must not encode previews or reread RGB")

        monkeypatch.setattr(video_outputs, "H264Writer", no_preview)
        monkeypatch.setattr(video_outputs, "load_frame_bgr", no_preview)
    predictor = Predictor()
    seen = []
    backend._segment_clip(
        predictor,
        tmp_path / "video.mkv",
        video_outputs.VideoInfo(10, 8, 3, 30),
        frames,
        tmp_path / "out",
        "hand",
        0,
        None,
        seen.append,
        render_preview=render_preview,
    )
    kinds = [r["type"] for r in predictor.requests]
    assert kinds == [
        "start_session",
        "add_prompt",
        "propagate_in_video",
        "close_session",
    ]
    metadata = json.loads((tmp_path / "out/metadata.json").read_text())
    assert metadata["frames_processed"] == 3
    assert [f["objects"][0]["tracking"] for f in metadata["mask_scores"]["frames"]] == [
        0,
        0,
        0,
    ]
    assert seen[-1] == 3
    assert (tmp_path / "out/masks.mkv").stat().st_size > 0
    assert (tmp_path / "out/result.mp4").exists() is render_preview
    assert ("visualization_video" in metadata["outputs"]) is render_preview
    interactions = json.loads((tmp_path / "out/interactions.json").read_text())
    assert interactions["outputs"] == metadata["outputs"]


def test_headless_closes_on_failure_and_never_publishes_success(tmp_path):
    predictor = Predictor(fail=True)
    with pytest.raises(RuntimeError, match="failed stream"):
        backend._segment_clip(
            predictor,
            tmp_path / "v",
            video_outputs.VideoInfo(10, 8, 3, 30),
            tmp_path,
            tmp_path / "out",
            "hand",
            0,
            None,
            lambda _: None,
        )
    assert predictor.requests[-1]["type"] == "close_session"
    assert not (tmp_path / "out/metadata.json").exists()


@pytest.mark.parametrize("async_frames", [False, True])
def test_native_progress_is_passed_to_synchronous_and_async_image_loaders(
    tmp_path, monkeypatch, async_frames
):
    from sam3.model import io_utils

    calls = []

    def tqdm(iterable=None, **kw):
        calls.append(kw["disable"])
        return iterable

    monkeypatch.setattr(io_utils, "tqdm", tqdm)
    for i in range(2):
        Image.new("RGB", (10, 8)).save(tmp_path / f"{i:06d}.png")
    frames, h, w = io_utils.load_resource_as_video_frames(
        str(tmp_path), 4, True, async_loading_frames=async_frames, progress=False
    )
    if async_frames:
        frames.thread.join()
    assert calls == [True] and (h, w) == (8, 10)
    assert len(frames) == 2


def test_png_editor_uses_public_frame_path_provider(tmp_path, monkeypatch):
    import sam3.mask_editor as editor
    import torch
    from sam3.model.utils import sam2_utils

    paths = []

    def load(path, size):
        paths.append(path)
        return torch.zeros((3, 4, 4)), 4, 4

    monkeypatch.setattr(sam2_utils, "_load_img_as_tensor", load)
    identity = lambda t: t
    model = SimpleNamespace(
        device=torch.device("cpu"),
        image_size=4,
        image_mean=(0.5,) * 3,
        image_std=(0.5,) * 3,
        detector=SimpleNamespace(
            backbone=SimpleNamespace(
                forward_image=lambda image: {
                    "sam2_backbone_out": {"backbone_fpn": [image, image]}
                }
            )
        ),
        tracker=SimpleNamespace(
            sam_mask_decoder=SimpleNamespace(conv_s0=identity, conv_s1=identity)
        ),
    )
    state = SimpleNamespace(version="sam3", confirmed_points=[])
    adapter = backend.create_mask_editor(
        SimpleNamespace(model=model), state, frame_path=lambda f: tmp_path / f"{f}.png"
    )
    adapter.prepare_features(2)
    assert paths == [str(tmp_path / "2.png")]
    assert isinstance(adapter, editor.LoadedMaskPredictor)


def test_backend_does_not_import_desktop_scripts():
    root = Path(backend.__file__).parent
    for name in (
        "video_backend.py",
        "video_outputs.py",
        "video_io.py",
        "mask_editor.py",
    ):
        text = (root / name).read_text()
        assert "from scripts" not in text and "import scripts" not in text
        assert "tqdm.__init__" not in text


@pytest.mark.parametrize(
    "start, directions",
    [(0, ["forward"]), (1, ["forward", "backward"]), (2, ["backward"])],
)
def test_custom_start_covers_all_frames_in_one_session(tmp_path, start, directions):
    frames = tmp_path / "frames"
    frames.mkdir()
    for frame in range(3):
        Image.new("RGB", (10, 8)).save(frames / f"{frame:05d}.jpg")
    predictor = Predictor()
    progress = []
    backend._segment_clip(
        predictor,
        tmp_path / "video",
        video_outputs.VideoInfo(10, 8, 3, 30),
        frames,
        tmp_path / "out",
        "hand",
        0,
        None,
        progress.append,
        start_frame_index=start,
    )
    streams = [r for r in predictor.requests if r["type"] == "propagate_in_video"]
    assert [r["propagation_direction"] for r in streams] == directions
    assert all(
        r["start_frame_index"] == start and r["session_id"] == "test" for r in streams
    )
    assert (
        next(r for r in predictor.requests if r["type"] == "add_prompt")["frame_index"]
        == start
    )
    assert sum(r["type"] == "start_session" for r in predictor.requests) == 1
    assert progress == sorted(progress) and progress[-1] == 3
    metadata = json.loads((tmp_path / "out/metadata.json").read_text())
    assert metadata["propagation"]["start_frame_index"] == start
    assert [f["frame_index"] for f in metadata["mask_scores"]["frames"]] == [0, 1, 2]


def test_single_frame_needs_no_stream(tmp_path):
    Image.new("RGB", (10, 8)).save(tmp_path / "00000.jpg")
    predictor = Predictor()
    backend._segment_clip(
        predictor,
        tmp_path / "video",
        video_outputs.VideoInfo(10, 8, 1, 30),
        tmp_path,
        tmp_path / "out",
        "hand",
        0,
        None,
        lambda _: None,
    )
    assert not any(r["type"] == "propagate_in_video" for r in predictor.requests)
    assert (
        json.loads((tmp_path / "out/metadata.json").read_text())["frames_processed"]
        == 1
    )


@pytest.mark.parametrize(
    "count, index, fraction, expected",
    [
        (101, 0, 0.5, 50),
        (100, 0, 0.5, 49),
        (101, 0, 0.58, 58),
        (101, 0, 0, 0),
        (101, 0, 1, 100),
        (3, 99, None, 2),
        (1, 0, 0.5, 0),
    ],
)
def test_public_backend_resolves_start_from_decoded_count(
    tmp_path, monkeypatch, count, index, fraction, expected
):
    from contextlib import nullcontext

    import torch

    calls = []
    monkeypatch.setattr(torch, "inference_mode", nullcontext)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        video_outputs, "probe_video", lambda _: video_outputs.VideoInfo(10, 8, 999, 30)
    )
    monkeypatch.setattr(video_outputs, "extract_frames", lambda *a: count)
    monkeypatch.setattr(
        backend, "_segment_clip", lambda *a, **kw: calls.append(kw["start_frame_index"])
    )
    result = backend.segment_video(
        object(),
        video=tmp_path / "video",
        prompts=["hand", "cup"],
        output_dirs=[tmp_path / "hand", tmp_path / "cup"],
        start_frame_index=index,
        start_fraction=fraction,
    )
    assert calls == [expected, expected]
    assert result["start_frame_index"] == expected and result["frames"] == count


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_frame_index": True},
        {"start_frame_index": -1},
        {"start_fraction": float("nan")},
        {"start_fraction": float("inf")},
        {"start_fraction": -0.1},
        {"start_fraction": 1.1},
        {"start_fraction": True},
        {"start_frame_index": 2, "start_fraction": 0.5},
        {"chunk_frames": 100, "start_frame_index": 2},
        {"chunk_frames": 100, "start_fraction": 0.5},
    ],
)
def test_invalid_custom_start_is_rejected_before_decoding(tmp_path, kwargs):
    with pytest.raises(ValueError, match="start"):
        backend.segment_video(
            object(),
            video=tmp_path / "video",
            prompts=["hand"],
            output_dirs=[tmp_path / "out"],
            **kwargs,
        )


@pytest.mark.parametrize("chunk_frames", [0, 2])
def test_public_mask_only_export_preserves_full_video_and_scores(
    tmp_path, monkeypatch, chunk_frames
):
    from contextlib import nullcontext

    import torch

    monkeypatch.setattr(torch, "inference_mode", nullcontext)
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    info = video_outputs.VideoInfo(10, 8, 5, 30)
    monkeypatch.setattr(video_outputs, "probe_video", lambda _: info)

    def extract(video, frames):
        frames.mkdir()
        for frame in range(info.frame_count):
            Image.new("RGB", (10, 8)).save(frames / f"{frame:05d}.jpg")
        return info.frame_count

    def no_preview(*args, **kwargs):
        pytest.fail("automatic export must not encode previews or reread RGB")

    monkeypatch.setattr(video_outputs, "extract_frames", extract)
    monkeypatch.setattr(video_outputs, "H264Writer", no_preview)
    monkeypatch.setattr(video_outputs, "load_frame_bgr", no_preview)

    class DecodedPredictor(Predictor):
        def handle_request(self, request):
            if request["type"] == "start_session":
                self.count = len(list(Path(request["resource_path"]).glob("*.jpg")))
            return super().handle_request(request)

        def handle_stream_request(self, request):
            for frame in range(self.count):
                yield {"frame_index": frame, "outputs": self.output()}

    output = tmp_path / "out"
    backend.segment_video(
        DecodedPredictor(),
        video=tmp_path / "rgb.mkv",
        prompts=["hand"],
        output_dirs=[output],
        chunk_frames=chunk_frames,
        render_preview=False,
    )
    assert {p.name for p in output.iterdir()} == {
        "masks.mkv",
        "metadata.json",
        "interactions.json",
    }
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "success" and metadata["frames_processed"] == 5
    assert "visualization_video" not in metadata["outputs"]
    scores = metadata["mask_scores"]["frames"]
    assert [f["frame_index"] for f in scores] == list(range(5))
    assert all(f["objects"][0]["tracking"] == 0 for f in scores)
    capture = cv2.VideoCapture(str(output / "masks.mkv"))
    try:
        for _ in range(5):
            ok, frame = capture.read()
            assert ok and np.all(frame == 1)
        assert not capture.read()[0]
    finally:
        capture.release()


@pytest.mark.parametrize("render_preview", [None, 0, 1, "false"])
def test_invalid_preview_flag_is_rejected_before_decoding(tmp_path, render_preview):
    with pytest.raises(ValueError, match="render_preview"):
        backend.segment_video(
            object(),
            video=tmp_path / "input",
            prompts=["hand"],
            output_dirs=[tmp_path / "out"],
            render_preview=render_preview,
        )

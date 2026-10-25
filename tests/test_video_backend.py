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
        for frame in range(3):
            yield {"frame_index": frame, "outputs": self.output()}


def test_headless_exports_scores_without_desktop_checkpoints_and_closes(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        Image.new("RGB", (10, 8)).save(frames / f"{i:05d}.jpg")
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

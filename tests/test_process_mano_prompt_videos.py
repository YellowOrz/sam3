import json
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image import Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_video_inference import (
    Sam3VideoInference,
    Sam3VideoInferenceWithInstanceInteractivity,
)
from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from scripts import (
    process_dataset_videos as dataset,
    process_mano_prompt_videos as mano,
)


def sample_npz(path, **overrides):
    joints = np.zeros((3, 21, 3))
    joints[:, :, 0] = np.linspace(-0.2, 0.2, 21)
    joints[:, :, 1] = np.linspace(-0.1, 0.1, 21)
    vertices = np.zeros((3, 778, 3))
    vertices[:, :, 0] = np.linspace(-0.4, 0.4, 778)
    vertices[:, :, 1] = np.linspace(-0.3, 0.3, 778)
    data = dict(
        width=100,
        height=80,
        hand="left_hand",
        fps=10.0,
        frame_indices=np.array([2, 0, 1]),
        has_hand=np.array([True, True, False]),
        camera_translation=np.tile([0, 0, 1], (3, 1)),
        joints=joints,
        vertices=vertices,
    )
    data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return path


def arguments(*extra):
    return mano.build_parser().parse_args(
        [
            "--text-prompt",
            "left hand",
            "--hand-side",
            "left",
            "--focal-length",
            "100",
            *extra,
        ]
    )


INFO = dict(width=100, height=80, frame_count=3, fps=10.0)


@pytest.mark.parametrize("mode", ["points", "box", "both"])
@pytest.mark.parametrize("source", ["mesh", "joints"])
def test_modes_projection_alignment_and_missing_frames(tmp_path, mode, source):
    path = sample_npz(tmp_path / "sample.npz")
    prompts, metadata = mano.load_geometry(
        path, INFO, arguments("--prompt-mode", mode, "--box-source", source)
    )
    assert set(prompts) == {0, 2}
    assert metadata["missing_frames"] == [1]
    assert metadata["unusable_prompt_frames"] == []
    expected_keys = (
        {"points", "boxes"}
        if mode == "both"
        else {"points" if mode == "points" else "boxes"}
    )
    assert set(prompts[0]) == expected_keys
    if mode != "box":
        # No second left-hand mirror; normalized projection uses translation once.
        np.testing.assert_allclose(prompts[0]["points"][0], [0.3, 0.375])
        assert len(prompts[0]["points"]) == 21
    if mode != "points":
        expected = (
            [0.06, 0.0875, 0.88, 0.825]
            if source == "mesh"
            else [0.28, 0.3625, 0.44, 0.275]
        )
        np.testing.assert_allclose(prompts[0]["boxes"][0], expected)
    sparse, _ = mano.load_geometry(path, INFO, arguments("--prompt-interval", "3"))
    assert set(sparse) == {0}


def test_invalid_coordinates_clipping_and_schema(tmp_path):
    xy = mano.project_points(
        np.array([[0, 0, 0], [2, 0, 0], [0, 0, -2], [np.nan, 0, 0]]),
        np.array([0, 0, 1]),
        100,
        80,
        100,
    )
    np.testing.assert_allclose(xy, [[50, 40], [250, 40]])
    path = sample_npz(tmp_path / "sample.npz")
    prompts, _ = mano.load_geometry(path, INFO, arguments("--box-padding", "5"))
    np.testing.assert_allclose(prompts[0]["boxes"], [[0, 0, 1, 1]])
    for override, message in [
        ({"width": 101}, "dimensions"),
        ({"hand": "right_hand"}, "hand"),
        ({"frame_indices": np.array([0, 0, 1])}, "unique"),
        ({"joints": np.zeros((3, 20, 3))}, "shape"),
        ({"fps": 30}, "FPS"),
    ]:
        sample_npz(path, **override)
        with pytest.raises(ValueError, match=message):
            mano.load_geometry(path, INFO, arguments())


def test_cli_invalid_options_and_ambiguous_files(tmp_path):
    for extra in [
        ["--version", "sam3.1"],
        ["--prompt-interval", "0"],
        ["--box-padding", "nan"],
        ["--focal-length", "0"],
        ["--mano-name", "../result.npz"],
        ["--mano-dir-name", "../MANO"],
        ["--mano-dir-name", ""],
        ["--mano-dir-name", "*"],
        ["--prompt-mode", "tracker"],
    ]:
        with pytest.raises(SystemExit):
            arguments(*extra)
    video = tmp_path / "color.mp4"
    with pytest.raises(FileNotFoundError):
        mano.find_mano(video, "left", None)
    first = sample_npz(tmp_path / "MANO_wilor/left_hand/result_mano_1.npz")
    assert mano.find_mano(video, "left", None) == first
    sample_npz(first.with_name("result_mano_2.npz"))
    with pytest.raises(ValueError, match="multiple"):
        mano.find_mano(video, "left", None)
    assert mano.find_mano(video, "left", first.name) == first


@pytest.mark.parametrize("text", ["left hand", "", "   ", None])
def test_api_preloads_geometry_once_and_preserves_tracker_dispatch(text):
    host = SimpleNamespace(device=torch.device("cpu"))
    host._prepare_geometry_prompts = MethodType(
        Sam3VideoInference._prepare_geometry_prompts, host
    )
    host.TEXT_ID_FOR_TEXT = 0
    host.TEXT_ID_FOR_VISUAL = 1
    empty = Prompt(box_embeddings=torch.zeros(0, 1, 4))
    state = {
        "num_frames": 3,
        "input_batch": SimpleNamespace(
            find_text_batch=["old"],
            find_inputs=[SimpleNamespace(text_ids=torch.zeros(1)) for _ in range(3)],
        ),
        "constants": {"empty_geometric_prompt": empty},
        "tracker_inference_states": ["old memory"],
        "tracker_metadata": {},
        "feature_cache": {},
        "cached_frame_outputs": {},
        "action_history": [],
    }
    for key in [
        "previous_stages_out",
        "per_frame_raw_point_input",
        "per_frame_raw_box_input",
        "per_frame_visual_prompt",
        "per_frame_geometric_prompt",
        "per_frame_cur_step",
    ]:
        state[key] = [None] * 3
    resets = []

    def reset(current):
        resets.append(True)
        Sam3VideoInference.reset_state(host, current)

    host.reset_state = reset

    def infer(current, frame_idx, reverse):
        assert current["text_prompt"] == (text.strip() or None if text else None)
        expected_id = (
            host.TEXT_ID_FOR_TEXT if text and text.strip() else host.TEXT_ID_FOR_VISUAL
        )
        assert all(
            stage.text_ids.item() == expected_id
            for stage in current["input_batch"].find_inputs
        )
        assert set(current["feature_cache"]) == {"per_frame_geometric_prompts"}
        current["tracker_inference_states"].append("new memory")
        return {"ok": True}

    host._run_single_frame_inference = infer
    host._postprocess_output = lambda current, out: out
    host.add_prompt = lambda **kwargs: Sam3VideoInference.add_prompt(host, **kwargs)
    predictor = SimpleNamespace(
        model=host,
        _get_session=lambda sid: {"state": state},
        _extend_expiration_time=lambda session: None,
    )
    host.detector = SimpleNamespace(backbone=SimpleNamespace())
    response = Sam3VideoPredictor.handle_request(
        predictor,
        {
            "type": "add_geometry_prompts",
            "session_id": "s",
            "frame_index": 0,
            "text": text,
            "geometry_prompts": {
                0: {"points": [[0.2, 0.3]], "boxes": [[0.1, 0.2, 0.4, 0.6]]},
                2: {"points": [[0.8, 0.7]]},
            },
        },
    )
    assert response["outputs"] == {"ok": True}
    assert len(resets) == 1
    assert state["tracker_inference_states"] == ["new memory"]
    schedule = state["feature_cache"]["per_frame_geometric_prompts"]
    assert schedule[1] is empty
    assert all(value is None for value in state["per_frame_visual_prompt"])
    torch.testing.assert_close(
        schedule[0].box_embeddings[:, 0], torch.tensor([[0.3, 0.5, 0.4, 0.6]])
    )
    assert schedule[0].point_labels.item() == 1
    assert schedule[0].box_labels.item() == 1
    for bad in [
        {3: {"points": [[0, 0]]}},
        {0: {"points": [[float("nan"), 0]]}},
        {0: {"boxes": [[0.9, 0, 0.2, 0.3]]}},
    ]:
        with pytest.raises(ValueError):
            host.add_prompt(
                inference_state=state,
                frame_idx=0,
                text_str="left hand",
                geometry_prompts=bad,
            )
    assert len(resets) == 1  # Bad input must not erase a valid session.
    host.use_prev_mem_frame = False
    host.add_tracker_new_points = lambda *a, **kw: kw["obj_id"]
    assert (
        Sam3VideoInferenceWithInstanceInteractivity.add_prompt(
            host,
            state,
            0,
            points=[[0.2, 0.3]],
            point_labels=[1],
            obj_id=7,
        )
        == 7
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_detector_prefetch_uses_destination_frame_geometry(reverse):
    host = SimpleNamespace(rank=0, world_size=1, gather_backbone_out=False)
    host._build_multigpu_buffer_next_chunk = MethodType(
        Sam3ImageOnVideoMultiGPU._build_multigpu_buffer_next_chunk, host
    )
    host._gather_tensor = lambda value: ([value], None)
    seen = []
    schedule = [object() for _ in range(3)]

    def grounding(**kwargs):
        frame = kwargs["find_input"]
        assert kwargs["geometric_prompt"] is schedule[frame]
        seen.append(frame)
        return {
            key: torch.tensor([frame])
            for key in ("pred_logits", "pred_boxes", "pred_boxes_xyxy", "pred_masks")
        }

    host.forward_grounding = grounding
    cache = {}
    order = [2, 1, 0] if reverse else [0, 1, 2]
    for index in order:
        out, _ = Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu(
            host,
            {},
            list(range(3)),
            schedule[index],
            index,
            3,
            cache,
            track_in_reverse=reverse,
            per_frame_geometric_prompts=schedule,
        )
        assert out["pred_masks"].item() == index
    assert seen == order  # Each frame is computed only once, including prefetch.


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 80))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((80, 100, 3), dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize("dir_name", ["MANO_wilor", "MANO_custom"])
def test_list_only_and_missing_file_skip(tmp_path, capsys, dir_name):
    video = tmp_path / "color.mp4"
    make_video(video)
    output = tmp_path / "out"
    cli = [
        "--input-root",
        str(tmp_path),
        "--output-root",
        str(output),
        "--text-prompt",
        "left hand",
        "--hand-side",
        "left",
        "--list-only",
    ]
    assert arguments().mano_dir_name == "MANO_wilor"
    if dir_name != "MANO_wilor":
        cli += ["--mano-dir-name", dir_name]
    assert mano.main(cli) == 0
    sample_npz(tmp_path / dir_name / "left_hand/result_mano_1.npz")
    assert mano.main(cli) == 0
    assert "result_mano_1.npz" in capsys.readouterr().out
    sample_npz(tmp_path / dir_name / "left_hand/result_mano_2.npz")
    assert mano.main(cli) == 1
    assert not output.exists()


@pytest.mark.parametrize("text", ["left hand", "", "   "])
def test_batch_records_missing_npz_and_continues(tmp_path, monkeypatch, text):
    import sam3

    root = tmp_path / "data"
    for name in ("a_missing", "b_valid"):
        directory = root / name
        directory.mkdir(parents=True)
        make_video(directory / "color.mp4")
    sample_npz(root / "b_valid/MANO_wilor/left_hand/result_mano_1.npz")
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(dataset, "require_cuda", lambda *args: True)
    monkeypatch.setattr(
        sam3,
        "build_sam3_predictor",
        lambda **kwargs: SimpleNamespace(shutdown=lambda: None),
    )

    def process(*args, **kwargs):
        calls.append(args[1])
        assert args[4] == text.strip()
        assert set(kwargs["geometry_prompts"]) == {0, 2}
        return "success"

    monkeypatch.setattr(dataset, "process_video", process)
    assert (
        mano.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--text-prompt",
                text,
                "--hand-side",
                "left",
            ]
        )
        == 0
    )
    records = json.loads((output / "batch_summary.json").read_text())["results"]
    assert [record["status"] for record in records] == ["skipped", "success"]
    assert "no MANO NPZ" in records[0]["reason"]
    assert not (output / "a_missing").exists()
    assert calls == [root / "b_valid/color.mp4"]


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("text", ["left hand", ""])
def test_output_videos_overlay_and_configuration_skip(tmp_path, direction, text):
    video = tmp_path / "color.mp4"
    make_video(video)
    output = tmp_path / "out"

    class Predictor:
        requests = []

        def handle_request(self, request):
            self.requests.append(request)
            return {"session_id": "s"}

        def handle_stream_request(self, request):
            indices = range(3) if direction == "forward" else range(2, -1, -1)
            for index in indices:
                yield {"frame_index": index, "outputs": dataset.empty_outputs()}

    predictor = Predictor()
    geometry = {0: {"points": [[0.5, 0.75]], "boxes": [[0.1, 0.4, 0.8, 0.5]]}}
    metadata = {"mano": {"mode": "both"}}

    def run(extra):
        return dataset.process_video(
            predictor,
            video,
            output,
            tmp_path,
            text,
            "sam3",
            None,
            False,
            direction,
            extra_metadata=extra,
            geometry_prompts=geometry,
        )

    assert run(metadata) == "success"
    request = next(r for r in predictor.requests if r["type"] == "add_geometry_prompts")
    assert request["text"] == text
    assert request["geometry_prompts"] == geometry
    assert request["frame_index"] == (0 if direction == "forward" else 2)
    assert dataset.probe_video(output / "masks.mkv")["frame_count"] == 3
    cap = cv2.VideoCapture(str(output / "result.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame[60, 50, 1] > 100  # Prompt dot visible, masks remain empty.
    data = json.loads((output / "metadata.json").read_text())
    assert data["mano"] == metadata["mano"] and data["status"] == "success"
    assert run(metadata) == "skipped"
    assert run({"mano": {"mode": "points"}}) == "success"

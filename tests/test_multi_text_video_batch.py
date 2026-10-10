"""Real batch scheduling and CLI output contracts, without weights or CUDA."""

import json
import sys
from collections import defaultdict
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_video_inference import (
    Sam3VideoInference,
    Sam3VideoInferenceWithInstanceInteractivity,
)
from scripts import process_dataset_videos as dataset


class FakeDetector:
    num_feature_levels = 1
    device = torch.device("cpu")
    _get_img_feats = Sam3Image._get_img_feats
    _get_dummy_prompt = Sam3Image._get_dummy_prompt

    def __init__(self):
        self.image_batches = []
        self.query_batches = []
        self.text_batches = []
        self.backbone = SimpleNamespace(
            forward_image=self.forward_image, forward_text=self.forward_text
        )

    def forward_image(self, images):
        self.image_batches.append(images.shape[0])
        feats = [torch.ones(1, 2, 2, 2)] * 3
        return {
            "backbone_fpn": feats,
            "vision_pos_enc": feats,
            "sam2_backbone_out": {"backbone_fpn": feats, "vision_pos_enc": feats},
        }

    def forward_text(self, texts, **kwargs):
        self.text_batches.append(texts)
        return {"texts": texts}

    def forward_video_grounding_multigpu(
        self, backbone_out, find_inputs, frame_idx, **kwargs
    ):
        stage = find_inputs[frame_idx]
        self.query_batches.append(stage.text_ids.tolist())
        backbone, *_ = self._get_img_feats(backbone_out, stage.img_ids)
        count = len(stage.text_ids)
        logits = torch.full((count, 1, 1), 8.0)
        masks = torch.zeros(count, 1, 2, 2)
        for i, text_id in enumerate(stage.text_ids.tolist()):
            text = backbone_out["texts"][text_id]
            if text == "no target":
                logits[i] = -8.0
            elif text == "right hand":
                masks[i] = 1.0
        return {
            "pred_logits": logits,
            "pred_boxes_xyxy": torch.zeros(count, 1, 4),
            "pred_masks": masks,
            **{
                f"tracker_backbone_fpn_{i}": f
                for i, f in enumerate(backbone["sam2_backbone_out"]["backbone_fpn"])
            },
            "tracker_backbone_pos_enc": backbone["sam2_backbone_out"]["vision_pos_enc"],
        }, backbone_out


class CPUVideoBatch(Sam3VideoInference):
    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.image_size = 8
        self.rank = 0
        self.world_size = 1
        self.det_nms_thresh = 0.0
        self.score_threshold_detection = 0.5
        self.hotstart_delay = 3
        self.masklet_confirmation_consecutive_det_thresh = 3
        self.detector = FakeDetector()
        self.tracker = SimpleNamespace(
            sam_mask_decoder=SimpleNamespace(conv_s0=lambda x: x, conv_s1=lambda x: x)
        )
        self.steps = []
        self.visual_features = []

    def _compile_model(self):
        pass

    def _run_single_frame_inference(self, state, frame_idx, reverse):
        detection = self.run_backbone_and_detection(
            frame_idx,
            state["num_frames"],
            state["input_batch"],
            state["constants"]["empty_geometric_prompt"],
            state["feature_cache"],
            reverse,
            True,
        )
        self.steps.append((frame_idx, state["text_prompt"]))
        self.visual_features.append(state["feature_cache"][frame_idx])
        metadata = state["tracker_metadata"]
        metadata.setdefault("frames", []).append(frame_idx)
        metadata.setdefault("obj_id_to_score", {0: 1.0})
        metadata.setdefault("obj_id_to_tracker_score_frame_wise", defaultdict(dict))
        metadata.setdefault("rank0_metadata", {"suppressed_obj_ids": defaultdict(set)})
        state["previous_stages_out"][frame_idx] = True
        return {
            "obj_id_to_mask": {0: detection["mask"]} if len(detection["mask"]) else {},
            "removed_obj_ids": set(),
            "suppressed_obj_ids": set(),
            "unconfirmed_obj_ids": [],
        }

    def _postprocess_output(self, state, out, *args, **kwargs):
        return {
            "text": state["text_prompt"],
            "mask": out["obj_id_to_mask"].get(0, torch.empty(0, 2, 2)),
        }


def initialized_batch(frame_count=8, start=0):
    model = CPUVideoBatch()
    state = model._init_state_from_images(
        torch.zeros(frame_count, 3, 8, 8), 8, 8, False, False
    )
    model.add_text_prompts(state, start, ["left hand", "right hand"])
    return model, state


@pytest.mark.parametrize(
    "reverse,start", [(False, 0), (True, 7), (False, 4), (True, 4), (True, 0)]
)
def test_batch_shares_one_image_and_advances_all_states_before_hotstart_yield(
    reverse, start
):
    model, state = initialized_batch(start=start)
    results = list(model.propagate_in_video_batch(state, start, 8, reverse))
    indices = list(range(start - 1, -1, -1) if reverse else range(start, 8))
    assert [i for i, _ in results] == indices
    assert model.detector.image_batches == [1] * (1 + len(indices))
    assert model.detector.query_batches == [[0, 1]] * (1 + len(indices))
    assert model.detector.text_batches == [["left hand", "right hand", "visual"]]
    for index, outputs in results:
        assert [output["text"] for output in outputs] == ["left hand", "right hand"]
        assert outputs[0]["mask"].eq(0).all()
        assert outputs[1]["mask"].eq(1).all()
    assert model.steps == [
        (index, prompt)
        for index in [start, *indices]
        for prompt in ["left hand", "right hand"]
    ]
    assert all(
        a is b for a, b in zip(model.visual_features[::2], model.visual_features[1::2])
    )
    left, right = state["text_prompt_states"]
    assert left["input_batch"].img_batch is right["input_batch"].img_batch
    assert left["feature_cache"] is not right["feature_cache"]
    assert left["tracker_metadata"] is not right["tracker_metadata"]
    assert left["tracker_inference_states"] is not right["tracker_inference_states"]
    assert not any(
        isinstance(k, int) and k != (indices[-1] if indices else start)
        for k in left["feature_cache"]
    )


def test_single_text_public_propagation_keeps_hotstart_and_flush_order():
    model, state = initialized_batch()
    left = state["text_prompt_states"][0]
    # Standalone testing uses detector outputs directly, without the group hook.
    outputs = list(model.propagate_in_video(left, 0, 8, False))
    assert [index for index, _ in outputs] == list(range(8))
    assert all(result["text"] == "left hand" for _, result in outputs)


@pytest.mark.parametrize(
    "texts", [[], "hand", [""], ["  "], [" visual "], ["hand", "hand"]]
)
def test_batch_rejects_invalid_texts_before_mutating_state(texts):
    model, state = initialized_batch()
    original = state["text_prompt_states"]
    with pytest.raises(ValueError):
        model.add_text_prompts(state, 0, texts)
    assert state["text_prompt_states"] is original


def test_reset_drops_batched_states_and_wrong_frame_detection_is_rejected():
    model, state = initialized_batch()
    child = state["text_prompt_states"][0]
    child["feature_cache"]["batched_detection"] = (5, {})
    with pytest.raises(RuntimeError, match="frame does not match"):
        model.run_backbone_and_detection(
            0, 8, None, None, child["feature_cache"], False, True
        )
    model.reset_state(state)
    assert "text_prompt_states" not in state
    assert "text_batch_input" not in state


@pytest.mark.parametrize(
    "prompts", [["left hand", "left_hand"], ["...", "hand"], ["a" * 256, "hand"]]
)
def test_directory_collisions_and_invalid_names_are_rejected(prompts):
    with pytest.raises(ValueError):
        dataset.text_prompt_directory_names(prompts)


def test_prompt_names_cannot_escape_output_root():
    assert dataset.text_prompt_directory_names(["../../hand", "right/hand"]) == {
        "../../hand": "hand",
        "right/hand": "right_hand",
    }


@pytest.mark.parametrize(
    "prompts,extra",
    [
        (["hand", "object"], ["--version", "sam3.1"]),
        (["", "hand"], []),
        (["  ", "hand"], []),
        (["left hand", "left_hand"], []),
        (["visual", "hand"], []),
        ([" visual ", "hand"], []),
    ],
)
def test_invalid_cli_fails_before_model_or_output_creation(prompts, extra, tmp_path):
    with pytest.raises(SystemExit):
        dataset.main(
            [
                "--text-prompt",
                *prompts,
                *extra,
                "--list-only",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )
    assert not (tmp_path / "out").exists()


def write_video(path, frames, color=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*("mp4v" if color else "FFV1")),
        12,
        (64, 48),
        color,
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def mask_output(prompt):
    mask = np.zeros((1, 48, 64), dtype=bool)
    mask[:, :, :32] = prompt == "left hand"
    mask[:, :, 32:] = prompt == "right hand"
    return {
        "out_obj_ids": np.array([0]),
        "out_probs": np.array([0.99]),
        "out_binary_masks": mask,
        "out_detector_mask": mask[0],
    }


@pytest.fixture
def fake_cli_model(monkeypatch):
    registry = SimpleNamespace(requests=[], builds=[], fail_next=False)
    requests, builds = registry.requests, registry.builds

    class Predictor:
        model = SimpleNamespace()
        texts = []

        def handle_request(self, request):
            requests.append(request)
            if request["type"] == "add_text_prompts":
                if registry.fail_next:
                    registry.fail_next = False
                    raise RuntimeError("CUDA out of memory")
                self.texts = request["texts"]
                return {"outputs": [mask_output(text) for text in self.texts]}
            return {"session_id": "test"}

        def handle_stream_request(self, request):
            assert request["type"] == "propagate_in_video_batch"
            requests.append(request)
            start = request["start_frame_index"]
            indices = (
                range(start, 4)
                if request["propagation_direction"] == "forward"
                else range(start - 1, -1, -1)
            )
            for index in indices:
                yield {
                    "frame_index": index,
                    "outputs": [mask_output(text) for text in self.texts],
                }

        def shutdown(self):
            pass

    def build(**kwargs):
        builds.append(kwargs)
        return Predictor()

    monkeypatch.setitem(
        sys.modules, "sam3", SimpleNamespace(build_sam3_predictor=build)
    )
    monkeypatch.setattr(dataset, "require_cuda", lambda *args: True)
    return registry


@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
def test_cli_batches_once_per_video_and_isolates_outputs_gt_and_partial_resume(
    tmp_path, fake_cli_model, direction
):
    root, out = tmp_path / "input", tmp_path / "out"
    video = root / "seq" / "color.mp4"
    write_video(video, [np.zeros((48, 64, 3), dtype=np.uint8)] * 4, True)
    write_video(
        video.parent / "masks_sam3" / "truth.mkv",
        [mask_output("left hand")["out_detector_mask"].astype(np.uint8)] * 4,
    )
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    requests, builds = fake_cli_model.requests, fake_cli_model.builds
    args = [
        "--input-root",
        str(root),
        "--output-root",
        str(out),
        "--checkpoint",
        str(checkpoint),
        "--text-prompt",
        "left hand",
        "right hand",
        "left hand",
        "--direction",
        direction,
        "--save-detector",
        "--compare-gt",
        "--gt-mask-name",
        "truth.mkv",
    ]
    assert dataset.main(args) == 0
    assert len(builds) == 1
    assert sum(r["type"] == "start_session" for r in requests) == 1
    assert next(r for r in requests if r["type"] == "add_text_prompts")["texts"] == [
        "left hand",
        "right hand",
    ]
    for prompt, slug, score in [
        ("left hand", "left_hand", 1.0),
        ("right hand", "right_hand", 0.0),
    ]:
        output = out / slug / "seq"
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["prompt"] == prompt
        assert metadata["status"] == "success"
        assert metadata["frames_processed"] == 4
        assert metadata["object_id_to_label"] == {"0": 1}
        assert dataset.probe_video(output / "detector_masks.mkv")["frame_count"] == 4
        summary = json.loads((out / slug / "gt_summary.json").read_text())
        assert len(summary["sequences"]) == 1
        assert summary["directions"][direction]["mean_iou"] == score
    left_before = (out / "left_hand" / "seq" / "metadata.json").read_bytes()
    assert dataset.main(args) == 0
    assert len(builds) == 1  # Complete cache needs no model.
    (out / "right_hand" / "seq" / "masks.mkv").unlink()
    assert dataset.main(args) == 0
    assert len(builds) == 2
    assert [r for r in requests if r["type"] == "add_text_prompts"][-1]["texts"] == [
        "right hand"
    ]
    assert (out / "left_hand" / "seq" / "metadata.json").read_bytes() == left_before


def test_oom_closes_session_marks_failure_and_continues_without_serial_fallback(
    tmp_path, fake_cli_model
):
    root, out = tmp_path / "input", tmp_path / "out"
    for name in ("a", "b"):
        write_video(
            root / name / "color.mp4", [np.zeros((48, 64, 3), dtype=np.uint8)] * 4, True
        )
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    fake_cli_model.fail_next = True
    assert (
        dataset.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(out),
                "--checkpoint",
                str(checkpoint),
                "--text-prompt",
                "left hand",
                "right hand",
            ]
        )
        == 1
    )
    assert len(fake_cli_model.builds) == 1
    assert sum(r["type"] == "close_session" for r in fake_cli_model.requests) == 2
    assert not any(r["type"] == "add_prompt" for r in fake_cli_model.requests)
    for slug in ("left_hand", "right_hand"):
        failed = json.loads((out / slug / "a" / "metadata.json").read_text())
        assert failed["status"] == "failed"
        assert "out of memory" in failed["error"]
        assert (
            json.loads((out / slug / "b" / "metadata.json").read_text())["status"]
            == "success"
        )


def test_factory_failure_preserves_completed_task_status(tmp_path, monkeypatch):
    args = dataset.build_parser().parse_args(
        ["--text-prompt", "left hand", "right hand"]
    )
    video = tmp_path / "color.mp4"
    monkeypatch.setattr(dataset, "probe_video", lambda *args: {"frame_count": 4})
    monkeypatch.setattr(
        dataset, "is_complete", lambda output, *args, **kwargs: output.name == "left"
    )

    def unavailable():
        raise RuntimeError("CUDA unavailable")

    assert dataset.process_video_text_batch(
        unavailable,
        video,
        tmp_path,
        {"left hand": tmp_path / "left", "right hand": tmp_path / "right"},
        args,
    ) == {
        "left hand": "skipped",
        "right hand": "failed",
    }
    assert not (tmp_path / "right").exists()


def test_derived_output_root_cannot_equal_input_even_in_list_only(tmp_path):
    root = tmp_path / "data" / "left_hand"
    video = root / "seq" / "result.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original RGB")
    assert (
        dataset.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(root.parent),
                "--rgb-name",
                "result.mp4",
                "--text-prompt",
                "left hand",
                "right hand",
                "--list-only",
            ]
        )
        == 2
    )
    assert video.read_bytes() == b"original RGB"


@pytest.mark.parametrize("alias", ["root", "sequence", "artifact", "temporary"])
def test_physical_output_aliases_are_rejected_before_processing(tmp_path, alias):
    root, out = tmp_path / "input", tmp_path / "out"
    video = root / "seq" / "color.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original RGB")
    left = out / "left_hand" / "seq"
    right = out / "right_hand" / "seq"
    left.mkdir(parents=True)
    if alias == "root":
        right.parent.symlink_to(left.parent, target_is_directory=True)
    elif alias == "sequence":
        right.parent.mkdir()
        right.symlink_to(left, target_is_directory=True)
    else:
        right.mkdir(parents=True)
        (
            right / ("metadata.json.tmp" if alias == "temporary" else "metadata.json")
        ).symlink_to(video)
    assert (
        dataset.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(out),
                "--text-prompt",
                "left hand",
                "right hand",
                "--list-only",
            ]
        )
        == 2
    )
    assert video.read_bytes() == b"original RGB"


@pytest.mark.parametrize("metadata", [[], None, "wrong type", 1])
def test_nonobject_metadata_is_incomplete_not_an_exception(tmp_path, metadata):
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert not dataset.is_complete(
        tmp_path, tmp_path / "color.mp4", "hand", "sam3", 4, "both"
    )


def test_gt_failure_report_error_does_not_stop_next_video(tmp_path, fake_cli_model):
    root, out = tmp_path / "input", tmp_path / "out"
    for name in ("a", "b"):
        video = root / name / "color.mp4"
        write_video(video, [np.zeros((48, 64, 3), dtype=np.uint8)] * 4, True)
        write_video(
            video.parent / "masks_sam3" / "truth.mkv",
            [np.zeros((48, 64), dtype=np.uint8)] * 4,
        )
    (out / "left_hand" / "a" / "gt_metrics.json").mkdir(parents=True)
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    assert (
        dataset.main(
            [
                "--input-root",
                str(root),
                "--output-root",
                str(out),
                "--checkpoint",
                str(checkpoint),
                "--text-prompt",
                "left hand",
                "right hand",
                "--compare-gt",
                "--gt-mask-name",
                "truth.mkv",
            ]
        )
        == 1
    )
    for slug in ("left_hand", "right_hand"):
        assert (
            json.loads((out / slug / "b" / "metadata.json").read_text())["status"]
            == "success"
        )


def test_rerun_without_gt_invalidates_previous_metrics(tmp_path, fake_cli_model):
    root, out = tmp_path / "input", tmp_path / "out"
    video = root / "seq" / "color.mp4"
    write_video(video, [np.zeros((48, 64, 3), dtype=np.uint8)] * 4, True)
    write_video(
        video.parent / "masks_sam3" / "truth.mkv",
        [np.zeros((48, 64), dtype=np.uint8)] * 4,
    )
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    args = [
        "--input-root",
        str(root),
        "--output-root",
        str(out),
        "--checkpoint",
        str(checkpoint),
        "--text-prompt",
        "left hand",
        "right hand",
    ]
    assert dataset.main([*args, "--compare-gt", "--gt-mask-name", "truth.mkv"]) == 0
    assert dataset.main([*args, "--overwrite"]) == 0
    for slug in ("left_hand", "right_hand"):
        assert not (out / slug / "gt_summary.json").exists()
        assert not (out / slug / "seq" / "gt_metrics.json").exists()
        assert not (out / slug / "seq" / "comparison.mp4").exists()


def test_empty_text_task_keeps_frame_order_without_consuming_other_detections():
    model = CPUVideoBatch()
    state = model._init_state_from_images(torch.zeros(8, 3, 8, 8), 8, 8, False, False)
    model.add_text_prompts(state, 0, ["left hand", "no target", "right hand"])
    results = list(model.propagate_in_video_batch(state, 0, 8))
    assert [index for index, _ in results] == list(range(8))
    assert model.detector.image_batches == [1] * 9
    for index, outputs in results:
        assert [output["text"] for output in outputs] == [
            "left hand",
            "no target",
            "right hand",
        ]
        assert outputs[0]["mask"].eq(0).all()
        assert len(outputs[1]["mask"]) == 0
        assert outputs[2]["mask"].eq(1).all()


class CPUInteractiveVideoBatch(
    CPUVideoBatch, Sam3VideoInferenceWithInstanceInteractivity
):
    pass


@pytest.mark.parametrize("frame_count", [1, 2, 8])
def test_interactive_model_batch_matches_independent_short_bidirectional_runs(
    frame_count,
):
    start = frame_count // 2
    batch = CPUInteractiveVideoBatch()
    state = batch._init_state_from_images(
        torch.zeros(frame_count, 3, 8, 8), 8, 8, False, False
    )
    batch.add_text_prompts(state, start, ["left hand", "right hand"])
    batched = dict(batch.propagate_in_video_batch(state, start, frame_count, False))
    if start:
        batched.update(batch.propagate_in_video_batch(state, start, frame_count, True))
    for index, prompt in enumerate(["left hand", "right hand"]):
        single = CPUInteractiveVideoBatch()
        single_state = single._init_state_from_images(
            torch.zeros(frame_count, 3, 8, 8), 8, 8, False, False
        )
        single.add_prompt(single_state, start, text_str=prompt)
        outputs = dict(
            single.propagate_in_video(single_state, start, frame_count, False)
        )
        if start:
            outputs.update(
                single.propagate_in_video(single_state, start, frame_count, True)
            )
        assert sorted(batched) == sorted(outputs) == list(range(frame_count))
        for frame in outputs:
            assert torch.equal(batched[frame][index]["mask"], outputs[frame]["mask"])
    calls = len(batch.detector.image_batches)
    # The real interactivity dispatcher fetches complete cached predictions on repeat.
    assert (
        len(list(batch.propagate_in_video_batch(state, start, frame_count, False)))
        == frame_count - start
    )
    assert len(batch.detector.image_batches) == calls

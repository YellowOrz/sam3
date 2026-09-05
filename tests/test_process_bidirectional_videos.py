import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import process_bidirectional_videos as processor


def square(x: int, *, size: int = 3, shape: tuple[int, int] = (12, 16)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[3 : 3 + size, x : x + size] = True
    return mask


def direction(
    name: str, obj_id: int, masks: list[np.ndarray], scores: list[float]
) -> processor.DirectionResult:
    return processor.DirectionResult(
        name,
        [{obj_id: mask} for mask in masks],
        [{obj_id: score} for score in scores],
        [{obj_id: score} for score in scores],
    )


def test_viterbi_prefers_one_stable_switch_over_framewise_flicker() -> None:
    forward = [square(1), square(2), square(3), square(4)]
    backward = [square(8), square(7), square(6), square(5)]
    result = processor.viterbi_select(
        {
            "F": forward,
            "B": backward,
            "O": [np.zeros((12, 16), dtype=bool) for _ in forward],
        },
        {
            "F": [0.9, 0.4, 0.9, 0.4],
            "B": [0.4, 0.9, 0.4, 0.9],
            "O": [0.0] * 4,
        },
        processor.FusionConfig(switch_penalty=1.0),
    )

    assert sum(a != b for a, b in zip(result.sources, result.sources[1:])) <= 1
    assert "O" not in result.sources


def test_empty_state_is_fail_closed_when_one_direction_is_confident() -> None:
    mask = square(2)
    empty = np.zeros_like(mask)

    result = processor.viterbi_select(
        {"F": [mask], "B": [empty], "O": [empty]},
        {"F": [0.8], "B": [0.0], "O": [0.0]},
        processor.FusionConfig(),
    )

    assert result.sources == ("F",)


def test_uncertainty_intervals_require_persistent_disagreement_and_failure() -> None:
    forward = [square(1) for _ in range(8)]
    backward = [square(1) for _ in range(8)]
    for index in range(2, 7):
        backward[index] = square(10)
    scores_forward = [0.9] * 8
    scores_backward = [0.9, 0.9] + [0.1] * 5 + [0.9]

    intervals = processor.uncertain_intervals(
        forward,
        backward,
        scores_forward,
        scores_backward,
        processor.FusionConfig(disagreement_min_frames=5),
    )

    assert intervals == [(2, 7)]


def test_primary_instances_are_matched_by_masks_not_object_ids() -> None:
    target = [square(1), square(2)]
    distractor = [square(10), square(10)]
    forward = processor.DirectionResult(
        "F",
        [{7: target[0], 8: distractor[0]}, {7: target[1], 8: distractor[1]}],
        [{7: 0.8, 8: 0.7}] * 2,
        [{7: 0.8, 8: 0.7}] * 2,
    )
    backward = processor.DirectionResult(
        "B",
        [{70: distractor[0], 80: target[0]}, {70: distractor[1], 80: target[1]}],
        [{70: 0.7, 80: 0.8}] * 2,
        [{70: 0.7, 80: 0.8}] * 2,
    )

    forward_id, backward_id, pairs = processor.match_primary_tracks(forward, backward)

    assert (forward_id, backward_id) == (7, 80)
    assert pairs[0]["mean_iou"] == 1.0


def test_run_direction_maps_reversed_indices_and_closes_session(tmp_path: Path) -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.requests = []

        def handle_request(self, request):
            self.requests.append(request)
            if request["type"] == "start_session":
                return {"session_id": "fresh"}
            if request["type"] == "add_prompt":
                index = request["frame_index"]
                return {
                    "outputs": {
                        "out_obj_ids": np.array([9]),
                        "out_probs": np.array([0.5]),
                        "out_tracker_probs": np.array([index / 10]),
                        "out_binary_masks": np.array([square(index)]),
                    }
                }
            return {"outputs": {}}

        def handle_stream_request(self, request):
            self.requests.append(request)
            start = request["start_frame_index"]
            for index in range(start, 3):
                yield {
                    "frame_index": index,
                    "outputs": {
                        "out_obj_ids": np.array([9]),
                        "out_probs": np.array([0.5]),
                        "out_tracker_probs": np.array([index / 10]),
                        "out_binary_masks": np.array([square(index)]),
                    },
                }

    predictor = FakePredictor()
    result = processor.run_direction(
        predictor, tmp_path, 3, "left hand", "B", reverse_index=True
    )

    assert result.tracker_scores[2][9] == 0.0
    assert result.tracker_scores[0][9] == pytest.approx(0.2)
    assert predictor.requests[2]["start_frame_index"] == 1
    assert predictor.requests[2]["max_frame_num_to_track"] == 2
    assert predictor.requests[0]["type"] == "start_session"
    assert predictor.requests[-1] == {"type": "close_session", "session_id": "fresh"}


def test_api_backward_preserves_prompt_frame_and_starts_before_it(
    tmp_path: Path,
) -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.stream_request = None

        def handle_request(self, request):
            if request["type"] == "start_session":
                return {"session_id": "fresh"}
            if request["type"] == "add_prompt":
                return {
                    "outputs": {
                        "out_obj_ids": np.array([9]),
                        "out_probs": np.array([0.8]),
                        "out_tracker_probs": np.array([0.9]),
                        "out_binary_masks": np.array([square(8)]),
                    }
                }
            return {"outputs": {}}

        def handle_stream_request(self, request):
            self.stream_request = request
            for index in range(request["start_frame_index"] - 1, -1, -1):
                yield {
                    "frame_index": index,
                    "outputs": {
                        "out_obj_ids": np.array([9]),
                        "out_probs": np.array([0.8]),
                        "out_tracker_probs": np.array([index / 10]),
                        "out_binary_masks": np.array([square(index)]),
                    },
                }

    predictor = FakePredictor()
    result = processor.run_direction(
        predictor,
        tmp_path,
        3,
        "left hand",
        "B-api",
        propagation_direction="backward",
    )

    assert result.tracker_scores[2][9] == pytest.approx(0.9)
    assert result.tracker_scores[1][9] == pytest.approx(0.1)
    assert predictor.stream_request["start_frame_index"] == 2
    assert predictor.stream_request["max_frame_num_to_track"] == 2


def test_backward_equivalence_detects_a_real_mask_difference() -> None:
    physical = direction("physical", 1, [square(1), square(2)], [0.8, 0.8])
    api = direction("api", 9, [square(1), square(4)], [0.8, 0.8])

    report = processor.compare_direction_results(physical, api, (12, 16), 0.999)

    assert not report["equivalent"]
    assert report["minimum_mask_iou"] < 0.999


@pytest.mark.parametrize(("minimum_gain", "accepted"), [(0.1, True), (2.0, False)])
def test_anchor_repair_is_only_applied_after_acceptance_gate(
    monkeypatch: pytest.MonkeyPatch, minimum_gain: float, accepted: bool
) -> None:
    shape = (12, 16)
    forward = [square(1), square(9), square(3)]
    backward = [square(1), square(10), square(3)]
    empty = [np.zeros(shape, dtype=bool) for _ in range(3)]
    repaired_candidate = [empty[0], square(2), empty[2]]

    monkeypatch.setattr(
        processor,
        "run_anchor_propagation",
        lambda *args, **kwargs: (repaired_candidate, [0.0, 0.99, 0.0]),
    )
    fused = [mask.copy() for mask in forward]
    sources = ["F"] * 3
    attempt = processor.try_repair_interval(
        object(),
        Path("frames"),
        3,
        "left hand",
        (1, 1),
        0,
        2,
        fused,
        sources,
        {"F": forward, "B": backward, "O": empty},
        {"F": [0.9, 0.1, 0.9], "B": [0.9, 0.1, 0.9], "O": [0.0] * 3},
        processor.FusionConfig(
            repair_min_gain=minimum_gain, recovery_iou_threshold=0.0
        ),
    )

    assert (attempt["status"] == "accepted") is accepted
    assert np.array_equal(fused[1], repaired_candidate[1] if accepted else forward[1])


def test_evaluation_reports_oracle_and_fused_metrics() -> None:
    target = [square(1), square(2)]
    wrong = [square(9), square(9)]
    report = processor.evaluate_masks(
        target,
        {
            "forward": [target[0], wrong[1]],
            "backward": [wrong[0], target[1]],
            "union": [np.logical_or(a, b) for a, b in zip(target, wrong)],
            "intersection": [np.logical_and(a, b) for a, b in zip(target, wrong)],
            "fused": target,
        },
    )

    assert report["per_frame_oracle_J_and_F"] == 1.0
    assert report["methods"]["fused"]["J_and_F"] == 1.0


def test_unlabeled_audit_does_not_treat_both_empty_as_evidence_of_quality() -> None:
    empty = np.zeros((12, 16), dtype=bool)
    audit = processor.bidirectional_audit([empty, square(1)], [empty, square(10)])

    assert audit["both_empty_frames"] == 1
    assert audit["frames_below_iou_0_5"] == 1


def test_result_video_is_two_by_two(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame = np.full((12, 16, 3), 80, dtype=np.uint8)
    assert cv2.imwrite(str(frame_dir / "000000.png"), frame)
    output = tmp_path / "result.mp4"
    mask = square(2)

    processor.write_result_video(
        output, frame_dir, [mask], [mask], [mask], ["F"], [0.1], [False], 12.0
    )

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 24
    finally:
        capture.release()


def test_process_video_writes_candidates_fusion_and_diagnostics(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    video_path = input_root / "color.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (16, 12)
    )
    assert writer.isOpened()
    for value in (30, 60, 90):
        writer.write(np.full((12, 16, 3), value, dtype=np.uint8))
    writer.release()

    class FakePredictor:
        def __init__(self) -> None:
            self.sessions = {}
            self.next_session = 0

        def handle_request(self, request):
            if request["type"] == "start_session":
                session_id = str(self.next_session)
                self.next_session += 1
                self.sessions[session_id] = "reversed" in request["resource_path"]
                return {"session_id": session_id}
            if request["type"] == "add_prompt":
                reversed_frames = self.sessions[request["session_id"]]
                index = request["frame_index"]
                source_index = 2 - index if reversed_frames else index
                return {
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_probs": np.array([0.8]),
                        "out_tracker_probs": np.array([0.8]),
                        "out_binary_masks": np.array([square(source_index + 1)]),
                    }
                }
            return {"outputs": {}}

        def handle_stream_request(self, request):
            reversed_frames = self.sessions[request["session_id"]]
            start = request["start_frame_index"]
            count = request["max_frame_num_to_track"]
            indices = (
                range(start - 1, max(start - count, 0) - 1, -1)
                if request["propagation_direction"] == "backward"
                else range(start, min(start + count, 3))
            )
            for index in indices:
                source_index = 2 - index if reversed_frames else index
                yield {
                    "frame_index": index,
                    "outputs": {
                        "out_obj_ids": np.array([7]),
                        "out_probs": np.array([0.8]),
                        "out_tracker_probs": np.array([0.8]),
                        "out_binary_masks": np.array([square(source_index + 1)]),
                    },
                }

    output_dir = tmp_path / "output"
    status = processor.process_video(
        FakePredictor(),
        video_path,
        input_root,
        output_dir,
        "left hand",
        "sam3",
        None,
        False,
        processor.FusionConfig(),
        False,
        None,
        1,
        "verify",
        0.999,
    )

    assert status == "success"
    assert all(
        (output_dir / name).is_file()
        for name in (
            "forward/masks.mkv",
            "backward/masks.mkv",
            "masks.mkv",
            "result.mp4",
            "frames.jsonl",
            "audit.json",
            "backward_equivalence.json",
        )
    )
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["backward_equivalence"]["equivalent"]
    assert metadata["status"] == "success"


def test_parser_defaults_to_verification_and_conservative_repair() -> None:
    args = processor.build_parser().parse_args(
        ["--input-root", "input", "--output-root", "output", "--prompt", "left hand"]
    )

    assert args.backward_mode == "verify"
    assert args.backward_equivalence_iou == pytest.approx(0.999)
    assert not args.repair_uncertain
    assert args.ground_truth_label == 1

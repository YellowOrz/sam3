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
        output, frame_dir, [mask], [mask], [mask], ["decoded"], 12.0
    )

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 24
    finally:
        capture.release()


def entry(frame, direction, *, quality=0.9, conditioning=False, object_id=7):
    return {
        "frame_index": frame,
        "direction": direction,
        "quality": quality,
        "conditioning": conditioning,
        "object_id": object_id,
        "path": f"{direction}_{frame}.pt",
        "spatial_tokens": 4,
        "pointer_values": 8,
    }


@pytest.mark.parametrize("side", ["both", "past", "future"])
def test_memory_selection_enforces_direction_budget_and_no_self(side):
    entries = [entry(i, d) for d in ("F", "B") for i in range(9)]
    entries += [entry(3, "F")]
    config = processor.MemoryConfig(side=side)
    selected = processor.select_memories(entries, 4, 4, config)
    assert len(selected) == 4
    assert len({(e["direction"], e["frame_index"]) for e in selected}) == 4
    assert all(
        e["frame_index"] < 4 if e["direction"] == "F" else e["frame_index"] > 4
        for e in selected
    )
    if side == "both":
        assert [e["direction"] for e in selected].count("F") == 2
    else:
        assert {e["direction"] for e in selected} == (
            {"F"} if side == "past" else {"B"}
        )


def test_memory_selection_refills_missing_side_without_deleting_invisibility():
    entries = [entry(0, "F", quality=None), entry(1, "F"), entry(2, "F", quality=0.1)]
    entries[1]["presence_logit"] = -10.0
    selected = processor.select_memories(
        entries, 3, 3, processor.MemoryConfig(min_quality=0.5)
    )
    assert selected == [entries[1]]
    assert len(processor.select_memories(entries, 3, 3, processor.MemoryConfig())) == 3
    assert processor.select_memories(entries, 3, 0, processor.MemoryConfig()) == []


def test_overlapping_windows_have_exactly_one_owner_per_frame():
    windows = list(
        processor.processing_windows(
            5, processor.MemoryConfig(chunk_frames=2, context_frames=1)
        )
    )
    assert windows == [(0, 2, 0, 3), (2, 4, 1, 5), (4, 5, 3, 5)]
    assert [t for a, b, _, _ in windows for t in range(a, b)] == list(range(5))
    with pytest.raises(ValueError, match="requires"):
        processor.MemoryConfig(context_frames=1)


class MemoryPredictor:
    """Exercise real disk/video orchestration without CUDA or model weights."""

    def __init__(self):
        self.sessions = {}
        self.next_session = 0
        self.decoded = []
        self.fail_decode = False

    def begin_memory_capture(self, session_id, directory):
        directory.mkdir()
        self.sessions[session_id]["capture"] = directory

    def finish_memory_capture(self, session_id):
        return self.sessions[session_id]["records"]

    def frame(self, session, index):
        state = self.sessions[session]
        image = cv2.imread(str(state["path"] / f"{index:06d}.png"))
        original = round(float(image.mean()) / 30) - 1
        obj = 9 if "reversed" in str(state["path"]) else 7
        if "capture" in state:
            state["records"][index] = {obj: entry(index, "F", object_id=obj)}
        return {
            "out_obj_ids": np.array([obj]),
            "out_probs": np.array([0.9]),
            "out_tracker_probs": np.array([0.9]),
            "out_binary_masks": np.array([square(original + 1)]),
        }

    def handle_request(self, request):
        kind = request["type"]
        if kind == "start_session":
            session = str(self.next_session)
            self.next_session += 1
            self.sessions[session] = {
                "path": Path(request["resource_path"]),
                "records": {},
            }
            return {"session_id": session}
        session = request["session_id"]
        if kind == "add_prompt":
            return {"outputs": self.frame(session, request["frame_index"])}
        if kind == "close_session":
            del self.sessions[session]
        return {}

    def handle_stream_request(self, request):
        start, count = request["start_frame_index"], request["max_frame_num_to_track"]
        indices = (
            range(start, start + count)
            if request["propagation_direction"] == "forward"
            else range(start - 1, start - count - 1, -1)
        )
        for i in indices:
            yield {"frame_index": i, "outputs": self.frame(request["session_id"], i)}

    def decode_memory_frame(self, session, index, spatial, pointers):
        if self.fail_decode:
            raise RuntimeError("decode failed")
        assert all(
            (
                e["frame_index"] < index
                if e["direction"] == "F"
                else e["frame_index"] > index
            )
            for e in spatial + pointers
        )
        self.decoded.append((session, index, spatial, pointers))
        return {
            "mask": square(10),
            "predicted_iou": 0.85,
            "presence_logit": 3.0,
            "spatial_tokens": 4 * len(spatial),
            "pointer_tokens": len(pointers),
        }


def make_video(tmp_path, count=5):
    root = tmp_path / "input"
    root.mkdir()
    video = root / "color.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (16, 12))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((12, 16, 3), (i + 1) * 30, dtype=np.uint8))
    writer.release()
    return root, video


@pytest.mark.parametrize("chunk,context", [(0, 0), (2, 0), (2, 1)])
def test_memory_pipeline_decodes_core_once_and_writes_provenance(
    tmp_path, chunk, context
):
    root, video = make_video(tmp_path)
    out = tmp_path / "output"
    predictor = MemoryPredictor()
    config = processor.MemoryConfig(chunk_frames=chunk, context_frames=context)
    assert (
        processor.process_video(predictor, video, root, out, "left hand", config)
        == "success"
    )
    assert not predictor.sessions
    records = [
        json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()
    ]
    assert [r["frame_index"] for r in records] == list(range(5))
    # With no context the final singleton block has no legal self-free memory.
    decoded_count = 4 if (chunk, context) == (2, 0) else 5
    assert len(predictor.decoded) == decoded_count
    for record in records:
        assert "selected_source" not in record
        for e in record["spatial_memory"] + record["pointer_memory"]:
            assert "path" not in e
            assert (
                e["frame_index"] < record["frame_index"]
                if e["direction"] == "F"
                else e["frame_index"] > record["frame_index"]
            )
    masks = processor.read_label_video(out / "masks.mkv", 5, (12, 16), 1)
    assert all(np.array_equal(m, square(10)) for m in masks[:decoded_count])
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["source_banks_frozen"] and metadata["training"] is False
    assert metadata["requires_review_frames"] == 5 - decoded_count
    assert (
        processor.process_video(predictor, video, root, out, "left hand", config)
        == "skipped"
    )
    with pytest.raises(FileExistsError):
        processor.process_video(predictor, video, root, out, "right hand", config)


def test_decode_failure_closes_session_and_marks_output_failed(tmp_path):
    root, video = make_video(tmp_path, 3)
    predictor = MemoryPredictor()
    predictor.fail_decode = True
    out = tmp_path / "output"
    with pytest.raises(RuntimeError, match="decode failed"):
        processor.process_video(
            predictor, video, root, out, "left hand", processor.MemoryConfig()
        )
    assert not predictor.sessions
    assert json.loads((out / "metadata.json").read_text())["status"] == "failed"


def test_parser_and_list_only_do_not_load_sam3(tmp_path):
    root, _ = make_video(tmp_path, 1)
    argv = [
        "--input-root",
        str(root),
        "--output-root",
        str(tmp_path / "out"),
        "--prompt",
        "left hand",
    ]
    args = processor.build_parser().parse_args(argv)
    assert args.backward_mode == "physical" and args.chunk_frames == 0
    assert processor.main(argv + ["--list-only"]) == 0
    with pytest.raises(SystemExit):
        processor.build_parser().parse_args(argv + ["--version", "sam3.1"])

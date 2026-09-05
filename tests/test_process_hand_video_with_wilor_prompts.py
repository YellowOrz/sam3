import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import process_hand_video_with_wilor_prompts as processor


def make_joint(
    index: int,
    x: int,
    y: int,
    status: str = "visible",
    reliability: float = 30.0,
    confidence: float = 0.8,
) -> dict:
    return {
        "joint_index": index,
        "joint_name": f"joint_{index}",
        "pixel": [float(x), float(y)],
        "sample_pixel": [x, y],
        "detection_confidence": confidence,
        "occlusion_status": status,
        "reliability_score_px": reliability,
    }


def make_hand(side: str, joints: list, confidence: float = 0.8) -> dict:
    return {
        "side": side,
        "detection_confidence": confidence,
        "bbox_xyxy": [0, 0, 20, 20],
        "joints": joints,
    }


def test_filtered_candidates_keep_only_supported_statuses_and_thresholds() -> None:
    frame = {
        "hands": [
            make_hand(
                "left",
                [
                    make_joint(0, 2, 2),
                    make_joint(1, 3, 3, "occluded", 26),
                    make_joint(2, 4, 4, "uncertain", 100),
                    make_joint(3, 5, 5, "visible", 24),
                    make_joint(6, 8, 8, confidence=0.6),
                ],
            ),
            make_hand(
                "right",
                [
                    make_joint(4, 6, 6),
                    make_joint(5, 7, 7, "occluded"),
                ],
            ),
        ]
    }

    candidates = processor.filtered_candidates(frame, "left", 10, 10, 0.7, 25)

    assert [(joint.side, joint.status, joint.joint_index) for joint in candidates] == [
        ("left", "visible", 0),
        ("left", "occluded", 1),
        ("right", "visible", 4),
    ]


def test_select_prompts_uses_spread_positives_and_supported_negative_joints() -> None:
    candidates = [
        processor.JointCandidate("left", "visible", 1, "pos", 2, 2, (2, 2), 0.8, 30),
        processor.JointCandidate("left", "visible", 2, "best", 3, 3, (3, 3), 0.8, 40),
        processor.JointCandidate(
            "left", "occluded", 3, "hidden", 8, 8, (8, 8), 0.8, 35
        ),
        processor.JointCandidate(
            "right", "visible", 4, "other", 12, 12, (12, 12), 0.8, 36
        ),
    ]
    mask = np.ones((20, 20), dtype=bool)

    points = processor.select_frame_prompts(candidates, {}, mask, "left", 0.5, 5)

    assert [(point.kind, point.label) for point in points] == [
        ("joint_positive", 1),
        ("joint_positive", 1),
        ("target_occluded_negative", 0),
        ("opposite_visible_negative", 0),
    ]
    assert points[0].joint.joint_name == "best"
    assert points[1].joint.joint_name == "pos"


def test_positive_prompts_are_capped_and_spatially_distributed() -> None:
    candidates = [
        processor.JointCandidate(
            "left", "visible", index, f"joint_{index}", x, y, (x, y), 0.8, reliability
        )
        for index, x, y, reliability in (
            (0, 5, 5, 100),
            (1, 6, 5, 90),
            (2, 20, 5, 80),
            (3, 5, 20, 70),
        )
    ]

    points = processor.select_frame_prompts(
        candidates, {}, np.ones((25, 25), dtype=bool), "left", 0.5, 5
    )

    assert [point.joint.joint_index for point in points] == [0, 2, 3]


def test_negative_prompts_require_a_same_frame_positive() -> None:
    candidate = processor.JointCandidate(
        "right", "visible", 4, "other", 5, 5, (5, 5), 0.8, 30
    )

    points = processor.select_frame_prompts(
        [candidate], {}, np.ones((10, 10), dtype=bool), "left", 0.5, 5
    )

    assert points == []


def test_positive_outside_the_mask_is_never_submitted() -> None:
    candidate = processor.JointCandidate(
        "left", "visible", 4, "tip", 8, 8, (8, 8), 0.8, 30
    )

    points = processor.select_frame_prompts(
        [candidate], {}, np.zeros((10, 10), dtype=bool), "left", 0.5, 5
    )

    assert points == []


def test_opposite_hand_negative_does_not_need_to_be_inside_mask() -> None:
    candidates = [
        processor.JointCandidate("left", "visible", 1, "pos", 5, 5, (5, 5), 0.8, 30),
        processor.JointCandidate(
            "right", "visible", 2, "other", 15, 5, (15, 5), 0.8, 30
        ),
    ]
    mask = np.zeros((12, 20), dtype=bool)
    mask[5, 5] = True

    points = processor.select_frame_prompts(candidates, {}, mask, "left", 0.25, 5)

    assert [point.kind for point in points] == [
        "joint_positive",
        "opposite_visible_negative",
    ]


def test_negative_near_any_positive_is_dropped() -> None:
    candidates = [
        processor.JointCandidate("left", "visible", 1, "pos", 5, 5, (5, 5), 0.8, 30),
        processor.JointCandidate("left", "visible", 2, "pos2", 15, 5, (15, 5), 0.8, 29),
        processor.JointCandidate("right", "visible", 3, "neg", 18, 5, (18, 5), 0.8, 30),
    ]

    points = processor.select_frame_prompts(
        candidates, {}, np.ones((12, 24), dtype=bool), "left", 0.5, 5
    )

    assert [point.kind for point in points] == ["joint_positive", "joint_positive"]


def test_arm_negative_is_on_forearm_side_and_clear_of_joints() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:90, 45:56] = True
    joints = {
        0: (50, 50),
        5: (44, 30),
        9: (48, 30),
        13: (52, 30),
        17: (56, 30),
    }

    point = processor.arm_negative_point(mask, joints, distance_ratio=0.5)

    assert point is not None
    assert point[1] > 50
    assert np.linalg.norm(np.asarray(point) - np.asarray(joints[0])) > 10


def test_choose_target_object_prefers_target_joint_coverage() -> None:
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 1, 1] = True
    masks[1, 2, 2] = True
    candidates = [
        processor.JointCandidate("left", "visible", 1, "target", 2, 2, (2, 2), 0.8, 30),
        processor.JointCandidate("right", "visible", 2, "other", 1, 1, (1, 1), 0.8, 30),
    ]
    outputs = {
        "out_obj_ids": np.asarray([10, 20]),
        "out_probs": np.asarray([0.9, 0.7]),
        "out_binary_masks": masks,
    }

    assert processor.choose_target_object(outputs, candidates, "left", 10, 10) == 20


def write_jsonl(path: Path, records: list) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_jsonl_validation_rejects_multiple_same_side_hands(tmp_path: Path) -> None:
    path = tmp_path / "joints.jsonl"
    write_jsonl(
        path,
        [
            {
                "record_type": "metadata",
                "schema_version": 1,
                "video": {"width": 10, "height": 8, "fps": 30},
            },
            {
                "record_type": "frame",
                "frame_index": 0,
                "hands": [make_hand("left", []), make_hand("left", [])],
            },
        ],
    )

    with pytest.raises(ValueError, match="multiple left hands.*frame 0"):
        processor.load_joint_jsonl(
            path, {"width": 10, "height": 8, "fps": 30, "frame_count": 1}
        )


def test_parser_defaults_to_agreed_thresholds_and_sam3() -> None:
    args = processor.build_parser().parse_args(
        ["--input-dir", "input", "--output-dir", "output", "--hand-side", "left"]
    )

    assert args.version == "sam3"
    assert args.segmentation_passes == 2
    assert args.detection_confidence_threshold == 0.7
    assert args.reliability_distance_threshold_px == 25
    assert args.arm_distance_ratio == 0.25
    assert not hasattr(args, "recovery_frames")

    with pytest.raises(SystemExit):
        processor.build_parser().parse_args(
            [
                "--input-dir",
                "input",
                "--output-dir",
                "output",
                "--hand-side",
                "left",
                "--segmentation-passes",
                "0",
            ]
        )


@pytest.mark.parametrize(
    ("segmentation_passes", "expected_point_requests"), [(1, 0), (2, 3), (3, 6)]
)
def test_iterative_process_writes_outputs_and_normalized_points(
    tmp_path: Path, segmentation_passes: int, expected_point_requests: int
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    joints_dir = input_dir / "MANO_wilor_occlusion"
    joints_dir.mkdir(parents=True)
    video_path = input_dir / "color.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 12)
    )
    assert writer.isOpened()
    for value in (20, 40, 60):
        writer.write(np.full((12, 16, 3), value, dtype=np.uint8))
    writer.release()

    frames = [
        {
            "record_type": "frame",
            "frame_index": frame_index,
            "hands": [
                make_hand("left", [make_joint(0, 3, 3)]),
                make_hand("right", [make_joint(1, 12, 8)]),
            ],
        }
        for frame_index in range(3)
    ]
    write_jsonl(
        joints_dir / "hand_joints_occlusion.jsonl",
        [
            {
                "record_type": "metadata",
                "schema_version": 1,
                "video": {"width": 16, "height": 12, "fps": 10},
            },
            *frames,
        ],
    )

    class FakePredictor:
        def __init__(self) -> None:
            self.requests = []

        @staticmethod
        def outputs() -> dict:
            return {
                "out_obj_ids": np.asarray([7]),
                "out_probs": np.asarray([0.8]),
                "out_binary_masks": np.ones((1, 12, 16), dtype=bool),
            }

        def handle_request(self, request: dict) -> dict:
            self.requests.append(request)
            if request["type"] == "start_session":
                return {"session_id": "fake"}
            if request["type"] in {"save_checkpoint", "restore_checkpoint"}:
                return {"is_success": True, "frame_index": request["frame_index"]}
            return {
                "frame_index": request.get("frame_index", 0),
                "outputs": self.outputs(),
            }

        def handle_stream_request(self, request: dict):
            self.requests.append(request)
            start = request["start_frame_index"]
            maximum = request["max_frame_num_to_track"]
            step = 1 if request["propagation_direction"] == "forward" else -1
            for offset in range(maximum + 1):
                yield {
                    "frame_index": start + step * offset,
                    "outputs": self.outputs(),
                }

    predictor = FakePredictor()
    args = processor.build_parser().parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--hand-side",
            "left",
            "--segmentation-passes",
            str(segmentation_passes),
        ]
    )

    processor.process_video(predictor, args)

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["status"] == "success"
    assert metadata["requested_segmentation_passes"] == segmentation_passes
    assert metadata["completed_segmentation_passes"] == segmentation_passes
    assert metadata["prompted_frames"] == (0 if segmentation_passes == 1 else 3)
    assert len(metadata["passes"]) == segmentation_passes
    assert metadata["target_obj_id"] == 7
    assert all(
        (output_dir / name).is_file()
        for name in ("result.mp4", "baseline_masks.mkv", "masks.mkv")
    )
    point_requests = [
        request
        for request in predictor.requests
        if request["type"] == "add_prompt" and "points" in request
    ]
    assert len(point_requests) == expected_point_requests
    if point_requests:
        assert point_requests[0]["points"] == [
            [3 / 16, 3 / 12],
            [12 / 16, 8 / 12],
        ]
        assert point_requests[0]["point_labels"] == [1, 0]
    result_info = processor.probe_video(output_dir / "result.mp4")
    assert (result_info["width"], result_info["height"]) == (48, 12)

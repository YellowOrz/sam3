"""uni-hoi loader for learned-prompt training; no SAM3 weights required."""

import json
from pathlib import Path

import numpy as np

from sam3.train.data.uni_hoi import (
    build_uni_hoi_raw_data,
    instance_ids_on_frame,
    split_for_sequence,
    UniHoiTargetCOCO,
)


def write_mini_dataset(root: Path) -> None:
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "split.json").write_text(
        json.dumps(
            {"toy": {"by_subject": {"s1": "train", "s2": "val"}}},
        ),
        encoding="utf-8",
    )
    layouts = (
        ("s1", "s1_a", 3),
        ("s2", "s2_a", 3),
    )
    for subject, seq_id, num_frames in layouts:
        seq_dir = root / "sequences" / "toy" / seq_id
        view = seq_dir / "cam0"
        view.mkdir(parents=True)
        (seq_dir / "sequence.json").write_text(
            json.dumps(
                {
                    "source": "toy",
                    "seq_id": seq_id,
                    "subject": subject,
                    "views": ["cam0"],
                    "num_frames": num_frames,
                    "intrinsics": {"cam0": {"w": 4, "h": 4}},
                }
            ),
            encoding="utf-8",
        )
        (view / "instances.json").write_text(
            json.dumps(
                {
                    "view_id": "cam0",
                    "instances": [
                        {
                            "kind": "hand_right",
                            "frame_map": [{"frames": [0, num_frames - 1], "id": 1}],
                        },
                        {
                            "kind": "hand_left",
                            "frame_map": [{"frames": [0, 0], "id": 2}],
                        },
                        {
                            "kind": "object",
                            "frame_map": [{"frames": [0, num_frames - 1], "id": 3}],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        (view / "rgb.mkv").write_bytes(b"rgb")
        (view / "mask.mkv").write_bytes(b"mask")


def decode_mini(_path: Path, width: int, height: int):
    assert (width, height) == (4, 4)
    frames = []
    for index in range(3):
        mask = np.zeros((height, width), dtype=np.uint16)
        if index == 0:
            mask[:2, :2] = 1
            mask[2:, 2:] = 2
            mask[:1, 2:] = 3
        elif index == 1:
            mask[:2, :2] = 1
            mask[2:, 2:] = 3
        frames.append(mask)
    return frames


def test_frame_map_and_subject_split():
    instances = [
        {"kind": "hand_right", "frame_map": [{"frames": [0, 2], "id": 1}]},
        {"kind": "hand_left", "frame_map": [{"frames": [4, 4], "id": 7}]},
    ]
    assert instance_ids_on_frame(instances, 2, "hand_right") == [1]
    assert instance_ids_on_frame(instances, 3, "hand_right") == []
    assert instance_ids_on_frame(instances, 4, "hand_left") == [7]
    assert (
        split_for_sequence(
            {"dexycb": {"by_subject": {"subject-09": "val"}}},
            {"source": "dexycb", "subject": "subject-09"},
        )
        == "val"
    )


def test_selected_kind_keeps_negatives_and_drops_other_instances(tmp_path):
    write_mini_dataset(tmp_path)
    raw = build_uni_hoi_raw_data(
        tmp_path, "train", "hand_right", 1, decode_mask=decode_mini
    )
    assert len(raw) == 3
    assert [len(item["annotations"]) for item in raw] == [1, 1, 0]
    assert raw[0]["annotations"][0]["bbox"] == [0.0, 0.0, 2.0, 2.0]
    assert raw[0]["image"]["file_name"].endswith("rgb.mkv@0")
    val = build_uni_hoi_raw_data(
        tmp_path, "val", "hand_left", 1, decode_mask=decode_mini
    )
    assert len(val) == 3
    assert [len(item["annotations"]) for item in val] == [1, 0, 0]


def test_uni_hoi_queries_match_target_coco_contract(tmp_path):
    write_mini_dataset(tmp_path)
    loader = UniHoiTargetCOCO(
        str(tmp_path / "metadata" / "split.json"),
        1,
        "right_hand",
        str(tmp_path),
        "train",
        "hand_right",
        decode_mask=decode_mini,
    )
    positive, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)
    negative, absent = loader.loadQueriesAndAnnotationsFromDatapoint(2)
    assert len(annotations) == 1
    assert positive[0]["query_text"] == "right_hand"
    assert negative[0]["object_ids_output"] == []
    assert absent == []
    assert loader.getDatapointIds() == [0, 1, 2]

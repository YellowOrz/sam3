"""CPU-only export/review checks; no model construction or checkpoint downloads."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import qualitative_test_interactive as qualitative


def make_outputs(tmp_path, ids=(0, 2), frame_count=2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    masks = {}
    for index, obj_id in enumerate(ids):
        mask = np.zeros((64, 96), dtype=bool)
        mask[35:55, 5 + index * 30 : 25 + index * 30] = True
        masks[obj_id] = mask
    for index in range(frame_count):
        assert cv2.imwrite(
            str(frame_dir / f"{index:05d}.jpg"), np.zeros((64, 96, 3), dtype=np.uint8)
        )
    app = SimpleNamespace(
        output_dir=tmp_path / "output",
        frame_dir=frame_dir,
        video_path=tmp_path / "input.mp4",
        video_info=qualitative.VideoInfo(96, 64, frame_count, 12.0),
        frame_count=frame_count,
        frame_offset=0,
        cache={index: dict(masks) for index in range(frame_count)},
        probability_cache={},
        prompt="hand",
        version="sam3",
        initial_propagation_mode="forward",
        confirmed_points=(
            [qualitative.PointEdit(1, 0, ids[-1], 40, 40, 1)] if ids else []
        ),
        events=[{"sequence": 2, "type": "remove_object", "obj_id": 1}],
    )
    qualitative.write_interactive_outputs(app)
    return app


def read_json(directory, name="metadata.json"):
    return json.loads((directory / name).read_text(encoding="utf-8"))


def read_masks(directory):
    capture = cv2.VideoCapture(str(directory / "masks.mkv"))
    frames = []
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame[:, :, 0])
    finally:
        capture.release()
    return frames


def test_compaction_updates_export_without_mutating_tracker_cache(tmp_path):
    app = make_outputs(tmp_path)
    metadata = read_json(app.output_dir)
    assert metadata["original_object_id_to_object_id"] == {"0": 0, "2": 1}
    assert metadata["object_id_to_label"] == {"0": 1, "1": 2}
    assert list(app.cache[0]) == [0, 2]
    assert app.confirmed_points[0].obj_id == 2
    labels = read_masks(app.output_dir)[0]
    assert labels[40, 10] == 1
    assert labels[40, 40] == 2
    interactions = read_json(app.output_dir, "interactions.json")
    point = interactions["confirmed_points"][0]
    assert (point["original_obj_id"], point["obj_id"]) == (2, 1)
    event = interactions["events"][0]
    assert event["original_obj_id"] == 1
    assert event["obj_id"] is None
    assert app.events[0]["obj_id"] == 1


def test_compaction_order_is_old_id_order_not_first_appearance(tmp_path):
    app = make_outputs(tmp_path, ids=(8, 3))
    app.cache[0] = {8: app.cache[0][8]}
    qualitative.write_interactive_outputs(app)
    assert read_json(app.output_dir)["original_object_id_to_object_id"] == {
        "3": 0,
        "8": 1,
    }
    assert read_masks(app.output_dir)[0][40, 10] == 2


def test_all_objects_deleted_exports_background(tmp_path):
    app = make_outputs(tmp_path, ids=())
    assert read_json(app.output_dir)["object_id_to_label"] == {}
    assert not read_masks(app.output_dir)[0].any()


@pytest.mark.parametrize(
    "command",
    [
        "map 2 0=1",
        "map 0 0=2",
        "map 3 0=2",
        "map 2 9=2",
        "map 2 0=-1",
        "map 2 0=1 0=2",
        "map 2 0=a",
        "map 2 0=1=2",
        "map 2",
    ],
)
def test_invalid_mapping_never_mutates_current_mapping(command):
    mappings = [{0: 0, 1: 1}, {0: 0, 1: 1}]
    with pytest.raises(ValueError):
        qualitative.parse_mapping_command(command, mappings)
    assert mappings == [{0: 0, 1: 1}, {0: 0, 1: 1}]


def test_atomic_swap_and_new_global_identity():
    mappings = [{0: 0, 1: 1}, {0: 0, 1: 1}]
    swapped = qualitative.parse_mapping_command("map 2 0=1 1=0", mappings)
    assert swapped == [{0: 0, 1: 1}, {0: 1, 1: 0}]
    separated = qualitative.parse_mapping_command("map 2 1=99", swapped)
    assert separated[1] == {0: 1, 1: 99}


def test_global_label_capacity_checked_before_export():
    mappings = [{index: index for index in range(255)}, {0: 0}]
    with pytest.raises(ValueError, match="255"):
        qualitative.parse_mapping_command("map 2 0=255", mappings)


def test_review_rewrites_masks_and_records_until_confirmed(tmp_path, monkeypatch):
    app = make_outputs(tmp_path / "source", frame_count=2)
    chunks = [(0, 1, app.output_dir), (1, 2, app.output_dir)]
    output_dir = tmp_path / "merged"
    replay_states = []

    def replay(directory, width):
        replay_states.append((read_json(directory), read_masks(directory)))

    commands = iter(["map 2 0=7 1=0", "replay", "ok"])
    monkeypatch.setattr(qualitative, "replay_merged_outputs", replay)
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert qualitative.review_chunk_outputs(
        chunks,
        output_dir,
        app.video_path,
        app.video_info,
        "hand",
        "sam3",
        1,
        app.frame_dir,
        960,
    )
    assert len(replay_states) == 2
    assert replay_states[0][0]["status"] == "unconfirmed"
    assert replay_states[0][1][1][40, 10] == 1
    assert replay_states[1][1][1][40, 10] == 3  # temporary global 7 -> label 3
    assert replay_states[1][1][1][40, 40] == 1  # local 1 -> global 0
    metadata = read_json(output_dir)
    assert metadata["status"] == "success"
    assert metadata["review_confirmed"] is True
    assert metadata["object_id_to_label"] == {"0": 1, "1": 2, "2": 3}
    assert metadata["chunks"][1]["local_object_id_to_global_object_id"] == {
        "0": 2,
        "1": 0,
    }
    assert metadata["outputs"]["label_scope"] == "video"
    interactions = read_json(output_dir, "interactions.json")
    point = interactions["confirmed_points"][1]
    assert (point["original_obj_id"], point["local_obj_id"], point["obj_id"]) == (
        2,
        1,
        0,
    )
    assert point["frame_index"] == 1
    assert point["chunk_index"] == 2
    assert interactions["events"][1]["obj_id"] is None


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_interrupted_review_retains_unconfirmed_mapping(
    tmp_path, monkeypatch, exception
):
    app = make_outputs(tmp_path / "source")
    output_dir = tmp_path / "merged"
    commands = iter(["map 1 1=8"])

    def interrupted_input(_):
        try:
            return next(commands)
        except StopIteration:
            raise exception()

    monkeypatch.setattr(qualitative, "replay_merged_outputs", lambda *_: None)
    monkeypatch.setattr("builtins.input", interrupted_input)
    assert not qualitative.review_chunk_outputs(
        [(0, 2, app.output_dir)],
        output_dir,
        app.video_path,
        app.video_info,
        "hand",
        "sam3",
        2,
        app.frame_dir,
        960,
    )
    metadata = read_json(output_dir)
    assert metadata["status"] == "unconfirmed"
    assert metadata["review_confirmed"] is False
    assert metadata["chunks"][0]["local_object_id_to_global_object_id"] == {
        "0": 0,
        "1": 8,
    }
    assert len(read_masks(output_dir)) == 2

from pathlib import Path

import pytest
import torch

from scripts import process_learned_prompt_videos as processor
from scripts.common import compare_gt_masks as evaluation


def test_list_only_does_not_require_learned_prompt(tmp_path: Path, capsys) -> None:
    input_root = tmp_path / "seq"
    input_root.mkdir()
    (input_root / "color.mp4").touch()

    assert (
        processor.main(
            [
                "--input-root",
                str(input_root),
                "--output-root",
                str(tmp_path / "out"),
                "--list-only",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "color.mp4"


def test_read_target_id_and_cli_mismatch(tmp_path: Path) -> None:
    prompt_path = tmp_path / "best.pt"
    torch.save(
        {"format": "sam3_learned_prompt_v1", "target_id": "left_hand"},
        prompt_path,
    )
    assert processor.read_target_id(prompt_path) == "left_hand"

    input_root = tmp_path / "seq"
    input_root.mkdir()
    (input_root / "color.mp4").touch()
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"ckpt")

    assert (
        processor.main(
            [
                "--input-root",
                str(input_root),
                "--output-root",
                str(tmp_path / "out"),
                "--learned-prompt",
                str(prompt_path),
                "--checkpoint",
                str(checkpoint),
                "--target-id",
                "right_hand",
            ]
        )
        == 2
    )


def test_parser_rejects_unknown_direction() -> None:
    parser = processor.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--learned-prompt", "best.pt", "--direction", "sideways"])


def test_rgb_name_and_invalid_options(tmp_path: Path, capsys) -> None:
    (tmp_path / "rgb.mkv").touch()
    (tmp_path / "color.mp4").touch()
    assert (
        processor.main(
            ["--input-root", str(tmp_path), "--rgb-name", "rgb.mkv", "--list-only"]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "rgb.mkv"
    for option, value in (
        ("--rgb-name", "../rgb.mkv"),
        ("--gt-dir-name", "/tmp"),
        ("--gt-mask-name", "*.mkv"),
        ("--compare-skip-frames", "-1"),
    ):
        with pytest.raises(SystemExit):
            processor.build_parser().parse_args([option, value])


def write_video(path, frames, fps=30, color=False):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height), color
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def comparison_fixture(tmp_path):
    import numpy as np

    rgb = tmp_path / "input" / "rgb.mkv"
    gt = rgb.parent / "truth" / "hand.mkv"
    output = tmp_path / "out"
    empty = np.zeros((48, 64), dtype=np.uint8)
    full = np.ones_like(empty)
    predictions = [empty, full, full, full, empty, empty, full]
    truth = [empty, full, full, empty, empty, empty, full]
    write_video(rgb, [np.zeros((48, 64, 3), dtype=np.uint8)] * 7, color=True)
    write_video(gt, truth)
    write_video(output / "masks.mkv", predictions)
    return rgb, gt, output, predictions


def test_comparison_sampling_metrics_and_frame_limit(tmp_path: Path) -> None:
    import csv
    import json

    from scripts.common.compare_gt_masks import compare_masks, metric_summary
    from scripts.common.video_utils import probe_video

    rgb, gt, output, predictions = comparison_fixture(tmp_path)
    result = compare_masks(rgb, gt, output, None, 2)
    assert [row["frame_index"] for row in result["rows"]] == [0, 3, 6]
    assert [row["iou"] for row in result["rows"]] == [1, 0, 1]
    assert result["summary"]["mean_iou"] == pytest.approx(2 / 3)
    assert result["summary"]["mean_dice"] == pytest.approx(2 / 3)
    assert result["summary"]["pixel_iou"] == 0.5
    assert result["summary"]["pixel_dice"] == pytest.approx(2 / 3)
    info = probe_video(output / "comparison.mp4")
    assert (info["frame_count"], info["fps"], info["width"]) == (3, 10, 192)
    with (output / "gt_metrics.csv").open() as handle:
        assert [int(row["frame_index"]) for row in csv.DictReader(handle)] == [0, 3, 6]
    assert json.loads((output / "gt_metrics.json").read_text())["status"] == "success"
    write_video(output / "masks.mkv", predictions[:4])
    limited = compare_masks(rgb, gt, output, 4, 2)
    assert len(limited["rows"]) == 2
    assert metric_summary([])["mean_iou"] is None


def test_comparison_includes_detector_column_and_metrics(tmp_path: Path) -> None:
    import json

    import numpy as np

    from scripts.common.compare_gt_masks import compare_masks
    from scripts.common.video_utils import probe_video

    rgb, gt, output, _ = comparison_fixture(tmp_path)
    empty = np.zeros((48, 64), dtype=np.uint8)
    full = np.ones_like(empty)
    write_video(
        output / "detector_masks.mkv",
        [full, empty, empty, full, empty, empty, empty],
    )
    result = compare_masks(
        rgb, gt, output, None, 2, detector_path=output / "detector_masks.mkv"
    )
    assert [row["detector_iou"] for row in result["rows"]] == [0, 0, 0]
    assert result["summary"]["detector_mean_iou"] == 0
    assert probe_video(output / "comparison.mp4")["width"] == 256
    metrics = json.loads((output / "gt_metrics.json").read_text())
    assert metrics["detector_video"].endswith("detector_masks.mkv")


@pytest.mark.parametrize(
    "problem", ["missing", "count", "fps", "dimensions", "channels"]
)
def test_invalid_gt(tmp_path: Path, problem: str) -> None:
    import numpy as np

    from scripts.common.compare_gt_masks import compare_masks

    rgb, gt, output, _ = comparison_fixture(tmp_path)
    if problem == "missing":
        gt.unlink()
    else:
        shape = (24, 32) if problem == "dimensions" else (48, 64)
        frame = np.ones(shape, dtype=np.uint8)
        if problem == "channels":
            frame = np.stack([frame, frame * 0, frame], axis=-1)
        write_video(
            gt,
            [frame] * (6 if problem == "count" else 7),
            fps=15 if problem == "fps" else 30,
            color=problem == "channels",
        )
    with pytest.raises((ValueError, RuntimeError)):
        compare_masks(rgb, gt, output, None, 0)
    assert not (output / "gt_metrics.json").exists()
    assert not (output / "comparison.mp4").exists()


def test_cached_cli_comparison_and_failure_continuation(
    tmp_path: Path, monkeypatch
) -> None:
    import json
    import shutil

    rgb, gt, output, _ = comparison_fixture(tmp_path)
    prompt = tmp_path / "prompt.pt"
    checkpoint = tmp_path / "sam3.pt"
    prompt.touch()
    checkpoint.touch()
    monkeypatch.setattr(processor, "read_target_id", lambda path: "left_hand")
    monkeypatch.setattr(
        processor, "require_cuda", lambda *args: pytest.fail("loaded CUDA")
    )
    for direction in ("forward", "backward"):
        destination = output / direction
        destination.mkdir()
        shutil.copyfile(output / "masks.mkv", destination / "masks.mkv")
        (destination / "result.mp4").touch()
        (destination / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "input_video": str(rgb),
                    "prompt": "left_hand",
                    "model_version": "sam3",
                    "frames_processed": 7,
                    "propagation_direction": direction,
                }
            )
        )
    args = [
        "--input-root",
        str(rgb.parent),
        "--output-root",
        str(output),
        "--rgb-name",
        "rgb.mkv",
        "--learned-prompt",
        str(prompt),
        "--checkpoint",
        str(checkpoint),
        "--compare-gt",
        "--gt-dir-name",
        "truth",
        "--gt-mask-name",
        "hand.mkv",
        "--compare-skip-frames",
        "2",
        "--direction",
        "both",
    ]
    assert processor.main(args) == 0
    batch = json.loads((output / "gt_summary.json").read_text())
    assert batch["directions"]["forward"]["frames_evaluated"] == 3
    assert batch["directions"]["backward"]["pixel_iou"] == 0.5
    original_compare = evaluation.compare_masks

    def fail_forward(*args):
        if args[2].name == "forward":
            raise ValueError("broken GT")
        return original_compare(*args)

    monkeypatch.setattr(evaluation, "compare_masks", fail_forward)
    assert processor.main(args) == 1
    batch = json.loads((output / "gt_summary.json").read_text())
    assert [seq["status"] for seq in batch["sequences"]] == ["failed", "success"]
    assert not (output / "forward" / "comparison.mp4").exists()
    assert (output / "backward" / "comparison.mp4").exists()


def test_inference_then_comparison(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys
    from types import SimpleNamespace

    from scripts.process_dataset_videos import empty_outputs

    rgb, gt, output, _ = comparison_fixture(tmp_path)
    gt.rename(gt.with_name("left_hand.mkv"))
    prompt = tmp_path / "prompt.pt"
    checkpoint = tmp_path / "sam3.pt"
    prompt.touch()
    checkpoint.touch()
    requests = []
    builds = []

    class Predictor:
        def handle_request(self, request):
            requests.append(request)
            return {"session_id": "test"}

        def handle_stream_request(self, request):
            requests.append(request)
            for index in range(7):
                yield {"frame_index": index, "outputs": empty_outputs()}

        def shutdown(self):
            requests.append({"type": "shutdown"})

    def build(**kwargs):
        builds.append(kwargs)
        return Predictor()

    monkeypatch.setitem(
        sys.modules,
        "sam3.model_builder",
        SimpleNamespace(build_sam3_video_predictor=build),
    )
    monkeypatch.setattr(processor, "read_target_id", lambda path: "left_hand")
    monkeypatch.setattr(processor, "require_cuda", lambda *args: True)
    assert (
        processor.main(
            [
                "--input-root",
                str(rgb.parent),
                "--output-root",
                str(output),
                "--rgb-name",
                "rgb.mkv",
                "--learned-prompt",
                str(prompt),
                "--checkpoint",
                str(checkpoint),
                "--compare-gt",
                "--gt-mask-name",
                "left_hand.mkv",
                "--gt-dir-name",
                "truth",
            ]
        )
        == 0
    )
    assert len(builds) == 1
    assert any(request["type"] == "add_learned_prompt" for request in requests)
    assert requests[-1]["type"] == "shutdown"
    summary = json.loads((output / "gt_metrics.json").read_text())
    assert summary["frames_evaluated"] == 7
    assert summary["mean_iou"] == pytest.approx(4 / 7)


def test_comparison_geometry_only_on_rgb_and_sampled_frames(tmp_path):
    import cv2

    from scripts.common.compare_gt_masks import compare_masks

    rgb, gt, output, _ = comparison_fixture(tmp_path)
    baseline = compare_masks(rgb, gt, output, None, 2)
    result = compare_masks(
        rgb,
        gt,
        output,
        None,
        2,
        geometry_prompts={
            0: {"points": [[0.5, 0.75]]},
            1: {"points": [[0.5, 0.75]]},
            6: {"boxes": [[0.25, 2 / 3, 0.5, 1 / 6]]},
        },
    )
    assert result["rows"] == baseline["rows"]
    capture = cv2.VideoCapture(str(output / "comparison.mp4"))
    try:
        for index in (0, 3, 6):
            ok, frame = capture.read()
            assert ok
            if index == 0:
                assert frame[36, 32, 1] > 100 and frame[36, 32, 2] > 100
            elif index == 3:
                assert frame[34:42, 24:40].max() < 40
            else:
                assert frame[32, 32, 0] > 100 and frame[32, 32, 1] > 100
            for offset in (64, 128):
                assert frame[36, offset + 32, 0] < 80
                assert frame[36, offset + 32, 2] < 80
    finally:
        capture.release()

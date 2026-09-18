"""Shared GT CLI contract and inference/cache integration without model weights."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import (
    process_dataset_videos as dataset,
    process_learned_prompt_videos as learned,
    process_mano_prompt_videos as mano,
)
from scripts.common import compare_gt_masks as evaluation


@pytest.mark.parametrize(
    "processor,required",
    [
        (dataset, ["--text-prompt", "hand"]),
        (learned, []),
        (mano, ["--text-prompt", "hand", "--hand-side", "left"]),
    ],
)
def test_shared_gt_arguments(processor, required, capsys):
    parser = processor.build_parser()
    defaults = parser.parse_args(required)
    if processor in (dataset, mano):
        assert defaults.prompt == "hand"
        with pytest.raises(SystemExit):
            parser.parse_args([*required, "--prompt", "hand"])
    assert (
        defaults.compare_gt,
        defaults.gt_dir_name,
        defaults.gt_mask_name,
        defaults.compare_skip_frames,
    ) == (False, "masks_sam3", None, 0)
    if processor is mano:
        assert defaults.save_detector is False
    else:
        assert not hasattr(defaults, "save_detector")
    with pytest.raises(SystemExit) as error:
        processor.main([*required, "--compare-gt", "--list-only"])
    assert error.value.code == 2
    assert "--gt-mask-name is required" in capsys.readouterr().err
    args = parser.parse_args(
        [
            *required,
            "--compare-gt",
            "--gt-mask-name",
            "custom.mkv",
            "--gt-dir-name",
            "truth",
            "--compare-skip-frames",
            "2",
        ]
    )
    evaluation.validate_gt_arguments(parser, args)
    assert (args.gt_mask_name, args.gt_dir_name, args.compare_skip_frames) == (
        "custom.mkv",
        "truth",
        2,
    )
    for option, value in (
        ("--gt-dir-name", "../truth"),
        ("--gt-mask-name", "*.mkv"),
        ("--gt-mask-name", ""),
        ("--compare-skip-frames", "-1"),
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*required, option, value])


def write_video(path, frames, color=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v" if path.suffix == ".mp4" else "FFV1"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), 12, (64, 48), color
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.parametrize(
    "processor,version", [(dataset, "sam3"), (dataset, "sam3.1"), (mano, "sam3")]
)
def test_inference_cached_comparison_and_failure(
    tmp_path, monkeypatch, processor, version
):
    root = tmp_path / "input"
    output = tmp_path / "out"
    video = root / "color.mp4"
    gt = root / "truth" / "custom.mkv"
    write_video(video, [np.zeros((48, 64, 3), dtype=np.uint8)] * 4, color=True)
    # Empty GT on frame 0; foreground on sampled frame 3.
    write_video(
        gt,
        [np.zeros((48, 64), dtype=np.uint8)] + [np.ones((48, 64), dtype=np.uint8)] * 3,
    )
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    builds = []
    requests = []
    shutdowns = []

    class Predictor:
        def handle_request(self, request):
            requests.append(request)
            return {"session_id": "test"}

        def handle_stream_request(self, request):
            indices = (
                range(4)
                if request["propagation_direction"] == "forward"
                else range(3, -1, -1)
            )
            for index in indices:
                yield {"frame_index": index, "outputs": dataset.empty_outputs()}

        def shutdown(self):
            shutdowns.append(True)

    def build(**kwargs):
        builds.append(kwargs)
        return Predictor()

    monkeypatch.setitem(
        sys.modules, "sam3", SimpleNamespace(build_sam3_predictor=build)
    )
    monkeypatch.setattr(dataset, "require_cuda", lambda *args: True)
    args = [
        "--input-root",
        str(root),
        "--output-root",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--text-prompt",
        "hand",
        "--version",
        version,
        "--direction",
        "both",
        "--compare-gt",
        "--gt-dir-name",
        "truth",
        "--gt-mask-name",
        "custom.mkv",
        "--compare-skip-frames",
        "2",
    ]
    if processor is mano:
        args += ["--hand-side", "left"]
        monkeypatch.setattr(mano, "find_mano", lambda *args: [root / "geometry.npz"])
        monkeypatch.setattr(
            mano,
            "load_geometry",
            lambda *args: (
                {0: {"points": [[0.5, 0.5]]}},
                {},
                {"missing_frames": [], "unusable_prompt_frames": []},
            ),
        )
    assert processor.main(args) == 0
    assert len(builds) == len(shutdowns) == 1
    assert builds[0]["version"] == version
    expected_type = "add_geometry_prompts" if processor is mano else "add_prompt"
    assert sum(r["type"] == expected_type for r in requests) == 2
    summary = json.loads((output / "gt_summary.json").read_text())
    for direction in ("forward", "backward"):
        metrics = summary["directions"][direction]
        assert metrics["frames_evaluated"] == 2
        assert metrics["mean_iou"] == metrics["mean_dice"] == 0.5
        assert (output / direction / "gt_metrics.csv").is_file()
        assert (
            dataset.probe_video(output / direction / "comparison.mp4")["frame_count"]
            == 2
        )
    masks_before = {
        d: (output / d / "masks.mkv").read_bytes() for d in ("forward", "backward")
    }
    monkeypatch.setattr(
        dataset,
        "require_cuda",
        lambda *args: pytest.fail("loaded CUDA for cached predictions"),
    )
    assert processor.main(args) == 0
    assert len(builds) == 1
    original_compare = evaluation.compare_masks

    def fail_forward(*args):
        if args[2].name == "forward":
            raise ValueError("broken GT")
        return original_compare(*args)

    monkeypatch.setattr(evaluation, "compare_masks", fail_forward)
    assert processor.main(args) == 1
    summary = json.loads((output / "gt_summary.json").read_text())
    assert summary["status"] == "failed"
    assert [row["status"] for row in summary["sequences"]] == ["failed", "success"]
    assert summary["directions"]["forward"]["mean_iou"] is None
    assert not (output / "forward" / "comparison.mp4").exists()
    assert not (output / "forward" / "gt_metrics.csv").exists()
    assert (output / "backward" / "comparison.mp4").is_file()
    for direction, data in masks_before.items():
        assert (output / direction / "masks.mkv").read_bytes() == data
    if processor is mano:
        records = json.loads((output / "batch_summary.json").read_text())["results"]
        assert [row["status"] for row in records] == ["failed", "skipped"]

        def missing_mano(*args):
            raise FileNotFoundError("missing MANO")

        monkeypatch.setattr(mano, "find_mano", missing_mano)
        assert processor.main(args) == 0
        summary = json.loads((output / "gt_summary.json").read_text())
        assert summary["sequences"] == []
        assert (output / "backward" / "comparison.mp4").is_file()
        records = json.loads((output / "batch_summary.json").read_text())["results"]
        assert [row["status"] for row in records] == ["skipped"]


def test_save_detector_cli_metrics_and_incomplete_cache(tmp_path, monkeypatch):
    root = tmp_path / "input"
    output = tmp_path / "out"
    video = root / "color.mp4"
    gt = root / "truth" / "custom.mkv"
    write_video(video, [np.zeros((48, 64, 3), dtype=np.uint8)] * 4, color=True)
    write_video(
        gt,
        [np.zeros((48, 64), dtype=np.uint8)] + [np.ones((48, 64), dtype=np.uint8)] * 3,
    )
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    builds = []

    class Predictor:
        def handle_request(self, request):
            return {
                "session_id": "test",
                "outputs": {"out_detector_mask": np.ones((48, 64), dtype=bool)},
            }

        def handle_stream_request(self, request):
            mask = np.ones((48, 64), dtype=bool)
            indices = (
                range(4)
                if request["propagation_direction"] == "forward"
                else range(3, -1, -1)
            )
            for index in indices:
                yield {
                    "frame_index": index,
                    "outputs": {
                        **dataset.empty_outputs(),
                        "out_detector_mask": mask,
                    },
                }

        def shutdown(self):
            return None

    def build(**kwargs):
        builds.append(kwargs)
        return Predictor()

    monkeypatch.setitem(
        sys.modules, "sam3", SimpleNamespace(build_sam3_predictor=build)
    )
    monkeypatch.setattr(dataset, "require_cuda", lambda *args: True)
    monkeypatch.setattr(mano, "find_mano", lambda *args: [root / "geometry.npz"])
    monkeypatch.setattr(
        mano,
        "load_geometry",
        lambda *args: (
            {0: {"points": [[0.5, 0.5]]}},
            {},
            {"missing_frames": [], "unusable_prompt_frames": []},
        ),
    )
    base = [
        "--input-root",
        str(root),
        "--output-root",
        str(output),
        "--checkpoint",
        str(checkpoint),
        "--text-prompt",
        "hand",
        "--hand-side",
        "left",
        "--compare-gt",
        "--gt-dir-name",
        "truth",
        "--gt-mask-name",
        "custom.mkv",
        "--compare-skip-frames",
        "2",
    ]
    assert mano.main(base) == 0
    assert len(builds) == 1
    assert not (output / "detector_masks.mkv").exists()
    assert dataset.probe_video(output / "comparison.mp4")["width"] == 192
    assert mano.main([*base, "--save-detector"]) == 0
    assert len(builds) == 2
    assert (output / "detector_masks.mkv").is_file()
    info = dataset.probe_video(output / "comparison.mp4")
    assert (info["width"], info["frame_count"]) == (256, 2)
    summary = json.loads((output / "gt_metrics.json").read_text())
    assert summary["detector_mean_iou"] == 0.5
    assert mano.main([*base, "--save-detector"]) == 0
    assert len(builds) == 2


@pytest.mark.parametrize(
    "processor,required",
    [
        (dataset, ["--text-prompt", "hand"]),
        (learned, []),
        (mano, ["--text-prompt", "", "--hand-side", "left"]),
    ],
)
def test_shared_io_discovery_and_validation(
    processor, required, tmp_path, monkeypatch, capsys
):
    root = tmp_path / "input"
    nested = root / "nested"
    nested.mkdir(parents=True)
    for directory in (root, nested):
        (directory / "rgb.mkv").touch()
        (directory / "color.mp4").touch()
    output = tmp_path / "output"
    argv = [
        *required,
        "--input-root",
        str(root),
        "--output-root",
        str(output),
        "--rgb-name",
        "rgb.mkv",
        "--list-only",
    ]
    checked = []

    def find_mano(video, side, dir_name):
        checked.append(video)
        return [video.with_suffix(".npz")]

    monkeypatch.setattr(mano, "find_mano", find_mano)
    monkeypatch.setattr(dataset, "probe_video", lambda path: {})
    monkeypatch.setattr(
        mano,
        "load_geometry",
        lambda *args: ({}, {}, {"missing_frames": [], "unusable_prompt_frames": []}),
    )
    assert processor.main(argv) == 0
    listing = capsys.readouterr().out
    assert "nested/rgb.mkv" in listing and "color.mp4" not in listing
    assert len(listing.splitlines()) == 2
    assert not output.exists()
    if processor is mano:
        assert checked == [nested / "rgb.mkv", root / "rgb.mkv"]
    assert processor.main([*argv, "--max-sequences", "1"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1
    for same_root in (root, tmp_path / "alias"):
        if same_root != root:
            same_root.symlink_to(root, target_is_directory=True)
        assert processor.main([*argv, "--output-root", str(same_root)]) == 2
    parser = processor.build_parser()
    for invalid in ("../rgb.mkv", "*.mp4"):
        with pytest.raises(SystemExit):
            parser.parse_args([*argv, "--rgb-name", invalid])
    help_text = parser.format_help()
    for group in (
        "Data input/output",
        "Batch execution",
        "Model loading",
        "GT evaluation",
    ):
        assert group in help_text
    assert parser.parse_args(required).output_root == (
        "~/sam3_learned_outputs" if processor is learned else "~/sam3_outputs"
    )
    if processor is mano:
        assert "Also validate MANO files" in help_text

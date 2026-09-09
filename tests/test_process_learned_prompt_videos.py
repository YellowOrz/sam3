from pathlib import Path

import pytest
import torch

from scripts import process_learned_prompt_videos as processor


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

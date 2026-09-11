import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_automatic_hands import (
    build_automatic_hands_predictor,
    draw_outputs,
    merge_instance_masks,
    parse_args,
    propagate_one_class,
)
from sam3.model.learnable_text_encoder import LearnableClassTextEncoder


class AutomaticHandsTest(unittest.TestCase):
    def test_merge_accepts_predictor_nchw_masks(self):
        masks = np.zeros((2, 1, 4, 5), dtype=bool)
        masks[0, 0, 1, 1] = True
        masks[1, 0, 2, 3] = True

        merged = merge_instance_masks(masks)

        self.assertEqual(merged.shape, (4, 5))
        self.assertTrue(merged[1, 1])
        self.assertTrue(merged[2, 3])

    def test_merge_returns_none_for_no_instances(self):
        self.assertIsNone(merge_instance_masks([]))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
    def test_draw_outputs_preserves_frame_shape(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        mask = np.zeros((1, 1, 4, 5), dtype=bool)
        mask[0, 0, 1:3, 1:4] = True
        outputs = {
            "left_hand": {"current": {"out_binary_masks": mask}},
            "right_hand": {"current": {"out_binary_masks": []}},
        }

        rendered = draw_outputs(frame, outputs)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(rendered.sum(), 0)

    def test_no_token_checkpoint_preserves_default_k_one(self):
        with patch("run_automatic_hands.build_sam3_predictor") as builder:
            predictor = build_automatic_hands_predictor("base.pt")
        self.assertIs(predictor, builder.return_value)
        self.assertEqual(builder.call_args.kwargs["tokens_per_class"], 1)
        self.assertEqual(builder.call_args.kwargs["checkpoint_path"], "base.pt")

    def test_explicit_k_without_token_checkpoint_reaches_builder(self):
        with patch("run_automatic_hands.build_sam3_predictor") as builder:
            build_automatic_hands_predictor("full-k4.pt", tokens_per_class=4)
        self.assertEqual(builder.call_args.kwargs["tokens_per_class"], 4)

    def test_token_checkpoint_is_overlaid_on_video_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch1.pt"
            tokens = torch.randn(2, 4, 256)
            torch.save(
                {"class_names": ["left_hand", "right_hand"], "class_tokens": tokens},
                checkpoint,
            )
            encoder = LearnableClassTextEncoder(tokens_per_class=4)
            predictor = SimpleNamespace(
                model=SimpleNamespace(
                    detector=SimpleNamespace(
                        backbone=SimpleNamespace(language_backbone=encoder)
                    )
                ),
                shutdown=Mock(),
                world_size=1,
            )
            with patch(
                "run_automatic_hands.build_sam3_predictor", return_value=predictor
            ) as builder:
                result = build_automatic_hands_predictor(
                    "base.pt", token_checkpoint=checkpoint
                )
            self.assertIs(result, predictor)
            self.assertEqual(builder.call_args.kwargs["tokens_per_class"], 4)
            self.assertTrue(torch.equal(encoder.class_tokens, tokens))
            predictor.shutdown.assert_not_called()

            with patch("run_automatic_hands.build_sam3_predictor") as builder:
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    build_automatic_hands_predictor(
                        "base.pt", token_checkpoint=checkpoint, tokens_per_class=1
                    )
                builder.assert_not_called()

    def test_invalid_explicit_k_is_rejected_before_building(self):
        with patch("run_automatic_hands.build_sam3_predictor") as builder:
            for tokens_per_class in (0, -1):
                with self.subTest(tokens_per_class=tokens_per_class):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        build_automatic_hands_predictor(
                            "base.pt", tokens_per_class=tokens_per_class
                        )
            builder.assert_not_called()

    def test_overlay_failure_shuts_down_predictor(self):
        predictor = SimpleNamespace(model=None, shutdown=Mock(), world_size=1)
        with (
            patch(
                "run_automatic_hands.load_class_token_checkpoint",
                return_value=torch.zeros(2, 4, 256),
            ),
            patch("run_automatic_hands.build_sam3_predictor", return_value=predictor),
        ):
            with self.assertRaises(AttributeError):
                build_automatic_hands_predictor("base.pt", token_checkpoint="tokens.pt")
        predictor.shutdown.assert_called_once_with()

    def test_session_closes_after_prompt_and_propagation_failures(self):
        for failure_stage in ("add_prompt", "propagate_in_video"):
            with self.subTest(failure_stage=failure_stage):
                predictor = Mock()

                def handle_request(*, request):
                    if request["type"] == "start_session":
                        return {"session_id": "test-session"}
                    if request["type"] == failure_stage:
                        raise RuntimeError("test failure")

                predictor.handle_request.side_effect = handle_request

                def propagate(*, request):
                    yield {"frame_index": 0, "outputs": {}}
                    raise RuntimeError("test failure")

                predictor.handle_stream_request.side_effect = propagate
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    propagate_one_class(predictor, "input.mp4", "left_hand")
                predictor.handle_request.assert_called_with(
                    request={"type": "close_session", "session_id": "test-session"}
                )

    def test_cli_default_does_not_override_checkpoint_k(self):
        args = parse_args(
            [
                "--video",
                "input.mp4",
                "--out",
                "out.mp4",
                "--checkpoint",
                "base.pt",
                "--token-checkpoint",
                "epoch1.pt",
            ]
        )
        self.assertIsNone(args.tokens_per_class)
        self.assertEqual(args.token_checkpoint, "epoch1.pt")


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import torch

from sam3.model.learnable_text_encoder import (
    LearnableClassTextEncoder,
    freeze_for_learnable_class_tokens,
)
from sam3.model.sam3_video_predictor import Sam3VideoPredictor
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model_builder import _create_text_encoder


class LearnableClassTextEncoderTest(unittest.TestCase):
    def test_class_prompts_keep_the_sam3_text_contract(self):
        for tokens_per_class in (1, 4, 8):
            with self.subTest(tokens_per_class=tokens_per_class):
                encoder = LearnableClassTextEncoder(tokens_per_class=tokens_per_class)
                padding_mask, features, raw_features = encoder(
                    ["left_hand", "right_hand"], device=torch.device("cpu")
                )
                self.assertEqual(features.shape, (tokens_per_class, 2, 256))
                self.assertEqual(raw_features.shape, features.shape)
                self.assertEqual(padding_mask.shape, (2, tokens_per_class))
                self.assertEqual(padding_mask.dtype, torch.bool)
                self.assertFalse(padding_mask.any())

    def test_each_class_selects_independent_parameters(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=1)
        _, features, _ = encoder(
            ["left_hand", "right_hand"], device=torch.device("cpu")
        )
        self.assertFalse(torch.equal(features[:, 0], features[:, 1]))

    def test_space_and_underscore_class_names_select_the_same_token(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=1)
        _, features, _ = encoder(
            ["left hand", "left_hand"], device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(features[:, 0], features[:, 1]))

    def test_video_placeholder_names_are_zero_and_do_not_select_a_class(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=1)
        _, features, _ = encoder(
            ["left_hand", "visual", "<text placeholder>"],
            device=torch.device("cpu"),
        )
        self.assertGreater(features[:, 0].abs().sum(), 0)
        self.assertEqual(features[:, 1:].abs().sum(), 0)

    def test_unknown_class_fails_with_allowed_names(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=1)
        with self.assertRaisesRegex(ValueError, "left_hand, right_hand"):
            encoder(["cup"], device=torch.device("cpu"))

    def test_only_the_selected_class_receives_gradient(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=1)
        _, features, _ = encoder(["left_hand"], device=torch.device("cpu"))
        features.square().sum().backward()
        gradient = encoder.class_tokens.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient[0].abs().sum(), 0)
        self.assertEqual(gradient[1:].abs().sum(), 0)

    def test_vl_backbone_exposes_the_existing_language_output_keys(self):
        backbone = SAM3VLBackbone(
            visual=None, text=LearnableClassTextEncoder(tokens_per_class=1)
        )
        output = backbone.forward_text(
            ["left_hand", "right_hand"], device=torch.device("cpu")
        )
        self.assertEqual(
            set(output), {"language_features", "language_mask", "language_embeds"}
        )
        self.assertEqual(output["language_features"].shape, (1, 2, 256))
        self.assertEqual(output["language_mask"].shape, (2, 1))

    def test_builder_selects_learnable_tokens_and_configures_length(self):
        encoder = _create_text_encoder(
            bpe_path="not-used-in-learnable-mode",
            text_encoder_type="learnable_class",
            tokens_per_class=4,
        )
        self.assertIsInstance(encoder, LearnableClassTextEncoder)
        self.assertEqual(encoder.class_tokens.shape, (2, 4, 256))

    def test_builder_rejects_unknown_text_encoder_type(self):
        with self.assertRaisesRegex(ValueError, "ve.*learnable_class"):
            _create_text_encoder(bpe_path="not-used", text_encoder_type="unknown")

    def test_freeze_helper_leaves_only_class_tokens_trainable(self):
        model = torch.nn.ModuleDict(
            {
                "backbone": torch.nn.Linear(4, 4),
                "language": LearnableClassTextEncoder(tokens_per_class=1),
                "head": torch.nn.Linear(4, 1),
            }
        )
        trainable_names = freeze_for_learnable_class_tokens(model)
        actual_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable_names, {"language.class_tokens"})
        self.assertEqual(actual_names, trainable_names)

    def test_gradient_reaches_tokens_through_a_frozen_downstream_module(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=1)
        downstream = torch.nn.Linear(256, 1)
        for parameter in downstream.parameters():
            parameter.requires_grad = False
        _, features, _ = encoder(["left_hand"], device=torch.device("cpu"))
        downstream(features).mean().square().backward()
        self.assertIsNotNone(encoder.class_tokens.grad)
        self.assertGreater(encoder.class_tokens.grad[0].abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in downstream.parameters()))

    def test_video_predictor_exposes_class_token_options(self):
        parameters = inspect.signature(Sam3VideoPredictor.__init__).parameters
        self.assertIn("text_encoder_type", parameters)
        self.assertIn("tokens_per_class", parameters)


if __name__ == "__main__":
    unittest.main()

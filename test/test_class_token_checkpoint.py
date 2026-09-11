import tempfile
import unittest
from pathlib import Path

import torch

from sam3.model.class_token_checkpoint import (
    copy_class_tokens,
    load_class_token_checkpoint,
)
from sam3.model.learnable_text_encoder import LearnableClassTextEncoder


class ClassTokenCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tokens.pt"

    def save(self, **overrides):
        state = {
            "class_names": ["left_hand", "right_hand"],
            "class_tokens": torch.randn(2, 4, 256),
            "tokens_per_class": 4,
            "d_model": 256,
        }
        state.update(overrides)
        torch.save(state, self.path)
        return state

    def test_infers_k_and_overlays_the_trained_values_without_grad(self):
        state = self.save()
        tokens = load_class_token_checkpoint(self.path)
        encoder = LearnableClassTextEncoder(tokens_per_class=tokens.shape[1])
        copy_class_tokens(encoder, tokens)
        self.assertEqual(tokens.device.type, "cpu")
        self.assertTrue(torch.equal(encoder.class_tokens, state["class_tokens"]))
        self.assertIsNone(encoder.class_tokens.grad)

    def test_tensor_can_supply_k_when_optional_metadata_is_absent(self):
        torch.save(
            {
                "class_names": ["left_hand", "right_hand"],
                "class_tokens": torch.zeros(2, 8, 256),
            },
            self.path,
        )
        self.assertEqual(load_class_token_checkpoint(self.path).shape[1], 8)

    def test_rejects_missing_or_reversed_class_names(self):
        for names in (None, ["right_hand", "left_hand"], ["right_hand"]):
            with self.subTest(names=names):
                self.save(class_names=names)
                with self.assertRaisesRegex(ValueError, "class_names"):
                    load_class_token_checkpoint(self.path)

    def test_rejects_invalid_shape_or_non_tensor_tokens(self):
        for tokens in (
            None,
            torch.zeros(2, 256),
            torch.zeros(1, 4, 256),
            torch.zeros(2, 0, 256),
            torch.zeros(2, 4, 255),
        ):
            with self.subTest(shape=getattr(tokens, "shape", None)):
                self.save(class_tokens=tokens)
                with self.assertRaisesRegex(ValueError, "shape"):
                    load_class_token_checkpoint(self.path)

    def test_rejects_non_finite_and_non_float_tokens(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.save(class_tokens=torch.full((2, 4, 256), value))
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    load_class_token_checkpoint(self.path)
        self.save(class_tokens=torch.zeros(2, 4, 256, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "floating-point"):
            load_class_token_checkpoint(self.path)

    def test_rejects_metadata_or_cli_k_conflict(self):
        for metadata in ({"tokens_per_class": 1}, {"d_model": 512}):
            with self.subTest(metadata=metadata):
                self.save(**metadata)
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    load_class_token_checkpoint(self.path)
        self.save()
        with self.assertRaisesRegex(ValueError, "conflicts"):
            load_class_token_checkpoint(self.path, tokens_per_class=1)
        self.assertEqual(
            load_class_token_checkpoint(self.path, tokens_per_class=4).shape[1], 4
        )

    def test_rejects_wrong_target_encoder_and_shape(self):
        tokens = self.save()["class_tokens"]
        with self.assertRaisesRegex(ValueError, "LearnableClassTextEncoder"):
            copy_class_tokens(torch.nn.Linear(256, 256), tokens)
        with self.assertRaisesRegex(ValueError, "does not match"):
            copy_class_tokens(LearnableClassTextEncoder(tokens_per_class=1), tokens)

    def test_rejects_non_finite_dtype_conversion_before_mutating_encoder(self):
        encoder = LearnableClassTextEncoder(tokens_per_class=4).half()
        initial = encoder.class_tokens.detach().clone()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            copy_class_tokens(encoder, torch.full((2, 4, 256), 1e10))
        self.assertTrue(torch.equal(initial, encoder.class_tokens))


if __name__ == "__main__":
    unittest.main()

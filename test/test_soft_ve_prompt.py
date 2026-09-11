"""Real small VE transformer tests; no pretrained weights or GPU required."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from scripts.soft_ve_prompt import SharedInputVETextEncoder, install_shared_input_ve, set_input_ve_training_mode


class SoftVEPromptTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(312)
        tokenizer = SimpleTokenizer(bpe_path=str(Path(__file__).resolve().parents[1] /
                                                "sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
        self.ve = VETextEncoder(d_model=8, tokenizer=tokenizer, width=16, heads=2,
                                layers=1, context_length=32, use_act_checkpoint=False).eval()
        with torch.no_grad():
            for parameter in self.ve.parameters():
                parameter.normal_(std=.04)
        self.adapter = SharedInputVETextEncoder(self.ve)

    def test_zero_residual_matches_original_all_three_outputs(self):
        prompts = ["left hand", "right hand", "cup"]
        with torch.no_grad():
            expected = self.ve(prompts, device="cpu")
            actual = self.adapter(prompts, device="cpu")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(self.adapter.input_delta.shape, (2, 16))

    def test_nonhand_rows_unchanged_even_in_mixed_batch_after_update(self):
        prompts = ["cup", "left hand", "right hand", "knife"]
        with torch.no_grad():
            expected = self.ve(prompts, device="cpu")
            self.adapter.input_delta.normal_(std=.2)
            actual = self.adapter(prompts, device="cpu")
        for a, b in zip(actual, expected):
            if a.ndim == 2:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                torch.testing.assert_close(a[:, [0, 3]], b[:, [0, 3]], rtol=0, atol=0)
        self.assertGreater(float((actual[1][:, 1:3] - expected[1][:, 1:3]).abs().max()), 0)

    def test_internal_aliases_map_to_exact_natural_words(self):
        with torch.no_grad():
            expected = self.ve(["left hand", "right hand"], device="cpu")
            actual = self.adapter(["left_hand", "right_hand"], device="cpu")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_only_word_positions_modified_not_side_names_or_padding(self):
        with torch.no_grad():
            expected = self.ve(["left hand", "right hand"], device="cpu")
            self.adapter.input_delta.fill_(.1)
            actual = self.adapter(["left_hand", "right_hand"], device="cpu")
        difference = actual[2] - expected[2]
        self.assertEqual(difference[[0, *range(3, 32)]].abs().sum().item(), 0.)
        torch.testing.assert_close(difference[1:3, 0], difference[1:3, 1], rtol=1e-6, atol=1e-7)
        self.assertFalse(torch.equal(actual[2][1, 0], actual[2][1, 1]))

    def test_gradients_reach_shared_residual_through_frozen_transformer(self):
        original = {name: p.detach().clone() for name, p in self.ve.named_parameters()}
        features = self.adapter(["left_hand", "right_hand"], device="cpu")[1]
        features.square().mean().backward()
        norms = self.adapter.input_delta.grad.norm(dim=1)
        self.assertTrue(torch.isfinite(norms).all())
        self.assertTrue((norms > 0).all())
        for name, parameter in self.ve.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)

    def test_nonhand_only_bypasses_residual_and_frozen_weights(self):
        with torch.no_grad():
            self.adapter.input_delta.fill_(.2)
            actual = self.adapter(["cup"], device="cpu")
            expected = self.ve(["cup"], device="cpu")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_small_checkpoint_round_trip_and_metadata_rejection(self):
        with torch.no_grad():
            self.adapter.input_delta.fill_(.02)
        state = self.adapter.residual_state()
        self.assertNotIn("original_ve", state)
        self.assertEqual(sum(x.numel() for x in state.values() if isinstance(x, torch.Tensor)), 96)
        with torch.no_grad():
            self.adapter.input_delta.zero_()
        self.adapter.load_residual_state(state)
        torch.testing.assert_close(self.adapter.input_delta, state["input_delta"])
        for key, value in [("shared_across_sides", False), ("positions", [0, 2]), ("input_delta", torch.zeros(2, 16, dtype=torch.bfloat16))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.adapter.load_residual_state({**state, key: value})

    def test_mode_and_dtype_guards_and_freeze_helper(self):
        model = nn.Module()
        model.backbone = nn.Module()
        model.backbone.language_backbone = self.ve
        model.visual = nn.Linear(4, 4)
        adapter = install_shared_input_ve(model)
        self.assertEqual(set_input_ve_training_mode(model, train_residual=True), {"backbone.language_backbone.input_delta"})
        self.assertFalse(model.visual.weight.requires_grad)
        adapter.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            adapter(["left_hand"])
        adapter.eval()
        with torch.no_grad():
            adapter.input_delta[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            adapter(["left_hand"])

    def test_preencoded_input_passthrough_and_invalid_strings_rejected(self):
        padding = torch.zeros(1, 32, dtype=torch.bool)
        features = torch.randn(32, 1, 8)
        raw = torch.randn(1, 32, 16)
        encoded = (padding, features, {"inputs_embeds": raw})
        actual = self.adapter(encoded)
        self.assertIs(actual[0], padding)
        self.assertIs(actual[1], features)
        torch.testing.assert_close(actual[2], raw.transpose(0, 1))
        for prompts in ([], ["left hand", None]):
            with self.assertRaises(ValueError):
                self.adapter(prompts)


if __name__ == "__main__":
    unittest.main()

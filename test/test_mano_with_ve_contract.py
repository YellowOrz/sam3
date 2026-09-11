"""CPU interface tests for original VE features plus opt-in MANO geometry.

Uses the real VE/SAM3 prompt-fusion classes with a small random VE transformer,
synthetic image features and the real geometry encoder. No pretrained weights,
GPU, optimizer step, dataset quality claim or full-model equivalence is involved.
"""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sam3.model.geometry_encoders import Prompt
from sam3.model.learnable_text_encoder import LearnableClassTextEncoder
from sam3.model.mano_geometry_encoder import (
    ManoGeometryEncoder, freeze_for_mano_geometry_encoder,
)
from sam3.model.mano_prompt_adapter import ManoPrompt, attach_mano_geometry_encoder
from sam3.model.sam3_image import Sam3Image
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model_builder import _create_geometry_encoder


class ManoWithVEContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(231)
        tokenizer = SimpleTokenizer(bpe_path=str(
            Path(__file__).resolve().parents[1] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        ))
        ve = VETextEncoder(d_model=256, tokenizer=tokenizer, width=16, heads=2,
                           layers=1, context_length=32, use_act_checkpoint=False).eval()
        # VE positional/projection tensors expect checkpoint loading in production.
        # Initialize the small random test instance explicitly, never load a model.
        with torch.no_grad():
            for parameter in ve.parameters():
                parameter.normal_(std=.02)
        self.model = Sam3Image.__new__(Sam3Image)
        nn.Module.__init__(self.model)
        self.model.num_feature_levels = 1
        self.model.geometry_encoder = _create_geometry_encoder().eval()
        self.model.backbone = nn.Module()
        self.model.backbone.language_backbone = ve
        self.model.test_downstream = nn.Linear(256, 1)
        self.model.eval()
        with torch.no_grad():
            padding, features, embeddings = ve(["left hand", "right hand"], device="cpu")
        self.backbone = {
            "language_mask": padding, "language_features": features,
            "language_embeds": embeddings,
            "backbone_fpn": [torch.randn(1, 256, 2, 2)],
            "vision_pos_enc": [torch.zeros(1, 256, 2, 2)],
        }
        self.stage = SimpleNamespace(img_ids=torch.tensor([0, 0]),
                                     text_ids=torch.tensor([0, 1]))
        self.plain = Prompt(point_embeddings=torch.empty(0, 2, 2))

    def encode(self, prompt):
        # The original CPU box encoder pins a scale tensor even for zero boxes.
        # Bypass only this transport hint; actual prompt/geometry operations run.
        with patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            return self.model._encode_prompt(self.backbone, self.stage, prompt)

    @staticmethod
    def mano(valid):
        result = {field: torch.zeros(2, size)
                  for field, size in ManoGeometryEncoder.FIELD_DIMS.items()}
        result["side"] = torch.tensor([0, 0])
        result["valid"] = torch.tensor(valid, dtype=torch.bool)
        for field in ManoGeometryEncoder.FIELD_DIMS:
            result[field][~result["valid"]] = float("nan")
        return result

    def test_wrapper_without_mano_preserves_full_ve_prompt_and_geometry_exactly(self):
        with torch.no_grad():
            expected, expected_mask, _ = self.encode(self.plain)
            ve = self.model.backbone.language_backbone
            attach_mano_geometry_encoder(self.model)
            for prompt in (self.plain, ManoPrompt.from_prompt(self.plain, mano=None)):
                actual, actual_mask, _ = self.encode(prompt)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_mask, expected_mask, rtol=0, atol=0)
        self.assertIs(self.model.backbone.language_backbone, ve)
        self.assertIsInstance(ve, VETextEncoder)
        self.assertFalse(any(isinstance(module, LearnableClassTextEncoder)
                             for module in self.model.modules()))
        self.assertEqual(tuple(expected.shape), (33, 2, 256))
        self.assertEqual(tuple(self.backbone["language_embeds"].shape), (32, 2, 16))
        self.assertEqual((~self.backbone["language_mask"]).sum(1).tolist(), [4, 4])

    def test_mano_appends_after_geometry_without_changing_original_ve_features(self):
        attach_mano_geometry_encoder(self.model)
        with torch.no_grad():
            plain, plain_mask, _ = self.encode(self.plain)
            augmented, mask, _ = self.encode(ManoPrompt(mano=self.mano([True, False])))
        self.assertEqual(tuple(augmented.shape), (38, 2, 256))
        torch.testing.assert_close(augmented[:33], plain, rtol=0, atol=0)
        torch.testing.assert_close(mask[:, :33], plain_mask, rtol=0, atol=0)
        self.assertEqual(mask[:, 33:].tolist(), [[False] * 5, [True] * 5])
        self.assertEqual(float(augmented[33:, 1].abs().sum()), 0.)
        # Invalid MANO retains original VE and geometry tokens, never all-masked.
        self.assertTrue((~mask).any(dim=1).all())
        self.assertTrue(torch.isfinite(augmented).all())

    def test_mano_only_freeze_also_freezes_original_ve_but_keeps_adapter_gradient(self):
        wrapper = attach_mano_geometry_encoder(self.model)
        trainable = freeze_for_mano_geometry_encoder(self.model)
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("geometry_encoder.mano_encoder.")
                            for name in trainable))
        original_ve = {name: parameter.detach().clone()
                       for name, parameter in self.model.backbone.language_backbone.named_parameters()}
        prompt, mask, _ = self.encode(ManoPrompt(mano=self.mano([True, True])))
        output = self.model.test_downstream(prompt).squeeze(-1)
        output.masked_fill(mask.T, 0).square().sum().backward()
        self.assertGreater(float(wrapper.mano_encoder.field_embedding.grad.abs().sum()), 0.)
        for name, parameter in self.model.backbone.language_backbone.named_parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, original_ve[name], rtol=0, atol=0)
        self.assertIsNone(self.model.test_downstream.weight.grad)


if __name__ == "__main__":
    unittest.main()

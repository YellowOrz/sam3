import unittest
from unittest.mock import patch

import torch
from torch import nn

from sam3.model.geometry_encoders import Prompt
from sam3.model.mano_geometry_encoder import ManoGeometryEncoder, freeze_for_mano_geometry_encoder
from sam3.model.mano_prompt_adapter import (
    ManoAugmentedGeometryEncoder, ManoPrompt, attach_mano_geometry_encoder,
)


class RecordingGeometryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.weight = nn.Parameter(torch.randn(2, 2, 256))
        self.mask_encoder = nn.Identity()
        self.last_output = None
        self.last_prompt = None

    def forward(self, geo_prompt, img_feats, img_sizes, img_pos_embeds=None):
        self.last_prompt = geo_prompt
        # First row has one genuine geometric token; second has two.
        self.last_output = (self.weight * 1.0, torch.tensor([[False, True], [False, False]]))
        return self.last_output


class ManoPromptAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def mano(self, valid=None):
        mano = {name: torch.randn(2, size, requires_grad=True) for name, size in ManoGeometryEncoder.FIELD_DIMS.items()}
        mano["side"] = torch.tensor([0, 1])
        if valid is not None:
            mano["valid"] = torch.tensor(valid, dtype=torch.bool)
        return mano

    def full_prompt(self):
        return Prompt(
            box_embeddings=torch.randn(1, 2, 4, requires_grad=True),
            point_embeddings=torch.randn(1, 2, 2, requires_grad=True),
            mask_embeddings=torch.randn(1, 2, 1, 4, 4, requires_grad=True),
            box_mask=torch.tensor([[False], [True]]),
            point_mask=torch.tensor([[False], [False]]),
            mask_mask=torch.tensor([[False], [True]]),
            box_labels=torch.tensor([[1, 0]]),
            point_labels=torch.tensor([[0, 1]]),
            mask_labels=torch.tensor([[1, 1]]),
        )

    def test_clone_preserves_every_prompt_field_mano_storage_and_gradient(self):
        original = self.full_prompt()
        mano = self.mano([True, False])
        augmented = ManoPrompt.from_prompt(original, mano=mano)
        cloned = augmented.clone()
        self.assertIsInstance(cloned, ManoPrompt)
        for name in ManoPrompt.PROMPT_FIELDS:
            source, copied = getattr(original, name), getattr(cloned, name)
            torch.testing.assert_close(source, copied)
            self.assertNotEqual(source.data_ptr(), copied.data_ptr())
        for name in mano:
            torch.testing.assert_close(mano[name], cloned.mano[name])
            self.assertNotEqual(mano[name].data_ptr(), cloned.mano[name].data_ptr())
        (cloned.mask_embeddings.sum() + cloned.mano["hand_pose"].sum()).backward()
        self.assertIsNotNone(original.mask_embeddings.grad)
        self.assertIsNotNone(mano["hand_pose"].grad)

    def test_mano_only_prompt_initializes_zero_length_ordinary_geometry(self):
        prompt = ManoPrompt(mano=self.mano())
        self.assertEqual(prompt.point_embeddings.shape, (0, 2, 2))
        self.assertEqual(prompt.box_embeddings.shape, (0, 2, 4))
        self.assertEqual(prompt.point_mask.shape, (2, 0))
        cloned = prompt.clone()
        self.assertEqual(cloned.point_embeddings.shape, (0, 2, 2))

    def test_no_mano_wrapper_returns_original_output_without_adapter_gradient(self):
        base = RecordingGeometryEncoder()
        wrapper = ManoAugmentedGeometryEncoder(base)
        for prompt in (self.full_prompt(), ManoPrompt.from_prompt(self.full_prompt())):
            result = wrapper(prompt, img_feats=[], img_sizes=[])
            self.assertIs(result, base.last_output)
            self.assertIs(base.last_prompt, prompt)
        self.assertIs(wrapper.mask_encoder, base.mask_encoder)
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.mano_encoder.parameters()))

    def test_appended_tokens_keep_padding_right_and_missing_rows_masked(self):
        base = RecordingGeometryEncoder()
        wrapper = ManoAugmentedGeometryEncoder(base)
        prompt = ManoPrompt.from_prompt(self.full_prompt(), mano=self.mano([True, False]))
        expected_mano, _ = wrapper.mano_encoder(**prompt.mano)
        result, mask = wrapper(prompt, img_feats=[], img_sizes=[])
        self.assertEqual(result.shape, (7, 2, 256))
        self.assertEqual(mask.tolist(), [[False] * 6 + [True], [False] * 2 + [True] * 5])
        torch.testing.assert_close(result[:1, 0], base.last_output[0][:1, 0])
        torch.testing.assert_close(result[1:6, 0], expected_mano[:, 0])
        torch.testing.assert_close(result[:2, 1], base.last_output[0][:2, 1])
        self.assertEqual(result[2:, 1].abs().sum(), 0)

    def test_downstream_loss_trains_only_adapter_when_base_is_frozen(self):
        base = RecordingGeometryEncoder()
        wrapper = ManoAugmentedGeometryEncoder(base)
        model = nn.ModuleDict({"geometry": wrapper, "downstream": nn.Linear(256, 1)})
        trainable = freeze_for_mano_geometry_encoder(model)
        self.assertTrue(all(name.startswith("geometry.mano_encoder.") for name in trainable))
        features, mask = wrapper(ManoPrompt(mano=self.mano()), img_feats=[], img_sizes=[])
        output = model["downstream"](features).squeeze(-1)
        output.masked_fill(mask.T, 0).square().sum().backward()
        self.assertGreater(wrapper.mano_encoder.field_embedding.grad.abs().sum(), 0)
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(model["downstream"].weight.grad)

    def test_actual_sequence_geometry_encoder_accepts_mano_only_and_plain_prompts(self):
        from sam3.model_builder import _create_geometry_encoder

        base = _create_geometry_encoder().eval()
        wrapper = ManoAugmentedGeometryEncoder(base).eval()
        image_features = [torch.randn(4, 2, 256)]
        image_positions = [torch.zeros_like(image_features[0])]
        image_sizes = [(2, 2)]
        mano = self.mano([True, False])
        plain = Prompt(point_embeddings=torch.empty(0, 2, 2))
        # The upstream box encoder unconditionally pins a four-element scale
        # tensor even with zero boxes. Bypass only that CUDA transport hint for
        # this CPU test; run its projections/attention/ROI operations unchanged.
        with torch.no_grad(), patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor):
            expected = base(plain, image_features, image_sizes, image_positions)
            actual = wrapper(plain, image_features, image_sizes, image_positions)
            augmented, mask = wrapper(
                ManoPrompt(mano=mano), image_features, image_sizes, image_positions
            )
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1])
        self.assertEqual(augmented.shape, (6, 2, 256))
        self.assertEqual(mask.tolist(), [[False] * 6, [False] + [True] * 5])
        torch.testing.assert_close(augmented[:1], expected[0])
        self.assertTrue(torch.isfinite(augmented).all())

    def test_bad_mapping_and_misaligned_query_batch_fail_explicitly(self):
        for bad_mano in ({}, {**self.mano(), "extra": torch.ones(2)}, {**self.mano(), "side": [0, 1]}):
            with self.subTest(keys=bad_mano.keys()):
                with self.assertRaises(ValueError):
                    ManoPrompt(mano=bad_mano)
        wrapper = ManoAugmentedGeometryEncoder(RecordingGeometryEncoder())
        mano = {name: value[:1] for name, value in self.mano().items()}
        with self.assertRaisesRegex(ValueError, "align MANO per query"):
            wrapper(ManoPrompt(mano=mano), img_feats=[], img_sizes=[])

    def test_explicit_installer_preserves_loaded_base_and_device_dtype_mode(self):
        model = nn.Module()
        base = RecordingGeometryEncoder().double().eval()
        model.geometry_encoder = base
        expected_weights = base.weight.detach().clone()
        wrapper = attach_mano_geometry_encoder(model)
        self.assertIs(model.geometry_encoder, wrapper)
        self.assertIs(wrapper.base_encoder, base)
        torch.testing.assert_close(wrapper.base_encoder.weight, expected_weights)
        self.assertEqual(wrapper.mano_encoder.field_embedding.dtype, torch.float64)
        self.assertEqual(wrapper.mano_encoder.field_embedding.device, base.weight.device)
        self.assertFalse(wrapper.training)
        self.assertFalse(wrapper.mano_encoder.training)
        with self.assertRaisesRegex(ValueError, "already has"):
            attach_mano_geometry_encoder(model)
        self.assertIs(model.geometry_encoder, wrapper)
        with self.assertRaisesRegex(ValueError, "geometry_encoder"):
            attach_mano_geometry_encoder(nn.Identity())


if __name__ == "__main__":
    unittest.main()

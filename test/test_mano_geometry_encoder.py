import unittest

import torch
from torch import nn

from sam3.model.mano_geometry_encoder import (
    ManoGeometryEncoder,
    freeze_for_mano_geometry_encoder,
)


class ManoGeometryEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.encoder = ManoGeometryEncoder()

    def inputs(self, valid=None):
        inputs = {
            name: torch.randn(2, size, requires_grad=True)
            for name, size in ManoGeometryEncoder.FIELD_DIMS.items()
        }
        inputs["side"] = torch.tensor([0, 1])
        if valid is not None:
            inputs["valid"] = torch.tensor(valid, dtype=torch.bool)
        return inputs

    def test_output_matches_geometry_layout_and_both_sides_receive_gradients(self):
        inputs = self.inputs()
        features, mask = self.encoder(**inputs)
        self.assertEqual(features.shape, (5, 2, 256))
        self.assertEqual(mask.shape, (2, 5))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(mask.any())
        (features * torch.randn_like(features)).sum().backward()
        for name in self.encoder.FIELD_DIMS:
            self.assertTrue(torch.isfinite(inputs[name].grad).all())
            self.assertTrue((inputs[name].grad.abs().sum(dim=1) > 0).all())
        for side in (0, 1):
            self.assertGreater(self.encoder.side_embedding.weight.grad[side].abs().sum(), 0)

    def test_invalid_nan_rows_are_zero_and_cannot_poison_parameter_gradients(self):
        inputs = self.inputs([True, False])
        with torch.no_grad():
            for name in self.encoder.FIELD_DIMS:
                inputs[name][1] = float("nan")
            inputs["side"][1] = -1
        features, mask = self.encoder(**inputs)
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(features[:, 1].abs().sum(), 0)
        self.assertTrue(mask[1].all())
        self.assertFalse(mask[0].any())
        (features * torch.randn_like(features)).sum().backward()
        for name in self.encoder.FIELD_DIMS:
            self.assertEqual(inputs[name].grad[1].abs().sum(), 0)
            self.assertGreater(inputs[name].grad[0].abs().sum(), 0)
        for parameter in self.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(self.encoder.side_embedding.weight.grad[1].abs().sum(), 0)

    def test_all_invalid_batch_still_has_finite_zero_gradient_graph(self):
        inputs = self.inputs([False, False])
        features, mask = self.encoder(**inputs)
        self.assertTrue(mask.all())
        self.assertEqual(features.abs().sum(), 0)
        features.sum().backward()
        for parameter in self.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.abs().sum(), 0)

    def test_prompt_rows_remain_independent_and_order_preserving(self):
        inputs = self.inputs()
        for name in self.encoder.FIELD_DIMS:
            inputs[name] = inputs[name][0:1].expand(2, -1).clone()
        features, _ = self.encoder(**inputs)
        # Same parameters with different physical sides differ only in the side token.
        torch.testing.assert_close(features[:-1, 0], features[:-1, 1])
        self.assertFalse(torch.equal(features[-1, 0], features[-1, 1]))
        permuted, _ = self.encoder(**{name: value.flip(0) for name, value in inputs.items()})
        torch.testing.assert_close(permuted, features.flip(1))

    def test_translation_has_its_own_token_and_unit_scale_survives_checkpoint(self):
        inputs = self.inputs()
        original, _ = self.encoder(**inputs)
        changed, _ = self.encoder(**{**inputs, "transl": inputs["transl"] + 0.2})
        torch.testing.assert_close(original[[0, 1, 2, 4]], changed[[0, 1, 2, 4]])
        self.assertFalse(torch.equal(original[3], changed[3]))
        encoder = ManoGeometryEncoder(translation_scale_m=0.5)
        restored = ManoGeometryEncoder(translation_scale_m=1.0)
        restored.load_state_dict(encoder.state_dict())
        self.assertEqual(float(restored.translation_scale_m), 0.5)
        torch.testing.assert_close(encoder(**inputs)[0], restored(**inputs)[0])

    def test_valid_nonfinite_fields_are_rejected(self):
        for name in self.encoder.FIELD_DIMS:
            for value in (float("nan"), float("inf")):
                with self.subTest(field=name, value=value):
                    inputs = self.inputs()
                    with torch.no_grad():
                        inputs[name][0, 0] = value
                    with self.assertRaisesRegex(ValueError, name + ".*nonfinite"):
                        self.encoder(**inputs)

    def test_malformed_shapes_dtypes_and_missing_side_are_rejected(self):
        bad_inputs = (
            ("global_orient", torch.zeros(2, 4)),
            ("hand_pose", torch.zeros(2, 48)),
            ("betas", torch.zeros(1, 10)),
            ("transl", torch.zeros(2, 3, dtype=torch.long)),
            ("side", torch.tensor([0.0, 1.0])),
            ("side", torch.tensor([[0], [1]])),
            ("side", torch.tensor([-1, 1])),
            ("side", torch.tensor([0, 2])),
            ("valid", torch.tensor([1, 0])),
            ("valid", torch.tensor([[True], [False]])),
        )
        for name, value in bad_inputs:
            with self.subTest(field=name, shape=value.shape):
                with self.assertRaisesRegex(ValueError, name):
                    self.encoder(**{**self.inputs(), name: value})
        inputs = self.inputs([False, True])
        inputs["side"][0] = 9
        with self.assertRaisesRegex(ValueError, "side"):
            self.encoder(**inputs)

    def test_float_cast_preserves_input_gradients_and_rejects_overflow(self):
        inputs = self.inputs()
        for name in self.encoder.FIELD_DIMS:
            inputs[name] = inputs[name].detach().double().requires_grad_()
        features, _ = self.encoder(**inputs)
        self.assertEqual(features.dtype, torch.float32)
        (features * torch.randn_like(features)).sum().backward()
        self.assertGreater(inputs["hand_pose"].grad.abs().sum(), 0)
        with torch.no_grad():
            inputs["hand_pose"][0, 0] = 1e100
        with self.assertRaisesRegex(ValueError, "hand_pose.*overflows"):
            self.encoder(**inputs)

    def test_empty_batch_and_invalid_constructor_settings_are_rejected(self):
        inputs = {name: value[:0] for name, value in self.inputs().items()}
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.encoder(**inputs)
        for kwargs in (
            {"d_model": 0}, {"hidden_dim": 0}, {"translation_scale_m": 0},
            {"translation_scale_m": float("nan")},
            {"translation_scale_m": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ManoGeometryEncoder(**kwargs)

    def test_freeze_helper_keeps_downstream_gradient_path_to_adapter(self):
        model = nn.ModuleDict({
            "image_backbone": nn.Linear(3, 256),
            "text_tokens": nn.Embedding(2, 256),
            "mano": self.encoder,
            "mask_head": nn.Linear(256, 1),
        })
        trainable = freeze_for_mano_geometry_encoder(model)
        self.assertEqual(trainable, {"mano." + name for name, _ in self.encoder.named_parameters()})
        features, _ = model["mano"](**self.inputs())
        model["mask_head"](features).square().mean().backward()
        self.assertGreater(model["mano"].field_embedding.grad.abs().sum(), 0)
        for name in ("image_backbone", "text_tokens", "mask_head"):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model[name].parameters()))

    def test_freeze_helper_without_adapter_fails_without_mutating_model(self):
        model = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "does not contain"):
            freeze_for_mano_geometry_encoder(model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()

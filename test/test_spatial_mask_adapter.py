import unittest
import torch
from torch import nn
from sam3.model.spatial_mask_adapter import SpatialMaskAdapter, attach_spatial_mask_adapter


class SpatialAdapterTests(unittest.TestCase):
    def test_exact_identity_both_layouts(self):
        adapter = SpatialMaskAdapter(8, 4)
        for shape in ((8, 7, 9), (2, 8, 7, 9)):
            x = torch.randn(shape)
            self.assertTrue(torch.equal(x, adapter(x)))

    def test_gradient_and_update(self):
        adapter = SpatialMaskAdapter(8, 4)
        x = torch.randn(2, 8, 7, 9)
        opt = torch.optim.AdamW(adapter.parameters(), lr=.001)
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0)
        # Zero last projection initially blocks upstream gradients, by design.
        self.assertEqual(adapter.down.weight.grad.abs().sum().item(), 0)
        opt.step()
        opt.zero_grad()
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.down.weight.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(x, adapter(x)))

    def test_freeze_and_reload(self):
        model = nn.Module()
        model.segmentation_head = nn.Module()
        original = nn.Conv2d(8, 8, 1)
        model.segmentation_head.instance_seg_head = original
        x = torch.randn(2, 8, 7, 9)
        before = original(x).detach()
        adapter = attach_spatial_mask_adapter(model, 4)
        self.assertTrue(torch.equal(before, model.segmentation_head.instance_seg_head(x)))
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                         {id(p) for p in adapter.parameters()})
        self.assertFalse(original.training)
        other = SpatialMaskAdapter(8, 4)
        other.load_state_dict(adapter.state_dict())
        self.assertTrue(torch.equal(adapter(x), other(x)))
        with self.assertRaises(ValueError):
            attach_spatial_mask_adapter(model)

    def test_shapes_and_configuration_rejected(self):
        for args in ((0, 4), (8, 0), (True, 4)):
            with self.assertRaises(ValueError):
                SpatialMaskAdapter(*args)
        with self.assertRaises(ValueError):
            SpatialMaskAdapter(8, 4)(torch.randn(1, 7, 4, 4))

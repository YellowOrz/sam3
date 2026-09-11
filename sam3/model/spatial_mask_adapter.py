"""Opt-in instance-mask-only adapter; original builders remain unchanged.

This deliberately cannot change boxes, concept presence or instance scores.
It is a boundary/shape ablation, not a proposed cure for missed detections.
"""
import torch
from torch import nn


class SpatialMaskAdapter(nn.Module):
    def __init__(self, channels=256, bottleneck=32):
        super().__init__()
        if type(channels) is not int or type(bottleneck) is not int or min(channels, bottleneck) < 1:
            raise ValueError("channels and bottleneck must be positive integers")
        self.channels = channels
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.local = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        if features.ndim not in (3, 4) or features.shape[-3] != self.channels:
            raise ValueError("Expected [C,H,W] or [B,C,H,W] with configured channels")
        residual = self.up(torch.nn.functional.gelu(self.local(
            torch.nn.functional.gelu(self.down(features)))))
        return features + residual


def attach_spatial_mask_adapter(model, bottleneck=32):
    """Call AFTER loading base weights. Freeze base, preserve its eval behavior.

    Wrap only instance_seg_head so the semantic head and detector stay intact.
    Save adapter.state_dict separately; never load it as an old token checkpoint.
    """
    head = model.segmentation_head
    if not isinstance(head.instance_seg_head, nn.Conv2d):
        raise ValueError("Expected unwrapped original Conv2d instance head")
    original = head.instance_seg_head
    adapter = SpatialMaskAdapter(original.in_channels, bottleneck).to(
        device=original.weight.device, dtype=original.weight.dtype)
    model.requires_grad_(False)
    head.instance_seg_head = nn.Sequential(adapter, original)
    model.eval()
    return adapter

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    # Fused kernel is not in autograd. Global grad can still be on when every
    # input is a constant (frozen ViT during learned-prompt training).
    needs_grad = torch.is_grad_enabled() and (
        mat1.requires_grad
        or linear.weight.requires_grad
        or (linear.bias is not None and linear.bias.requires_grad)
    )
    if needs_grad:
        raise ValueError("Expected grad to be disabled.")
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    raise ValueError(f"Unexpected activation {activation}")

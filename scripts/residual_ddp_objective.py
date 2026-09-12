"""Output-residual objective with an explicit large-batch/DDP loss contract.

Only delta is trainable. All SAM3 modules remain in eval mode; eval does not
disable autograd. Target-normalized losses use the GLOBAL number of targets,
whereas classification/presence are query means (equal queries per rank).
"""

from __future__ import annotations

import torch
from torch import nn
from torch import distributed as dist

from scripts import train_ve_initialized_tokens as shared
from scripts.train_nakehand_semantic_tokens import anchor_penalty

COMPONENTS = ("loss_mask", "loss_dice", "loss_bbox", "loss_giou", "loss_ce", "presence_loss")
TARGET_COMPONENTS = COMPONENTS[:4]
NORMALIZATION = "global_target_sum_clamp_before_world_division__query_mean_v1"


def global_target_denominator(local_count: torch.Tensor, *, distributed=True):
    """DDP averages gradients: divide each local numerator by max(global_N,1)/W.

    Clamping AFTER division by W incorrectly shrinks sparse-positive gradients.
    All ranks, including entirely empty batches, must call this collective.
    """
    value = local_count.detach().float().reshape(()).clone()
    world = 1
    if distributed and dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        dist.all_reduce(value)
    return value.clamp(min=1) / world


def loss_components(model, batch, functions, denominator):
    targets = model.back_convert(batch.find_targets[0])
    counts = targets["num_boxes"]
    if counts.numel() != 2 * len(batch.img_batch):
        raise ValueError("Every RGB image must have both independent hand queries")
    prediction = model(batch)[0]
    indices = model.matcher(prediction, targets)
    prediction["indices"] = indices
    terms = [fn(outputs=prediction, targets=targets, indices=indices, num_boxes=denominator)
             for fn in functions]
    values = {key: term[key] for term in terms for key in COMPONENTS if key in term}
    if set(values) != set(COMPONENTS):
        raise RuntimeError("The six original task losses must all be present")
    vector = torch.stack([values[key].reshape(()) for key in COMPONENTS])
    return vector.sum(), vector, prediction


class ResidualObjective(nn.Module):
    """Wrap this WHOLE objective in DDP, then call loss.backward(), not autograd.grad."""

    def __init__(self, model, *, anchor_weight=0.0, functions=None):
        super().__init__()
        if anchor_weight < 0 or not torch.isfinite(torch.tensor(anchor_weight)):
            raise ValueError("anchor_weight must be finite and nonnegative")
        self.model = model
        self.anchor_weight = float(anchor_weight)
        self.functions = nn.ModuleList(shared.build_loss_functions() if functions is None else functions)
        self.eval()
        self.check_frozen_contract()

    @property
    def encoder(self):
        return self.model.backbone.language_backbone

    def check_frozen_contract(self):
        delta = self.encoder.delta
        shape = shared.cached.expected_delta_shape(self.encoder.mode)
        if delta is None or tuple(delta.shape) != shape or delta.dtype != torch.float32:
            raise RuntimeError(f"Expected one FP32 output residual with declared shape {shape}")
        if [id(p) for p in self.model.parameters() if p.requires_grad] != [id(delta)]:
            raise RuntimeError("Only the output residual may be trainable")
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("Frozen SAM3 must stay in eval mode")

    def forward(self, batch):
        denominator = global_target_denominator(batch.find_targets[0].num_boxes.sum())
        task, components, _ = loss_components(self.model, batch, self.functions, denominator)
        anchor, _ = anchor_penalty(self.encoder)
        return task + self.anchor_weight * anchor, components.detach(), anchor.detach()


def frozen_versions(model, delta):
    return [(name, p, p._version) for name, p in model.named_parameters() if p is not delta]


def assert_frozen_versions(versions):
    if any(p.requires_grad or p.grad is not None or p._version != version for _, p, version in versions):
        raise RuntimeError("A frozen parameter changed or received a gradient")


def mean_across_ranks(value):
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result)
        result /= dist.get_world_size()
    return result


def require_finite_everywhere(*values):
    """One numerical gate on every rank before stepping; do not silently skip a rank."""
    if not values:
        raise ValueError("A finite gate needs at least one tensor")
    bad = torch.stack([(~torch.isfinite(value)).any() for value in values]).any().int()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bool(bad):
        raise FloatingPointError("Nonfinite value on a rank; retain last completed checkpoint")

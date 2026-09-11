"""Real SAM3 loss gradients under global-target and query-mean DDP reduction."""

from functools import partial
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.train.loss import loss_fns
from scripts import cached_ve_text_features as cached
from scripts import residual_ddp_objective as objective
from scripts import train_ve_initialized_tokens as legacy


def make_encoder():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, :4] = False
    return cached.CachedVETextEncoder(padding, torch.ones(32, 2, 256), torch.ones(32, 2, 1024),
        mode="zero_delta", metadata={"base_checkpoint_sha256": "a" * 64,
                                      "tokenizer_sha256": "b" * 64})


class SyntheticSAM3(nn.Module):
    """Tiny differentiable predictions with the actual cached output residual."""

    def __init__(self):
        super().__init__()
        self.frozen_scale = nn.Parameter(torch.tensor(.8), requires_grad=False)
        self.dropout = nn.Dropout(.9)
        self.backbone = nn.Module()
        self.backbone.language_backbone = nn.Identity()
        cached.install_cached_ve_text_encoder(self, make_encoder())

    @staticmethod
    def back_convert(target):
        return target.payload

    @staticmethod
    def matcher(prediction, targets):
        rows = torch.nonzero(targets["num_boxes"], as_tuple=True)[0]
        return rows, torch.zeros_like(rows), torch.arange(len(rows))

    def forward(self, batch):
        _, features, _ = self.backbone.language_backbone(list(cached.CLASS_NAMES))
        side_features = features[:4].float().mean(dim=(0, 2))
        signal = self.dropout(side_features.repeat(len(batch.img_batch))) * self.frozen_scale
        signal = signal + batch.img_batch[:, 0, 0, 0].repeat_interleave(2) * .03
        offsets = torch.tensor([-.2, .3])
        logits = signal[:, None] + offsets
        boxes = torch.sigmoid(logits[:, :, None] * .1 + torch.tensor([.1, -.2, -.8, -.7]))
        pattern = torch.linspace(-.8, .8, 12).reshape(3, 4)
        return [{"pred_logits": logits[:, :, None], "presence_logit_dec": signal[:, None] * .7,
                 "pred_masks": logits[:, :, None, None] + pattern,
                 "pred_boxes": boxes, "pred_boxes_xyxy": box_cxcywh_to_xyxy(boxes)}]


def make_batch(side_counts, image_indices=None):
    if image_indices is None:
        image_indices = list(range(len(side_counts)))
    counts = torch.tensor([count for pair in side_counts for count in pair])
    count = int(counts.sum())
    boxes = torch.tensor([[.45, .4, .3, .35]]).repeat(count, 1)
    masks = torch.zeros(count, 3, 4, dtype=torch.bool)
    masks[:, 1:, 1:3] = True
    positive_rows = torch.nonzero(counts, as_tuple=True)[0]
    padded = torch.zeros(len(counts), 1 if count else 0, 4)
    ids = torch.full((len(counts), 1 if count else 0), -1, dtype=torch.long)
    if count:
        padded[positive_rows, 0] = boxes
        ids[positive_rows, 0] = torch.arange(count)
    target = {"num_boxes": counts, "boxes": boxes, "boxes_xyxy": box_cxcywh_to_xyxy(boxes),
              "boxes_padded": padded, "object_ids_padded": ids, "masks": masks,
              "is_valid_mask": torch.ones(count, dtype=torch.bool),
              "is_exhaustive": torch.ones(len(counts), dtype=torch.bool)}
    pixels = torch.tensor(image_indices, dtype=torch.float32)[:, None, None, None].expand(-1, 3, 2, 2)
    return SimpleNamespace(img_batch=pixels, find_targets=[SimpleNamespace(num_boxes=counts, payload=target)])


class CPUReferenceLossCase(unittest.TestCase):
    def setUp(self):
        # Use SAM3's own mathematically identical PyTorch branch. The default
        # focal-loss branch launches Triton kernels and cannot run on CPU.
        reference_focal = partial(loss_fns.sigmoid_focal_loss, triton=False)
        self.focal_patch = patch.object(loss_fns, "sigmoid_focal_loss", reference_focal)
        self.focal_patch.start()
        self.addCleanup(self.focal_patch.stop)


class GlobalNormalizationTest(CPUReferenceLossCase):
    def test_clamp_precedes_world_division_even_when_positive_count_is_sparse(self):
        for world in (3, 4):
            for total in (0, 1, 2, world + 2):
                local = torch.tensor(0., requires_grad=True)
                with patch.object(objective.dist, "is_available", return_value=True), \
                     patch.object(objective.dist, "is_initialized", return_value=True), \
                     patch.object(objective.dist, "get_world_size", return_value=world), \
                     patch.object(objective.dist, "all_reduce", side_effect=lambda value: value.fill_(total)) as reduce:
                    denominator = objective.global_target_denominator(local)
                self.assertAlmostEqual(float(denominator), max(total, 1) / world)
                self.assertFalse(denominator.requires_grad)
                self.assertEqual(float(local.detach()), 0.)
                reduce.assert_called_once()

    def test_real_six_losses_and_gradients_equal_global_batch_for_three_and_four_ranks(self):
        for world, local_batch in ((3, 2), (4, 1)):
            size = world * local_batch
            distributions = {
                "all_empty": [(0, 0)] * size,
                "one_target_less_than_world": [(1, 0)] + [(0, 0)] * (size - 1),
                "two_targets_less_than_world": [(1, 1)] + [(0, 0)] * (size - 1),
                "mixed": ([(0, 0), (1, 0), (1, 1), (0, 1)] * size)[:size],
            }
            for name, counts in distributions.items():
                with self.subTest(world=world, distribution=name):
                    model = SyntheticSAM3()
                    functions = legacy.build_loss_functions()
                    global_count = sum(sum(pair) for pair in counts)
                    total, terms, _ = objective.loss_components(model, make_batch(counts), functions,
                                                               torch.tensor(max(global_count, 1.)))
                    total.backward()
                    global_gradient = model.backbone.language_backbone.delta.grad.clone()
                    rank_gradients, rank_terms = [], []
                    for rank in range(world):
                        local_model = SyntheticSAM3()
                        begin, end = rank * local_batch, (rank + 1) * local_batch
                        loss, components, _ = objective.loss_components(
                            local_model, make_batch(counts[begin:end], list(range(begin, end))), functions,
                            torch.tensor(max(global_count, 1.) / world))
                        loss.backward()
                        rank_gradients.append(local_model.backbone.language_backbone.delta.grad)
                        rank_terms.append(components.detach())
                    # DDP averages the gradients and metrics of equal-sized ranks.
                    torch.testing.assert_close(torch.stack(rank_terms).mean(0), terms.detach(), rtol=2e-6, atol=2e-7)
                    torch.testing.assert_close(torch.stack(rank_gradients).mean(0), global_gradient,
                                               rtol=3e-6, atol=2e-9)
                    self.assertTrue(bool(torch.isfinite(global_gradient).all()))
                    self.assertGreater(float(global_gradient.norm()), 0.)

    def test_single_image_matches_original_pilot_task_loss(self):
        for counts in (((0, 0),), ((1, 0),), ((0, 1),)):
            model = SyntheticSAM3()
            batch = make_batch(counts)
            functions = legacy.build_loss_functions()
            old_loss, _ = legacy.compute_loss(model, batch, functions)
            new_loss, _, _ = objective.loss_components(model, batch, functions,
                                                       objective.global_target_denominator(batch.find_targets[0].num_boxes.sum(),
                                                                                           distributed=False))
            torch.testing.assert_close(new_loss, old_loss)


class FrozenObjectiveTest(CPUReferenceLossCase):
    def test_standard_backward_updates_only_delta_and_total_is_six_terms_plus_anchor(self):
        model = SyntheticSAM3()
        with torch.no_grad():
            model.backbone.language_backbone.delta.fill_(.2)
        wrapped = objective.ResidualObjective(model, anchor_weight=.3)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        versions = objective.frozen_versions(model, wrapped.encoder.delta)
        optimizer = torch.optim.AdamW([wrapped.encoder.delta], lr=.01, weight_decay=0.)
        total, terms, anchor = wrapped(make_batch(((1, 1), (0, 0))))
        torch.testing.assert_close(total.detach(), terms.sum() + .3 * anchor)
        self.assertEqual(len(terms), 6)
        self.assertFalse(terms.requires_grad)
        self.assertFalse(anchor.requires_grad)
        self.assertGreater(float(anchor), 0.)
        total.backward()
        optimizer.step()
        objective.assert_frozen_versions(versions)
        for name, parameter in model.named_parameters():
            if parameter is wrapped.encoder.delta:
                self.assertFalse(torch.equal(parameter, before[name]))
                self.assertIsNotNone(parameter.grad)
            else:
                self.assertTrue(torch.equal(parameter, before[name]))
                self.assertIsNone(parameter.grad)
        self.assertTrue(all(not module.training for module in wrapped.modules()))

    def test_accidental_model_train_and_additional_trainable_parameter_are_rejected(self):
        wrapped = objective.ResidualObjective(SyntheticSAM3())
        wrapped.model.train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            wrapped(make_batch(((1, 0),)))
        model = SyntheticSAM3()
        model.frozen_scale.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Only the output residual"):
            objective.ResidualObjective(model)

    def test_frozen_parameter_version_and_nonfinite_values_fail_closed(self):
        model = SyntheticSAM3()
        versions = objective.frozen_versions(model, model.backbone.language_backbone.delta)
        with torch.no_grad():
            model.frozen_scale.add_(1)
        with self.assertRaisesRegex(RuntimeError, "frozen parameter"):
            objective.assert_frozen_versions(versions)
        objective.require_finite_everywhere(torch.tensor(1.))
        for value in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                objective.require_finite_everywhere(torch.tensor(value))


if __name__ == "__main__":
    unittest.main()

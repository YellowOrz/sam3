"""CPU-only boundary math, real collator, independent gradients and resume tests."""
from argparse import Namespace
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import train_nakehand_prompt_ablation as training


def encoder_fixture():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, :4] = False
    return training.cached.CachedVETextEncoder(padding, torch.ones(32, 2, 256, dtype=torch.bfloat16),
                                               torch.ones(32, 2, 1024), mode="zero_delta",
                                               metadata={"base_checkpoint_sha256": "a" * 64,
                                                         "tokenizer_sha256": "b" * 64})


def configuration_fixture(initial, weight=4.):
    core = {"sam3/fake.py": "d" * 64}
    config = {"experiment": "nakehand_natural_ve_boundary_focal_ablation", "learning_rate": .001,
              "optimizer": "AdamW", "weight_decay": 0., "batch_size": 1, "seed": 123,
              "amp": True, "amp_dtype": "bfloat16", "planned_samples": 2000,
              "float32_matmul_precision": "high", "network_mode": "eval_with_delta_autograd",
              "delta_shape": [2, 4, 256], "context_length": 32, "anchor_weight": 0.,
              "boundary_weight": weight, "boundary_radius": 4, "training_mask_size": [1008, 1008],
              "boundary_definition": training.BOUNDARY_DEFINITION,
              "boundary_reduction": training.BOUNDARY_REDUCTION, "boundary_alpha": .25, "boundary_gamma": 2.,
              "loss_weights": dict(training.shared.LOSS_WEIGHTS), "base_checkpoint_sha256": "a" * 64,
              "tokenizer_sha256": "b" * 64, "initial_cache_sha256": "c" * 64,
              "initial_cache_state_sha256": training.shared.cache_fingerprint(initial),
              "annotations_sha256": "e" * 64, "data_provenance": {"dataset_role": "train"},
              "core_sources_sha256": training.shared.json_hash(core)}
    return config, core


def step_fixture(encoder, optimizer, histories, weight):
    optimizer.zero_grad(set_to_none=True)
    features = encoder(list(training.cached.CLASS_NAMES))[1].float()
    task = (features * torch.rand_like(features)).mean()
    boundary = features.sin().square().mean()
    task_value, boundary_value = float(task.detach()), float(boundary.detach())
    norms = training.apply_loss_gradients(task, boundary, encoder.delta, weight)
    optimizer.step()
    with torch.no_grad():
        _, ratios = training.reference.anchor_penalty(encoder)
        drift = ratios.sqrt().tolist()
    components = {name: 0. for name in training.COMPONENT_NAMES}
    components["loss_mask"] = task_value
    entries = {"task_loss_history": task_value, "boundary_loss_history": boundary_value,
               "loss_history": task_value + weight * boundary_value, "relative_drift_history": drift,
               "task_grad_norm_history": norms["task"], "boundary_grad_norm_history": norms["boundary"],
               "total_grad_norm_history": norms["total"], "loss_component_history": components,
               "boundary_support_history": {"matched_masks": 2, "boundary_pixels": 100,
                                             "pixels_per_mask": 1008 * 1008}}
    for name, value in entries.items():
        histories[name].append(value)


def checkpoint_fixture(weight=4., steps=20):
    torch.manual_seed(123)
    encoder = encoder_fixture()
    initial = training.shared.cpu_state(encoder.state_dict())
    config, core = configuration_fixture(initial, weight)
    optimizer = torch.optim.AdamW([encoder.delta], lr=.001, weight_decay=0.)
    histories = {name: [] for name in training.HISTORY_NAMES}
    order = training.legacy.build_epoch_order(9092, 123)[:2000]
    image_ids = list(range(20000, 29092))
    for _ in range(steps):
        step_fixture(encoder, optimizer, histories, weight)
    checkpoint = training.make_checkpoint(
        encoder=encoder, optimizer=optimizer, config=config, initial_state=initial,
        annotation_summary={"images": 9092, "sha256": "e" * 64}, order=order,
        observed_ids=[image_ids[index] for index in order[:steps]], histories=histories, core_hashes=core)
    return checkpoint, encoder, optimizer, histories, image_ids


class BoundaryPromptTrainingTest(unittest.TestCase):
    def test_symmetric_band_includes_inside_and_outside_not_whole_region(self):
        mask = torch.zeros(1, 9, 9)
        mask[:, 3:6, 3:6] = 1
        band = training.boundary_band(mask, radius=1)
        self.assertEqual(int(band.sum()), 24)
        self.assertTrue(bool(band[0, 3, 3]))
        self.assertTrue(bool(band[0, 2, 2]))
        self.assertFalse(bool(band[0, 4, 4]))
        self.assertFalse(bool(band[0, 0, 0]))

    def test_image_exterior_is_background_and_empty_masks_have_no_band(self):
        full = torch.ones(1, 5, 5)
        band = training.boundary_band(full, radius=1)
        self.assertEqual(int(band.sum()), 16)
        self.assertTrue(bool(band[0, 0, 0]))
        self.assertFalse(bool(band[0, 2, 2]))
        self.assertFalse(bool(training.boundary_band(torch.zeros_like(full), 1).any()))
        self.assertEqual(training.boundary_band(torch.empty(0, 5, 5)).shape, (0, 5, 5))
        for invalid in (torch.full((1, 5, 5), .5), torch.full((1, 5, 5), float("nan"))):
            with self.assertRaises(ValueError):
                training.boundary_band(invalid)

    def test_extra_focal_uses_all_pixels_and_weight_four_is_five_in_band(self):
        from sam3.train.loss.loss_fns import sigmoid_focal_loss

        source = torch.linspace(-1, 1, 81).view(1, 1, 9, 9).requires_grad_()
        target = torch.zeros(1, 9, 9)
        target[:, 3:6, 3:6] = 1
        indices = (torch.tensor([0]), torch.tensor([0]), None)
        extra, support = training.boundary_focal_loss({"pred_masks": source},
                    {"masks": target, "is_valid_mask": torch.ones(1, dtype=torch.bool)}, indices, 1, radius=1)
        base = sigmoid_focal_loss(source[:, 0], target, 1, reduce=False, triton=False)
        band = training.boundary_band(target, 1)
        torch.testing.assert_close(extra, (base * band).mean())
        torch.testing.assert_close(base.mean() + 4 * extra, (base * (1 + 4 * band.float())).mean())
        self.assertNotAlmostEqual(float(extra.detach()), float((base * band).sum().detach()) / int(band.sum()))
        self.assertEqual(support, {"matched_masks": 1, "boundary_pixels": 24, "pixels_per_mask": 81})

    def test_only_matched_visible_side_receives_extra_mask_gradient(self):
        source = torch.zeros(2, 1, 9, 9, requires_grad=True)
        target = torch.zeros(1, 9, 9)
        target[:, 3:6, 3:6] = 1
        extra, _ = training.boundary_focal_loss({"pred_masks": source},
                    {"masks": target, "is_valid_mask": torch.ones(1, dtype=torch.bool)},
                    (torch.tensor([1]), torch.tensor([0]), None), 1, radius=1)
        gradient = torch.autograd.grad(extra, source)[0]
        self.assertEqual(float(gradient[0].abs().sum()), 0)
        self.assertGreater(float(gradient[1].abs().sum()), 0)
        empty, support = training.boundary_focal_loss({"pred_masks": source},
                    {"masks": torch.empty(0), "is_valid_mask": torch.empty(0, dtype=torch.bool)},
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), None), 1)
        self.assertEqual(float(empty.detach()), 0)
        self.assertEqual(support["matched_masks"], 0)
        self.assertTrue(torch.equal(torch.autograd.grad(empty, source)[0], torch.zeros_like(source)))

    def test_independent_gradients_equal_one_combined_objective(self):
        for weight in (0., 4.):
            encoder = encoder_fixture()
            features = encoder(list(training.cached.CLASS_NAMES))[1].float()
            task, boundary = features.square().mean(), features.sin().mean()
            expected = torch.autograd.grad(task + weight * boundary, encoder.delta, retain_graph=True)[0]
            norms = training.apply_loss_gradients(task, boundary, encoder.delta, weight)
            torch.testing.assert_close(encoder.delta.grad, expected)
            self.assertTrue(all(value > 0 for value in norms["task"]))
            self.assertTrue(all(value > 0 for value in norms["boundary"]))

    def test_weight_zero_matches_old_unconstrained_optimizer_updates_on_cpu(self):
        first, second = encoder_fixture(), encoder_fixture()
        first_opt = torch.optim.AdamW([first.delta], lr=.001, weight_decay=0.)
        second_opt = torch.optim.AdamW([second.delta], lr=.001, weight_decay=0.)
        for _ in range(20):
            a = first(list(training.cached.CLASS_NAMES))[1].float()
            b = second(list(training.cached.CLASS_NAMES))[1].float()
            old_anchor, _ = training.reference.anchor_penalty(first)
            first_opt.zero_grad(set_to_none=True)
            second_opt.zero_grad(set_to_none=True)
            training.reference.apply_loss_gradients(a.square().mean(), old_anchor, first.delta, 0.)
            training.apply_loss_gradients(b.square().mean(), b.sin().mean(), second.delta, 0.)
            first_opt.step()
            second_opt.step()
        self.assertTrue(torch.equal(first.delta, second.delta))
        for name in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(first_opt.state[first.delta][name], second_opt.state[second.delta][name]))

    def test_zero_twenty_and_2000_budget_resume_validation(self):
        state, _, _, _, image_ids = checkpoint_fixture()
        info = training.validate_checkpoint_schema(state, minimum_samples=20, base_hash="a" * 64, tokenizer_hash="b" * 64)
        self.assertEqual(info["boundary_weight"], 4.)
        self.assertFalse(info["pilot_complete"])
        self.assertEqual(training.validate_resume(state, state["training_config"], state["planned_dataset_indices"],
                                                  image_ids, state["initial_cache_state_dict"]), 20)
        with self.assertRaises(ValueError):
            training.validate_checkpoint_schema(state, minimum_samples=2000)
        empty, *_ = checkpoint_fixture(steps=0)
        self.assertEqual(training.validate_checkpoint_schema(empty)["completed_steps"], 0)

    def test_twenty_step_resume_matches_uninterrupted_forty_with_rng_and_moments(self):
        state, first, first_opt, history, image_ids = checkpoint_fixture()
        saved = deepcopy(state)
        for _ in range(20):
            step_fixture(first, first_opt, history, 4.)
        second = encoder_fixture()
        second.load_state_dict(saved["cache_state_dict"])
        second_opt = torch.optim.AdamW([second.delta], lr=.001, weight_decay=0.)
        second_opt.load_state_dict(saved["optimizer"])
        resumed_history = {name: deepcopy(saved[name]) for name in training.HISTORY_NAMES}
        training.shared.restore_rng(saved["rng"])
        for _ in range(20):
            step_fixture(second, second_opt, resumed_history, 4.)
        self.assertTrue(torch.equal(first.delta, second.delta))
        self.assertEqual(history, resumed_history)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(first_opt.state[first.delta][name], second_opt.state[second.delta][name]))
        continued = training.make_checkpoint(encoder=second, optimizer=second_opt, config=saved["training_config"],
            initial_state=saved["initial_cache_state_dict"], annotation_summary=saved["annotation_summary"],
            order=saved["planned_dataset_indices"], observed_ids=[image_ids[index] for index in saved["planned_dataset_indices"][:40]],
            histories=resumed_history, core_hashes=saved["core_source_hashes"])
        self.assertEqual(continued["next_step"], 40)

    def test_schema_rejects_different_method_changed_loss_and_corrupt_provenance(self):
        state, *_ = checkpoint_fixture()
        mutations = [lambda x: x.update(format=training.reference.FORMAT),
                     lambda x: x["training_config"].update(boundary_weight=1.),
                     lambda x: x["training_config"].update(boundary_radius=8),
                     lambda x: x["training_config"].update(learning_rate=.003),
                     lambda x: x["training_config"].update(anchor_weight=1.),
                     lambda x: x["cache_state_dict"].update(delta=x["cache_state_dict"]["delta"].half()),
                     lambda x: x["initial_cache_state_dict"]["delta"].fill_(.01),
                     lambda x: x["loss_history"].__setitem__(0, 999.),
                     lambda x: x["boundary_support_history"][0].update(pixels_per_mask=81),
                     lambda x: x["observed_image_ids"].__setitem__(1, x["observed_image_ids"][0]),
                     lambda x: x["optimizer"]["param_groups"][0].update(lr=.01),
                     lambda x: x["boundary_grad_norm_history"].pop()]
        for mutation in mutations:
            changed = deepcopy(state)
            mutation(changed)
            with self.assertRaises((ValueError, RuntimeError)):
                training.validate_checkpoint_schema(changed)

    def test_real_bilateral_collator_and_single_forward_include_empty_images(self):
        from test_train_nakehand_semantic_tokens import coco_fixture, save_json
        from sam3.train.data.collator import collate_fn_api

        class FakeModel:
            def __init__(self):
                self.parameter, self.calls = torch.ones((), requires_grad=True), 0

            def back_convert(self, target):
                return {"num_boxes": target.num_boxes, "masks": target.segments, "is_valid_mask": target.is_valid_segment}

            def __call__(self, _batch):
                self.calls += 1
                return [{"pred_masks": self.parameter * torch.ones(2, 1, 4, 4)}]

            def matcher(self, _prediction, targets):
                positive = targets["num_boxes"].nonzero().flatten()
                return positive, torch.zeros_like(positive), None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coco = coco_fixture(root)
            save_json(root / "annotations.json", coco)
            dataset = training.evaluation.make_dataset(root)
            for index, expected in enumerate((2, 1, 1, 0)):
                model = FakeModel()
                batch = collate_fn_api([dataset[index]], dict_key="train", with_seg_masks=True)["train"]
                functions = [lambda **kwargs: {"core_loss": model.parameter.square()} for _ in range(3)]
                task, boundary, _, support = training.compute_losses(model, batch, functions)
                self.assertEqual(model.calls, 1)
                self.assertEqual(support["matched_masks"], expected)
                self.assertTrue(bool(torch.isfinite(task + boundary)))
                if not expected:
                    self.assertEqual(float(boundary.detach()), 0)

    def test_cli_limits_one_factor_and_snapshot_covers_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = ["--data-root", str(root / "train"), "--base-checkpoint", str(root / "base.pt"),
                         "--initial-cache", str(root / "cache.pt"), "--output-dir", str(root / "output"),
                         "--boundary-weight", "4", "--max-steps", "20"]
            args = training.parse_args(arguments)
            self.assertEqual(args.anchor_weight, 0.)
            self.assertEqual(args.max_steps, 20)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(arguments + ["--boundary-weight", "1"])
            snapshot_root = root / "snapshots"
            snapshot_root.mkdir()
            rows = training.snapshot_scripts(snapshot_root)
            self.assertEqual({Path(row["source"]).name for row in rows},
                             {"train_nakehand_prompt_ablation.py", "train_nakehand_semantic_tokens.py",
                              "train_ve_initialized_tokens.py", "cached_ve_text_features.py",
                              "evaluate_bilateral_tokens.py", "train_learnable_tokens.py"})
            self.assertTrue(all(training.evaluation.sha256(Path(row["snapshot"])) == row["sha256"] for row in rows))


if __name__ == "__main__":
    unittest.main()

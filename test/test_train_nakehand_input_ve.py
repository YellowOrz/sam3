"""CPU contracts for the independent input-word residual trainer; no GPU jobs."""
from argparse import Namespace
from copy import deepcopy
from functools import lru_cache
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from scripts import train_nakehand_input_ve as training


@lru_cache(maxsize=1)
def initial_fixture():
    return training.initial_residual_state(Path(__file__).resolve().parents[1] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")


class TinyAdapter(nn.Module):
    """Small differentiable stand-in with exact production checkpoint shapes."""
    def __init__(self):
        super().__init__()
        self.input_delta = nn.Parameter(torch.zeros(2, 1024))
        self.register_buffer("basis", torch.arange(32).float().view(32, 1, 1) / 32 + 1)

    def forward(self):
        words = self.input_delta.reshape(2, 4, 256).mean(1)
        return (1 + self.basis * words[0] + self.basis.square() * words[1]).expand(32, 2, 256)

    def residual_state(self):
        state = training.shared.cpu_state(initial_fixture())
        state["input_delta"] = self.input_delta.detach().cpu().clone()
        return state


def configuration_fixture():
    initial_input = training.shared.cpu_state(initial_fixture())
    cache = training.cached.CachedVETextEncoder(initial_input["natural_token_ids"].eq(0),
        torch.ones(32, 2, 256, dtype=torch.bfloat16), torch.ones(32, 2, 1024), mode="zero_delta",
        metadata={"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64})
    initial_cache = training.shared.cpu_state(cache.state_dict())
    core = {"sam3/fake.py": "d" * 64}
    config = deepcopy(training.FIXED_CONFIG)
    config.update(learning_rate=.001, data_root="/placeholder/train", base_checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64, initial_cache_sha256="c" * 64, annotations_sha256="e" * 64,
        initial_input_state_sha256=training.shared.cache_fingerprint(initial_input),
        initial_cache_state_sha256=training.shared.cache_fingerprint(initial_cache),
        core_sources_sha256=training.shared.json_hash(core), data_provenance={"dataset_role": "train"})
    return config, core, initial_input, initial_cache


def step_fixture(encoder, optimizer, histories):
    optimizer.zero_grad(set_to_none=True)
    features = encoder()
    loss = (features * torch.rand_like(features)).mean()
    value = float(loss.detach())
    norms = training.apply_task_gradients(loss, encoder.input_delta, features)
    optimizer.step()
    components = {name: 0. for name in training.COMPONENT_NAMES}
    components["loss_mask"] = value
    rows = {"loss_history": value, "task_loss_history": value, "loss_component_history": components,
            "task_grad_norm_history": norms["word_roles"],
            "left_right_feature_grad_norm_history": norms["hand_sides"],
            "input_delta_norm_history": encoder.input_delta.detach().norm(dim=1).tolist()}
    for name, row in rows.items():
        histories[name].append(row)


def checkpoint_fixture(steps=20, learning_rate=.001):
    torch.manual_seed(123)
    encoder = TinyAdapter()
    config, core, initial_input, initial_cache = configuration_fixture()
    config["learning_rate"] = learning_rate
    optimizer = torch.optim.AdamW([encoder.input_delta], lr=learning_rate, weight_decay=0.)
    histories = {name: [] for name in training.HISTORY_NAMES}
    order = training.legacy.build_epoch_order(9092, 123)[:2000]
    image_ids = list(range(20000, 29092))
    for _ in range(steps):
        step_fixture(encoder, optimizer, histories)
    state = training.make_checkpoint(encoder=encoder, optimizer=optimizer, config=config,
        initial_input_state=initial_input, initial_cache_state=initial_cache,
        annotation_summary={"images": 9092, "sha256": "e" * 64}, order=order,
        observed_ids=[image_ids[index] for index in order[:steps]], histories=histories, core_hashes=core)
    return state, encoder, optimizer, histories, image_ids


class InputVETrainingTests(unittest.TestCase):
    def test_zero_twenty_checkpoint_schema_and_roles(self):
        for steps in (0, 20):
            state, *_ = checkpoint_fixture(steps)
            result = training.validate_checkpoint_schema(state, base_hash="a" * 64, tokenizer_hash="b" * 64)
            self.assertEqual(result["completed_steps"], steps)
            self.assertEqual(result["parameter_row_semantics"], ["side_word", "hand_word"])
            self.assertEqual(state["gradient_nonzero_steps"], [steps, steps])
            self.assertNotIn("cache_state_dict", state)
            self.assertEqual(state["input_residual_state"]["input_delta"].numel(), 2048)
            with self.assertRaises(ValueError):
                training.validate_checkpoint_schema(state, minimum_samples=2000)

    def test_twenty_plus_resume_equals_uninterrupted_forty_with_rng_and_moments(self):
        state, first, first_optimizer, first_history, image_ids = checkpoint_fixture()
        saved = deepcopy(state)
        for _ in range(20):
            step_fixture(first, first_optimizer, first_history)
        config = saved["training_config"]
        self.assertEqual(training.validate_resume(saved, config, saved["planned_dataset_indices"], image_ids,
            saved["initial_input_residual_state"], saved["initial_cache_state_dict"]), 20)
        second = TinyAdapter()
        with torch.no_grad():
            second.input_delta.copy_(saved["input_residual_state"]["input_delta"])
        optimizer = torch.optim.AdamW([second.input_delta], lr=.001, weight_decay=0.)
        optimizer.load_state_dict(saved["optimizer"])
        histories = {name: deepcopy(saved[name]) for name in training.HISTORY_NAMES}
        training.shared.restore_rng(saved["rng"])
        for _ in range(20):
            step_fixture(second, optimizer, histories)
        self.assertTrue(torch.equal(first.input_delta, second.input_delta))
        self.assertEqual(first_history, histories)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(first_optimizer.state[first.input_delta][name], optimizer.state[second.input_delta][name]))

    def test_task_backward_tracks_both_word_roles_and_both_feature_sides(self):
        encoder = TinyAdapter()
        features = encoder()
        loss = features.square().mean()
        expected = torch.autograd.grad(loss, encoder.input_delta, retain_graph=True)[0]
        norms = training.apply_task_gradients(loss, encoder.input_delta, features)
        self.assertTrue(torch.equal(encoder.input_delta.grad, expected))
        self.assertTrue(all(value > 0 for value in norms["word_roles"] + norms["hand_sides"]))

    def test_disconnected_nan_and_malformed_feature_gradients_fail(self):
        encoder = TinyAdapter()
        features = encoder()
        with self.assertRaises(RuntimeError):
            training.apply_task_gradients(torch.tensor(1., requires_grad=True), encoder.input_delta, features)
        with self.assertRaises(RuntimeError):
            training.apply_task_gradients(features.mean() * float("nan"), encoder.input_delta, features)
        with self.assertRaises(RuntimeError):
            training.apply_task_gradients(features.mean(), encoder.input_delta, features[:4])
        bad = encoder()
        # Finite forward value with infinite backward derivative at zero.
        loss = encoder.input_delta.abs().sqrt().sum() + bad.sum() * 0
        with self.assertRaises(RuntimeError):
            training.apply_task_gradients(loss, encoder.input_delta, bad)

    def test_zero_individual_frames_allowed_but_missing_windows_fail(self):
        history = [[0., 0.]] * 19 + [[1., 1.]]
        self.assertEqual(training.gradient_coverage(history, 20, "roles"), [1, 1])
        with self.assertRaises(ValueError):
            training.gradient_coverage([[0., 1.]] * 20, 20, "roles")
        with self.assertRaises(ValueError):
            training.gradient_coverage([[1., 1.]] * 20 + [[0., 0.]] * 180, 200, "roles")

    def test_nonzero_initialization_changed_tokenizer_and_frozen_base_weights_rejected(self):
        initial = training.shared.cpu_state(initial_fixture())
        for mutation in (
            lambda s: s["input_delta"].fill_(.01),
            lambda s: s.update(positions=[0, 2]),
            lambda s: s.update(shared_across_sides=False),
            lambda s: s.update(input_delta=s["input_delta"].bfloat16()),
            lambda s: s.update(original_ve={}),
            lambda s: s["natural_token_ids"].__setitem__((0, 1), s["natural_token_ids"][1, 1]),
        ):
            state = deepcopy(initial)
            mutation(state)
            with self.assertRaises(ValueError):
                training.validate_residual_state(state, require_zero=True)

    def test_schema_corruption_and_false_progress_rejected(self):
        state, *_ = checkpoint_fixture()
        mutations = [
            lambda s: s.update(format=training.reference.FORMAT),
            lambda s: s.update(cache_state_dict={}),
            lambda s: s["training_config"].update(seed=124),
            lambda s: s["training_config"].update(anchor_weight=1.),
            lambda s: s["training_config"]["loss_weights"].update(mask=4.),
            lambda s: s["training_config"].update(learning_rate=float("nan")),
            lambda s: s["optimizer"]["param_groups"][0].update(lr=.003),
            lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
            lambda s: s["initial_input_residual_state"]["input_delta"].fill_(.01),
            lambda s: s["input_residual_state"]["natural_token_ids"].__setitem__((0, 1), 100),
            lambda s: s["input_delta_norm_history"][-1].__setitem__(0, 99.),
            lambda s: s["loss_history"].__setitem__(0, 999.),
            lambda s: s["loss_component_history"][0].pop("loss_mask"),
            lambda s: s["observed_image_ids"].__setitem__(1, s["observed_image_ids"][0]),
            lambda s: s["planned_dataset_indices"].reverse(),
            lambda s: s["progress"].update(full_epoch_completed=True),
            lambda s: s["core_source_hashes"].update(extra="f" * 64),
            lambda s: s.update(rng={}),
            lambda s: s["left_right_feature_grad_norm_history"].pop(),
        ]
        for mutation in mutations:
            changed = deepcopy(state)
            mutation(changed)
            with self.assertRaises((ValueError, RuntimeError)):
                training.validate_checkpoint_schema(changed)

    def test_learning_rate_configurable_but_resume_cannot_change_it(self):
        state, _, _, _, ids = checkpoint_fixture(steps=1, learning_rate=.0003)
        self.assertEqual(training.validate_checkpoint_schema(state)["learning_rate"], .0003)
        changed = deepcopy(state["training_config"])
        changed["learning_rate"] = .001
        with self.assertRaises(ValueError):
            training.validate_resume(state, changed, state["planned_dataset_indices"], ids,
                state["initial_input_residual_state"], state["initial_cache_state_dict"])

    def test_six_named_losses_and_no_extra_loss(self):
        terms = [{name: torch.tensor(float(i + 1)) for i, name in enumerate(training.COMPONENT_NAMES)}]
        value, components = training.named_loss_values(terms, torch.tensor(21.))
        self.assertEqual(value, 21.)
        self.assertEqual(set(components), set(training.COMPONENT_NAMES))
        with self.assertRaises(RuntimeError):
            training.named_loss_values(terms, torch.tensor(22.))

    def test_frozen_parameter_guard_ignores_only_the_residual(self):
        model = nn.Linear(2, 2)
        model.requires_grad_(False)
        delta = nn.Parameter(torch.zeros(2, 1024))
        model.register_parameter("input_delta", delta)
        versions = training.frozen_parameter_versions(model, delta)
        with torch.no_grad():
            delta.add_(1)
        training.verify_frozen_parameters(versions)
        with torch.no_grad():
            model.weight.add_(1)
        with self.assertRaises(RuntimeError):
            training.verify_frozen_parameters(versions)

    def test_cli_limits_new_output_and_resumable_twenty_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flags = ["--data-root", str(root / "train"), "--base-checkpoint", str(root / "base.pt"),
                     "--initial-cache", str(root / "cache.pt"), "--output-dir", str(root / "run")]
            args = training.parse_args(flags + ["--max-samples", "20"])
            self.assertEqual(args.max_steps, 20)
            self.assertEqual(args.gpu_memory_fraction, .35)
            for suffix in (["--max-steps", "2001"], ["--seed", "0"], ["--learning-rate", "nan"],
                           ["--learning-rate", "0"], ["--gpu-memory-fraction", ".36"], ["--checkpoint-every", "0"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    training.parse_args(flags + suffix)

    def test_snapshot_hashes_and_atomic_checkpoint_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = training.snapshot_scripts(root)
            self.assertIn("soft_ve_prompt.py", {Path(row["source"]).name for row in rows})
            self.assertTrue(all(training.evaluation.sha256(Path(row["snapshot"])) == row["sha256"] for row in rows))
            state, *_ = checkpoint_fixture(steps=1)
            path = root / "checkpoint.pt"
            training.shared.atomic_save(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(training.validate_checkpoint_schema(loaded)["completed_steps"], 1)
            with self.assertRaises(FileExistsError):
                training.shared.atomic_save(path, state)


if __name__ == "__main__":
    unittest.main()

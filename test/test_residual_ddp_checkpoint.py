"""Token-only checkpoint schema, actual sample accounting and exact CPU resume."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from scripts import cached_ve_text_features as cached
from scripts import residual_ddp_checkpoint as checkpoint
from scripts.residual_ddp_runtime import (
    DeterministicGlobalBatchSampler, capture_rng_state, restore_rng_state,
)
from scripts.residual_selection import new_selection_state, update_selection


def make_encoder():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, :4] = False
    return cached.CachedVETextEncoder(padding, torch.ones(32, 2, 256), torch.ones(32, 2, 1024),
        mode="zero_delta", metadata={"base_checkpoint_sha256": "a" * 64,
                                      "tokenizer_sha256": "b" * 64})


def configuration():
    return {"world_size": 3, "batch_size_per_rank": 2, "global_batch_size": 6,
            "dataset_size": 14, "steps_per_epoch": 2, "epochs": 3,
            "seed": 123, "learning_rate": .001, "weight_decay": 0., "anchor_weight": 0.,
            "base_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "annotations_sha256": "c" * 64, "approval_sha256": "d" * 64,
            "implementation_sha256": "e" * 64, "initial_cache_file_sha256": "f" * 64}


def advance(encoder, optimizer):
    optimizer.zero_grad(set_to_none=True)
    target = torch.rand_like(encoder.delta) + random.random() + float(np.random.random())
    loss = (encoder.delta - target).square().mean()
    loss.backward()
    optimizer.step()


def make_fixture(step=3):
    random.seed(841)
    np.random.seed(841)
    torch.manual_seed(841)
    config = configuration()
    image_ids = [200 + 7 * index for index in range(config["dataset_size"])]
    encoder = make_encoder()
    initial = deepcopy(encoder.state_dict())
    optimizer = torch.optim.AdamW([encoder.delta], lr=config["learning_rate"], weight_decay=0.)
    for _ in range(step):
        advance(encoder, optimizer)
    ranks = [{"image_ids": checkpoint.expected_rank_ids(image_ids, config, step, rank),
              "rng": capture_rng_state("cpu")} for rank in range(config["world_size"])]
    state = checkpoint.make_checkpoint(encoder, optimizer, config, step, initial, ranks)
    return state, config, image_ids, initial, encoder, optimizer


class CheckpointContractTest(unittest.TestCase):
    def test_optimizer_steps_and_actual_images_are_distinct_at_epoch_and_partial_boundaries(self):
        config = configuration()
        for step, epoch, offset in ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1), (6, 3, 0)):
            value = checkpoint.progress_at(step, config)
            self.assertEqual(value["global_step"], step)
            self.assertEqual(value["samples_seen"], step * 6)
            self.assertEqual(value["next_epoch"], epoch)
            self.assertEqual(value["next_step_in_epoch"], offset)
            self.assertEqual(value["training_complete"], step == 6)
            self.assertEqual(value["dropped_images_per_epoch"], 2)
            self.assertEqual(value["planned_samples"], 36)
        for invalid in (-1, 7, 1.5, True):
            with self.assertRaises(ValueError):
                checkpoint.progress_at(invalid, config)

    def test_actual_rank_histories_match_sampler_across_epoch_resume(self):
        state, config, ids, initial, _, _ = make_fixture(step=3)
        self.assertEqual(checkpoint.validate_resume(state, config, ids, initial), 3)
        expected = [[] for _ in range(config["world_size"])]
        for epoch in range(2):
            for rank in range(config["world_size"]):
                batches = list(DeterministicGlobalBatchSampler(len(ids), 2, rank=rank,
                    world_size=3, seed=config["seed"], epoch=epoch))
                for batch in batches[:2 if epoch == 0 else 1]:
                    expected[rank].extend(ids[index] for index in batch)
        self.assertEqual([rank["image_ids"] for rank in state["rank_states"]], expected)
        self.assertEqual(sum(len(rank["image_ids"]) for rank in state["rank_states"]), 18)
        self.assertEqual(state["progress"]["global_step"], 3)
        self.assertEqual(state["progress"]["samples_seen"], 18)

    def test_world_batch_learning_rate_and_source_fingerprint_changes_are_rejected(self):
        state, config, ids, initial, _, _ = make_fixture()
        for field, replacement in (("world_size", 4), ("batch_size_per_rank", 3),
                ("global_batch_size", 8), ("learning_rate", .02), ("seed", 9),
                ("base_sha256", "0" * 64), ("tokenizer_sha256", "0" * 64),
                ("annotations_sha256", "0" * 64), ("approval_sha256", "0" * 64),
                ("implementation_sha256", "0" * 64), ("initial_cache_file_sha256", "0" * 64)):
            changed = {**config, field: replacement}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "config mismatch"):
                checkpoint.validate_resume(state, changed, ids, initial)

    def test_progress_rng_and_observed_identity_corruption_are_rejected(self):
        original, config, ids, initial, _, _ = make_fixture()
        for mutation in ("samples", "epoch", "ids", "missing_rank", "rng", "rng_tensor", "config_hash"):
            state = deepcopy(original)
            if mutation == "samples":
                state["progress"]["samples_seen"] = state["progress"]["global_step"]
            elif mutation == "epoch":
                state["progress"]["next_step_in_epoch"] = 0
            elif mutation == "ids":
                state["rank_states"][1]["image_ids"][0] = -1
            elif mutation == "missing_rank":
                state["rank_states"].pop()
            elif mutation == "rng":
                state["rank_states"][0]["rng"] = {"bogus": 1}
            elif mutation == "rng_tensor":
                state["rank_states"][0]["rng"]["torch_cpu"] = torch.ones(4)
            else:
                state["config_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                checkpoint.validate_resume(state, config, ids, initial)

    def test_current_and_initial_cache_tampering_are_rejected(self):
        original, config, ids, initial, _, _ = make_fixture()
        for mutation in ("delta_dtype", "delta_nan", "frozen", "metadata", "initial", "initial_missing"):
            state = deepcopy(original)
            if mutation == "delta_dtype":
                state["cache_state_dict"]["delta"] = state["cache_state_dict"]["delta"].half()
            elif mutation == "delta_nan":
                state["cache_state_dict"]["delta"][0, 0, 0] = float("nan")
            elif mutation == "frozen":
                state["cache_state_dict"]["resized_cache"][0, 0, 0] += 1
            elif mutation == "metadata":
                state["cache_state_dict"]["_extra_state"]["metadata"]["base_checkpoint_sha256"] = "0" * 64
            elif mutation == "initial":
                state["initial_cache_state_dict"]["delta"][0, 0, 0] = .1
            else:
                state.pop("initial_cache_state_dict")
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                checkpoint.validate_resume(state, config, ids, initial)

    def test_adamw_strategy_step_and_moment_corruption_are_rejected(self):
        original, config, ids, initial, _, _ = make_fixture()
        for mutation in ("lr", "betas", "eps", "weight_decay", "maximize", "amsgrad", "step", "moment_dtype", "moment_nan"):
            state = deepcopy(original)
            group = state["optimizer"]["param_groups"][0]
            moments = state["optimizer"]["state"][0]
            if mutation == "lr":
                group["lr"] = .5
            elif mutation == "betas":
                group["betas"] = (.8, .9)
            elif mutation == "eps":
                group["eps"] = .01
            elif mutation == "weight_decay":
                group["weight_decay"] = .1
            elif mutation in ("maximize", "amsgrad"):
                group[mutation] = True
            elif mutation == "step":
                moments["step"] += 1
            elif mutation == "moment_dtype":
                moments["exp_avg"] = moments["exp_avg"].half()
            else:
                moments["exp_avg_sq"][0, 0, 0] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "optimizer|Optimizer|AdamW"):
                checkpoint.validate_resume(state, config, ids, initial)

    def test_zero_step_empty_optimizer_and_complete_checkpoint_validate(self):
        for step in (0, 6):
            state, config, ids, initial, _, _ = make_fixture(step)
            self.assertEqual(checkpoint.validate_resume(state, config, ids, initial), step)
            self.assertEqual(state["progress"]["training_complete"], step == 6)


class ValidationRecoveryContractTest(unittest.TestCase):
    def config(self, *, initial=True, every=1):
        return {**configuration(), "validation": {
            "enabled": True, "initial_validation": initial, "every_epochs": every,
            "annotations_sha256": "1" * 64,
            "selection_policy": {"patience": 3, "min_delta": .001}}}

    def value(self, steps=(), *, last=None, config=None):
        config = config or self.config()
        policy = config["validation"]
        selection = new_selection_state(policy["annotations_sha256"], **policy["selection_policy"])
        metrics = {"images": 3, "queries": 6, "targets": 4,
                   **{f"{side}/{key}": number for side in ("left_hand", "right_hand")
                      for key, number in {"positive_count": 2, "absent_count": 1,
                                          "false_negative_count": 0, "false_positive_count": 0,
                                          "miss_zero_dice": .5}.items()}}
        for step in steps:
            selection, _ = update_selection(selection, metrics, step=step,
                epoch=step // config["steps_per_epoch"], validation_sha256=policy["annotations_sha256"])
        return {"last_validation_step": last if last is not None else (steps[-1] if steps else None),
                "selection": selection}

    def test_baseline_epoch_pending_and_validated_boundaries(self):
        config = self.config()
        for step, selected_steps, expected_pending in (
                (0, (), 0), (0, (0,), None), (1, (0,), None),
                (2, (0,), 2), (2, (0, 2), None), (3, (0, 2), None),
                (4, (0, 2), 4), (4, (0, 2, 4), None)):
            with self.subTest(step=step, selected=selected_steps):
                value = self.value(selected_steps)
                validated = checkpoint.validate_validation_state(value, config, step)
                self.assertEqual(validated, value)
                self.assertIsNot(validated, value)
                self.assertEqual(checkpoint.pending_selection_validation_step(value, config, step), expected_pending)

    def test_skipped_required_epoch_or_baseline_rejected_after_training_advances(self):
        for step, selected in ((1, ()), (3, (0,)), (5, (0, 2)), (6, (0, 4))):
            with self.subTest(step=step, selected=selected), self.assertRaisesRegex(ValueError, "skipped"):
                checkpoint.validate_validation_state(self.value(selected), self.config(), step)

    def test_partial_diagnostic_is_retained_without_selecting_partial_epoch(self):
        value = self.value((0,), last=1)
        self.assertEqual(checkpoint.validate_validation_state(value, self.config(), 1), value)
        config = self.config(initial=False)
        partial_selection = self.value((1,), config=config)
        with self.assertRaisesRegex(ValueError, "partial epoch"):
            checkpoint.validate_validation_state(partial_selection, config, 1)

    def test_disabled_initial_and_every_two_epochs_schedules(self):
        config = self.config(initial=False, every=2)
        for step in (0, 1, 2, 3):
            self.assertIsNone(checkpoint.pending_selection_validation_step(self.value(config=config), config, step))
        self.assertEqual(checkpoint.pending_selection_validation_step(self.value(config=config), config, 4), 4)
        selected = self.value((4,), config=config)
        self.assertIsNone(checkpoint.pending_selection_validation_step(selected, config, 6))

    def test_terminal_selection_cannot_be_followed_by_more_optimizer_steps(self):
        config = self.config()
        config["validation"]["selection_policy"]["patience"] = 1
        stopped = self.value((0, 2), config=config)
        self.assertTrue(stopped["selection"]["stopped"])
        checkpoint.validate_validation_state(stopped, config, 2)
        with self.assertRaisesRegex(ValueError, "terminal"):
            checkpoint.validate_validation_state(stopped, config, 3)

    def test_invalid_last_step_history_epoch_scope_digest_and_policy_rejected(self):
        config = self.config()
        original = self.value((0, 2))
        for last in (None, -1, 1, 4, True, 3.0):
            changed = deepcopy(original)
            changed["last_validation_step"] = last
            with self.subTest(last=last), self.assertRaises(ValueError):
                checkpoint.validate_validation_state(changed, config, 3)
        for key, replacement in (("annotations_sha256", "2" * 64),
                                 ("selection_policy", {"patience": 4, "min_delta": .001}),
                                 ("every_epochs", True), ("initial_validation", 1)):
            changed = deepcopy(config)
            changed["validation"][key] = replacement
            with self.subTest(field=key), self.assertRaises(ValueError):
                checkpoint.validate_validation_state(original, changed, 3)
        wrong_epoch = self.value((2,), config=self.config(initial=False))
        wrong_epoch["selection"]["history"][0]["epoch"] = 2
        with self.assertRaises(ValueError):
            checkpoint.validate_validation_state(wrong_epoch, self.config(initial=False), 2)

    def test_new_checkpoint_requires_selection_and_roundtrip_preserves_pending_phase(self):
        original, old_config, ids, initial, encoder, optimizer = make_fixture(step=2)
        config = self.config()
        value = self.value((0,))
        saved = checkpoint.make_checkpoint(encoder, optimizer, config, 2, initial,
                                           original["rank_states"], validation_state=value)
        self.assertEqual(checkpoint.validate_resume(saved, config, ids, initial), 2)
        self.assertEqual(checkpoint.pending_selection_validation_step(saved["validation_state"], config, 2), 2)
        value["selection"]["bad_epochs"] = 999
        self.assertEqual(saved["validation_state"]["selection"]["bad_epochs"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.pt"
            checkpoint.atomic_checkpoint(path, saved)
            loaded = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint.validate_resume(loaded, config, ids, initial), 2)
            self.assertEqual(loaded["validation_state"], saved["validation_state"])
        missing = deepcopy(saved)
        del missing["validation_state"]
        with self.assertRaisesRegex(ValueError, "Missing checkpoint validation state"):
            checkpoint.validate_resume(missing, config, ids, initial)
        with self.assertRaises(ValueError):
            checkpoint.make_checkpoint(encoder, optimizer, config, 2, initial, original["rank_states"])
        # Old contract remains readable only with that exact old config, not
        # silently promoted to the new selection-aware training configuration.
        self.assertEqual(checkpoint.validate_resume(original, old_config, ids, initial), 2)
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            checkpoint.validate_resume(original, config, ids, initial)

    def test_disabled_validation_has_explicit_none_and_rejects_selector(self):
        original, config, ids, initial, encoder, optimizer = make_fixture(step=2)
        config = {**config, "validation": {"enabled": False}}
        saved = checkpoint.make_checkpoint(encoder, optimizer, config, 2, initial, original["rank_states"])
        self.assertIn("validation_state", saved)
        self.assertIsNone(saved["validation_state"])
        self.assertEqual(checkpoint.validate_resume(saved, config, ids, initial), 2)
        with self.assertRaisesRegex(ValueError, "Disabled validation"):
            checkpoint.validate_validation_state(self.value((0,)), config, 2)


class ExactResumeTest(unittest.TestCase):
    def test_weights_only_roundtrip_restores_optimizer_rng_and_next_updates_exactly(self):
        state, config, ids, initial, encoder, optimizer = make_fixture(step=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.pt"
            checkpoint.atomic_checkpoint(path, state)
            with self.assertRaises(FileExistsError):
                checkpoint.atomic_checkpoint(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            rng_before_validation = capture_rng_state("cpu")
            self.assertEqual(checkpoint.validate_resume(loaded, config, ids, initial), 3)
            after_validation = capture_rng_state("cpu")
            self.assertEqual(rng_before_validation["python"], after_validation["python"])
            torch.testing.assert_close(rng_before_validation["numpy"]["keys"], after_validation["numpy"]["keys"])
            torch.testing.assert_close(rng_before_validation["torch_cpu"], after_validation["torch_cpu"])
            for _ in range(2):
                advance(encoder, optimizer)
            expected_delta = encoder.delta.detach().clone()
            expected_optimizer = deepcopy(optimizer.state_dict())
            expected_random = (random.random(), float(np.random.random()), torch.rand(5))

            resumed = make_encoder()
            resumed_optimizer = torch.optim.AdamW([resumed.delta], lr=config["learning_rate"], weight_decay=0.)
            resumed.load_state_dict(loaded["cache_state_dict"])
            resumed_optimizer.load_state_dict(loaded["optimizer"])
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restore_rng_state(loaded["rank_states"][0]["rng"], "cpu")
            for _ in range(2):
                advance(resumed, resumed_optimizer)
            self.assertTrue(torch.equal(resumed.delta, expected_delta))
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(resumed_optimizer.state_dict()["state"][0][key],
                                            expected_optimizer["state"][0][key]))
            actual_random = (random.random(), float(np.random.random()), torch.rand(5))
            self.assertEqual(actual_random[:2], expected_random[:2])
            self.assertTrue(torch.equal(actual_random[2], expected_random[2]))
            self.assertEqual(loaded["rank_states"][0]["image_ids"], state["rank_states"][0]["image_ids"])


class RankCacheAuditTest(unittest.TestCase):
    def test_three_four_rank_actual_states_and_gathered_hashes_produce_same_audit(self):
        state = deepcopy(make_encoder().state_dict())
        state["delta"][0, 0, 0] = .03125
        fingerprint = checkpoint.shared.cache_fingerprint(state)
        rng_before = capture_rng_state("cpu")
        for world in (3, 4):
            with self.subTest(world=world):
                rank_states = [deepcopy(state) for _ in range(world)]
                audit = checkpoint.validate_rank_cache_consistency(rank_states, world_size=world)
                from_hashes = checkpoint.validate_rank_cache_consistency([fingerprint] * world, world_size=world)
                self.assertEqual(audit, from_hashes)
                self.assertEqual(audit["rank_cache_sha256"], [fingerprint] * world)
                self.assertEqual(audit["cache_sha256"], fingerprint)
                self.assertEqual(audit["world_size"], world)
                self.assertTrue(audit["all_ranks_identical"])
                for unchanged in rank_states:
                    self.assertEqual(checkpoint.shared.cache_fingerprint(unchanged), fingerprint)
        rng_after = capture_rng_state("cpu")
        self.assertEqual(rng_before["python"], rng_after["python"])
        self.assertTrue(torch.equal(rng_before["numpy"]["keys"], rng_after["numpy"]["keys"]))
        self.assertTrue(torch.equal(rng_before["torch_cpu"], rng_after["torch_cpu"]))

    def test_one_rank_residual_frozen_buffer_or_metadata_divergence_is_rejected(self):
        for mutation in ("delta", "resized_cache", "metadata"):
            with self.subTest(mutation=mutation):
                states = [deepcopy(make_encoder().state_dict()) for _ in range(4)]
                if mutation == "metadata":
                    states[2]["_extra_state"]["metadata"]["base_checkpoint_sha256"] = "0" * 64
                else:
                    states[2][mutation].reshape(-1)[0] += .125
                with self.assertRaisesRegex(ValueError, r"rank 0 on ranks \[2\]"):
                    checkpoint.validate_rank_cache_consistency(states, world_size=4)
                hashes = [checkpoint.shared.cache_fingerprint(state) for state in states]
                with self.assertRaisesRegex(ValueError, r"rank 0 on ranks \[2\]"):
                    checkpoint.validate_rank_cache_consistency(hashes, world_size=4)

    def test_missing_rank_malformed_digest_and_invalid_actual_state_are_rejected(self):
        for values, world in (([], 3), (["a" * 64] * 2, 3), (["a" * 64] * 4, 3),
                              (["a" * 64], True), (["a" * 64], 0),
                              (["A" * 64], 1), (["a" * 63], 1), ([None], 1), ([{}], 1)):
            with self.subTest(values=values, world=world), self.assertRaises(ValueError):
                checkpoint.validate_rank_cache_consistency(values, world_size=world)
        state = deepcopy(make_encoder().state_dict())
        for delta in (torch.zeros(2, 4, 256, dtype=torch.float64), torch.zeros(2048),
                      torch.full((2, 4, 256), float("nan"))):
            with self.assertRaisesRegex(ValueError, "invalid output residual"):
                checkpoint.validate_rank_cache_consistency([{**state, "delta": delta}])

    def test_checkpoint_audit_binds_actual_saved_encoder_and_weights_only_roundtrip(self):
        state, config, ids, initial, encoder, optimizer = make_fixture(step=3)
        hashes = [checkpoint.shared.cache_fingerprint(encoder.state_dict())] * config["world_size"]
        audit = checkpoint.validate_rank_cache_consistency(hashes, world_size=config["world_size"])
        state = checkpoint.make_checkpoint(encoder, optimizer, config, 3, initial,
                                           state["rank_states"], rank_cache_audit=audit)
        self.assertEqual(checkpoint.validate_resume(state, config, ids, initial), 3)
        # make_checkpoint snapshots the supplied audit instead of aliasing it.
        audit["rank_cache_sha256"][0] = "0" * 64
        self.assertNotEqual(audit, state["rank_cache_audit"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audited.pt"
            checkpoint.atomic_checkpoint(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint.validate_resume(loaded, config, ids, initial), 3)
            self.assertEqual(loaded["rank_cache_audit"], state["rank_cache_audit"])
        stale = checkpoint.validate_rank_cache_consistency(["b" * 64] * config["world_size"])
        with self.assertRaisesRegex(ValueError, "actual encoder state"):
            checkpoint.make_checkpoint(encoder, optimizer, config, 3, initial,
                                       state["rank_states"], rank_cache_audit=stale)

    def test_audit_or_saved_delta_tampering_fails_but_legacy_checkpoint_remains_compatible(self):
        legacy, config, ids, initial, _, _ = make_fixture(step=3)
        self.assertNotIn("rank_cache_audit", legacy)
        self.assertEqual(checkpoint.validate_resume(legacy, config, ids, initial), 3)
        audit = checkpoint.validate_rank_cache_consistency(
            [checkpoint.shared.cache_fingerprint(legacy["cache_state_dict"])] * config["world_size"])
        for field in ("world_size", "rank_cache_sha256", "cache_sha256", "all_ranks_identical", "delta"):
            with self.subTest(field=field):
                changed = deepcopy(legacy)
                changed["rank_cache_audit"] = deepcopy(audit)
                if field == "delta":
                    changed["cache_state_dict"]["delta"][0, 0, 0] += .001
                elif field == "rank_cache_sha256":
                    changed["rank_cache_audit"][field].pop()
                else:
                    changed["rank_cache_audit"][field] = {"world_size": 4, "cache_sha256": "f" * 64,
                                                           "all_ranks_identical": False}[field]
                with self.assertRaisesRegex(ValueError, "Rank cache audit"):
                    checkpoint.validate_resume(changed, config, ids, initial)


if __name__ == "__main__":
    unittest.main()

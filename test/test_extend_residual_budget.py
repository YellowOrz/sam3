"""A budget increase is explicit, identity-bound and exact-resume compatible."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts import extend_residual_budget as extension
from scripts import residual_ddp_checkpoint as checkpoint
from scripts import train_ve_initialized_tokens as shared
from scripts.residual_ddp_runtime import restore_rng_state
from scripts.residual_selection import new_selection_state, update_selection
from test_residual_ddp_checkpoint import advance, make_encoder, make_fixture


def fixture(step=6, *, pending=False, scores=(.3, .4, .5, .5005)):
    state, config, ids, initial, encoder, optimizer = make_fixture(step)
    # Match the production initializer's verified precision contract.
    initial["resized_cache"] = initial["resized_cache"].bfloat16()
    initial["_extra_state"]["metadata"]["resized_dtype"] = "torch.bfloat16"
    state["initial_cache_state_dict"] = deepcopy(initial)
    state["cache_state_dict"]["resized_cache"] = initial["resized_cache"].clone()
    state["cache_state_dict"]["_extra_state"] = deepcopy(initial["_extra_state"])
    state["initial_cache_sha256"] = shared.cache_fingerprint(initial)
    config.update(image_order_sha256=checkpoint.canonical_hash(ids),
        initial_state_sha256=shared.cache_fingerprint(initial), validation={
            "enabled": True, "initial_validation": True, "every_epochs": 1,
            "annotations_sha256": "1" * 64,
            "selection_policy": {"patience": 3, "min_delta": .001}})
    policy = config["validation"]
    selection = new_selection_state(policy["annotations_sha256"], **policy["selection_policy"])
    selected_steps = list(range(0, step + 1, config["steps_per_epoch"]))
    if pending and selected_steps[-1] == step:
        selected_steps.pop()
    for index, selected_step in enumerate(selected_steps):
        metrics = {"images": 3, "queries": 6, "targets": 4,
            **{f"{side}/{key}": number for side in ("left_hand", "right_hand")
               for key, number in {"positive_count": 2, "absent_count": 1,
                   "false_negative_count": 0, "false_positive_count": 0,
                   "miss_zero_dice": scores[index]}.items()}}
        selection, _ = update_selection(selection, metrics, step=selected_step,
            epoch=selected_step // config["steps_per_epoch"],
            validation_sha256=policy["annotations_sha256"])
    state["validation_state"] = {"last_validation_step": selected_steps[-1], "selection": selection}
    state["training_config"] = deepcopy(config)
    state["config_sha256"] = checkpoint.canonical_hash(config)
    checkpoint.validate_resume(state, config, ids, initial)
    return state, config, ids, initial


def extend(state, ids, initial, epochs=5, **kwargs):
    return extension.extend_budget(state, epochs, ids, initial,
        kwargs.pop("source_sha256", "9" * 64), kwargs.pop("reason", "Two additional epochs; all else fixed"),
        annotations_sha256=kwargs.pop("annotations_sha256", "c" * 64),
        initial_cache_file_sha256=kwargs.pop("initial_cache_file_sha256", "f" * 64), **kwargs)


class ExtendResidualBudgetTest(unittest.TestCase):
    def assert_nested_equal(self, first, second):
        self.assertIs(type(first), type(second))
        if isinstance(first, torch.Tensor):
            self.assertEqual(first.dtype, second.dtype)
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_only_budget_derived_fields_change_and_input_is_independent(self):
        state, config, ids, initial = fixture()
        before = deepcopy(state)
        result = extend(state, ids, initial)
        self.assert_nested_equal(state, before)
        self.assertEqual(result["training_config"], dict(config, epochs=5))
        self.assertEqual(result["progress"], checkpoint.progress_at(6, result["training_config"]))
        self.assertFalse(result["progress"]["training_complete"])
        self.assertEqual(result["progress"]["planned_steps"], 10)
        self.assertEqual(result["progress"]["samples_seen"], 36)
        self.assertEqual(result["validation_state"]["selection"]["bad_epochs"], 1)
        for key in state.keys() - {"training_config", "config_sha256", "progress"}:
            self.assert_nested_equal(result[key], state[key])
        self.assertEqual(result["budget_extension"]["source_sha256"], "9" * 64)
        checkpoint.validate_resume(result, result["training_config"], ids, initial)
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            checkpoint.validate_resume(state, result["training_config"], ids, initial)
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            checkpoint.validate_resume(result, config, ids, initial)
        result["cache_state_dict"]["delta"].add_(1)
        result["rank_states"][0]["image_ids"][0] = -1
        self.assert_nested_equal(state, before)

    def test_rejects_nonincrease_nonintegers_reason_and_bad_digest(self):
        state, _, ids, initial = fixture()
        for epochs in (0, 2, 3, True, 5., "5"):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                extend(state, ids, initial, epochs=epochs)
        for reason in ("", "  ", None, 3):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                extend(state, ids, initial, reason=reason)
        for digest in ("", "z" * 64, "A" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                extend(state, ids, initial, source_sha256=digest)

    def test_rejects_partial_unvalidated_stopped_and_nonbest(self):
        for args, expected in (({"step": 5}, "completed training budgets"),
                ({"pending": True}, "final epoch"),
                ({"scores": (.3, .3, .3, .3)}, "terminal early-stop"),
                ({"scores": (.3, .4, .5, .49)}, "selected best")):
            state, _, ids, initial = fixture(**args)
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, expected):
                extend(state, ids, initial)
        state, config, ids, initial = fixture()
        config["validation"] = {"enabled": False}
        state["training_config"] = config
        state["config_sha256"] = checkpoint.canonical_hash(config)
        state["validation_state"] = None
        with self.assertRaisesRegex(ValueError, "fully validated"):
            extend(state, ids, initial)

    def test_rejects_external_identity_cache_bindings_and_corrupt_recovery(self):
        original, _, ids, initial = fixture()
        for kwargs in ({"annotations_sha256": "d" * 64},
                       {"initial_cache_file_sha256": "a" * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                extend(original, ids, initial, **kwargs)
        for changed_ids in (list(reversed(ids)), ids[:-1], [*ids[:-1], ids[0]],
                            [*ids[:-1], 9999]):
            with self.subTest(ids=changed_ids), self.assertRaises(ValueError):
                extend(original, changed_ids, initial)
        modified_initial = deepcopy(initial)
        modified_initial["resized_cache"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "semantic state"):
            extend(original, ids, modified_initial)
        for mutation in ("optimizer", "rng", "rank_ids", "frozen", "config", "selection"):
            state = deepcopy(original)
            if mutation == "optimizer":
                state["optimizer"]["state"][0]["exp_avg"][0, 0, 0] = float("nan")
            elif mutation == "rng":
                del state["rank_states"][0]["rng"]["torch_cpu"]
            elif mutation == "rank_ids":
                state["rank_states"][0]["image_ids"][0] = -1
            elif mutation == "frozen":
                state["cache_state_dict"]["raw_cache"][0, 0, 0] += 1
            elif mutation == "config":
                state["config_sha256"] = "0" * 64
            else:
                state["validation_state"]["selection"]["bad_epochs"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                extend(state, ids, initial)

    def test_roundtrip_and_next_update_preserve_adamw_rng(self):
        state, _, ids, initial = fixture()
        result = extend(state, ids, initial)
        outputs = []
        for source in (state, result):
            encoder = make_encoder()
            encoder.resized_cache = encoder.resized_cache.bfloat16()
            encoder.load_state_dict(source["cache_state_dict"])
            optimizer = torch.optim.AdamW([encoder.delta], lr=source["training_config"]["learning_rate"], weight_decay=0.)
            optimizer.load_state_dict(deepcopy(source["optimizer"]))
            restore_rng_state(source["rank_states"][0]["rng"], "cpu")
            advance(encoder, optimizer)
            outputs.append((encoder.delta.detach().clone(), deepcopy(optimizer.state_dict())))
        self.assert_nested_equal(outputs[0], outputs[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.pt"
            checkpoint.atomic_checkpoint(path, result)
            restored = torch.load(path, weights_only=True)
            checkpoint.validate_resume(restored, result["training_config"], ids, initial)
            self.assert_nested_equal(restored, result)

    def test_cli_binds_real_annotation_cache_files_and_refuses_overwrite(self):
        state, config, ids, initial = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = json.dumps({"images": [{"id": value} for value in reversed(ids)]}).encode()
            (root / "annotations.json").write_bytes(annotations)
            cache = root / "cache.pt"
            torch.save(initial, cache)
            config["annotations_sha256"] = hashlib.sha256(annotations).hexdigest()
            config["initial_cache_file_sha256"] = shared.evaluation.sha256(cache)
            state["training_config"] = config
            state["config_sha256"] = checkpoint.canonical_hash(config)
            source, output = root / "source.pt", root / "extended.pt"
            torch.save(state, source)
            source_hash = shared.evaluation.sha256(source)
            argv = ["--source", str(source), "--output", str(output), "--epochs", "5",
                "--training-data-root", str(root), "--initial-cache", str(cache),
                "--reason", "Explicit two-epoch continuation"]
            with redirect_stdout(io.StringIO()):
                extension.main(argv)
            result = torch.load(output, weights_only=True)
            self.assertEqual(result["budget_extension"]["source_sha256"], source_hash)
            self.assertEqual(shared.evaluation.sha256(source), source_hash)
            checkpoint.validate_resume(result, dict(config, epochs=5), ids, initial)
            output_hash = shared.evaluation.sha256(output)
            with self.assertRaises(FileExistsError):
                extension.main(argv)
            self.assertEqual(shared.evaluation.sha256(output), output_hash)
            altered_argv = [str(root / "bad.pt") if value == str(output) else value for value in argv]
            (root / "annotations.json").write_bytes(annotations + b"\n")
            with self.assertRaisesRegex(ValueError, "annotation SHA256"):
                extension.main(altered_argv)
            self.assertFalse((root / "bad.pt").exists())
            (root / "annotations.json").write_bytes(annotations)
            bad_initial = deepcopy(initial)
            bad_initial["_extra_state"]["metadata"]["base_checkpoint_sha256"] = "d" * 64
            torch.save(bad_initial, cache)
            with self.assertRaisesRegex(ValueError, "base/tokenizer"):
                extension.main(altered_argv)
            self.assertFalse((root / "bad.pt").exists())


if __name__ == "__main__":
    unittest.main()

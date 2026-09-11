"""CPU-only standard-library tests; does not import the repository or PyTorch."""

import copy
import json
import math
import unittest

from scripts.residual_selection import (
    METRIC_NAME,
    new_selection_state,
    restore_selection_state,
    selection_metric,
    serialize_selection_state,
    update_selection,
)


DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def metrics(left=0.5, right=None):
    # Deliberately unequal left/right populations distinguish macro averaging
    # from a pooled, positive-count-weighted score.
    return {
        "images": 12, "queries": 24, "targets": 12,
        "left_hand/positive_count": 10, "left_hand/absent_count": 2,
        "left_hand/false_negative_count": 1, "left_hand/false_positive_count": 1,
        "right_hand/positive_count": 2, "right_hand/absent_count": 10,
        "right_hand/false_negative_count": 0, "right_hand/false_positive_count": 3,
        "left_hand/miss_zero_dice": left,
        "right_hand/miss_zero_dice": left if right is None else right,
        "left_hand/candidate_dice": 0.9, "right_hand/candidate_dice": 0.9,
        "left_hand/false_negative_rate": 0.1,
        "right_hand/false_negative_rate": 0.0,
        "left_hand/false_positive_rate": 0.5,
        "right_hand/false_positive_rate": 0.3,
        "loss_mask": 0.8, "total_loss": 2.3,
    }


def advance(state, score, step, epoch, **kwargs):
    return update_selection(state, metrics(score), step=step, epoch=epoch,
                            validation_sha256=DIGEST, **kwargs)


class SelectionMetricTests(unittest.TestCase):
    def test_macro_equal_side_not_weighted_or_candidate_dice(self):
        values = metrics(0.2, 0.8)
        self.assertEqual(selection_metric(values), 0.5)
        self.assertNotAlmostEqual(selection_metric(values), (10 * .2 + 2 * .8) / 12)
        self.assertNotEqual(selection_metric(values), values["left_hand/candidate_dice"])

    def test_full_missing_or_nonfinite_metrics_are_rejected(self):
        for key in metrics():
            for value in (None, float("nan"), float("inf"), -float("inf"), True, "0.2"):
                with self.subTest(key=key, value=value):
                    values = metrics()
                    values[key] = value
                    with self.assertRaises(ValueError):
                        selection_metric(values)
        for side in ("left_hand", "right_hand"):
            for name in ("miss_zero_dice", "positive_count", "absent_count",
                         "false_negative_count", "false_positive_count"):
                with self.subTest(side=side, name=name):
                    values = metrics()
                    del values[f"{side}/{name}"]
                    with self.assertRaises(ValueError):
                        selection_metric(values)

    def test_main_metric_range_and_positive_requirement(self):
        for side in ("left_hand", "right_hand"):
            for value in (-0.001, 1.001):
                values = metrics()
                values[f"{side}/miss_zero_dice"] = value
                with self.assertRaises(ValueError):
                    selection_metric(values)
            values = metrics()
            values[f"{side}/positive_count"] = 0
            values[f"{side}/absent_count"] = 12
            with self.assertRaises(ValueError):
                selection_metric(values)

    def test_counts_integrity_and_query_coverage(self):
        bad_values = {
            "images": [0, 12.0, True], "queries": [23, 25], "targets": [11, 13],
            "left_hand/positive_count": [11, -1, 10.0],
            "left_hand/absent_count": [1, 3],
            "left_hand/false_negative_count": [-1, 11, .5],
            "right_hand/false_positive_count": [-1, 11, .5],
        }
        for key, replacements in bad_values.items():
            for value in replacements:
                with self.subTest(key=key, value=value):
                    values = metrics()
                    values[key] = value
                    with self.assertRaises(ValueError):
                        selection_metric(values)

    def test_empty_or_nonnumeric_payload_rejected(self):
        for value in (None, {}, [], "metrics"):
            with self.assertRaises(ValueError):
                selection_metric(value)
        values = metrics()
        values[None] = 1
        with self.assertRaises(ValueError):
            selection_metric(values)

    def test_overflowing_integer_metric_is_rejected_as_invalid(self):
        values = metrics()
        values["total_loss"] = 10 ** 10000
        with self.assertRaises(ValueError):
            selection_metric(values)


class SelectionTransitionTests(unittest.TestCase):
    def test_initial_defaults_and_nonzero_initial_step(self):
        state = new_selection_state(DIGEST)
        self.assertEqual(state["policy"], {"patience": 3, "min_delta": .001})
        self.assertEqual(state["metric_name"], METRIC_NAME)
        self.assertIsNone(state["best_metric"])
        updated, decision = advance(state, .4, 500, 7)
        self.assertTrue(decision["is_baseline"])
        self.assertTrue(decision["is_best"])
        self.assertEqual(updated["best_step"], 500)
        self.assertEqual(updated["best_epoch"], 7)
        self.assertEqual(updated["bad_epochs"], 0)
        self.assertFalse(decision["should_stop"])

    def test_default_patience_counts_three_bad_evaluated_epochs(self):
        state, _ = advance(new_selection_state(DIGEST), .6, 0, 0)
        for epoch, score in enumerate((.5, .6, .59), 1):
            state, decision = advance(state, score, 100 * epoch, epoch)
            self.assertEqual(state["bad_epochs"], epoch)
            self.assertEqual(decision["should_stop"], epoch == 3)
        self.assertEqual(state["best_step"], 0)
        self.assertEqual(decision["stop_reason"], "patience_exhausted")
        with self.assertRaisesRegex(ValueError, "terminal"):
            advance(state, .9, 400, 4)

    def test_true_best_small_gain_saved_without_resetting_patience(self):
        state, _ = advance(new_selection_state(DIGEST, min_delta=.01), .5, 0, 0)
        state, decision = advance(state, .501, 10, 1)
        self.assertTrue(decision["is_best"])
        self.assertFalse(decision["significant_improvement"])
        self.assertEqual(state["best_metric"], .501)
        self.assertEqual(state["best_step"], 10)
        self.assertEqual(state["early_stop_reference_metric"], .5)
        self.assertEqual(state["bad_epochs"], 1)

    def test_small_gains_accumulate_against_separate_reference(self):
        state, _ = advance(new_selection_state(DIGEST), .6, 0, 0)
        state, _ = advance(state, .6006, 10, 1)
        state, decision = advance(state, .6012, 20, 2)
        self.assertTrue(decision["significant_improvement"])
        self.assertEqual(state["early_stop_reference_metric"], .6012)
        self.assertEqual(state["bad_epochs"], 0)

    def test_min_delta_is_strict_greater_and_ties_keep_earlier_best(self):
        # Binary-exact values make the equality boundary unambiguous.
        state, _ = advance(new_selection_state(DIGEST, min_delta=.125), .5, 0, 0)
        state, decision = advance(state, .625, 10, 1)
        self.assertFalse(decision["significant_improvement"])
        self.assertTrue(decision["is_best"])
        state, decision = advance(state, .625, 20, 2)
        self.assertFalse(decision["is_best"])
        self.assertEqual(state["best_step"], 10)
        state, decision = advance(state, .75, 30, 3)
        self.assertTrue(decision["significant_improvement"])
        self.assertEqual(state["bad_epochs"], 0)

    def test_new_best_and_early_stop_can_coincide(self):
        state, _ = advance(new_selection_state(DIGEST, patience=1, min_delta=.01), .5, 0, 0)
        state, decision = advance(state, .501, 10, 1)
        self.assertTrue(decision["is_best"])
        self.assertTrue(decision["should_stop"])
        self.assertEqual(state["best_step"], 10)

    def test_zero_delta_accepts_any_positive_gain_but_not_tie(self):
        state, _ = advance(new_selection_state(DIGEST, min_delta=0), .5, 0, 0)
        state, decision = advance(state, .500001, 10, 1)
        self.assertTrue(decision["significant_improvement"])
        state, decision = advance(state, .500001, 20, 2)
        self.assertFalse(decision["significant_improvement"])
        self.assertFalse(decision["is_best"])
        self.assertEqual(state["bad_epochs"], 1)

    def test_duplicate_or_regressing_steps_and_epochs_rejected(self):
        state, _ = advance(new_selection_state(DIGEST), .5, 10, 2)
        for step, epoch in ((10, 2), (10, 3), (9, 3), (11, 2), (11, 1),
                            (-1, 3), (11, -1), (True, 3), (11.0, 3), (11, 3.0)):
            with self.subTest(step=step, epoch=epoch):
                with self.assertRaises(ValueError):
                    advance(state, .5, step, epoch)

    def test_skipped_epochs_are_not_invented_as_bad_epochs(self):
        state, _ = advance(new_selection_state(DIGEST), .5, 0, 0)
        state, _ = advance(state, .4, 100, 10)
        self.assertEqual(state["bad_epochs"], 1)

    def test_scope_and_fixed_validation_digest_enforced(self):
        for scope in ("validation", "test", "realsense", "dexycb_test", ""):
            with self.subTest(scope=scope):
                with self.assertRaises(ValueError):
                    advance(new_selection_state(DIGEST), .5, 0, 0, scope=scope)
        with self.assertRaises(ValueError):
            update_selection(new_selection_state(DIGEST), metrics(), step=0, epoch=0,
                             validation_sha256=OTHER_DIGEST)

    def test_fixed_validation_population_counts_enforced(self):
        state, _ = advance(new_selection_state(DIGEST), .5, 0, 0)
        values = metrics()
        values["left_hand/positive_count"] = 9
        values["left_hand/absent_count"] = 3
        values["targets"] = 11
        with self.assertRaisesRegex(ValueError, "population"):
            update_selection(state, values, step=10, epoch=1, validation_sha256=DIGEST)

    def test_full_metrics_and_fp_fn_retained_but_not_selection_criteria(self):
        values = metrics(.2, .8)
        state, decision = update_selection(new_selection_state(DIGEST), values,
                                          step=0, epoch=0, validation_sha256=DIGEST)
        self.assertEqual(decision["metric"], .5)
        self.assertEqual(state["history"][0]["metrics"], values)
        values = metrics(.2, .8)
        values["left_hand/false_negative_count"] = 2
        values["right_hand/false_positive_count"] = 0
        state, decision = update_selection(state, values, step=10, epoch=1,
                                          validation_sha256=DIGEST)
        self.assertFalse(decision["is_best"])
        self.assertEqual(state["history"][1]["metrics"]["left_hand/false_negative_count"], 2)

    def test_transition_is_pure_and_returned_decision_does_not_alias_history(self):
        state = new_selection_state(DIGEST)
        values = metrics()
        original_state, original_values = copy.deepcopy(state), copy.deepcopy(values)
        updated, decision = update_selection(state, values, step=0, epoch=0,
                                             validation_sha256=DIGEST)
        self.assertEqual(state, original_state)
        self.assertEqual(values, original_values)
        values["left_hand/miss_zero_dice"] = 0
        decision["is_best"] = False
        self.assertEqual(updated["history"][0]["metrics"]["left_hand/miss_zero_dice"], .5)
        self.assertTrue(updated["history"][0]["decision"]["is_best"])
        before = copy.deepcopy(updated)
        with self.assertRaises(ValueError):
            update_selection(updated, {"bad": math.nan}, step=10, epoch=1,
                             validation_sha256=DIGEST)
        self.assertEqual(updated, before)


class SelectionCheckpointTests(unittest.TestCase):
    def populated(self):
        state = new_selection_state(DIGEST)
        for epoch, score in enumerate((.5, .49, .5005, .6)):
            state, _ = advance(state, score, epoch * 10, epoch)
        return state

    def test_empty_and_populated_roundtrip_and_resume_equivalence(self):
        for state in (new_selection_state(DIGEST), self.populated()):
            for payload in (state, serialize_selection_state(state)):
                with self.subTest(empty=not state["history"], text=isinstance(payload, str)):
                    restored = restore_selection_state(payload, validation_sha256=DIGEST)
                    self.assertEqual(restored, state)
                    self.assertIsNot(restored, state)
                    self.assertEqual(serialize_selection_state(restored), serialize_selection_state(state))
                    next_step = 0 if state["last_validation_step"] is None else state["last_validation_step"] + 10
                    next_epoch = 0 if state["last_validation_epoch"] is None else state["last_validation_epoch"] + 1
                    self.assertEqual(advance(state, .61, next_step, next_epoch),
                                     advance(restored, .61, next_step, next_epoch))
        restored["history"][0]["metrics"]["total_loss"] = 500
        self.assertNotEqual(restored, state)

    def test_json_checkpoint_contains_no_special_values(self):
        payload = serialize_selection_state(new_selection_state(DIGEST))
        self.assertNotIn("Infinity", payload)
        self.assertNotIn("NaN", payload)
        self.assertIsNone(json.loads(payload)["best_metric"])

    def test_duplicate_json_object_keys_rejected(self):
        payload = serialize_selection_state(new_selection_state(DIGEST))
        # The duplicate's final value matches the true value; accepting it
        # would hide an ambiguous checkpoint even though replay can succeed.
        payload = payload.replace('"bad_epochs":0', '"bad_epochs":99,"bad_epochs":0')
        with self.assertRaises(ValueError):
            restore_selection_state(payload, validation_sha256=DIGEST)

    def test_stopped_state_restores_terminal_without_consuming_patience(self):
        state, _ = advance(new_selection_state(DIGEST, patience=1), .5, 0, 0)
        state, _ = advance(state, .4, 10, 1)
        restored = restore_selection_state(serialize_selection_state(state),
                                           validation_sha256=DIGEST, patience=1)
        self.assertTrue(restored["stopped"])
        self.assertEqual(restored["bad_epochs"], 1)
        with self.assertRaises(ValueError):
            advance(restored, .8, 20, 2)

    def test_resume_rejects_changed_policy_validation_or_legacy_missing_state(self):
        state = self.populated()
        for kwargs in ({"patience": 4}, {"min_delta": .002},
                       {"validation_sha256": OTHER_DIGEST}):
            options = {"validation_sha256": DIGEST, **kwargs}
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    restore_selection_state(state, **options)
        for payload in (None, {}, "{}", "not json", []):
            with self.assertRaises(ValueError):
                restore_selection_state(payload, validation_sha256=DIGEST)

    def test_replay_rejects_inconsistent_derived_state_fields(self):
        original = self.populated()
        edits = {
            "schema": "v999", "metric_name": "accuracy", "mode": "min",
            "validation_scope": "test", "best_metric": .99, "best_step": 29,
            "best_epoch": 2, "early_stop_reference_metric": .59,
            "bad_epochs": 3, "last_validation_step": 31,
            "last_validation_epoch": 4, "stopped": True,
        }
        for key, value in edits.items():
            with self.subTest(key=key):
                state = copy.deepcopy(original)
                state[key] = value
                with self.assertRaises(ValueError):
                    restore_selection_state(state, validation_sha256=DIGEST)
                with self.assertRaises(ValueError):
                    serialize_selection_state(state)
                with self.assertRaises(ValueError):
                    advance(state, .7, 40, 4)

    def test_replay_rejects_corrupt_history_and_unknown_fields(self):
        def add_unknown(state):
            state["unrecognized"] = 1

        def omit_required(state):
            del state["best_metric"]

        def change_decision(state):
            state["history"][1]["decision"]["is_best"] = True

        def change_score(state):
            state["history"][1]["metrics"]["left_hand/miss_zero_dice"] = math.nan

        def duplicate_step(state):
            state["history"][1]["step"] = 0

        def different_scope(state):
            state["history"][1]["scope"] = "test"

        def unknown_record_field(state):
            state["history"][1]["unknown"] = 0

        def missing_record_field(state):
            del state["history"][1]["metrics"]

        def nonlist_history(state):
            state["history"] = {}

        def malformed_record(state):
            state["history"][0] = 42

        for mutate in (add_unknown, omit_required, change_decision, change_score,
                       duplicate_step, different_scope, unknown_record_field,
                       missing_record_field, nonlist_history, malformed_record):
            with self.subTest(mutation=mutate.__name__):
                state = self.populated()
                mutate(state)
                with self.assertRaises(ValueError):
                    restore_selection_state(state, validation_sha256=DIGEST)

    def test_bool_cannot_impersonate_derived_integer_on_restore(self):
        state = new_selection_state(DIGEST)
        state["bad_epochs"] = False
        with self.assertRaises(ValueError):
            restore_selection_state(state, validation_sha256=DIGEST)

    def test_invalid_policy_and_digest_rejected(self):
        for patience in (0, -1, 1.0, True, None):
            with self.assertRaises(ValueError):
                new_selection_state(DIGEST, patience=patience)
        for delta in (-.1, math.nan, math.inf, True, "0.1", None):
            with self.assertRaises(ValueError):
                new_selection_state(DIGEST, min_delta=delta)
        for digest in ("", "a" * 63, "A" * 64, "g" * 64, None):
            with self.assertRaises(ValueError):
                new_selection_state(digest)


if __name__ == "__main__":
    unittest.main()

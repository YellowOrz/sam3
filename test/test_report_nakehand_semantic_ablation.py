from copy import deepcopy
import unittest

import torch

from scripts.report_nakehand_semantic_ablation import compare_training_contracts, fraction, validate_summary


class SemanticAblationReportTests(unittest.TestCase):
    def states(self):
        state = {
            "format": "sam3-nakehand-semantic-delta-training-v1",
            "training_config": {"anchor_weight": 0., "learning_rate": .001, "seed": 123},
            "progress": {"completed_steps": 2000, "samples_seen": 2000, "pilot_complete": True},
            "next_step": 2000, "planned_dataset_indices": list(range(2000)),
            "observed_image_ids": list(range(4000, 6000)),
            "initial_cache_state_dict": {"delta": torch.zeros(2, 4, 256), "_extra_state": {"mode": "zero_delta"}},
        }
        other = deepcopy(state)
        other["training_config"]["anchor_weight"] = 1.
        return state, other

    def test_exact_single_variable_comparison(self):
        result = compare_training_contracts(*self.states())
        self.assertTrue(result["only_anchor_weight_differs"])
        self.assertEqual(result["actual_samples_per_trial"], 2000)

    def test_reject_changed_lr_sample_order_or_initial_semantics(self):
        a, b = self.states()
        b["training_config"]["learning_rate"] = .01
        with self.assertRaises(ValueError):
            compare_training_contracts(a, b)
        a, b = self.states()
        b["observed_image_ids"].reverse()
        with self.assertRaises(ValueError):
            compare_training_contracts(a, b)
        a, b = self.states()
        b["initial_cache_state_dict"]["delta"][0, 0, 0] = 1
        with self.assertRaises(ValueError):
            compare_training_contracts(a, b)

    def test_reject_partial_training_and_partial_evaluation(self):
        a, b = self.states()
        b["progress"]["pilot_complete"] = False
        with self.assertRaises(ValueError):
            compare_training_contracts(a, b)
        with self.assertRaises(ValueError):
            validate_summary({"format": "sam3-nakehand-semantic-evaluation-v1", "status": "completed",
                              "full_val_evaluated": False}, {})

    def test_zero_denominator_is_not_a_perfect_score(self):
        self.assertEqual(fraction(0, 0), "0/0（N/A）")
        self.assertIn("50.00%", fraction(1, 2))


if __name__ == "__main__":
    unittest.main()

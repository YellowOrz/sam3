"""CPU-only checks for strict forward-equivalence bookkeeping."""

import unittest

import torch
from torch import nn

from scripts.check_ve_prompt_equivalence import (
    OUTPUT_KEYS, compare_outputs, parameter_versions, verify_parameter_versions,
)


class VEPromptEquivalenceHelpersTest(unittest.TestCase):
    def outputs(self):
        return {name: torch.tensor([-.1, 0., .1]) for name in OUTPUT_KEYS}

    def test_equal_outputs_keep_mask_threshold_boundary(self):
        result = compare_outputs(self.outputs(), self.outputs())
        self.assertTrue(result["passed"])
        self.assertTrue(result["outputs"]["pred_masks"]["binary_masks_equal"])
        self.assertEqual(result["outputs"]["pred_logits"]["max_abs_difference"], 0.)

    def test_small_difference_fails_even_when_binary_mask_unchanged(self):
        expected, actual = self.outputs(), self.outputs()
        actual["pred_masks"][2] += 1e-6
        result = compare_outputs(expected, actual)
        self.assertFalse(result["passed"])
        self.assertTrue(result["outputs"]["pred_masks"]["binary_masks_equal"])
        self.assertGreater(result["outputs"]["pred_masks"]["max_abs_difference"], 0.)

    def test_shape_dtype_and_nonfinite_each_fail(self):
        for value in (torch.ones(4), torch.ones(3, dtype=torch.float64),
                      torch.tensor([-.1, float("nan"), .1])):
            actual = self.outputs()
            actual["pred_logits"] = value
            self.assertFalse(compare_outputs(self.outputs(), actual)["passed"])

    def test_parameter_version_detects_in_place_mutation_not_mode_change(self):
        model = nn.Linear(2, 1)
        original = parameter_versions(model)
        model.eval().requires_grad_(False)
        verify_parameter_versions(original)
        with torch.no_grad():
            model.weight.add_(.01)
        with self.assertRaisesRegex(RuntimeError, "modified"):
            verify_parameter_versions(original)


if __name__ == "__main__":
    unittest.main()

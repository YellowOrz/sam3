import unittest
import numpy as np
from scripts.diagnose_validation_gates import paired_metrics, validate_comparison_steps, boundary_iou


def row(image_id=1, score=.8, dice=.9, present=True):
    return dict(image_id=image_id, top_class_probability=1., presence_probability=score,
        top_confidence=score, detected=score >= .5, reference_present=present,
        candidate_dice=dice if present else None,
        candidate_boundary_iou_4px=dice / 2 if present else None,
        miss_zero_dice=(dice if score >= .5 else 0.) if present else None)


class ValidationGateTest(unittest.TestCase):
    def test_exact_gate_shape_decomposition_and_transitions(self):
        m = paired_metrics([(row(), row(score=.4, dice=.8)),
            (row(2, .4, .6), row(2, .8, .7)), (row(3, .7, .9), row(3, .8, .8)),
            (row(4, .1), row(4, .2)), (row(5, .8, present=False), row(5, .2, present=False))])
        self.assertEqual(m['positive_queries'], 4)
        self.assertEqual(m['transitions']['new_miss']['count'], 1)
        self.assertEqual(m['transitions']['recovered']['count'], 1)
        self.assertEqual(m['transitions']['new_miss']['candidate_dice_ge_0_7_after'], 1)
        self.assertAlmostEqual(m['dice_delta'], -.075)
        self.assertAlmostEqual(m['dice_delta'], m['gate_contribution_with_old_candidate']+m['candidate_contribution_under_new_gate'])
        self.assertEqual(m['false_positive_before'], 1)
        self.assertEqual(m['false_positive_after'], 0)
        self.assertAlmostEqual(m['boundary_delta'], m['dice_delta'] / 2)
        self.assertAlmostEqual(m['boundary_delta'], m['boundary_gate_contribution_with_old_candidate']
                               + m['boundary_candidate_contribution_under_new_gate'])
        self.assertEqual(m['negative_transitions']['removed_false_positive'], 1)
        self.assertEqual(m['false_negative_before'], 2)
        self.assertEqual(m['false_negative_after'], 2)

    def test_reject_changed_reference_and_bad_probability(self):
        with self.assertRaises(ValueError): paired_metrics([(row(1), row(2))])
        with self.assertRaises(ValueError): paired_metrics([(row(), row(present=False))])
        a = row();a['top_class_probability'] = .1
        with self.assertRaises(ValueError): paired_metrics([(a, row())])
        a = row();a['candidate_dice'] = float('nan')
        with self.assertRaises(ValueError): paired_metrics([(a, row())])
        a = row();a['candidate_boundary_iou_4px'] = float('nan')
        with self.assertRaises(ValueError): paired_metrics([(a, row())])

    def test_no_positive_denominator_is_undefined(self):
        m = paired_metrics([(row(present=False), row(present=False))])
        self.assertIsNone(m['dice_delta'])
        self.assertIsNone(m['gate_contribution_with_old_candidate'])
        self.assertIsNone(m['boundary_delta'])

    def test_ordered_trained_baseline_requires_opt_in(self):
        validate_comparison_steps(0, 55680)
        validate_comparison_steps(55680, 62640, allow_trained_baseline=True)
        with self.assertRaises(ValueError): validate_comparison_steps(55680, 62640)
        for before, after in [(2, 2), (3, 2), (-1, 2), (True, 2), (0, 0), (0, float('nan'))]:
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                validate_comparison_steps(before, after, allow_trained_baseline=True)

    def test_boundary_reference_and_empty_output(self):
        reference = np.zeros((20, 20), bool)
        reference[3:17, 3:17] = True
        self.assertEqual(boundary_iou(reference, reference), 1.)
        self.assertEqual(boundary_iou(np.zeros_like(reference), reference), 0.)
        self.assertIsNone(boundary_iou(reference, np.zeros_like(reference)))

    def test_negative_only_transitions_are_separate(self):
        m = paired_metrics([(row(1, .2, present=False), row(1, .8, present=False)),
                            (row(2, .9, present=False), row(2, .8, present=False))])
        self.assertEqual(m['negative_transitions']['new_false_positive'], 1)
        self.assertEqual(m['negative_transitions']['persistent_false_positive'], 1)
        self.assertEqual(m['false_positive_after'], 2)
        self.assertIsNone(m['boundary_delta'])


if __name__ == '__main__': unittest.main()

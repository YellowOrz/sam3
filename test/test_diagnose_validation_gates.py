import unittest
from scripts.diagnose_validation_gates import paired_metrics


def row(image_id=1, score=.8, dice=.9, present=True):
    return dict(image_id=image_id, top_class_probability=1., presence_probability=score,
        top_confidence=score, detected=score >= .5, reference_present=present,
        candidate_dice=dice if present else None,
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

    def test_reject_changed_reference_and_bad_probability(self):
        with self.assertRaises(ValueError): paired_metrics([(row(1), row(2))])
        with self.assertRaises(ValueError): paired_metrics([(row(), row(present=False))])
        a = row();a['top_class_probability'] = .1
        with self.assertRaises(ValueError): paired_metrics([(a, row())])
        a = row();a['candidate_dice'] = float('nan')
        with self.assertRaises(ValueError): paired_metrics([(a, row())])

    def test_no_positive_denominator_is_undefined(self):
        m = paired_metrics([(row(present=False), row(present=False))])
        self.assertIsNone(m['dice_delta'])
        self.assertIsNone(m['gate_contribution_with_old_candidate'])


if __name__ == '__main__': unittest.main()

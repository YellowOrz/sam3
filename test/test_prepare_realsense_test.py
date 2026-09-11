import unittest

from scripts.prepare_realsense_test import exclusions, select_indices, FRAME_COUNTS, PER_RECORDING


def review_evidence():
    plan = {"samples": [{"recording": "basket", "frame_index": 121},
                        {"recording": "left_hand", "frame_index": 580}]}
    issues = {"format": "realsense-review-issue-register-v1", "confirmed_current_annotation_issues": [
        {"id": "D1", "recording": "basket", "frame_index": 622},
        {"id": "D2", "recording": "cup", "frame_index": 319}],
        "uncertainties": [{"recording": "basket", "frame_index": 609}],
        "version_findings_not_current_error_counts": [{"recording": "cup",
            "historical_side_correspondence_interval_inclusive": [171, 186], "current_version_error_confirmed": False}]}
    return plan, issues


class RealSensePreparationTest(unittest.TestCase):
    def test_balanced_unique_fixed_selection(self):
        blocked = exclusions(*review_evidence())
        first = select_indices(FRAME_COUNTS, blocked)
        self.assertEqual(first, select_indices(dict(reversed(list(FRAME_COUNTS.items()))), blocked))
        self.assertEqual(sum(map(len, first.values())), 128)
        for name, indices in first.items():
            self.assertEqual(len(indices), PER_RECORDING)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertFalse(set(indices) & set(blocked[name]))
            self.assertTrue(all(0 <= frame < FRAME_COUNTS[name] for frame in indices))

    def test_known_disputes_not_relabelled(self):
        blocked = exclusions(*review_evidence())
        self.assertIn(609, blocked["basket"])
        self.assertIn(622, blocked["basket"])
        self.assertIn(319, blocked["cup"])
        self.assertIn(171, blocked["cup"])
        self.assertIn(186, blocked["cup"])
        self.assertNotIn(170, blocked["cup"])
        self.assertNotIn(187, blocked["cup"])
        self.assertNotIn("left_hand", blocked)

    def test_changed_issue_contract_rejected(self):
        plan, issues = review_evidence()
        issues["confirmed_current_annotation_issues"][0]["frame_index"] = 621
        with self.assertRaises(ValueError):
            exclusions(plan, issues)

    def test_invalid_counts(self):
        for counts, blocks in (({"s": 3}, {}), ({"s": -1}, {}), ({"s": 20}, {"s": [20]})):
            with self.assertRaises(ValueError):
                select_indices(counts, blocks)

    def test_changed_seed_changes_selection(self):
        self.assertNotEqual(select_indices(FRAME_COUNTS, {}, seed=1), select_indices(FRAME_COUNTS, {}, seed=2))


if __name__ == "__main__":
    unittest.main()

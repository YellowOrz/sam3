import copy
import unittest

from scripts.report_bilateral_evaluation import merge_summaries, render_report


def example(label="epoch1"):
    return {
        "data_root": "/dataset/val", "annotations_sha256": "abc",
        "base_checkpoint": "/models/base.pt", "evaluated_images": 2,
        "evaluated_dataset_indices": [1, 2], "detection_threshold": 0.5,
        "mask_threshold": 0.5, "confidence_definition": "class * presence",
        "models": {label: {"kind": "learned"}},
        "metrics": {label: {"correct_prompt_mean_top_dice": 0.8}},
    }


class BilateralReportTest(unittest.TestCase):
    def test_merges_comparable_runs_without_modifying_input(self):
        first, second = example(), example("epoch2")
        original = copy.deepcopy(first)
        second["evaluated_dataset_indices"] = [2, 1]
        merged = merge_summaries([first, second])
        self.assertEqual(set(merged["metrics"]), {"epoch1", "epoch2"})
        self.assertEqual(first, original)

    def test_rejects_different_data_and_thresholds(self):
        for key, value in (("annotations_sha256", "other"), ("mask_threshold", 0.4),
                           ("detection_threshold", 0.7), ("evaluated_dataset_indices", [1, 3])):
            changed = example("epoch2")
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                merge_summaries([example(), changed])

    def test_conflicting_duplicate_labels_are_not_silently_overwritten(self):
        changed = example()
        changed["metrics"]["epoch1"]["correct_prompt_mean_top_dice"] = 0.9
        with self.assertRaisesRegex(ValueError, "Conflicting model label"):
            merge_summaries([example(), changed])
        self.assertEqual(len(merge_summaries([example(), example()])["metrics"]), 1)

    def test_incomplete_evaluation_is_rejected_and_report_explains_metrics(self):
        changed = example()
        changed["models"] = {}
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            merge_summaries([changed])
        report = render_report(example(), [("/run/summary.json", "hash")])
        self.assertIn("0.8000", report)
        self.assertIn("不是 COCO mask AP", report)
        self.assertIn("不是“任一侧检出”的每图误检率", report)
        self.assertIn("路径相同不能独立证明", report)


if __name__ == "__main__":
    unittest.main()

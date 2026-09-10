import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from scripts.train_learnable_tokens import (
    build_epoch_order,
    build_training_config,
    build_training_order,
    inspect_training_annotations,
    validate_resume_training_config,
)


class TrainLearnableTokensTest(unittest.TestCase):
    def test_resume_training_config_rejects_changed_loss_weight(self):
        args = Namespace(
            tokens_per_class=4,
            batch_size=1,
            amp=False,
            learning_rate=0.01,
            epochs=2,
            seed=123,
            mask_weight=1.0,
            dice_weight=1.0,
            bbox_weight=1.0,
            giou_weight=1.0,
            classification_weight=1.0,
            presence_weight=1.0,
            data_root=Path("/tmp/data"),
            base_checkpoint=Path("/tmp/sam3.pt"),
        )
        saved = build_training_config(args)
        current = dict(saved, presence_weight=0.5)

        with self.assertRaisesRegex(ValueError, "presence_weight"):
            validate_resume_training_config(saved, current)

    def test_epoch_order_is_deterministic_and_visits_every_sample_once(self):
        first = build_epoch_order(dataset_size=20, seed=123)
        second = build_epoch_order(dataset_size=20, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(20)))

    def test_multi_epoch_order_visits_every_sample_once_per_epoch(self):
        order = build_training_order(dataset_size=20, seed=123, epochs=2)

        self.assertEqual(len(order), 40)
        self.assertEqual(sorted(order[:20]), list(range(20)))
        self.assertEqual(sorted(order[20:]), list(range(20)))
        self.assertNotEqual(order[:20], order[20:])

    def test_training_contract_requires_both_sides_and_allows_empty_images(self):
        data = {
            "categories": [
                {"id": 1, "name": "left_hand"},
                {"id": 2, "name": "right_hand"},
            ],
            "images": [{"id": 10}, {"id": 11}, {"id": 12}],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 1},
                {"id": 2, "image_id": 11, "category_id": 2},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(data))

            summary = inspect_training_annotations(path)

        self.assertEqual(summary["negative_images"], 1)
        self.assertEqual(
            summary["positive_images_by_class"],
            {"left_hand": 1, "right_hand": 1},
        )


if __name__ == "__main__":
    unittest.main()

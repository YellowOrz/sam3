import unittest
import hashlib
import json
import tempfile
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.train_learnable_tokens import (
    build_epoch_order,
    build_training_config,
    build_training_order,
    describe_training_progress,
    inspect_training_annotations,
    observed_identity_provenance,
    sha256_argument,
    validate_initial_token_hash,
    validate_training_batch_identity,
    validate_resume_training_config,
)


class TrainLearnableTokensTest(unittest.TestCase):
    def test_initial_token_hash_accepts_exact_float32_bytes_without_rng_change(self):
        tokens = torch.zeros(2, 4, 256, dtype=torch.float32)
        expected = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        rng = torch.get_rng_state().clone()
        self.assertEqual(validate_initial_token_hash(tokens, expected.upper()), expected)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(float(tokens.sum()), 0)

    def test_initial_token_hash_rejects_mismatch_or_invalid_tensor(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            validate_initial_token_hash(torch.zeros(2, 4, 256), "0" * 64)
        for tokens in (torch.zeros(2, 4, 256, dtype=torch.float64), torch.full((2, 4, 256), float("nan"))):
            with self.assertRaisesRegex(ValueError, "finite float32"):
                validate_initial_token_hash(tokens)

    def test_initial_token_expected_sha256_rejects_malformed_cli_values(self):
        for value in ("", "a" * 63, "g" * 64, "1" * 65):
            with self.subTest(value=value), self.assertRaises(ArgumentTypeError):
                sha256_argument(value)

    @staticmethod
    def identity_batch(images=(10, 11, 10, 11), categories=(1, 1, 2, 2),
                       local_indices=(0, 1, 0, 1), text_indices=(0, 0, 1, 1)):
        return SimpleNamespace(
            find_text_batch=["left_hand", "right_hand"],
            find_inputs=[SimpleNamespace(img_ids=torch.tensor(local_indices), text_ids=torch.tensor(text_indices))],
            find_metadatas=[SimpleNamespace(coco_image_id=torch.tensor(images),
                                           original_category_id=torch.tensor(categories))],
        )

    def test_training_identity_accepts_query_major_order_without_changing_rng(self):
        rng = torch.get_rng_state().clone()
        observed = validate_training_batch_identity(
            self.identity_batch(), [10, 11], {"left_hand": 1, "right_hand": 2}
        )
        self.assertEqual(observed, [10, 11])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_training_identity_rejects_substituted_or_swapped_images(self):
        for images in ((99, 11, 99, 11), (11, 10, 11, 10)):
            with self.subTest(images=images), self.assertRaisesRegex(RuntimeError, "substituted image"):
                validate_training_batch_identity(
                    self.identity_batch(images=images), [10, 11], {"left_hand": 1, "right_hand": 2}
                )

    def test_training_identity_rejects_category_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "Prompt/category mismatch"):
            validate_training_batch_identity(
                self.identity_batch(categories=(1, 2, 2, 2)), [10, 11], {"left_hand": 1, "right_hand": 2}
            )

    def test_training_identity_requires_both_queries_exactly_once(self):
        batch = self.identity_batch(text_indices=(0, 0, 0, 1), categories=(1, 1, 1, 2))
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            validate_training_batch_identity(batch, [10, 11], {"left_hand": 1, "right_hand": 2})
        batch = self.identity_batch(images=(10, 11), categories=(1, 1),
                                    local_indices=(0, 1), text_indices=(0, 0))
        with self.assertRaisesRegex(RuntimeError, "exactly two"):
            validate_training_batch_identity(batch, [10, 11], {"left_hand": 1, "right_hand": 2})
        batch = self.identity_batch(local_indices=(0, 1, -1, 1))
        with self.assertRaisesRegex(RuntimeError, "invalid image/text"):
            validate_training_batch_identity(batch, [10, 11], {"left_hand": 1, "right_hand": 2})

    def test_observed_identity_on_resume_does_not_claim_historical_samples(self):
        result = observed_identity_provenance(100, 102, [12, 15])
        self.assertEqual(result["start_step"], 100)
        self.assertEqual(result["end_step"], 102)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["queries"], 4)
        self.assertEqual(result["image_ids_sha256"], hashlib.sha256(b"12\n15\n").hexdigest())
        self.assertEqual(result["scope"], "current_process_completed_steps_only")

    def test_short_run_is_not_reported_as_two_completed_epochs(self):
        progress = describe_training_progress(1000, 23265, 2, 2)
        self.assertEqual(progress["samples_seen"], 2000)
        self.assertEqual(progress["epochs_completed"], 0)
        self.assertEqual(progress["steps_planned"], 23266)
        self.assertFalse(progress["full_training_complete"])

    def test_progress_accounts_for_odd_last_batch(self):
        self.assertEqual(describe_training_progress(11633, 23265, 2, 2)["samples_seen"], 23265)
        progress = describe_training_progress(23266, 23265, 2, 2)
        self.assertEqual(progress["samples_seen"], 46530)
        self.assertEqual(progress["epochs_completed"], 2)
        self.assertTrue(progress["full_training_complete"])
        self.assertEqual(describe_training_progress(0, 3, 2, 2)["samples_seen"], 0)

    def test_progress_rejects_invalid_or_excess_steps(self):
        for arguments in ((-1, 3, 2, 2), (5, 3, 2, 2), (1, 0, 2, 2), (1, 3, 0, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                describe_training_progress(*arguments)

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

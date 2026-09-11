from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts.evaluate_residual_test import add_boundary, summarize, load_residual
from scripts import evaluate_nakehand_tokens as bilateral
from scripts import residual_ddp_checkpoint as checkpoint
from scripts import train_ve_initialized_tokens as initializer
from scripts.cached_ve_text_features import CachedVETextEncoder
from scripts.prepare_nakehand_test import side_annotation
from scripts.residual_ddp_runtime import capture_rng_state


class ResidualTestMetricsTest(unittest.TestCase):
    def test_missed_perfect_mask_is_zero(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        row = bilateral.measure_query(mask, mask, np.zeros_like(mask), .49)
        add_boundary(row, mask, mask)
        self.assertEqual(row["top_dice"], 1.)
        self.assertEqual(row["miss_zero_dice"], 0.)
        self.assertEqual(row["candidate_boundary_iou_4px"], 1.)
        self.assertEqual(row["miss_zero_boundary_iou_4px"], 0.)

    def test_threshold_equal_counts_detected(self):
        mask = np.ones((8, 8), dtype=bool)
        row = bilateral.measure_query(mask, mask, np.zeros_like(mask), .5)
        add_boundary(row, mask, mask)
        self.assertTrue(row["detected"])
        self.assertEqual(row["miss_zero_boundary_iou_4px"], 1.)

    def test_empty_reference_not_in_dice_mean(self):
        empty = np.zeros((12, 12), dtype=bool)
        row = bilateral.measure_query(empty, empty, empty, .9)
        add_boundary(row, empty, empty)
        self.assertIsNone(row["miss_zero_dice"])
        self.assertIsNone(row["candidate_boundary_iou_4px"])
        row.update(image_id=1, prompt_key="left_hand", recording_id="s")
        result = summarize([row])
        self.assertEqual(result["overall"]["false_positive_queries"], 1)
        self.assertIsNone(result["overall"]["present_mean_miss_zero_dice"])

    def test_raw_instance_two_preserved_in_semantic_union(self):
        raw = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        annotation = side_annotation(raw, 1, 1, 1)
        self.assertEqual(annotation["area"], 2)
        self.assertEqual(annotation["source_instance_values"], [1, 2])

    def test_original_and_residual_metrics_same_for_identical_masks(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2:6, 3:8] = True
        rows = []
        for side in ("left_hand", "right_hand"):
            row = bilateral.measure_query(mask, mask, np.zeros_like(mask), .7)
            add_boundary(row, mask, mask)
            row.update(image_id=1, prompt_key=side, recording_id="s")
            rows.append(row)
        result = summarize(rows)
        self.assertEqual(result["overall"]["present_mean_miss_zero_dice"], 1.)
        self.assertEqual(result["recording_macro_miss_zero_dice"], 1.)

    def test_wrong_checkpoint_format_rejected_before_loading_initializer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save({"format": "legacy_random_class_tokens"}, path)
            with self.assertRaises(ValueError):
                load_residual(path, Path(directory) / "unused", directory,
                              base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_actual_nonzero_delta_is_loaded_and_frozen_cache_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            padding = torch.ones(2, 32, dtype=torch.bool)
            padding[:, :4] = False
            encoder = CachedVETextEncoder(padding, torch.ones(32, 2, 256, dtype=torch.bfloat16),
                torch.ones(32, 2, 1024), mode="zero_delta",
                metadata={"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64})
            initial = deepcopy(encoder.state_dict())
            initial_path = root / "initial.pt"
            torch.save(initial, initial_path)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps({"images": [{"id": 1}, {"id": 2}]}))
            config = {"world_size": 1, "batch_size_per_rank": 1, "global_batch_size": 1,
                "dataset_size": 2, "steps_per_epoch": 2, "epochs": 2, "seed": 123,
                "learning_rate": .001, "base_sha256": "a"*64, "tokenizer_sha256": "b"*64,
                "initial_cache_file_sha256": initializer.evaluation.sha256(initial_path),
                "annotations_sha256": initializer.evaluation.sha256(annotation_path),
                "image_order_sha256": checkpoint.canonical_hash([1, 2])}
            optimizer = torch.optim.AdamW([encoder.delta], lr=.001, weight_decay=0.)
            encoder.delta.square().sum().add(encoder.delta.sum()).backward()
            optimizer.step()
            ranks = [{"image_ids": checkpoint.expected_rank_ids([1, 2], config, 1, 0),
                      "rng": capture_rng_state("cpu")}]
            state = checkpoint.make_checkpoint(encoder, optimizer, config, 1, initial, ranks)
            target = root / "delta.pt"
            torch.save(state, target)
            loaded, metadata = load_residual(target, initial_path, root, base_hash="a"*64, tokenizer_hash="b"*64)
            self.assertTrue(torch.equal(loaded.delta, encoder.delta))
            self.assertTrue(bool(loaded.delta.abs().sum() > 0))
            self.assertEqual(metadata["progress"]["global_step"], 1)
            state["cache_state_dict"]["resized_cache"][0, 0, 0] += 1
            torch.save(state, target)
            with self.assertRaises(ValueError):
                load_residual(target, initial_path, root, base_hash="a"*64, tokenizer_hash="b"*64)


if __name__ == "__main__":
    unittest.main()

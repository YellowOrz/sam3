"""CPU contracts for fixed-split, bilateral semantic-delta evaluation."""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import evaluate_nakehand_semantic_tokens as evaluate
from test_evaluate_nakehand_tokens import write_fixture, four_image_records
from test_evaluate_ve_initialized_tokens import state as old_state


def state(anchor=0.):
    value = old_state()
    value["format"] = evaluate.FORMAT
    for name in ("cache_state_dict", "initial_cache_state_dict"):
        value[name]["resized_cache"] = value[name]["resized_cache"].to(torch.bfloat16)
        value[name]["_extra_state"]["metadata"]["resized_dtype"] = "torch.bfloat16"
    value["training_config"].update(anchor_weight=anchor, amp_dtype="bfloat16", seed=123,
                                     learning_rate=.001, planned_samples=2000, float32_matmul_precision="high")
    value["progress"]["full_epoch_completed"] = False
    return value


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return evaluate.shared.sha256(path)


def published_val(root):
    split = root / "val"
    split.mkdir()
    plan = {"splits": {"val": {"coco_split": "val", "images": 3449,
                                 "recordings": list(evaluate.VAL_RECORDINGS)}},
            "recordings": {evaluate.VAL_RECORDINGS[0]: {"frame_count": 3449, "global_image_id_offset": 9000}}}
    plan_sha = write_json(root / "frozen-plan.json", plan)
    images = [{"id": index + 9000, "recording_id": evaluate.VAL_RECORDINGS[0],
               "file_name": f"images/{index}.png", "source_frame_index": index,
               "frame_index": index} for index in range(3449)]
    data = {"images": images, "info": {"split": "val", "dataset_role": "validation",
                                         "frozen_plan_sha256": plan_sha}}
    annotation_sha = write_json(split / "annotations.json", data)
    manifest = {"status": "complete", "sources_unchanged": True,
                "image_outputs": [{"image_id": row["id"], "files": {
                    "rgb": {"path": row["file_name"], "sha256": "a" * 64}}} for row in images]}
    manifest_sha = write_json(split / "manifest.json", manifest)
    ready = {"status": "complete", "annotations_sha256": annotation_sha,
             "manifest_sha256": manifest_sha, "frozen_plan_sha256": plan_sha}
    ready_sha = write_json(split / "READY.json", ready)
    root_manifest_sha = write_json(root / "manifest.json", {"sources_unchanged": True})
    write_json(root / "READY.json", {"status": "complete", "manifest_sha256": root_manifest_sha,
                                    "frozen_plan_sha256": plan_sha,
                                    "splits": {"val": {**ready, "ready_sha256": ready_sha}}})
    return split


class SemanticNakehandEvaluationTest(unittest.TestCase):
    def validate(self, value, *, minimum=20, anchor=0.):
        return evaluate.validate_checkpoint(value, minimum_samples=minimum, expected_anchor=anchor,
                                             base_hash="a" * 64, tokenizer_hash="b" * 64)

    def test_partial_checkpoint_never_passes_formal_and_labels_bind_anchor(self):
        trained, initial = self.validate(state())
        self.assertGreater(float(trained.delta.detach().abs().sum()), 0)
        self.assertEqual(float(initial.delta.detach().abs().sum()), 0)
        with self.assertRaises(ValueError):
            self.validate(state(), minimum=2000)
        with self.assertRaisesRegex(ValueError, "Anchor"):
            self.validate(state(1.))
        self.validate(state(1.), anchor=1.)

    def test_new_trainer_checkpoint_constructor_matches_evaluator_contract(self):
        from scripts import train_nakehand_semantic_tokens as trainer

        value = state()
        encoder = evaluate.semantic.cache_from_state(value["cache_state_dict"])
        config = dict(value["training_config"], initial_cache_sha256="c" * 64, data_provenance={})
        optimizer = torch.optim.AdamW([encoder.delta], lr=.001, weight_decay=0.)
        checkpoint = trainer.make_checkpoint(
            encoder=encoder, optimizer=optimizer, config=config,
            initial_state=value["initial_cache_state_dict"], annotation_summary={"images": 9092},
            order=list(range(2000)), observed_ids=list(range(20)), loss_history=[1.] * 20,
            gradient_counts=[20, 20], core_hashes={}, task_history=[1.] * 20,
            anchor_history=[0.] * 20, drift_history=[[0., 0.]] * 20,
            task_grad_history=[[1., 1.]] * 20)
        self.validate(checkpoint)
        self.assertEqual(checkpoint["format"], evaluate.FORMAT)
        self.assertFalse(checkpoint["progress"]["pilot_complete"])

    def test_wrong_dataset_lr_amp_epoch_claim_and_initial_cache_are_rejected(self):
        for field in ("format", "lr", "amp", "epoch", "initial"):
            value = state()
            if field == "format":
                value["format"] = evaluate.semantic.FORMAT
            elif field == "lr":
                value["training_config"]["learning_rate"] = .01
            elif field == "amp":
                value["training_config"]["amp_dtype"] = "float16"
            elif field == "epoch":
                value["progress"]["full_epoch_completed"] = True
            else:
                value["initial_cache_state_dict"]["delta"].add_(1.)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(value)

    def test_comparison_requires_same_initial_features_numerics_and_actual_prefix(self):
        original, constrained = state(), state(1.)
        evaluate.verify_comparable([original, constrained])
        for field in ("prefix", "base", "cache"):
            changed = copy.deepcopy(constrained)
            if field == "prefix":
                changed["observed_image_ids"][0] += 1
            elif field == "base":
                changed["training_config"]["base_checkpoint_sha256"] = "c" * 64
            else:
                changed["initial_cache_state_dict"]["raw_cache"].add_(1)
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate.verify_comparable([original, changed])

    def test_lazy_reference_union_and_memory_bounded_with_zero_one_two_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            data = write_fixture(Path(directory))
            references = evaluate.LazyReferences(data, cache_size=1)
            expected_presence = ((True, True), (True, False), (False, True), (False, False))
            for index, expected in enumerate(expected_presence):
                masks = references[index]
                self.assertEqual(tuple(bool(masks[side].any()) for side in evaluate.shared.CLASS_NAMES), expected)
                self.assertEqual(len(references.cache), 1)
            first = references[0]["right_hand"].copy()
            data["annotations"].append(dict(data["annotations"][1], id=100))
            duplicate = evaluate.LazyReferences(data)
            np.testing.assert_array_equal(duplicate[0]["right_hand"], first)

    def test_real_cpu_loader_preserves_global_ids_bilateral_presence_and_no_prompts(self):
        from sam3.train.data.collator import collate_fn_api

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = write_fixture(root)
            for image in data["images"]:
                image["id"] += 9092
            for annotation in data["annotations"]:
                annotation["image_id"] += 9092
            write_json(root / "annotations.json", data)
            raw = evaluate.shared.make_dataset(root)
            dataset = evaluate.semantic.IdentityCheckedDataset(raw, data["images"])
            samples = [dataset[index] for index in range(4)]
            self.assertEqual([[len(query.object_ids_output) for query in sample.find_queries] for sample in samples],
                             [[1, 1], [1, 0], [0, 1], [0, 0]])
            batch = collate_fn_api(samples, dict_key="eval", with_seg_masks=True)["eval"]
            evaluate.bilateral.validate_batch_identity(batch, [0, 1, 2, 3], data["images"])
            self.assertEqual(dataset.observed_indices, [0, 1, 2, 3])

    def test_publication_gate_binds_root_split_plan_manifest_and_rejects_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = published_val(root)
            data, provenance = evaluate.verify_split(split, "val")
            self.assertEqual(len(data["images"]), 3449)
            self.assertEqual(provenance["dataset_role"], "validation")
            self.assertEqual(len(provenance["rgb_files"]), 3449)
            with self.assertRaises(ValueError):
                evaluate.verify_split(split, "test")
            (split / "annotations.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hashes"):
                evaluate.verify_split(split, "val")

    def test_real_planned_prefix_uses_global_image_ids_not_indices(self):
        value = state()
        data = {"images": [{"id": index + 10000} for index in range(9092)]}
        provenance = {"root_ready_sha256": "r", "split_ready_sha256": "s", "split_manifest_sha256": "m",
                      "frozen_plan_sha256": "p", "recordings": list(evaluate.TRAIN_RECORDINGS),
                      "dataset_role": "train", "annotations_sha256": "d"}
        recorded = {key: item for key, item in provenance.items() if key != "annotations_sha256"}
        value["training_config"].update(data_root="/unused/train", annotations_sha256="d", data_provenance=recorded)
        value["data_provenance"] = recorded
        value["annotation_summary"] = {"sha256": "d", "images": 9092}
        order = list(range(9092))
        random.Random(123).shuffle(order)
        value["planned_dataset_indices"] = order[:2000]
        value["planned_dataset_indices_sha256"] = evaluate.object_hash(order[:2000])
        value["observed_image_ids"] = [index + 10000 for index in order[:20]]
        with patch.object(evaluate, "verify_split", return_value=(data, provenance)):
            self.assertTrue(evaluate.verify_training_identity(value)["actual_prefix_verified"])
            value["observed_image_ids"][0] -= 10000
            with self.assertRaisesRegex(ValueError, "prefix"):
                evaluate.verify_training_identity(value)

    def test_validation_alias_does_not_turn_confidence_detection_into_correct_mask(self):
        metrics = evaluate.summarize_validation(four_image_records(swapped=True))["epoch2"]
        self.assertIs(metrics["validation"], metrics["primary_test"])
        overall = metrics["validation"]["overall"]
        self.assertEqual(overall["correct_side_detection_rate"], 1)
        self.assertEqual(overall["present_mean_candidate_dice"], .5)
        self.assertEqual(overall["simultaneous_two_query_swap_proxy_images"], 1)

    def test_cli_diagnostic_requires_indices_memory_cap_and_optional_aware_deadline(self):
        base = ["--data-root", "/unused/val", "--base-checkpoint", "/unused/base.pt",
                "--output-dir", "/tmp/semantic-eval-unit-never-created", "--variant", evaluate.LABELS[0],
                "--baseline-checkpoint", "/unused/checkpoint.pt"]
        self.assertEqual(evaluate.parse_args(base).minimum_samples_seen, 2000)
        self.assertIsNone(evaluate.parse_args(base).deadline)
        self.assertIsNotNone(evaluate.parse_args(
            base + ["--deadline", "2026-09-11T21:41:00+08:00"]).deadline)
        for extra in (["--minimum-samples-seen", "20"], ["--gpu-memory-fraction", ".5"],
                      ["--deadline", "2026-09-11T21:41:00"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                evaluate.parse_args(base + extra)
        args = evaluate.parse_args(base + ["--minimum-samples-seen", "20", "--indices", "0,1"])
        self.assertEqual(args.labels, [evaluate.LABELS[0]])


if __name__ == "__main__":
    unittest.main()

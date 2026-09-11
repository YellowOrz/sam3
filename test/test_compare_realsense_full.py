"""CPU contracts for complete RealSense reference/shard comparisons."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import compare_realsense_full as compare
from scripts import evaluate_realsense_full as evaluator


SIDES = ("left_hand", "right_hand")
SHAPE = (480, 640)


def rle(mask):
    result = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    result["counts"] = result["counts"].decode("ascii")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def freeze_publication(root, data, manifest, plan):
    write_json(root / "frozen-plan.json", plan)
    plan_hash = compare.sha256(root / "frozen-plan.json")
    data["info"]["frozen_plan_sha256"] = plan_hash
    manifest["frozen_plan_sha256"] = plan_hash
    write_json(root / "annotations.json", data)
    write_json(root / "manifest.json", manifest)
    write_json(root / "READY.json", {
        "format": compare.DATA_FORMAT, "status": "complete", "dataset_role": "external_test_only",
        "evaluation_scope": "fixed_development_benchmark",
        "annotations_sha256": compare.sha256(root / "annotations.json"),
        "manifest_sha256": compare.sha256(root / "manifest.json"), "frozen_plan_sha256": plan_hash,
    })


def make_publication(root):
    """Six real RGB/side PNGs cover known-empty, unknown and flagged references."""
    empty = np.zeros(SHAPE, dtype=bool)
    left, right = empty.copy(), empty.copy()
    left[100:132, 100:132] = True
    right[200:224, 200:224] = True
    pairs = [(left, right), (left, empty), (left, None), (None, right),
             (empty, empty), (left, left)]
    data = {"info": {"format": compare.DATA_FORMAT, "dataset_role": "external_test_only",
                     "evaluation_scope": "fixed_development_benchmark"},
            "categories": [{"id": index, "name": side} for index, side in enumerate(SIDES, 1)],
            "images": [], "annotations": []}
    manifest = {"format": compare.DATA_FORMAT, "status": "complete", "sources_unchanged": True,
                "image_outputs": []}
    plan = {"format": compare.DATA_FORMAT, "dataset_role": "external_test_only",
            "evaluation_scope": "fixed_development_benchmark", "frame_counts": {"recording_a": 3, "recording_b": 3},
            "reference_description": compare.REFERENCE_DESCRIPTION,
            "selection_uses_predictions_or_mask_pixels": False}
    for image_id, pair in enumerate(pairs, 1):
        recording, frame = ("recording_a" if image_id <= 3 else "recording_b"), (image_id - 1) % 3
        provided = {side: mask is not None for side, mask in zip(SIDES, pair)}
        side_flags = {"left_hand": ["D1_confirmed_wrong_side_reference"] if image_id == 6 else [],
                      "right_hand": []}
        flags = list(side_flags["left_hand"])
        group = "both" if all(provided.values()) else "left_only" if provided[SIDES[0]] else "right_only"
        mapping = {"source_frame_index": frame, "frame_numbering": "zero_based_exported_video_not_bag_message",
                   "rgb_video": f"/original/{recording}/color.mp4",
                   "mask_videos": {side: f"/original/{recording}/{side}.mkv" for side in SIDES if provided[side]}}
        directory = root / "images" / str(image_id)
        directory.mkdir(parents=True)
        Image.new("RGB", (640, 480), (image_id, 20, 30)).save(directory / "rgb.png")
        files = {"rgb": {"path": f"images/{image_id}/rgb.png", "sha256": compare.sha256(directory / "rgb.png")}}
        image = {"id": image_id, "file_name": files["rgb"]["path"], "width": 640, "height": 480,
                 "recording_id": recording, "source_frame_index": frame, "source_mapping": mapping,
                 "source_dataset": "realsense", "reference_provided": provided, "quality_flags": flags,
                 "previously_displayed_random_review": image_id == 2,
                 "reference_quality_flags": side_flags, "raw_provided_group": group,
                 "has_both_reference": all(provided.values()), "legacy_fixed128": image_id <= 2,
                 "legacy_fixed128_image_id": image_id if image_id <= 2 else None,
                 "render_preselected": image_id <= 2}
        data["images"].append(image)
        for category, (side, mask) in enumerate(zip(SIDES, pair), 1):
            if mask is None:
                continue
            path = directory / f"{side}.png"
            Image.fromarray(mask.astype(np.uint8)).save(path)
            files[side] = {"path": str(path.relative_to(root)), "sha256": compare.sha256(path)}
            if mask.any():
                data["annotations"].append({"id": len(data["annotations"]) + 1, "image_id": image_id,
                    "category_id": category, "segmentation": rle(mask), "area": int(mask.sum()), "iscrowd": 0})
        manifest["image_outputs"].append({"image_id": image_id, "files": files,
            "left_right_overlap_pixels": int((pair[0] & pair[1]).sum()) if all(provided.values()) else None,
            **{key: deepcopy(image[key]) for key in ("recording_id", "source_frame_index", "source_mapping",
                "reference_provided", "quality_flags", "reference_quality_flags", "raw_provided_group",
                "legacy_fixed128", "legacy_fixed128_image_id")}})
    freeze_publication(root, data, manifest, plan)
    return data, manifest, plan


def make_evaluations(root, contract, mode, group, count=4):
    """Actual evaluator output schema, without a model, network or CUDA operation."""
    base_hash, tokenizer_hash = "1" * 64, "2" * 64
    model = {"base_checkpoint_sha256": base_hash, "tokenizer_sha256": tokenizer_hash}
    inputs = {"/run-inputs/base.pt": base_hash, "/run-inputs/tokenizer.gz": tokenizer_hash}
    if mode == "residual":
        step, checkpoint_hash = (5021, "a" * 64) if group == "mixed" else (3082, "b" * 64)
        cache_hash, annotations_hash = "3" * 64, ("4" if group == "mixed" else "5") * 64
        model.update(checkpoint_sha256=checkpoint_hash, checkpoint_path=f"/run-inputs/{group}.pt",
            progress={"global_step": step, "completed_epochs": 1, "next_epoch": 1,
                      "next_step_in_epoch": 0, "samples_seen": step * 6},
            training_config={"steps_per_epoch": step, "global_batch_size": 6, "dataset_size": step * 6,
                "epochs": 5, "base_sha256": base_hash, "tokenizer_sha256": tokenizer_hash,
                "initial_cache_file_sha256": cache_hash, "annotations_sha256": annotations_hash},
            initial_cache_verified=True, actual_training_identities_verified=True,
            rank_cache_consistency_audit_present_and_verified=True, delta_l2_per_side=[.1, .2])
        inputs.update({f"/run-inputs/{group}.pt": checkpoint_hash,
                       "/run-inputs/cache.pt": cache_hash,
                       f"/run-inputs/{group}/annotations.json": annotations_hash})
    else:
        model["source"] = "actual original VE natural prompts; no cached replacement"
    source_root = Path(__file__).resolve().parents[1]
    logical_sources = ("scripts/evaluate_realsense_full.py", "scripts/evaluate_residual_test.py",
        "scripts/evaluate_nakehand_tokens.py", "scripts/evaluate_bilateral_tokens.py",
        "scripts/cached_ve_text_features.py", "scripts/residual_ddp_validation.py", "sam3/__init__.py")
    inference = {name: compare.sha256(source_root / name) for name in logical_sources}
    implementation = {str(source_root / name): value for name, value in inference.items()}
    outputs = {row["image_id"]: row for row in contract["manifest"]["image_outputs"]}
    paths = []
    for index in range(count):
        positions = evaluator.shard_indices(contract["images"], index, count)
        sides = evaluator.selected_sides(mode)
        sources = {str(contract["root"] / name): value for name, value in contract["fingerprints"].items()}
        sources.update(inputs)
        records = []
        for position in positions:
            image = contract["images"][position]
            for asset in outputs[image["id"]]["files"].values():
                sources[str(contract["root"] / asset["path"])] = asset["sha256"]
            for side in sides:
                other_side = SIDES[1 - SIDES.index(side)]
                reference = compare.reference_mask(contract, image["id"], side)
                other = compare.reference_mask(contract, image["id"], other_side)
                candidate = np.zeros(SHAPE, dtype=bool) if reference is None else reference.copy()
                score = float(np.float32(.8))
                own_flags = image["reference_quality_flags"][side]
                records.append({"model": mode, "mode": mode, "dataset_role": "external_test_only",
                    "dataset_index": position, "image_id": image["id"], "identity_verified": True,
                    "file_name": image["file_name"], "recording_id": image["recording_id"],
                    "source_frame_index": image["source_frame_index"], "source_mapping": image["source_mapping"],
                    "prompt_key": side, "prompt_text": ("left hand", "right hand")[SIDES.index(side)],
                    "reference_provided": reference is not None, "other_reference_provided": other is not None,
                    "has_both_reference": image["has_both_reference"], "raw_provided_group": image["raw_provided_group"],
                    "quality_flags": image["quality_flags"], "reference_quality_flags": own_flags,
                    "pair_quality_flags": sorted(set(own_flags + image["reference_quality_flags"][other_side])),
                    "quality_flag_scope": "side", "primary_test": reference is not None and not own_flags,
                    "legacy_fixed128": image["legacy_fixed128"], "legacy_fixed128_image_id": image["legacy_fixed128_image_id"],
                    "top_class_probability": score, "presence_probability": 1., "selected_decoder_query": 0,
                    "prediction_rle": rle(candidate), "detections_above_threshold": 1,
                    **evaluator.measure_partial(candidate, reference, other, score)})
        directory = root / group / str(index)
        directory.mkdir(parents=True)
        record_path = directory / "records.jsonl"
        record_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8")
        summary = {"format": evaluator.FORMAT, "status": "complete", "dataset_role": "external_test_only",
            "evaluation_scope": "fixed_development_benchmark", "mode": mode,
            "created_at_utc": "2026-09-11T00:00:00+00:00", "completed_at_utc": "2026-09-11T00:00:01+00:00",
            "whole_dataset_images": len(contract["images"]), "shard_images": len(positions),
            "shard_index": index, "shard_count": count,
            "shard_rule": "sorted whole-image ID position modulo shard_count",
            "image_ids": [contract["images"][i]["id"] for i in positions], "emitted_sides": list(sides),
            "expected_emitted_queries": len(positions) * len(sides), "actual_inference_queries_per_image": 2,
            "expected_actual_inference_queries": len(positions) * 2, "batch_size": 1,
            "detection_threshold": .5, "mask_threshold": .5, "boundary_width_original_pixels": 4,
            "precision": "BF16 autocast; FP32 logits for sigmoid/interpolation; FP32 delta",
            "prediction_selection": "argmax(sigmoid(class)*sigmoid(presence)); no reference-dependent selection",
            "protocol": contract["plan"], "source_fingerprints": sources, "implementation_sha256": implementation,
            "dataset_fingerprints": contract["fingerprints"], "source_inference_sha256": inference,
            "checkpoint_selection_note": "Predeclared completed epoch 1, not best." if mode == "residual" else None,
            "model_metadata": model, "reference_description": contract["plan"]["reference_description"],
            "limitations": [], "elapsed_seconds": 1., "actual_complete_query_coverage_verified": True,
            "records_file": "records.jsonl", "records_sha256": compare.sha256(record_path),
            "metrics": evaluator.summarize(records), "visualizations": []}
        path = directory / "summary.json"
        write_json(path, summary)
        paths.append(path)
    return paths


class MaskAndMetricTests(unittest.TestCase):
    def test_compressed_rle_roundtrips_signed_deltas_and_empty_full_masks(self):
        checker = np.indices((31, 43)).sum(axis=0) % 2 == 0
        for mask in (checker, np.zeros_like(checker), np.ones_like(checker)):
            with self.subTest(pixels=int(mask.sum())):
                np.testing.assert_array_equal(compare.decode_rle(rle(mask), mask.shape), mask)

    def test_malformed_rle_rejected_before_decoder(self):
        valid = rle(np.zeros((8, 10), dtype=bool))
        invalid = [None, {**valid, "size": [True, 10]}, {**valid, "size": [8.0, 10]},
                   {**valid, "size": [10, 8]}, {**valid, "counts": ""}, {**valid, "counts": "P"},
                   {**valid, "counts": "0"}, {**valid, "counts": "\x00"},
                   {**valid, "counts": [80]}, {**valid, "extra": 1}]
        for candidate in invalid:
            with self.subTest(rle=candidate), self.assertRaises(ValueError):
                compare.decode_rle(candidate, (8, 10))

    def test_missing_reference_is_unknown_even_for_high_confidence(self):
        empty = np.zeros((32, 32), dtype=bool)
        unknown = compare.measure(empty, None, empty, .9)
        absent = compare.measure(empty, empty, empty, .9)
        self.assertTrue(unknown["detected"])
        self.assertFalse(unknown["reference_provided"])
        for key in (*compare.METRIC_FIELDS, "target_present", "reference_pixels", "false_negative",
                    "absent_false_positive", "opposite_overlap_dominant_proxy"):
            self.assertIsNone(unknown[key], key)
        self.assertTrue(absent["absent_false_positive"])
        result = compare.aggregate([{**unknown, "image_id": 1}, {**absent, "image_id": 2}])
        self.assertEqual(result["missing_reference_queries"], 1)
        self.assertEqual(result["absent_queries"], 1)
        self.assertEqual(result["absent_false_positive_queries"], 1)

    def test_missing_other_reference_does_not_become_a_false_swap_proxy(self):
        mask = np.ones((32, 32), dtype=bool)
        row = compare.measure(mask, mask, None, .8)
        self.assertEqual(row["candidate_dice"], 1.)
        self.assertIsNone(row["opposite_overlap_dominant_proxy"])
        self.assertIsNone(row["detected_opposite_overlap_dominant_proxy"])

    def test_threshold_equal_detects_and_low_confidence_perfect_candidate_is_zero(self):
        mask = np.ones((32, 32), dtype=bool)
        for score, detected in ((.49, False), (.5, True)):
            row = compare.measure(mask, mask, np.zeros_like(mask), score)
            self.assertIs(row["detected"], detected)
            self.assertEqual(row["candidate_dice"], 1.)
            self.assertEqual(row["candidate_boundary_iou_4px"], 1.)
            self.assertEqual(row["miss_zero_dice"], float(detected))
            self.assertEqual(row["miss_zero_boundary_iou_4px"], float(detected))

    def test_boundary_width_is_four_original_pixels(self):
        reference = np.ones((32, 32), dtype=bool)
        candidate = np.zeros_like(reference)
        candidate[1:31, 1:31] = True
        # Four-pixel interior rings: areas 448 and 416, intersection 324, union 540.
        row = compare.measure(candidate, reference, None, .8)
        self.assertAlmostEqual(row["candidate_boundary_iou_4px"], 324 / 540)

    def test_swap_ties_and_suppressed_detection(self):
        own, other = np.zeros((32, 32), dtype=bool), np.zeros((32, 32), dtype=bool)
        own[2:10, 2:10], other[20:28, 20:28] = True, True
        swap = compare.measure(other, own, other, .2)
        self.assertTrue(swap["opposite_overlap_dominant_proxy"])
        self.assertFalse(swap["detected_opposite_overlap_dominant_proxy"])
        self.assertFalse(compare.measure(own | other, own, other, .8)["opposite_overlap_dominant_proxy"])

    def test_invalid_confidence_rejected(self):
        mask = np.ones((8, 10), dtype=bool)
        for score in (True, "0.5", -.1, 1.1, float("nan"), float("inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                compare.measure(mask, mask, None, score)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.manifest, self.plan = make_publication(self.root)

    def load(self):
        return compare.load_contract(self.root, expected_images=6, expected_legacy_images=2)

    def republish(self):
        freeze_publication(self.root, self.data, self.manifest, self.plan)

    def test_known_empty_and_missing_are_distinct_and_legacy_is_preserved(self):
        contract = self.load()
        self.assertEqual(contract["legacy"], {1: 1, 2: 2})
        self.assertFalse(compare.reference_mask(contract, 2, "right_hand").any())
        self.assertIsNone(compare.reference_mask(contract, 3, "right_hand"))
        self.assertIsNone(compare.reference_mask(contract, 4, "left_hand"))
        self.assertTrue(compare.reference_mask(contract, 3, "left_hand").any())

    def test_default_full_and_fixed128_sizes_cannot_accept_small_fixture(self):
        with self.assertRaises(ValueError):
            compare.load_contract(self.root)
        with self.assertRaises(ValueError):
            compare.load_contract(self.root, expected_images=6)

    def test_changed_publication_hash_rejected(self):
        self.data["images"][0]["source_frame_index"] += 1
        write_json(self.root / "annotations.json", self.data)
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_reference_cannot_acquire_annotation(self):
        annotation = deepcopy(self.data["annotations"][0])
        annotation.update(id=100, image_id=3, category_id=2)
        self.data["annotations"].append(annotation)
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_semantic_side_annotation_rejected(self):
        annotation = deepcopy(self.data["annotations"][0])
        annotation["id"] = 100
        self.data["annotations"].append(annotation)
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_empty_annotation_is_not_a_positive_reference(self):
        self.data["annotations"][0]["segmentation"] = rle(np.zeros(SHAPE, dtype=bool))
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_manifest_mapping_and_stream_presence_must_agree(self):
        for mutation in ("mapping", "missing_stream"):
            with self.subTest(mutation=mutation):
                original = deepcopy(self.manifest)
                if mutation == "mapping":
                    self.manifest["image_outputs"][0]["source_frame_index"] += 1
                else:
                    self.manifest["image_outputs"][2]["files"]["right_hand"] = deepcopy(
                        self.manifest["image_outputs"][0]["files"]["right_hand"])
                self.republish()
                with self.assertRaises(ValueError):
                    self.load()
                self.manifest = original

    def test_legacy_duplicate_or_missing_id_rejected(self):
        self.data["images"][1]["legacy_fixed128_image_id"] = 1
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_reference_provided_requires_boolean_values(self):
        self.data["images"][0]["reference_provided"]["left_hand"] = 1
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_side_quality_flags_required_without_image_level_fallback(self):
        del self.data["images"][0]["reference_quality_flags"]
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_review_exposure_is_not_a_quality_flag(self):
        self.data["images"][1]["quality_flags"] = ["previously_displayed_random_review"]
        self.data["images"][1]["reference_quality_flags"]["left_hand"] = ["previously_displayed_random_review"]
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_reference_png_hash_and_semantic_union_both_checked(self):
        entry = self.manifest["image_outputs"][0]["files"]["left_hand"]
        path = self.root / entry["path"]
        Image.fromarray(np.zeros(SHAPE, dtype=np.uint8)).save(path)
        with self.assertRaises(ValueError):
            self.load()
        entry["sha256"] = compare.sha256(path)
        self.republish()
        with self.assertRaises(ValueError):
            self.load()

    def test_scopes_exclude_only_flagged_side_and_keep_review_exposure(self):
        contract = self.load()
        rows = []
        for image in contract["images"]:
            for side in SIDES:
                reference = compare.reference_mask(contract, image["id"], side)
                other = compare.reference_mask(contract, image["id"], SIDES[1 - SIDES.index(side)])
                candidate = np.zeros(SHAPE, dtype=bool) if reference is None else reference
                rows.append({"image_id": image["id"], "prompt_key": side, "recording_id": image["recording_id"],
                    "quality_flags": image["quality_flags"], "reference_quality_flags": image["reference_quality_flags"][side],
                    "pair_quality_flags": image["quality_flags"], "legacy_fixed128": image["legacy_fixed128"],
                    "raw_provided_group": image["raw_provided_group"],
                    **compare.measure(candidate, reference, other, .8)})
        result = compare.scopes(rows)
        self.assertEqual(result["all_predictions_coverage"]["overall"]["queries"], 12)
        self.assertEqual(result["raw_all_provided"]["overall"]["queries"], 10)
        self.assertEqual(result["primary_quality_eligible"]["overall"]["queries"], 9)
        self.assertEqual(result["knownwrong_or_uncertain_raw"]["overall"]["queries"], 1)
        self.assertEqual(result["missing_reference_predictions_only"]["overall"]["queries"], 2)
        self.assertEqual(result["primary_quality_eligible"]["per_side"]["right_hand"]["queries"], 5)
        self.assertEqual(result["primary_quality_eligible"]["overall"]["opposite_proxy_eligible_queries"], 6)
        self.assertEqual(result["legacy_fixed128"]["overall"]["queries"], 4)


class FullComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "publication"
        make_publication(self.data_root)
        self.contract = compare.load_contract(self.data_root, expected_images=6, expected_legacy_images=2)
        self.ve = make_evaluations(self.root, self.contract, "ve-both", "ve-both")
        self.mixed = make_evaluations(self.root, self.contract, "residual", "mixed")
        self.nake = make_evaluations(self.root, self.contract, "residual", "nake")
        self.arguments = dict(ve_left_summaries=self.ve, ve_right_summaries=self.ve,
            mixed_summaries=self.mixed, nake_summaries=self.nake,
            mixed_checkpoint_sha256="a" * 64, nake_checkpoint_sha256="b" * 64,
            expected_images=6, expected_legacy_images=2)

    def run_comparison(self, **overrides):
        return compare.compare(self.data_root, **{**self.arguments, **overrides})

    def change_summary(self, path, change):
        summary = json.loads(path.read_text())
        change(summary)
        write_json(path, summary)

    def change_records(self, path, change):
        record_path = path.parent / "records.jsonl"
        records = [json.loads(line) for line in record_path.read_text().splitlines()]
        change(records)
        record_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8")
        self.change_summary(path, lambda summary: summary.update(records_sha256=compare.sha256(record_path)))

    def test_complete_uneven_shards_allow_shared_ve_both_and_preserve_actual_progress(self):
        result = self.run_comparison(mixed_summaries=list(reversed(self.mixed)))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evaluation_scope"], "fixed_development_benchmark")
        self.assertEqual(result["record_coverage"], {"ve-left": 6, "ve-right": 6, "mixed": 12, "nake": 12})
        self.assertEqual([json.loads(path.read_text())["image_ids"] for path in self.ve],
                         [[1, 5], [2, 6], [3], [4]])
        self.assertEqual(result["model_progress"]["mixed"]["actual_global_step"], 5021)
        self.assertEqual(result["model_progress"]["mixed"]["actual_image_exposures"], 30126)
        self.assertEqual(result["model_progress"]["nake"]["actual_global_step"], 3082)
        self.assertEqual(result["model_progress"]["nake"]["actual_image_exposures"], 18492)
        metrics = result["metrics_recomputed_from_saved_rle"]
        self.assertEqual(metrics["mixed"]["primary_quality_eligible"]["overall"]["queries"], 9)
        self.assertEqual(metrics["mixed"]["missing_reference_predictions_only"]["overall"]["queries"], 2)
        self.assertEqual(metrics["mixed"]["raw_all_provided"]["overall"]["absent_queries"], 3)
        self.assertEqual(metrics["mixed"]["primary_quality_eligible"]["overall"]["opposite_proxy_eligible_queries"], 6)
        for group in ("mixed", "nake"):
            for side in SIDES:
                delta = result["difference_residual_minus_corresponding_ve_side"][group][side]
                self.assertEqual(delta["primary_quality_eligible"]["candidate_dice"], 0.)
                self.assertEqual(delta["legacy_fixed128"]["miss_zero_dice"], 0.)

    def test_single_side_ve_reports_match_shared_both_results(self):
        expected = self.run_comparison()["metrics_recomputed_from_saved_rle"]
        left = make_evaluations(self.root, self.contract, "ve-left", "ve-left")
        right = make_evaluations(self.root, self.contract, "ve-right", "ve-right")
        result = self.run_comparison(ve_left_summaries=left, ve_right_summaries=right)
        for group in ("ve-left", "ve-right"):
            self.assertEqual(result["metrics_recomputed_from_saved_rle"][group], expected[group])

    def test_explicit_second_epoch_preserves_per_epoch_identity_and_report(self):
        for path in self.nake:
            self.change_summary(path, lambda summary: summary["model_metadata"]["progress"].update(
                global_step=6164, completed_epochs=2, next_epoch=2, samples_seen=36984))
        with self.assertRaises(ValueError):
            self.run_comparison()
        result = self.run_comparison(expected_nake_epoch=2)
        progress = result["model_progress"]["nake"]
        self.assertEqual(progress["actual_completed_epochs"], 2)
        self.assertEqual(progress["actual_global_step"], 6164)
        self.assertEqual(progress["actual_image_exposures"], 36984)
        self.assertIn("| nake | 2 | 6164 | 36984 |", compare.markdown_report(result))
        self.assertIn("nake epoch2=6164", " ".join(result["limitations"]))
        self.change_summary(self.nake[0], lambda summary: summary["model_metadata"]["progress"].update(next_step_in_epoch=1))
        with self.assertRaises(ValueError):
            self.run_comparison(expected_nake_epoch=2)

    def test_second_epoch_request_rejects_first_epoch_or_invalid_selection(self):
        for epoch in (2, 0, 3, True, 1.0):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                self.run_comparison(expected_nake_epoch=epoch)

    def test_second_epoch_must_fit_original_training_plan(self):
        for path in self.nake:
            self.change_summary(path, lambda summary: summary["model_metadata"]["progress"].update(
                global_step=6164, completed_epochs=2, next_epoch=2, samples_seen=36984))
            self.change_summary(path, lambda summary: summary["model_metadata"]["training_config"].update(epochs=1))
        with self.assertRaises(ValueError):
            self.run_comparison(expected_nake_epoch=2)

    def test_missing_duplicate_or_wrong_model_shard_rejected(self):
        for paths in (self.mixed[:-1], self.mixed + self.mixed[:1], self.ve):
            with self.subTest(paths=[str(path) for path in paths]), self.assertRaises(ValueError):
                self.run_comparison(mixed_summaries=paths)

    def test_explicit_checkpoint_sha_and_epoch_progress_are_not_interchangeable(self):
        with self.assertRaises(ValueError):
            self.run_comparison(mixed_checkpoint_sha256="b" * 64)
        self.change_summary(self.nake[0], lambda row: row["model_metadata"]["progress"].update(global_step=5021))
        with self.assertRaises(ValueError):
            self.run_comparison()

    def test_threshold_precision_and_dataset_scope_drift_rejected(self):
        original = self.mixed[0].read_text()
        for key, value in (("detection_threshold", .49), ("mask_threshold", .51),
                           ("boundary_width_original_pixels", 3), ("precision", "FP16"),
                           ("evaluation_scope", "blind_test"), ("expected_actual_inference_queries", 2)):
            with self.subTest(key=key):
                self.mixed[0].write_text(original)
                self.change_summary(self.mixed[0], lambda row: row.update({key: value}))
                with self.assertRaises(ValueError):
                    self.run_comparison()

    def test_ve_both_hidden_other_side_coverage_is_checked_before_side_filter(self):
        self.change_records(self.ve[0], lambda rows: rows.pop(1))
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_comparison()

    def test_duplicate_and_wrong_shard_records_rejected_even_with_new_record_hash(self):
        original = (self.mixed[0].parent / "records.jsonl").read_text()
        for wrong_shard in (False, True):
            with self.subTest(wrong_shard=wrong_shard):
                (self.mixed[0].parent / "records.jsonl").write_text(original)
                if wrong_shard:
                    self.change_records(self.mixed[0], lambda rows: rows[0].update(image_id=2))
                else:
                    self.change_records(self.mixed[0], lambda rows: rows.append(deepcopy(rows[0])))
                with self.assertRaises(ValueError):
                    self.run_comparison()

    def test_source_sha_drift_across_model_shards_rejected_when_local_binding_is_consistent(self):
        def change(summary):
            logical = "scripts/evaluate_realsense_full.py"
            summary["source_inference_sha256"][logical] = "f" * 64
            absolute = next(path for path in summary["implementation_sha256"] if path.endswith(logical))
            summary["implementation_sha256"][absolute] = "f" * 64
        self.change_summary(self.mixed[1], change)
        with self.assertRaisesRegex(ValueError, "inference source"):
            self.run_comparison()

    def test_declared_model_sha_cannot_disagree_with_actual_source_fingerprint(self):
        self.change_summary(self.mixed[0], lambda row: row["source_fingerprints"].update({"/run-inputs/mixed.pt": "f" * 64}))
        with self.assertRaises(ValueError):
            self.run_comparison()

    def test_saved_rle_and_unknown_reference_metrics_recomputed(self):
        self.change_records(self.mixed[0], lambda rows: rows[0].update(prediction_rle=rle(np.zeros(SHAPE, dtype=bool))))
        with self.assertRaisesRegex(ValueError, "recomput"):
            self.run_comparison()
        # Image 3/right has no supplied reference, not an observed absent hand.
        self.change_records(self.ve[2], lambda rows: rows[1].update(target_present=False, reference_pixels=0))
        with self.assertRaisesRegex(ValueError, "recomput"):
            self.run_comparison()

    def test_identical_checkpoint_bytes_can_be_relocated_between_shards(self):
        self.change_summary(self.mixed[1], lambda row: row["model_metadata"].update(checkpoint_path="/another-host/checkpoint.pt"))
        self.assertEqual(self.run_comparison()["status"], "complete")


if __name__ == "__main__":
    unittest.main()

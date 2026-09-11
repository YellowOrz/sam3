from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import compare_residual_test_results as comparison
from scripts import evaluate_residual_test as evaluation
from scripts import prepare_realsense_test as preparation
from scripts import evaluate_nakehand_tokens as bilateral
from scripts.prepare_nakehand_test import side_annotation


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fixture(root):
    data = root / "data"
    data.mkdir()
    selected = preparation.select_indices(preparation.FRAME_COUNTS, {})
    plan = {"format": preparation.FORMAT, "dataset_role": "external_test_only",
        "seed": preparation.SEED, "samples_per_recording": preparation.PER_RECORDING,
        "frame_counts": preparation.FRAME_COUNTS, "exclusions": {}, "selection": selected,
        "selection_uses_predictions_or_mask_pixels": False,
        "render_selection": {name: [frames[0], frames[len(frames)//2]] for name, frames in selected.items()},
        "reference_description": preparation.REFERENCE_DESCRIPTION}
    write(data / "frozen-plan.json", plan)
    plan_hash = comparison.sha256(data / "frozen-plan.json")
    coco = {"info": {"dataset_role": "external_test_only", "frozen_plan_sha256": plan_hash},
        "categories": preparation.CATEGORIES, "images": [], "annotations": []}
    manifest = {"status": "complete", "sources_unchanged": True, "frozen_plan_sha256": plan_hash,
                "statistics": {"images_with_left_right_overlap": 21}, "image_outputs": []}
    refs = {}
    for recording, frames in selected.items():
        for frame in frames:
            image_id = len(coco["images"]) + 1
            left, right = (np.zeros((480, 640), dtype=np.uint8) for _ in range(2))
            if image_id <= 83 or image_id % 2:
                left[10:20, 10:20] = 1
            if image_id <= 83 or image_id % 2 == 0:
                location = 15 if image_id <= 21 else 40
                right[location:location+10, location:location+10] = 1
            refs[image_id] = {"left_hand": left.astype(bool), "right_hand": right.astype(bool)}
            directory = data / "images" / recording / f"frame-{frame:06d}"
            directory.mkdir(parents=True)
            files = {}
            arrays = {"rgb": np.zeros((480, 640, 3), dtype=np.uint8), "left_hand": left, "right_hand": right}
            for key, array in arrays.items():
                path = directory / ("rgb.png" if key == "rgb" else f"{key}_raw.png")
                Image.fromarray(array).save(path)
                files[key] = {"path": path.relative_to(data).as_posix(), "sha256": comparison.sha256(path)}
            mapping = {"source_frame_index": frame, "frame_numbering": "zero_based_exported_video_not_bag_message",
                       "rgb_video": f"/original/{recording}/color.mp4",
                       "mask_videos": {side: f"/original/{recording}/{side}.mkv" for side in comparison.SIDES},
                       "pts_seconds": {key: frame / 30 for key in ("rgb", *comparison.SIDES)}}
            image = {"id": image_id, "file_name": files["rgb"]["path"], "width": 640, "height": 480,
                "source_dataset": "realsense", "recording_id": recording, "source_frame_index": frame,
                "source_mapping": mapping, "reference_provided": {side: True for side in comparison.SIDES},
                "render_preselected": frame in plan["render_selection"][recording]}
            coco["images"].append(image)
            manifest["image_outputs"].append({"image_id": image_id, "recording_id": recording,
                "source_frame_index": frame, "source_mapping": mapping, "files": files,
                "left_right_overlap_pixels": int((left.astype(bool) & right.astype(bool)).sum())})
            for category, side in enumerate(comparison.SIDES, 1):
                annotation = side_annotation(arrays[side], image_id, category, len(coco["annotations"]) + 1, mapping)
                if annotation is not None:
                    coco["annotations"].append(annotation)
    write(data / "annotations.json", coco)
    write(data / "manifest.json", manifest)
    write(data / "READY.json", {"format": preparation.FORMAT, "status": "complete",
        "dataset_role": "external_test_only", "frozen_plan_sha256": plan_hash,
        "annotations_sha256": comparison.sha256(data / "annotations.json"),
        "manifest_sha256": comparison.sha256(data / "manifest.json")})
    contract = comparison.load_contract(data)
    code_names = ("evaluate_residual_test.py", "evaluate_nakehand_tokens.py", "evaluate_bilateral_tokens.py",
        "cached_ve_text_features.py", "evaluate_ve_initialized_tokens.py", "residual_ddp_validation.py",
        "residual_ddp_checkpoint.py", "prepare_realsense_test.py", "train_ve_initialized_tokens.py")
    code = {f"/source/repo/scripts/{name}": "f" * 64 for name in code_names}
    code["/source/repo/sam3/model.py"] = "0" * 64
    base, tokenizer, cache, delta, train = (char * 64 for char in "abcde")
    config = {"steps_per_epoch": 10, "epochs": 100, "global_batch_size": 6, "dataset_size": 61,
              "base_sha256": base, "tokenizer_sha256": tokenizer, "initial_cache_file_sha256": cache,
              "annotations_sha256": train}
    progress = {"global_step": 15, "next_epoch": 1, "next_step_in_epoch": 5, "samples_seen": 90,
        "planned_steps": 1000, "planned_samples": 6000, "completed_epochs": 1,
        "training_complete": False, "dropped_images_per_epoch": 1}
    summaries, all_records = {}, {}
    for label in comparison.LABELS:
        directory = root / label
        directory.mkdir()
        records = []
        for image in coco["images"]:
            for index, side in enumerate(comparison.SIDES):
                reference, other = refs[image["id"]][side], refs[image["id"]][comparison.SIDES[1-index]]
                candidate = reference.copy()
                score = .9 if reference.any() else .1
                if label == "ve-natural":
                    if image["id"] == 1 and side == "left_hand":
                        score = .49
                    if not reference.any():
                        candidate, score = other.copy(), .7
                rle = mask_utils.encode(np.asfortranarray(candidate.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                record = {"model": label, "dataset_role": "external_test_only", "dataset_index": image["id"]-1,
                    "image_id": image["id"], "identity_verified": True, "primary_test": True, "diagnostic_ids": [],
                    **{key: image[key] for key in ("file_name", "recording_id", "source_frame_index", "source_mapping")},
                    "prompt_key": side, "prompt_text": evaluation.cached.NATURAL_PROMPTS[index],
                    "reference_description": preparation.REFERENCE_DESCRIPTION, "top_class_probability": score,
                    "presence_probability": 1., "selected_decoder_query": 2, "prediction_rle": rle,
                    "detections_above_threshold": int(score >= .5),
                    **bilateral.measure_query(candidate, reference, other, score)}
                records.append(evaluation.add_boundary(record, candidate, reference))
        metadata = ({"source": "actual original text encoder, natural prompts, no cached replacement"}
                    if label == "ve-natural" else {"checkpoint_sha256": delta, "progress": progress,
                    "training_config": config, "initial_cache_verified": True, "actual_training_identities_verified": True,
                    "rank_cache_consistency_audit_present_and_verified": True, "delta_l2_per_side": [1., 2.]})
        fingerprints = {**contract["fingerprints"], "/source/base.pt": base, "/source/tokenizer.gz": tokenizer}
        if label == "residual-preselected":
            fingerprints.update({"/source/initial.pt": cache, "/source/delta.pt": delta, "/source/train/annotations.json": train})
        visuals = []
        for image in coco["images"]:
            if image["render_preselected"]:
                target = directory / "visuals" / f"image-{image['id']:06d}"
                visuals.append({"image_id": image["id"], "dataset_index": image["id"]-1, "diagnostic_ids": [],
                                "directory": str(target), "comparison": str(target / "comparison.png")})
        summary = {"format": evaluation.FORMAT, "status": "complete", "dataset_role": "external_test_only",
            "images": 128, "queries_per_model": 256, "actual_complete_query_coverage_verified": True,
            "created_at_utc": "2026-09-11T00:00:00+00:00", "completed_at_utc": "2026-09-11T00:01:00+00:00",
            "elapsed_seconds": 60., "source_fingerprints": fingerprints, "implementation_sha256": deepcopy(code),
            "reference_description": preparation.REFERENCE_DESCRIPTION, "protocol": deepcopy(plan),
            "checkpoint_selection_note": None if label == "ve-natural" else "预先按 DexYCB val 选择；best_note 原样保留",
            "detection_threshold": .5, "mask_threshold": .5, "boundary_width_original_pixels": 4,
            "prediction_selection": comparison.PREDICTION_SELECTION, "precision": comparison.PRECISION,
            "batch_size": 1, "models": {label: metadata}, "metrics": {label: evaluation.summarize(records)},
            "limitations": ["Auxiliary-reference agreement, not independent human GT accuracy"], "visualizations": visuals}
        summaries[label], all_records[label] = summary, records
    return data, contract, summaries, all_records


class FixedComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="sam3-comparison-test-")
        cls.root = Path(cls.temporary.name)
        cls.data, cls.contract, cls.original_summaries, cls.original_records = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.summaries = deepcopy(self.original_summaries)
        self.records = deepcopy(self.original_records)

    def run_comparison(self, *, real_contract=False):
        for label in comparison.LABELS:
            write(self.root / label / "summary.json", self.summaries[label])
            write(self.root / label / f"records-{label}.json", self.records[label])
        args = [self.root / label / "summary.json" for label in comparison.LABELS]
        if real_contract:
            return comparison.compare(*args, self.data)
        with mock.patch.object(comparison, "load_contract", return_value=self.contract):
            return comparison.compare(*args, self.data)

    def test_real_128_image_schema_recomputes_metrics_overlap_and_actual_epoch(self):
        result = self.run_comparison(real_contract=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["reference_overlap"]["images"], 21)
        self.assertEqual(result["residual_checkpoint"]["actual_epochs"], 1.5)
        self.assertEqual(result["residual_checkpoint"]["planned_epochs"], 100)
        self.assertEqual(result["residual_checkpoint"]["best_note"], self.summaries[comparison.LABELS[1]]["checkpoint_selection_note"])
        delta = result["difference_residual_minus_ve"]["overall"]
        self.assertEqual(delta["false_negative_queries"], -1)
        self.assertEqual(delta["false_positive_queries"], -45)
        self.assertAlmostEqual(delta["present_mean_miss_zero_dice"], 1 / 211)
        self.assertEqual(delta["present_mean_candidate_dice"], 0.)
        self.assertEqual(len(result["visualizations"]["ve-natural"]), 16)
        self.assertEqual(len(result["metrics_recomputed_from_candidate_rle_and_frozen_references"]["ve-natural"]["per_recording"]), 8)
        report = comparison.markdown_report(result)
        self.assertIn("21 张", report)
        self.assertIn("实际完成 step 15", report)
        self.assertIn("1.5000 epoch", report)
        self.assertIn("不是独立全人工", report)
        self.assertIn("图像待汇入", report)

    def test_missing_duplicate_and_wrong_image_identity_rejected(self):
        for mode in ("missing", "duplicate", "wrong", "bool"):
            self.records = deepcopy(self.original_records)
            rows = self.records[comparison.LABELS[1]]
            if mode == "missing":
                rows.pop()
            elif mode == "duplicate":
                rows[-1] = deepcopy(rows[0])
            else:
                rows[0]["image_id"] = True if mode == "bool" else 129
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_comparison()

    def test_source_frame_recording_and_side_changes_rejected(self):
        for key, value in (("source_frame_index", 999), ("recording_id", "changed"), ("prompt_key", "object"),
                           ("dataset_index", True), ("identity_verified", 1)):
            self.records = deepcopy(self.original_records)
            self.records[comparison.LABELS[1]][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_comparison()

    def test_threshold_boundary_precision_and_incomplete_status_rejected(self):
        for key, value in (("detection_threshold", .49), ("mask_threshold", .4),
                           ("boundary_width_original_pixels", 5), ("status", "running"),
                           ("actual_complete_query_coverage_verified", 1), ("precision", "FP32")):
            self.summaries = deepcopy(self.original_summaries)
            self.summaries[comparison.LABELS[1]][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_comparison()

    def test_corrupt_row_aggregate_and_rle_are_not_trusted(self):
        for mode in ("row", "summary", "rle"):
            self.summaries, self.records = deepcopy(self.original_summaries), deepcopy(self.original_records)
            if mode == "row":
                self.records[comparison.LABELS[1]][0]["top_dice"] = .25
            elif mode == "summary":
                self.summaries[comparison.LABELS[1]]["metrics"][comparison.LABELS[1]]["overall"]["present_mean_candidate_dice"] = .25
            else:
                self.records[comparison.LABELS[1]][0]["prediction_rle"]["size"] = [8, 8]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_comparison()

    def test_reference_and_plan_fingerprint_change_rejected(self):
        for mode in ("fingerprint", "plan", "source_mapping"):
            self.summaries, self.records = deepcopy(self.original_summaries), deepcopy(self.original_records)
            if mode == "fingerprint":
                key = str(self.data / "annotations.json")
                self.summaries[comparison.LABELS[1]]["source_fingerprints"][key] = "1" * 64
            elif mode == "plan":
                self.summaries[comparison.LABELS[1]]["protocol"]["seed"] += 1
            else:
                self.records[comparison.LABELS[1]][0]["source_mapping"]["source_frame_index"] += 1
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_comparison()

    def test_visuals_cannot_be_reselected_from_results(self):
        rows = self.summaries[comparison.LABELS[1]]["visualizations"]
        rows[0]["image_id"] = 2
        with self.assertRaisesRegex(ValueError, "Visualization selection"):
            self.run_comparison()

    def test_checkpoint_progress_and_pretest_note_required(self):
        for mode in ("step", "epoch", "note", "model"):
            self.summaries = deepcopy(self.original_summaries)
            summary = self.summaries[comparison.LABELS[1]]
            if mode == "step":
                summary["models"][comparison.LABELS[1]]["progress"]["global_step"] = 0
            elif mode == "epoch":
                summary["models"][comparison.LABELS[1]]["progress"]["completed_epochs"] = 100
            elif mode == "note":
                summary["checkpoint_selection_note"] = None
            else:
                summary["models"]["picked-by-test-dice"] = summary["models"].pop(comparison.LABELS[1])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_comparison()

    def test_base_weight_and_inference_implementation_changes_rejected(self):
        for mode in ("base", "code"):
            self.summaries = deepcopy(self.original_summaries)
            summary = self.summaries[comparison.LABELS[0]]
            if mode == "base":
                summary["source_fingerprints"]["/source/base.pt"] = "2" * 64
            else:
                summary["implementation_sha256"]["/source/repo/sam3/model.py"] = "2" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_comparison()

    def test_nonfinite_and_confidence_count_inconsistency_rejected(self):
        for key, value in (("top_confidence", float("nan")), ("presence_probability", float("inf")),
                           ("detections_above_threshold", 0), ("selected_decoder_query", True)):
            self.records = deepcopy(self.original_records)
            self.records[comparison.LABELS[1]][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.run_comparison()

    def test_near_threshold_probability_cannot_change_detection(self):
        row = self.records[comparison.LABELS[1]][0]
        row["top_class_probability"] = .49999994
        row["top_confidence"] = .5
        with self.assertRaisesRegex(ValueError, "Confidence differs"):
            self.run_comparison()

    def test_missing_nullable_metrics_rejected(self):
        for name in ("top_dice", "miss_zero_dice", "candidate_boundary_iou_4px"):
            self.records = deepcopy(self.original_records)
            row = next(row for row in self.records[comparison.LABELS[1]] if not row["target_present"])
            del row[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing recomputed record field"):
                self.run_comparison()

    def test_encoder_metadata_audit_type_and_extraneous_fingerprints_rejected(self):
        for mode in ("encoder", "audit", "fingerprint"):
            self.summaries = deepcopy(self.original_summaries)
            if mode == "encoder":
                self.summaries[comparison.LABELS[0]]["models"][comparison.LABELS[0]]["source"] = "cached replacement"
            elif mode == "audit":
                self.summaries[comparison.LABELS[1]]["models"][comparison.LABELS[1]]["rank_cache_consistency_audit_present_and_verified"] = 1
            else:
                self.summaries[comparison.LABELS[0]]["source_fingerprints"]["/extra/file"] = "2" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.run_comparison()

    def test_reordered_records_preserve_all_metrics(self):
        self.records[comparison.LABELS[1]].reverse()
        result = self.run_comparison()
        self.assertEqual(result["difference_residual_minus_ve"]["overall"]["false_positive_queries"], -45)

    def test_cli_writes_json_markdown_and_refuses_existing_output(self):
        self.run_comparison()
        output = self.root / "comparison-output"
        args = ["--ve-summary", str(self.root / comparison.LABELS[0] / "summary.json"),
                "--residual-summary", str(self.root / comparison.LABELS[1] / "summary.json"),
                "--data-root", str(self.data), "--output-dir", str(output)]
        with mock.patch.object(comparison, "load_contract", return_value=self.contract):
            comparison.main(args)
            self.assertEqual(json.loads((output / "comparison.json").read_text())["status"], "complete")
            self.assertIn("原 VE", (output / "comparison.md").read_text())
            with self.assertRaisesRegex(ValueError, "new directory"):
                comparison.main(args)


if __name__ == "__main__":
    unittest.main()

from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from scripts import evaluate_realsense_full as full
from scripts.prepare_nakehand_test import side_annotation


def fixture(root):
    folder = root / "images" / "s" / "frame-000000"
    folder.mkdir(parents=True)
    pixels = np.zeros((12, 16), dtype=np.uint8)
    pixels[3:8, 4:10] = 2  # Preserve arbitrary positive source-instance IDs.
    Image.new("RGB", (16, 12), (50, 90, 120)).save(folder / "rgb.png")
    Image.fromarray(pixels).save(folder / "left_hand_raw.png")
    image = {"id": 1, "file_name": "images/s/frame-000000/rgb.png", "width": 16, "height": 12,
        "source_dataset": "realsense", "recording_id": "s", "frame_index": 0, "source_frame_index": 0,
        "source_mapping": {"rgb_video": "/explicit/source/s/rgb.mp4"},
        "reference_provided": {"left_hand": True, "right_hand": False}, "has_both_reference": False,
        "raw_provided_group": "left_only", "quality_flags": [],
        "reference_quality_flags": {"left_hand": [], "right_hand": []},
        "legacy_fixed128": False, "legacy_fixed128_image_id": None, "render_preselected": True}
    plan = {"frame_counts": {"s": 1}, "selection_uses_predictions_or_mask_pixels": False}
    (root / "frozen-plan.json").write_text(json.dumps(plan))
    plan_hash = full.shared.sha256(root / "frozen-plan.json")
    data = {"info": {"dataset_role": "external_test_only", "evaluation_scope": "fixed_development_benchmark",
             "frozen_plan_sha256": plan_hash}, "images": [image],
        "categories": full.fixed.preparation.CATEGORIES, "annotations": [side_annotation(pixels, 1, 1, 1)]}
    (root / "annotations.json").write_text(json.dumps(data))
    files = {name: {"path": str((folder / filename).relative_to(root)),
                   "sha256": full.shared.sha256(folder / filename)}
             for name, filename in (("rgb", "rgb.png"), ("left_hand", "left_hand_raw.png"))}
    manifest = {"status": "complete", "sources_unchanged": True, "frozen_plan_sha256": plan_hash,
        "image_outputs": [{"image_id": 1, "recording_id": "s", "source_frame_index": 0, "files": files}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    ready = {"format": full.DATA_FORMAT, "status": "complete", "dataset_role": "external_test_only",
             "evaluation_scope": "fixed_development_benchmark", "frozen_plan_sha256": plan_hash,
             "manifest_sha256": full.shared.sha256(root / "manifest.json"),
             "annotations_sha256": full.shared.sha256(root / "annotations.json")}
    (root / "READY.json").write_text(json.dumps(ready))
    return image, pixels > 0


def metric_row(reference, other, *, flags=(), pair_flags=(), score=.8):
    mask = np.ones((12, 16), dtype=bool)
    row = full.measure_partial(mask, reference, other, score)
    row.update(image_id=1, prompt_key="left_hand", recording_id="s", reference_provided=reference is not None,
               has_both_reference=reference is not None and other is not None,
               reference_quality_flags=list(flags), pair_quality_flags=list(pair_flags), legacy_fixed128=True,
               primary_test=reference is not None and not flags)
    return row


class FullRealSenseTest(unittest.TestCase):
    def test_spatial_mode_retains_both_sides(self):
        self.assertEqual(full.selected_sides('spatial'), full.SIDES)

    def test_shards_are_disjoint_complete_and_seed_free(self):
        images = [{"id": i} for i in range(1, 6205)]
        shards = [full.shard_indices(images, rank, 4) for rank in range(4)]
        self.assertEqual(sorted(index for shard in shards for index in shard), list(range(6204)))
        self.assertTrue(all(len(shard) == 1551 for shard in shards))
        with self.assertRaises(ValueError):
            full.shard_indices(images, 4, 4)
        with self.assertRaises(ValueError):
            full.shard_indices(list(reversed(images)), 0, 1)

    def test_selected_sides_and_ve_both(self):
        self.assertEqual(full.selected_sides("ve-left"), ("left_hand",))
        self.assertEqual(full.selected_sides("ve-right"), ("right_hand",))
        self.assertEqual(full.selected_sides("ve-both"), full.selected_sides("residual"))

    def test_missing_reference_is_null_and_never_false_positive(self):
        mask = np.ones((12, 16), dtype=bool)
        row = metric_row(None, mask)
        for field in ("target_present", "reference_pixels", "top_dice", "miss_zero_dice",
                      "candidate_boundary_iou_4px", "any_hand_present", "opposite_overlap_dominant_proxy"):
            self.assertIsNone(row[field], field)
        summary = full.summarize([row])
        self.assertEqual(summary["raw_all_provided"]["overall"]["queries"], 0)
        unknown = summary["unknown_reference_predictions_only"]["overall"]
        self.assertEqual(unknown["unknown_reference_queries"], 1)
        self.assertEqual(unknown["false_positive_queries"], 0)
        self.assertEqual(unknown["detections_all_queries"], 1)

    def test_provided_empty_is_a_real_negative(self):
        empty = np.zeros((12, 16), dtype=bool)
        row = metric_row(empty, None)
        summary = full.summarize([row])["primary_provided_nonflagged"]["overall"]
        self.assertEqual(summary["false_positive_queries"], 1)
        self.assertEqual(summary["absent_side_false_positive_rate"], 1.)
        self.assertEqual(summary["known_both_reference_queries_for_side_proxy"], 0)

    def test_missed_perfect_candidate_dice_is_retained(self):
        mask = np.ones((12, 16), dtype=bool)
        row = metric_row(mask, None, score=.49)
        self.assertEqual(row["top_dice"], 1.)
        self.assertEqual(row["miss_zero_dice"], 0.)
        self.assertEqual(row["candidate_boundary_iou_4px"], 1.)
        self.assertEqual(row["miss_zero_boundary_iou_4px"], 0.)

    def test_flagged_own_reference_only_in_raw(self):
        mask = np.ones((12, 16), dtype=bool)
        row = metric_row(mask, mask, flags=["known_wrong"], pair_flags=["known_wrong"])
        groups = full.summarize([row])
        self.assertEqual(groups["primary_provided_nonflagged"]["overall"]["queries"], 0)
        self.assertEqual(groups["raw_all_provided"]["overall"]["queries"], 1)
        self.assertEqual(groups["known_issue_provided"]["overall"]["queries"], 1)

    def test_other_flag_does_not_remove_own_dice_but_removes_confusion_proxy(self):
        mask = np.ones((12, 16), dtype=bool)
        row = metric_row(mask, mask, pair_flags=["other_wrong"])
        groups = full.summarize([row])
        self.assertEqual(groups["primary_provided_nonflagged"]["overall"]["queries"], 1)
        self.assertEqual(groups["primary_provided_nonflagged"]["overall"]["known_both_reference_queries_for_side_proxy"], 0)
        self.assertEqual(groups["raw_all_provided"]["overall"]["known_both_reference_queries_for_side_proxy"], 1)

    def test_quality_flags_are_side_specific(self):
        image = {"quality_flags": ["shown", "left_wrong"],
                 "reference_quality_flags": {"left_hand": ["left_wrong"], "right_hand": []}}
        self.assertEqual(full.quality_flags(image, "right_hand"), [])
        self.assertEqual(full.quality_flags(image, "left_hand"), ["left_wrong"])

    def test_loader_is_lazy_and_checks_positive_id_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, expected = fixture(root)
            with mock.patch.object(full.shared, "decode_gt_mask", side_effect=AssertionError("must stay lazy")):
                images, annotations, outputs, _, hashes = full.load_publication(root)
            refs = full.batch_references(root, images, [0], annotations, outputs, hashes)
            self.assertTrue(np.array_equal(refs[1]["left_hand"], expected))
            self.assertIsNone(refs[1]["right_hand"])
            self.assertEqual(len(hashes), 6)

    def test_hash_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            (root / "annotations.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                full.load_publication(root)

    def test_asset_changes_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            images, annotations, outputs, _, hashes = full.load_publication(root)
            (root / images[0]["file_name"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Asset changed"):
                full.batch_references(root, images, [0], annotations, outputs, hashes)

    def test_path_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("../out.png", "/tmp/out.png"):
                with self.assertRaises(ValueError):
                    full.local_file(root, relative)
            (root / "link").symlink_to(root / "target")
            with self.assertRaises(ValueError):
                full.local_file(root, "link")

    def test_each_batch_flushes_complete_json_lines(self):
        handle = mock.Mock(wraps=io.StringIO())
        full.append_batch(handle, [{"image_id": 1}, {"image_id": 2}])
        handle.flush.assert_called_once()
        self.assertEqual([json.loads(line)["image_id"] for line in handle.getvalue().splitlines()], [1, 2])

    def test_unknown_reference_visual_is_gray_not_negative_black(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, candidate = fixture(root)
            result = full.render_one(root, root/"out", image, "right_hand", None, candidate,
                                     {"detected": False, "top_confidence": .49, "reference_quality_flags": []})
            reference = np.asarray(Image.open(Path(result["comparison"]).parent/"reference.png"))
            detected = np.asarray(Image.open(Path(result["comparison"]).parent/"detected.png"))
            self.assertTrue(bool((reference == 128).all()))
            self.assertFalse(bool(detected.any()))

    def test_real_cpu_loader_and_mock_model_use_confidence_not_gt(self):
        class FakeModel(torch.nn.Module):
            num_interactive_steps_val = 0
            def forward(self, batch):
                masks = torch.full((2, 2, 12, 16), -10.)
                masks[:, 1, 3:8, 4:10] = 10.  # Low-score decoder matches GT; must NOT be selected.
                return [{"pred_logits": torch.tensor([[[3.], [-2.]], [[3.], [-2.]]]),
                         "presence_logit_dec": torch.tensor([[3.], [3.]]), "pred_masks": masks}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            images, annotations, outputs, _, hashes = full.load_publication(root)
            for mode in ("ve-both", "ve-left", "ve-right"):
                dataset = full.semantic.IdentityCheckedDataset(full.shared.make_dataset(root), images)
                out = root/mode
                out.mkdir()
                with mock.patch("sam3.model.utils.misc.copy_data_to_device", side_effect=lambda batch, *a, **k: batch), \
                     mock.patch.object(torch.amp, "autocast", side_effect=lambda *a, **k: nullcontext()):
                    records, _ = full.evaluate(FakeModel().eval(), mode, dataset, images, [0], annotations,
                                               outputs, root, out, hashes, 1)
                self.assertEqual([row["prompt_key"] for row in records], list(full.selected_sides(mode)))
                for row in records:
                    self.assertEqual(row["selected_decoder_query"], 0)
                    if row["prompt_key"] == "left_hand":
                        self.assertEqual(row["top_dice"], 0.)
                    else:
                        self.assertIsNone(row["target_present"])
                saved = [json.loads(line) for line in (out/"records.jsonl").read_text().splitlines()]
                self.assertTrue(all("prediction_rle" in row for row in saved))
                self.assertFalse(any("prediction_rle" in row for row in records))


if __name__ == "__main__":
    unittest.main()

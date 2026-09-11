import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts import diagnose_generic_hand_prompt as diagnostic
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as DataImage, InferenceMetadata, Object
from sam3.train.data.collator import collate_fn_api


def reference_masks():
    left = np.zeros((6, 8), dtype=bool)
    right = left.copy()
    left[1:5, 1:3] = True
    right[1:5, 5:7] = True
    return {"left_hand": left, "right_hand": right}


def outputs():
    refs = reference_masks()
    left, right = refs.values()
    masks = torch.full((3, 3, 6, 8), -8.)
    for row in range(3):
        masks[row, 0][torch.from_numpy(left)] = 8.
        masks[row, 1][torch.from_numpy(right)] = 8.
    # The left prompt's top-score prediction deliberately identifies the right hand.
    masks[1, 0] = masks[1, 1]
    probabilities = torch.tensor([[[.9], [.8], [.2]], [[.9], [.1], [.2]], [[.1], [.9], [.2]]])
    return {"pred_logits": torch.logit(probabilities), "presence_logit_dec": torch.logit(torch.tensor([[.9], [.9], [.9]])),
            "pred_boxes": torch.full((3, 3, 4), .5), "pred_masks": masks}


def sample():
    queries = []
    for category, side in enumerate(("left_hand", "right_hand"), 1):
        metadata = InferenceMetadata(coco_image_id=77, original_image_id=77, original_category_id=category,
                                     original_size=(6, 8), object_id=0, frame_index=0)
        queries.append(FindQueryLoaded(query_text=side, image_id=0, object_ids_output=[0],
                                       is_exhaustive=True, inference_metadata=metadata,
                                       semantic_target=torch.ones(6, 8, dtype=torch.bool)))
    obj = Object(bbox=torch.tensor([.5, .5, .3, .4]), area=8., segment=torch.ones(6, 8, dtype=torch.bool))
    return Datapoint(find_queries=queries, images=[DataImage(data=torch.zeros(3, 6, 8), objects=[obj], size=(6, 8))])


class GenericSelectionTest(unittest.TestCase):
    def test_fixed_abc_then_earliest_both_visible_at_most_eight(self):
        images = [{"id": i, "diagnostic_id": {7: "A", 8: "B", 9: "C"}.get(i)} for i in range(10)]
        refs = {i: reference_masks() for i in range(10)}
        manifest = [{"image_id": i, "dataset_index": i, "diagnostic_ids": diagnostic.evaluation.diagnostic_ids(images[i])}
                    for i in reversed(range(10))]
        refs[0]["left_hand"][:] = False
        self.assertEqual(diagnostic.select_diagnostic_indices(images, refs, manifest), [7, 8, 9, 1, 2, 3, 4, 5])

    def test_missing_diagnostic_and_duplicate_manifest_rejected(self):
        images = [{"id": i, "diagnostic_id": name} for i, name in enumerate(("A", "B", "C"))]
        refs = {i: reference_masks() for i in range(3)}
        manifest = [{"image_id": i, "dataset_index": i, "diagnostic_ids": [image["diagnostic_id"]]} for i, image in enumerate(images)]
        for invalid in (manifest[:2], manifest + manifest[:1]):
            with self.assertRaises(ValueError):
                diagnostic.select_diagnostic_indices(images, refs, invalid)


class GenericModelInputsTest(unittest.TestCase):
    def test_real_dataclasses_collator_strip_targets_without_mutating_source(self):
        original = sample()
        prepared = diagnostic.make_prompt_only_sample(original)
        self.assertEqual(len(original.images[0].objects), 1)
        self.assertEqual(original.find_queries[0].object_ids_output, [0])
        self.assertIsNotNone(original.find_queries[0].semantic_target)
        self.assertIs(prepared.images[0].data, original.images[0].data)
        self.assertEqual(prepared.images[0].objects, [])
        self.assertEqual([query.query_text for query in prepared.find_queries], list(diagnostic.PROMPTS))
        for query in prepared.find_queries:
            self.assertEqual(query.object_ids_output, [])
            self.assertIsNone(query.semantic_target)
        batch = collate_fn_api([prepared], dict_key="eval", with_seg_masks=True)["eval"]
        diagnostic.validate_prompt_batch(batch, {"id": 77, "height": 6, "width": 8})
        self.assertEqual(batch.find_targets[0].num_boxes.tolist(), [0, 0, 0])

    def test_prompt_inputs_and_wrong_observed_identity_rejected(self):
        original = sample()
        original.find_queries[0].input_bbox = torch.ones(4)
        with self.assertRaises(ValueError):
            diagnostic.make_prompt_only_sample(original)
        batch = collate_fn_api([diagnostic.make_prompt_only_sample(sample())], dict_key="eval", with_seg_masks=True)["eval"]
        with self.assertRaises(RuntimeError):
            diagnostic.validate_prompt_batch(batch, {"id": 78, "height": 6, "width": 8})


class GenericPostprocessTest(unittest.TestCase):
    def test_keep_all_generic_instances_and_own_top_side_without_gt_selection(self):
        metadata, predictions = diagnostic.postprocess_outputs(outputs(), (6, 8))
        refs = reference_masks()
        self.assertEqual(metadata["hand"]["retained_decoder_queries"], [0, 1])
        self.assertEqual(metadata["hand"]["all_score_passing_count"], 2)
        self.assertTrue(np.array_equal(predictions["hand"]["detected_union"], refs["left_hand"] | refs["right_hand"]))
        self.assertEqual(metadata["left_hand"]["top_decoder_query"], 0)
        self.assertTrue(np.array_equal(predictions["left_hand"]["detected_union"], refs["right_hand"]))
        measured = diagnostic.measure_diagnostic(predictions, refs)
        self.assertEqual(measured["generic_all_detected_union_vs_reference_union"]["own_dice"], 1.)
        self.assertEqual(measured["per_side"]["left_hand"]["detected_union"]["own_dice"], 0.)

    def test_presence_product_and_threshold_boundary(self):
        output = outputs()
        output["pred_logits"][:] = 0.  # sigmoid=0.5
        output["presence_logit_dec"][:] = 80.  # sigmoid rounds to 1
        output["pred_masks"][:] = 0.
        metadata, predictions = diagnostic.postprocess_outputs(output, (6, 8))
        self.assertEqual(metadata["hand"]["all_score_passing_count"], 3)
        self.assertEqual(metadata["hand"]["top_score"], .5)
        self.assertTrue(predictions["hand"]["detected_union"].all())
        output["presence_logit_dec"][:] = 0.
        metadata, predictions = diagnostic.postprocess_outputs(output, (6, 8))
        self.assertEqual(metadata["hand"]["top_score"], .25)
        self.assertEqual(metadata["hand"]["all_score_passing_count"], 0)
        self.assertFalse(predictions["hand"]["detected_union"].any())
        self.assertTrue(predictions["hand"]["top_candidate"].all())

    def test_no_nms_or_oracle_deduplication_of_generic_instances(self):
        output = outputs()
        output["pred_masks"][0, 1] = output["pred_masks"][0, 0]
        metadata, predictions = diagnostic.postprocess_outputs(output, (6, 8))
        self.assertEqual(metadata["hand"]["all_score_passing_count"], 2)
        self.assertEqual(len(predictions["hand"]["instances"]), 2)

    def test_resize_is_existing_evaluator_logit_interpolation(self):
        output = outputs()
        generator = torch.Generator().manual_seed(94)
        output["pred_masks"] = torch.randn(3, 3, 3, 4, generator=generator)
        _, predictions = diagnostic.postprocess_outputs(output, (6, 8))
        expected = F.interpolate(output["pred_masks"][1, 0][None, None].float(), size=(6, 8),
                                 mode="bilinear", align_corners=False)[0, 0].sigmoid().numpy() >= .5
        self.assertTrue(np.array_equal(predictions["left_hand"]["top_candidate"], expected))

    def test_nonfinite_or_wrong_shapes_rejected(self):
        for mutation in ("nan", "presence", "boxes", "rows"):
            output = outputs()
            if mutation == "nan":
                output["pred_masks"][2, 2, 0, 0] = float("nan")
            elif mutation == "presence":
                output["presence_logit_dec"] = torch.ones(3, 2)
            elif mutation == "boxes":
                output["pred_boxes"] = torch.ones(3, 3, 5)
            else:
                output["pred_logits"] = output["pred_logits"][:2]
            with self.assertRaises(ValueError):
                diagnostic.postprocess_outputs(output, (6, 8))

    def test_png_outputs_are_separate_binary_roundtrips(self):
        metadata, predictions = diagnostic.postprocess_outputs(outputs(), (6, 8))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rgb = root / "source.png"
            Image.new("RGB", (8, 6), (4, 5, 6)).save(rgb)
            artifacts = diagnostic.save_visuals(root / "visuals", rgb, reference_masks(), predictions)
            self.assertTrue((root / "visuals/left_hand__reference.png").is_file())
            self.assertTrue((root / "visuals/hand__instance-query-000.png").is_file())
            self.assertTrue((root / "visuals/right_hand__detected_union.png").is_file())
            for artifact in artifacts:
                self.assertEqual(len(artifact["sha256"]), 64)
            row = {"image_id": 77, "diagnostic_ids": ["A"], "prompts": metadata,
                   "metrics": diagnostic.measure_diagnostic(predictions, reference_masks())}
            report = diagnostic.markdown_report({"records": [row]})
            self.assertIn("不是完整测试集结论", report)
            self.assertIn("不能直接称为前臂泄漏", report)
            self.assertIn("严格 >0.5", report)
            json.dumps(row, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

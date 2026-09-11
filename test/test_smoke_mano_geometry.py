import copy
import unittest

import torch

from scripts.prepare_mano_sidecar import FORMAT, PARAMETER_SIZES
from scripts.smoke_mano_geometry import (
    adapter_gradient_norm,
    adapter_snapshot,
    mano_for_two_queries,
    parameter_max_change,
    select_smoke_rows,
    validate_row_identity,
)


def row(image_id=0, side="left", visible=True, valid=True):
    return {
        "format": FORMAT, "split": "train", "image_id": image_id,
        "file_name": f"images/{image_id}.jpg", "source": "dexycb",
        "sequence": "sequence-01", "view": "camera-01", "frame_index": image_id,
        "source_frame": image_id, "source_frame_mapping": "dexycb_contiguous_zero_based_identity",
        "segmentation_target_present": visible,
        "segmentation_annotation_ids": [image_id] if visible else [],
        "hands": [{
            "side": side, "valid": valid, "root_frame": "camera", "pose_representation": "axis-angle",
            **{name: [0.1] * size if valid else None for name, size in PARAMETER_SIZES.items()},
        }],
    }


class SmokeManoGeometryHelpersTest(unittest.TestCase):
    def test_selection_is_deterministic_and_prefers_invalid_empty(self):
        rows = [row(index, side="right" if index % 2 else "left") for index in range(8)]
        rows.extend([row(10, visible=False), row(11, visible=False, valid=False)])
        first = select_smoke_rows(iter(rows), seed=17)
        second = select_smoke_rows(iter(rows), seed=17)
        self.assertEqual(first, second)
        self.assertEqual([name for name, _ in first], ["left_valid", "right_valid", "empty"])
        self.assertEqual(first[-1][1]["image_id"], 11)
        self.assertTrue(first[0][1]["segmentation_target_present"])
        self.assertTrue(first[1][1]["hands"][0]["valid"])

    def test_selection_allows_valid_empty_but_requires_both_visible_sides(self):
        rows = [row(0), row(1, "right"), row(2, visible=False)]
        self.assertTrue(select_smoke_rows(rows, 1)[-1][1]["hands"][0]["valid"])
        with self.assertRaisesRegex(ValueError, "left and right"):
            select_smoke_rows([row(), row(2, visible=False)], 1)

    def test_both_prompts_get_identical_geometry_even_when_text_order_reverses(self):
        sample = row(side="right")
        mano = mano_for_two_queries(sample, torch.tensor([0, 0]), torch.tensor([1, 0]), "cpu")
        self.assertEqual(mano["side"].tolist(), [1, 1])
        self.assertEqual(mano["valid"].tolist(), [True, True])
        for name, dimension in PARAMETER_SIZES.items():
            self.assertEqual(mano[name].shape, (2, dimension))
            torch.testing.assert_close(mano[name][0], mano[name][1])
        self.assertEqual(sample["hands"][0]["hand_pose"], [0.1] * 45)

    def test_missing_mano_is_masked_nan_not_a_valid_zero_hand(self):
        mano = mano_for_two_queries(row(valid=False, visible=False), torch.tensor([0, 0]), torch.tensor([0, 1]), "cpu")
        self.assertEqual(mano["valid"].tolist(), [False, False])
        for name in PARAMETER_SIZES:
            self.assertTrue(torch.isnan(mano[name]).all())
        sample = row(valid=False)
        sample["hands"][0]["global_orient"] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "must be null"):
            mano_for_two_queries(sample, torch.tensor([0, 0]), torch.tensor([0, 1]), "cpu")

    def test_wrong_batch_or_coordinate_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            mano_for_two_queries(row(), torch.tensor([0, 1]), torch.tensor([0, 1]), "cpu")
        sample = row()
        sample["hands"][0]["root_frame"] = "world"
        with self.assertRaisesRegex(ValueError, "camera-frame"):
            mano_for_two_queries(sample, torch.tensor([0, 0]), torch.tensor([0, 1]), "cpu")

    def test_row_identity_rejects_wrong_frame_visibility_and_side(self):
        sample = row(7)
        image = {"id": 7, **{key: sample[key] for key in ("file_name", "source", "sequence", "view", "frame_index")}}
        annotations = [{"id": 7, "category_id": 1}]
        validate_row_identity(sample, image, annotations, "train")
        invalid = copy.deepcopy(sample)
        invalid["frame_index"] = 9
        with self.assertRaisesRegex(ValueError, "frame_index mismatch"):
            validate_row_identity(invalid, image, annotations, "train")
        with self.assertRaisesRegex(ValueError, "visibility mismatch"):
            validate_row_identity(sample, image, [], "train")
        with self.assertRaisesRegex(ValueError, "physical side contradicts"):
            validate_row_identity(sample, image, [{"id": 7, "category_id": 2}], "train")

    def test_gradient_measurement_does_not_claim_missing_or_zero_gradient_update(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.assertEqual(adapter_gradient_norm([parameter]), 0.0)
        parameter.grad = torch.zeros(2)
        self.assertEqual(adapter_gradient_norm([parameter]), 0.0)
        parameter.grad = torch.tensor([3.0, 4.0])
        self.assertEqual(adapter_gradient_norm([parameter]), 5.0)
        parameter.grad[0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            adapter_gradient_norm([parameter])

    def test_parameter_change_compares_independent_snapshots(self):
        adapter = torch.nn.Linear(2, 2)
        before = adapter_snapshot(adapter)
        self.assertEqual(parameter_max_change(adapter, before), 0.0)
        with torch.no_grad():
            adapter.weight.add_(1.0)
        self.assertAlmostEqual(parameter_max_change(adapter, before), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

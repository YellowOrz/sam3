from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import audit_realsense_mano_numeric as audit


def fixture(path, *, legacy=False):
    orientation = np.stack([np.eye(3), np.full((3, 3), np.nan)]).astype(np.float32)[:, None]
    fields = {"hand": np.array("left_hand"), "pose_format": np.array("rotation_matrix"),
              "frame_indices": np.arange(2), "total_frames": np.array(2), "width": np.array(8), "height": np.array(6),
              "bbox_xyxy": np.array([[1, 1, 3, 4], [np.nan] * 4], dtype=np.float32),
              "global_orient": orientation, "hand_pose": np.repeat(orientation, 15, axis=1),
              "confidence": np.array([.8, np.nan], np.float32)}
    if not legacy:
        fields.update(has_hand=np.array([True, False]), instance_label=np.array(1, np.uint8), mask_source=np.array("left_hand"))
    np.savez_compressed(path, **fields)
    return fields


class ManoNumericAuditTest(unittest.TestCase):
    def test_mask_box_is_exclusive_and_label_union_are_not_conflated(self):
        pixels = np.zeros((6, 8), np.uint8)
        pixels[1:4, 1:3] = 1
        pixels[5, 7] = 2
        np.testing.assert_array_equal(audit.tight_box(pixels == 1), [1, 1, 3, 4])
        np.testing.assert_array_equal(audit.tight_box(pixels > 0), [1, 1, 8, 6])
        self.assertTrue(np.isnan(audit.tight_box(pixels == 3)).all())

    def test_full_npz_body_expected_absent_nan_is_not_valid_frame_corruption(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "test.npz"
            fixture(path)
            report, arrays, label = audit.inspect_npz(path, 2)
            self.assertEqual(label, 1)
            self.assertEqual(report["schema"], "instance_label_and_has_hand")
            self.assertEqual(report["arrays"]["global_orient"]["nonfinite_total"], 9)
            self.assertEqual(report["arrays"]["global_orient"]["nonfinite_valid_frame_indices"], [])
            self.assertEqual(report["arrays"]["global_orient"]["invalid_frame_all_nan_count"], 1)
            self.assertEqual(report["rotations"]["hand_pose"]["matrices_outside_tolerance"], 0)

    def test_legacy_missing_flags_is_explicit_not_fabricated_ground_truth(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "legacy.npz"
            fixture(path, legacy=True)
            report, arrays, label = audit.inspect_npz(path, 2)
            self.assertIsNone(label)
            self.assertEqual(report["schema"], "legacy_no_instance_label_no_has_hand")
            self.assertIn("assumption", report["mask_selector"])
            self.assertIn("inferred for audit only", report["validity_source"])
            self.assertIsNone(audit.compare_mask_contract(arrays["bbox_xyxy"], arrays)["has_hand_disagrees_current_mask"])

    def test_rotations_reject_scaling_reflections_and_missing_valid_values(self):
        rotations = np.stack([np.eye(3), np.diag([-1, 1, 1]), np.eye(3) * 2, np.full((3, 3), np.nan)])[:, None]
        result = audit.rotation_statistics(rotations, np.ones(4, bool))
        self.assertEqual(result["matrices_outside_tolerance"], 2)
        self.assertEqual(result["nonfinite_rotation_matrices_on_valid_frames"], 1)
        self.assertEqual(result["determinant_min"], -1.)

    def test_bbox_and_presence_revision_mismatch_are_separate(self):
        stored = {"bbox_xyxy": np.array([[1, 1, 3, 4], [2, 2, 4, 5], [np.nan] * 4]),
                  "has_hand": np.array([True, True, False])}
        current = np.array([[1, 1, 4, 4], [np.nan] * 4, [1, 1, 2, 2]])
        result = audit.compare_mask_contract(current, stored)
        self.assertEqual(result["bbox_mismatch_frame_indices"], [0])
        self.assertEqual(result["mask_empty_but_bbox_finite"], [1])
        self.assertEqual(result["mask_present_but_bbox_nonfinite"], [2])
        self.assertEqual(result["has_hand_disagrees_current_mask"], [1, 2])

    def test_object_arrays_are_not_unpickled(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "object.npz"
            np.savez(path, unsafe=np.array([{"do_not": "load"}], dtype=object))
            with self.assertRaises(ValueError):
                audit.inspect_npz(path, 2)


if __name__ == "__main__":
    unittest.main()

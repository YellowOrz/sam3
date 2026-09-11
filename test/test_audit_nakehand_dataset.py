"""CPU regression checks for read-only nakehand diagnostics."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.audit_nakehand_dataset import extract, npz_summary, sha256


class NakehandAuditTest(unittest.TestCase):
    def test_npz_distinguishes_invalid_nan_and_records_source_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.npz"
            rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None], 3, axis=0)
            rotations[1] = np.nan
            betas = np.zeros((3, 10), dtype=np.float32)
            betas[1] = np.nan
            np.savez(path, has_hand=np.array([True, False, True]), frame_indices=np.arange(3),
                     global_orient=rotations, pose_format=np.array("rotation_matrix"),
                     camera_translation=np.array([[0, 0, 10], [np.nan] * 3, [0, 0, 70]], dtype=np.float32),
                     confidence=np.array([1, np.nan, 1], dtype=np.float32),
                     bbox_xyxy=np.array([[0, 0, 20, 30], [np.nan] * 4, [630, 450, 640, 480]], dtype=np.float32),
                     betas=betas, wilor_right_canonical_beta10=np.zeros(10, dtype=np.float32))
            before = sha256(path)
            summary = npz_summary(path)
            self.assertEqual(sha256(path), before)
            self.assertEqual(summary["valid_rows"], 2)
            self.assertEqual(summary["invalid_rows"], 1)
            translation = summary["arrays"]["camera_translation"]
            self.assertTrue(translation["finite_valid_values"])
            self.assertFalse(translation["finite_invalid_values"])
            self.assertEqual(translation["max_z_source_frame"], 2)
            self.assertEqual(translation["max_z_confidence"], 1)
            self.assertEqual(translation["z_quantiles_source_units"][-1], 70)
            self.assertIn("not inferred", translation["unit_status"])
            self.assertEqual(summary["arrays"]["betas"]["max_abs_delta_fixed_shape"], 0)
            self.assertEqual(summary["arrays"]["global_orient"]["max_RtR_error"], 0)

    def test_npz_does_not_assume_sparse_frame_indices_are_row_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.npz"
            np.savez(path, has_hand=np.array([False, False]), frame_indices=np.array([3, 8]))
            summary = npz_summary(path)
            self.assertFalse(summary["frame_indices_identity"])
            self.assertEqual(summary["frame_indices_range"], [3, 8])

    @patch("scripts.audit_nakehand_dataset.subprocess.run")
    def test_extract_keeps_values_and_requires_requested_pts(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, bytes([0, 2, 1, 0]), b"n: 0 pts: 1000 pts_time:1.000")
        array, pts = extract(Path("mask.mkv"), 30, 30, 2, 2, gray=True)
        self.assertEqual(array.tolist(), [[0, 2], [1, 0]])
        self.assertEqual(pts, 1)
        run.return_value.stderr = b"n: 0 pts: 1033 pts_time:1.033"
        with self.assertRaisesRegex(ValueError, "expected frame"):
            extract(Path("mask.mkv"), 30, 30, 2, 2, gray=True)

    @patch("scripts.audit_nakehand_dataset.subprocess.run")
    def test_extract_rejects_decoded_dimensions(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, bytes([0, 2]), b"pts_time:0")
        with self.assertRaisesRegex(ValueError, "byte length"):
            extract(Path("rgb.mkv"), 0, 30, 2, 2)


if __name__ == "__main__":
    unittest.main()

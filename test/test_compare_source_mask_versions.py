"""CPU tests for raw-ID versus foreground comparison, timestamps and mutation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import compare_source_mask_versions as compare


class CompareSourceMaskVersionsTest(unittest.TestCase):
    def test_relabeling_keeps_identical_foreground(self):
        a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        b = np.array([[0, 2], [3, 0]], dtype=np.uint8)
        row = compare.compare_frame(a, b)
        self.assertEqual(row["raw_different_pixels"], 2)
        self.assertEqual(row["binary_different_pixels"], 0)
        self.assertEqual(row["binary_dice"], 1.)
        self.assertEqual(compare.classification(True, False, False), "same_nonzero_foreground_masks_but_raw_label_values_differ")

    def test_spatial_changes_are_not_instance_relabeling(self):
        a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        b = np.array([[0, 2], [0, 2]], dtype=np.uint8)
        row = compare.compare_frame(a, b)
        self.assertEqual(row["binary_different_pixels"], 2)
        self.assertEqual(row["a_only_pixels"], 1)
        self.assertEqual(row["b_only_pixels"], 1)
        self.assertEqual(row["binary_dice"], .5)
        self.assertAlmostEqual(row["binary_iou"], 1 / 3)
        self.assertEqual(row["binary_difference_bbox_xyxy_exclusive"], [0, 1, 2, 2])
        self.assertIn("visible_foreground_masks_differ", compare.classification(True, True, False))

    def test_empty_and_shape_contract(self):
        empty = np.zeros((2, 3), dtype=np.uint8)
        row = compare.compare_frame(empty, empty)
        self.assertEqual(row["binary_iou"], 1.)
        self.assertIsNone(row["binary_difference_bbox_xyxy_exclusive"])
        with self.assertRaises(ValueError):
            compare.compare_frame(empty, empty.astype(np.uint16))
        with self.assertRaises(ValueError):
            compare.compare_frame(empty, empty.T)

    def test_same_decoded_values_different_file_bytes(self):
        self.assertEqual(compare.classification(False, False, False), "files_differ_but_all_decoded_raw_mask_frames_identical")
        self.assertEqual(compare.classification(False, False, True), "files_identical_and_decoded_frames_identical")

    def test_source_hash_stat_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mkv"
            source.write_bytes(b"fixture")
            first = compare.source_receipt(source)
            self.assertEqual(compare.source_receipt(source), first)
            source.write_bytes(b"changed")
            self.assertNotEqual(compare.source_receipt(source), first)
            with self.assertRaises(FileExistsError):
                compare.main(["--source-a", str(source), "--source-b", str(source), "--output", str(source)])

    def test_source_mutation_during_hash_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mkv"
            source.write_bytes(b"fixture")
            with patch.object(compare, "file_state", side_effect=[{"bytes": 7}, {"bytes": 8}]):
                with self.assertRaises(RuntimeError):
                    compare.source_receipt(source)

    def test_pts_mismatch_stops_before_decoder(self):
        base = {"stream": {"width": 2, "height": 2, "time_base": "1/1000"}, "pts": [0, 33]}
        changed = {**base, "pts": [0, 34]}
        with patch.object(compare, "source_receipt", return_value={"sha256": "same"}), \
                patch.object(compare, "probe", side_effect=[base, changed]), \
                patch.object(compare, "MaskDecoder") as decoder:
            with self.assertRaises(ValueError):
                compare.compare_sources(Path("a"), Path("b"))
            decoder.assert_not_called()

    def test_mutation_after_decode_refuses_stable_conclusion(self):
        metadata = {"stream": {"width": 2, "height": 2, "time_base": "1/1000"}, "pts": [0]}
        first = {"sha256": "first"}
        changed = {"sha256": "changed"}
        decoders = [Mock(), Mock()]
        for decoder in decoders:
            decoder.frame.side_effect = [np.zeros((2, 2), dtype=np.uint8), None]
            decoder.command = ["fixture-decoder"]
        with patch.object(compare, "source_receipt", side_effect=[first, first, first, changed]), \
                patch.object(compare, "probe", return_value=metadata), \
                patch.object(compare, "MaskDecoder", side_effect=decoders):
            with self.assertRaises(RuntimeError):
                compare.compare_sources(Path("a"), Path("b"))
        for decoder in decoders:
            decoder.close.assert_called_once()

    def test_second_decoder_start_failure_closes_first(self):
        metadata = {"stream": {"width": 2, "height": 2, "time_base": "1/1000"}, "pts": [0]}
        decoder = Mock()
        with patch.object(compare, "source_receipt", return_value={"sha256": "same"}), \
                patch.object(compare, "probe", return_value=metadata), \
                patch.object(compare, "MaskDecoder", side_effect=[decoder, OSError("fixture spawn failure")]):
            with self.assertRaises(OSError):
                compare.compare_sources(Path("a"), Path("b"))
        decoder.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

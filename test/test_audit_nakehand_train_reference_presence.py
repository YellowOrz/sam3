from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_utils

from scripts import audit_nakehand_train_reference_presence as audit


def fixture():
    images = [{"id": index, "recording_id": "ego/a", "source_frame_index": index} for index in range(10)]
    annotations = {index: {category: {} for category in (() if index < 3 else ((1,) if index < 6 else (1, 2)))}
                   for index in range(10)}
    return images, annotations


class TrainReferencePresenceTests(unittest.TestCase):
    def test_four_modes_and_invalid_category(self):
        self.assertEqual([audit.presence_mode({key: {} for key in keys})
                          for keys in ((), (1,), (2,), (1, 2))], list(audit.MODES))
        with self.assertRaises(ValueError):
            audit.presence_mode({3: {}})

    def test_two_queries_per_empty_and_one_per_single(self):
        images, annotations = fixture()
        result = audit.summarize_presence(images, annotations, [0, 1, 3, 8])
        self.assertEqual(result["reference_absent_training_queries"], 5)
        self.assertEqual(result["total"]["reference_empty"]["frames"], 3)
        self.assertEqual(result["total"]["both_references"]["frames"], 4)

    def test_repeated_consumption_count_not_deduplicated(self):
        images, annotations = fixture()
        result = audit.summarize_presence(images, annotations, [0, 0, 3])
        self.assertEqual(result["consumed_training_samples"], 3)
        self.assertEqual(result["consumed_unique_images"], 2)
        self.assertEqual(result["reference_absent_training_queries"], 5)

    def test_consumed_identity_must_be_train(self):
        images, annotations = fixture()
        with self.assertRaises(ValueError):
            audit.summarize_presence(images, annotations, [999])

    def test_first_middle_last_selection_is_deterministic(self):
        images, annotations = fixture()
        result = audit.select_samples(images, annotations)
        self.assertEqual([row["image_id"] for row in result], [3, 4, 5, 0, 1, 2])
        self.assertEqual(result, audit.select_samples(images[::-1], annotations))
        self.assertNotIn("both_references", {row["reference_mode"] for row in result})

    def test_singleton_group_deduplicates_selection_roles(self):
        images, _ = fixture()
        result = audit.select_samples(images[:1], {0: {}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["selection_roles"], ["first", "middle", "last"])

    def test_sampling_cap_rejects_not_truncates(self):
        images, annotations = fixture()
        with self.assertRaises(ValueError):
            audit.select_samples(images, annotations, maximum=2)

    def test_spans_preserve_gaps_and_exposure(self):
        images, _ = fixture()
        result = audit.runs([images[0], images[1], images[4]], Counter([0, 4, 4]))
        self.assertEqual([(row["first_frame"], row["last_frame"]) for row in result], [(0, 1), (4, 4)])
        self.assertEqual([row["consumed_training_samples"] for row in result], [1, 2])

    def test_zero_reference_has_no_annotation_but_is_still_audited(self):
        empty = np.zeros((4, 5), np.uint8)
        result = audit.compare_reference(empty, empty, None, empty.shape)
        self.assertEqual(result["reference_pixels"], 0)
        bad = empty.copy()
        bad[0, 0] = 2
        with self.assertRaises(ValueError):
            audit.compare_reference(bad, empty, None, empty.shape)

    def test_raw_instance_union_matches_binary_and_coco(self):
        raw = np.zeros((4, 5), np.uint8)
        raw[1, 1], raw[2, 2] = 1, 2
        binary = (raw > 0).astype(np.uint8) * 255
        rle = mask_utils.encode(np.asfortranarray((raw > 0).astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        annotation = {"segmentation": rle, "area": 2}
        result = audit.compare_reference(raw, binary, annotation, raw.shape)
        self.assertEqual(result["raw_instance_values"], [0, 1, 2])
        self.assertEqual(result["binary_vs_rle_mismatch_pixels"], 0)
        annotation["area"] = 3
        with self.assertRaises(ValueError):
            audit.compare_reference(raw, binary, annotation, raw.shape)

    def test_binary_value_or_shape_corruption_rejected(self):
        raw = np.zeros((4, 5), np.uint8)
        binary = raw.copy()
        binary[0, 0] = 1
        with self.assertRaises(ValueError):
            audit.compare_reference(raw, binary, None, raw.shape)
        with self.assertRaises(ValueError):
            audit.compare_reference(raw, raw[:2], None, raw.shape)

    def test_path_traversal_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(audit.inside(root, "images/1.png"), root / "images/1.png")
            for relative in ("../bad", "/etc/passwd", ""):
                with self.assertRaises(ValueError):
                    audit.inside(root, relative)
            with self.assertRaises(ValueError):
                audit.audit(root, (), root)


if __name__ == "__main__":
    unittest.main()

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_mano_sidecar import (
    SourceSnapshots,
    hand_target,
    prepare_sidecars,
)


def valid_record(frame=0):
    return {
        "frame": frame, "side": "left", "hand_id": "subject-01_left",
        "track_id": "h2" if frame == 0 else None,
        "global_orient": [0.1, 0.2, 0.3], "hand_pose": [0.01] * 45,
        "betas": [0.2] * 10, "transl": [0.01, 0.02, 1.0],
        "root_frame": "camera", "source": "gt",
    }


def valid_document():
    return {
        "view_id": "camera-01", "root_frame": "camera",
        "pose_representation": "axis-angle",
        "pca_conversion": {
            "file": "MANO_LEFT.pkl", "sha256": "a" * 64, "flat_hand_mean": False,
        },
        "records": [valid_record(0), valid_record(2)],
    }


class PrepareManoSidecarTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.unified = self.root / "unified"
        self.data = self.root / "derived"
        self.output = self.root / "sidecar"
        self.sequence_path = self.unified / "sequences/dexycb/sequence-01/sequence.json"
        self.mano_path = self.sequence_path.parent / "camera-01/mano.json"
        self.annotation_path = self.data / "val/annotations.json"
        self.sequence = {
            "source": "dexycb", "seq_id": "sequence-01", "subject": "subject-01",
            "views": ["camera-01"], "num_frames": 4,
            "extra": {"mano_sides": ["left"]},
            "source_paths": {"mano_calib": "subject-01_right"},
        }
        self.document = valid_document()
        self.coco = {
            "categories": [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}],
            "images": [
                {
                    "id": image_id, "source": "dexycb", "sequence": "sequence-01",
                    "view": "camera-01", "frame_index": frame,
                    "file_name": f"images/dexycb__sequence-01__camera-01__{frame:08d}.jpg",
                }
                for image_id, frame in ((30, 2), (10, 0), (40, 3), (20, 1))
            ],
            "annotations": [
                {"id": 0, "image_id": 10, "category_id": 1, "track_id": "legacy-h0"},
                {"id": 1, "image_id": 20, "category_id": 1, "track_id": "legacy-h0"},
            ],
        }
        self.write(self.sequence_path, self.sequence)
        self.write(self.mano_path, self.document)
        self.write(self.annotation_path, self.coco)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def run_export(self, **kwargs):
        return prepare_sidecars(self.data, self.unified, self.output, splits=("val",), **kwargs)

    def test_frame_identity_four_way_validity_current_tracks_and_hashes(self):
        summary = self.run_export()
        rows = [json.loads(line) for line in (self.output / "val.mano.jsonl").read_text().splitlines()]
        self.assertEqual([row["image_id"] for row in rows], [10, 20, 30, 40])
        self.assertEqual([row["source_frame"] for row in rows], [0, 1, 2, 3])
        self.assertEqual([row["hands"][0]["valid"] for row in rows], [True, False, True, False])
        counts = summary["splits"]["val"]["counts"]
        for key in ("visible_mask_valid_mano", "visible_mask_invalid_mano",
                    "no_mask_valid_mano", "no_mask_invalid_mano"):
            self.assertEqual(counts[key], 1, key)
        self.assertEqual(rows[0]["hands"][0]["track_id"], "h2")
        self.assertEqual(rows[0]["segmentation_track_ids"], ["legacy-h0"])
        self.assertIsNone(rows[2]["hands"][0]["track_id"])
        self.assertEqual(rows[2]["hands"][0]["hand_id"], "subject-01_left")
        self.assertEqual(rows[0]["hands"][0]["physical_hand_id"], rows[2]["hands"][0]["physical_hand_id"])
        self.assertIsNone(rows[1]["hands"][0]["hand_pose"])
        self.assertEqual(rows[0]["source_files"]["mano"]["sha256"],
                         hashlib.sha256(self.mano_path.read_bytes()).hexdigest())
        self.assertTrue(summary["sources_rechecked_unchanged"])
        self.assertEqual(summary["splits"]["val"]["sidecar_sha256"],
                         hashlib.sha256((self.output / "val.mano.jsonl").read_bytes()).hexdigest())

    def test_missing_file_stays_invalid_with_null_arrays(self):
        self.mano_path.unlink()
        summary = self.run_export()
        rows = [json.loads(line) for line in (self.output / "val.mano.jsonl").read_text().splitlines()]
        self.assertEqual(summary["splits"]["val"]["counts"]["invalid_mano"], 4)
        self.assertEqual(summary["splits"]["val"]["counts"]["visible_mask_invalid_mano"], 2)
        for row in rows:
            target = row["hands"][0]
            self.assertFalse(target["valid"])
            self.assertIn("missing_mano_file", target["invalid_reasons"])
            for field in ("global_orient", "hand_pose", "betas", "transl"):
                self.assertIsNone(target[field])
            self.assertFalse(row["source_files"]["mano"]["exists"])

    def test_axis_angle_pca_provenance_and_finite_vectors_are_required(self):
        cases = [
            ("pose_representation_not_axis_angle", lambda doc, rec: doc.pop("pose_representation")),
            ("pose_representation_not_axis_angle", lambda doc, rec: doc.update(pose_representation="pca")),
            ("missing_pca_provenance", lambda doc, rec: doc.pop("pca_conversion")),
            ("pca_model_side_mismatch", lambda doc, rec: doc["pca_conversion"].update(file="MANO_RIGHT.pkl")),
            ("invalid_pca_model_sha256", lambda doc, rec: doc["pca_conversion"].update(sha256="bad")),
            ("pca_flat_hand_mean_must_be_false", lambda doc, rec: doc["pca_conversion"].update(flat_hand_mean=True)),
            ("mano_side_differs_from_sequence", lambda doc, rec: rec.update(side="right")),
            ("root_frame_requires_camera_conversion", lambda doc, rec: (doc.update(root_frame="world"), rec.update(root_frame="world"))),
            ("record_root_frame_mismatch", lambda doc, rec: rec.update(root_frame="world")),
            ("invalid_global_orient", lambda doc, rec: rec.update(global_orient=[0, 0])),
            ("invalid_hand_pose", lambda doc, rec: rec.update(hand_pose=[float("nan")] * 45)),
            ("invalid_betas", lambda doc, rec: rec.update(betas=[True] * 10)),
            ("invalid_transl", lambda doc, rec: rec.update(transl=[float("inf"), 0, 1])),
            ("source_marked_invalid", lambda doc, rec: rec.update(valid=False)),
        ]
        for reason, mutate in cases:
            with self.subTest(reason=reason):
                document = copy.deepcopy(self.document)
                record = document["records"][0]
                mutate(document, record)
                target = hand_target(document, record, "left", self.sequence)
                self.assertFalse(target["valid"])
                self.assertIn(reason, target["invalid_reasons"])
                for field in ("global_orient", "hand_pose", "betas", "transl"):
                    self.assertIsNone(target[field])

    def test_missing_side_authority_is_fatal_and_leaves_no_artifact(self):
        self.sequence["extra"] = {}
        self.write(self.sequence_path, self.sequence)
        with self.assertRaisesRegex(ValueError, "authoritative"):
            self.run_export()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_coco_side_contradiction_is_fatal(self):
        self.coco["annotations"][0]["category_id"] = 2
        self.write(self.annotation_path, self.coco)
        with self.assertRaisesRegex(ValueError, "contradicts sequence side"):
            self.run_export()

    def test_duplicate_mano_frame_is_not_silently_selected(self):
        self.document["records"].append(valid_record(0))
        self.write(self.mano_path, self.document)
        with self.assertRaisesRegex(ValueError, "Duplicate physical-hand"):
            self.run_export()

    def test_explicit_nonidentity_mapping_is_rejected(self):
        self.sequence["frame_map"] = [10, 20, 30, 40]
        self.write(self.sequence_path, self.sequence)
        with self.assertRaisesRegex(ValueError, "frame_map"):
            self.run_export()

    def test_snapshots_detect_source_changes_and_newly_created_missing_file(self):
        snapshots = SourceSnapshots()
        snapshots.read(self.sequence_path)
        self.sequence["extra"]["mano_sides"] = ["right"]
        self.write(self.sequence_path, self.sequence)
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            snapshots.verify_unchanged()
        missing = self.root / "was-missing.json"
        snapshots = SourceSnapshots()
        self.assertIsNone(snapshots.read(missing, optional=True))
        self.write(missing, {})
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            snapshots.verify_unchanged()

    def test_limit_and_refuse_overwrite_or_shared_output(self):
        summary = self.run_export(limit=1)
        self.assertEqual(summary["splits"]["val"]["available_images"], 4)
        self.assertEqual(summary["splits"]["val"]["counts"]["images"], 1)
        with self.assertRaises(FileExistsError):
            self.run_export()
        with self.assertRaisesRegex(ValueError, "shared unified"):
            prepare_sidecars(self.data, self.unified, self.unified / "new", splits=("val",))


if __name__ == "__main__":
    unittest.main()

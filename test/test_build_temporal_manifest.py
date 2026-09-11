import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_temporal_manifest import build_temporal_manifests


class BuildTemporalManifestTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.data_root = root / "coco"
        self.unified_root = root / "unified"
        self.output_dir = root / "clips"
        self.coco = {
            split: {
                "images": [], "annotations": [],
                "categories": [
                    {"id": 1, "name": "left_hand"},
                    {"id": 2, "name": "right_hand"},
                ],
            }
            for split in ("train", "val", "test")
        }

    def add_group(
        self, split, sequence="sequence-01", subject="subject-01", view="camera-01",
        frames=(0, 1), side="left", negatives=(),
    ):
        sequence_path = (
            self.unified_root / "sequences" / "dexycb" / sequence / "sequence.json"
        )
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": "dexycb", "seq_id": sequence, "subject": subject,
            "views": [view], "num_frames": max(frames) + 1,
            "extra": {"mano_sides": [side]},
        }
        if sequence_path.exists():
            previous = json.loads(sequence_path.read_text())
            metadata["views"] = sorted(set(previous["views"] + [view]))
            metadata["num_frames"] = max(metadata["num_frames"], previous["num_frames"])
        sequence_path.write_text(json.dumps(metadata), encoding="utf-8")
        document = self.coco[split]
        for frame in frames:
            image_id = len(document["images"])
            document["images"].append({
                "id": image_id, "source": "dexycb", "sequence": sequence,
                "view": view, "frame_index": frame,
                "file_name": f"images/dexycb__{sequence}__{view}__{frame:08d}.jpg",
            })
            if frame not in negatives:
                document["annotations"].append({
                    "id": len(document["annotations"]), "image_id": image_id,
                    "category_id": 1 if side == "left" else 2,
                    "track_id": "h0",
                })
        return sequence_path

    def write_coco(self):
        for split, document in self.coco.items():
            destination = self.data_root / split / "annotations.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(document), encoding="utf-8")

    def build(self, **kwargs):
        self.write_coco()
        return build_temporal_manifests(
            self.data_root, self.unified_root, self.output_dir, **kwargs
        )

    def clips(self, split="train"):
        return [
            json.loads(line)
            for line in (self.output_dir / f"{split}.clips.jsonl").read_text().splitlines()
        ]

    def test_preserves_source_order_gaps_negatives_tails_and_identity(self):
        frames = [11, 0, 10, 2, 8, 1, 9, 3, 4, 5, 6, 7, 15, 18, 16]
        self.add_group("train", frames=frames, negatives=(0, 9, 15))
        summary = self.build()
        clips = self.clips()
        self.assertEqual(
            [clip["original_frame_indices"] for clip in clips],
            [list(range(8)), list(range(8, 12)), [15, 16]],
        )
        self.assertEqual({clip["physical_hand_id"] for clip in clips}, {"sequence-01:left"})
        source_images = {image["id"]: image for image in self.coco["train"]["images"]}
        for clip in clips:
            self.assertEqual(clip["prompt_names"], ["left_hand", "right_hand"])
            for frame in clip["frames"]:
                self.assertEqual(frame["file_name"], source_images[frame["image_id"]]["file_name"])
                self.assertEqual(frame["source_frame"], frame["frame_index"])
                self.assertEqual(frame["prompt_annotation_ids"]["right_hand"], [])
                self.assertEqual(frame["prompt_annotation_ids"]["left_hand"], frame["annotation_ids"])
                if frame["frame_index"] in (0, 9, 15):
                    self.assertEqual(frame["annotation_ids"], [])
                    self.assertFalse(frame["segmentation_target_present"])
        counts = summary["splits"]["train"]["counts"]
        self.assertEqual(counts["clips"], 3)
        self.assertEqual(counts["input_frames"], 15)
        self.assertEqual(counts["covered_unique_frames"], 14)
        self.assertEqual(counts["covered_negative_frames"], 3)
        self.assertEqual(counts["frame_gaps"], 2)
        self.assertEqual(counts["missing_frames_inside_gaps"], 4)
        self.assertEqual(counts["short_contiguous_segments"], 1)
        self.assertEqual(summary["splits"]["train"]["omitted_frames"][0]["frame_index"], 18)
        self.assertEqual(
            summary["splits"]["train"]["manifest_sha256"],
            hashlib.sha256((self.output_dir / "train.clips.jsonl").read_bytes()).hexdigest(),
        )

    def test_singleton_tail_is_recorded_as_uncovered_not_padded(self):
        self.add_group("train", frames=list(range(9)))
        counts = self.build()["splits"]["train"]["counts"]
        self.assertEqual([clip["num_frames"] for clip in self.clips()], [8])
        self.assertEqual(counts["short_tail_windows"], 1)
        self.assertEqual(counts["uncovered_frames"], 1)

    def test_overlap_counts_unique_coverage_separately(self):
        self.add_group("train", frames=list(range(9)))
        counts = self.build(stride=4)["splits"]["train"]["counts"]
        self.assertEqual([clip["num_frames"] for clip in self.clips()], [8, 5])
        self.assertEqual(counts["covered_unique_frames"], 9)
        self.assertEqual(counts["clip_frame_occurrences"], 13)
        self.assertEqual(counts["uncovered_frames"], 0)

    def test_completely_negative_view_uses_sequence_side_and_both_prompts(self):
        self.add_group("train", frames=(0, 1), negatives=(0, 1), side="right")
        self.build()
        clip = self.clips()[0]
        self.assertEqual(clip["physical_hand_id"], "sequence-01:right")
        self.assertEqual(clip["side_authority"], "sequence.extra.mano_sides[0]")
        for frame in clip["frames"]:
            self.assertEqual(frame["prompt_annotation_ids"], {"left_hand": [], "right_hand": []})

    def test_views_never_merge_but_physical_identity_remains_shared(self):
        self.add_group("train", view="camera-01", frames=(0, 1))
        self.add_group("train", view="camera-02", frames=(2, 3))
        self.build()
        clips = self.clips()
        self.assertEqual([clip["original_frame_indices"] for clip in clips], [[0, 1], [2, 3]])
        self.assertEqual({clip["view"] for clip in clips}, {"camera-01", "camera-02"})
        self.assertEqual({clip["physical_hand_id"] for clip in clips}, {"sequence-01:left"})

    def test_rejects_subject_leakage_across_sequences_and_splits(self):
        self.add_group("train", sequence="sequence-01", subject="subject-01")
        self.add_group("val", sequence="sequence-02", subject="subject-01")
        with self.assertRaisesRegex(ValueError, "Subject leaks"):
            self.build()
        self.assertFalse(self.output_dir.exists())

    def test_rejects_duplicate_source_frame_with_distinct_image_ids(self):
        self.add_group("train", frames=(0, 0, 1))
        with self.assertRaisesRegex(ValueError, "Duplicate source frame"):
            self.build()

    def test_rejects_annotation_side_disagreement(self):
        self.add_group("train", side="left")
        self.coco["train"]["annotations"][0]["category_id"] = 2
        with self.assertRaisesRegex(ValueError, "authoritative side"):
            self.build()

    def test_missing_side_is_rejected_even_when_masks_are_negative(self):
        sequence_path = self.add_group("train", negatives=(0, 1))
        sequence = json.loads(sequence_path.read_text())
        sequence["extra"] = {}
        sequence_path.write_text(json.dumps(sequence))
        with self.assertRaisesRegex(ValueError, "mano_sides"):
            self.build()

    def test_non_identity_frame_map_is_not_silently_assumed_contiguous(self):
        sequence_path = self.add_group("train")
        sequence = json.loads(sequence_path.read_text())
        sequence["frame_map"] = {"0": 10, "1": 20}
        sequence_path.write_text(json.dumps(sequence))
        with self.assertRaisesRegex(ValueError, "frame_map"):
            self.build()

    def test_existing_output_is_never_overwritten(self):
        self.add_group("train")
        self.build()
        path = self.output_dir / "train.clips.jsonl"
        expected = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(path.read_bytes(), expected)

    def test_stride_cannot_silently_skip_frames(self):
        with self.assertRaisesRegex(ValueError, "stride"):
            self.build(clip_length=8, stride=9)


if __name__ == "__main__":
    unittest.main()

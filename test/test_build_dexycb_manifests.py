import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_dexycb_manifests import build_manifests, expand_frame_map


class BuildDexycbManifestsTest(unittest.TestCase):
    def test_source_mano_side_overrides_legacy_hardcoded_right_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "metadata" / "split.json").write_text(
                json.dumps(
                    {"dexycb": {"by_subject": {"subject-01": "train"}}}
                )
            )
            sequence_dir = root / "sequences" / "dexycb" / "sequence-01"
            view_dir = sequence_dir / "camera-01"
            view_dir.mkdir(parents=True)
            (sequence_dir / "sequence.json").write_text(
                json.dumps(
                    {
                        "seq_id": "sequence-01",
                        "subject": "subject-01",
                        "views": ["camera-01"],
                        "num_frames": 3,
                        "extra": {"mano_sides": ["left"]},
                    }
                )
            )
            (view_dir / "instances.json").write_text(
                json.dumps(
                    {
                        "instances": [
                            {
                                "track_id": "h0",
                                "kind": "hand_right",
                                "frame_map": [{"frames": [1, 2], "id": 3}],
                            }
                        ]
                    }
                )
            )

            manifests, excluded = build_manifests(root)

            self.assertEqual(excluded, [])
            record = manifests["train"][0]
            self.assertEqual(record["hand_side"], "left")
            self.assertEqual(record["hand_kind"], "hand_left")
            self.assertEqual(record["source_hand_kind"], "hand_right")
            self.assertEqual(record["hand_visible_frames"], [1, 2])

    def test_overlapping_hand_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            expand_frame_map(
                {
                    "track_id": "h0",
                    "frame_map": [
                        {"frames": [0, 1], "id": 1},
                        {"frames": [1, 2], "id": 2},
                    ],
                },
                num_frames=3,
            )

    def test_hand_empty_view_is_retained_for_negative_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "metadata" / "split.json").write_text(
                json.dumps(
                    {"dexycb": {"by_subject": {"subject-01": "train"}}}
                )
            )
            sequence_dir = root / "sequences" / "dexycb" / "sequence-01"
            view_dir = sequence_dir / "camera-01"
            view_dir.mkdir(parents=True)
            (sequence_dir / "sequence.json").write_text(
                json.dumps(
                    {
                        "seq_id": "sequence-01",
                        "subject": "subject-01",
                        "views": ["camera-01"],
                        "num_frames": 3,
                        "extra": {"mano_sides": ["right"]},
                    }
                )
            )
            (view_dir / "instances.json").write_text(
                json.dumps({"instances": [{"kind": "object"}]})
            )

            manifests, excluded = build_manifests(root)

            self.assertEqual(len(manifests["train"]), 1)
            self.assertEqual(manifests["train"][0]["hand_visible_frames"], [])
            self.assertEqual(excluded[0]["reason"], "no_hand_instance")


if __name__ == "__main__":
    unittest.main()

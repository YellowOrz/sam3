import unittest

import numpy as np
from pycocotools import mask as mask_utils

from scripts.unified_to_sam3 import build_frame_annotations


class UnifiedToSam3Test(unittest.TestCase):
    def test_builds_one_annotation_per_matching_instance(self):
        instance_mask = np.array(
            [
                [0, 0, 0, 0, 6],
                [0, 5, 5, 0, 0],
                [0, 5, 5, 0, 0],
            ],
            dtype=np.uint16,
        )
        instances = [
            {
                "track_id": "o0",
                "kind": "object",
                "frame_map": [{"frames": [0, 2], "id": 6}],
            },
            {
                "track_id": "h0",
                "kind": "hand_right",
                "frame_map": [{"frames": [1, 2], "id": 5}],
            },
            {
                "track_id": "h1",
                "kind": "hand_right",
                "frame_map": [{"frames": [1, 1], "id": 6}],
            },
            {
                "track_id": "h2",
                "kind": "hand_left",
                "frame_map": [{"frames": [0, 2], "id": 7}],
            },
        ]

        annotations = build_frame_annotations(
            instance_mask,
            frame_index=1,
            instances=instances,
            category_ids={"right_hand": 1},
            image_id=12,
        )

        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0]["image_id"], 12)
        self.assertEqual(annotations[0]["category_id"], 1)
        self.assertEqual(annotations[0]["bbox"], [1, 1, 2, 2])
        self.assertEqual(annotations[0]["area"], 4)
        self.assertEqual(annotations[0]["track_id"], "h0")
        decoded = mask_utils.decode(annotations[0]["segmentation"])
        np.testing.assert_array_equal(decoded, instance_mask == 5)

        self.assertEqual(annotations[1]["bbox"], [4, 0, 1, 1])
        self.assertEqual(annotations[1]["area"], 1)
        self.assertEqual(annotations[1]["track_id"], "h1")

    def test_frame_map_end_is_inclusive(self):
        instance_mask = np.array([[5]], dtype=np.uint16)
        instances = [
            {
                "track_id": "h0",
                "kind": "hand_right",
                "frame_map": [{"frames": [3, 3], "id": 5}],
            }
        ]

        self.assertEqual(
            len(
                build_frame_annotations(
                    instance_mask,
                    frame_index=3,
                    instances=instances,
                    category_ids={"right_hand": 1},
                    image_id=0,
                )
            ),
            1,
        )
        self.assertEqual(
            build_frame_annotations(
                instance_mask,
                frame_index=4,
                instances=instances,
                category_ids={"right_hand": 1},
                image_id=0,
            ),
            [],
        )

    def test_unified_hand_kind_maps_to_learnable_token_kind(self):
        instance_mask = np.array([[5]], dtype=np.uint16)
        instances = [
            {
                "track_id": "h0",
                "kind": "hand_right",
                "frame_map": [{"frames": [0, 0], "id": 5}],
            }
        ]

        annotations = build_frame_annotations(
            instance_mask,
            frame_index=0,
            instances=instances,
            category_ids={"right_hand": 1},
            image_id=0,
        )

        self.assertEqual(annotations[0]["kind"], "right_hand")
        self.assertEqual(annotations[0]["category_id"], 1)


if __name__ == "__main__":
    unittest.main()

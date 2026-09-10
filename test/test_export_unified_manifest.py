import unittest

from scripts.export_unified_manifest import CATEGORY_IDS, image_file_name


class ExportUnifiedManifestTest(unittest.TestCase):
    def test_bilateral_category_ids_follow_token_encoder_order(self):
        self.assertEqual(CATEGORY_IDS, {"left_hand": 1, "right_hand": 2})

    def test_image_name_contains_view_and_frame_to_avoid_collisions(self):
        record = {
            "source": "dexycb",
            "sequence": "subject-01_20200709_145132",
            "view": "836212060125",
        }

        self.assertEqual(
            image_file_name(record, 3),
            "images/dexycb__subject-01_20200709_145132__"
            "836212060125__00000003.jpg",
        )


if __name__ == "__main__":
    unittest.main()

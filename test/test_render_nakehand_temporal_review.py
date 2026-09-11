from pathlib import Path
import unittest
from scripts import render_nakehand_temporal_review as review


class TemporalReviewTest(unittest.TestCase):
    def test_full_rgb_is_complete_not_a_selected_clip_and_refuses_overwrite(self):
        command = review.transcode_command(Path("/tmp/in.mkv"), Path("/tmp/new.mp4"))
        self.assertNotIn("-vf", command)
        self.assertNotIn("-t", command)
        self.assertNotIn("-frames:v", command)
        self.assertIn("-n", command)
        self.assertEqual(command[command.index("-r") + 1], "30")
        self.assertEqual(command[command.index("-preset") + 1], "veryfast")

    def test_mask_display_is_nonzero_not_instance_one_only(self):
        command = review.transcode_command(Path("/tmp/mask.mkv"), Path("/tmp/new.mp4"), mask=True)
        filters = command[command.index("-vf") + 1]
        self.assertIn("if(gt(val,0),255,0)", filters)
        self.assertNotIn("eq(val,1)", filters)

    def test_slow_preserves_every_first300_frame_with_source_frame_number(self):
        command = review.transcode_command(Path("/tmp/rgb.mkv"), Path("/tmp/new.mp4"), slow=True)
        filters = command[command.index("-vf") + 1]
        self.assertIn("trim=start_frame=0:end_frame=300", filters)
        self.assertIn("setpts=3*(PTS-STARTPTS)", filters)
        self.assertIn("SOURCE frame %{n}", filters)
        self.assertIn("side unconfirmed", filters)
        self.assertEqual(command[command.index("-r") + 1], "10")

    def test_still_selection_frozen_unique_with_both_sides_of_18_transition(self):
        self.assertEqual(tuple(sorted(set(review.STILLS))), review.STILLS)
        self.assertTrue({0, 17, 18, 19, 20, 299}.issubset(review.STILLS))
        self.assertLess(max(review.STILLS), review.FRAME_COUNT)


if __name__ == "__main__":
    unittest.main()

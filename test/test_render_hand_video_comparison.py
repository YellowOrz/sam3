import unittest
import numpy as np
from scripts.render_hand_video_comparison import compose_frame, CELL, HEADER, ROW


class RenderHandVideoTest(unittest.TestCase):
    def test_reference_and_predictions_are_separate_not_overlaid(self):
        rgb = np.zeros((12,16,3),np.uint8);rgb[...,0]=200
        on = np.ones((12,16),bool);off = np.zeros_like(on)
        image = compose_frame(rgb,{'left_hand':on,'right_hand':None},
            [('ve',{'left_hand':off,'right_hand':on}),('spatial',{'left_hand':on,'right_hand':off})], 'test')
        self.assertEqual(image.size,(CELL[0]*4,HEADER+2*ROW))
        a=np.asarray(image);y=HEADER+36+10
        self.assertTrue(np.array_equal(a[y,10],[200,0,0]))
        self.assertTrue(np.all(a[y,CELL[0]+10]==255))
        self.assertTrue(np.all(a[y,CELL[0]*2+10]==0))
        self.assertTrue(np.all(a[y,CELL[0]*3+10]==255))
        self.assertTrue(np.all(a[y+ROW,CELL[0]+10]==110))


if __name__ == '__main__': unittest.main()

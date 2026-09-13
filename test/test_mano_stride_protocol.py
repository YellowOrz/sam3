import copy
import unittest
from pathlib import Path

import numpy as np

from scripts import mano_stride_protocol as p
from scripts.evaluate_mano_stride import actual_output, metrics, encode, code_hashes
from pycocotools import mask as mask_utils


class StrideProtocolTests(unittest.TestCase):
    def test_source_hashes_accept_relative_root_and_keep_portable_keys(self):
        hashes=code_hashes('.')
        self.assertIn('scripts/evaluate_mano_stride.py',hashes)
        self.assertTrue(all(not Path(k).is_absolute() and len(v)==64 for k,v in hashes.items()))

    def test_selection_keeps_zero_empty_unknown_and_recording_end(self):
        images=[dict(id=i+1,recording_id='a',source_frame_index=i,has_hand=False) for i in range(10)]
        selected,render=p.select_images(images)
        self.assertEqual([r['source_frame_index'] for r in selected],[0,3,6,9])
        self.assertEqual(render,[1,4,10])
        with self.assertRaises(ValueError):p.select_images(images[1:])
        with self.assertRaises(ValueError):p.select_images(images+[images[0]])

    def test_two_recordings_reset_offset(self):
        images=[dict(id=j*10+i+1,recording_id=str(j),source_frame_index=i) for j in range(2) for i in range(5)]
        self.assertEqual([(r['recording_id'],r['source_frame_index']) for r in p.select_images(images)[0]],
                         [('0',0),('0',3),('1',0),('1',3)])

    def test_no_points_does_not_invalidate_box(self):
        joints=np.tile([100.,100.,1.],(21,1))
        vertices=np.zeros((778,3));vertices[1,:2]=[.005,.005]
        g=p.projected_geometry(joints,vertices,np.array([0.,0.,1.]),640,480)
        self.assertIsNone(g['points']);self.assertIsNotNone(g['mesh_box'])
        self.assertEqual(g['point_reason'],'no_projected_joint_in_image')
        g['reference_box']=[.5,.5,.1,.1]
        self.assertFalse(p.method_condition('text_mesh_box',g)['fallback'])
        self.assertTrue(p.method_condition('text_points',g)['fallback'])
        self.assertFalse(p.method_condition('visual_points',g)['evaluated'])

    def test_projection_allows_points_without_valid_box(self):
        g=p.projected_geometry(np.zeros((21,3)),np.zeros((778,3)),np.array([0.,0.,1.]),640,480)
        self.assertEqual(len(g['points']),21);self.assertIsNone(g['mesh_box'])
        np.testing.assert_allclose(g['points'][0],[(320+.5)/640,(240+.5)/480])

    def test_unavailable_fallback_only_core(self):
        g=dict(points=None,mesh_box=None,reference_box=None,point_reason='no_npz',box_reason='no_npz')
        self.assertEqual(p.method_condition('text',g),dict(evaluated=True,fallback=False,reasons=[]))
        self.assertEqual(p.method_condition('text_points_mesh_box',g),dict(evaluated=True,fallback=True,reasons=['no_npz']))
        self.assertFalse(p.method_condition('wrong_text_points_mesh_box',g)['evaluated'])
        self.assertFalse(p.method_condition('text_reference_box',g)['evaluated'])

    def test_text_targets_anatomical_side_and_wrong_text_is_opposite(self):
        self.assertEqual(p.text_for('text_points','right_hand'),'right hand')
        self.assertEqual(p.text_for('wrong_text_points_mesh_box','right_hand'),'left hand')
        self.assertEqual(p.text_for('visual_points','left_hand'),'visual')
        with self.assertRaises(ValueError):p.text_for('text','screen_left')

    def test_strict_threshold_and_unknown_not_negative(self):
        candidate=np.ones((12,12),bool)
        self.assertFalse(actual_output(candidate,.5).any())
        self.assertTrue(actual_output(candidate,.500001).any())
        for score in (np.nan,np.inf,-.1,1.1):
            with self.assertRaises(ValueError):actual_output(candidate,score)
        row=metrics(candidate,.9,None,None)
        self.assertIsNone(row['false_positive']);self.assertIsNone(row['dice'])
        self.assertIsNone(row['all_provided_dice']);self.assertIsNone(row['reference_pixels'])

    def test_empty_reference_positive_miss_and_codec(self):
        full=np.ones((12,12),bool);empty=np.zeros_like(full)
        row=metrics(full,.5,full,empty)
        self.assertEqual(row['dice'],0);self.assertTrue(row['false_negative'])
        self.assertEqual(row['candidate_dice'],1)
        row=metrics(full,.5,empty,None)
        self.assertIsNone(row['dice']);self.assertEqual(row['all_provided_dice'],1)
        self.assertFalse(row['false_positive'])
        self.assertTrue(metrics(full,.9,empty,None)['false_positive'])
        np.testing.assert_array_equal(mask_utils.decode(encode(full)),full)

    def test_modern_contract_does_not_accept_wrong_side_frame_or_legacy(self):
        a=dict(hand=np.array('left_hand'),total_frames=np.array(2),width=np.array(640),height=np.array(480),
               frame_indices=np.arange(2),mask_source=np.array('left_hand'),instance_label=np.array(1),
               has_hand=np.zeros(2,dtype=bool),joints=np.zeros((2,21,3)),vertices=np.zeros((2,778,3)),
               camera_translation=np.zeros((2,3)),bbox_xyxy=np.zeros((2,4)))
        p.validate_arrays(a,'left_hand',2,640,480,True)
        for key,value in [('hand',np.array('right_hand')),('frame_indices',np.array([1,2])),
                          ('mask_source',np.array('right_hand')),('has_hand',np.zeros(2,dtype=int)),
                          ('camera_intrinsics',np.eye(3))]:
            b=copy.deepcopy(a);b[key]=value
            with self.assertRaises(ValueError):p.validate_arrays(b,'left_hand',2,640,480,True)
        del a['has_hand']
        with self.assertRaises(KeyError):p.validate_arrays(a,'left_hand',2,640,480,True)
        p.validate_arrays(a,'left_hand',2,640,480,False)

    def test_area_bins_do_not_claim_true_occlusion(self):
        self.assertEqual([p.area_group(n,100,100) for n in (None,0,49,50,199,200)],
                         ['unknown','empty','small_lt_0.5pct','medium_lt_2pct','medium_lt_2pct','large_ge_2pct'])


if __name__=='__main__':unittest.main()

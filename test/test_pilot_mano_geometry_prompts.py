import unittest
import numpy as np
from scripts.pilot_mano_geometry_prompts import (project, normalized_box, prompt_coordinates,
    mask_metrics, geometry_prompt)


class ManoPromptPilotTests(unittest.TestCase):
    def test_projection_applies_no_second_left_mirror(self):
        points=np.array([[.01,.02,0],[-.01,-.02,0]])
        np.testing.assert_allclose(project(points,[0,0,1],1000,640,480),[[330,260],[310,220]])

    def test_projection_rejects_bad_depth_and_values(self):
        for points in (np.zeros((1,3)),np.full((1,3),np.nan)):
            with self.assertRaises(ValueError): project(points,[0,0,0],1000,640,480)

    def test_normalized_box_clips_without_expansion(self):
        np.testing.assert_allclose(normalized_box([-10,10,80,60],100,100),[.4,.35,.8,.5])
        with self.assertRaises(ValueError): normalized_box([5,5,1,1],100,100)

    def test_points_drop_only_out_of_image_not_reference(self):
        joints=np.tile([10.,20.],(21,1));joints[0]=[-1,20]
        vertices=np.tile([5.,5.],(778,1));vertices[1]=[50,60]
        points,box,ids=prompt_coordinates(joints,vertices,100,100)
        self.assertEqual(len(points),20);self.assertNotIn(0,ids)
        np.testing.assert_allclose(points[0],[.105,.205])
        np.testing.assert_allclose(box,[.275,.325,.45,.55])

    def test_actual_empty_output_zero_and_identical_one(self):
        reference=np.zeros((20,20),bool);reference[5:15,5:15]=True
        self.assertEqual(mask_metrics(reference,reference)['dice'],1)
        row=mask_metrics(np.zeros_like(reference),reference)
        self.assertEqual(row['dice'],0);self.assertEqual(row['boundary_iou_4px'],0)
        self.assertTrue(row['false_negative'])

    def test_prompt_reaches_geometry_shape_with_positive_default(self):
        points=np.full((21,2),.5,np.float32);box=np.array([.5,.5,.2,.2],np.float32)
        prompt=geometry_prompt('text_points_mesh_box',points,box,box,'cpu')
        self.assertEqual(tuple(prompt.point_embeddings.shape),(21,1,2))
        self.assertEqual(tuple(prompt.box_embeddings.shape),(1,1,4))
        self.assertTrue(prompt.point_labels.bool().all())
        self.assertTrue(prompt.box_labels.bool().all())
        self.assertIsNone(geometry_prompt('text',points,box,box,'cpu'))


if __name__=='__main__': unittest.main()

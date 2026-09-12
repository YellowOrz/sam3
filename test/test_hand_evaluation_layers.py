import unittest
import numpy as np
import torch
from torch import nn
from scripts.hand_evaluation_metrics import summarize_outputs, temporal_diagnostics, validate_reference_role
from scripts.video_hand_routes import VideoResidualTextEncoder, union_video_outputs, select_video_indices
from scripts.audit_hand_route_outputs import actual_metrics, thresholded_output


def row(frame=0, ref=5, pred=5, side='left_hand', provided=True, flags=None):
    return dict(source_frame_index=frame, recording_id='synthetic', prompt_key=side,
        reference_provided=provided, reference_quality_flags=flags or [],
        reference_pixels=ref if provided else None, detected_mask_pixels=pred,
        detected=pred>0, miss_zero_dice=.8 if ref and provided and pred else 0 if ref and provided else None,
        miss_zero_boundary_iou_4px=.4 if ref and provided and pred else 0 if ref and provided else None,
        top_dice=.8 if ref and provided else None, output_object_ids=[1] if pred else [])


class HandEvaluationLayersTest(unittest.TestCase):
    def test_audit_preserves_video_outputs_below_raw_score_gate(self):
        a=np.ones((2,2),bool)
        self.assertTrue(thresholded_output(a,.4,True,'video_system').all())
        self.assertFalse(thresholded_output(a,.4,False,'image_ablation').any())
        self.assertTrue(thresholded_output(a,.5,True,'image_ablation').all())

    def test_audit_rejects_invalid_score_and_inconsistent_detection(self):
        a=np.ones((2,2),bool)
        for score in (float('nan'),float('inf'),-.1,1.1,True):
            with self.assertRaises(ValueError): thresholded_output(a,score,True,'image_ablation')
        with self.assertRaises(ValueError): thresholded_output(a,.4,True,'image_ablation')
        with self.assertRaises(ValueError): thresholded_output(a,.4,False,'video_system')

    def test_independent_mask_metrics(self):
        a=np.zeros((12,12),bool);a[2:10,2:10]=True
        self.assertEqual(actual_metrics(a,a),(1.,1.))
        self.assertEqual(actual_metrics(np.zeros_like(a),a),(0.,0.))
        self.assertEqual(actual_metrics(a,None),(None,None))
        self.assertEqual(actual_metrics(a,np.zeros_like(a)),(None,None))

    def test_actual_output_separates_empty_reference_and_misses(self):
        rows=[row(),row(1,pred=0),row(2,ref=0,pred=0),row(3,ref=0),
              row(4,provided=False),row(5,flags=['uncertain'])]
        result=summarize_outputs(rows)
        m=result['primary']['overall']
        self.assertEqual(m['positive_queries'],2)
        self.assertEqual(m['negative_queries'],2)
        self.assertAlmostEqual(m['actual_positive_dice'],.4)
        self.assertAlmostEqual(m['all_provided_dice_empty_empty_one'],.45)
        self.assertEqual(m['false_negative_empty_output'],1)
        self.assertEqual(m['false_positive_nonempty_output'],1)
        self.assertEqual(result['unknown_queries'],1)
        self.assertEqual(result['excluded_flagged_queries'],1)
        self.assertIsNone(result['primary']['per_side']['right_hand']['actual_positive_dice'])

    def test_rejects_nonfinite_positive_metric(self):
        r=row();r['miss_zero_dice']=float('nan')
        with self.assertRaises(ValueError): summarize_outputs([r])

    def test_cannot_relabel_inspected_or_assisted_as_blind(self):
        validate_reference_role('external_development','sam3_assisted',True)
        validate_reference_role('independent_holdout','independent_manual',False)
        for source,inspected in [('sam3_assisted',False),('independent_manual',True)]:
            with self.assertRaises(ValueError): validate_reference_role('independent_holdout',source,inspected)

    def test_temporal_gap_flags_and_proxy_not_identity_accuracy(self):
        rows=[row(0),row(1,pred=0),row(2),row(4)]
        m=temporal_diagnostics(rows)
        self.assertEqual(m['eligible_adjacent_pairs'],2)
        self.assertEqual(m['isolated_one_frame_empty_outputs'],1)
        self.assertIsNone(m['identity_switches'])
        rows[1]['reference_quality_flags']=['uncertain']
        self.assertEqual(temporal_diagnostics(rows)['eligible_adjacent_pairs'],0)
        with self.assertRaises(ValueError): temporal_diagnostics([row(),row()])

    def test_video_union_preserves_all_accepted_instances(self):
        masks=np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=bool)
        union,ids,scores=union_video_outputs(dict(out_obj_ids=[3,7],out_probs=[.4,.9],out_binary_masks=masks),(2,2))
        self.assertEqual(union.sum(),2)
        self.assertEqual(ids,[3,7])  # No second score threshold or top-one selection.
        self.assertEqual(scores,[.4,.9])
        self.assertEqual(union_video_outputs({},(2,2))[0].sum(),0)
        with self.assertRaises(ValueError): union_video_outputs(dict(out_obj_ids=[3],out_probs=[float('nan')],out_binary_masks=masks[:1]),(2,2))
        with self.assertRaises(ValueError): union_video_outputs(dict(out_obj_ids=[3,3],out_probs=[1,1],out_binary_masks=masks),(2,2))

    def test_explicit_contiguous_recordings(self):
        images=[dict(recording_id='a',source_frame_index=i) for i in range(4)]
        self.assertEqual(select_video_indices(images,['a'],2),{'a':[0,1]})
        for names,limit in [(['missing'],None),(['a','a'],None),(['a'],0)]:
            with self.assertRaises(ValueError): select_video_indices(images,names,limit)
        with self.assertRaises(ValueError): select_video_indices(images[1:],['a'])

    def test_video_residual_preserves_visual_and_other_text(self):
        class Fake(nn.Module):
            def __init__(self,value): super().__init__();self.value=value
            def forward(self,captions,input_boxes=None,device=None):
                n=len(captions)
                return (torch.zeros(n,32,dtype=torch.bool),torch.full((32,n,256),self.value),
                        torch.full((32,n,1024),self.value))
        model=VideoResidualTextEncoder(Fake(7.),Fake(9.))
        _,features,raw=model(['right hand','visual','cat','left_hand'])
        self.assertTrue(torch.all(features[:,[0,3]]==9))
        self.assertTrue(torch.all(features[:,[1,2]]==7))
        self.assertTrue(torch.all(raw[:,[1,2]]==7))
        self.assertFalse(model.training)


if __name__=='__main__': unittest.main()

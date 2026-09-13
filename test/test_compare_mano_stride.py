import copy
import unittest

import numpy as np

from scripts import mano_stride_protocol as p
from scripts import compare_mano_stride as c
from scripts.evaluate_mano_stride import metrics


class ManoStrideAuditTests(unittest.TestCase):
    def test_independent_formulas_match_nontrivial_masks(self):
        ref=np.zeros((30,40),bool);ref[5:24,8:32]=True
        mask=np.zeros_like(ref);mask[10:28,3:27]=True
        other=np.flip(ref,0)
        for score in (.499,.5,.501):
            for reference in (ref,np.zeros_like(ref),None):
                c.equal_metrics(metrics(mask,score,reference,other),c.independent_metrics(mask,score,reference,other))

    def test_rejects_forged_missing_empty_and_metric(self):
        mask=np.ones((12,12),bool)
        values=c.independent_metrics(mask,.5,mask,None)
        for key,value in [('dice',1.),('prediction_pixels',144),('false_negative',False),('other_dice',0)]:
            forged=dict(values);forged[key]=value
            with self.assertRaises(ValueError):c.equal_metrics(forged,values)
        for score in (None,float('nan'),True,-1,2):
            with self.assertRaises(ValueError):c.independent_metrics(mask,score,mask,None)

    def test_complete_disjoint_shards_only(self):
        spec=dict(format=p.FORMAT,contract=p.CONTRACT,shard_count=2,selected_image_ids=[1,4,7])
        summaries=[dict(protocol=spec,shard_index=i,status='complete',frozen_parameters_verified=True,
                        assigned_image_ids=spec['selected_image_ids'][i::2],frames=len(spec['selected_image_ids'][i::2]),
                        rows=len(spec['selected_image_ids'][i::2])*18) for i in range(2)]
        self.assertEqual(c.validate_shards(summaries),spec)
        for bad in (summaries[:1],[summaries[0],summaries[0]]):
            with self.assertRaises(ValueError):c.validate_shards(bad)
        for field,value in [('status','running'),('rows',1),('assigned_image_ids',[1]),('frozen_parameters_verified',False)]:
            bad=copy.deepcopy(summaries);bad[1][field]=value
            with self.assertRaises(ValueError):c.validate_shards(bad)

    def test_summary_excludes_unevaluated_and_unknown_from_denominators(self):
        base=dict(evaluated=True,reference_pixels=1,fallback=False,dice=.5,boundary_iou_4px=.1,
                  candidate_dice=.7,recording_id='a',all_provided_dice=.5,false_negative=False,
                  false_positive=False,other_dice=None,prediction_pixels=1)
        rows=[base,dict(base,evaluated=False),dict(base,reference_pixels=None,dice=None,all_provided_dice=None),
              dict(base,reference_pixels=0,dice=None,all_provided_dice=0.,false_positive=True)]
        s=c.summarize(rows)
        self.assertEqual((s['queries'],s['evaluated'],s['positive'],s['empty'],s['unknown']),(4,3,1,1,1))
        self.assertEqual(s['mean_dice'],.5);self.assertEqual(s['false_positive'],1)
        self.assertEqual(s['all_provided_mean_dice_empty_empty_one'],.25)


if __name__=='__main__':unittest.main()

import ast
import io
import json
from pathlib import Path
import types
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval.recondition_consistency import ConsistentReconditioning


SOURCE = Path(__file__).parents[1] / 'sam3/model/sam3_video_base.py'
TREE = ast.parse(SOURCE.read_text())
BASE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'Sam3VideoBase')
METHODS = {n.name: n for n in BASE.body if isinstance(n, ast.FunctionDef)}
ENV = dict(np=np, torch=torch, F=F, logger=types.SimpleNamespace(debug=lambda *a: None),
           fill_holes_in_mask_scores=lambda x, **kw: x)
for name in ('_recondition_masklets', '_tracker_update_memories', 'build_outputs'):
    exec('from __future__ import annotations\n'+ast.unparse(METHODS[name]), ENV)
BODY = METHODS['run_tracker_update_planning_phase'].body
START = next(i for i, n in enumerate(BODY) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'should_recondition_iou' for t in n.targets))
END = next(i for i, n in enumerate(BODY[START:], START)
           if isinstance(n, ast.If) and ast.unparse(n.test) == 'batch_size > 0')
SEGMENT = compile(ast.Module(body=BODY[START:END+1], type_ignores=[]), str(SOURCE), 'exec')


class Tracker:
    input_mask_size = 4
    maskmem_backbone = types.SimpleNamespace(mask_downsampler=types.SimpleNamespace(interpol_size=(4, 4)))

    def __init__(self):
        self.added = {}; self.fail = False; self.memory_input = None

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        if self.fail:
            raise RuntimeError('native add failure')
        self.added[obj_id] = mask.clone()

    def propagate_in_video_preflight(self, state, run_mem_encoder):
        state['output_dict']['cond_frame_outputs'][16] = {'maskmem_features': 'corrected', 'maskmem_pos_enc': []}
        state['output_dict']['non_cond_frame_outputs'].pop(16, None)

    def _suppress_object_pw_area_shrinkage(self, x):
        return x

    def _run_memory_encoder(self, state, frame_idx, batch_size, mask, score, is_mask_from_pts):
        self.memory_input = mask.clone()
        return (mask > 0).clone(), []

    def _add_output_per_object(self, **kw):
        pass


class NativeSubset:
    world_size = 1; rank = 0; is_multiplex = False
    reconstruction_bbox_iou_thresh = 0.
    recondition_every_nth_frame = 16
    suppress_overlapping_based_on_recent_occlusion_threshold = 0.
    fill_hole_area = 0
    _recondition_masklets = ENV['_recondition_masklets']
    _tracker_update_memories = ENV['_tracker_update_memories']
    build_outputs = ENV['build_outputs']

    def __init__(self):
        self.tracker = Tracker()

    def _suppress_overlapping_based_on_recent_occlusion(self, frame_idx, masks, *args):
        masks.fill_(-1)
        return masks

    def run_tracker_update_planning_phase(self, frame_idx, reverse, tracker_low_res_masks_global,
            tracker_metadata_prev, det_out, tracker_states_local, tracker_obj_scores_global, matches):
        # Execute the actual native planning correction/memory block, not a
        # hand-reimplementation of the behavior under test. Neural ops are mocks.
        scope = dict(self=self, frame_idx=frame_idx, reverse=reverse,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_metadata_prev=tracker_metadata_prev, tracker_metadata_new={}, det_out=det_out,
            tracker_states_local=tracker_states_local, tracker_obj_scores_global=tracker_obj_scores_global,
            trk_id_to_max_iou_high_conf_det=matches, reconditioned_obj_ids=set(), obj_ids_newly_removed=set())
        exec(SEGMENT, ENV, scope)
        plan = dict(new_det_fa_inds=np.array([], np.int64), new_det_obj_ids=np.array([], np.int64),
                    reconditioned_obj_ids=scope['reconditioned_obj_ids'], trk_id_to_max_iou_high_conf_det=matches)
        return plan, {}


def inputs(two=False, score=2., dtype=torch.float32):
    ids = [42, 7] if two else [7]
    old = -torch.ones(len(ids), 4, 4, dtype=dtype); old[:, :2, :2] = 1
    new = -torch.ones(1, 2, 2, dtype=dtype); new[:, 1:, 1:] = 1
    state = dict(obj_ids=ids, output_dict=dict(cond_frame_outputs={}, non_cond_frame_outputs={16: {}}))
    return dict(frame_idx=16, reverse=False, tracker_low_res_masks_global=old,
        tracker_metadata_prev={'obj_ids_all_gpu': np.array(ids), 'num_obj_per_gpu': [len(ids)]},
        det_out={'mask': new}, tracker_states_local=[state], tracker_obj_scores_global=torch.full((len(ids),), score), matches={7: 0})


def execute(model, args):
    plan, _ = model.run_tracker_update_planning_phase(**args)
    result = model.build_outputs(args['frame_idx'], 32, False, args['det_out'],
        args['tracker_low_res_masks_global'], args['tracker_obj_scores_global'],
        args['tracker_metadata_prev'], plan, 4, 4, plan['reconditioned_obj_ids'], {})
    return plan, result


class ConsistencyTests(unittest.TestCase):
    def test_legacy_behavior_reproduces_overwrite_without_install(self):
        model = NativeSubset(); args = inputs(); old = args['tracker_low_res_masks_global'].clone()
        plan, result = execute(model, args)
        self.assertEqual(plan['reconditioned_obj_ids'], set())
        self.assertTrue(torch.equal(result[7], old > 0))
        self.assertTrue(torch.equal(model.tracker.memory_input, old[:, None]))

    def test_success_shared_source_memory_output_and_trace(self):
        model = NativeSubset(); args = inputs(); stream = io.StringIO()
        expected = F.interpolate(args['det_out']['mask'][:, None], (4,4), mode='bilinear', align_corners=False)
        with ConsistentReconditioning(model, stream) as policy:
            plan, result = execute(model, args)
            self.assertEqual(plan['reconditioned_obj_ids'], {7})
            self.assertTrue(torch.equal(model.tracker.memory_input, expected))
            self.assertTrue(torch.equal(result[7], expected[0] > 0))
            self.assertEqual(policy.counts['corrected_objects'], 1)
        records = [json.loads(s) for s in stream.getvalue().splitlines()]
        mem = next(r for r in records if r['event']=='memory_input')
        out = next(r for r in records if r['event']=='output_source')
        self.assertEqual(mem['masks'], out['masks'])
        self.assertNotIn('_recondition_consistency_active', model.__dict__)

    def test_native_confidence_rejection_no_false_success(self):
        model = NativeSubset(); args = inputs(score=-2); old=args['tracker_low_res_masks_global'].clone()
        with ConsistentReconditioning(model) as policy:
            plan, out = execute(model, args)
            self.assertEqual(plan['reconditioned_obj_ids'], set())
            self.assertEqual(policy.counts['corrected_objects'], 0)
            self.assertTrue(torch.equal(out[7], old > 0))

    def test_nonperiodic_frame_keeps_propagation(self):
        model=NativeSubset(); args=inputs(); args['frame_idx']=17
        with ConsistentReconditioning(model) as policy:
            execute(model,args)
            self.assertEqual(policy.counts['correction_calls'],0)

    def test_no_cross_object_replacement_and_dtype_preserved(self):
        model=NativeSubset(); args=inputs(two=True,dtype=torch.bfloat16); old=args['tracker_low_res_masks_global'][0].clone()
        with ConsistentReconditioning(model):
            plan,out=execute(model,args)
            self.assertEqual(plan['reconditioned_obj_ids'],{7})
            self.assertTrue(torch.equal(out[42],old[None]>0))
            self.assertEqual(args['tracker_low_res_masks_global'].dtype,torch.bfloat16)

    def test_native_occlusion_is_not_undone_by_raw_detection_override(self):
        model=NativeSubset(); model.suppress_overlapping_based_on_recent_occlusion_threshold=.7
        with ConsistentReconditioning(model):
            plan,out=execute(model,inputs())
            self.assertEqual(plan['reconditioned_obj_ids'],{7})
            self.assertFalse(out[7].any())
            self.assertFalse((model.tracker.memory_input>0).any())

    def test_exception_restores_all_instance_and_tracker_methods(self):
        model=NativeSubset(); model.tracker.fail=True
        before=set(model.__dict__); before_tracker=set(model.tracker.__dict__)
        with self.assertRaisesRegex(RuntimeError,'native add failure'):
            with ConsistentReconditioning(model):execute(model,inputs())
        self.assertEqual(set(model.__dict__),before)
        self.assertEqual(set(model.tracker.__dict__),before_tracker)
        self.assertIs(model.build_outputs.__func__, NativeSubset.build_outputs)

    def test_failed_preflight_does_not_select_correction(self):
        model=NativeSubset(); args=inputs(); old=args['tracker_low_res_masks_global'].clone()
        def fail(*args,**kwargs):raise RuntimeError('preflight failure')
        model.tracker.propagate_in_video_preflight=fail
        with self.assertRaisesRegex(RuntimeError,'preflight failure'):
            with ConsistentReconditioning(model):execute(model,args)
        self.assertTrue(torch.equal(args['tracker_low_res_masks_global'],old))

    def test_reject_multiplex_distributed_reverse_and_nested(self):
        for attr,value in [('world_size',2),('is_multiplex',True)]:
            model=NativeSubset();setattr(model,attr,value)
            with self.assertRaises(ValueError):
                with ConsistentReconditioning(model):pass
            self.assertNotIn('_recondition_consistency_active',model.__dict__)
        model=NativeSubset()
        with ConsistentReconditioning(model):
            with self.assertRaises(ValueError):
                with ConsistentReconditioning(model):pass
            args=inputs();args['reverse']=True
            with self.assertRaises(ValueError):execute(model,args)

    def test_bad_signature_is_rejected_before_any_patch(self):
        model=NativeSubset();model.build_outputs=lambda:None
        before=set(model.__dict__)
        with self.assertRaises(ValueError):
            with ConsistentReconditioning(model):pass
        self.assertEqual(set(model.__dict__),before)


if __name__=='__main__':unittest.main()

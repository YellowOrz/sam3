from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from scripts.senior_video_evaluation import frame_metrics, score_video_rows, require_validated_step
from scripts.compare_senior_video_routes import compare, label


def row(frame=0, side='left_hand', p=4, g=4, i=4, provided=True):
    return dict(image_id=frame, recording_id='recording', source_frame_index=frame,
        prompt_key=side, reference_provided=provided, reference_quality_flags=[],
        detected_mask_pixels=p, reference_pixels=g if provided else None,
        actual_reference_intersection_pixels=i if provided else None,
        miss_zero_dice=2*i/(p+g) if provided and g else None,
        miss_zero_boundary_iou_4px=.5 if provided and g else None,
        top_dice=2*i/(p+g) if provided and g else None, detected=bool(p))


class SeniorVideoEvaluationTest(unittest.TestCase):
    def test_empty_policy_and_frame_not_pixel_weighting(self):
        result = frame_metrics([row(p=0,g=0,i=0), row(p=0,i=0), row(provided=False)])
        self.assertEqual(result['mean_dice'], .5)
        self.assertEqual(result['pixel_dice'], 0.)
        self.assertEqual(result['unknown_reference_queries'], 1)
        self.assertEqual(result['frames_evaluated'], 2)
        self.assertIsNone(frame_metrics([])['mean_dice'])

    def test_stride_only_after_complete_propagation(self):
        rows = [row(i, s) for i in range(7) for s in ('left_hand','right_hand')]
        result = score_video_rows(rows)
        self.assertEqual(result['propagated_queries'], 14)
        self.assertEqual(result['sampled_queries'], 6)
        self.assertEqual(result['raw_provided']['per_side']['left_hand']['frames_evaluated'], 3)
        for bad in (rows[2:], rows + [rows[0]], rows[::2], rows[:3]):
            with self.assertRaises(ValueError): score_video_rows(bad)
        for stride in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): score_video_rows(rows, stride)

    def test_flags_and_missing_are_not_silently_scored_negative(self):
        rows = [row(0), row(0, 'right_hand', provided=False)]
        rows[0]['reference_quality_flags'] = ['uncertain']
        result = score_video_rows(rows)
        self.assertEqual(result['raw_provided']['overall']['mean_dice'], 1.)
        self.assertIsNone(result['primary_nonflagged']['overall']['mean_dice'])
        self.assertEqual(result['actual_output_metrics']['excluded_flagged_queries'], 1)
        self.assertEqual(result['actual_output_metrics']['unknown_queries'], 1)

    def test_counts_and_dice_must_agree(self):
        for key, value in [('actual_reference_intersection_pixels',5),('reference_pixels',True),
                           ('miss_zero_dice',.2),('detected_mask_pixels',-1)]:
            bad = row(); bad[key] = value
            with self.assertRaises(ValueError): frame_metrics([bad])

    def test_validated_partial_step_does_not_fake_epoch(self):
        state = dict(progress=dict(global_step=600, completed_epochs=0),
                     validation_state=dict(last_validation_step=600))
        before = deepcopy(state)
        require_validated_step(state, 600)
        self.assertEqual(state,before)
        for step in (None,True,0,599):
            with self.assertRaises(ValueError): require_validated_step(state,step)
        state['validation_state']['last_validation_step'] = 2
        with self.assertRaises(ValueError): require_validated_step(state,600)

    def test_sides_must_bind_same_images(self):
        rows = [row(), row(side='right_hand')]
        rows[1]['image_id'] = 100
        with self.assertRaises(ValueError): score_video_rows(rows)

    def test_comparison_requires_audit_and_keeps_step_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [row(), row(side='right_hand')]
            spec = dict(evaluation_layer='video_system', frame_stride=1, image_ids=[0],
                scoring_frame_stride=3, scoring_source_anchor=0, sampled_mean_dice_empty_empty_one=True)
            runs = []
            for method in ('ve','residual'):
                path = root/method; path.mkdir()
                (path/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                (path/'cpu-audit.json').write_text(json.dumps(dict(status='complete',checked_queries=2,records_sha256='hash')))
                visuals = []
                for side in ('left_hand','right_hand'):
                    directory = path/'visuals/image-000000'/side; directory.mkdir(parents=True)
                    for key in ('rgb','reference','detected'):
                        Image.new('RGB',(30,20),'white').save(directory/f'{key}.png')
                    visuals.append(dict(image_id=0,side=side))
                summary = dict(protocol=spec, queries=2, records_sha256='hash', visuals=visuals,
                    method=method, epoch=None, step=600 if method=='residual' else None,
                    residual_metadata=dict(residual_positions='all'), sampled_video_metrics=score_video_rows(rows))
                runs.append((path,summary))
            with patch('scripts.compare_senior_video_routes.load_runs',return_value=runs):
                result = compare([],root/'comparison')
                self.assertEqual(result['status'],'complete')
                self.assertIn('partial epoch',(root/'comparison/README.md').read_text())
                with Image.open(root/'comparison/000000-left_hand.png') as im:
                    self.assertEqual(im.size,(120,64))
                (runs[1][0]/'cpu-audit.json').write_text('{}')
                with self.assertRaises(ValueError): compare([],root/'no-audit')
            self.assertIn('all step 600',label(runs[1][1]))


if __name__ == '__main__':
    unittest.main()

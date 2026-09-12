from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from scripts.evaluate_hand_routes import require_residual_epoch, protocol
from scripts.compare_hand_routes import load_runs, export
from scripts import plot_residual_training as plotting


class HandRoutesTest(unittest.TestCase):
    def test_epoch_requires_validation_and_boundary(self):
        state = dict(progress=dict(global_step=20), training_config=dict(steps_per_epoch=10),
                     validation_state=dict(last_validation_step=20))
        require_residual_epoch(state, 2)
        for epoch in (1, 0, True, None):
            with self.assertRaises(ValueError): require_residual_epoch(state, epoch)
        state['validation_state']['last_validation_step'] = 10
        with self.assertRaises(ValueError): require_residual_epoch(state, 2)

    def test_protocol_tracks_subset_and_precision(self):
        images = [{'id': 8}, {'id': 9}]
        a = protocol(images, [0], 'b', 't', 'a', 1)
        self.assertNotEqual(a, protocol(images, [1], 'b', 't', 'a', 1))
        self.assertNotEqual(a, protocol(images, [0], 'b', 't', 'a', 2))
        self.assertEqual(a['precision'], 'bf16')

    def test_compare_rejects_legacy_incomplete_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('a', 'b')]
            p = protocol([{'id': 8}], [0], 'b', 't', 'a', 1)
            for path in paths:
                path.mkdir()
                (path / 'records.jsonl').write_text('\n'.join(json.dumps(dict(image_id=8, prompt_key=s))
                    for s in ('left_hand', 'right_hand')))
                summary = dict(status='complete', actual_complete_query_coverage_verified=True,
                    protocol=p, records_sha256='hash', metrics={})
                (path / 'summary.json').write_text(json.dumps(summary))
            with patch('scripts.compare_hand_routes.full.shared.sha256', return_value='hash'), \
                 patch('scripts.compare_hand_routes.full.summarize', return_value={}):
                self.assertEqual(len(load_runs(paths)), 2)
                for change in ({'status': 'running'}, {'protocol': {}}, {'records_sha256': 'bad'},
                               {'protocol': {**p, 'batch_size': 2}}):
                    bad = {**deepcopy(summary), **change}
                    (paths[1] / 'summary.json').write_text(json.dumps(bad))
                    with self.assertRaises(ValueError): load_runs(paths)

    def test_spatial_plot_uses_real_tag_without_total_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / 'logs'
            logs.mkdir()
            record = dict(kind='train', global_step=20, samples_seen=120, wall_seconds=10.,
                values={'loss/mask_objective': .2, 'loss/loss_mask': .05, 'loss/loss_dice': .15})
            (logs / 'metrics.jsonl').write_text(json.dumps(record) + '\n')
            result = plotting.export_plots(logs, root / 'plots')
            self.assertTrue((root / 'plots/spatial_mask_objective.png').exists())
            self.assertNotIn('train/total_loss', result['series'])

    def test_comparison_panels_are_separate_and_reference_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            metric = dict(present_mean_candidate_dice=.8, present_mean_miss_zero_dice=.7,
                candidate_boundary_iou_4px=.5, false_negative_queries=1, present_queries=2,
                false_positive_queries=0, absent_queries=1)
            for method in ('ve', 'spatial'):
                path = root / method
                directory = path / 'visuals/image-000008/left_hand'
                directory.mkdir(parents=True)
                for name in ('rgb', 'reference', 'candidate', 'detected'):
                    Image.new('RGB', (20, 10), 'white').save(directory / f'{name}.png')
                runs.append((path, dict(method=method, epoch=None if method == 've' else 1,
                    protocol={}, visuals=[dict(image_id=8, side='left_hand')],
                    metrics=dict(primary_provided_nonflagged=dict(overall=metric)))))
            with patch('scripts.compare_hand_routes.load_runs', return_value=runs):
                export([], root / 'out')
                with Image.open(root / 'out/000008-left_hand.png') as canvas:
                    self.assertEqual(canvas.size, (120, 50))
                Image.new('RGB', (20, 10), 'black').save(directory / 'reference.png')
                with self.assertRaises(ValueError): export([], root / 'bad')


if __name__ == '__main__':
    unittest.main()

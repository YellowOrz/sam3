"""Small CPU fixtures; no model or GPU and no project data mutations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as masks

from scripts import visualize_residual_validation as visual


def rle(array):
    result = masks.encode(np.asfortranarray(array.astype(np.uint8)))
    result['counts'] = result['counts'].decode('ascii')
    return result


class ValidationVisualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root/'data'
        self.data.mkdir()
        self.base = self.root/'baseline'
        self.later = self.root/'later'
        self.out = self.root/'out'
        self.mask = np.zeros((10, 12), bool)
        self.mask[2:7, 3:9] = True
        images, annotations = [], []
        for index, image_id in enumerate((100, 110, 120, 130)):
            name = f'{image_id}.png'
            Image.fromarray(np.full((10, 12, 3), index*40, np.uint8)).save(self.data/name)
            images.append({'id': image_id, 'file_name': name, 'height': 10, 'width': 12,
                           'source_dataset': 'dexycb',
                           'source_rgb_sha256': hashlib.sha256((self.data/name).read_bytes()).hexdigest()})
            annotations.append({'id': index+1, 'image_id': image_id, 'category_id': 1,
                                'segmentation': rle(self.mask)})
        data = {'info': {'dataset_role': 'val'}, 'images': images, 'annotations': annotations,
                'categories': [{'id': 1, 'name': 'left_hand'}, {'id': 2, 'name': 'right_hand'}]}
        (self.data/'annotations.json').write_text(json.dumps(data))
        self.annotation_sha = hashlib.sha256((self.data/'annotations.json').read_bytes()).hexdigest()
        for step, directory in ((0, self.base), (5021, self.later)):
            directory.mkdir()
            summary = {'dataset_role': 'validation', 'scope': 'dexycb_val', 'world_size': 2,
                       'annotations_sha256': self.annotation_sha, 'global_step': step,
                       'detection_threshold': .5, 'mask_threshold': .5,
                       'metrics': {'images': 4, 'queries': 8}, 'verified_query_records': 8}
            (directory/'summary.json').write_text(json.dumps(summary))
            grouped = [[], []]
            for index, image in enumerate(images):
                for side in visual.SIDES:
                    present = side == 'left_hand'
                    detected = present == (step > 0)
                    grouped[index%2].append({'dataset_role': 'validation', 'rank': index%2,
                        'dataset_index': index, 'image_id': image['id'], 'file_name': image['file_name'],
                        'prompt_key': side, 'original_size': [10, 12], 'identity_verified': True,
                        'provenance': {'annotations_sha256': self.annotation_sha},
                        'top_confidence': .8 if detected else .2, 'detected': detected,
                        'prediction_rle': rle(self.mask), 'reference_present': present,
                        'candidate_dice': 1. if present else None,
                        'miss_zero_dice': float(detected) if present else None})
            for rank, rows in enumerate(grouped):
                (directory/f'rank-{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def tearDown(self):
        self.tmp.cleanup()

    def run_visual(self, **kwargs):
        options = dict(data_root=self.data, baseline_dir=self.base, comparison_dir=self.later,
                       output_dir=self.out, count=2)
        options.update(kwargs)
        return visual.visualize(**options)

    def test_real_png_roundtrip_and_detection_suppression(self):
        result = self.run_visual(count=4)
        self.assertEqual(result['scope'], 'validation_diagnostic')
        self.assertEqual(result['steps'], [0, 5021])
        self.assertEqual(result['png_vs_rle_pixel_verification']['composite_content_panels'], 28)
        self.assertTrue(result['source_files_verified_unchanged'])
        self.assertEqual(len(result['browse_pages']), 1)
        directory = self.out/'image-00000100'
        with Image.open(directory/'step-00000000__left_hand__candidate.png') as image:
            self.assertGreater(np.asarray(image).sum(), 0)
        with Image.open(directory/'step-00000000__left_hand__detected.png') as image:
            self.assertEqual(np.asarray(image).sum(), 0)
        with Image.open(directory/'step-00005021__left_hand__detected.png') as image:
            np.testing.assert_array_equal(np.asarray(image), self.mask.astype(np.uint8)*255)
        self.assertEqual(len(result['artifacts']), 4*12+1)
        self.assertIn('Dex原始', (self.out/'README.md').read_text())
        self.assertFalse((self.out/'manifest.json').read_text().find('validation_diagnostic') < 0)

    def test_selection_independent_of_order_and_metrics(self):
        ids = list(range(100))
        self.assertEqual(visual.select_ids(ids), visual.select_ids(ids[::-1]))
        for invalid in ([1, 1], ['1', 2]):
            with self.assertRaises((ValueError, TypeError)):
                visual.select_ids(invalid, count=1)
        with self.assertRaises(ValueError):
            visual.select_ids([1], count=2)

    def test_duplicate_record_rejected_after_plan_before_manifest(self):
        path = self.base/'rank-0.jsonl'
        path.write_text(path.read_text()+path.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_visual()
        plan = json.loads((self.out/'plan.json').read_text())
        self.assertFalse(plan['selection_uses_metrics_or_predictions'])
        self.assertFalse((self.out/'manifest.json').exists())

    def test_missing_tail_query_rejected(self):
        path = self.later/'rank-1.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.run_visual()

    def test_nonfinite_and_wrong_gate_rejected(self):
        path = self.base/'rank-0.jsonl'
        original = path.read_text()
        path.write_text(original.replace('"top_confidence": 0.2', '"top_confidence": NaN', 1))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.run_visual()
        path.write_text(original.replace('"detected": false', '"detected": true', 1))
        with self.assertRaisesRegex(ValueError, 'Detection gate'):
            self.run_visual(output_dir=self.root/'other')

    def test_wrong_validation_hash_rejected(self):
        path = self.base/'summary.json'
        summary = json.loads(path.read_text())
        summary['annotations_sha256'] = '0'*64
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'matching completed'):
            self.run_visual()
        self.assertFalse(self.out.exists())

    def test_source_rgb_change_rejected(self):
        Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(self.data/'130.png')
        with self.assertRaisesRegex(ValueError, 'Source RGB'):
            self.run_visual(count=4)
        self.assertFalse((self.out/'manifest.json').exists())

    def test_existing_output_and_test_scope_rejected(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'new external'):
            self.run_visual()
        path = self.data/'annotations.json'
        data = json.loads(path.read_text())
        data['info']['dataset_role'] = 'external_test_only'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'validation annotations'):
            self.run_visual(output_dir=self.root/'new')

    def test_bad_rle_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shape'):
            visual.decode_rle(rle(self.mask), (12, 10))


if __name__ == '__main__':
    unittest.main()

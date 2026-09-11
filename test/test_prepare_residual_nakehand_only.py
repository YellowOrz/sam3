"""Synthetic byte-preservation/publication contracts; no GPU or original videos."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts import prepare_residual_nakehand_only as prep


class NakehandOnlyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mixed, self.test, self.output = (self.root/name for name in ('mixed', 'test', 'nake-only'))
        self.original = self.root/'original-nake'
        for split in ('train', 'val', 'development_holdout'):
            (self.original/split/'images').mkdir(parents=True)
        for split in ('train', 'val'):
            (self.mixed/split/'images').mkdir(parents=True)
        self.test.mkdir()
        images, annotations = [], []
        self.nake_images = []
        for index in range(10):
            nake = index >= 2
            image_id = index+1
            filename = f'images/{image_id:08d}.png'
            source_split = ('train', 'val', 'development_holdout')[index%3] if nake else 'train'
            path = self.mixed/'train'/filename
            payload = f'actual-fixture-RGB-bytes-{index}'.encode()
            if nake:
                original = self.original/source_split/'images'/f'{index}.png'
                original.write_bytes(payload)
                path.symlink_to(original)
            else:
                path.write_bytes(payload)
            image = {'id': image_id, 'height': 4, 'width': 4, 'file_name': filename,
                'dataset_role': 'train', 'source_dataset': 'nakehand' if nake else 'dexycb',
                'source_split': source_split, 'recording': 'other-recording', 'frame': index,
                'source_image_id': 100+index, 'source_rgb_sha256': prep.sha256(path),
                'provenance': {'preserve': ['all', 'fields'], 'original_dataset_role': source_split}}
            images.append(image)
            if nake:
                self.nake_images.append(image)
            for category in (() if index == 2 else (1,) if index%2 else (1, 2)):
                annotations.append({'id': len(annotations)+1, 'image_id': image_id,
                    'category_id': category, 'iscrowd': 0, 'bbox': [0, 0, 2, 2], 'area': 4,
                    'segmentation': {'size': [4, 4], 'counts': '02208'},
                    'source_dataset': image['source_dataset'], 'provenance': {'unaltered': True}})
        self.coco = {'info': {'dataset_role': 'train', 'split': 'train'},
                     'categories': prep.CATEGORIES, 'images': images, 'annotations': annotations}
        (self.mixed/'train/annotations.json').write_text(json.dumps(self.coco))
        val_images = []
        for image_id in (101, 102, 103):
            name = f'images/{image_id:08d}.png'
            (self.mixed/'val'/name).write_bytes(b'unchanged-Dex-val')
            val_images.append({'id': image_id, 'height': 4, 'width': 4, 'file_name': name,
                               'source_dataset': 'dexycb'})
        val = {'info': {'dataset_role': 'val'}, 'categories': prep.CATEGORIES,
               'images': val_images, 'annotations': []}
        (self.mixed/'val/annotations.json').write_text(json.dumps(val))
        self._bind_mixed()
        (self.test/'annotations.json').write_text(json.dumps({'images': [{'id': n} for n in range(128)]}))
        (self.test/'manifest.json').write_text('{}')
        (self.test/'frozen-plan.json').write_text('{"seed":20260911}')
        (self.test/'READY.json').write_text(json.dumps({'status': 'complete',
            **{key: prep.sha256(self.test/name) for name, key in [
                ('annotations.json', 'annotations_sha256'), ('manifest.json', 'manifest_sha256'),
                ('frozen-plan.json', 'frozen_plan_sha256')]}}))

    def _bind_mixed(self):
        approval = {'format': prep.APPROVAL_FORMAT, 'approved_by': 'user'}
        for split in ('train', 'val'):
            approval[split] = {'root': str(self.mixed/split), 'exhaustive_hand_labels': True,
                'annotations_sha256': prep.sha256(self.mixed/split/'annotations.json'),
                'allowed_image_roots': [str(self.original)] if split == 'train' else []}
        (self.mixed/'training-approval.json').write_text(json.dumps(approval))
        manifest = {'status': 'complete', 'excluded_frames': [prep.EXCLUDED_FRAME],
                    'training_approval_sha256': prep.sha256(self.mixed/'training-approval.json')}
        (self.mixed/'manifest.json').write_text(json.dumps(manifest))
        (self.mixed/'READY.json').write_text(json.dumps({'status': 'complete',
            'manifest_sha256': prep.sha256(self.mixed/'manifest.json'),
            'training_approval_sha256': manifest['training_approval_sha256'],
            'splits': {split: {'annotations_sha256': approval[split]['annotations_sha256']}
                       for split in ('train', 'val')}}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self, **kwargs):
        options = dict(mixed_root=self.mixed, test_root=self.test, output_dir=self.output,
            expected_nake_images=8, expected_val_images=3, expected_source_shas={
                'train': prep.sha256(self.mixed/'train/annotations.json'),
                'val': prep.sha256(self.mixed/'val/annotations.json'),
                'test_annotations': prep.sha256(self.test/'annotations.json')})
        options.update(kwargs)
        return prep.prepare(**options)

    def test_exact_subset_rows_rgb_aliases_val_and_approval(self):
        before = {p: p.read_bytes() for p in self.mixed.rglob('*.json')}
        result = self.run_prepare()
        derived = json.loads((self.output/'train/annotations.json').read_bytes())
        self.assertEqual(derived['images'], self.nake_images)
        expected_annotations = [row for row in self.coco['annotations'] if row['source_dataset'] == 'nakehand']
        self.assertEqual(derived['annotations'], expected_annotations)
        self.assertEqual(result['train_counts']['images'], 8)
        self.assertEqual(result['train_counts']['empty_images'], 1)
        self.assertEqual(result['source_dataset_counts'], {'nakehand': 8})
        self.assertEqual((self.output/'val').resolve(), self.mixed/'val')
        self.assertEqual((self.output/'val/annotations.json').read_bytes(), (self.mixed/'val/annotations.json').read_bytes())
        approval = json.loads((self.output/'training-approval.json').read_bytes())
        roots = set(approval['train']['allowed_image_roots'])
        self.assertEqual(roots, {str(self.mixed/'train/images'), *(str(self.original/split/'images')
                          for split in ('train', 'val', 'development_holdout'))})
        for image in derived['images']:
            path = self.output/'train'/image['file_name']
            self.assertTrue(path.is_symlink())
            self.assertEqual(path.read_bytes(), (self.mixed/'train'/image['file_name']).read_bytes())
        self.assertEqual(prep.validate_publication(self.output)['status'], 'complete')
        for path, raw in before.items():
            self.assertEqual(path.read_bytes(), raw)

    def test_preregistered_two_epochs_not_best_or_test_selection(self):
        result = self.run_prepare()
        plan = result['training_plan']
        self.assertEqual((plan['epochs'], plan['world_size'], plan['batch_size_per_rank']), (2, 3, 2))
        self.assertEqual((plan['steps_per_epoch'], plan['dropped_images_per_epoch'], plan['planned_image_exposures']), (1, 2, 12))
        self.assertFalse(plan['same_exposure_as_mixed_experiment'])
        self.assertFalse(plan['resume_from_mixed_checkpoint'])
        self.assertIn('zero output delta', plan['initialization'])
        test = result['preregistered_external_test_plan']
        self.assertEqual(test['completed_epochs'], [1, 2])
        self.assertEqual(test['expected_global_steps'], [1, 2])
        self.assertTrue(test['report_both_epochs'])
        self.assertFalse(test['test_used_for_selection_or_tuning'])
        self.assertIn('not best.pt', test['checkpoint_policy'])

    def test_existing_trainer_approval_contracts_accept_new_publication(self):
        self.run_prepare()
        from scripts.train_residual_ddp import approval_contracts
        approval, approval_sha, contracts = approval_contracts(SimpleNamespace(
            approval=self.output/'training-approval.json', validation=True,
            data_root=None, val_root=None))
        self.assertEqual(len(contracts['train'].images), 8)
        self.assertEqual(len(contracts['val'].images), 3)
        self.assertEqual(contracts['val'].root, self.mixed/'val')
        self.assertEqual(approval_sha, prep.sha256(self.output/'training-approval.json'))
        self.assertEqual(contracts['train'].annotations_sha256, approval['train']['annotations_sha256'])

    def test_bad_original_frame_is_rejected_not_doubly_removed(self):
        data = deepcopy(self.coco)
        data['images'][-1].update(source_image_id=4713, recording=prep.EXCLUDED_FRAME['recording'], frame=0)
        (self.mixed/'train/annotations.json').write_text(json.dumps(data))
        self._bind_mixed()
        with self.assertRaisesRegex(ValueError, 'excluded source frame'):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_changed_source_rgb_prevents_ready(self):
        image = self.nake_images[-1]
        (self.mixed/'train'/image['file_name']).resolve().write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError, 'Source RGB SHA'):
            self.run_prepare()
        self.assertFalse((self.output/'READY.json').exists())

    def test_missing_rgb_and_existing_output_fail_closed(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        missing = (self.mixed/'train'/self.nake_images[0]['file_name']).resolve()
        missing.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_prepare(output_dir=self.root/'new')

    def test_source_sha_and_ready_bindings_required(self):
        with self.assertRaisesRegex(ValueError, 'Source annotation SHA'):
            self.run_prepare(expected_source_shas={'train': '0'*64, 'val': '0'*64})
        ready = self.mixed/'READY.json'
        data = json.loads(ready.read_text())
        data['manifest_sha256'] = '0'*64
        ready.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'completed approved'):
            self.run_prepare()

    def test_foreign_alias_rejected_and_original_annotations_unchanged(self):
        self.run_prepare()
        image = self.nake_images[0]
        path = self.output/'train'/image['file_name']
        foreign = self.root/'foreign.png'
        foreign.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(foreign)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            prep.validate_publication(self.output)

    def test_wrong_count_or_test_publication_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'counts changed'):
            self.run_prepare(expected_nake_images=7)
        (self.test/'annotations.json').write_text('{"images":[]}')
        with self.assertRaisesRegex(ValueError, 'test publication hash'):
            self.run_prepare()

    def test_mutated_geometry_changes_per_image_equivalence(self):
        image = self.nake_images[1]
        rows = [row for row in self.coco['annotations'] if row['image_id'] == image['id']]
        original = prep._image_equivalence(image, rows)
        changed = deepcopy(rows)
        changed[0]['bbox'][2] += 1
        self.assertNotEqual(original, prep._image_equivalence(image, changed))
        changed = deepcopy(rows)
        changed[0]['segmentation']['counts'] = 'different'
        self.assertNotEqual(original, prep._image_equivalence(image, changed))

    def test_output_cannot_nest_in_sources_or_repository(self):
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.run_prepare(output_dir=self.mixed/'new')

    def test_cli_preparation_requires_explicit_source_roots(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            prep.main(['--output-dir', str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_cli_verification_does_not_require_personal_source_defaults(self):
        self.run_prepare()
        stream = io.StringIO()
        with redirect_stdout(stream):
            prep.main(['--output-dir', str(self.output), '--verify-only'])
        self.assertEqual(json.loads(stream.getvalue())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()

"""CPU fixtures: no training, evaluation, remote calls, or existing artifact writes."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from scripts import finish_overnight_review as finalizer


class OvernightReviewTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        (self.repo / 'docs').mkdir(parents=True)
        (self.repo / 'scripts').mkdir()
        (self.repo / 'docs/goal.md').write_text('# 手物分割目标\n固定实验目标。\n')
        shutil.copyfile(Path(finalizer.__file__).with_name('package_review_docs.py'),
                        self.repo / 'scripts/package_review_docs.py')
        self.run = self.root / 'explicit-run'
        self.run.mkdir()
        self.output = self.root / 'new-publication'
        self.created = datetime.now(timezone.utc)

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return finalizer.fingerprint(path)

    def complete_report(self, name='boundary-comparison', *, input_format=False):
        output = self.run / name
        fmt = 'sam3-input-ve-boundary-validation-v1' if input_format else 'sam3-hand-boundary-validation-v1'
        labels = finalizer.KNOWN_REPORTS[fmt]
        summary = {'format': fmt, 'status': 'completed', 'full_val_evaluated': True,
                   'evaluated_images': 3449, 'completed_images_per_model': {label:3449 for label in labels},
                   'all_sources_unchanged': True, 'observed_identity_verified': True,
                   'parameters_unchanged_by_version_counter': True,
                   'training_performed': False, 'dataset_role': 'validation',
                   'evaluated_image_ids': list(range(3449)), 'models': {label:{} for label in labels},
                   'record_files': {}}
        for label in labels:
            rows = [{'image_id': image_id, 'prompt_key': side, 'model': label,
                     'identity_verified': True, 'observed_coco_image_id': image_id}
                    for image_id in range(3449) for side in finalizer.SIDES]
            path = output / 'records' / f'{label}.json'
            summary['record_files'][label] = {'path':str(path), 'sha256':self.write_json(path, rows)}
        summary_path = output / 'summary.json'
        report = output / 'REPORT.md'
        report.write_text('# 实际完成的验证报告\n这是 CPU 合同夹具。\n')
        return {'summary':str(summary_path), 'summary_sha256':self.write_json(summary_path, summary),
                'report':str(report), 'report_sha256':finalizer.fingerprint(report)}

    def state(self, status='running', **extra):
        value = {'format':'sam3-nakehand-boundary-supervisor-v1', 'status':status,
                 'training_comparison': finalizer.QUEUE_TRAINING_COMPARISONS['sam3-nakehand-boundary-supervisor-v1'], **extra}
        self.write_json(self.run / 'state.json', value)
        return finalizer.snapshot_run(self.run)

    def args(self, **kwargs):
        return SimpleNamespace(run=[self.run], output_dir=self.output, project_root=self.repo,
                               deadline=kwargs.get('deadline', self.created+timedelta(hours=1)),
                               once=kwargs.get('once', True))

    def execute(self, args=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return finalizer.execute(args or self.args(), **kwargs)

    def test_actual_three_complete_record_sets_accepted(self):
        entry = self.complete_report()
        actual = finalizer.validate_report(entry, self.run)
        self.assertEqual(actual['actual_images_per_variant'], 3449)
        self.assertEqual({item['queries'] for item in actual['actual_records'].values()}, {6898})

    def test_success_requires_both_distinct_boundary_reports(self):
        first = self.complete_report()
        snapshot = self.state('complete', verified_reports={'boundary-comparison':first})
        self.assertFalse(finalizer.verified_run(snapshot)['validated_complete'])
        # Reusing the same full summary under two labels is not a second experiment.
        snapshot = self.state('complete', verified_reports={'boundary-comparison':first, 'semantic-comparison':first})
        self.assertFalse(finalizer.verified_run(snapshot)['validated_complete'])
        second = self.complete_report('semantic-comparison')
        snapshot = self.state('complete', verified_reports={'boundary-comparison':first, 'semantic-comparison':second})
        self.assertTrue(finalizer.verified_run(snapshot)['validated_complete'])

    def test_input_format_uses_actual_three_model_inventory(self):
        entry = self.complete_report('lr1e-3-validation', input_format=True)
        actual = finalizer.validate_report(entry, self.run)
        self.assertEqual(set(actual['actual_records']), {'baseline','output-delta','input-ve'})
        summary = json.loads(Path(entry['summary']).read_text())
        summary['parameters_unchanged_by_version_counter'] = False
        entry['summary_sha256'] = self.write_json(Path(entry['summary']), summary)
        with self.assertRaisesRegex(ValueError, 'unchanged model'):
            finalizer.validate_report(entry, self.run)

    def test_input_queue_requires_both_lr_reports_and_training_comparison(self):
        first = self.complete_report('lr1e-3-validation', input_format=True)
        second = self.complete_report('lr3e-4-validation', input_format=True)
        fmt = 'sam3-nakehand-input-ablation-supervisor-v1'
        settings = {'format':fmt, 'verified_reports':{'lr1e-3':first, 'lr3e-4':second},
                    'training_comparison':finalizer.QUEUE_TRAINING_COMPARISONS[fmt],
                    'commands':{'lr1e-3-probe':{'status':'completed','verified_result':{'full':False}}}}
        self.assertTrue(finalizer.verified_run(self.state('complete', **settings))['validated_complete'])
        settings['training_comparison'] = {}
        self.assertFalse(finalizer.verified_run(self.state('complete', **settings))['validated_complete'])
        settings['training_comparison'] = finalizer.QUEUE_TRAINING_COMPARISONS[fmt]
        settings['verified_reports'] = {'lr1e-3':first}
        self.assertFalse(finalizer.verified_run(self.state('complete', **settings))['validated_complete'])

    def test_incomplete_records_rejected_despite_full_progress_counts(self):
        entry = self.complete_report()
        summary = json.loads(Path(entry['summary']).read_text())
        record = summary['record_files']['baseline']
        rows = json.loads(Path(record['path']).read_text())
        rows[-1] = rows[0]
        record['sha256'] = self.write_json(Path(record['path']), rows)
        entry['summary_sha256'] = self.write_json(Path(entry['summary']), summary)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            finalizer.validate_report(entry, self.run)

    def test_model_identity_mismatch_rejected_even_with_fresh_hashes(self):
        entry = self.complete_report()
        summary = json.loads(Path(entry['summary']).read_text())
        record = summary['record_files']['baseline']
        rows = json.loads(Path(record['path']).read_text())
        rows[0]['observed_coco_image_id'] = 999999
        record['sha256'] = self.write_json(Path(record['path']), rows)
        entry['summary_sha256'] = self.write_json(Path(entry['summary']), summary)
        with self.assertRaisesRegex(ValueError, 'identity/model'):
            finalizer.validate_report(entry, self.run)

    def test_digest_report_escape_and_symlink_rejected(self):
        entry = self.complete_report()
        for field in ('summary_sha256','report_sha256'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                finalizer.validate_report({**entry,field:'0'*64}, self.run)
        outside = self.root / 'foreign.md'
        outside.write_text('not approved')
        with self.assertRaises(ValueError):
            finalizer.validate_report({**entry,'report':str(outside)}, self.run)
        link = self.run / 'link.md'
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            finalizer.validate_report({**entry,'report':str(link)}, self.run)

    def test_unknown_contract_is_never_certified_complete(self):
        entry = self.complete_report()
        snapshot = self.state('complete', format='unregistered-queue', verified_reports={'one':entry})
        result = finalizer.verified_run(snapshot)
        self.assertFalse(result['validated_complete'])
        self.assertEqual(len(result['verified_reports']), 1)
        summary = json.loads(Path(entry['summary']).read_text())
        summary['format'] = 'future-unknown-format'
        entry['summary_sha256'] = self.write_json(Path(entry['summary']), summary)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            finalizer.validate_report(entry, self.run)

    def test_failed_queue_keeps_real_partial_report_without_upgrading_status(self):
        entry = self.complete_report()
        snapshot = self.state('failed_or_stopped', commands={
            'boundary-comparison':{'status':'completed','verified_result':entry},
            'semantic-comparison':{'status':'failed','command':['never','execute']}})
        result = finalizer.verified_run(snapshot)
        self.assertFalse(result['validated_complete'])
        self.assertEqual(len(result['verified_reports']), 1)
        self.assertEqual(result['status'], 'failed_or_stopped')

    def test_malformed_state_fails_closed_without_crashing_finalization(self):
        for extra in ({'verified_reports':[]}, {'commands':'bad'}, {'commands':{'bad':[]}}):
            with self.subTest(extra=extra):
                result = finalizer.verified_run(self.state('complete', **extra))
                self.assertFalse(result['validated_complete'])
        (self.run/'state.json').write_text('{unfinished')
        self.assertEqual(finalizer.snapshot_run(self.run)['status'], 'unreadable_or_not_started')

    def test_once_publishes_real_zip_without_touching_job_state(self):
        self.state('running', commands={'untrusted':['never','execute']})
        before = (self.run/'state.json').read_bytes()
        self.assertEqual(self.execute(), 0)
        self.assertEqual((self.run/'state.json').read_bytes(), before)
        state = json.loads((self.output/'finalizer-state.json').read_text())
        self.assertEqual(state['status'], 'snapshot_published')
        self.assertFalse(state['all_runs_validated_complete'])
        self.assertFalse(state['gpu_work'])
        self.assertFalse(state['windows_transfer_performed'])
        notes = (self.output/'notes/OVERNIGHT_REVIEW.md').read_text()
        self.assertIn('用户已确认 frame 0 的可见手是右手', notes)
        self.assertIn('缺标与错侧同时存在', notes)
        self.assertIn('没有收录可严格验收的完整结果', notes)
        with zipfile.ZipFile(state['receipt']['archive']) as archive:
            self.assertIn('START_HERE.md', archive.namelist())
            self.assertFalse(any(name.endswith('.pt') for name in archive.namelist()))
        with self.assertRaises(FileExistsError):
            self.execute()

    def test_polling_is_at_most_30_seconds_and_reserves_publication_time(self):
        self.state()
        current, sleeps, calls = [self.created], [], []
        def advance(seconds):
            sleeps.append(seconds)
            current[0] += timedelta(seconds=seconds)
        def package(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.run(command, **kwargs)
        self.execute(self.args(once=False, deadline=self.created+timedelta(seconds=181)),
                     now=lambda:current[0], sleep=advance, run_command=package)
        self.assertEqual(sleeps, [30,30,1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['timeout'], 120)
        self.assertEqual(Path(calls[0][0][1]).name, 'package_review_docs.py')

    def test_packaging_failure_preserves_notes_and_honest_failed_state(self):
        self.state()
        def fail(command, **kwargs):
            return subprocess.CompletedProcess(command, 7, 'partial output', 'fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'Packaging failed'):
            self.execute(run_command=fail)
        state = json.loads((self.output/'finalizer-state.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertTrue((self.output/'notes/OVERNIGHT_REVIEW.md').is_file())
        self.assertIn('fixture failure', (self.output/'packaging.log').read_text())

    def test_previously_verified_artifact_mutation_detected(self):
        entry = self.complete_report()
        verified = finalizer.verified_run(self.state('failed', verified_reports={'one':entry}))
        Path(entry['report']).write_text('changed after acceptance')
        with self.assertRaisesRegex(RuntimeError, 'Previously verified artifact changed'):
            finalizer.verify_receipt_sources([verified])

    def test_fixed_visual_selection_keeps_rgb_refs_and_models_separate(self):
        entry = self.complete_report()
        summary_path = Path(entry['summary'])
        summary = json.loads(summary_path.read_text())
        summary['evaluated_dataset_indices'] = list(range(3449))
        summary['visuals'] = []
        for index in (0,1000,2000,3448):
            directory = summary_path.parent/'visuals'/f'image-{index:06d}'
            directory.mkdir(parents=True)
            for name in ('comparison.png','rgb.png','left_hand__reference.png','right_hand__reference.png'):
                (directory/name).write_bytes(b'PNG fixture bytes')
            for label in finalizer.KNOWN_REPORTS[summary['format']]:
                (directory/label).mkdir()
                for side in finalizer.SIDES:
                    (directory/label/f'{side}__detected.png').write_bytes(b'separate mask bytes')
            summary['visuals'].append({'dataset_index':index,'image_id':index,
                                       'directory':str(directory),'comparison':str(directory/'comparison.png')})
        entry['summary_sha256'] = self.write_json(summary_path, summary)
        snapshot = self.state('failed', verified_reports={'boundary-comparison':entry})
        accepted = finalizer.verified_run(snapshot)
        report = accepted['verified_reports'][0]
        self.assertEqual([group['dataset_index'] for group in report['visuals']], [0,2000,3448])
        self.assertEqual([len(group['assets']) for group in report['visuals']], [10,10,10])
        notes = finalizer.render_notes([accepted], self.created, self.args().deadline, True)
        self.assertIn('baseline left_hand 达阈预测', notes)
        self.assertIn('left_hand__reference.png', notes)
        before = finalizer.fingerprint(Path(report['visuals'][0]['assets'][0]['path']))
        self.execute()
        published = json.loads((self.output/'finalizer-state.json').read_text())
        self.assertEqual(published['receipt']['asset_count'], 30)
        self.assertEqual(finalizer.fingerprint(Path(report['visuals'][0]['assets'][0]['path'])), before)
        summary['visuals'][0]['directory'] = str(self.root/'outside')
        entry['summary_sha256'] = self.write_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            finalizer.validate_report(entry, self.run)

    def test_cli_rejects_naive_deadline_long_window_existing_and_nested_outputs(self):
        base = ['--run',str(self.run),'--project-root',str(self.repo),'--output-dir',str(self.output)]
        for deadline in ('2026-09-11T08:55:00', (self.created+timedelta(hours=10)).isoformat()):
            with self.subTest(deadline=deadline), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                finalizer.parse_args(base+['--deadline',deadline])
        deadline = (self.created+timedelta(hours=1)).isoformat()
        for output in (self.run/'nested', self.repo/'docs/new', self.run, self.root):
            with self.subTest(output=output), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                finalizer.parse_args(base+['--output-dir',str(output),'--deadline',deadline])
        parsed = finalizer.parse_args(base+['--deadline',deadline,'--once'])
        self.assertTrue(parsed.once)

    def test_external_docs_cli_and_finalizer_pass_physical_root_to_packager(self):
        sibling = self.root / 'docs'
        (self.repo / 'docs').rename(sibling)
        (self.repo / 'docs').symlink_to(sibling, target_is_directory=True)
        self.state('failed')
        parsed = finalizer.parse_args([
            '--run',str(self.run),'--project-root',str(self.repo),'--docs-root',str(sibling),
            '--output-dir',str(self.output),'--deadline',(self.created+timedelta(hours=1)).isoformat(),'--once'])
        self.assertEqual(parsed.docs_root, sibling)
        self.execute(parsed)
        published = json.loads((self.output / 'finalizer-state.json').read_text())
        self.assertEqual(published['source_docs_root'], str(sibling))
        index = published['command'].index('--docs-root')
        self.assertEqual(published['command'][index+1], str(sibling))
        manifest = json.loads((self.output / 'bundle/review/manifest.json').read_text())
        self.assertEqual(manifest['source_docs_root'], str(sibling))

    def test_sibling_docs_default_and_nested_publication_rejected(self):
        sibling = self.root / 'docs'
        (self.repo / 'docs').rename(sibling)
        base = ['--run',str(self.run),'--project-root',str(self.repo),
                '--deadline',(self.created+timedelta(hours=1)).isoformat()]
        parsed = finalizer.parse_args(base+['--output-dir',str(self.output),'--once'])
        self.assertEqual(parsed.docs_root, sibling)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            finalizer.parse_args(base+['--output-dir',str(sibling / 'nested')])
        args = self.args()
        args.output_dir = sibling / 'nested'
        with self.assertRaisesRegex(ValueError, 'must not contain'):
            self.execute(args)


if __name__ == '__main__':
    unittest.main()

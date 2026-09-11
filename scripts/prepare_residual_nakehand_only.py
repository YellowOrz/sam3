"""Publish the already-approved nakehand subset without altering pixels or labels.

Only container metadata changes. IDs, image rows and annotation rows remain
exactly the mixed publication's nakehand records. No original video decoding,
new labels, GPU work, checkpoint reuse, or changes to the existing mixed run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.prepare_residual_mixed_training import (
    APPROVAL_FORMAT, CATEGORIES, EXCLUDED_FRAME, LABEL_LIMITATIONS,
    _coco_counts, _validate_coco, atomic_json, sha256,
)

FORMAT = 'sam3-residual-nakehand-only-dataset-v1'
SOURCE_SHAS = {
    'train': '6ed07c5010fe04b52c6d35f2c28288afd638b783d813112f6e9859199e8da866',
    'val': 'dd6ed2af7448dc4541ea1901446fc0d6dd84a4828fa4b4eb41c75b7fa56c5ddd',
    'test_annotations': '96656cc7f2791c99ef1b577b4294bfd4ffa8e3e1be3cc366db0deeb2a5dc86b7',
}
TEST_POLICY = ('RealSense fixed128 external test at BOTH completed epoch1 and epoch2 checkpoints; '
               'report both, never select/tune from test and never substitute best.pt. '
               'Dex validation remains diagnostic/best tracking, not an independent nakehand holdout.')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_document(path, sources):
    raw = path.read_bytes()
    sources[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)


def image_path(root, image, approved_roots):
    name = Path(image['file_name'])
    if name.is_absolute() or '..' in name.parts or name.parts[:1] != ('images',):
        raise ValueError('Image filename must remain inside the images directory')
    path = root/name
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not any(resolved.is_relative_to(allowed) for allowed in approved_roots):
        raise ValueError('Source RGB escapes approved image roots')
    return path, resolved


def _image_equivalence(image, annotations):
    return {'image_id': image['id'], 'image_record_sha256': digest(image),
            'annotation_records_sha256': digest(annotations),
            'annotation_ids': [row['id'] for row in annotations],
            'source_rgb_sha256': image['source_rgb_sha256']}


def validate_publication(output_dir, *, verify_rgb=True):
    output = Path(output_dir).resolve()
    ready = json.loads((output/'READY.json').read_bytes())
    manifest = json.loads((output/'manifest.json').read_bytes())
    approval = json.loads((output/'training-approval.json').read_bytes())
    if (ready.get('format') != FORMAT or ready.get('status') != 'complete'
            or manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
            or ready['manifest_sha256'] != sha256(output/'manifest.json')
            or ready['training_approval_sha256'] != sha256(output/'training-approval.json')
            or manifest['training_approval_sha256'] != ready['training_approval_sha256']):
        raise ValueError('Incomplete or changed nakehand-only publication')
    for path, expected in manifest['source_documents'].items():
        path = Path(path)
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('Source metadata changed after publication')
    data = json.loads((output/'train/annotations.json').read_bytes())
    if sha256(output/'train/annotations.json') != approval['train']['annotations_sha256']:
        raise ValueError('Derived annotation SHA mismatch')
    grouped = _validate_coco(data)
    by_id = {row['id']: row for row in data['images']}
    expected = manifest['image_equivalence']
    if (len(expected) != len(by_id) or {row['image_id'] for row in expected} != set(by_id)
            or _coco_counts(data) != manifest['train_counts']):
        raise ValueError('Derived subset coverage/counts changed')
    allowed = [Path(path).resolve() for path in approval['train']['allowed_image_roots']]
    for row in expected:
        image = by_id[row['image_id']]
        if (image.get('source_dataset') != 'nakehand'
                or _image_equivalence(image, grouped[image['id']]) != row):
            raise ValueError('Image/annotation records are no longer byte-equivalent')
        path, resolved = image_path(output/'train', image, allowed)
        if not path.is_symlink() or resolved != (Path(manifest['source_mixed_root'])/'train'/image['file_name']).resolve():
            raise ValueError('Derived RGB no longer references the same mixed source')
        if verify_rgb and sha256(path) != row['source_rgb_sha256']:
            raise ValueError('RGB bytes changed')
    if (not (output/'val').is_symlink()
            or (output/'val').resolve() != Path(manifest['source_val_root'])
            or sha256(output/'val/annotations.json') != approval['val']['annotations_sha256']):
        raise ValueError('Original Dex validation alias/bytes changed')
    return manifest


def prepare(*, mixed_root, test_root, output_dir, expected_nake_images=18497,
            expected_val_images=2909, expected_source_shas=None):
    expected_source_shas = SOURCE_SHAS if expected_source_shas is None else expected_source_shas
    mixed, test = Path(mixed_root).resolve(strict=True), Path(test_root).resolve(strict=True)
    given_output = Path(output_dir)
    if given_output.exists() or given_output.is_symlink():
        raise FileExistsError('Require a completely new dataset directory')
    output = given_output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo) or any(output.is_relative_to(root) or root.is_relative_to(output)
                                         for root in (mixed, test)):
        raise ValueError('Output must be external and separate from immutable source roots')
    sources = {}
    old_approval = read_document(mixed/'training-approval.json', sources)
    old_manifest = read_document(mixed/'manifest.json', sources)
    old_ready = read_document(mixed/'READY.json', sources)
    if (old_approval.get('format') != APPROVAL_FORMAT or old_approval.get('approved_by') != 'user'
            or old_manifest.get('status') != 'complete' or old_ready.get('status') != 'complete'
            or old_ready['manifest_sha256'] != sources[str(mixed/'manifest.json')]['sha256']
            or old_ready['training_approval_sha256'] != sources[str(mixed/'training-approval.json')]['sha256']
            or old_manifest['training_approval_sha256'] != old_ready['training_approval_sha256']
            or old_manifest.get('excluded_frames') != [EXCLUDED_FRAME]):
        raise ValueError('Require the completed approved mixed publication and exact existing exclusion')
    original, original_val = (read_document(mixed/role/'annotations.json', sources) for role in ('train', 'val'))
    for role in ('train', 'val'):
        actual = sources[str(mixed/role/'annotations.json')]['sha256']
        if (actual != expected_source_shas[role] or actual != old_approval[role]['annotations_sha256']
                or old_approval[role].get('exhaustive_hand_labels') is not True
                or old_ready['splits'][role]['annotations_sha256'] != actual):
            raise ValueError('Source annotation SHA or exhaustive-label approval changed')
    _validate_coco(original)
    _validate_coco(original_val)
    selected = [row for row in original['images'] if row.get('source_dataset') == 'nakehand']
    if len(selected) != expected_nake_images or len(original_val['images']) != expected_val_images:
        raise ValueError('Approved nakehand/validation image counts changed')
    if any(row.get('source_dataset') != 'dexycb' for row in original_val['images']):
        raise ValueError('Validation must remain the original Dex subset')
    identities = [(row['source_split'], row['recording'], row['frame'], row['source_image_id']) for row in selected]
    if len(set(identities)) != len(selected):
        raise ValueError('Duplicate nakehand source identity')
    if any((row['recording'], row['frame']) == (EXCLUDED_FRAME['recording'], EXCLUDED_FRAME['frame'])
           or row['source_image_id'] == EXCLUDED_FRAME['source_image_id'] for row in selected):
        raise ValueError('Previously excluded source frame was restored; refuse rather than silently remove twice')
    selected_ids = {row['id'] for row in selected}
    if selected_ids & {row['id'] for row in original_val['images']}:
        raise ValueError('Train and validation image IDs overlap')
    selected_annotations = [row for row in original['annotations'] if row['image_id'] in selected_ids]
    derived = deepcopy(original)
    derived['images'] = deepcopy(selected)
    derived['annotations'] = deepcopy(selected_annotations)
    derived['info'].update(description='Approved nakehand-only subset of immutable mixed-v1',
        dataset_role='train', split='train', no_independent_nakehand_validation=True,
        label_limitations=LABEL_LIMITATIONS, test_policy=TEST_POLICY)
    grouped = _validate_coco(derived)
    if canonical(derived['images']) != canonical(selected) or canonical(derived['annotations']) != canonical(selected_annotations):
        raise RuntimeError('Selected record bytes changed')
    test_ready = read_document(test/'READY.json', sources)
    for name, key in [('annotations.json', 'annotations_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('frozen-plan.json', 'frozen_plan_sha256')]:
        document = read_document(test/name, sources)
        if test_ready.get('status') != 'complete' or test_ready[key] != sources[str(test/name)]['sha256']:
            raise ValueError('Fixed test publication hash binding changed')
        if name == 'annotations.json' and (len(document['images']) != 128 or
                test_ready[key] != expected_source_shas['test_annotations']):
            raise ValueError('Require unchanged RealSense fixed128 test annotations')
    global_batch, epochs = 6, 2
    steps = len(selected)//global_batch
    if steps == 0:
        raise ValueError('Not enough selected images for one global batch')
    training_plan = {'approved_by': 'user', 'epochs': epochs, 'world_size': 3,
        'batch_size_per_rank': 2, 'global_batch_size': global_batch, 'gradient_accumulation_steps': 1,
        'seed': 123, 'learning_rate': .001, 'anchor_weight': 0., 'weight_decay': 0.,
        'initialization': 'original natural VE cache with zero output delta; fresh optimizer and global_step0',
        'resume_from_mixed_checkpoint': False, 'trainable_parameters': 2048,
        'steps_per_epoch': steps, 'dropped_images_per_epoch': len(selected)%global_batch,
        'consumed_images_per_epoch': steps*global_batch, 'planned_steps': epochs*steps,
        'planned_image_exposures': epochs*steps*global_batch,
        'same_exposure_as_mixed_experiment': False,
        'enforcement': 'Approval records the user plan; caller must pass matching trainer CLI values.'}
    test_plan = {'dataset_role': 'external_test_only', 'root': str(test), 'images': 128,
        'ready_sha256': sources[str(test/'READY.json')]['sha256'],
        'annotations_sha256': test_ready['annotations_sha256'],
        'frozen_plan_sha256': test_ready['frozen_plan_sha256'],
        'completed_epochs': [1, 2], 'expected_global_steps': [steps, 2*steps],
        'checkpoint_policy': 'immutable actual completed-epoch checkpoint; bind SHA after completion; not best.pt',
        'report_both_epochs': True, 'test_used_for_selection_or_tuning': False,
        'do_not_change_remaining_training_after_epoch1_test': True,
        'missing_epoch_policy': 'report not completed/not evaluated; never fabricate a checkpoint or result'}
    source_train = mixed/'train'
    source_val = mixed/'val'
    approved_train = [source_train.resolve(), *(Path(p).resolve() for p in old_approval['train'].get('allowed_image_roots', []))]
    approved_val = [source_val.resolve(), *(Path(p).resolve() for p in old_approval['val'].get('allowed_image_roots', []))]
    train_roots, val_roots = {source_train/'images'}, {source_val/'images'}
    paths, source_stats = {}, {}
    for image in selected:
        path, resolved = image_path(source_train, image, approved_train)
        if not isinstance(image.get('source_rgb_sha256'), str) or len(image['source_rgb_sha256']) != 64:
            raise ValueError('Every selected RGB must have its original byte fingerprint')
        paths[image['id']] = path
        train_roots.add(resolved.parent)
    val_paths = set()
    for image in original_val['images']:
        _, resolved = image_path(source_val, image, approved_val)
        val_roots.add(resolved.parent)
        val_paths.add(resolved)
    if {path.resolve() for path in paths.values()} & val_paths:
        raise ValueError('Train and validation resolve to overlapping RGB files')
    output.mkdir(parents=True, exist_ok=False)
    (output/'train/images').mkdir(parents=True)
    atomic_json(output/'IN_PROGRESS.json', {'format': FORMAT, 'status': 'building',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_mixed_root': str(mixed)})
    equivalence = []
    for number, image in enumerate(selected, 1):
        source = paths[image['id']]
        before = source.stat()
        if sha256(source) != image['source_rgb_sha256']:
            raise ValueError(f'Source RGB SHA changed: {source}')
        after = source.stat()
        signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != signature:
            raise RuntimeError('Source RGB changed during verification')
        source_stats[str(source)] = signature
        link = output/'train'/image['file_name']
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
        if link.resolve() != source.resolve():
            raise RuntimeError('New RGB alias points to a different source')
        equivalence.append(_image_equivalence(image, grouped[image['id']]))
        if number % 2000 == 0:
            print(f'Verified unchanged nakehand RGB: {number}/{len(selected)}', flush=True)
    (output/'val').symlink_to(source_val, target_is_directory=True)
    atomic_json(output/'train/annotations.json', derived)
    # Reload serialized rows to verify that masks/bbox/classes/IDs/provenance did not change.
    reread = json.loads((output/'train/annotations.json').read_bytes())
    if canonical(reread['images']) != canonical(selected) or canonical(reread['annotations']) != canonical(selected_annotations):
        raise RuntimeError('Serialized image/annotation records differ from source subset')
    approval = {'format': APPROVAL_FORMAT, 'approved_by': 'user',
        'request': 'User changed to nakehand-only two epochs, retaining original approved subset; fixed RealSense test after each completed epoch.',
        'label_limitations': LABEL_LIMITATIONS, 'test_policy': TEST_POLICY,
        'training_plan': training_plan, 'preregistered_external_test_plan': test_plan,
        'validation_policy': 'same full Dex validation diagnostic and existing best tracking; do not substitute best for epoch tests',
        'source_approval_sha256': sources[str(mixed/'training-approval.json')]['sha256'],
        'train': {'root': str(output/'train'), 'annotations_sha256': sha256(output/'train/annotations.json'),
                  'exhaustive_hand_labels': True, 'allowed_image_roots': sorted(str(path) for path in train_roots)},
        'val': {'root': str(output/'val'), 'annotations_sha256': expected_source_shas['val'],
                'exhaustive_hand_labels': True, 'allowed_image_roots': sorted(str(path) for path in val_roots)}}
    atomic_json(output/'training-approval.json', approval)
    for path, expected in sources.items():
        if Path(path).stat().st_size != expected['bytes'] or sha256(Path(path)) != expected['sha256']:
            raise RuntimeError('Source metadata changed during publication')
    for path, expected in source_stats.items():
        status = Path(path).stat()
        if (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns) != expected:
            raise RuntimeError('Source RGB changed after verification')
    manifest = {'format': FORMAT, 'status': 'complete', 'source_mixed_root': str(mixed),
        'source_val_root': str(source_val), 'source_documents': sources,
        'training_approval_sha256': sha256(output/'training-approval.json'),
        'train_annotations_sha256': approval['train']['annotations_sha256'],
        'train_counts': _coco_counts(derived), 'val_counts': _coco_counts(original_val),
        'source_dataset_counts': dict(Counter(row['source_dataset'] for row in selected)),
        'source_split_counts': dict(Counter(row['source_split'] for row in selected)),
        'recording_counts': dict(Counter(row['recording'] for row in selected)),
        'excluded_frames': [EXCLUDED_FRAME], 'excluded_frame_already_absent_no_second_removal': True,
        'image_equivalence': equivalence, 'equivalence_encoding': 'canonical JSON UTF-8 per unchanged image and associated annotation records',
        'all_selected_rgb_sha256_verified': True, 'original_pixels_labels_and_mixed_run_unchanged': True,
        'val_annotations_bytes_unchanged': True, 'training_plan': training_plan,
        'preregistered_external_test_plan': test_plan,
        'label_limitations': LABEL_LIMITATIONS, 'test_policy': TEST_POLICY,
        'portability': 'RGB and val are symlink references; deployment must keep the referenced mixed data or copy bytes and approve exact relocated image roots.',
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(output/'manifest.json', manifest)
    atomic_json(output/'READY.json', {'format': FORMAT, 'status': 'complete',
        'manifest_sha256': sha256(output/'manifest.json'),
        'training_approval_sha256': manifest['training_approval_sha256'],
        'train_annotations_sha256': approval['train']['annotations_sha256'],
        'val_annotations_sha256': approval['val']['annotations_sha256'],
        'train_counts': manifest['train_counts'], 'planned_steps': epochs*steps})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed-root', type=Path, help='Explicit approved mixed publication root')
    parser.add_argument('--test-root', type=Path, help='Explicit fixed RealSense publication root')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    if not args.verify_only and (args.mixed_root is None or args.test_root is None):
        parser.error('--mixed-root and --test-root are required when preparing a publication')
    result = (validate_publication(args.output_dir) if args.verify_only else
              prepare(mixed_root=args.mixed_root, test_root=args.test_root, output_dir=args.output_dir))
    print(json.dumps({'status': result['status'], 'train': result['train_counts'], 'val': result['val_counts'],
                      'training_plan': result['training_plan'], 'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

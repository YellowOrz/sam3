#!/usr/bin/env python3
"""Freeze a stratified random sample before decoding unmodified reference masks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scripts.prepare_nakehand_test import (
        DIAGNOSTICS, atomic_json, frame_chunk, load_recording, select_frames,
        sha256, side_annotation, source_record, verify_sources,
    )
except ModuleNotFoundError:
    from prepare_nakehand_test import (
        DIAGNOSTICS, atomic_json, frame_chunk, load_recording, select_frames,
        sha256, side_annotation, source_record, verify_sources,
    )


def sample_records(frame_counts, excluded, seed=20260910):
    rng = random.Random(seed)
    result, candidates = [], {}
    for recording, count in sorted(frame_counts.items()):
        view = recording.split('/')[0]
        number = {'nakehandego': 2, 'nakehandexo': 1}[view]
        available = [index for index in range(count) if (recording, index) not in excluded]
        if len(available) < number:
            raise ValueError(f'Insufficient unreviewed candidate frames: {recording}')
        candidates[recording] = available
        for frame in sorted(rng.sample(available, number)):
            result.append({'review_id': f'{len(result)+1:02d}', 'recording_id': recording, 'frame_index': frame})
    if len(result) != 10 or Counter(item['recording_id'].split('/')[0] for item in result) != {'nakehandego': 8, 'nakehandexo': 2}:
        raise ValueError('Expected four ego recordings (2 each) and two exo recordings (1 each)')
    return result, candidates


def freeze_plan(root, test_annotations, audit_path, output, seed):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError('Output must be outside read-only original dataset')
    if output.exists():
        raise FileExistsError(output)
    formal = json.loads(test_annotations.read_text())
    audit = json.loads(audit_path.read_text())
    if len(formal['images']) != 602 or len(audit['samples']) != 42:
        raise ValueError('Expected frozen formal 602 frames and original 42-frame audit')
    formal_pairs = {(im['recording_id'], im['frame_index']) for im in formal['images']}
    audit_pairs = {(im['recording'], im['frame']) for im in audit['samples']}
    excluded = formal_pairs | audit_pairs | set(DIAGNOSTICS)
    metadata_sources, counts = [], {}
    for path in sorted(root.glob('*/*/metadata.json')):
        metadata_sources.append(source_record(path))
        counts[path.parent.relative_to(root).as_posix()] = json.loads(path.read_text())['frame_count']
    if len(counts) != 6 or sum(counts.values()) != 18498:
        raise ValueError('Frozen source inventory changed')
    selected, candidates = sample_records(counts, excluded, seed)
    plan = {'format': 'nakehand-manual-review-plan-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'root': str(root), 'seed': seed, 'random_algorithm': 'random.Random(seed), sorted recording order; random.sample without replacement; selected rows sorted by frame within recording',
            'strata': '2 frames from each of four ego recordings, 1 from each of two exo recordings',
            'selection_before_decode': True, 'no_mask_quality_or_presence_selection': True,
            'empty_masks_retained': True, 'total_source_frames': 18498, 'frame_counts': counts,
            'excluded_sources': {'formal_test_602': source_record(test_annotations), 'original_audit_42': source_record(audit_path)},
            'exclusion_rule': 'Union of all 602 formal external-test frames, all 42 original audit frames (conservative superset of displayed samples), and explicit accepted A/B/C. Never filter on mask presence or predicted quality.',
            'excluded_unique_frames': len(excluded), 'excluded_frames': [{'recording_id': r, 'frame_index': f} for r, f in sorted(excluded)],
            'candidate_indices_by_recording': candidates, 'candidate_count': sum(len(v) for v in candidates.values()),
            'metadata_sources': metadata_sources, 'samples': selected,
            'review_status': 'pending for all ten; previous A/B/C acceptance is not inherited',
            'label_source': 'Existing SAM3 prompted/propagated reference masks; not independent human pixel ground truth'}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'frozen-plan.json', plan)
    print(json.dumps({'plan': str(output/'frozen-plan.json'), 'plan_sha256': sha256(output/'frozen-plan.json'),
                      'excluded_unique_frames': len(excluded), 'candidate_count': plan['candidate_count'], 'samples': selected}, indent=2), flush=True)


def comparison_sheet(rgb, left, right, sample):
    height, width = rgb.shape[:2]
    header = 74
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    sheet = Image.new('RGB', (width*3, height+header), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 5), f"{sample['review_id']} | {sample['recording_id']} | frame {sample['frame_index']}", font=font, fill='black')
    for column, (array, title) in enumerate(zip((rgb, left, right), ('Original RGB', 'LEFT reference mask (>0 union)', 'RIGHT reference mask (>0 union)'))):
        tile = Image.fromarray(array).convert('RGB')
        sheet.paste(tile, (column*width, header))
        draw.text((column*width+10, 32), title, font=font, fill='black')
        if column:
            draw.text((column*width+10, 56), 'SAM3-assisted reference; human review pending', font=small, fill=(75,75,75))
    return sheet


def render_plan(plan_path):
    output = plan_path.resolve().parent
    if (output/'manifest.json').exists() or (output/'samples').exists():
        raise FileExistsError('Refusing to overwrite generated review artifacts')
    plan_sha = sha256(plan_path)
    plan = json.loads(plan_path.read_text())
    verify_sources(plan['metadata_sources'])
    verify_sources(list(plan['excluded_sources'].values()))
    root = Path(plan['root'])
    excluded = {(x['recording_id'],x['frame_index']) for x in plan['excluded_frames']}
    selected, candidates = sample_records(plan['frame_counts'], excluded, plan['seed'])
    if selected != plan['samples'] or candidates != plan['candidate_indices_by_recording']:
        raise ValueError('Frozen random draw no longer reproduces')
    recordings = {name: load_recording(root, root/name/'metadata.json') for name in sorted(plan['frame_counts'])}
    sources = [source for record in recordings.values() for source in record['sources']]
    manifest = {'format': 'nakehand-manual-review-v1', 'frozen_plan_sha256': plan_sha,
                'label_source': plan['label_source'], 'sources': sources, 'samples': [], 'overview_pages': [],
                'not_model_predictions': True, 'no_model_or_gpu_used': True, 'human_review_status': 'pending'}
    for helper in ('prepare_nakehand_manual_review.py', 'prepare_nakehand_test.py', 'audit_nakehand_dataset.py'):
        shutil.copyfile(Path(__file__).resolve().with_name(helper), output/helper)
    sheets = []
    for sample in selected:
        recording = recordings[sample['recording_id']]
        frame = sample['frame_index']
        decoded, pts = {}, {}
        for stream in ('rgb','left','right'):
            values, timestamps = select_frames(recording['paths'][stream], [frame], recording['fps'], recording['width'], recording['height'], stream!='rgb')
            decoded[stream], pts[stream] = values[0], timestamps[0]
        if len(set(pts.values())) != 1:
            raise ValueError('RGB/left/right source PTS mismatch')
        subdir = output/'samples'/sample['review_id']
        subdir.mkdir(parents=True)
        arrays = {'rgb.png': decoded['rgb'], 'left_reference.png': (decoded['left']>0).astype(np.uint8)*255,
                  'right_reference.png': (decoded['right']>0).astype(np.uint8)*255,
                  'left_instance_raw.png': decoded['left'], 'right_instance_raw.png': decoded['right']}
        files = {}
        for name, array in arrays.items():
            path=subdir/name
            Image.fromarray(array).save(path)
            if not np.array_equal(np.asarray(Image.open(path)),array):
                raise ValueError('PNG is not pixel-identical')
            files[name]={'path': str(path.relative_to(output)), 'sha256': sha256(path)}
        sheet=comparison_sheet(decoded['rgb'], arrays['left_reference.png'], arrays['right_reference.png'], sample)
        sheet_path=output/f"review-{sample['review_id']}.png"
        sheet.save(sheet_path)
        sheets.append(sheet)
        references={}
        for side,category in [('left',1),('right',2)]:
            chunk=frame_chunk(recording['sidecars'][side]['metadata'],frame)
            raw=decoded[side]
            allowed={0}|{int(v) for v in chunk['object_id_to_label'].values()}
            if not set(np.unique(raw).tolist()) <= allowed: raise ValueError('Undeclared source instance label')
            references[side]={'source_mask_path':str(recording['paths'][side]), 'source_values':np.unique(raw).tolist(),
                              'source_mapping':chunk, 'positive_pixels':int((raw>0).sum()),
                              'annotation':side_annotation(raw,int(sample['review_id']),category,category)}
        manifest['samples'].append({**sample,'human_review_status':'pending','files':files,'references':references,
                                    'source_rgb_path':str(recording['paths']['rgb']), 'video_pts_seconds':pts,
                                    'source_frame_index':recording['metadata']['frames'][frame]['source_frame_index'],
                                    'source_capture_timestamp_seconds':recording['metadata']['frames'][frame]['timestamp'],
                                    'comparison':{'path':sheet_path.name,'sha256':sha256(sheet_path)}})
        print(f"rendered {sample['review_id']}: {sample['recording_id']} frame={frame}",flush=True)
    for page, start in enumerate((0,5),1):
        width=max(sheet.width for sheet in sheets[start:start+5])
        overview=Image.new('RGB',(width,sum(sheet.height for sheet in sheets[start:start+5])+16*4),'white')
        y=0
        for sheet in sheets[start:start+5]:
            overview.paste(sheet,(0,y)); y+=sheet.height+16
        path=output/f'review-page-{page}.png'
        overview.save(path)
        manifest['overview_pages'].append({'path':path.name,'sha256':sha256(path),'review_ids':[f'{i+1:02d}' for i in range(start,start+5)]})
    verify_sources(sources)
    verify_sources(plan['metadata_sources'])
    verify_sources(list(plan['excluded_sources'].values()))
    if sha256(plan_path)!=plan_sha: raise ValueError('Frozen plan changed during rendering')
    manifest.update(status='complete',sources_unchanged=True,completed_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'READY.json',{'status':'complete','manifest_sha256':sha256(output/'manifest.json'),'frozen_plan_sha256':plan_sha})
    print(json.dumps({'output':str(output),'samples':len(selected),'pages':2,'manifest_sha256':sha256(output/'manifest.json')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/xuzhefeng/Datasets/wanqing_datasets/nakehand'))
    parser.add_argument('--test-annotations',type=Path,default=Path('/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/external-test/nakehand-systematic600-20260910-v2/annotations.json'))
    parser.add_argument('--audit',type=Path,default=Path('/home/zhengyuxi/datasets/sam3-dexycb-bilateral-v1/external-data-audit/nakehand-local-20260910/audit.json'))
    parser.add_argument('--seed',type=int,default=20260910)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan-output-dir',type=Path)
    group.add_argument('--render-plan',type=Path)
    args=parser.parse_args()
    if args.render_plan: render_plan(args.render_plan)
    else: freeze_plan(args.root,args.test_annotations,args.audit,args.plan_output_dir,args.seed)


if __name__=='__main__': main()

"""Publish existing nake RGB/references and uniquely active MANO mesh boxes.

No training, RGB re-encoding, GT-box substitution, or inferred physical camera.
The producer projection is explicitly selected and hashed. Historical split names
are only source locations; this publication is evaluation-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

from scripts.eval import mano_box_factorial as f
from scripts.prepare_nakehand_training import FRAME_COUNTS

EXCLUDED = ('nakehandego/20260907_142020', 0)


def sample_and_exclude(recording, count):
    excluded = [EXCLUDED[1]] if recording == EXCLUDED[0] else []
    return [i for i in range(0, count, 3) if i not in excluded], excluded


def merge_unique(parts):
    """Presence overlap is an ambiguity even when one projected box is invalid."""
    active, prompts, sources = set(), {}, {}
    for path, present, geometry in parts:
        present = set(present)
        if active & present or not set(geometry).issubset(present):
            raise ValueError('Ambiguous overlapping active instances or orphan geometry')
        active |= present
        for i, value in geometry.items():
            if set(value) != {'boxes'} or len(value['boxes']) != 1:
                raise ValueError('Exactly one mesh box per active original frame')
            f.box_cxcywh(value['boxes'][0])
            prompts[i] = value
            sources[i] = str(path)
    return prompts, sources, active


def local_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe relative source path')
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Source escapes dataset root')
    return resolved


def load_projection(path):
    path = path.resolve(strict=True)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('declared_nake_projection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(a):
    source, raw, output = a.source.resolve(), a.raw_root.resolve(), a.output.resolve()
    project = Path(__file__).resolve().parents[2]
    if (output.exists() or output.is_relative_to(project) or output.is_relative_to(source)
            or source.is_relative_to(output) or output.is_relative_to(raw)):
        raise ValueError('Require new external output separate from source and project')
    fingerprints = {}
    def document(path):
        fingerprints[str(path)] = f.sha(path)
        return json.loads(path.read_text())
    ready = document(source / 'READY.json')
    source_manifest = document(source / 'manifest.json')
    source_plan = document(source / 'frozen-plan.json')
    if (ready.get('status') != 'complete' or source_manifest.get('status') != 'complete'
            or not source_manifest.get('sources_unchanged')
            or ready['manifest_sha256'] != fingerprints[str(source / 'manifest.json')]
            or ready['frozen_plan_sha256'] != fingerprints[str(source / 'frozen-plan.json')]
            or {k:v['frame_count'] for k,v in source_plan['recordings'].items()} != FRAME_COUNTS):
        raise ValueError('Require complete six-recording audited source')
    records = {name:{} for name in FRAME_COUNTS}
    for split in ('train', 'val', 'development_holdout'):
        folder = source / split
        split_ready = document(folder / 'READY.json')
        manifest = document(folder / 'manifest.json')
        coco = document(folder / 'annotations.json')
        for field, name in [('ready_sha256','READY.json'),('manifest_sha256','manifest.json'),
                            ('annotations_sha256','annotations.json')]:
            if ready['splits'][split][field] != fingerprints[str(folder / name)]:
                raise ValueError('Source split metadata hash mismatch')
        if split_ready['status'] != 'complete' or manifest['status'] != 'complete':
            raise ValueError('Incomplete source split')
        files = {x['image_id']:x['files'] for x in manifest['image_outputs']}
        if len(files) != len(manifest['image_outputs']) or len(files) != len(coco['images']):
            raise ValueError('Duplicate or missing source image files')
        for im in coco['images']:
            recording, frame = im['recording_id'], im['source_frame_index']
            if recording not in records or frame in records[recording]:
                raise ValueError('Unknown/duplicate recording frame')
            if (im['width'], im['height']) != (640,480):
                raise ValueError('Unexpected source resolution')
            records[recording][frame] = (folder, im, files[im['id']])
    for name, count in FRAME_COUNTS.items():
        if set(records[name]) != set(range(count)):
            raise ValueError('Missing original RGB context')
    producer_hashes = {str(p.relative_to(a.projection_source.parent)): f.sha(p)
                      for p in [a.projection_source,
                                a.projection_source.parent/'process_dataset_videos.py',
                                *sorted((a.projection_source.parent/'common').glob('*.py'))]}
    projection = load_projection(a.projection_source)
    model_hashes = f.code_hashes(a.model_source)
    output.mkdir(parents=True, exist_ok=False)

    def sequence(recording):
        count = FRAME_COUNTS[recording]
        name = recording.replace('/', '__')
        sampled, excluded = sample_and_exclude(recording, count)
        info = dict(width=640, height=480, frame_count=count, fps=30.0)
        args = argparse.Namespace(max_frames=None, focal_length=None, prompt_mode='box',
            box_source='mesh', hand_side='right', prompt_interval=1, box_padding=.05)
        paths = sorted((raw/recording/'MANO_wilor/right_hand').glob('result_mano_*.npz'))
        if not paths:
            raise ValueError('Missing declared MANO files')
        parts, mano_meta, mano_hashes = [], [], {}
        for path in paths:
            mano_hashes[str(path)] = f.sha(path)
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z['frame_indices'], np.arange(count)):
                    raise ValueError('MANO row does not preserve original frame index')
                present = np.asarray(z['has_hand'])
                if present.dtype != np.bool_ or present.shape != (count,):
                    raise ValueError('Invalid MANO presence mask')
                camera = z['vertices'][present] + z['camera_translation'][present,None,:]
                if not np.isfinite(camera).all() or (camera[...,2] <= 0).any():
                    raise ValueError('Invalid active MANO mesh camera coordinates')
                indices = np.flatnonzero(present).tolist()
            geometry, metadata = projection.load_geometry(path, info, args)
            parts.append((path,indices,geometry))
            mano_meta.append(metadata)
        prompts, prompt_sources, active = merge_unique(parts)
        if any(i in prompts for i in excluded):
            raise ValueError('Excluded ambiguous context unexpectedly has geometry')
        files_out = {}
        for sub in ('rgb','right','left'):
            (output/name/sub).mkdir(parents=True)
        for i in range(count):
            folder, im, files = records[recording][i]
            names = [('rgb','rgb')] if i in excluded else [
                ('rgb','rgb'),('right','right_binary'),('left','left_binary')]
            for sub, key in names:
                asset = files[key]
                src = local_file(folder,asset['path'])
                if f.sha(src) != asset['sha256']:
                    raise ValueError('Source PNG changed')
                if sub != 'rgb':
                    with Image.open(src) as img:
                        pixels = np.asarray(img)
                        if pixels.shape != (480,640) or not np.isin(pixels,[0,255]).all():
                            raise ValueError('Reference must be original binary mask')
                rel = f'{name}/{sub}/{i:06d}.png'
                dst = output / rel
                try:
                    os.link(src,dst)
                except OSError:
                    shutil.copyfile(src,dst)
                if f.sha(dst) != asset['sha256']:
                    raise ValueError('Published source bytes differ')
                files_out[rel] = asset['sha256']
        for path,digest in mano_hashes.items():
            if f.sha(path) != digest:
                raise ValueError('MANO changed during publication')
        if paths != sorted((raw/recording/'MANO_wilor/right_hand').glob('result_mano_*.npz')):
            raise ValueError('MANO file set changed during publication')
        print(json.dumps(dict(prepared=name,frames=count,boxes=len(prompts),scored=len(sampled))),flush=True)
        return dict(name=name,recording_id=recording,**info,prompts=prompts,
            prompt_sources=prompt_sources,mano=mano_meta,source_sha256=mano_hashes,
            file_sha256=files_out,sampled_indices=sampled,reference_excluded_indices=excluded,
            render_indices=[i for i in sorted({0,count//4,count//2,count*3//4,count-1}) if i not in excluded],
            active_mano_frames=len(active),source_role='SAM3-assisted external evaluation only')

    with ThreadPoolExecutor(max_workers=2) as pool:
        sequences = list(pool.map(sequence,sorted(FRAME_COUNTS)))
    for path,digest in fingerprints.items():
        if f.sha(path) != digest:
            raise ValueError('Source metadata changed during publication')
    if f.code_hashes(a.model_source) != model_hashes:
        raise ValueError('Frozen model source changed')
    for relative, digest in producer_hashes.items():
        if f.sha(a.projection_source.parent / relative) != digest:
            raise ValueError('Projection producer changed during publication')
    plan = dict(contract=f.NAKE_CONTRACT,sequences=sequences,
        created_at=datetime.now(timezone.utc).isoformat(),base_sha256=a.base_sha256,
        model_source_sha256=model_hashes,projection_source_sha256=producer_hashes,
        tokenizer_sha256=f.sha(a.model_source/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        source_metadata_sha256=fingerprints,preparer_sha256=f.sha(__file__),
        raw_frames=sum(s['frame_count'] for s in sequences),
        scoring_frames=sum(len(s['sampled_indices']) for s in sequences),
        projection_caveat='Virtual focal 12500 matches current producer; not verified physical calibration',
        training_permitted=False,independent_ground_truth=False)
    f.check_inputs(output,plan)
    f.write_json(output/'plan.json',plan)
    print(json.dumps(dict(status='complete',raw_frames=plan['raw_frames'],scoring_frames=plan['scoring_frames'],
                          plan_sha256=f.sha(output/'plan.json'))),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','raw-root','model-source','projection-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--base-sha256',required=True)
    publish(p.parse_args())


if __name__ == '__main__':
    main()

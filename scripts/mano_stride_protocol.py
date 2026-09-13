"""Frozen, all-time stride-three MANO geometry diagnostic (not a blind test).

CPU preparation binds existing published RGB/reference assets to producer NPZs.
Geometry absence never removes a frame or invents a negative reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from scripts import pilot_mano_geometry_prompts as pilot

FORMAT = 'sam3-mano-stride3-oracle-v1'
SIDES = ('left_hand', 'right_hand')
METHODS = tuple(pilot.METHODS)
CORE = METHODS[:4]
CONTRACT = dict(stride=3, offset=0, threshold=.5, score_comparison='strict_gt',
                mask_threshold=.5, boundary_pixels=4, precision='bfloat16', batch_size=1,
                training=False, independent_ground_truth=False,
                projection='producer_virtual_camera_already_mirrored', focal_pixels=12500.,
                geometry_absence='core_same_side_text_fallback_controls_not_evaluated')


def select_images(images):
    groups = defaultdict(list)
    seen = set()
    for image in images:
        if image['id'] in seen:
            raise ValueError('Duplicate image identity')
        seen.add(image['id'])
        groups[image['recording_id']].append(image)
    selected, render = [], []
    for name, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['source_frame_index'])
        if [r['source_frame_index'] for r in rows] != list(range(len(rows))):
            raise ValueError('Require complete zero-based recording, including empty/unknown frames')
        subset = rows[::3]
        selected.extend(subset)
        render.extend(subset[i]['id'] for i in sorted(set(np.linspace(0, len(subset)-1, 3).astype(int))))
    if not selected:
        raise ValueError('No images')
    return selected, render


def validate_arrays(arrays, side, total, width, height, modern):
    if (str(arrays['hand']) != side or int(arrays['total_frames']) != total
            or int(arrays['width']) != width or int(arrays['height']) != height
            or not np.array_equal(arrays['frame_indices'], np.arange(total))):
        raise ValueError('NPZ side/frame/dimension mismatch')
    if not modern:
        return  # Legacy geometry is explicitly unavailable, not silently interpreted.
    if (str(arrays['mask_source']) != side or int(arrays['instance_label']) != 1
            or arrays['has_hand'].shape != (total,) or arrays['has_hand'].dtype != np.bool_):
        raise ValueError('Modern NPZ reference/validity mismatch')
    for key, shape in [('joints', (total,21,3)), ('vertices',(total,778,3)),
                       ('camera_translation',(total,3)), ('bbox_xyxy',(total,4))]:
        if arrays[key].shape != shape:
            raise ValueError(f'NPZ shape mismatch: {key}')
    if 'camera_intrinsics' in arrays:
        raise ValueError('Embedded calibration requires a separately registered projection contract')


def projected_geometry(joints, vertices, translation, width, height):
    result = dict(points=None, mesh_box=None, point_reason=None, box_reason=None,
                  selected_joint_indices=[], joints_pixel=None, mesh_box_xyxy_pixel=None)
    try:
        xy = pilot.project(joints, translation, CONTRACT['focal_pixels'], width, height)
        if xy.shape != (21,2):
            raise ValueError('Expected 21 joints')
        inside = ((xy >= [0,0]) & (xy < [width,height])).all(axis=1)
        result['joints_pixel'] = xy.tolist()
        result['selected_joint_indices'] = np.flatnonzero(inside).tolist()
        if inside.any():
            result['points'] = np.clip((xy[inside]+.5)/[width,height],0,1).astype(np.float32).tolist()
        else:
            result['point_reason'] = 'no_projected_joint_in_image'
    except ValueError:
        result['point_reason'] = 'invalid_joint_projection'
    try:
        xy = pilot.project(vertices, translation, CONTRACT['focal_pixels'], width, height)
        if xy.shape != (778,2):
            raise ValueError('Expected 778 vertices')
        extent = np.r_[xy.min(0),xy.max(0)]
        result['mesh_box_xyxy_pixel'] = extent.tolist()
        result['mesh_box'] = pilot.normalized_box(extent,width,height).tolist()
    except ValueError:
        result['box_reason'] = 'invalid_or_outside_mesh_box'
    return result


def method_condition(method, geometry):
    if method not in METHODS:
        raise ValueError('Unknown method')
    _, points, box, reference = pilot.METHODS[method]
    missing = []
    if points and geometry['points'] is None:
        missing.append(geometry['point_reason'])
    if box and geometry['mesh_box'] is None:
        missing.append(geometry['box_reason'])
    if reference and geometry['reference_box'] is None:
        missing.append('no_positive_provided_reference_box')
    reasons = sorted(set(missing))
    return dict(evaluated=not missing or method in CORE,
                fallback=bool(missing) and method in CORE, reasons=reasons)


def text_for(method, side):
    if side not in SIDES or method not in METHODS:
        raise ValueError('Invalid method/side')
    if method.startswith('visual_'):
        return 'visual'
    if method.startswith('wrong_text_'):
        side = SIDES[1-SIDES.index(side)]
    return side.replace('_',' ')


def area_group(pixels, width, height):
    if pixels is None:
        return 'unknown'
    if pixels == 0:
        return 'empty'
    ratio = pixels/(width*height)
    return 'small_lt_0.5pct' if ratio < .005 else 'medium_lt_2pct' if ratio < .02 else 'large_ge_2pct'


def verify_plan(plan, images, hashes):
    selected, render = select_images(images)
    if (plan.get('format') != FORMAT or plan.get('contract') != CONTRACT
            or plan.get('methods') != list(METHODS)
            or plan.get('publication_sha256') != {Path(k).name:v for k,v in hashes.items()}
            or plan.get('render_ids') != render
            or [r['image_id'] for r in plan['frames']] != [r['id'] for r in selected]):
        raise ValueError('Frozen stride protocol/publication/selection mismatch')
    for item, image in zip(plan['frames'], selected):
        if (item['recording_id'],item['frame_index'],item['width'],item['height']) != (
                image['recording_id'],image['source_frame_index'],image['width'],image['height']):
            raise ValueError('Prepared frame identity mismatch')
        if set(item['geometry']) != set(SIDES):
            raise ValueError('Missing explicit side geometry')
        for g in item['geometry'].values():
            for field,shape in [('mesh_box',(4,)),('reference_box',(4,))]:
                if g[field] is not None:
                    a=np.asarray(g[field])
                    if a.shape!=shape or not np.isfinite(a).all() or np.any((a<0)|(a>1)) or np.any(a[2:]<=0):
                        raise ValueError('Invalid prepared normalized box')
            if g['points'] is not None:
                a=np.asarray(g['points'])
                if (a.ndim!=2 or a.shape[1]!=2 or not 1<=len(a)<=21
                        or not np.isfinite(a).all() or np.any((a<0)|(a>1))):
                    raise ValueError('Invalid prepared points')
            if (g['points'] is None) != bool(g['point_reason']) or (g['mesh_box'] is None) != bool(g['box_reason']):
                raise ValueError('Missing geometry reason mismatch')
    return selected


def prepare(root, source_root, producer, config, base_checkpoint, tokenizer, output):
    from scripts import evaluate_realsense_full as pub
    if output.exists():
        raise ValueError('New prepared-plan path required')
    images, annotations, outputs, frozen, hashes = pub.load_publication(root)
    publication_hashes = {Path(k).name:v for k,v in hashes.items()}
    manifest = json.loads((root/'manifest.json').read_text())
    if Path(frozen['source_root']).resolve()!=source_root.resolve():
        raise ValueError('Original source root differs from publication')
    inputs = {row['path']:row['sha256'] for row in manifest['sources']}
    for path,digest in inputs.items():
        if pilot.sha256(path)!=digest:
            raise ValueError(f'Original source changed since frozen publication: {path}')
    for path in (producer,config,base_checkpoint,tokenizer,Path(__file__),Path(pilot.__file__)):
        inputs[str(path.resolve())] = pilot.sha256(path)
    selected, render = select_images(images)
    moderns, legacy, archives = [], [], {}
    for name,total in frozen['frame_counts'].items():
        directory=source_root/name/'MANO_wilor'
        for side in SIDES:
            modern=directory/side/'result_mano_1.npz'
            old=directory/side/'result_mano.npz'
            path=modern if modern.exists() else old if old.exists() else None
            if path is None:
                archives[name,side]=(None,False,'no_npz')
                continue
            inputs[str(path)]=pilot.sha256(path)
            with np.load(path,allow_pickle=False) as data:
                arrays={key:data[key] for key in data.files}
            validate_arrays(arrays,side,total,640,480,path==modern)
            (moderns if path==modern else legacy).append(str(path))
            archives[name,side]=(arrays,path==modern,'legacy_projection_unverified' if path==old else None)
    by_id={r['id']:i for i,r in enumerate(images)}
    frames=[]; reasons=Counter(); valid_boxes=0
    for image in selected:
        if (image['width'],image['height'])!=(640,480):
            raise ValueError('Projection pre-registration is 640x480 only')
        refs=pub.batch_references(root,images,[by_id[image['id']]],annotations,outputs,hashes)[image['id']]
        item=dict(image_id=image['id'],recording_id=image['recording_id'],frame_index=image['source_frame_index'],
                  width=image['width'],height=image['height'],geometry={})
        for side in SIDES:
            ref=refs[side]; arrays,modern,reason=archives[item['recording_id'],side]; i=item['frame_index']
            reference_box=None if ref is None or not ref.any() else pilot.normalized_box(pilot.tight_box(ref),640,480).tolist()
            if modern and not arrays['has_hand'][i]:
                reason='producer_has_hand_false'
            if reason:
                g=dict(points=None,mesh_box=None,point_reason=reason,box_reason=reason,
                       selected_joint_indices=[],joints_pixel=None,mesh_box_xyxy_pixel=None)
                reasons[reason]+=1
            else:
                if ref is None or not ref.any() or not np.array_equal(pilot.tight_box(ref),arrays['bbox_xyxy'][i]):
                    raise ValueError(f'Current reference and modern NPZ bbox disagree: {item["recording_id"]}/{side}/{i}')
                valid_boxes+=1
                g=projected_geometry(arrays['joints'][i],arrays['vertices'][i],arrays['camera_translation'][i],640,480)
                for key in ('point_reason','box_reason'):
                    if g[key]: reasons[g[key]]+=1
            g.update(reference_box=reference_box,reference_pixels=None if ref is None else int(ref.sum()),
                     reference_provided=ref is not None,reference_quality_flags=pub.quality_flags(image,side))
            item['geometry'][side]=g
        frames.append(item)
        if len(frames)%100==0:
            print(json.dumps(dict(prepared=len(frames),total=len(selected))),flush=True)
    for path,digest in {**inputs,**hashes}.items():
        if pilot.sha256(path)!=digest:
            raise ValueError(f'Input changed during preparation: {path}')
    plan=dict(format=FORMAT,contract=CONTRACT,methods=list(METHODS),frames=frames,render_ids=render,
              publication_sha256=publication_hashes,input_sha256=inputs,
              base_sha256=pilot.sha256(base_checkpoint),tokenizer_sha256=pilot.sha256(tokenizer),
              modern_npz=moderns,legacy_npz_geometry_unavailable=legacy,
              audit=dict(selected_frames=len(frames),side_queries=2*len(frames),verified_modern_reference_boxes=valid_boxes,
                         geometry_unavailable_reasons=dict(reasons),
                         frames_per_recording=dict(Counter(r['recording_id'] for r in frames))),
              note='SAM3-assisted references and reference-assisted MANO: location-oracle diagnostic, not blind generalization')
    verify_plan(plan,images,{str(root/k):v for k,v in publication_hashes.items()})
    pilot.write_json(output,plan)
    print(json.dumps(plan['audit']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source-root','producer','config','base-checkpoint','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    prepare(a.root,a.source_root,a.producer,a.config,a.base_checkpoint,a.tokenizer,a.output)

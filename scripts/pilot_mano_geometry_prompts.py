"""Frozen detector-geometry oracle diagnostic; never a blind MANO benchmark.

Projection requires an explicit, pre-registered producer contract. No MANO fit,
tracker, training, reference-driven point filtering, or threshold search occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw


METHODS = {
    'text': ('left hand', False, False, False),
    'text_points': ('left hand', True, False, False),
    'text_mesh_box': ('left hand', False, True, False),
    'text_points_mesh_box': ('left hand', True, True, False),
    'text_reference_box': ('left hand', False, False, True),
    'visual_points': ('visual', True, False, False),
    'visual_mesh_box': ('visual', False, True, False),
    'visual_points_mesh_box': ('visual', True, True, False),
    'wrong_text_points_mesh_box': ('right hand', True, True, False),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def project(points, translation, focal, width, height):
    points, translation = np.asarray(points), np.asarray(translation)
    if (points.ndim != 2 or points.shape[1] != 3 or translation.shape != (3,)
            or not np.isfinite(points).all() or not np.isfinite(translation).all()
            or not np.isfinite(focal) or focal <= 0 or width <= 0 or height <= 0):
        raise ValueError('Invalid explicit projection inputs')
    camera = points + translation
    if np.any(camera[:, 2] <= 0):
        raise ValueError('Projection behind or on camera plane')
    return camera[:, :2] / camera[:, 2:] * focal + [width / 2, height / 2]


def normalized_box(xyxy, width, height):
    box = np.asarray(xyxy, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or width <= 0 or height <= 0:
        raise ValueError('Invalid box')
    box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    if not (box[2] > box[0] and box[3] > box[1]):
        raise ValueError('Empty box after clipping')
    return np.array([(box[0]+box[2])/2/width, (box[1]+box[3])/2/height,
                     (box[2]-box[0])/width, (box[3]-box[1])/height], dtype=np.float32)


def prompt_coordinates(joints_xy, vertices_xy, width, height):
    joints_xy, vertices_xy = np.asarray(joints_xy), np.asarray(vertices_xy)
    if (joints_xy.shape != (21, 2) or vertices_xy.shape != (778, 2)
            or not np.isfinite(joints_xy).all() or not np.isfinite(vertices_xy).all()):
        raise ValueError('Require finite 21 joints / 778 vertices')
    in_image = ((joints_xy >= [0, 0]) & (joints_xy < [width, height])).all(axis=1)
    if not in_image.any():
        raise ValueError('No projected point in image')
    # Pixel-centre offset follows grid_sample(align_corners=False). No mask lookup.
    points = ((joints_xy[in_image] + .5) / [width, height]).astype(np.float32)
    points = np.clip(points, 0, 1)
    extent = np.r_[vertices_xy.min(axis=0), vertices_xy.max(axis=0)]
    return points, normalized_box(extent, width, height), np.flatnonzero(in_image)


def tight_box(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Selected reference is empty')
    return np.array([x.min(), y.min(), x.max()+1, y.max()+1], dtype=np.float32)


def decode_selected(path, indices, total, width, height, rgb):
    channels = 3 if rgb else 1
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
        '-map', '0:v:0', '-threads', '1', '-f', 'rawvideo', '-pix_fmt',
        'rgb24' if rgb else 'gray', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = {}
    try:
        for i in range(total):
            raw = process.stdout.read(width*height*channels)
            if len(raw) != width*height*channels:
                raise ValueError(f'Video truncated at frame {i}')
            if i in indices:
                shape = (height, width, 3) if rgb else (height, width)
                output[i] = np.frombuffer(raw, np.uint8).reshape(shape).copy()
        if process.stdout.read(1):
            raise ValueError('Video has extra frames')
        error = process.stderr.read()
        if process.wait(timeout=30) or error:
            raise ValueError(f'Video decode failed: {error[-1000:]!r}')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    return output


def mask_metrics(prediction, reference):
    from scipy.ndimage import binary_erosion
    p, r = np.asarray(prediction, bool), np.asarray(reference, bool)
    if p.shape != r.shape or p.ndim != 2 or not r.any():
        raise ValueError('This pilot requires matching positive references')
    def boundary(mask):
        return mask & ~binary_erosion(mask, structure=np.ones((3, 3)), iterations=4, border_value=0)
    pb, rb = boundary(p), boundary(r)
    return dict(dice=float(2*(p&r).sum()/(p.sum()+r.sum())),
        boundary_iou_4px=float((pb&rb).sum()/(pb|rb).sum()),
        false_negative=not bool(p.any()), prediction_pixels=int(p.sum()))


def geometry_prompt(method, points, mesh_box, reference_box, device):
    import torch
    from sam3.model.geometry_encoders import Prompt
    _, use_points, use_mesh, use_reference = METHODS[method]
    kwargs = {}
    if use_points:
        kwargs['point_embeddings'] = torch.tensor(points, device=device).view(-1, 1, 2)
    if use_mesh or use_reference:
        kwargs['box_embeddings'] = torch.tensor(reference_box if use_reference else mesh_box,
                                               device=device).view(1, 1, 4)
    return Prompt(**kwargs) if kwargs else None


def render_panel(path, rgb, reference, predictions, records, names):
    cell_w, cell_h, title_h = 320, 240, 40
    tiles = [('Original RGB', Image.fromarray(rgb)), ('SAM3 assisted reference', Image.fromarray(reference*255))]
    for name in names:
        tiles.append((f'{name}\nDice {records[name]["dice"]:.4f}', Image.fromarray(predictions[name]*255)))
    canvas = Image.new('RGB', (cell_w*len(tiles), cell_h+title_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, tile) in enumerate(tiles):
        draw.text((index*cell_w+5, 3), label, fill='black')
        canvas.paste(tile.convert('RGB').resize((cell_w, cell_h), Image.Resampling.NEAREST),
                     (index*cell_w, title_h))
    canvas.save(path)


def run(plan_path, output):
    import torch
    import torch.nn.functional as functional
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    plan = json.loads(plan_path.read_text())
    if (plan.get('format') != 'sam3-mano-oracle-pilot-v1' or plan.get('training') is not False
            or plan.get('independent_ground_truth') is not False
            or plan.get('projection_contract') != 'producer_virtual_camera_already_mirrored'
            or plan.get('methods') != list(METHODS)):
        raise ValueError('Explicit frozen oracle plan required')
    if output.exists():
        raise ValueError('Use new output path')
    source = {key: Path(plan[key]) for key in ('npz', 'rgb', 'reference', 'producer', 'producer_config',
                                                   'base_checkpoint', 'tokenizer')}
    hashes = {str(path): sha256(path) for path in source.values()}
    hashes[str(plan_path)] = sha256(plan_path)
    with np.load(source['npz'], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    total, width, height = (int(arrays[k]) for k in ('total_frames', 'width', 'height'))
    if (str(arrays['hand']) != 'left_hand' or str(arrays['mask_source']) != 'left_hand'
            or not np.array_equal(arrays['frame_indices'], np.arange(total))
            or arrays['has_hand'].shape != (total,) or arrays['has_hand'].dtype != np.bool_):
        raise ValueError('Unexpected frame/side/validity contract')
    selected = plan['frame_indices']
    if len(selected) != 12 or len(set(selected)) != 12 or not all(arrays['has_hand'][i] for i in selected):
        raise ValueError('Require twelve pre-registered valid frames')
    valid = np.flatnonzero(arrays['has_hand'])
    if selected != valid[np.linspace(0, len(valid)-1, 12).astype(int)].tolist():
        raise ValueError('Selection must be uniform over declared valid indices, not outcomes')
    rgbs = decode_selected(source['rgb'], set(selected), total, width, height, True)
    masks = decode_selected(source['reference'], set(selected), total, width, height, False)
    output.mkdir(parents=True, exist_ok=False)
    geometry = {}
    for i in selected:
        reference = masks[i] == int(arrays['instance_label'])
        if not np.array_equal(tight_box(reference), arrays['bbox_xyxy'][i]):
            raise ValueError(f'Stored bbox differs from current reference at {i}; stop, no correction')
        projected = [project(arrays[key][i], arrays['camera_translation'][i], plan['focal_pixels'], width, height)
                     for key in ('joints', 'vertices')]
        points, box, ids = prompt_coordinates(*projected, width, height)
        geometry[i] = (points, box, normalized_box(arrays['bbox_xyxy'][i], width, height))
        folder = output / f'frame-{i:06d}'
        folder.mkdir()
        Image.fromarray(rgbs[i]).save(folder/'rgb.png')
        Image.fromarray(reference.astype(np.uint8)*255).save(folder/'reference.png')
        overlay = Image.fromarray(rgbs[i]); draw = ImageDraw.Draw(overlay)
        for number, xy in enumerate(projected[0]):
            draw.ellipse((xy[0]-2, xy[1]-2, xy[0]+2, xy[1]+2), fill='cyan')
            draw.text(tuple(xy), str(number), fill='white')
        mesh_xyxy = np.r_[projected[1].min(0), projected[1].max(0)]
        draw.rectangle(tuple(mesh_xyxy), outline='cyan', width=2)
        overlay.save(folder/'prompt_locations_only.png')
        write_json(folder/'geometry.json', dict(frame_index=i, joints_pixel=projected[0].tolist(),
            mesh_box_xyxy_pixel=mesh_xyxy.tolist(), selected_joint_indices=ids.tolist(),
            point_count=len(points), points_normalized=points.tolist(), mesh_box_cxcywh=box.tolist(),
            stored_reference_box_xyxy=arrays['bbox_xyxy'][i].tolist(),
            stored_reference_box_exactly_matches_current_mask=True,
            projected_points_inside_reference_diagnostic_only=int(sum(reference[int(y), int(x)]
                for x,y in projected[0] if 0 <= x < width and 0 <= y < height))))
    write_json(output/'preflight.json', dict(status='passed', hashes=hashes, frames=selected,
        decoded_rgb_and_reference_frames=total, shape=[height,width],
        projection_is_virtual_not_calibrated=True, source_run_config_not_embedded_in_npz=True))
    torch.cuda.set_per_process_memory_fraction(plan['gpu_memory_fraction'])
    torch.manual_seed(123)
    model = build_sam3_image_model(checkpoint_path=str(source['base_checkpoint']),
        bpe_path=str(source['tokenizer']), load_from_HF=False, device='cuda', eval_mode=True,
        enable_segmentation=True, enable_inst_interactivity=False, text_encoder_type='ve')
    model.eval().requires_grad_(False)
    versions = [(p, p._version) for p in model.parameters()]
    geometry_calls = []
    def hook(module, args, kwargs, result):
        prompt = kwargs['geo_prompt']
        geometry_calls.append(dict(points=0 if prompt.point_embeddings is None else len(prompt.point_embeddings),
            boxes=0 if prompt.box_embeddings is None else len(prompt.box_embeddings),
            encoded_shape=list(result[0].shape)))
    handle = model.geometry_encoder.register_forward_hook(hook, with_kwargs=True)
    processor = Sam3Processor(model, device='cuda', confidence_threshold=.5)
    rows = []; started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        text_outputs = {text: model.backbone.forward_text([text], device='cuda')
                        for text in ('left hand', 'right hand', 'visual')}
        for i in selected:
            image_state = processor.set_image(Image.fromarray(rgbs[i]))
            reference = masks[i] == int(arrays['instance_label'])
            predictions, by_method = {}, {}
            for name, (text, _, _, _) in METHODS.items():
                geo = geometry_prompt(name, *geometry[i], device='cuda') or model._get_dummy_prompt()
                backbone = dict(image_state['backbone_out']); backbone.update(text_outputs[text])
                out = model.forward_grounding(backbone_out=backbone, find_input=processor.find_stage,
                    geometric_prompt=geo, find_target=None)
                scores = (out['pred_logits'].sigmoid() * out['presence_logit_dec'].sigmoid().unsqueeze(1)).flatten()
                index = int(scores.argmax()); score = float(scores[index])
                candidate = functional.interpolate(out['pred_masks'][0,index][None,None].float(),
                    (height,width), mode='bilinear', align_corners=False)[0,0].sigmoid().gt(.5).cpu().numpy()
                prediction = candidate if score > .5 else np.zeros_like(candidate)
                metric = mask_metrics(prediction, reference)
                row = dict(frame_index=i, method=name, score=score, candidate_index=index,
                    candidate_dice=mask_metrics(candidate, reference)['dice'], reference_pixels=int(reference.sum()),
                    geometry_call=geometry_calls[-1], **metric)
                expected = geometry[i][0].shape[0] if METHODS[name][1] else 0
                if geometry_calls[-1]['points'] != expected:
                    raise RuntimeError('Detector geometry did not receive points')
                rows.append(row); by_method[name] = row; predictions[name] = prediction.astype(np.uint8)
                folder = output/f'frame-{i:06d}'
                Image.fromarray(prediction.astype(np.uint8)*255).save(folder/f'{name}.png')
                Image.fromarray(candidate.astype(np.uint8)*255).save(folder/f'{name}-candidate.png')
                del out
            render_panel(output/f'frame-{i:06d}'/'comparison.png', rgbs[i], reference.astype(np.uint8),
                         predictions, by_method, list(METHODS)[:4])
            render_panel(output/f'frame-{i:06d}'/'controls.png', rgbs[i], reference.astype(np.uint8),
                         predictions, by_method, list(METHODS)[4:])
            print(json.dumps(dict(frame=i, completed=len(rows), total=len(selected)*len(METHODS))), flush=True)
    handle.remove()
    if any(p._version != version or p.requires_grad for p,version in versions):
        raise RuntimeError('Frozen parameter contract violated')
    for path,digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError('Input changed during pilot')
    if len(rows) != len(selected)*len(METHODS):
        raise RuntimeError('Incomplete coverage')
    write_json(output/'records.json', rows)
    summary = {name: dict(queries=len(selected), mean_dice=float(np.mean([r['dice'] for r in rows if r['method']==name])),
        mean_boundary_iou_4px=float(np.mean([r['boundary_iou_4px'] for r in rows if r['method']==name])),
        false_negative=sum(r['false_negative'] for r in rows if r['method']==name),
        mean_candidate_dice=float(np.mean([r['candidate_dice'] for r in rows if r['method']==name]))) for name in METHODS}
    write_json(output/'summary.json', dict(status='complete', frames=len(selected), queries=len(rows),
        all_parameters_frozen_and_version_unchanged=True, geometry_calls=len(geometry_calls),
        elapsed_seconds=time.monotonic()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        metrics=summary, protocol=plan, source_hashes=hashes,
        note='GT-assisted location oracle; not independent generalization or MANO 3D accuracy'))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan.resolve(), args.output.resolve())

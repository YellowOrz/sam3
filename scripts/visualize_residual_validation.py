"""CPU-only, fixed-seed Dex validation diagnostics from existing prediction RLE.

No model, CUDA, new inference, test data, or metric-based image selection.
The selection plan is committed before prediction records are opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

SIDES = ('left_hand', 'right_hand')
SEED = 20260911
FORMAT = 'sam3-residual-validation-visual-diagnostic-v1'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_json(raw):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    return json.loads(raw, parse_constant=invalid)


def select_ids(image_ids, *, seed=SEED, count=12):
    ids = sorted(image_ids)
    if len(set(ids)) != len(ids) or any(type(value) is not int for value in ids):
        raise ValueError('Image IDs must be unique integers')
    if type(count) is not int or not 1 <= count <= min(24, len(ids)):
        raise ValueError('Select 1..24 available images')
    return sorted(random.Random(seed).sample(ids, count))


def decode_rle(rle, shape):
    if rle is None:
        return np.zeros(shape, dtype=bool)
    if rle.get('size') != list(shape):
        raise ValueError('RLE shape differs from original image')
    if isinstance(rle.get('counts'), list):
        rle = mask_utils.frPyObjects(rle, *shape)
    mask = mask_utils.decode(rle)
    if mask.shape != tuple(shape):
        raise ValueError('Require one two-dimensional semantic mask')
    return mask.astype(bool)


def load_records(directory, summary, images, selected, annotation_sha, fingerprints):
    expected = {(image_id, side) for image_id in images for side in SIDES}
    index = {image_id: number for number, image_id in enumerate(sorted(images))}
    observed, kept = set(), {}
    paths = [directory / f'rank-{rank}.jsonl' for rank in range(summary['world_size'])]
    if set(directory.glob('rank-*.jsonl')) != set(paths):
        raise ValueError('Missing or extra rank record files')
    for rank, path in enumerate(paths):
        raw = path.read_bytes()
        fingerprints[str(path)] = sha256(raw)
        for line in raw.splitlines():
            row = parse_json(line)
            key = row['image_id'], row['prompt_key']
            if key not in expected or key in observed:
                raise ValueError('Duplicate or unexpected validation identity')
            image = images[key[0]]
            if (row.get('dataset_role') != 'validation' or row.get('identity_verified') is not True
                    or row.get('rank') != rank or row.get('dataset_index') != index[key[0]]
                    or row.get('file_name') != image['file_name']
                    or row.get('original_size') != [image['height'], image['width']]
                    or row.get('provenance', {}).get('annotations_sha256') != annotation_sha):
                raise ValueError('Validation identity/provenance mismatch')
            score = row['top_confidence']
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detection score')
            if type(row['detected']) is not bool or row['detected'] != (score >= .5):
                raise ValueError('Detection gate differs from fixed threshold')
            observed.add(key)
            if key[0] in selected:
                kept[key] = row
    if observed != expected or summary['verified_query_records'] != len(expected):
        raise ValueError('Incomplete validation identity coverage')
    return kept


def save_png(path, array, artifacts):
    Image.fromarray(array).save(path)
    with Image.open(path) as saved:
        actual = np.asarray(saved)
        if actual.shape != array.shape or not np.array_equal(actual, array):
            raise RuntimeError(f'PNG round-trip differs: {path}')
    artifacts.append({'path': str(path), 'sha256': sha256(path.read_bytes()),
                      'shape': list(array.shape), 'pixel_exact_verified': True})


def visualize(*, data_root, baseline_dir, comparison_dir, output_dir, seed=SEED, count=12):
    data_root, baseline_dir, comparison_dir = (Path(p).resolve() for p in (data_root, baseline_dir, comparison_dir))
    output = Path(output_dir).resolve()
    forbidden = [data_root, baseline_dir, comparison_dir, Path(__file__).resolve().parents[1]]
    if output.exists() or any(output.is_relative_to(root) for root in forbidden):
        raise ValueError('Require a new external output directory outside all source roots')
    fingerprints = {}
    annotation_path = data_root / 'annotations.json'
    raw = annotation_path.read_bytes()
    annotation_sha = sha256(raw)
    fingerprints[str(annotation_path)] = annotation_sha
    data = parse_json(raw)
    if data.get('info', {}).get('dataset_role') != 'val' or data.get('categories') != [
            {'id': 1, 'name': 'left_hand'}, {'id': 2, 'name': 'right_hand'}]:
        raise ValueError('Require original approved two-side validation annotations')
    images = {image['id']: image for image in data['images']}
    if len(images) != len(data['images']) or any(image.get('source_dataset') != 'dexycb' for image in images.values()):
        raise ValueError('Require unique Dex validation images, never RealSense test')
    selected = select_ids(images, seed=seed, count=count)
    summaries = []
    for directory in (baseline_dir, comparison_dir):
        path = directory / 'summary.json'
        raw = path.read_bytes()
        fingerprints[str(path)] = sha256(raw)
        summary = parse_json(raw)
        if (summary.get('dataset_role') != 'validation' or summary.get('scope') != 'dexycb_val'
                or summary.get('annotations_sha256') != annotation_sha
                or summary.get('detection_threshold') != .5 or summary.get('mask_threshold') != .5
                or summary.get('metrics', {}).get('images') != len(images)
                or summary.get('metrics', {}).get('queries') != 2*len(images)):
            raise ValueError('Require matching completed full Dex validations and fixed thresholds')
        summaries.append(summary)
    steps = [summary['global_step'] for summary in summaries]
    if steps[0] != 0 or type(steps[1]) is not int or steps[1] <= 0:
        raise ValueError('Require actual step0 baseline and a later validated checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    plan = {'format': FORMAT, 'scope': 'validation_diagnostic', 'seed': seed,
            'selection': 'random.Random(seed).sample(sorted(full_validation_ids), count), then sort selected IDs',
            'selection_uses_metrics_or_predictions': False, 'image_ids': selected,
            'full_validation_image_count': len(images), 'annotations_sha256': annotation_sha,
            'steps': steps, 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    # Save before reading any prediction records. Keep plan on any later failure.
    (output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    records = [load_records(directory, summary, images, set(selected), annotation_sha, fingerprints)
               for directory, summary in zip((baseline_dir, comparison_dir), summaries)]
    annotations = {}
    for row in data['annotations']:
        key = row['image_id'], row['category_id']
        if key in annotations or key[0] not in images or key[1] not in (1, 2):
            raise ValueError('Duplicate/unknown reference annotation')
        annotations[key] = row
    artifacts, visual_rows, comparisons = [], [], []
    for image_id in selected:
        image = images[image_id]
        directory = output / f'image-{image_id:08d}'
        directory.mkdir()
        source_path = data_root / image['file_name']
        source_bytes = source_path.read_bytes()
        fingerprints[str(source_path)] = sha256(source_bytes)
        if image.get('source_rgb_sha256') and image['source_rgb_sha256'] != sha256(source_bytes):
            raise ValueError('Source RGB differs from approved image fingerprint')
        with Image.open(source_path) as loaded:
            rgb = np.asarray(loaded.convert('RGB')).copy()
        shape = image['height'], image['width']
        if rgb.shape != (*shape, 3):
            raise ValueError('Source RGB dimensions changed')
        save_png(directory/'rgb.png', rgb, artifacts)
        panels, titles = [rgb], [f'Dex validation RGB | ID {image_id}']
        refs = {}
        for category, side in enumerate(SIDES, 1):
            row = annotations.get((image_id, category))
            reference = decode_rle(row['segmentation'] if row else None, shape)
            refs[side] = reference
            pixels = reference.astype(np.uint8)*255
            save_png(directory/f'{side}__reference.png', pixels, artifacts)
            panels.append(np.repeat(pixels[..., None], 3, axis=2))
            titles.append(f'Dex original reference | {side}')
        per_query = []
        for model_index, step in enumerate(steps):
            for side in SIDES:
                row = records[model_index][image_id, side]
                candidate = decode_rle(row['prediction_rle'], shape)
                displayed = candidate if row['detected'] else np.zeros_like(candidate)
                reference = refs[side]
                dice = (2*int((candidate & reference).sum()) / (int(candidate.sum())+int(reference.sum()))) if reference.any() else None
                miss = dice if row['detected'] else 0.
                miss = miss if reference.any() else None
                for key, value in [('candidate_dice', dice), ('miss_zero_dice', miss)]:
                    if (value is None) != (row[key] is None) or (value is not None and abs(value-row[key]) > 1e-10):
                        raise ValueError('Selected query Dice differs from RLE/reference')
                if row['reference_present'] != bool(reference.any()):
                    raise ValueError('Reference presence differs from validation record')
                prefix = f'step-{step:08d}__{side}'
                save_png(directory/f'{prefix}__candidate.png', candidate.astype(np.uint8)*255, artifacts)
                pixels = displayed.astype(np.uint8)*255
                save_png(directory/f'{prefix}__detected.png', pixels, artifacts)
                panels.append(np.repeat(pixels[..., None], 3, axis=2))
                dice_text = 'absent reference' if miss is None else f'miss-zero Dice={miss:.3f}'
                titles.append(f'step {step} | {side} | score={row["top_confidence"]:.3f}\n'
                              f'det={int(row["detected"])} | {dice_text}')
                per_query.append({'step': step, 'side': side, 'detected': row['detected'],
                                  'score': row['top_confidence'], 'candidate_dice': dice,
                                  'miss_zero_dice': miss, 'candidate_pixels': int(candidate.sum()),
                                  'detected_pixels': int(displayed.sum())})
        height, width = shape
        canvas = Image.new('RGB', (width*7, height+52), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, (pixels, title) in enumerate(zip(panels, titles)):
            canvas.paste(Image.fromarray(pixels), (column*width, 52))
            draw.multiline_text((column*width+6, 5), title, fill='black', spacing=3)
        path = directory/'comparison.png'
        save_png(path, np.asarray(canvas), artifacts)
        with Image.open(path) as saved:
            pixels = np.asarray(saved)
            if any(not np.array_equal(pixels[52:, column*width:(column+1)*width], panel)
                   for column, panel in enumerate(panels)):
                raise RuntimeError('Composite content differs from source/RLE')
        comparisons.append((image_id, canvas.copy()))
        visual_rows.append({'image_id': image_id, 'file_name': image['file_name'],
                           'comparison': str(path.relative_to(output)), 'queries': per_query})
    pages = []
    for start in range(0, len(comparisons), 4):
        group = comparisons[start:start+4]
        cell_width, cell_height = 1792, 280
        page = Image.new('RGB', (cell_width*2, cell_height*2), 'white')
        for index, (image_id, original) in enumerate(group):
            thumbnail = original.copy()
            thumbnail.thumbnail((cell_width, cell_height-28), Image.Resampling.LANCZOS)
            x, y = (index%2)*cell_width, (index//2)*cell_height
            page.paste(thumbnail, (x, y+28))
            ImageDraw.Draw(page).text((x+8, y+5), f'ID {image_id} | preview only; inspect full-resolution individual PNG', fill='black')
        name = f'browse-{len(pages)+1:02d}.png'
        save_png(output/name, np.asarray(page), artifacts)
        pages.append(name)
    for path, expected in fingerprints.items():
        if sha256(Path(path).read_bytes()) != expected:
            raise RuntimeError('Input changed during rendering; incomplete output retained')
    manifest = {**plan, 'status': 'complete', 'input_sha256': fingerprints,
                'source_files_verified_unchanged': True, 'visuals': visual_rows, 'browse_pages': pages,
                'artifacts': artifacts, 'png_vs_rle_pixel_verification': {'status': 'pass',
                    'rgb': len(selected), 'references': 2*len(selected), 'candidate': 4*len(selected),
                    'detected': 4*len(selected), 'composite_content_panels': 7*len(selected)},
                'limitations': ['Fixed random validation diagnostics, not test results or representative accuracy estimates.',
                    'Not selected by improvement; do not infer gains from a chosen example.',
                    'Dex references retain original rendered-label/minimum-area policy; not reannotated here.',
                    'Browse pages are resized previews; only full-resolution masks are pixel-exact to RLE.',
                    'No new inference, GPU use, training, or RealSense access.']}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# Dex validation：baseline／epoch 对照诊断（固定随机样本）', '',
             f'范围：validation_diagnostic；step {steps[0]} → {steps[1]}；seed={seed}；固定{len(selected)}图。', '',
             '先写plan.json再读预测，按完整val ID随机抽样，不按提升/退化指标挑图。不运行新推理，不访问RealSense。', '',
             f'选图ID：{selected}。验证标注SHA256：{annotation_sha}。', '',
             '每图七列：RGB、Dex原始左参考、右参考、baseline左/右检测、后续step左/右检测。黑白分栏，无彩色叠加。', '',
             'candidate文件另外保留；detected=score>=0.5，低于阈值显示空mask。手腕/手指形态与漏检应分别检查，不能以零散例图宣布提升。', '',
             '[冻结选图计划](plan.json) · [记录SHA、PNG校验与完整清单](manifest.json)', '',
             '全尺寸PNG已与源RGB、左右参考RLE、预测RLE及检测门控逐像素核验；浏览页仅缩略，非逐像素参考。', '']
    for name in pages:
        lines += [f'![两行浏览页]({name})', '']
    lines += ['## 全部固定样本', '', '| image ID | 全尺寸七列图 |', '|---:|---|']
    lines += [f'| {row["image_id"]} | [打开]({row["comparison"]}) |' for row in visual_rows]
    lines += ['', '仅为固定验证集诊断，不是新的test结果，不改变19:00按完整Dex验证选点再做固定RealSense测试的规则。', '']
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'baseline-dir', 'comparison-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args(argv)
    result = visualize(**vars(args))
    print(json.dumps({'status': result['status'], 'image_ids': result['image_ids'],
                      'verification': result['png_vs_rle_pixel_verification'], 'pages': result['browse_pages']}))


if __name__ == '__main__':
    main()

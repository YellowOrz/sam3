"""Independent CPU recomputation from saved RLE and unchanged reference PNGs.

This does not run a model or reuse the inference metric implementation.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from scripts import evaluate_realsense_full as publication
from scripts.compare_realsense_full import decode_rle


def actual_metrics(mask, reference):
    if reference is None or not reference.any():
        return None, None
    intersection = int((mask & reference).sum())
    dice = 2 * intersection / (int(mask.sum()) + int(reference.sum()))
    boundary = lambda m: m & ~binary_erosion(m, structure=np.ones((3,3),bool), iterations=4, border_value=0)
    a,b = boundary(mask),boundary(reference)
    return dice, int((a&b).sum()) / int((a|b).sum())


def audit(data_root, run, output):
    if output.exists():
        raise ValueError('New audit output required')
    root = data_root.resolve()
    images, annotations, outputs, _, hashes = publication.load_publication(root)
    by_id = {r['id']:i for i,r in enumerate(images)}
    summary = json.loads((run/'summary.json').read_text())
    spec = summary['protocol']
    records_path = run/'records.jsonl'
    if (summary['status']!='complete' or spec['format']!='sam3-hand-routes-v2'
            or summary['records_sha256'] != publication.shared.sha256(records_path)
            or spec['annotations_sha256'] != publication.shared.sha256(root/'annotations.json')):
        raise ValueError('Incomplete, changed, or non-v2 evaluation')
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    expected={(i,s) for i in spec['image_ids'] for s in publication.SIDES}
    if len(records)!=len(expected) or {(r['image_id'],r['prompt_key']) for r in records}!=expected:
        raise ValueError('Query coverage mismatch')
    checked=0
    for r in records:
        index=by_id[r['image_id']];im=images[index];side=r['prompt_key']
        if (r['recording_id'],r['source_frame_index'],r['reference_quality_flags'],r['reference_provided']) != (
                im['recording_id'],im['source_frame_index'],publication.quality_flags(im,side),im['reference_provided'][side]):
            raise ValueError('Reference identity/quality mismatch')
        refs=publication.batch_references(root,images,[index],annotations,outputs,hashes)[im['id']]
        mask=decode_rle(r['prediction_rle'],(im['height'],im['width']))
        if spec['evaluation_layer']=='image_ablation':
            detected=r['top_confidence']>=.5
            if detected!=r['detected']:
                raise ValueError('Threshold mismatch')
            if not detected: mask=np.zeros_like(mask)
        elif spec['evaluation_layer']!='video_system':
            raise ValueError('Unknown evaluation layer')
        if int(mask.sum())!=r['detected_mask_pixels']:
            raise ValueError('Actual-output pixel count mismatch')
        if refs[side] is not None and int(refs[side].sum())!=r['reference_pixels']:
            raise ValueError('Reference pixel count mismatch')
        dice,boundary=actual_metrics(mask,refs[side])
        for field,value in [('miss_zero_dice',dice),('miss_zero_boundary_iou_4px',boundary)]:
            supplied=r[field]
            if value is None:
                if supplied is not None: raise ValueError('Unknown/empty reference must not have positive-only score')
            elif supplied is None or not math.isclose(value,supplied,rel_tol=1e-9,abs_tol=1e-9):
                raise ValueError(f'Independent RLE metric mismatch: {field}')
        checked+=1
    for path,digest in hashes.items():
        if publication.shared.sha256(Path(path))!=digest:
            raise ValueError('Publication changed during audit')
    publication.shared.atomic_write_json(output,dict(status='complete',checked_queries=checked,
        evaluation_layer=spec['evaluation_layer'],records_sha256=summary['records_sha256'],
        reference_role=spec['reference_role'],note='CPU RLE/PNG Dice/boundary recomputation; not independent annotation certification'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit(args.data_root,args.run,args.output)

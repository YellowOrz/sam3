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


def thresholded_output(mask, score, detected, layer):
    if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
            or type(detected) is not bool):
        raise ValueError('Invalid output confidence/detection flag')
    if layer == 'image_ablation':
        if detected != (score >= .5):
            raise ValueError('Threshold mismatch')
        return mask if detected else np.zeros_like(mask)
    if layer != 'video_system' or detected != bool(mask.any()):
        raise ValueError('Invalid video actual-output presence or evaluation layer')
    return mask  # Video predictor already filtered; never apply a second threshold.


def audit(data_root, run, output):
    if output.exists():
        raise ValueError('New audit output required')
    auditor_hash = publication.shared.sha256(Path(__file__))
    root = data_root.resolve()
    images, annotations, outputs, _, hashes = publication.load_publication(root)
    by_id = {r['id']:i for i,r in enumerate(images)}
    summary_path = run/'summary.json'
    summary_hash = publication.shared.sha256(summary_path)
    summary = json.loads(summary_path.read_text())
    spec = summary['protocol']
    records_path = run/'records.jsonl'
    if (summary['status']!='complete' or not summary.get('actual_complete_query_coverage_verified')
            or spec['format']!='sam3-hand-routes-v2' or spec['threshold'] != .5
            or spec['mask_threshold'] != .5 or spec['boundary_pixels'] != 4
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
        mask = thresholded_output(mask, r['top_confidence'], r['detected'], spec['evaluation_layer'])
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
    if (publication.shared.sha256(summary_path) != summary_hash
            or publication.shared.sha256(records_path) != summary['records_sha256']
            or publication.shared.sha256(Path(__file__)) != auditor_hash):
        raise ValueError('Evaluation output changed during audit')
    publication.shared.atomic_write_json(output,dict(status='complete',checked_queries=checked,
        evaluation_layer=spec['evaluation_layer'],records_sha256=summary['records_sha256'],
        auditor_sha256=auditor_hash,
        reference_role=spec['reference_role'],note='CPU RLE/PNG Dice/boundary recomputation; not independent annotation certification'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit(args.data_root,args.run,args.output)

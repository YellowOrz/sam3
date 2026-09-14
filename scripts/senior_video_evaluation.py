"""Score complete video propagation at source frames 0, stride, 2*stride, ...

Sampling belongs to scoring, never the model's input or memory progression.
Keep the senior's provided-reference, empty/empty=1 frame mean alongside
positive-reference, boundary, FP/FN and known-issue sensitivity results.
"""
from collections import defaultdict
import math

from scripts.hand_evaluation_metrics import summarize_outputs


def require_validated_step(state, step):
    """Explicit diagnostic step selection, without calling it a complete epoch.

    This complements, not replaces, the full checkpoint/provenance loader.
    Epoch-based callers retain their original completed-epoch requirement.
    """
    if (type(step) is not int or step < 1
            or state.get('progress', {}).get('global_step') != step
            or state.get('validation_state', {}).get('last_validation_step') != step):
        raise ValueError('Require explicit matching and fully validated diagnostic step')


def frame_metrics(rows):
    known = [r for r in rows if r['reference_provided']]
    intersection = prediction = reference = 0
    dices, ious = [], []
    for r in known:
        p, g, i = (r[k] for k in ('detected_mask_pixels', 'reference_pixels',
                                  'actual_reference_intersection_pixels'))
        if (any(type(v) is not int or v < 0 for v in (p, g, i))
                or i > min(p, g)):
            raise ValueError('Invalid actual-output overlap pixel counts')
        dice = 2*i/(p+g) if p+g else 1.
        iou = i/(p+g-i) if p+g-i else 1.
        if g and not math.isclose(dice, r['miss_zero_dice'], abs_tol=1e-7):
            raise ValueError('Actual Dice differs from overlap pixels')
        intersection += i
        prediction += p
        reference += g
        dices.append(dice)
        ious.append(iou)
    union = prediction + reference - intersection
    return dict(frames_evaluated=len(known), unknown_reference_queries=len(rows)-len(known),
        intersection=intersection, prediction_pixels=prediction, gt_pixels=reference,
        mean_dice=sum(dices)/len(dices) if dices else None,
        mean_iou=sum(ious)/len(ious) if ious else None,
        pixel_dice=(2*intersection/(prediction+reference) if prediction+reference else 1.) if known else None,
        pixel_iou=(intersection/union if union else 1.) if known else None)


def score_video_rows(records, stride=3):
    if type(stride) is not int or stride < 1:
        raise ValueError('Positive integer scoring stride required')
    groups = defaultdict(list)
    for r in records:
        if r['prompt_key'] not in ('left_hand', 'right_hand'):
            raise ValueError('Unknown hand side')
        if type(r['source_frame_index']) is not int or r['source_frame_index'] < 0:
            raise ValueError('Invalid source frame index')
        groups[(r['recording_id'], r['prompt_key'])].append(r)
    if not groups:
        raise ValueError('Require nonempty complete video records')
    recordings = sorted({name for name, _ in groups})
    for name in recordings:
        identities = []
        for side in ('left_hand', 'right_hand'):
            rows = sorted(groups.get((name, side), []), key=lambda r: r['source_frame_index'])
            if not rows or [r['source_frame_index'] for r in rows] != list(range(len(rows))):
                raise ValueError('Require all continuous source frames for both sides before sampling')
            identities.append([r['image_id'] for r in rows])
        if identities[0] != identities[1] or len(set(identities[0])) != len(identities[0]):
            raise ValueError('Side/source image identities differ or repeat')
    sampled = [r for r in records if r['source_frame_index'] % stride == 0]
    def group(rows):
        return dict(overall=frame_metrics(rows),
            per_side={s: frame_metrics([r for r in rows if r['prompt_key'] == s])
                      for s in ('left_hand', 'right_hand')},
            per_recording={name: {s: frame_metrics([r for r in rows
                if r['recording_id'] == name and r['prompt_key'] == s])
                for s in ('left_hand', 'right_hand')} for name in recordings})
    primary = [r for r in sampled if not r['reference_quality_flags']]
    return dict(format='sam3-continuous-video-sampled-score-v1', inference_frame_stride=1,
        scoring_frame_stride=stride, scoring_source_anchor=0, empty_masks_score=1.,
        propagated_queries=len(records), sampled_queries=len(sampled),
        raw_provided=group(sampled), primary_nonflagged=group(primary),
        actual_output_metrics=summarize_outputs(sampled),
        note='Frame-weighted means, not equal-video means; auxiliary, previously viewed references')

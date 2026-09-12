"""Actual-output metrics shared by image ablations and video-system evaluation.

Candidate masks are diagnostics only. Missing and flagged references never
become negative labels. Adjacent-frame diagnostics are not tracking accuracy.
"""
from collections import defaultdict
import math

SIDES = ('left_hand', 'right_hand')


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def validate_reference_role(role, provenance, previously_inspected):
    if role not in ('external_development', 'independent_holdout'):
        raise ValueError('Unknown evaluation role')
    if role == 'independent_holdout' and (previously_inspected or provenance != 'independent_manual'):
        raise ValueError('Inspected/assisted references cannot be relabelled as blind independent holdout')


def output_metrics(rows):
    positive = [r for r in rows if r['reference_pixels'] > 0]
    negative = [r for r in rows if r['reference_pixels'] == 0]
    for r in rows:
        for key in ('miss_zero_dice', 'miss_zero_boundary_iou_4px'):
            value = r[key]
            if r['reference_pixels'] > 0 and (value is None or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Positive reference requires finite actual-output metrics')
    emitted = lambda r: r['detected_mask_pixels'] > 0
    return dict(queries=len(rows), positive_queries=len(positive), negative_queries=len(negative),
        actual_positive_dice=mean(r['miss_zero_dice'] for r in positive),
        actual_positive_boundary_iou_4px=mean(r['miss_zero_boundary_iou_4px'] for r in positive),
        false_negative_empty_output=sum(not emitted(r) for r in positive),
        false_positive_nonempty_output=sum(emitted(r) for r in negative),
        false_negative_rate=mean(not emitted(r) for r in positive),
        false_positive_rate=mean(emitted(r) for r in negative),
        all_provided_dice_empty_empty_one=mean(
            r['miss_zero_dice'] if r['reference_pixels'] > 0 else float(not emitted(r)) for r in rows),
        candidate_positive_dice_diagnostic=mean(r['top_dice'] for r in positive),
        confidence_detection_count=sum(r['detected'] for r in rows))


def summarize_outputs(records):
    provided = [r for r in records if r['reference_provided']]
    primary = [r for r in provided if not r['reference_quality_flags']]
    def group(rows):
        return dict(overall=output_metrics(rows),
            per_side={s: output_metrics([r for r in rows if r['prompt_key'] == s]) for s in SIDES})
    return dict(definition='Actual thresholded/system output; candidate Dice diagnostic only',
        reference_role='external_development', label_provenance='sam3_assisted',
        independent_ground_truth=False, primary=group(primary), raw_provided=group(provided),
        unknown_queries=len(records)-len(provided), excluded_flagged_queries=len(provided)-len(primary))


def temporal_diagnostics(records):
    """Detection discontinuities / ID-set changes, explicitly NOT ID switches.

    Only consecutive frames with both usable references contribute. Normal
    motion and occlusion may change masks/IDs; correspondence GT is required
    to decide whether a change is actually erroneous.
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r['recording_id'], r['prompt_key'])].append(r)
    pairs = changes = id_changes = triplets = holes = 0
    for rows in groups.values():
        rows.sort(key=lambda r: r['source_frame_index'])
        if len({r['source_frame_index'] for r in rows}) != len(rows):
            raise ValueError('Duplicate frame/side in temporal records')
        eligible = lambda r: r['reference_provided'] and not r['reference_quality_flags']
        for a, b in zip(rows, rows[1:]):
            if b['source_frame_index'] != a['source_frame_index'] + 1 or not (eligible(a) and eligible(b)):
                continue
            pairs += 1
            changes += (a['detected_mask_pixels'] > 0) != (b['detected_mask_pixels'] > 0)
            id_changes += set(a.get('output_object_ids', [])) != set(b.get('output_object_ids', []))
        for a, b, c in zip(rows, rows[1:], rows[2:]):
            if (c['source_frame_index'] != b['source_frame_index'] + 1
                    or b['source_frame_index'] != a['source_frame_index'] + 1
                    or not all(eligible(r) and r['reference_pixels'] > 0 for r in (a,b,c))):
                continue
            triplets += 1
            holes += a['detected_mask_pixels'] > 0 and b['detected_mask_pixels'] == 0 and c['detected_mask_pixels'] > 0
    return dict(eligible_adjacent_pairs=pairs, output_presence_changes=changes,
        output_id_set_changes=id_changes, positive_reference_triplets=triplets,
        isolated_one_frame_empty_outputs=holes, identity_switches=None,
        identity_switches_unavailable_reason='No independently verified instance correspondence GT',
        contact_object_metrics=None, contact_object_metrics_unavailable_reason='This protocol evaluates hands only')

"""CPU-only paired Dex validation diagnosis; never tune on external benchmarks.

Decomposes actual Dice changes into gate and candidate terms, not causal effects.
All provided query identities and candidate RLE Dice are verified before output.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from scripts import visualize_residual_validation as validation

SIDES = validation.SIDES


def mean(values):
    return sum(values) / len(values) if values else None


def validate_comparison_steps(before, after, *, allow_trained_baseline=False):
    if (type(before) is not int or type(after) is not int or before < 0
            or after <= before or (before != 0 and not allow_trained_baseline)):
        raise ValueError('Require increasing nonnegative steps; trained baseline needs explicit opt-in')


def boundary_iou(mask, reference):
    if not reference.any():
        return None
    boundary = lambda m: m & ~binary_erosion(
        m, structure=np.ones((3, 3), bool), iterations=4, border_value=0)
    a, b = boundary(mask), boundary(reference)
    return int((a & b).sum()) / int((a | b).sum())


def paired_metrics(pairs):
    """Input pairs have the same positive reference and fixed score threshold."""
    positive = [(a, b) for a, b in pairs if a['reference_present']]
    negative = [(a, b) for a, b in pairs if not a['reference_present']]
    for a, b in pairs:
        if a['image_id'] != b['image_id']:
            raise ValueError('Paired image identity changed')
        if a['reference_present'] != b['reference_present']:
            raise ValueError('Reference presence changed between runs')
        for r in (a, b):
            c, p, s = (r[k] for k in ('top_class_probability', 'presence_probability', 'top_confidence'))
            if (not all(math.isfinite(x) and 0 <= x <= 1 for x in (c, p, s))
                    or not math.isclose(c*p, s, abs_tol=2e-7, rel_tol=2e-6)
                    or r['detected'] != (s >= .5)):
                raise ValueError('Invalid class/presence/product/detection relationship')
            if r['reference_present']:
                for key in ('candidate_dice', 'candidate_boundary_iou_4px'):
                    d = r[key]
                    if d is None or not math.isfinite(d) or not 0 <= d <= 1:
                        raise ValueError('Invalid positive candidate metric')
                d = r['candidate_dice']
                if not math.isclose(r['miss_zero_dice'], d if r['detected'] else 0., abs_tol=1e-10):
                    raise ValueError('Stored actual Dice inconsistent with gate')
    groups = {}
    for name, before, after in [('kept', True, True), ('new_miss', True, False),
                                ('recovered', False, True), ('persistent_miss', False, False)]:
        selected = [(a, b) for a, b in positive if a['detected'] == before and b['detected'] == after]
        groups[name] = dict(count=len(selected), image_ids=[a['image_id'] for a, _ in selected],
            candidate_dice_ge_0_7_after=sum(b['candidate_dice'] >= .7 for _, b in selected),
            before={k: mean([a[k] for a, _ in selected]) for k in
                ('candidate_dice', 'candidate_boundary_iou_4px', 'top_class_probability', 'presence_probability', 'top_confidence')},
            after={k: mean([b[k] for _, b in selected]) for k in
                ('candidate_dice', 'candidate_boundary_iou_4px', 'top_class_probability', 'presence_probability', 'top_confidence')})
    gate = mean([(int(b['detected'])-int(a['detected']))*a['candidate_dice'] for a, b in positive])
    shape = mean([int(b['detected'])*(b['candidate_dice']-a['candidate_dice']) for a, b in positive])
    delta = mean([b['miss_zero_dice']-a['miss_zero_dice'] for a, b in positive])
    if positive and not math.isclose(gate+shape, delta, abs_tol=1e-10):
        raise ValueError('Additive decomposition failed')
    bk = 'candidate_boundary_iou_4px'
    boundary_gate = mean([(int(b['detected'])-int(a['detected']))*a[bk] for a, b in positive])
    boundary_shape = mean([int(b['detected'])*(b[bk]-a[bk]) for a, b in positive])
    boundary_delta = mean([int(b['detected'])*b[bk]-int(a['detected'])*a[bk] for a, b in positive])
    if positive and not math.isclose(boundary_gate+boundary_shape, boundary_delta, abs_tol=1e-10):
        raise ValueError('Boundary additive decomposition failed')
    return dict(positive_queries=len(positive), absent_queries=len(negative), transitions=groups,
        mean_actual_dice_before=mean([a['miss_zero_dice'] for a, _ in positive]),
        mean_actual_dice_after=mean([b['miss_zero_dice'] for _, b in positive]),
        dice_delta=delta, gate_contribution_with_old_candidate=gate,
        candidate_contribution_under_new_gate=shape,
        mean_actual_boundary_before=mean([int(a['detected'])*a[bk] for a, _ in positive]),
        mean_actual_boundary_after=mean([int(b['detected'])*b[bk] for _, b in positive]),
        boundary_delta=boundary_delta, boundary_gate_contribution_with_old_candidate=boundary_gate,
        boundary_candidate_contribution_under_new_gate=boundary_shape,
        false_negative_before=sum(not a['detected'] for a, _ in positive),
        false_negative_after=sum(not b['detected'] for _, b in positive),
        false_positive_before=sum(a['detected'] for a, _ in negative),
        false_positive_after=sum(b['detected'] for _, b in negative),
        negative_transitions={name: sum(a['detected'] == old and b['detected'] == new for a, b in negative)
            for name, old, new in [('new_false_positive', False, True), ('removed_false_positive', True, False),
                                   ('persistent_false_positive', True, True), ('kept_empty', False, False)]},
        negative_probabilities={label: {k: mean([pair[index][k] for pair in negative]) for k in
            ('top_class_probability', 'presence_probability', 'top_confidence')}
            for index, label in enumerate(('before', 'after'))})


def diagnose(data_root, baseline, comparison, output, *, allow_trained_baseline=False):
    data_root = data_root.resolve()
    if output.exists():
        raise ValueError('Require new output directory')
    annotation_path = data_root/'annotations.json'
    raw = annotation_path.read_bytes()
    digest = validation.sha256(raw)
    doc = validation.parse_json(raw)
    images = {r['id']: r for r in doc['images']}
    if (len(images) != len(doc['images']) or doc.get('info', {}).get('dataset_role') != 'val'
            or any(r.get('source_dataset') != 'dexycb' for r in images.values())
            or doc['categories'] != [{'id': 1, 'name': 'left_hand'}, {'id': 2, 'name': 'right_hand'}]):
        raise ValueError('Require fixed Dex validation data, never external benchmark')
    fingerprints = {str(annotation_path): digest}
    for source in (Path(__file__).resolve(), Path(validation.__file__).resolve()):
        fingerprints[str(source)] = validation.sha256(source.read_bytes())
    records, summaries = [], []
    for directory in (baseline, comparison):
        path = directory/'summary.json'
        raw = path.read_bytes()
        fingerprints[str(path)] = validation.sha256(raw)
        summary = validation.parse_json(raw)
        if (summary.get('dataset_role') != 'validation' or summary.get('scope') != 'dexycb_val'
                or summary.get('annotations_sha256') != digest or summary.get('detection_threshold') != .5
                or summary.get('mask_threshold') != .5 or summary.get('boundary_band_original_pixels') != 4
                or summary.get('metrics', {}).get('images') != len(images)
                or summary.get('metrics', {}).get('queries') != len(images)*2):
            raise ValueError('Validation protocol/coverage mismatch')
        records.append(validation.load_records(directory, summary, images, set(images), digest, fingerprints))
        summaries.append(summary)
    validate_comparison_steps(summaries[0]['global_step'], summaries[1]['global_step'],
                              allow_trained_baseline=allow_trained_baseline)
    annotations = {}
    for r in doc['annotations']:
        key = r['image_id'], r['category_id']
        if key in annotations or key[0] not in images or key[1] not in (1, 2):
            raise ValueError('Invalid/duplicate reference')
        annotations[key] = r['segmentation']
    absent_overlaps = [dict.fromkeys(SIDES, 0) for _ in records]
    empty_candidate = [dict.fromkeys(SIDES, 0) for _ in records]
    for image_id, image in images.items():
        shape = (image['height'], image['width'])
        refs = {side: validation.decode_rle(annotations.get((image_id, index)), shape)
                for index, side in enumerate(SIDES, 1)}
        for j, rows in enumerate(records):
            for side in SIDES:
                r = rows[image_id, side]
                mask = validation.decode_rle(r['prediction_rle'], shape)
                reference = refs[side]
                if r['reference_present'] != bool(reference.any()):
                    raise ValueError('Reference presence mismatch')
                if reference.any():
                    d = 2*int((mask & reference).sum())/(int(mask.sum())+int(reference.sum()))
                    if not math.isclose(d, r['candidate_dice'], abs_tol=1e-10):
                        raise ValueError('Candidate RLE Dice mismatch')
                    if not math.isclose(boundary_iou(mask, reference),
                                        r['candidate_boundary_iou_4px'], abs_tol=1e-10):
                        raise ValueError('Candidate RLE boundary mismatch')
                if r['detected'] and not mask.any():
                    empty_candidate[j][side] += 1
                other = refs[SIDES[1-SIDES.index(side)]]
                if not reference.any() and r['detected'] and mask.any() and other.any():
                    # More than half of output overlaps the other known hand.
                    absent_overlaps[j][side] += int((mask & other).sum()/mask.sum() > .5)
    sides = {side: paired_metrics([(records[0][i, side], records[1][i, side]) for i in sorted(images)])
             for side in SIDES}
    for j, summary in enumerate(summaries):
        for side in SIDES:
            rows = [records[j][i, side] for i in sorted(images)]
            expected = {'positive_count': sum(r['reference_present'] for r in rows),
                'false_negative_count': sum(r['reference_present'] and not r['detected'] for r in rows),
                'false_positive_count': sum(not r['reference_present'] and r['detected'] for r in rows),
                'miss_zero_dice': mean([r['miss_zero_dice'] for r in rows if r['reference_present']]),
                'candidate_boundary_iou_4px': mean([r['candidate_boundary_iou_4px'] for r in rows if r['reference_present']]),
                'miss_zero_boundary_iou_4px': mean([int(r['detected'])*r['candidate_boundary_iou_4px'] for r in rows if r['reference_present']])}
            for key, value in expected.items():
                if not math.isclose(summary['metrics'][f'{side}/{key}'], value, abs_tol=1e-9):
                    raise ValueError('Records and summary disagree')
    sequences = {}
    for i in sorted(images):
        name = images[i].get('source_image_file_name', '')
        parts = Path(name).stem.split('__')
        if len(parts) != 4:
            raise ValueError('Require explicit Dex sequence/view/frame provenance')
        sequence = parts[1]
        sequences.setdefault(sequence, []).append(i)
    grouped = {seq: {side: paired_metrics([(records[0][i, side], records[1][i, side]) for i in ids])
                     for side in SIDES} for seq, ids in sequences.items()}
    for path, expected in fingerprints.items():
        if validation.sha256(Path(path).read_bytes()) != expected:
            raise ValueError('Input changed during diagnosis')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='complete', scope='dexycb_validation_diagnostic', steps=[s['global_step'] for s in summaries],
        trained_baseline_explicitly_allowed=allow_trained_baseline,
        verified_queries_per_model=len(images)*2, metrics=sides, by_sequence=grouped, input_sha256=fingerprints,
        absent_prompt_outputs_mostly_overlapping_other_hand=absent_overlaps,
        detected_but_empty_candidate_counts=empty_candidate,
        limitations=['Gate/shape split is descriptive, not causal; query selection may change.',
            'Other-hand overlap is a suspicion, not a verified anatomical confusion matrix.',
            'Correlated frames are not independent statistical observations; no RealSense tuning.',
            'Dice>=0.7 is a diagnostic bin only, not a deployment or selection threshold.'])
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Dex残差门控与候选形状诊断', '',
        '|侧别|原实际Dice|新实际Dice|门控贡献|候选贡献|新增漏检|恢复检出|新增漏检中候选Dice≥0.7|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for side, m in sides.items():
        t = m['transitions']
        lines.append(f"|{side}|{m['mean_actual_dice_before']:.5f}|{m['mean_actual_dice_after']:.5f}|"
            f"{m['gate_contribution_with_old_candidate']:+.5f}|{m['candidate_contribution_under_new_gate']:+.5f}|"
            f"{t['new_miss']['count']}|{t['recovered']['count']}|{t['new_miss']['candidate_dice_ge_0_7_after']}|")
    lines.extend(['', '|侧别|原实际边界IoU|新实际边界IoU|边界门控贡献|边界候选贡献|原漏检→新漏检|原误报→新误报|',
        '|---|---:|---:|---:|---:|---|---|'])
    for side, m in sides.items():
        lines.append(f"|{side}|{m['mean_actual_boundary_before']:.5f}|{m['mean_actual_boundary_after']:.5f}|"
            f"{m['boundary_gate_contribution_with_old_candidate']:+.5f}|{m['boundary_candidate_contribution_under_new_gate']:+.5f}|"
            f"{m['false_negative_before']}→{m['false_negative_after']}|{m['false_positive_before']}→{m['false_positive_after']}|")
    lines.extend(['', '固定0.5分数门槛；全量RLE Dice及4px边界重算和摘要核验通过。门控贡献＋候选贡献严格等于对应实际指标变化；不是因果消融。',
        '', '候选Dice≥0.7仅用于诊断，不调阈值；空侧与另一手重叠仅为错手嫌疑。完整逐序列统计与输入SHA见 [metrics.json](metrics.json)。'])
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-trained-baseline', action='store_true',
                   help='Explicitly compare an earlier trained checkpoint against a later step')
    args = p.parse_args()
    diagnose(args.data_root, args.baseline, args.comparison, args.output,
             allow_trained_baseline=args.allow_trained_baseline)

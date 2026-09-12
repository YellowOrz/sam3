"""Compare explicit unified evaluation directories; never search for a best epoch.

CPU-only report and fixed, separate RGB/reference/candidate/detected panels.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
from scripts import evaluate_realsense_full as full


def load_runs(paths):
    runs = []
    for path in paths:
        summary = json.loads((path / 'summary.json').read_text())
        if summary.get('status') != 'complete' or not summary.get('actual_complete_query_coverage_verified'):
            raise ValueError('Incomplete evaluation')
        if summary.get('protocol', {}).get('format') != 'sam3-hand-routes-v1':
            raise ValueError('Unified protocol required; legacy summaries cannot be mixed')
        records_path = path / 'records.jsonl'
        if full.shared.sha256(records_path) != summary['records_sha256']:
            raise ValueError('Records hash mismatch')
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        expected = {(i, side) for i in summary['protocol']['image_ids'] for side in full.SIDES}
        observed = [(r['image_id'], r['prompt_key']) for r in records]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError('Query coverage mismatch')
        if full.summarize(records) != summary['metrics']:
            raise ValueError('Metrics differ from records')
        if runs and summary['protocol'] != runs[0][1]['protocol']:
            raise ValueError('Different evaluation protocols')
        runs.append((path, summary))
    if len(runs) < 2:
        raise ValueError('At least two explicit runs required')
    return runs


def export(paths, output):
    runs = load_runs(paths)
    visual_keys = [{(v['image_id'], v['side']) for v in s['visuals']} for _, s in runs]
    if any(keys != visual_keys[0] for keys in visual_keys):
        raise ValueError('Different preselected visualization samples')
    if not visual_keys[0]:
        raise ValueError('No preselected visualizations')
    output.mkdir(parents=True, exist_ok=False)
    lines = ['# 同口径手分割对照', '',
             '参考为 SAM3 辅助标签，不是独立精标。固定样本不按模型效果挑选；不据此选择 checkpoint。', '',
             '|方法／阶段轮次|候选 Dice|漏检计零 Dice|边界 IoU（4px）|漏检／有手查询|误报／无手查询|',
             '|---|---:|---:|---:|---:|---:|']
    for path, summary in runs:
        metrics = summary['metrics']['primary_provided_nonflagged']['overall']
        values = [metrics[k] for k in ('present_mean_candidate_dice', 'present_mean_miss_zero_dice',
                                      'candidate_boundary_iou_4px')]
        cells = ['N/A' if v is None else f'{v:.5f}' for v in values]
        lines.append(f"|{summary['method']} / {summary['epoch']}|" + '|'.join(cells)
            + f"|{metrics['false_negative_queries']}/{metrics['present_queries']}"
            + f"|{metrics['false_positive_queries']}/{metrics['absent_queries']}|")
    for image_id, side in sorted(visual_keys[0]):
        panels = []
        reference_bytes = None
        for index, (path, summary) in enumerate(runs):
            directory = path / 'visuals' / f'image-{image_id:06d}' / side
            current = tuple((directory / f'{name}.png').read_bytes() for name in ('rgb', 'reference'))
            if reference_bytes is not None and reference_bytes != current:
                raise ValueError('Visualization RGB/reference differ')
            reference_bytes = current
            names = ('rgb', 'reference', 'candidate', 'detected') if index == 0 else ('candidate', 'detected')
            for name in names:
                with Image.open(directory / f'{name}.png') as image:
                    panel = image.convert('RGB')
                label = name if name in ('rgb', 'reference') else f"{summary['method']} e{summary['epoch']} {name}"
                panels.append((label, panel))
        width, height = panels[0][1].size
        if any(p.size != (width, height) for _, p in panels):
            raise ValueError('Panel dimensions differ')
        canvas = Image.new('RGB', (width * len(panels), height + 40), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, (label, panel) in enumerate(panels):
            canvas.paste(panel, (index * width, 40))
            draw.text((index * width + 8, 8), label, fill='black')
        name = f'{image_id:06d}-{side}.png'
        canvas.save(output / name)
        lines.extend(['', f'![{image_id} {side}]({name})'])
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    full.shared.atomic_write_json(output / 'comparison.json', {
        'protocol': runs[0][1]['protocol'], 'runs': [s for _, s in runs],
        'sources': [str(p.resolve()) for p, _ in runs]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output_dir)


if __name__ == '__main__':
    main()

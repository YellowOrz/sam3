"""Compare explicit CPU-audited full video runs using the source-anchored score.

No checkpoint searching, tuning, missing-run substitution or mask regeneration.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

from scripts.compare_hand_routes import load_runs
from scripts.senior_video_evaluation import score_video_rows


def label(summary):
    if summary['method'] == 've':
        return 'original VE'
    if summary.get('step') is not None:
        mode = summary.get('residual_metadata', {}).get('residual_positions', 'unknown')
        return f"residual {mode} step {summary['step']} (partial epoch)"
    return f"{summary['method']} epoch {summary['epoch']}"


def compare(paths, output):
    runs = load_runs(paths)
    if output.exists():
        raise ValueError('New comparison directory required')
    samples = []
    for path, summary in runs:
        spec = summary['protocol']
        if spec['evaluation_layer'] != 'video_system' or spec['frame_stride'] != 1:
            raise ValueError('Continuous video-system runs required')
        if not spec.get('sampled_mean_dice_empty_empty_one') or spec.get('scoring_source_anchor') != 0:
            raise ValueError('Explicit source-anchored scoring protocol required')
        audit = json.loads((path / 'cpu-audit.json').read_text())
        if (audit.get('status') != 'complete' or audit.get('records_sha256') != summary['records_sha256']
                or audit.get('checked_queries') != summary['queries']):
            raise ValueError('Require independent complete RLE/PNG audit for each run')
        records = [json.loads(line) for line in (path / 'records.jsonl').read_text().splitlines()]
        sampled = score_video_rows(records, spec['scoring_frame_stride'])
        if sampled != summary.get('sampled_video_metrics'):
            raise ValueError('Sampled summary differs from records')
        samples.append(sampled)
    labels = [label(s) for _, s in runs]
    if len(set(labels)) != len(labels):
        raise ValueError('Ambiguous duplicate model labels')
    output.mkdir(parents=True, exist_ok=False)
    spec = runs[0][1]['protocol']
    lines = ['# 连续视频：同口径进一步对比', '',
        f"每个模型连续传播全部 {len(spec['image_ids'])} 帧，左右独立会话；"
        f"仅评分各录像原始0、{spec['scoring_frame_stride']}、{2*spec['scoring_frame_stride']}…帧。", '',
        '使用相同SAM3权重、视频流程与默认阈值，实际输出实例取并集，不用参考挑选候选。', '',
        '主表复现学长的已提供参考、双方为空计1的逐帧mean Dice；已知问题仍明确单列。'
        '这些是反复查看过的SAM3辅助参考，不是独立人工精标或盲测。', '',
        '|模型|侧别|评分帧数|mean Dice（含空帧）|pixel Dice|正参考实际Dice|漏检/有手|误报/空参考|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'N/A' if v is None else f'{v:.5f}'
    for name, sample in zip(labels, samples):
        for side, m in sample['raw_provided']['per_side'].items():
            o = sample['actual_output_metrics']['raw_provided']['per_side'][side]
            lines.append(f"|{name}|{side}|{m['frames_evaluated']}|{fmt(m['mean_dice'])}|"
                f"{fmt(m['pixel_dice'])}|{fmt(o['actual_positive_dice'])}|"
                f"{o['false_negative_empty_output']}/{o['positive_queries']}|"
                f"{o['false_positive_nonempty_output']}/{o['negative_queries']}|")
    lines += ['', '## 排除已知问题的敏感性结果', '',
        '|模型|侧别|mean Dice（含空帧）|正参考实际Dice|边界IoU4px|', '|---|---|---:|---:|---:|']
    for name, sample in zip(labels, samples):
        for side, m in sample['primary_nonflagged']['per_side'].items():
            o = sample['actual_output_metrics']['primary']['per_side'][side]
            lines.append(f"|{name}|{side}|{fmt(m['mean_dice'])}|{fmt(o['actual_positive_dice'])}|"
                f"{fmt(o['actual_positive_boundary_iou_4px'])}|")
    lines += ['', '## 同帧分离可视化', '', '依次为RGB、辅助参考、各模型实际输出；灰色表示参考缺失。', '']
    keys = [{(v['image_id'],v['side']) for v in s['visuals']} for _,s in runs]
    if any(k != keys[0] for k in keys) or not keys[0]:
        raise ValueError('Predeclared visualization identity differs or is empty')
    for image_id, side in sorted(keys[0]):
        panels, reference = [], None
        for idx, ((path, _), name) in enumerate(zip(runs, labels)):
            folder = path/'visuals'/f'image-{image_id:06d}'/side
            source = tuple((folder/f'{key}.png').read_bytes() for key in ('rgb','reference'))
            if reference is not None and source != reference:
                raise ValueError('RGB/reference bytes differ across models')
            reference = source
            for key in (('rgb','reference','detected') if idx == 0 else ('detected',)):
                with Image.open(folder/f'{key}.png') as im:
                    panels.append((name if key == 'detected' else key, im.convert('RGB')))
        width,height = panels[0][1].size
        if any(im.size != (width,height) for _,im in panels):
            raise ValueError('Panel dimensions differ')
        canvas = Image.new('RGB',(width*len(panels),height+44),'white')
        draw = ImageDraw.Draw(canvas)
        for idx,(title,im) in enumerate(panels):
            canvas.paste(im,(width*idx,44)); draw.text((width*idx+8,8),title,fill='black')
        filename = f'{image_id:06d}-{side}.png'
        canvas.save(output/filename)
        lines += [f'### image {image_id} / {side}', '', f'![对照]({filename})', '']
    result = dict(status='complete', protocol=spec, models={name: sampled for name,sampled in zip(labels,samples)},
        source_runs=[str(p) for p,_ in runs],
        note='Independent pixel audit passed; no independent annotation certification')
    (output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='append',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    compare(args.run,args.output)

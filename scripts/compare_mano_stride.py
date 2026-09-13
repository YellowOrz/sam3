"""Independent CPU audit and paired report of completed MANO stride-three shards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from scripts import mano_stride_protocol as protocol
from scripts import pilot_mano_geometry_prompts as pilot
from scripts.compare_realsense_full import decode_rle


def independent_metrics(candidate, score, reference, other):
    if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('Invalid stored probability')
    prediction=candidate if score>.5 else np.zeros_like(candidate)
    n=int(prediction.sum())
    row=dict(prediction_pixels=n,reference_pixels=None if reference is None else int(reference.sum()),
             dice=None,boundary_iou_4px=None,candidate_dice=None,all_provided_dice=None,
             false_negative=None,false_positive=None,other_dice=None)
    if reference is not None:
        r=int(reference.sum());intersection=int(np.count_nonzero(prediction&reference))
        row['all_provided_dice']=2*intersection/(n+r) if n+r else 1.
        row['false_negative']=bool(r and not n);row['false_positive']=bool(not r and n)
        if r:
            row['dice']=2*intersection/(n+r)
            row['candidate_dice']=2*int(np.count_nonzero(candidate&reference))/(int(candidate.sum())+r)
            pb=prediction & ~binary_erosion(prediction,np.ones((3,3),bool),iterations=4,border_value=0)
            rb=reference & ~binary_erosion(reference,np.ones((3,3),bool),iterations=4,border_value=0)
            row['boundary_iou_4px']=np.count_nonzero(pb&rb)/np.count_nonzero(pb|rb)
    if other is not None and other.any():
        row['other_dice']=2*int(np.count_nonzero(prediction&other))/(n+int(other.sum()))
    return row


def equal_metrics(supplied, computed):
    for key,value in computed.items():
        actual=supplied.get(key)
        if value is None:
            good=actual is None
        elif isinstance(value,(bool,int)):
            good=type(actual)==type(value) and actual==value
        else:
            good=type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,value,rel_tol=1e-9,abs_tol=1e-9)
        if not good:
            raise ValueError(f'Independent metric mismatch: {key}: {actual} != {value}')


def validate_shards(summaries):
    if not summaries:raise ValueError('No shards')
    spec=summaries[0]['protocol'];count=spec['shard_count']
    if (spec.get('format')!=protocol.FORMAT or spec.get('contract')!=protocol.CONTRACT
            or len(summaries)!=count or sorted(r['shard_index'] for r in summaries)!=list(range(count))):
        raise ValueError('Missing/duplicate shard or incompatible protocol')
    ids=spec['selected_image_ids']
    if len(ids)!=len(set(ids)):raise ValueError('Repeated selected identity')
    for summary in summaries:
        assigned=ids[summary['shard_index']::count]
        if (summary['status']!='complete' or summary['protocol']!=spec
                or summary.get('frozen_parameters_verified') is not True
                or summary['assigned_image_ids']!=assigned
                or summary['frames']!=len(assigned)
                or summary['rows']!=len(assigned)*2*len(protocol.METHODS)):
            raise ValueError('Incomplete, inconsistent, or overlapping shard')
    return spec


def average(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def summarize(rows):
    evaluated=[r for r in rows if r['evaluated']]
    positive=[r for r in evaluated if r['reference_pixels'] is not None and r['reference_pixels']>0]
    empty=[r for r in evaluated if r['reference_pixels']==0]
    provided=[r for r in evaluated if r['reference_pixels'] is not None]
    recordings=sorted({r['recording_id'] for r in positive})
    return dict(queries=len(rows),evaluated=len(evaluated),not_evaluated=len(rows)-len(evaluated),
        positive=len(positive),empty=len(empty),unknown=len(evaluated)-len(provided),
        fallback=sum(r['fallback'] for r in evaluated),
        mean_dice=average(r['dice'] for r in positive),mean_boundary_iou_4px=average(r['boundary_iou_4px'] for r in positive),
        mean_candidate_dice=average(r['candidate_dice'] for r in positive),
        equal_recording_macro_dice=average(average(r['dice'] for r in positive if r['recording_id']==name) for name in recordings),
        all_provided_mean_dice_empty_empty_one=average(r['all_provided_dice'] for r in provided),
        false_negative=sum(r['false_negative'] for r in positive),false_positive=sum(r['false_positive'] for r in empty),
        other_overlap_dominant_proxy=sum(r['other_dice'] is not None and r['dice'] is not None
                                       and r['other_dice']>r['dice'] and r['prediction_pixels']>0 for r in positive))


def grouped(rows):
    return {name:summarize([r for r in rows if r['method']==name]) for name in protocol.METHODS}


def audit(root, plan_path, runs, output):
    from scripts import evaluate_realsense_full as pub
    if output.exists():raise ValueError('New audit/report directory required')
    root=root.resolve();images,annotations,outputs,_,hashes=pub.load_publication(root)
    plan=json.loads(plan_path.read_text());selected=protocol.verify_plan(plan,images,hashes)
    publication_hashes={Path(k).name:v for k,v in hashes.items()}
    input_hashes={str(plan_path):pilot.sha256(plan_path),str(Path(__file__)):pilot.sha256(__file__)}
    summaries=[json.loads((run/'summary.json').read_text()) for run in runs]
    spec=validate_shards(summaries)
    if (spec['plan_sha256']!=pilot.sha256(plan_path) or spec['publication_sha256']!=publication_hashes
            or spec['base_sha256']!=plan['base_sha256'] or spec['tokenizer_sha256']!=plan['tokenizer_sha256']):
        raise ValueError('Plan/data/base binding differs')
    expected_selected=([r['id'] for r in selected] if not spec['engineering'] else
        [r['id'] for r in selected if r['id'] in [plan['render_ids'][i] for i in range(1,len(plan['render_ids']),3)]+[selected[0]['id']]])
    if spec['selected_image_ids']!=expected_selected:raise ValueError('Actual selection does not match registered protocol')
    by_frame=defaultdict(list);seen=set()
    for run,summary in zip(runs,summaries):
        records=run/'records.jsonl';input_hashes[str(run/'summary.json')]=pilot.sha256(run/'summary.json')
        input_hashes[str(records)]=pilot.sha256(records)
        if input_hashes[str(records)]!=summary['records_sha256']:raise ValueError('Records SHA mismatch')
        shard_rows=[json.loads(line) for line in records.read_text().splitlines()]
        expected={(i,s,m) for i in summary['assigned_image_ids'] for s in protocol.SIDES for m in protocol.METHODS}
        found={(r['image_id'],r['side'],r['method']) for r in shard_rows}
        if len(shard_rows)!=len(expected) or found!=expected or seen&found:raise ValueError('Duplicate/missing/foreign query')
        seen|=found
        if sum(r['evaluated'] and not r['fallback'] for r in shard_rows)!=summary['forwards']:
            raise ValueError('Actual forward count mismatch')
        for row in shard_rows:by_frame[row['image_id']].append(row)
    frames={r['image_id']:r for r in plan['frames']};by_id={r['id']:i for i,r in enumerate(images)}
    checked=[];baselines={}
    for done,image_id in enumerate(spec['selected_image_ids'],1):
        image=images[by_id[image_id]];refs=pub.batch_references(root,images,[by_id[image_id]],annotations,outputs,hashes)[image_id]
        for row in by_frame[image_id]:
            side=row['side'];name=row['method'];g=frames[image_id]['geometry'][side]
            condition=protocol.method_condition(name,g)
            for k,v in condition.items():
                if row[k]!=v:raise ValueError('Geometry availability/fallback mismatch')
            if (row['recording_id'],row['frame_index'],row['reference_provided'],row['reference_quality_flags'],row['text']) != (
                    image['recording_id'],image['source_frame_index'],refs[side] is not None,pub.quality_flags(image,side),protocol.text_for(name,side)):
                raise ValueError('Reference identity/side/quality/text mismatch')
            pixels=None if refs[side] is None else int(refs[side].sum())
            if (row['reference_pixels']!=pixels or g['reference_pixels']!=pixels
                    or row['common_geometry_eligible']!=all(g[k] is not None for k in ('points','mesh_box','reference_box'))
                    or row['reference_area_group']!=protocol.area_group(pixels,image['width'],image['height'])):
                raise ValueError('Reference size/eligibility group mismatch')
            if row['evaluated']:
                candidate=decode_rle(row['candidate_rle'],(image['height'],image['width']))
                metric=independent_metrics(candidate,row['score'],refs[side],refs[protocol.SIDES[1-protocol.SIDES.index(side)]])
                equal_metrics(row,metric)
                effective='text' if row['fallback'] else name
                _,pts,box,refbox=pilot.METHODS[effective]
                pc=len(g['points']) if pts else 0;bc=int(box or refbox)
                if row['geometry_call']!=dict(points=pc,boxes=bc,encoded_shape=[1+pc+bc,1,256]):
                    raise ValueError('Geometry runtime hook differs')
                if type(row['candidate_index']) is not int or row['candidate_index']<0:raise ValueError('Invalid query index')
                if name=='text':baselines[image_id,side]=row
            else:
                for key in ('candidate_rle','candidate_index','score','geometry_call',*independent_metrics(np.zeros((1,1),bool),0,None,None)):
                    if key!='reference_pixels' and row[key] is not None:raise ValueError('Unevaluated control has invented output/score')
            checked.append(row)
        if done%100==0:print(json.dumps(dict(audited_frames=done,total=len(spec['selected_image_ids']))),flush=True)
    for row in checked:
        if row['fallback']:
            baseline=baselines[row['image_id'],row['side']]
            if any(row[k]!=baseline[k] for k in ('candidate_rle','score','candidate_index','geometry_call')):
                raise ValueError('Fallback must be exact same-side text output, not a fabricated empty mask')
    for path,digest in {**hashes,**input_hashes}.items():
        if pilot.sha256(path)!=digest:raise ValueError('Input changed during independent audit')
    clean=[r for r in checked if not r['reference_quality_flags']]
    common=[r for r in clean if r['common_geometry_eligible']]
    summary=dict(status='complete',engineering=spec['engineering'],frames=len(spec['selected_image_ids']),rows=len(checked),
        independently_audited_rle_queries=sum(r['evaluated'] for r in checked),
        protocol=spec,input_sha256=input_hashes,primary_all_times=grouped(clean),raw_provided_sensitivity=grouped(checked),
        common_geometry_eligible=grouped(common),
        per_side={s:grouped([r for r in clean if r['side']==s]) for s in protocol.SIDES},
        per_recording={s:grouped([r for r in clean if r['recording_id']==s]) for s in sorted({r['recording_id'] for r in clean})},
        per_reference_area={s:grouped([r for r in clean if r['reference_area_group']==s]) for s in sorted({r['reference_area_group'] for r in clean})},
        geometry_unavailable_reasons=dict(Counter(reason for r in checked if r['fallback'] for reason in r['reasons'])),
        note='All-time core methods use explicit same-side text fallback. Controls only applicable queries; use common subset for nine-way comparisons. Auxiliary references; no independent/causal/significance/temporal claim.')
    pairs={}
    for method in protocol.METHODS[1:]:
        subset=[r for r in (clean if method in protocol.CORE else common) if r['method']==method and r['dice'] is not None]
        delta=[r['dice']-baselines[r['image_id'],r['side']]['dice'] for r in subset]
        pairs[method]=dict(queries=len(delta),mean_dice_change=average(delta),
                          improved=sum(d>1e-9 for d in delta),degraded=sum(d< -1e-9 for d in delta),unchanged=sum(abs(d)<=1e-9 for d in delta))
    summary['paired_vs_text']=pairs
    output.mkdir(parents=True)
    pilot.write_json(output/'metrics.json',summary)
    with (output/'comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['scope','method',*next(iter(summary['primary_all_times'].values()))]);writer.writeheader()
        for scope in ('primary_all_times','common_geometry_eligible','raw_provided_sensitivity'):
            for method,values in summary[scope].items():writer.writerow(dict(scope=scope,method=method,**values))
    def table(values,names):
        lines=['| 方法 | 正参考查询 | 实际Dice | 4px边界IoU | 漏检 | 空参考误报 | 文字回退 |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name in names:
            v=values[name];d='—' if v['mean_dice'] is None else f'{v["mean_dice"]:.5f}'
            b='—' if v['mean_boundary_iou_4px'] is None else f'{v["mean_boundary_iou_4px"]:.5f}'
            lines.append(f'| {name} | {v["positive"]} | {d} | {b} | {v["false_negative"]} | {v["false_positive"]}/{v["empty"]} | {v["fallback"]} |')
        return '\n'.join(lines)
    lines=['# MANO原生geometry：每3帧取1扩大评估',
           f'\n状态：{"工程检查，不是正式实验结果" if spec["engineering"] else "完整推理＋独立CPU复算通过"}。{summary["frames"]}帧、{summary["rows"]}方法×侧记录，实际保存并验算{summary["independently_audited_rle_queries"]}次输出。',
           '\n原SAM3全部冻结；不训练、不用Tracker、阈值固定。MANO来自辅助参考mask裁剪后的WiLoR；参考框控制直接使用参考mask，不是独立估计。不能称盲测或MANO独立定位精度。',
           '\n## 全时段主比较\n\n每段零基0、3、6…；几何缺失保留帧，后三组显式回退同侧原文字。正参考Dice漏检计零；空参考误报另外列，不用空空得1掩盖问题。质量标记排除主指标、原样敏感性表在CSV/JSON中。\n',
           table(summary['primary_all_times'],protocol.CORE),
           '\n## 九组共同几何可用子集\n\n所有组使用同一批具备点、框、正参考框的查询；只代表条件定位诊断，不能替代上表全时段结果。wrong_text仍相对几何目标侧评分，visual是原生占位文本。\n',
           table(summary['common_geometry_eligible'],protocol.METHODS)]
    for side in protocol.SIDES:
        lines.extend([f'\n## {side} 全时段\n',table(summary['per_side'][side],protocol.CORE)])
    lines.append('\n## 每段录像\n\n帧间相关，另在JSON报告逐录像等权宏平均。面积分组仅是可见参考面积代理，不是独立遮挡标签。')
    for recording,values in summary['per_recording'].items():
        lines.extend([f'\n### {recording}\n',table(values,protocol.CORE)])
    lines.extend(['\n## 复算与可视化\n',f'- [完整指标](metrics.json)；[可复制对比表](comparison.csv)。',
        '- 原图、辅助参考、模型预测分栏；灰色表示未知或条件不可用，不是负mask。每段3个时间点在推理前固定。',
        '- 原始结果及分离图目录：'])
    for run in runs:lines.append(f'  - [{run.name}]({run.resolve()})')
    lines.append('\n## 限制\n\n不说明接触物体、独立解剖学手别、3D精度或时序稳定性。更多覆盖减轻12帧抽样偶然性，但stride3不保证比全帧误差更小。RealSense参考为SAM3辅助且已反复查看，不用于学习率、阈值或选训练轮次。既有混合训练保持；未来新实验Dex训练／Dex验证，nake和RealSense只作外部评估，旧混合权重不能在nake称未见测试。')
    with (output/'README.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',frames=summary['frames'],rows=len(checked),primary=summary['primary_all_times'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    a=p.parse_args();audit(a.root,a.plan,a.runs,a.output)

"""Finite, frozen original-SAM3 geometry evaluation, independent image GPU shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from scripts import mano_stride_protocol as protocol
from scripts import pilot_mano_geometry_prompts as pilot


def code_hashes(root):
    root=Path(root).resolve()
    return {str(p.relative_to(root)):pilot.sha256(p) for prefix in ('sam3','scripts')
            for p in (root/prefix).rglob('*.py')}


def actual_output(candidate, score):
    if not np.isfinite(score) or not 0<=score<=1:
        raise ValueError('Invalid score')
    return candidate if score>.5 else np.zeros_like(candidate)


def metrics(candidate, score, reference, other):
    prediction=actual_output(candidate,score)
    fields=dict(prediction_pixels=int(prediction.sum()),reference_pixels=None if reference is None else int(reference.sum()),
                dice=None,boundary_iou_4px=None,candidate_dice=None,all_provided_dice=None,
                false_negative=None,false_positive=None,other_dice=None)
    if reference is not None:
        denominator=int(prediction.sum())+int(reference.sum())
        fields['all_provided_dice']=float(2*(prediction&reference).sum()/denominator) if denominator else 1.
        fields['false_negative']=bool(reference.any() and not prediction.any())
        fields['false_positive']=bool(not reference.any() and prediction.any())
        if reference.any():
            scores=pilot.mask_metrics(prediction,reference)
            fields.update(dice=scores['dice'],boundary_iou_4px=scores['boundary_iou_4px'],
                          candidate_dice=float(2*(candidate&reference).sum()/(int(candidate.sum())+int(reference.sum()))))
    if other is not None and other.any():
        fields['other_dice']=float(2*(prediction&other).sum()/(int(prediction.sum())+int(other.sum())))
    return fields


def encode(mask):
    rle=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=rle['size'],counts=rle['counts'].decode('ascii'))


def render(folder, rgb, reference, rows, geometry):
    folder.mkdir(parents=True,exist_ok=False)
    Image.fromarray(rgb).save(folder/'rgb.png')
    if reference is not None:
        Image.fromarray(reference.astype(np.uint8)*255).save(folder/'reference.png')
    tiles=[('RGB',Image.fromarray(rgb)),('Assisted reference' if reference is not None else 'UNKNOWN reference',
            Image.fromarray(reference.astype(np.uint8)*255) if reference is not None else Image.new('L',(640,480),100))]
    for name in protocol.METHODS:
        row=rows[name]
        if row['evaluated']:
            candidate=mask_utils.decode(row['candidate_rle']).astype(bool)
            mask=actual_output(candidate,row['score'])
            img=Image.fromarray(mask.astype(np.uint8)*255);img.save(folder/(name+'.png'))
            dice=row['dice']; label=f'{name}\nDice {dice:.3f}' if dice is not None else f'{name}\nDice N/A'
            if row['fallback']:label+=' TEXT FALLBACK'
        else:
            img=Image.new('L',(640,480),100);label=name+'\nNOT EVALUATED'
        tiles.append((label,img))
    for filename, indices in [('comparison.png',list(range(6))),('controls.png',[0,1,*range(6,11)])]:
        canvas=Image.new('RGB',(320*len(indices),290),'white');draw=ImageDraw.Draw(canvas)
        for col,index in enumerate(indices):
            label,img=tiles[index];draw.text((col*320+4,3),label,fill='black')
            canvas.paste(img.convert('RGB').resize((320,240),Image.Resampling.NEAREST),(col*320,50))
        canvas.save(folder/filename)
    canvas=Image.fromarray(rgb);draw=ImageDraw.Draw(canvas)
    for x,y in geometry['joints_pixel'] or []:
        if 0<=x<640 and 0<=y<480:draw.ellipse((x-2,y-2,x+2,y+2),fill='cyan')
    extent=geometry['mesh_box_xyxy_pixel']
    if extent is not None:
        extent=np.clip(extent,[0,0,0,0],[639,479,639,479]);draw.rectangle(tuple(extent),outline='cyan',width=2)
    draw.text((8,8),'PROMPT LOCATIONS ONLY - NOT PREDICTION',fill='cyan')
    canvas.save(folder/'prompt_locations_only.png')


def run(a):
    import torch
    import torch.nn.functional as functional
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from scripts import evaluate_realsense_full as pub
    if (a.output.exists() or not 0<=a.shard_index<a.shard_count or not 1<=a.shard_count<=4
            or not 0<a.gpu_memory_fraction<=.8):
        raise ValueError('New output and valid independent shard required')
    root=a.root.resolve();plan=json.loads(a.plan.read_text())
    images,annotations,outputs,_,hashes=pub.load_publication(root)
    selected=protocol.verify_plan(plan,images,hashes)
    if pilot.sha256(a.base_checkpoint)!=plan['base_sha256'] or pilot.sha256(a.tokenizer)!=plan['tokenizer_sha256']:
        raise ValueError('Base checkpoint/tokenizer changed')
    engineering_ids=[plan['render_ids'][i] for i in range(1,len(plan['render_ids']),3)] + [selected[0]['id']]
    chosen=[r for r in selected if not a.engineering or r['id'] in engineering_ids]
    assigned=chosen[a.shard_index::a.shard_count]
    if not assigned:
        raise ValueError('Empty shard')
    code=code_hashes(Path.cwd())
    signature=dict(format=protocol.FORMAT,contract=plan['contract'],plan_sha256=pilot.sha256(a.plan),
                   publication_sha256=plan['publication_sha256'],base_sha256=plan['base_sha256'],
                   tokenizer_sha256=plan['tokenizer_sha256'],code_sha256=code,engineering=a.engineering,
                   shard_count=a.shard_count,selected_image_ids=[r['id'] for r in chosen])
    a.output.mkdir(parents=True)
    pilot.write_json(a.output/'run.json',dict(protocol=signature,shard_index=a.shard_index,
                                           assigned_image_ids=[r['id'] for r in assigned]))
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction)
    torch.manual_seed(123)
    model=build_sam3_image_model(checkpoint_path=str(a.base_checkpoint),bpe_path=str(a.tokenizer),
        load_from_HF=False,device='cuda',eval_mode=True,enable_segmentation=True,
        enable_inst_interactivity=False,text_encoder_type='ve')
    model.eval().requires_grad_(False); versions=[(p,p._version) for p in model.parameters()]
    calls=[]
    def hook(module,args,kwargs,result):
        prompt=kwargs['geo_prompt']
        calls.append(dict(points=0 if prompt.point_embeddings is None else len(prompt.point_embeddings),
                          boxes=0 if prompt.box_embeddings is None else len(prompt.box_embeddings),
                          encoded_shape=list(result[0].shape)))
    handle=model.geometry_encoder.register_forward_hook(hook,with_kwargs=True)
    processor=Sam3Processor(model,device='cuda',confidence_threshold=.5)
    frames={r['image_id']:r for r in plan['frames']};by_id={r['id']:i for i,r in enumerate(images)}
    started=time.monotonic();row_count=0;forwards=0
    try:
        with (a.output/'records.jsonl').open('x') as stream,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            texts={text:model.backbone.forward_text([text],device='cuda') for text in ('left hand','right hand','visual')}
            for done,image in enumerate(assigned,1):
                if time.monotonic()-started>10800:
                    raise RuntimeError('Finite three-hour evaluation budget exceeded')
                refs=pub.batch_references(root,images,[by_id[image['id']]],annotations,outputs,hashes)[image['id']]
                rgb=np.asarray(Image.open(pub.local_file(root,image['file_name'])).convert('RGB'))
                state=processor.set_image(Image.fromarray(rgb));item=frames[image['id']]
                for side in protocol.SIDES:
                    g=item['geometry'][side];rows={};baseline=None
                    if g['reference_pixels']!=(None if refs[side] is None else int(refs[side].sum())):
                        raise ValueError('Prepared reference no longer matches')
                    for name in protocol.METHODS:
                        condition=protocol.method_condition(name,g)
                        row=dict(image_id=image['id'],recording_id=image['recording_id'],frame_index=image['source_frame_index'],
                                 side=side,method=name,text=protocol.text_for(name,side),**condition,
                                 reference_provided=refs[side] is not None,reference_quality_flags=pub.quality_flags(image,side),
                                 common_geometry_eligible=all(g[k] is not None for k in ('points','mesh_box','reference_box')),
                                 reference_area_group=protocol.area_group(g['reference_pixels'],image['width'],image['height']))
                        if not condition['evaluated']:
                            row.update(score=None,candidate_index=None,candidate_rle=None,geometry_call=None,
                                       **{key:None for key in metrics(np.zeros((1,1),bool),0,None,None)})
                            row['reference_pixels']=g['reference_pixels']
                        else:
                            if condition['fallback']:
                                candidate,score,index,geo_call=baseline
                            else:
                                geo=pilot.geometry_prompt(name,g['points'],g['mesh_box'],g['reference_box'],'cuda') or model._get_dummy_prompt()
                                backbone=dict(state['backbone_out']);backbone.update(texts[row['text']])
                                before=len(calls)
                                out=model.forward_grounding(backbone_out=backbone,find_input=processor.find_stage,
                                                           geometric_prompt=geo,find_target=None)
                                scores=(out['pred_logits'].sigmoid()*out['presence_logit_dec'].sigmoid().unsqueeze(1)).flatten()
                                if not torch.isfinite(scores).all() or not torch.isfinite(out['pred_masks']).all():
                                    raise RuntimeError('Nonfinite SAM3 output')
                                index=int(scores.argmax());score=float(scores[index])
                                candidate=functional.interpolate(out['pred_masks'][0,index][None,None].float(),
                                    (image['height'],image['width']),mode='bilinear',align_corners=False)[0,0].sigmoid().gt(.5).cpu().numpy()
                                if len(calls)!=before+1:
                                    raise RuntimeError('Unexpected geometry call count')
                                geo_call=calls[-1];_,use_points,use_box,use_ref=pilot.METHODS[name]
                                point_count=len(g['points']) if use_points else 0;box_count=int(use_box or use_ref)
                                if geo_call!=dict(points=point_count,boxes=box_count,encoded_shape=[1+point_count+box_count,1,256]):
                                    raise RuntimeError(f'Geometry input/token shape mismatch: {geo_call}')
                                forwards+=1;del out
                            if name=='text':baseline=candidate,score,index,geo_call
                            row.update(score=score,candidate_index=index,candidate_rle=encode(candidate),geometry_call=geo_call,
                                       **metrics(candidate,score,refs[side],refs[protocol.SIDES[1-protocol.SIDES.index(side)]]))
                        rows[name]=row;stream.write(json.dumps(row,allow_nan=False)+'\n');row_count+=1
                    if image['id'] in plan['render_ids'] or a.engineering:
                        render(a.output/'visualizations'/f'{image["recording_id"]}-{image["source_frame_index"]:06d}-{side}',rgb,refs[side],rows,g)
                stream.flush();del state
                if done%10==0 or done==len(assigned):
                    print(json.dumps(dict(frames=done,total=len(assigned),rows=row_count,forwards=forwards,
                                          elapsed_seconds=time.monotonic()-started)),flush=True)
        if any(p.requires_grad or p._version!=version for p,version in versions):
            raise RuntimeError('Frozen model mutated')
        for path,digest in {**hashes,**code,str(a.plan):signature['plan_sha256'],str(a.base_checkpoint):plan['base_sha256'],
                            str(a.tokenizer):plan['tokenizer_sha256']}.items():
            if pilot.sha256(path)!=digest:raise RuntimeError(f'Input/code changed: {path}')
        if row_count!=len(assigned)*len(protocol.SIDES)*len(protocol.METHODS):
            raise RuntimeError('Incomplete side/method coverage')
        pilot.write_json(a.output/'summary.json',dict(status='complete',protocol=signature,shard_index=a.shard_index,
            assigned_image_ids=[r['id'] for r in assigned],frames=len(assigned),rows=row_count,forwards=forwards,
            frozen_parameters_verified=True,records_sha256=pilot.sha256(a.output/'records.jsonl'),
            elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
    finally:
        handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','base-checkpoint','tokenizer','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--gpu-memory-fraction',type=float,default=.8)
    p.add_argument('--engineering',action='store_true')
    run(p.parse_args())

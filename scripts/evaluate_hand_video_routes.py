"""Independent, forward-only video-system evaluation of predeclared hand routes.

Each side starts its own fresh session; no GT prompts, no frame subsampling,
no changing tracker settings per method. --frame-limit is a labelled pilot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts import evaluate_realsense_full as full
from scripts.evaluate_hand_routes import selected_checkpoint, require_residual_epoch, protocol
from scripts.hand_evaluation_metrics import summarize_outputs, temporal_diagnostics
from scripts.video_hand_routes import VideoResidualTextEncoder, union_video_outputs, select_video_indices
from sam3.model.spatial_mask_adapter import attach_spatial_mask_adapter


def run(args):
    if args.output_dir.exists():
        raise ValueError('New output directory required')
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError('Invalid memory limit')
    if (args.method == 've' and any((args.adapter_checkpoint, args.delta_checkpoint, args.epoch,
                                   args.initial_cache, args.training_data_root))
            or args.method == 'spatial' and (not args.adapter_checkpoint or args.epoch is None
                or any((args.delta_checkpoint, args.initial_cache, args.training_data_root)))
            or args.method == 'residual' and (args.adapter_checkpoint or args.epoch is None
                or not all((args.delta_checkpoint, args.initial_cache, args.training_data_root)))):
        raise ValueError('Invalid method/checkpoint/cache/epoch combination')
    root = args.data_root.resolve()
    images, annotations, outputs, _, hashes = full.load_publication(root)
    publication_hash = hashlib.sha256(json.dumps(sorted(hashes.values())).encode()).hexdigest()
    selected = select_video_indices(images, args.recording, args.frame_limit)
    indices = [i for values in selected.values() for i in values]
    base_hash, token_hash = full.shared.sha256(args.base_checkpoint), full.shared.sha256(args.tokenizer_path)
    state = cache = metadata = None
    paths = [args.base_checkpoint, args.tokenizer_path]
    if args.method == 'spatial':
        state = selected_checkpoint(args.adapter_checkpoint, args.epoch, base_hash, token_hash)
        paths.append(args.adapter_checkpoint)
    if args.method == 'residual':
        require_residual_epoch(torch.load(args.delta_checkpoint, map_location='cpu', weights_only=True), args.epoch)
        cache, metadata = full.fixed.load_residual(args.delta_checkpoint, args.initial_cache,
            args.training_data_root, base_hash=base_hash, tokenizer_hash=token_hash)
        paths.extend([args.delta_checkpoint, args.initial_cache, args.training_data_root / 'annotations.json'])
    hashes.update({str(p.resolve()): full.shared.sha256(p) for p in paths})
    project = Path(__file__).resolve().parents[1]
    implementation = {str(p): full.shared.sha256(p) for p in
        [*sorted((project / 'sam3').rglob('*.py')), *sorted((project / 'scripts').glob('*.py'))]}
    spec = protocol(images, indices, base_hash, token_hash, full.shared.sha256(root / 'annotations.json'), 1)
    spec.update(evaluation_layer='video_system', selection='union of all accepted video output instances',
        confidence='video predictor filtering; no additional mask/GT selection',
        direction='forward', separate_side_sessions=True, temporal_disambiguation=True,
        start_frame=0, frame_stride=1, frame_limit=args.frame_limit,
        scope='bounded_continuous_pilot' if args.frame_limit is not None else 'complete_selected_recordings',
        recordings=list(selected), publication_sha256=publication_hash,
        implementation_sha256=hashlib.sha256(json.dumps({str(Path(p).relative_to(project)): d
            for p,d in sorted(implementation.items())}, sort_keys=True).encode()).hexdigest(),
        candidate_metrics='No separate candidate: these fields alias the actual system mask')
    visual_ids = {images[values[j]]['id'] for values in selected.values()
                  for j in sorted({0, len(values)//4, len(values)//2, 3*len(values)//4, len(values)-1})}
    spec['visual_image_ids'] = sorted(visual_ids)
    summary = dict(status='running', method=args.method, epoch=args.epoch, protocol=spec,
        residual_metadata=metadata, images=len(indices), queries=2*len(indices),
        checkpoint_sha256=full.shared.sha256(args.adapter_checkpoint or args.delta_checkpoint)
            if args.method != 've' else None,
        note='External development, SAM3-assisted references; not independent or full identity/HOI acceptance')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full.shared.atomic_write_json(args.output_dir / 'progress.json', summary)
    started = time.monotonic()
    predictor = None
    try:
        # Same published decoded RGB as image ablations; link only into new output.
        for name, values in selected.items():
            frame_dir = args.output_dir / 'input-frames' / name
            frame_dir.mkdir(parents=True)
            for number, i in enumerate(values):
                asset = outputs[images[i]['id']]['files']['rgb']
                source = full.local_file(root, asset['path'])
                if full.shared.sha256(source) != asset['sha256']:
                    raise ValueError('RGB changed')
                hashes[str(source)] = asset['sha256']
                (frame_dir / f'{number:06d}.png').symlink_to(source)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.manual_seed(123)
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        predictor = Sam3VideoPredictor(checkpoint_path=str(args.base_checkpoint),
            bpe_path=str(args.tokenizer_path), async_loading_frames=False,
            apply_temporal_disambiguation=True, compile=False, text_encoder_type='ve')
        detector = predictor.model.detector
        if state is not None:
            attach_spatial_mask_adapter(detector).load_state_dict(state['adapter'])
        if cache is not None:
            detector.backbone.language_backbone = VideoResidualTextEncoder(
                detector.backbone.language_backbone, cache.to('cuda'))
        predictor.model.eval().requires_grad_(False)
        versions = [(p,p._version) for p in predictor.model.parameters()]
        cache_hash = full.initializer.cache_fingerprint(cache.state_dict()) if cache is not None else None
        records, visuals = [], []
        with (args.output_dir / 'records.jsonl').open('x') as handle:
            for name, values in selected.items():
                for side in full.SIDES:
                    sid = predictor.handle_request(dict(type='start_session',
                        resource_path=str(args.output_dir / 'input-frames' / name),
                        offload_video_to_cpu=True, offload_state_to_cpu=True))['session_id']
                    seen = []
                    try:
                        predictor.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0,
                            text=full.cached.NATURAL_PROMPTS[full.SIDES.index(side)], output_prob_thresh=.5))
                        stream = predictor.handle_stream_request(dict(type='propagate_in_video', session_id=sid,
                            propagation_direction='forward', start_frame_index=0,
                            max_frame_num_to_track=len(values), output_prob_thresh=.5))
                        for response in stream:
                            frame = response['frame_index']
                            if type(frame) is not int or frame != len(seen) or frame >= len(values):
                                raise ValueError('Missing/duplicate/out-of-order video frame')
                            seen.append(frame)
                            index = values[frame]
                            image = images[index]
                            mask, ids, scores = union_video_outputs(response['outputs'], (image['height'],image['width']))
                            refs = full.batch_references(root, images, [index], annotations, outputs, hashes)[image['id']]
                            other = full.SIDES[1-full.SIDES.index(side)]
                            own_flags, other_flags = full.quality_flags(image, side), full.quality_flags(image, other)
                            row = dict(model=args.method, mode=args.method, dataset_role='external_development',
                                dataset_index=index, image_id=image['id'], identity_verified=True,
                                file_name=image['file_name'], recording_id=name, source_frame_index=frame,
                                prompt_key=side, prompt_text=full.cached.NATURAL_PROMPTS[full.SIDES.index(side)],
                                reference_provided=refs[side] is not None, other_reference_provided=refs[other] is not None,
                                has_both_reference=all(x is not None for x in refs.values()),
                                primary_test=refs[side] is not None and not own_flags,
                                reference_quality_flags=own_flags, pair_quality_flags=sorted(set(own_flags+other_flags)),
                                legacy_fixed128=image['legacy_fixed128'], output_object_ids=ids,
                                output_object_probabilities=scores,
                                prediction_rle=full.mask_utils.encode(np.asfortranarray(mask.astype('uint8'))),
                                **full.measure_partial(mask, refs[side], refs[other], float(mask.any())))
                            row['prediction_rle']['counts'] = row['prediction_rle']['counts'].decode('ascii')
                            # System accepted masks are not gated again by their scores.
                            row['top_confidence'] = max(scores, default=0.)
                            full.append_batch(handle, [row])
                            if image['id'] in visual_ids:
                                visuals.append(full.render_one(root,args.output_dir,image,side,refs[side],mask,row))
                            records.append({k:v for k,v in row.items() if k != 'prediction_rle'})
                            if (frame+1) % 24 == 0:
                                print(f'{args.method} {name} {side} {frame+1}/{len(values)}', flush=True)
                        if seen != list(range(len(values))):
                            raise ValueError('Incomplete video coverage')
                    finally:
                        predictor.handle_request(dict(type='close_session', session_id=sid))
        expected = {(images[i]['id'],s) for i in indices for s in full.SIDES}
        if len(records) != len(expected) or {(r['image_id'],r['prompt_key']) for r in records} != expected:
            raise ValueError('Incomplete video/side query coverage')
        if any(p._version != v for p,v in versions):
            raise RuntimeError('Frozen model weights changed')
        if cache is not None and full.initializer.cache_fingerprint(cache.state_dict()) != cache_hash:
            raise RuntimeError('Residual cache changed')
        for path, digest in {**hashes, **implementation}.items():
            if full.shared.sha256(Path(path)) != digest:
                raise RuntimeError('Input/code changed during video evaluation')
        summary.update(status='complete', elapsed_seconds=time.monotonic()-started,
            metrics=full.summarize(records), output_metrics=summarize_outputs(records),
            temporal_diagnostics=temporal_diagnostics(records), visuals=visuals,
            records_sha256=full.shared.sha256(args.output_dir/'records.jsonl'),
            actual_complete_query_coverage_verified=True)
        full.shared.atomic_write_json(args.output_dir/'summary.json', summary)
    except BaseException as error:
        full.shared.atomic_write_json(args.output_dir/'failure.json',
            {**summary, 'status':'failed', 'error':f'{type(error).__name__}: {error}'})
        raise
    finally:
        if predictor is not None:
            predictor.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','base-checkpoint','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('adapter-checkpoint','delta-checkpoint','initial-cache','training-data-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--tokenizer-path',type=Path,default=Path(__file__).resolve().parents[1]/'sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    parser.add_argument('--method',choices=('ve','spatial','residual'),required=True)
    parser.add_argument('--recording',action='append',required=True)
    parser.add_argument('--frame-limit',type=int)
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--gpu-memory-fraction',type=float,default=.5)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

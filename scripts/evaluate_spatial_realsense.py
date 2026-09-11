#!/usr/bin/env python3
"""Predeclared completed-epoch spatial/VE comparison on the fixed benchmark.

Defaults to original fixed128; --full uses all6204. No threshold or best picking.
Original VE is used directly at inference (not a trainable text replacement).
"""
import argparse
from pathlib import Path
import time
import torch
from scripts import evaluate_realsense_full as full
from scripts.spatial_training_state import validate_state
from sam3.model.spatial_mask_adapter import SpatialMaskAdapter, attach_spatial_mask_adapter


def selected_checkpoint(path, epoch, base_hash, tokenizer_hash):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state.get("config", {})
    validate_state(state, config, SpatialMaskAdapter().state_dict())
    if (type(epoch) is not int or epoch < 1 or config["base_sha256"] != base_hash
            or config["tokenizer_sha256"] != tokenizer_hash
            or state["step"] != epoch * config["steps_per_epoch"]
            or state["last_validation_step"] != state["step"]):
        raise ValueError("Require explicit completed AND validated epoch, matching base/tokenizer")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tokenizer-path', type=Path,
        default=Path(__file__).resolve().parents[1] / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    parser.add_argument('--adapter-checkpoint', type=Path)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.55)
    args = parser.parse_args()
    if bool(args.adapter_checkpoint) != (args.epoch is not None):
        parser.error('Supply adapter-checkpoint and epoch together, or neither for original VE')
    if args.batch_size < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid batch or memory fraction')
    if args.output_dir.exists():
        raise ValueError('New output directory required')
    root = args.data_root.resolve()
    images, annotations, outputs, plan, hashes = full.load_publication(root)
    if len(images) != 6204 or sum(x['legacy_fixed128'] for x in images) != 128:
        raise ValueError('Unexpected fixed benchmark identity')
    indices = [i for i, image in enumerate(images) if args.full or image['legacy_fixed128']]
    hashes.update({str(p.resolve()): full.shared.sha256(p) for p in (args.base_checkpoint, args.tokenizer_path)})
    state = None
    if args.adapter_checkpoint:
        state = selected_checkpoint(args.adapter_checkpoint, args.epoch,
            full.shared.sha256(args.base_checkpoint), full.shared.sha256(args.tokenizer_path))
        hashes[str(args.adapter_checkpoint.resolve())] = full.shared.sha256(args.adapter_checkpoint)
    project = Path(__file__).resolve().parents[1]
    implementation = {str(p): full.shared.sha256(p) for p in
        [*sorted((project / 'sam3').rglob('*.py')), *sorted((project / 'scripts').glob('*.py'))]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', method='spatial' if state else 've',
        scope='fixed_development_benchmark', images=len(indices), queries=2*len(indices),
        epoch=args.epoch, threshold=.5, mask_threshold=.5, implementation=implementation,
        checkpoint_sha256=full.shared.sha256(args.adapter_checkpoint) if state else None,
        note='SAM3-assisted reference, not independent manual ground truth; no model selection')
    full.shared.atomic_write_json(args.output_dir / 'progress.json', summary)
    started = time.monotonic()
    try:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.manual_seed(123)
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint), bpe_path=str(args.tokenizer_path),
            load_from_HF=False, device='cuda', eval_mode=True, enable_segmentation=True,
            enable_inst_interactivity=False, text_encoder_type='ve')
        if state:
            attach_spatial_mask_adapter(model).load_state_dict(state['adapter'])
        model.eval().requires_grad_(False)
        versions = [(p, p._version) for p in model.parameters()]
        raw = full.shared.make_dataset(root)
        if len(raw) != len(images): raise ValueError('Dataset length mismatch')
        dataset = full.semantic.IdentityCheckedDataset(raw, images)
        records, visuals = full.evaluate(model, 'spatial' if state else 've-both', dataset,
            images, indices, annotations, outputs, root, args.output_dir, hashes, args.batch_size)
        if any(p._version != v for p, v in versions): raise RuntimeError('Model changed')
        for path, digest in {**hashes, **implementation}.items():
            if full.shared.sha256(Path(path)) != digest: raise RuntimeError('Input/code changed')
        summary.update(status='complete', elapsed_seconds=time.monotonic()-started,
            metrics=full.summarize(records), visuals=visuals)
        full.shared.atomic_write_json(args.output_dir / 'summary.json', summary)
    except Exception as error:
        summary.update(status='failed', error=str(error))
        full.shared.atomic_write_json(args.output_dir / 'failure.json', summary)
        raise


if __name__ == '__main__':
    main()

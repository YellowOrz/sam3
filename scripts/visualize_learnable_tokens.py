#!/usr/bin/env python3
"""Render qualitative SAM3 right-hand masks for learned token checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def load_model(base_checkpoint: Path, token_checkpoint: Path, tokens_per_class: int):
    from sam3.model.learnable_text_encoder import LearnableClassTextEncoder
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(base_checkpoint),
        load_from_HF=False,
        device="cuda",
        # Keep the training matcher attached so we can associate the predicted
        # mask with the single right-hand ground-truth instance for rendering.
        eval_mode=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        text_encoder_type="learnable_class",
        tokens_per_class=tokens_per_class,
    )
    state = torch.load(token_checkpoint, map_location="cpu", weights_only=True)
    encoder = next(
        module
        for module in model.modules()
        if isinstance(module, LearnableClassTextEncoder)
    )
    with torch.no_grad():
        encoder.class_tokens.copy_(state["class_tokens"].to(encoder.class_tokens))
    model.eval()
    return model


def load_ve_model(base_checkpoint: Path):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(base_checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        text_encoder_type="ve",
    )
    model.eval()
    return model


def make_dataset(root: Path):
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
    from sam3.train.transforms.basic_for_api import (
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )
    from sam3.train.transforms.segmentation import DecodeRle

    return Sam3ImageDataset(
        img_folder=str(root),
        ann_file=str(root / "annotations.json"),
        transforms=[
            DecodeRle(),
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
        max_ann_per_img=100,
        multiplier=1,
        training=False,
        load_segmentation=True,
    )


def restore_rgb(tensor: torch.Tensor) -> np.ndarray:
    rgb = ((tensor.detach().cpu().float().permute(1, 2, 0) + 1.0) * 127.5).clamp(
        0, 255
    )
    return rgb.numpy().astype(np.uint8)


def matched_mask(model, batch, targets, output):
    prediction = output[0]
    indices = model.matcher(prediction, targets)
    if isinstance(indices, tuple) and len(indices) == 3:
        batch_idx, src, tgt = indices
        src = src[batch_idx == 0]
        tgt = torch.arange(len(src), device=src.device) if tgt is None else tgt
    else:
        raise TypeError(f"Unsupported matcher output: {type(indices)} {repr(indices)[:300]}")
    match = src[tgt == 0]
    if len(match) == 0:
        return None, None
    query = int(match[0])
    logits = prediction["pred_masks"][0, query]
    mask = torch.sigmoid(logits)[None, None]
    mask = F.interpolate(mask, size=(1008, 1008), mode="bilinear", align_corners=False)
    return mask[0, 0].cpu().numpy() > 0.5, float(prediction["pred_logits"][0, query, 0].sigmoid().cpu())


def render(rgb, gt, pred, title):
    out = rgb.astype(np.float32).copy()
    # Green = ground truth only, red = prediction only, yellow = overlap.
    fp = pred & ~gt
    fn = gt & ~pred
    both = pred & gt
    colors = [(fp, (255, 50, 50)), (fn, (50, 255, 80)), (both, (255, 220, 40))]
    for mask, color in colors:
        out[mask] = 0.45 * out[mask] + 0.55 * np.asarray(color, dtype=np.float32)
    image = Image.fromarray(out.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1008, 30), fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--k4-checkpoint", type=Path, required=True)
    parser.add_argument("--k1-checkpoint", type=Path, default=None)
    parser.add_argument("--include-ve", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(args.data_root)

    # Deterministic spread over the validation set: first, middle, last and evenly spaced.
    count = min(args.count, len(dataset))
    indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1] + [
        int(round(i * (len(dataset) - 1) / max(count - 1, 1))) for i in range(count)
    ]))

    models = {"k4": load_model(args.base_checkpoint, args.k4_checkpoint, 4)}
    if args.k1_checkpoint is not None:
        models["k1"] = load_model(args.base_checkpoint, args.k1_checkpoint, 1)
    if args.include_ve:
        models["ve"] = load_ve_model(args.base_checkpoint)

    runs = [(name, model, "right_hand") for name, model in models.items() if name != "ve"]
    if "ve" in models:
        runs.extend(
            [
                ("ve_right_hand", models["ve"], "right_hand"),
                ("ve_right_space", models["ve"], "right hand"),
            ]
        )

    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    manifest = []
    for index in indices:
        sample = dataset[index]
        batch = collate_fn_api([sample], dict_key="train", with_seg_masks=True)["train"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        targets = models["k4"].back_convert(batch.find_targets[0])
        gt = targets["masks"][0].cpu().numpy().astype(bool)
        rgb = restore_rgb(batch.img_batch[0])
        raw_name = f"val_index_{index:04d}"
        item = {"index": index, "output": raw_name, "gt_pixels": int(gt.sum())}

        rendered = []
        with torch.no_grad():
            for name, model, prompt in runs:
                batch.find_text_batch = [prompt]
                output = model(batch)
                pred, score = matched_mask(model, batch, targets, output)
                if pred is None:
                    item[name] = {"matched": False}
                    continue
                inter = int((pred & gt).sum())
                dice = (2.0 * inter) / max(int(pred.sum()) + int(gt.sum()), 1)
                item[name] = {
                    "matched": True,
                    "score": score,
                    "pred_pixels": int(pred.sum()),
                    "dice": dice,
                }
                rendered.append(
                    render(
                        rgb,
                        gt,
                        pred,
                        f"{name.upper()} | dice={dice:.3f} | score={score:.3f}",
                    )
                )

        if rendered:
            width = 1008 * len(rendered)
            canvas = Image.new("RGB", (width, 1008))
            for i, image in enumerate(rendered):
                canvas.paste(image, (i * 1008, 0))
            canvas.save(args.output_dir / f"{raw_name}.png")
        manifest.append(item)
        del sample, batch
        torch.cuda.empty_cache()

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "samples": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

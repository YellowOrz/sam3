#!/usr/bin/env python3
"""Train SAM3 learnable class tokens on an exported COCO image dataset."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch


def build_epoch_order(dataset_size: int, seed: int) -> list[int]:
    """Return one deterministic shuffled pass over every dataset item."""
    order = list(range(dataset_size))
    random.Random(seed).shuffle(order)
    return order


REQUIRED_CLASS_NAMES = ("left_hand", "right_hand")


def build_training_config(args) -> dict:
    """Return the optimization settings that must stay fixed when resuming."""
    return {
        "tokens_per_class": args.tokens_per_class,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "mask_weight": args.mask_weight,
        "dice_weight": args.dice_weight,
        "bbox_weight": args.bbox_weight,
        "giou_weight": args.giou_weight,
        "classification_weight": args.classification_weight,
        "presence_weight": args.presence_weight,
        "data_root": str(args.data_root.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
    }


def validate_resume_training_config(saved: dict, current: dict) -> None:
    """Reject a resume command that silently changes training semantics."""
    if saved == current:
        return
    changed = sorted(
        key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
    )
    raise ValueError(f"resume checkpoint 训练配置不一致: {', '.join(changed)}")


def inspect_training_annotations(annotation_path: Path) -> dict:
    """Validate the bilateral COCO contract used by token training."""
    raw = annotation_path.read_bytes()
    data = json.loads(raw)
    categories = data.get("categories")
    images = data.get("images")
    annotations = data.get("annotations")
    if not all(isinstance(value, list) for value in (categories, images, annotations)):
        raise ValueError("annotations.json must contain categories/images/annotations lists")

    category_names = {int(item["id"]): item["name"] for item in categories}
    if set(category_names.values()) != set(REQUIRED_CLASS_NAMES):
        raise ValueError(
            "Training data must declare exactly left_hand and right_hand categories"
        )
    image_ids = [int(image["id"]) for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Training data contains duplicate image ids")
    image_id_set = set(image_ids)
    annotations_per_image = Counter()
    positives = Counter()
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_id_set:
            raise ValueError(f"Annotation references unknown image id {image_id}")
        annotations_per_image[image_id] += 1
        if annotations_per_image[image_id] > 1:
            raise ValueError(
                f"DexYCB training expects at most one hand annotation per image: {image_id}"
            )
        try:
            positives[category_names[int(annotation["category_id"])]] += 1
        except KeyError as error:
            raise ValueError("Annotation references an unknown category") from error
    if any(positives[name] == 0 for name in REQUIRED_CLASS_NAMES):
        raise ValueError(f"Both hand sides need positive samples, got {dict(positives)}")

    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "images": len(images),
        "annotations": len(annotations),
        "negative_images": len(images) - len(annotations_per_image),
        "positive_images_by_class": {
            name: positives[name] for name in REQUIRED_CLASS_NAMES
        },
    }


def build_training_order(dataset_size: int, seed: int, epochs: int) -> list[int]:
    """Return deterministic shuffled passes, visiting every item once per epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    order = []
    for epoch in range(epochs):
        order.extend(build_epoch_order(dataset_size, seed + epoch))
    return order


def save_training_checkpoint(
    path: Path,
    token_encoder,
    optimizer,
    next_step: int,
    epoch_order: list[int],
    loss_history: list[float],
    initial_left_tokens: torch.Tensor,
    initial_class_tokens: torch.Tensor,
    annotation_summary: dict,
    gradient_nonzero_steps: list[int],
    args,
) -> None:
    """Atomically save enough state to resume the current epoch."""
    state = {
        "format": "sam3-learnable-class-tokens-v2",
        "class_names": list(token_encoder.CLASS_NAMES),
        "tokens_per_class": token_encoder.tokens_per_class,
        "d_model": token_encoder.d_model,
        "class_tokens": token_encoder.class_tokens.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "next_step": next_step,
        "epoch_order": epoch_order,
        "loss_history": loss_history,
        "initial_left_tokens": initial_left_tokens,
        "initial_class_tokens": initial_class_tokens,
        "annotation_summary": annotation_summary,
        "gradient_nonzero_steps": gradient_nonzero_steps,
        "training_config": build_training_config(args),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "epochs": args.epochs,
        "data_root": str(args.data_root),
        "base_checkpoint": str(args.base_checkpoint),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens-per-class", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use BF16 autocast for the frozen SAM3 forward/loss computation",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--bbox-weight", type=float, default=1.0)
    parser.add_argument("--giou-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens_per_class < 1:
        raise ValueError("tokens-per-class must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    annotation_path = args.data_root / "annotations.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    annotation_summary = inspect_training_annotations(annotation_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"k{args.tokens_per_class}_epoch{args.epochs}"
    latest_path = args.output_dir / f"{stem}_latest.pt"
    final_path = args.output_dir / f"{stem}_final.pt"
    summary_path = args.output_dir / f"{stem}_summary.json"

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    from sam3.model.learnable_text_encoder import (
        LearnableClassTextEncoder,
        freeze_for_learnable_class_tokens,
    )
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.transforms.basic_for_api import (
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )
    from sam3.train.transforms.segmentation import DecodeRle

    print("1. 创建训练数据集", flush=True)
    dataset = Sam3ImageDataset(
        img_folder=str(args.data_root),
        ann_file=str(args.data_root / "annotations.json"),
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
    print("训练集图片数量:", len(dataset), flush=True)
    print("训练数据契约:", json.dumps(annotation_summary, ensure_ascii=False), flush=True)

    print("2. 加载 SAM3", flush=True)
    model = build_sam3_image_model(
        checkpoint_path=str(args.base_checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        text_encoder_type="learnable_class",
        tokens_per_class=args.tokens_per_class,
    )
    model.eval()
    trainable_names = freeze_for_learnable_class_tokens(model)
    token_encoder = next(
        module
        for module in model.modules()
        if isinstance(module, LearnableClassTextEncoder)
    )
    optimizer = torch.optim.AdamW(
        [token_encoder.class_tokens],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    mask_loss_function = Masks(
        weight_dict={
            "loss_mask": args.mask_weight,
            "loss_dice": args.dice_weight,
        },
        compute_aux=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )
    box_loss_function = Boxes(
        weight_dict={
            "loss_bbox": args.bbox_weight,
            "loss_giou": args.giou_weight,
        },
        compute_aux=False,
    )
    classification_loss_function = IABCEMdetr(
        weight_dict={
            "loss_ce": args.classification_weight,
            "presence_loss": args.presence_weight,
        },
        compute_aux=False,
        pos_weight=5.0,
        alpha=0.25,
        gamma=2.0,
        weak_loss=False,
        use_presence=True,
        presence_alpha=0.5,
        presence_gamma=0.0,
        pos_focal=False,
    )
    print("token shape:", tuple(token_encoder.class_tokens.shape), flush=True)
    print("允许训练的参数:", sorted(trainable_names), flush=True)
    print(
        "允许训练的参数数量:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        flush=True,
    )

    epoch_order = build_training_order(len(dataset), args.seed, args.epochs)
    batch_spans = []
    for epoch_start in range(0, len(epoch_order), len(dataset)):
        epoch_end = min(epoch_start + len(dataset), len(epoch_order))
        batch_spans.extend(
            (start, min(start + args.batch_size, epoch_end))
            for start in range(epoch_start, epoch_end, args.batch_size)
        )
    start_step = 0
    loss_history: list[float] = []
    initial_class_tokens = token_encoder.class_tokens.detach().cpu().clone()
    initial_left_tokens = initial_class_tokens[0].clone()
    gradient_nonzero_steps = [0 for _ in REQUIRED_CLASS_NAMES]

    if args.resume is not None:
        print("3. 恢复训练:", args.resume, flush=True)
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_state.get("format") != "sam3-learnable-class-tokens-v2":
            raise ValueError("resume checkpoint 不是双手训练 v2 格式")
        validate_resume_training_config(
            resume_state.get("training_config", {}), build_training_config(args)
        )
        if resume_state["tokens_per_class"] != args.tokens_per_class:
            raise ValueError("resume checkpoint 的 K 与当前参数不一致")
        if resume_state["epoch_order"] != epoch_order:
            raise ValueError("resume checkpoint 的数据顺序与当前配置不一致")
        if resume_state["annotation_summary"]["sha256"] != annotation_summary["sha256"]:
            raise ValueError("resume checkpoint 的 annotations.json 指纹不一致")
        with torch.no_grad():
            token_encoder.class_tokens.copy_(
                resume_state["class_tokens"].to(token_encoder.class_tokens)
            )
        optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state["next_step"])
        loss_history = [float(value) for value in resume_state["loss_history"]]
        initial_class_tokens = resume_state["initial_class_tokens"].clone()
        initial_left_tokens = initial_class_tokens[0].clone()
        gradient_nonzero_steps = [
            int(value) for value in resume_state["gradient_nonzero_steps"]
        ]
    else:
        print(
            f"3. K={args.tokens_per_class} 左右手 token 从头随机初始化",
            flush=True,
        )

    total_steps = len(batch_spans)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    print(f"训练范围: step {start_step + 1} 到 {total_steps}", flush=True)
    print("4. 开始训练", flush=True)

    completed_steps = start_step
    training_started_at = time.monotonic()
    try:
        for step in range(start_step, total_steps):
            span_start, span_end = batch_spans[step]
            dataset_indices = epoch_order[span_start:span_end]
            samples = [dataset[index] for index in dataset_indices]
            batch = collate_fn_api(
                samples, dict_key="train", with_seg_masks=True
            )["train"]
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            targets = model.back_convert(batch.find_targets[0])
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.amp
            ):
                output = model(batch)
                prediction = output[0]
                if tuple(batch.find_text_batch) != REQUIRED_CLASS_NAMES:
                    raise RuntimeError(
                        "每张图必须同时生成 left_hand/right_hand 查询，实际为 "
                        f"{tuple(batch.find_text_batch)!r}"
                    )
                positive_queries = int((targets["num_boxes"] > 0).sum().item())
                if positive_queries < 0 or positive_queries > len(samples):
                    raise RuntimeError(
                        "DexYCB 每个样本应有 0 或 1 个正手别查询，"
                        f"批次实际为 {positive_queries}"
                    )
                prediction["indices"] = model.matcher(prediction, targets)
                num_boxes = targets["num_boxes"].sum().float().clamp(min=1)
                mask_losses = mask_loss_function(
                    outputs=prediction,
                    targets=targets,
                    indices=prediction["indices"],
                    num_boxes=num_boxes,
                )
                box_losses = box_loss_function(
                    outputs=prediction,
                    targets=targets,
                    indices=prediction["indices"],
                    num_boxes=num_boxes,
                )
                classification_losses = classification_loss_function(
                    outputs=prediction,
                    targets=targets,
                    indices=prediction["indices"],
                    num_boxes=num_boxes,
                )
                loss = (
                    mask_losses["core_loss"]
                    + box_losses["core_loss"]
                    + classification_losses["core_loss"]
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"step {step + 1}: loss 不是有限值")

            loss.backward()
            gradient = token_encoder.class_tokens.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError(f"step {step + 1}: token 梯度无效")
            gradient_norms = gradient.flatten(1).norm(dim=1).detach().cpu()
            for class_index, gradient_norm in enumerate(gradient_norms):
                if float(gradient_norm) > 0.0:
                    gradient_nonzero_steps[class_index] += 1
            if not bool((gradient_norms > 0).all()):
                raise RuntimeError(
                    f"step {step + 1}: 左右手 token 未同时获得梯度: "
                    f"{gradient_norms.tolist()}"
                )
            optimizer.step()

            completed_steps = step + 1
            loss_value = float(loss.detach().cpu())
            loss_history.append(loss_value)
            current_epoch = (span_end - 1) // len(dataset) + 1
            epoch_step = span_end - (current_epoch - 1) * len(dataset)

            if step == start_step or completed_steps % args.log_every == 0:
                recent = loss_history[-args.log_every :]
                print(
                    f"step={completed_steps:05d}/{total_steps} "
                    f"epoch={current_epoch}/{args.epochs} "
                    f"epoch_step={epoch_step}/{len(dataset)} "
                    f"batch_samples={len(samples)} "
                    f"sample_indices={dataset_indices[0]}..{dataset_indices[-1]} "
                    f"loss={loss_value:.6f} "
                    f"rolling={sum(recent) / len(recent):.6f} "
                    f"mask={float(mask_losses['loss_mask'].detach().cpu()):.6f} "
                    f"dice={float(mask_losses['loss_dice'].detach().cpu()):.6f} "
                    f"bbox={float(box_losses['loss_bbox'].detach().cpu()):.6f} "
                    f"giou={float(box_losses['loss_giou'].detach().cpu()):.6f} "
                    f"cls={float(classification_losses['loss_ce'].detach().cpu()):.6f} "
                    f"presence={float(classification_losses['presence_loss'].detach().cpu()):.6f} "
                    f"left_grad={float(gradient_norms[0]):.6e} "
                    f"right_grad={float(gradient_norms[1]):.6e} "
                    f"elapsed={time.monotonic() - training_started_at:.1f}s",
                    flush=True,
                )

            epoch_complete = span_end % len(dataset) == 0
            if completed_steps % args.save_every == 0 or epoch_complete:
                save_training_checkpoint(
                    latest_path,
                    token_encoder,
                    optimizer,
                    completed_steps,
                    epoch_order,
                    loss_history,
                    initial_left_tokens,
                    initial_class_tokens,
                    annotation_summary,
                    gradient_nonzero_steps,
                    args,
                )
                print("恢复点已保存:", latest_path, flush=True)
                if epoch_complete:
                    epoch_path = (
                        args.output_dir
                        / f"k{args.tokens_per_class}_epoch{current_epoch}_complete.pt"
                    )
                    save_training_checkpoint(
                        epoch_path,
                        token_encoder,
                        optimizer,
                        completed_steps,
                        epoch_order,
                        loss_history,
                        initial_left_tokens,
                        initial_class_tokens,
                        annotation_summary,
                        gradient_nonzero_steps,
                        args,
                    )
                    print(f"第 {current_epoch} 轮完成 checkpoint: {epoch_path}", flush=True)

            del (
                samples,
                batch,
                output,
                prediction,
                targets,
                mask_losses,
                box_losses,
                classification_losses,
                loss,
            )
            if completed_steps % 100 == 0:
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        save_training_checkpoint(
            latest_path,
            token_encoder,
            optimizer,
            completed_steps,
            epoch_order,
            loss_history,
            initial_left_tokens,
            initial_class_tokens,
            annotation_summary,
            gradient_nonzero_steps,
            args,
        )
        print("训练被中断，恢复点已保存:", latest_path, flush=True)
        raise

    class_token_changes = (
        token_encoder.class_tokens.detach().cpu() - initial_class_tokens
    ).abs().flatten(1).max(dim=1).values
    if not bool((class_token_changes > 0).all()):
        raise RuntimeError(
            f"左右手 token 必须都发生更新，实际为 {class_token_changes.tolist()}"
        )

    save_training_checkpoint(
        final_path,
        token_encoder,
        optimizer,
        completed_steps,
        epoch_order,
        loss_history,
        initial_left_tokens,
        initial_class_tokens,
        annotation_summary,
        gradient_nonzero_steps,
        args,
    )
    window = min(100, len(loss_history))
    first_average = sum(loss_history[:window]) / window
    last_average = sum(loss_history[-window:]) / window
    summary = {
        "experiment": f"DexYCB bilateral-hand K={args.tokens_per_class}",
        "steps": completed_steps,
        "samples_seen": batch_spans[completed_steps - 1][1] if completed_steps else 0,
        "dataset_size": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "annotation_summary": annotation_summary,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "first_average_loss": first_average,
        "last_average_loss": last_average,
        "loss_decreased": last_average < first_average,
        "class_token_max_change": {
            name: float(class_token_changes[index])
            for index, name in enumerate(REQUIRED_CLASS_NAMES)
        },
        "class_gradient_nonzero_steps": {
            name: gradient_nonzero_steps[index]
            for index, name in enumerate(REQUIRED_CLASS_NAMES)
        },
        "final_checkpoint": str(final_path),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "training_elapsed_seconds": time.monotonic() - training_started_at,
        "training_steps_per_second": (
            completed_steps - start_step
        ) / max(time.monotonic() - training_started_at, 1e-9),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"5. {args.epochs} 轮训练完成", flush=True)
    print("前 100 步平均 loss:", first_average, flush=True)
    print("后 100 步平均 loss:", last_average, flush=True)
    print("loss 是否下降:", last_average < first_average, flush=True)
    print("左右手 token 最大变化:", summary["class_token_max_change"], flush=True)
    print("最终 checkpoint:", final_path, flush=True)
    print("训练摘要:", summary_path, flush=True)
    print("GPU 峰值显存 MiB:", summary["peak_gpu_memory_mib"], flush=True)


if __name__ == "__main__":
    main()

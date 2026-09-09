"""Single-device, single-target feature training using SAM3 data and losses."""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.coco_json_loaders import ann_to_rle, COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.data.uni_hoi import UniHoiTargetCOCO
from sam3.train.loss.loss_fns import Boxes, CORE_LOSS_KEY, IABCEMdetr, Masks
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryOneToManyMatcher
from sam3.train.transforms.basic_for_api import (
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.transforms.filter_query_transforms import (
    FilterCrowds,
    FlexibleFilterFindGetQueries,
)
from sam3.train.transforms.segmentation import DecodeRle
from torch.utils.data import DataLoader


class TargetCOCO(COCO_FROM_JSON):
    """Restrict the native loader to one category, retaining negative images."""

    def __init__(self, annotation_file: str, category_id: int, target_id: str) -> None:
        super().__init__(annotation_file, include_negatives=True)
        if category_id not in self._cat_idx_to_text:
            raise ValueError(f"Category {category_id} is absent from {annotation_file}")
        self.category_chunks = [[category_id]]
        self._cat_idx_to_text = {category_id: target_id}
        from pycocotools import mask as mask_util

        for item in self._raw_data:
            item["annotations"] = [
                ann for ann in item["annotations"] if ann["category_id"] == category_id
            ]
            for ann in item["annotations"]:
                ann.setdefault("iscrowd", 0)
                if not ann.get("segmentation"):
                    raise ValueError(f"Selected annotation {ann['id']} has no mask")
                if "bbox" not in ann:
                    rle = ann_to_rle(ann["segmentation"], item["image"])
                    ann["bbox"] = mask_util.toBbox(rle).tolist()


def make_loader(config: DictConfig, split: str, target_id: str) -> DataLoader:
    spec = config[split]
    category_id = int(config.category_id)
    if spec.get("annotations"):
        img_folder = spec["images"]
        ann_file = spec["annotations"]
        coco_json_loader = partial(
            TargetCOCO, category_id=category_id, target_id=target_id
        )
    else:
        root = str(Path(str(config.dataset_root)).expanduser())
        img_folder = root
        ann_file = str(Path(root) / "metadata" / "split.json")
        coco_json_loader = partial(
            UniHoiTargetCOCO,
            category_id=category_id,
            target_id=target_id,
            dataset_root=root,
            split_name=str(spec["split"]),
            kind=str(config.kind),
        )
    dataset = Sam3ImageDataset(
        img_folder=img_folder,
        ann_file=ann_file,
        coco_json_loader=coco_json_loader,
        transforms=[
            FlexibleFilterFindGetQueries(query_filter=FilterCrowds()),
            DecodeRle(),
            RandomResizeAPI(sizes=1008, consistent_transform=False, square=True),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3),
        ],
        max_ann_per_img=200,
        multiplier=1,
        training=split == "train",
        load_segmentation=True,
    )
    if not len(dataset):
        raise ValueError(f"Empty {split} dataset")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=split == "train",
        num_workers=config.num_workers,
        collate_fn=partial(collate_fn_api, dict_key="target", with_seg_masks=True),
    )


def make_loss(model: torch.nn.Module, weights: DictConfig) -> Sam3LossWrapper:
    return Sam3LossWrapper(
        normalization="local",  # no distributed process group needed
        matcher=model.matcher,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        scale_by_find_batch_size=True,
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": weights.box, "loss_giou": weights.giou}),
            IABCEMdetr(
                pos_weight=5.0,
                weak_loss=False,
                gamma=2,
                alpha=0.25,
                use_presence=True,
                pad_n_queries=200,
                weight_dict={
                    "loss_ce": weights.classification,
                    "presence_loss": weights.presence,
                },
            ),
            Masks(weight_dict={"loss_mask": weights.mask, "loss_dice": weights.dice}),
        ],
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Sam3LossWrapper,
    config: DictConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    # Retain matching/DAC training outputs without updating frozen visual/BN state.
    model.backbone.vision_backbone.eval()
    for module in model.modules():
        if isinstance(
            module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)
        ):
            module.eval()
    totals, count = {}, 0
    for batch in loader:
        batch = copy_data_to_device(batch["target"], config.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=torch.device(config.device).type,
            dtype=torch.bfloat16,
            enabled=config.amp,
        ):
            outputs = model(batch)
            targets = [model.back_convert(target) for target in batch.find_targets]
            if not training:
                for output, target in zip(outputs, targets):
                    model._compute_matching(output, target)
            losses = loss_fn(outputs, targets)
            loss = losses[CORE_LOSS_KEY]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss: {loss.item()}")
        if training:
            loss.backward()
            features = model.backbone.learned_prompt.features
            if features.grad is None or not torch.isfinite(features.grad).all():
                raise RuntimeError("Learned feature gradient is missing or nonfinite")
            torch.nn.utils.clip_grad_norm_([features], config.max_grad_norm)
            optimizer.step()
        size = batch.img_batch.shape[0]
        count += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
    return {key: value / count for key, value in totals.items()}


def train(config: DictConfig, resume: str | None = None) -> None:
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0:
        raise ValueError(
            "epochs/batch_size must be positive and num_workers nonnegative"
        )
    if config.lr <= 0 or config.max_grad_norm <= 0:
        raise ValueError("lr and max_grad_norm must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(config.output_dir)
    if not resume and (output / "last.pt").exists():
        raise ValueError("Existing run: use --resume or choose a new output_dir")
    model = build_sam3_image_model(
        checkpoint_path=config.checkpoint,
        load_from_HF=False,
        learned_prompt_path=resume or config.initial_feature,
        eval_mode=False,
        device=config.device,
    ).to(config.device)
    prompt = model.backbone.learned_prompt
    prompt.require_target(str(config.target_id))
    previous_category = prompt.metadata.get("category_id")
    if previous_category is not None and previous_category != int(config.category_id):
        raise ValueError("Feature was trained for a different dataset category_id")
    prompt.metadata["category_id"] = int(config.category_id)
    if config.get("kind"):
        previous_kind = prompt.metadata.get("kind")
        if previous_kind is not None and previous_kind != str(config.kind):
            raise ValueError("Feature was trained for a different uni-hoi kind")
        prompt.metadata["kind"] = str(config.kind)
    trainable = [param for param in model.parameters() if param.requires_grad]
    if len(trainable) != 1 or trainable[0] is not prompt.features:
        raise RuntimeError("Only the selected target feature may be trainable")
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.0)
    start_epoch, best_loss = 0, float("inf")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if "training_state" not in saved:
            raise ValueError("--resume requires last.pt, including optimizer state")
        state = saved["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_loss = state["epoch"], state["best_loss"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None and torch.device(config.device).type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device=config.device)
    if start_epoch >= config.epochs:
        raise ValueError("epochs must exceed the completed epochs in the resume file")
    train_loader = make_loader(config, "train", prompt.target_id)
    val_loader = (
        make_loader(config, "val", prompt.target_id) if config.get("val") else None
    )
    loss_fn = make_loss(model, config.loss)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml")
    print(f"Training target {prompt.target_id}: {prompt.features.numel()} parameters")
    for epoch in range(start_epoch, config.epochs):
        train_metrics = run_epoch(model, train_loader, loss_fn, config, optimizer)
        val_metrics = (
            run_epoch(model, val_loader, loss_fn, config) if val_loader else None
        )
        if val_metrics is not None and val_metrics[CORE_LOSS_KEY] < best_loss:
            best_loss = val_metrics[CORE_LOSS_KEY]
            prompt.save(output / "best.pt")
        prompt.save(output / "learned_prompt.pt")
        prompt.save(
            output / "last.pt",
            training_state={
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "cuda_rng": (
                    torch.cuda.get_rng_state(config.device)
                    if torch.device(config.device).type == "cuda"
                    else None
                ),
            },
        )
        metrics = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(OmegaConf.load(args.config), resume=args.resume)


if __name__ == "__main__":
    main()

"""Single-target feature training using SAM3 data and losses."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.coco_json_loaders import ann_to_rle, COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.data.uni_hoi import UniHoiTargetCOCO
from sam3.train.loss.loss_fns import Boxes, CORE_LOSS_KEY, IABCEMdetr, Masks
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.utils.logger import make_tensorboard_logger
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
    sampler = (
        DistributedSampler(dataset, shuffle=True)
        if split == "train" and dist.is_available() and dist.is_initialized()
        else None
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None and split == "train",
        sampler=sampler,
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


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _cuda_ids(device: str) -> list[int] | None:
    text = str(device).strip()
    if text == "cpu" or not text.startswith("cuda"):
        return None
    _, _, rest = text.partition(":")
    if not rest:
        return [0]
    return [int(part.strip()) for part in rest.split(",") if part.strip()]


def _rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _run_vision_as_constant(backbone: torch.nn.Module) -> None:
    # Frozen ViT uses inference fused kernels; only the learned tokens need autograd.
    if getattr(backbone, "_learned_prompt_frozen_vision", False):
        return
    encode = backbone.forward_image

    def forward_image(samples, *args, **kwargs):
        with torch.no_grad():
            return encode(samples, *args, **kwargs)

    backbone.forward_image = forward_image
    backbone._learned_prompt_frozen_vision = True


def _tb_payload(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Sam3LossWrapper,
    config: DictConfig,
    optimizer: torch.optim.Optimizer | None = None,
    progress_desc: str = "",
    tb_logger=None,
    tb_step: int = 0,
) -> tuple[dict, int]:
    training = optimizer is not None
    core = _unwrap(model)
    model.train(training)
    # Retain matching/DAC training outputs without updating frozen visual/BN state.
    core.backbone.vision_backbone.eval()
    _run_vision_as_constant(core.backbone)
    for module in core.modules():
        if isinstance(
            module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)
        ):
            module.eval()
    totals, count = {}, 0
    interval = int(config.get("tensorboard_interval", 100))
    batch_index = 0
    real_out, real_err = sys.stdout, sys.stderr
    try:
        if _rank0():
            sys.stdout = sys.stderr = DummyTqdmFile(real_err)
        with tqdm(
            loader,
            desc=progress_desc,
            disable=not _rank0(),
            leave=False,
            dynamic_ncols=True,
            position=0,
            file=real_err,
        ) as batches:
            for batch in batches:
                batch = copy_data_to_device(batch["target"], config.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.autocast(
                    device_type=torch.device(config.device).type,
                    dtype=torch.bfloat16,
                    enabled=config.amp,
                ):
                    outputs = model(batch)
                    targets = [core.back_convert(target) for target in batch.find_targets]
                    if not training:
                        for output, target in zip(outputs, targets):
                            core._compute_matching(output, target)
                    losses = loss_fn(outputs, targets)
                    loss = losses[CORE_LOSS_KEY]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {loss.item()}")
                if training:
                    loss.backward()
                    features = core.backbone.learned_prompt.features
                    if features.grad is None or not torch.isfinite(features.grad).all():
                        raise RuntimeError(
                            "Learned feature gradient is missing or nonfinite"
                        )
                    torch.nn.utils.clip_grad_norm_([features], config.max_grad_norm)
                    optimizer.step()
                size = batch.img_batch.shape[0]
                count += size
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
                if training:
                    tb_step += 1
                    batch_index += 1
                    if (
                        tb_logger is not None
                        and interval > 0
                        and batch_index % interval == 0
                    ):
                        tb_logger.log_dict(
                            _tb_payload(
                                "train",
                                {key: value / count for key, value in totals.items()},
                            ),
                            tb_step,
                        )
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (totals, count))
        totals, count = {}, 0
        for part, size in gathered:
            count += size
            for key, value in part.items():
                totals[key] = totals.get(key, 0.0) + value
    if count == 0:
        raise RuntimeError("No samples in epoch")
    return {key: value / count for key, value in totals.items()}, tb_step


def train(config: DictConfig, resume: str | None = None) -> None:
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0:
        raise ValueError(
            "epochs/batch_size must be positive and num_workers nonnegative"
        )
    if config.lr <= 0 or config.max_grad_norm <= 0:
        raise ValueError("lr and max_grad_norm must be positive")
    if int(config.get("tensorboard_interval", 100)) < 1:
        raise ValueError("tensorboard_interval must be positive")
    requested_device = str(config.device)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        config.device = f"cuda:{local_rank}"
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
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
    if world_size > 1:
        index = torch.device(config.device).index
        model = DistributedDataParallel(
            model,
            device_ids=[index],
            output_device=index,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    prompt = _unwrap(model).backbone.learned_prompt
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
    start_epoch, best_loss, tb_step = 0, float("inf"), 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if "training_state" not in saved:
            raise ValueError("--resume requires last.pt, including optimizer state")
        state = saved["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_loss = state["epoch"], state["best_loss"]
        tb_step = int(state.get("tb_step", 0))
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
    loss_fn = make_loss(_unwrap(model), config.loss)
    if _rank0():
        output.mkdir(parents=True, exist_ok=True)
        saved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        saved.device = requested_device
        OmegaConf.save(saved, output / "config.yaml")
        tqdm.write(
            f"Training target {prompt.target_id}: {prompt.features.numel()} parameters"
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    tb_logger = make_tensorboard_logger(str(output / "tensorboard"))
    try:
        for epoch in range(start_epoch, config.epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            epoch_id = epoch + 1
            train_metrics, tb_step = run_epoch(
                model,
                train_loader,
                loss_fn,
                config,
                optimizer,
                progress_desc=f"epoch {epoch_id}/{config.epochs} train",
                tb_logger=tb_logger,
                tb_step=tb_step,
            )
            val_metrics = None
            if val_loader:
                val_metrics, tb_step = run_epoch(
                    model,
                    val_loader,
                    loss_fn,
                    config,
                    progress_desc=f"epoch {epoch_id}/{config.epochs} val",
                    tb_step=tb_step,
                )
            tb_logger.log_dict(_tb_payload("train", train_metrics), tb_step)
            if val_metrics is not None:
                tb_logger.log_dict(_tb_payload("val", val_metrics), tb_step)
            tb_logger.flush()
            if _rank0():
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
                        "tb_step": tb_step,
                        "torch_rng": torch.get_rng_state(),
                        "python_rng": random.getstate(),
                        "cuda_rng": (
                            torch.cuda.get_rng_state(config.device)
                            if torch.device(config.device).type == "cuda"
                            else None
                        ),
                    },
                )
                metrics = {"epoch": epoch_id, "train": train_metrics, "val": val_metrics}
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                tqdm.write(json.dumps(metrics))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _spawn_train(rank: int, world_size: int, config_dict: dict, resume: str | None) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    train(OmegaConf.create(config_dict), resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    ids = _cuda_ids(config.device)
    if ids is not None and len(ids) > 1 and "LOCAL_RANK" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in ids)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sock:
                sock.bind(("", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
        torch.multiprocessing.spawn(
            _spawn_train,
            args=(
                len(ids),
                OmegaConf.to_container(config, resolve=True),
                args.resume,
            ),
            nprocs=len(ids),
            join=True,
        )
        return
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()

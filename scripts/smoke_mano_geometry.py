#!/usr/bin/env python3
"""Run exactly three engineering smoke cases for opt-in MANO token injection.

This is NOT a training experiment or evidence of better geometry/segmentation.
It uses ground-truth MANO parameters as oracle inputs: one visible valid left
hand, one visible valid right hand and one no-mask image (prefer invalid MANO).
Each case uses one RGB image, both text queries, and identical MANO input for
the two queries. Physical side is never used to route MANO to the correct query.

Load base and epoch-token weights before explicitly wrapping geometry_encoder.
Only ManoGeometryEncoder parameters may receive gradients/optimizer updates.
Invalid MANO arrays become NaN placeholders with valid=False, never valid zeros.
An all-invalid case may have zero adapter gradient; its optimizer step is then
skipped so Adam momentum cannot masquerade as an effective MANO update.

Writes smoke_summary.json, per-step progress.json and adapter_smoke.pt under a
new --output-dir. The saved adapter is only a smoke-reproduction artifact.
CUDA_VISIBLE_DEVICES must select the GPU externally. The default allocator cap
is 20% of that GPU; this is a memory bound, not compute/scheduling isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch

# Prefer the reviewed checkout when invoked directly as `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_mano_sidecar import (
    DEFAULT_DATA_ROOT, FORMAT as SIDECAR_FORMAT, PARAMETER_SIZES,
    SIDES, finite_vector, load_coco, sequence_side,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_smoke_rows(rows, seed: int) -> list[tuple[str, dict]]:
    """Deterministic reservoir selection; prefer invalid MANO for empty case."""
    rng = random.Random(seed)
    choices, counts = {}, {}
    for row in rows:
        hands = row.get("hands")
        if not isinstance(hands, list) or len(hands) != 1:
            raise ValueError("Smoke requires exactly one physical hand slot per image")
        hand = hands[0]
        if type(hand.get("valid")) is not bool or hand.get("side") not in SIDES:
            raise ValueError("Invalid MANO validity/physical side")
        if row["segmentation_target_present"] and hand["valid"]:
            group = hand["side"]
        elif not row["segmentation_target_present"]:
            group = "empty_valid" if hand["valid"] else "empty_invalid"
        else:
            continue
        counts[group] = counts.get(group, 0) + 1
        if rng.randrange(counts[group]) == 0:
            choices[group] = row
    if "left" not in choices or "right" not in choices:
        raise ValueError("Need visible MANO-valid left and right examples")
    empty = choices.get("empty_invalid", choices.get("empty_valid"))
    if empty is None:
        raise ValueError("Need a no-mask image for the negative smoke case")
    return [("left_valid", choices["left"]), ("right_valid", choices["right"]), ("empty", empty)]


def validate_row_identity(row: dict, image: dict, annotations: list[dict], split: str):
    if row.get("format") != SIDECAR_FORMAT or row.get("split") != split:
        raise ValueError("Sidecar format/split mismatch")
    for key in ("file_name", "source", "sequence", "view", "frame_index"):
        if row.get(key) != image.get(key):
            raise ValueError(f"Sidecar/COCO {key} mismatch for image {image['id']}")
    if row.get("source_frame") != image["frame_index"] or row.get("source_frame_mapping") != "dexycb_contiguous_zero_based_identity":
        raise ValueError("Unsupported source-frame mapping")
    if row.get("segmentation_target_present") is not bool(annotations):
        raise ValueError("Sidecar/COCO segmentation visibility mismatch")
    if row.get("segmentation_annotation_ids") != [annotation["id"] for annotation in annotations]:
        raise ValueError("Sidecar/COCO annotation IDs mismatch")
    hands = row.get("hands")
    if not isinstance(hands, list) or len(hands) != 1:
        raise ValueError("Need one MANO physical hand slot")
    side = hands[0].get("side")
    if side not in SIDES or any(annotation["category_id"] != SIDES.index(side) + 1 for annotation in annotations):
        raise ValueError("Sidecar physical side contradicts COCO")


def load_inputs(data_root: Path, sidecar_root: Path, split: str, seed: int):
    summary_path = sidecar_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("format") != SIDECAR_FORMAT or summary.get("sources_rechecked_unchanged") is not True:
        raise ValueError("Need a successfully published, source-verified sidecar")
    annotation_path = data_root / split / "annotations.json"
    sidecar_path = sidecar_root / f"{split}.mano.jsonl"
    split_summary = summary["splits"][split]
    annotation_hash = file_sha256(annotation_path)
    sidecar_hash = file_sha256(sidecar_path)
    if annotation_hash != split_summary["annotations_sha256"]:
        raise ValueError("COCO annotations SHA256 differs from sidecar")
    if sidecar_hash != split_summary["sidecar_sha256"]:
        raise ValueError("Sidecar SHA256 differs from its summary")
    images, annotations = load_coco(json.loads(annotation_path.read_text(encoding="utf-8")))
    images_by_id = {image["id"]: image for image in images}
    seen = set()

    def checked_rows():
        with sidecar_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                image_id = row["image_id"]
                if image_id in seen or image_id not in images_by_id:
                    raise ValueError(f"Duplicate/unknown sidecar image id: {image_id}")
                seen.add(image_id)
                validate_row_identity(row, images_by_id[image_id], annotations[image_id], split)
                yield row

    selected = select_smoke_rows(checked_rows(), seed)
    if seen != set(images_by_id):
        raise ValueError("Smoke requires complete sidecar coverage of the selected split")
    verified_sources = {}
    for _, row in selected:
        for descriptor in row["source_files"].values():
            path = Path(descriptor["path"])
            if descriptor["exists"]:
                if file_sha256(path) != descriptor["sha256"]:
                    raise ValueError(f"Source changed since sidecar export: {path}")
            elif path.exists():
                raise ValueError(f"Previously missing source now exists: {path}")
            verified_sources[str(path)] = descriptor
        sequence = json.loads(Path(row["source_files"]["sequence"]["path"]).read_text())
        authority = sequence_side(sequence, row["source"], row["sequence"], row["view"])
        if row["hands"][0]["side"] != authority:
            raise ValueError("Selected side differs from sequence.extra.mano_sides")
    provenance = {
        "annotations": {"path": str(annotation_path.resolve()), "sha256": annotation_hash},
        "sidecar": {"path": str(sidecar_path.resolve()), "sha256": sidecar_hash},
        "sidecar_summary": {"path": str(summary_path.resolve()), "sha256": file_sha256(summary_path)},
        "selected_source_files": list(verified_sources.values()),
    }
    return images, selected, provenance


def mano_for_two_queries(row: dict, img_ids: torch.Tensor, text_ids: torch.Tensor, device) -> dict:
    """Duplicate physical-hand data identically; text_ids never route geometry."""
    if img_ids.tolist() != [0, 0] or sorted(text_ids.tolist()) != [0, 1]:
        raise ValueError("Smoke requires batch size 1 and exactly the two side queries")
    hand = row["hands"][0]
    valid = hand["valid"]
    if type(valid) is not bool or hand["side"] not in SIDES:
        raise ValueError("Invalid MANO validity/side")
    if valid and (hand["root_frame"] != "camera" or hand["pose_representation"] != "axis-angle"):
        raise ValueError("Valid MANO must be camera-frame axis-angle")
    fields = {}
    for name, dimension in PARAMETER_SIZES.items():
        values = hand[name]
        if valid and not finite_vector(values, dimension):
            raise ValueError(f"Invalid valid-MANO {name}")
        if not valid and values is not None:
            raise ValueError("Invalid sidecar MANO arrays must be null")
        values = values if valid else [float("nan")] * dimension
        fields[name] = torch.tensor([values, values], dtype=torch.float32, device=device)
    fields["side"] = torch.full((2,), SIDES.index(hand["side"]), dtype=torch.long, device=device)
    fields["valid"] = torch.full((2,), valid, dtype=torch.bool, device=device)
    return fields


def adapter_gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Adapter gradient is nonfinite")
            squared += float(gradient.square().sum().cpu())
    return math.sqrt(squared)


def adapter_snapshot(adapter) -> dict:
    return {name: parameter.detach().cpu().clone() for name, parameter in adapter.named_parameters()}


def parameter_max_change(adapter, before: dict) -> float:
    return max(float((parameter.detach().cpu() - before[name]).abs().max())
               for name, parameter in adapter.named_parameters())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_DATA_ROOT / "mano-sidecar-v1")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--token-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("--gpu-memory-fraction must be in (0, 1]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new smoke output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU helper tests do not run the SAM3 model")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from scripts.evaluate_bilateral_tokens import CLASS_NAMES, make_dataset, validate_batch_identity
    from sam3.model.learnable_text_encoder import LearnableClassTextEncoder
    from sam3.model.mano_geometry_encoder import freeze_for_mano_geometry_encoder
    from sam3.model.mano_prompt_adapter import ManoAugmentedGeometryEncoder, ManoPrompt
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks

    images, selected, provenance = load_inputs(args.data_root, args.sidecar_root, args.split, args.seed)
    provenance["base_checkpoint"] = {"path": str(args.base_checkpoint.resolve()), "sha256": file_sha256(args.base_checkpoint)}
    provenance["token_checkpoint"] = {"path": str(args.token_checkpoint.resolve()), "sha256": file_sha256(args.token_checkpoint)}
    state = torch.load(args.token_checkpoint, map_location="cpu", weights_only=True)
    class_tokens = state.get("class_tokens")
    if state.get("class_names") != list(CLASS_NAMES) or not isinstance(class_tokens, torch.Tensor):
        raise ValueError("Token checkpoint must contain bilateral class tokens")
    if class_tokens.ndim != 3 or class_tokens.shape[0] != 2 or class_tokens.shape[-1] != 256 or not torch.isfinite(class_tokens).all():
        raise ValueError("Expected finite token tensor [2,K,256]")
    model = build_sam3_image_model(
        checkpoint_path=str(args.base_checkpoint), load_from_HF=False, device="cuda",
        eval_mode=False, enable_segmentation=True, enable_inst_interactivity=False,
        text_encoder_type="learnable_class", tokens_per_class=int(class_tokens.shape[1]),
    )
    token_encoder = next(module for module in model.modules() if isinstance(module, LearnableClassTextEncoder))
    with torch.no_grad():
        token_encoder.class_tokens.copy_(class_tokens.to(token_encoder.class_tokens))
    model.geometry_encoder = ManoAugmentedGeometryEncoder(model.geometry_encoder).cuda()
    trainable_names = freeze_for_mano_geometry_encoder(model)
    model.eval()
    adapter = model.geometry_encoder.mano_encoder
    parameters = list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    initial = adapter_snapshot(adapter)
    initial_tokens = token_encoder.class_tokens.detach().cpu().clone()
    mask_loss_fn = Masks(weight_dict={"loss_mask": 1.0, "loss_dice": 1.0}, compute_aux=False, focal_alpha=0.25, focal_gamma=2.0)
    box_loss_fn = Boxes(weight_dict={"loss_bbox": 1.0, "loss_giou": 1.0}, compute_aux=False)
    class_loss_fn = IABCEMdetr(
        weight_dict={"loss_ce": 1.0, "presence_loss": 1.0}, compute_aux=False,
        pos_weight=5.0, alpha=0.25, gamma=2.0, weak_loss=False,
        use_presence=True, presence_alpha=0.5, presence_gamma=0.0, pos_focal=False,
    )
    dataset = make_dataset(args.data_root / args.split)
    image_indices = {image["id"]: index for index, image in enumerate(images)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    steps = []
    started = time.monotonic()
    for case, row in selected:
        dataset_index = image_indices[row["image_id"]]
        batch = collate_fn_api([dataset[dataset_index]], dict_key="smoke", with_seg_masks=True)["smoke"]
        validate_batch_identity(batch, [dataset_index], images)
        if tuple(batch.find_text_batch) != CLASS_NAMES:
            raise RuntimeError("Both original side text queries must be present")
        mano = mano_for_two_queries(row, batch.find_inputs[0].img_ids, batch.find_inputs[0].text_ids, torch.device("cuda"))
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        stage = batch.find_inputs[0]
        prompt = ManoPrompt(mano=mano, box_embeddings=stage.input_boxes,
                            box_mask=stage.input_boxes_mask, box_labels=stage.input_boxes_label)
        targets = model.back_convert(batch.find_targets[0])
        optimizer.zero_grad(set_to_none=True)
        before = adapter_snapshot(adapter)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
            # Image and text encoders are frozen; grounding remains differentiable.
            with torch.no_grad():
                backbone = {"img_batch_all_stages": batch.img_batch}
                backbone.update(model.backbone.forward_image(batch.img_batch))
                backbone.update(model.backbone.forward_text(batch.find_text_batch, device=torch.device("cuda")))
            prediction = model.forward_grounding(
                backbone_out=backbone, find_input=stage,
                find_target=batch.find_targets[0], geometric_prompt=prompt,
            )
            indices = model.matcher(prediction, targets)
            num_boxes = targets["num_boxes"].sum().float().clamp(min=1)
            loss_kwargs = {"outputs": prediction, "targets": targets, "indices": indices, "num_boxes": num_boxes}
            mask_losses = mask_loss_fn(**loss_kwargs)
            box_losses = box_loss_fn(**loss_kwargs)
            class_losses = class_loss_fn(**loss_kwargs)
            loss = mask_losses["core_loss"] + box_losses["core_loss"] + class_losses["core_loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss in {case}")
        if loss.requires_grad:
            loss.backward()
        grad_norm = adapter_gradient_norm(parameters)
        valid = row["hands"][0]["valid"]
        if valid and grad_norm == 0:
            raise RuntimeError(f"Valid MANO case {case} has no adapter gradient")
        frozen_no_grad = all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        if not frozen_no_grad:
            raise RuntimeError("A frozen model parameter received a gradient")
        optimizer_step = grad_norm > 0
        if optimizer_step:
            optimizer.step()
        change = parameter_max_change(adapter, before)
        if valid and not change > 0:
            raise RuntimeError(f"Valid MANO case {case} did not update the adapter")
        tokens_unchanged = torch.equal(initial_tokens, token_encoder.class_tokens.detach().cpu())
        if not tokens_unchanged:
            raise RuntimeError("Frozen class tokens changed")
        step = {
            "case": case, "image_id": row["image_id"], "dataset_index": dataset_index,
            "file_name": row["file_name"], "physical_side": row["hands"][0]["side"],
            "segmentation_target_present": row["segmentation_target_present"], "mano_valid": valid,
            "query_count": 2, "identical_mano_for_both_queries": True,
            "loss": float(loss.detach().cpu()), "loss_requires_grad": loss.requires_grad,
            "adapter_gradient_norm": grad_norm, "optimizer_step": optimizer_step,
            "adapter_parameter_max_change": change, "frozen_no_grad": frozen_no_grad,
            "class_tokens_unchanged": tokens_unchanged,
            "loss_components": {key: float(value.detach().cpu())
                                for group in (mask_losses, box_losses, class_losses)
                                for key, value in group.items() if key != "core_loss" and isinstance(value, torch.Tensor) and value.numel() == 1},
        }
        steps.append(step)
        write_json(args.output_dir / "progress.json", {"status": "in_progress", "steps": steps})
        print(json.dumps(step), flush=True)
        del batch, stage, prompt, targets, backbone, prediction, indices, loss_kwargs, mask_losses, box_losses, class_losses, loss, mano
        torch.cuda.empty_cache()
    summary = {
        "format": "sam3-mano-geometry-engineering-smoke-v1", "status": "passed",
        "purpose": "Three sample forward/backward engineering check; NOT a training/effectiveness result",
        "input_limitation": "Ground-truth MANO is oracle input; not an RGB-only inference pipeline",
        "seed": args.seed, "split": args.split, "batch_size": 1,
        "learning_rate": args.learning_rate, "amp": args.amp,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameter_names": sorted(trainable_names),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        "adapter_total_max_change": parameter_max_change(adapter, initial),
        "class_tokens_unchanged": torch.equal(initial_tokens, token_encoder.class_tokens.detach().cpu()),
        "steps": steps, "provenance": provenance,
    }
    artifact = args.output_dir / "adapter_smoke.pt"
    temporary = artifact.with_suffix(".pt.tmp")
    torch.save({"format": summary["format"], "smoke_only": True,
                "adapter_state_dict": {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()},
                "initial_adapter_parameters": initial, "optimizer_state_dict": optimizer.state_dict(),
                "summary": summary}, temporary)
    temporary.replace(artifact)
    summary["adapter_artifact"] = {"path": str(artifact.resolve()), "sha256": file_sha256(artifact)}
    write_json(args.output_dir / "smoke_summary.json", summary)
    write_json(args.output_dir / "progress.json", {"status": "passed", "steps": steps})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

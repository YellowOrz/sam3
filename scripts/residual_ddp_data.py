"""CPU data contract and identity-preserving loaders for bilateral residual DDP.

This module never scans image files or touches CUDA on import. Annotation loading
is explicit; callers must first approve the manifest and its exhaustive labels.
Pixels, resizing, normalization and target collation use the existing evaluator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("left_hand", "right_hand")
SIDE_BY_CATEGORY = {1: "left_hand", 2: "right_hand"}


def _check_completeness(value: Any, location: str, *, label_context=False) -> None:
    """Explicit incomplete side labels override a global approval declaration.

    COCO has no standard completeness field. Recognize common label/annotation
    status fields and nested per-side declarations while permitting unrelated
    metadata. Absent declarations require the caller's exhaustive approval.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            path = f"{location}.{key}"
            if name in {"is_exhaustive", "is_instance_exhaustive", "is_pixel_exhaustive",
                        "exhaustive", "labels_exhaustive", "annotations_exhaustive"}:
                if child is not True and not (type(child) is int and child == 1):
                    raise ValueError(f"Non-exhaustive labels at {path}")
            context = label_context or name in {"side_status", "hand_status"} or any(
                token in name for token in (*CLASS_NAMES, "label", "annotation", "completeness"))
            if context and any(token in name for token in ("missing", "unknown", "unannotated", "incomplete")):
                if child is not False and child is not None and child != [] and child != {} and child != 0:
                    raise ValueError(f"Unknown/missing or incomplete labels at {path}")
            if context and ("status" in name or "completeness" in name):
                if not isinstance(child, (Mapping, list, tuple)):
                    allowed = {"complete", "exhaustive", "approved", "verified", "annotated",
                               "present", "absent", "positive", "negative", "empty",
                               "visible", "occluded"}
                    if not isinstance(child, str) or child.lower() not in allowed:
                        raise ValueError(f"Unknown/missing or incomplete label status at {path}: {child!r}")
                context = True
            if context and isinstance(child, str) and child.lower() in {
                "unknown", "missing", "unannotated", "incomplete", "partial", "unlabeled"}:
                raise ValueError(f"Unknown/missing or incomplete labels at {path}")
            _check_completeness(child, path, label_context=context)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_completeness(child, f"{location}[{index}]", label_context=label_context)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class CocoContract:
    root: Path
    images: tuple[dict, ...]
    annotations_by_image: dict[int, tuple[dict, ...]]
    annotations_sha256: str
    side_counts: dict[int, tuple[int, int]]
    summary: dict
    allowed_image_roots: tuple[Path, ...] = ()


def _approved_image_path(root: Path, image: dict, allowed_image_roots: Sequence[Path]) -> Path:
    path = (root / image["file_name"]).resolve()
    if not any(path.is_relative_to(allowed) for allowed in (root, *allowed_image_roots)):
        raise ValueError(f"Image path escapes approved image roots: {image['file_name']}")
    return path


def load_coco_contract(root: Path, *, approved_exhaustive: bool,
                       allowed_image_roots: Sequence[Path] | None = None) -> CocoContract:
    """Read only annotations.json; validate generic 0/1/2-hand COCO metadata.

    ``approved_exhaustive`` is a required assertion from the calling CLI's
    hash-bound approval manifest, not an inference from absent annotations.
    Both sides of every image must have complete labels, including true negatives.
    """
    if approved_exhaustive is not True:
        raise ValueError("A manifest approving exhaustive labels for both sides is required")
    root = Path(root).resolve()
    allowed_image_roots = tuple(Path(path).resolve() for path in (allowed_image_roots or ()))
    raw = (root / "annotations.json").read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("COCO annotations must be an object")
    _check_completeness(data, "COCO")
    categories = data.get("categories")
    if (not isinstance(categories, list) or len(categories) != 2
            or any(not isinstance(row, dict) or type(row.get("id")) is not int for row in categories)
            or {row["id"]: row.get("name") for row in categories} != SIDE_BY_CATEGORY):
        raise ValueError("Expected categories 1=left_hand and 2=right_hand")
    images = data.get("images")
    annotations = data.get("annotations")
    if not isinstance(images, list) or not images or not isinstance(annotations, list):
        raise ValueError("COCO requires a nonempty images list and annotations list")
    seen = set()
    for row in images:
        if not isinstance(row, dict):
            raise ValueError("Each COCO image must be an object")
        image_id = _integer(row.get("id"), "image id")
        if image_id in seen:
            raise ValueError(f"Duplicate image id {image_id}")
        seen.add(image_id)
        _integer(row.get("height"), "image height", minimum=1)
        _integer(row.get("width"), "image width", minimum=1)
        filename = row.get("file_name")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Missing image file_name for {image_id}")
        _approved_image_path(root, row, allowed_image_roots)
        source_hash = row.get("source_rgb_sha256")
        if (source_hash is not None and (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash))):
            raise ValueError("source_rgb_sha256 must be a lowercase SHA256 digest")
    images = tuple(sorted(images, key=lambda row: row["id"]))
    grouped = {row["id"]: [] for row in images}
    counts = {row["id"]: [0, 0] for row in images}
    annotation_ids = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("Each COCO annotation must be an object")
        annotation_id = _integer(annotation.get("id"), "annotation id")
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation id {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _integer(annotation.get("image_id"), "annotation image_id")
        category_id = _integer(annotation.get("category_id"), "annotation category_id", minimum=1)
        if image_id not in grouped or category_id not in SIDE_BY_CATEGORY:
            raise ValueError("Annotation references an unknown image or category")
        if annotation.get("iscrowd") != 0:
            raise ValueError("Complete separate hand masks require iscrowd=0")
        bbox = annotation.get("bbox")
        if (not isinstance(bbox, (tuple, list)) or len(bbox) != 4
                or any(type(number) not in (int, float) or not math.isfinite(number) for number in bbox)
                or bbox[2] <= 0 or bbox[3] <= 0):
            raise ValueError("A finite positive-area COCO bbox is required")
        if not annotation.get("segmentation"):
            raise ValueError("Every annotated hand requires a segmentation mask")
        counts[image_id][category_id - 1] += 1
        if counts[image_id][category_id - 1] > 1:
            raise ValueError(f"Expected at most one physical hand per side in image {image_id}")
        grouped[image_id].append(annotation)
    digest = hashlib.sha256(raw).hexdigest()
    side_counts = {image_id: tuple(value) for image_id, value in counts.items()}
    summary = {"images": len(images), "annotations": len(annotations), "sha256": digest,
               "categories": dict(SIDE_BY_CATEGORY), "exhaustive_labels_approved": True,
               "empty_images": sum(sum(value) == 0 for value in counts.values()),
               "single_hand_images": sum(sum(value) == 1 for value in counts.values()),
               "both_hand_images": sum(sum(value) == 2 for value in counts.values())}
    return CocoContract(root, images, {key: tuple(rows) for key, rows in grouped.items()},
                        digest, side_counts, summary, allowed_image_roots)


@dataclass(frozen=True)
class IndexedSample:
    sample: Any
    dataset_index: int
    image_id: int
    provenance: dict
    side_counts: tuple[int, int]


def _validate_sample(item: IndexedSample) -> None:
    sample = item.sample
    if len(sample.images) != 1:
        raise RuntimeError("Expected one image per dataset index")
    queries = sample.find_queries
    if len(queries) != 2 or tuple(query.query_text for query in queries) != CLASS_NAMES:
        raise RuntimeError("Expected exactly left_hand then right_hand queries per image")
    image = sample.images[0]
    if not isinstance(image.data, torch.Tensor) or image.data.device.type != "cpu":
        raise RuntimeError("Dataset workers must return CPU image tensors")
    used_objects = []
    for side, query in enumerate(queries):
        metadata = query.inference_metadata
        if (query.image_id != 0 or metadata is None
                or metadata.coco_image_id != item.image_id
                or metadata.original_category_id != side + 1):
            raise RuntimeError("Loader substituted an image or prompt category")
        if query.query_processing_order != 0 or query.is_exhaustive is not True:
            raise RuntimeError("Only a single stage with exhaustive bilateral queries is supported")
        if query.is_pixel_exhaustive is False:
            raise RuntimeError("Non-exhaustive pixel labels are unsupported")
        for name in ("input_bbox", "input_points"):
            geometry = getattr(query, name, None)
            if geometry is not None and geometry.numel():
                raise RuntimeError("Reference-derived geometry prompts are forbidden")
        indices = query.object_ids_output
        if len(indices) != item.side_counts[side]:
            raise RuntimeError("Sample target count differs from approved annotation side count")
        for object_index in indices:
            if type(object_index) is not int or not 0 <= object_index < len(image.objects):
                raise RuntimeError("Query references an invalid target object")
            obj = image.objects[object_index]
            if (not isinstance(obj.bbox, torch.Tensor) or obj.bbox.device.type != "cpu"
                    or obj.bbox.numel() != 4 or not bool(torch.isfinite(obj.bbox).all())):
                raise RuntimeError("Invalid CPU target box")
            mask = obj.segment
            if (not isinstance(mask, torch.Tensor) or mask.device.type != "cpu"
                    or mask.dtype not in (torch.bool, torch.uint8)
                    or tuple(mask.shape) != tuple(image.data.shape[-2:])
                    or (mask.dtype == torch.uint8 and bool((mask > 1).any()))
                    or not bool(mask.any())):
                raise RuntimeError("Positive hand target needs a valid, nonempty CPU mask")
            used_objects.append(object_index)
    if len(used_objects) != len(set(used_objects)) or sorted(used_objects) != list(range(len(image.objects))):
        raise RuntimeError("Hand targets must belong to exactly one side query")


class IdentityCheckedDataset(Dataset):
    """Return requested identity alongside the sample, including across workers."""

    def __init__(self, dataset, contract: CocoContract):
        if len(dataset) != len(contract.images):
            raise ValueError("Dataset size differs from approved COCO image index")
        self.dataset = dataset
        self.contract = contract
        # The existing loader retries by advancing idx. One attempt stops this
        # before a substituted image can even be read; identity checks remain.
        if hasattr(dataset, "_MAX_RETRIES"):
            dataset._MAX_RETRIES = 1

    def __len__(self):
        return len(self.contract.images)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(f"Invalid dataset index {index}")
        image = self.contract.images[index]
        image_id = image["id"]
        path = _approved_image_path(self.contract.root, image, self.contract.allowed_image_roots)
        source_hash = image.get("source_rgb_sha256")
        if source_hash is not None:
            with path.open("rb") as handle:
                observed = hashlib.file_digest(handle, "sha256").hexdigest()
            if observed != source_hash:
                raise RuntimeError(f"Source RGB hash differs for requested image {image_id}")
        item = IndexedSample(
            self.dataset[index], index, image_id,
            {"dataset_root": str(self.contract.root), "annotations_sha256": self.contract.annotations_sha256,
             "file_name": image["file_name"], "image_id": image_id, "dataset_index": index,
             "source_rgb_sha256": source_hash,
             "source": {key: value for key, value in image.items() if key.startswith("source_") or key == "provenance"}},
            self.contract.side_counts[image_id])
        _validate_sample(item)
        return item


def make_identity_dataset(root: Path, contract: CocoContract) -> IdentityCheckedDataset:
    """Reuse the established 1008-square evaluator pixel and label transforms."""
    try:
        from scripts.evaluate_bilateral_tokens import make_dataset
    except ModuleNotFoundError:
        from evaluate_bilateral_tokens import make_dataset
    root = Path(root).resolve()
    if root != contract.root:
        raise ValueError("Dataset root differs from approved annotation provenance")
    annotation_path = root / "annotations.json"
    if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != contract.annotations_sha256:
        raise RuntimeError("Annotations changed before dataset creation")
    dataset = make_dataset(root)
    if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != contract.annotations_sha256:
        raise RuntimeError("Annotations changed during dataset creation")
    return IdentityCheckedDataset(dataset, contract)


def map_tensors(value: Any, function: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Preserve nested SAM3 dataclasses, including frozen/init=False fields."""
    if isinstance(value, torch.Tensor):
        return function(value)
    if is_dataclass(value) and not isinstance(value, type):
        result = copy.copy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, map_tensors(getattr(value, field.name), function))
        return result
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, child in value.items():
            result[key] = map_tensors(child, function)
        return result
    if isinstance(value, list):
        return [map_tensors(child, function) for child in value]
    if isinstance(value, tuple):
        children = [map_tensors(child, function) for child in value]
        return type(value)(*children) if hasattr(value, "_fields") else tuple(children)
    return value


def pin_memory_recursive(value: Any) -> Any:
    return map_tensors(value, lambda tensor: tensor.pin_memory())


@dataclass(frozen=True)
class ResidualBatch:
    datapoint: Any
    dataset_indices: tuple[int, ...]
    image_ids: tuple[int, ...]
    provenance: tuple[dict, ...]

    def pin_memory(self):
        # DataLoader does not descend into arbitrary dataclasses by itself.
        return pin_memory_recursive(self)

    def to(self, device, *, non_blocking=True):
        return map_tensors(self, lambda tensor: tensor.to(device, non_blocking=non_blocking))


def validate_collated_batch(batch, items: Sequence[IndexedSample]) -> None:
    if (batch.img_batch.shape[0] != len(items) or tuple(batch.find_text_batch) != CLASS_NAMES
            or len(batch.find_inputs) != 1 or len(batch.find_targets) != 1
            or len(batch.find_metadatas) != 1):
        raise RuntimeError("Collation changed image count or bilateral query stages")
    stage, target, metadata = batch.find_inputs[0], batch.find_targets[0], batch.find_metadatas[0]
    expected_ids = [item.image_id for item in items for _ in CLASS_NAMES]
    expected_counts = [count for item in items for count in item.side_counts]
    if (metadata.coco_image_id.tolist() != expected_ids
            or metadata.original_category_id.tolist() != [1, 2] * len(items)
            or stage.img_ids.tolist() != [index for index in range(len(items)) for _ in CLASS_NAMES]
            or stage.text_ids.tolist() != [0, 1] * len(items)):
        raise RuntimeError("Collation changed requested image identity or query category order")
    count = sum(expected_counts)
    if (target.num_boxes.tolist() != expected_counts or target.boxes.numel() != count * 4
            or target.is_valid_segment is None or target.segments is None
            or target.is_valid_segment.numel() != count or len(target.segments) != count
            or not bool(target.is_valid_segment.all())
            or target.is_exhaustive.tolist() != [True] * len(expected_counts)):
        raise RuntimeError("Collated target count, exhaustive labels or valid masks differ")
    if count:
        expected_masks = torch.stack([item.sample.images[0].objects[index].segment
                                      for item in items for query in item.sample.find_queries
                                      for index in query.object_ids_output]).bool()
        if not torch.equal(target.segments, expected_masks):
            raise RuntimeError("Collation reordered hand masks relative to image and side queries")


def collate_indexed_samples(items: Sequence[IndexedSample]) -> ResidualBatch:
    from sam3.train.data.collator import collate_fn_api

    if not items:
        raise ValueError("Cannot collate an empty image batch")
    for item in items:
        _validate_sample(item)
    batch = collate_fn_api([item.sample for item in items], dict_key="train", with_seg_masks=True)["train"]
    validate_collated_batch(batch, items)
    return ResidualBatch(batch, tuple(item.dataset_index for item in items),
                         tuple(item.image_id for item in items), tuple(item.provenance for item in items))


def seed_worker(worker_id: int) -> None:
    """Top-level spawn-picklable initializer using DataLoader's seeded generator."""
    del worker_id
    seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.set_num_threads(1)


def make_loader(dataset, *, batch_sampler, num_workers=0, pin_memory=True,
                persistent_workers=False, prefetch_factor=2, seed=0) -> DataLoader:
    """Build one rank's loader; the supplied DDP sampler defines equal batches.

    Restarting workers is the default for reproducible epoch-boundary resumes.
    The underlying evaluator transforms are deterministic even with persistence.
    """
    _integer(num_workers, "num_workers")
    _integer(seed, "seed")
    _integer(prefetch_factor, "prefetch_factor", minimum=1)
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    kwargs = {"dataset": dataset, "batch_sampler": batch_sampler,
              "collate_fn": collate_indexed_samples, "num_workers": num_workers,
              "pin_memory": pin_memory, "worker_init_fn": seed_worker,
              "generator": torch.Generator().manual_seed(seed)}
    if num_workers:
        kwargs.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
                      multiprocessing_context="spawn")
    return DataLoader(**kwargs)

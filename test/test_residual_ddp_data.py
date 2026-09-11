"""CPU-only coverage of DDP image identity, labels and worker transport."""

from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch.utils.data import Dataset

from sam3.train.data.sam3_image_dataset import (
    Datapoint, FindQueryLoaded, Image as DataImage, InferenceMetadata, Object,
)
from scripts import residual_ddp_data as data


def write_fixture(root):
    images, annotations = [], []
    for index, sides in enumerate(((1, 2), (1,), (2,), ())):
        image_id = 20 + index * 3
        name = f"image-{image_id}.png"
        Image.new("RGB", (8, 8), (30 + index * 10, 60, 90)).save(root / name)
        images.append({"id": image_id, "file_name": name, "height": 8, "width": 8,
                       "source_dataset": "generic", "extra": {"arbitrary": "metadata"}})
        for side in sides:
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[1:5, 1:3 if side == 1 else 7] = 1
            if side == 2:
                mask[:, :5] = 0
            rle = mask_utils.encode(np.asfortranarray(mask))
            bbox = mask_utils.toBbox(rle).tolist()
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({"id": len(annotations), "image_id": image_id, "category_id": side,
                                "segmentation": rle, "bbox": bbox, "area": int(mask.sum()), "iscrowd": 0})
    document = {"images": list(reversed(images)), "annotations": annotations,
                "categories": [{"id": 2, "name": "right_hand", "supercategory": "hand"},
                               {"id": 1, "name": "left_hand", "supercategory": "hand"}]}
    (root / "annotations.json").write_text(json.dumps(document))
    return document


def make_sample(image_id, counts=(0, 0)):
    objects, queries = [], []
    for category, (side, count) in enumerate(zip(data.CLASS_NAMES, counts), 1):
        indices = []
        if count:
            indices.append(len(objects))
            mask = torch.zeros(6, 8, dtype=torch.uint8)
            mask[1:4, category:category + 2] = 1
            objects.append(Object(bbox=torch.tensor([.5, .5, .25, .5]), area=6., segment=mask))
        metadata = InferenceMetadata(coco_image_id=image_id, original_image_id=image_id,
                                     original_category_id=category, original_size=(6, 8),
                                     object_id=0, frame_index=0)
        queries.append(FindQueryLoaded(query_text=side, image_id=0, object_ids_output=indices,
                                       is_exhaustive=True, inference_metadata=metadata))
    return Datapoint(queries, [DataImage(torch.zeros(3, 6, 8), objects, (6, 8))])


class RandomWorkerDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        sample = make_sample(100 + index)
        values = (random.random(), float(np.random.random()), float(torch.rand(())))
        sample.images[0].data[:, 0, 0] = torch.tensor(values)
        return data.IndexedSample(sample, index, 100 + index,
                                  {"rng": values, "threads": torch.get_num_threads()}, (0, 0))


class CocoContractTest(unittest.TestCase):
    def test_generic_categories_metadata_and_zero_one_two_hands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            contract = data.load_coco_contract(root, approved_exhaustive=True)
            self.assertEqual([row["id"] for row in contract.images], [20, 23, 26, 29])
            self.assertEqual(contract.side_counts, {20: (1, 1), 23: (1, 0), 26: (0, 1), 29: (0, 0)})
            self.assertEqual(contract.summary["both_hand_images"], 1)
            self.assertEqual(contract.summary["single_hand_images"], 2)
            self.assertEqual(contract.summary["empty_images"], 1)
            with self.assertRaisesRegex(ValueError, "approving exhaustive"):
                data.load_coco_contract(root, approved_exhaustive=False)

    def test_explicit_unknown_or_missing_labels_override_global_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = write_fixture(root)
            for status in ({"label_status": {"left_hand": "unknown", "right_hand": "complete"}},
                           {"annotations_exhaustive": False},
                           {"left_hand": {"status": "missing"}},
                           {"left_hand_status": "unknown"},
                           {"side_status": {"right_hand": "missing"}},
                           {"missing_labels": True},
                           {"annotation_status": "partial"}):
                document = deepcopy(original)
                document["images"][0].update(status)
                (root / "annotations.json").write_text(json.dumps(document))
                with self.subTest(status=status), self.assertRaisesRegex(ValueError, "exhaustive|label"):
                    data.load_coco_contract(root, approved_exhaustive=True)

    def test_invalid_category_duplicate_side_and_missing_segmentation_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = write_fixture(root)
            for mutation in ("category", "duplicate", "segmentation"):
                document = deepcopy(original)
                if mutation == "category":
                    document["categories"][0]["name"] = "left_hand"
                elif mutation == "duplicate":
                    extra = {**document["annotations"][0], "id": 400}
                    document["annotations"].append(extra)
                else:
                    document["annotations"][0]["segmentation"] = None
                (root / "annotations.json").write_text(json.dumps(document))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    data.load_coco_contract(root, approved_exhaustive=True)

    def test_external_symlinks_require_approved_root_and_rgb_hash_is_checked_on_access(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root, source = parent / "combined", parent / "source"
            root.mkdir()
            source.mkdir()
            document = write_fixture(root)
            image = document["images"][0]
            original = root / image["file_name"]
            target = source / "source.png"
            original.replace(target)
            original.symlink_to(target)
            image["source_rgb_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            (root / "annotations.json").write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "approved image roots"):
                data.load_coco_contract(root, approved_exhaustive=True)
            contract = data.load_coco_contract(root, approved_exhaustive=True, allowed_image_roots=[source])
            samples = [make_sample(row["id"], contract.side_counts[row["id"]]) for row in contract.images]
            checked = data.IdentityCheckedDataset(samples, contract)
            item = checked[3]
            self.assertEqual(item.provenance["source"]["source_dataset"], "generic")
            target.write_bytes(b"changed approved source")
            with self.assertRaisesRegex(RuntimeError, "Source RGB hash"):
                checked[3]


class IdentityAndCollationTest(unittest.TestCase):
    def test_real_pixel_pipeline_multiple_images_and_query_masks_keep_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            contract = data.load_coco_contract(root, approved_exhaustive=True)
            dataset = data.make_identity_dataset(root, contract)
            items = [dataset[index] for index in (2, 0, 3, 1)]
            batch = data.collate_indexed_samples(items)
            self.assertEqual(batch.dataset_indices, (2, 0, 3, 1))
            self.assertEqual(batch.image_ids, (26, 20, 29, 23))
            self.assertEqual(tuple(batch.datapoint.img_batch.shape), (4, 3, 1008, 1008))
            self.assertEqual(batch.datapoint.find_metadatas[0].coco_image_id.tolist(),
                             [26, 26, 20, 20, 29, 29, 23, 23])
            self.assertEqual(batch.datapoint.find_targets[0].num_boxes.tolist(), [0, 1, 1, 1, 0, 0, 1, 0])
            repeat = data.collate_indexed_samples([dataset[index] for index in (2, 0, 3, 1)])
            self.assertTrue(torch.equal(batch.datapoint.img_batch, repeat.datapoint.img_batch))
            self.assertTrue(torch.equal(batch.datapoint.find_targets[0].segments,
                                        repeat.datapoint.find_targets[0].segments))
            moved = batch.to("cpu")
            self.assertEqual(moved.image_ids, batch.image_ids)
            self.assertIsInstance(moved.datapoint, type(batch.datapoint))

    def test_loader_error_never_advances_to_another_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            contract = data.load_coco_contract(root, approved_exhaustive=True)
            dataset = data.make_identity_dataset(root, contract)
            requested = []
            original = dataset.dataset._load_datapoint

            def broken(index):
                requested.append(index)
                if index == 0:
                    raise OSError("deliberate read error")
                return original(index)

            with patch.object(dataset.dataset, "_load_datapoint", side_effect=broken):
                with self.assertRaisesRegex(RuntimeError, "Failed 1 times"):
                    dataset[0]
            self.assertEqual(requested, [0])

    def test_substitution_geometry_invalid_masks_and_unknown_query_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            contract = data.load_coco_contract(root, approved_exhaustive=True)
            original = [make_sample(row["id"], contract.side_counts[row["id"]]) for row in contract.images]
            for mutation in ("identity", "category", "geometry", "mask", "count", "exhaustive", "order"):
                samples = deepcopy(original)
                query = samples[0].find_queries[0]
                if mutation == "identity":
                    query.inference_metadata.coco_image_id = 999
                elif mutation == "category":
                    query.inference_metadata.original_category_id = 2
                elif mutation == "geometry":
                    query.input_bbox = torch.ones(4)
                elif mutation == "mask":
                    samples[0].images[0].objects[0].segment = None
                elif mutation == "count":
                    query.object_ids_output = []
                elif mutation == "exhaustive":
                    query.is_exhaustive = False
                else:
                    samples[0].find_queries.reverse()
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                    data.IdentityCheckedDataset(samples, contract)[0]

    def test_collated_metadata_counts_validity_and_mask_order_tampering_rejected(self):
        items = [data.IndexedSample(make_sample(4, (1, 1)), 0, 4, {}, (1, 1)),
                 data.IndexedSample(make_sample(9), 1, 9, {}, (0, 0))]
        original = data.collate_indexed_samples(items).datapoint
        for mutation in ("identity", "count", "valid", "mask_order"):
            batch = deepcopy(original)
            if mutation == "identity":
                batch.find_metadatas[0].coco_image_id[0] = 9
            elif mutation == "count":
                batch.find_targets[0].num_boxes[0] = 0
            elif mutation == "valid":
                batch.find_targets[0].is_valid_segment[0] = False
            else:
                batch.find_targets[0].segments = batch.find_targets[0].segments.flip(0)
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                data.validate_collated_batch(batch, items)


@dataclass(frozen=True)
class Nested:
    tensor: torch.Tensor
    children: object
    metadata: str = "unchanged"
    hidden: torch.Tensor = field(init=False, default_factory=lambda: torch.tensor(8))


class WorkerTransportTest(unittest.TestCase):
    def test_recursive_pinning_preserves_dataclasses_containers_and_metadata(self):
        Pair = namedtuple("Pair", "tensor label")
        nested = Nested(torch.tensor(1), {"list": [torch.tensor(2), Pair(torch.tensor(3), "pair")],
                                          "tuple": (torch.tensor(4), None)})
        batch = data.ResidualBatch(nested, (7,), (77,), ({"source": "fixture"},))
        calls = []

        def fake_pin(tensor):
            calls.append(int(tensor))
            return tensor + 100

        with patch.object(torch.Tensor, "pin_memory", fake_pin):
            pinned = batch.pin_memory()
        self.assertEqual(sorted(calls), [1, 2, 3, 4, 8])
        self.assertIsInstance(pinned.datapoint, Nested)
        self.assertIsInstance(pinned.datapoint.children["list"][1], Pair)
        self.assertEqual(int(pinned.datapoint.hidden), 108)
        self.assertEqual(pinned.provenance, batch.provenance)
        self.assertEqual(int(batch.datapoint.tensor), 1)

    def test_worker_seed_is_picklable_deterministic_and_limits_torch_threads(self):
        self.assertIs(pickle.loads(pickle.dumps(data.seed_worker)), data.seed_worker)
        with patch("torch.initial_seed", return_value=1234), patch("torch.set_num_threads") as set_threads:
            data.seed_worker(0)
            first = (random.random(), np.random.random())
            data.seed_worker(2)
            self.assertEqual(first, (random.random(), np.random.random()))
            set_threads.assert_called_with(1)

    def test_spawn_worker_loader_is_deterministic_and_reports_real_identities(self):
        def consume():
            loader = data.make_loader(RandomWorkerDataset(), batch_sampler=[[2, 0], [3, 1]],
                                      num_workers=1, pin_memory=False, seed=572)
            return [(batch.image_ids, batch.datapoint.img_batch.clone(), batch.provenance) for batch in loader]

        first, second = consume(), consume()
        self.assertEqual([row[0] for row in first], [(102, 100), (103, 101)])
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left[1], right[1]))
            self.assertEqual(left[2], right[2])
            self.assertTrue(all(item["threads"] == 1 for item in left[2]))

    def test_zero_worker_loader_omits_worker_only_settings(self):
        loader = data.make_loader(RandomWorkerDataset(), batch_sampler=[[0, 1]],
                                  pin_memory=False, seed=8)
        self.assertIsNone(loader.multiprocessing_context)
        self.assertIsNone(loader.prefetch_factor)
        self.assertEqual(next(iter(loader)).dataset_indices, (0, 1))
        with self.assertRaisesRegex(ValueError, "persistent_workers"):
            data.make_loader(RandomWorkerDataset(), batch_sampler=[[0]], persistent_workers=True)


if __name__ == "__main__":
    unittest.main()

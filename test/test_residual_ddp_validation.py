"""Synthetic CPU/Gloo verification of exhaustive validation and loss weighting."""

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from sam3.train.data.sam3_image_dataset import (
    Datapoint, FindQueryLoaded, Image as DataImage, InferenceMetadata, Object,
)
from scripts import residual_ddp_data as data
from scripts import residual_ddp_validation as validation
from scripts.residual_ddp_runtime import DistributedContext


class SyntheticDataset:
    def __init__(self, contract):
        self.contract = contract

    def __len__(self):
        return len(self.contract.images)

    def __getitem__(self, index):
        image = self.contract.images[index]
        counts = self.contract.side_counts[image["id"]]
        objects, queries = [], []
        for category, (side, count) in enumerate(zip(data.CLASS_NAMES, counts), 1):
            outputs = []
            if count:
                outputs = [len(objects)]
                objects.append(Object(bbox=torch.tensor([.5, .5, .5, .5]), area=16.,
                                      segment=torch.ones(4, 4, dtype=torch.bool)))
            metadata = InferenceMetadata(coco_image_id=image["id"], original_image_id=image["id"],
                                         original_category_id=category,
                                         original_size=(image["height"], image["width"]),
                                         object_id=0, frame_index=index)
            queries.append(FindQueryLoaded(query_text=side, image_id=0, object_ids_output=outputs,
                                           is_exhaustive=True, inference_metadata=metadata))
        sample = Datapoint(queries, [DataImage(torch.full((3, 4, 4), float(index + 1)), objects,
                                              (image["height"], image["width"]))])
        return data.IndexedSample(sample, index, image["id"], {"fixture": True}, counts)


class FakeModel(nn.Module):
    def back_convert(self, targets):
        return {"num_boxes": targets.num_boxes}

    def matcher(self, outputs, targets):
        # A matcher would choose the lower-scoring perfect mask for positives.
        # Prediction selection must remain independent of these loss indices.
        return "prefer_decoder_1_for_loss_only"

    def forward(self, batch):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Validation must run eval with gradients disabled")
        values = batch.img_batch[:, 0, 0, 0].repeat_interleave(2)
        query_count = len(values)
        logits = torch.tensor([8., -8.]).repeat(query_count, 1).unsqueeze(-1)
        # One positive (image index 1, left) is a low-confidence false negative.
        logits[values == 2] -= 12.
        masks = torch.full((query_count, 2, 2, 2), -10.)
        masks[:, 1] = 10.
        # Highest-score mask is perfect except index 4, whose lower-score
        # candidate would be perfect: useful to reject reference-based choice.
        masks[values != 5, 0] = 10.
        return [{"pred_logits": logits, "presence_logit_dec": torch.full((query_count, 1), 8.),
                 "pred_masks": masks, "fixture_values": values}]


class FakeLoss(nn.Module):
    def forward(self, *, outputs, targets, indices, num_boxes):
        if num_boxes != 1. or indices != "prefer_decoder_1_for_loss_only":
            raise AssertionError("Loss must use unnormalized target numerators and the matcher")
        values = outputs["fixture_values"]
        target_sum = (values * targets["num_boxes"]).sum()
        return {"loss_mask": target_sum, "loss_dice": target_sum * 2,
                "loss_bbox": target_sum * 3, "loss_giou": target_sum * 4,
                "loss_ce": values.mean(), "presence_loss": values.mean() * .5}


class RecordingMonitor:
    def __init__(self):
        self.images, self.validations = [], []

    def log_images(self, step, images):
        self.images.append((step, images))

    def log_validation(self, step, metrics, scope):
        self.validations.append((step, metrics, scope))


def fixture(root, size=5):
    root.mkdir()
    images, annotations = [], []
    for index, counts in enumerate(((1, 1), (1, 0), (0, 1), (0, 0), (1, 0))[:size]):
        height, width = 6 + index, 8 + index
        image = {"id": 20 + 3 * index, "height": height, "width": width,
                 "file_name": f"image-{index}.png", "source": "dexycb"}
        images.append(image)
        Image.new("RGB", (width, height), (30 + index, 60, 90)).save(root / image["file_name"])
        for category, count in enumerate(counts, 1):
            if count:
                rle = mask_utils.encode(np.asfortranarray(np.ones((height, width), dtype=np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                annotations.append({"id": len(annotations), "image_id": image["id"],
                                    "category_id": category, "segmentation": rle,
                                    "bbox": [0, 0, width, height], "area": height * width, "iscrowd": 0})
    (root / "annotations.json").write_text(json.dumps({
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}],
    }))
    return data.load_coco_contract(root, approved_exhaustive=True)


def run_single(contract, output, batch_size=2, monitor=None):
    context = DistributedContext(0, 0, 1, torch.device("cpu"), None)
    with patch.object(validation.objective.shared, "build_loss_functions", return_value=(FakeLoss(),)):
        return validation.evaluate_validation(FakeModel(), contract, SyntheticDataset(contract),
            context=context, batch_size=batch_size, num_workers=0, output_dir=output,
            monitor=monitor, step=90, amp=False)


def distributed_worker(rank, world_size, rendezvous, dataset_root, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=45))
    try:
        context = DistributedContext(rank, rank, world_size, torch.device("cpu"), "gloo")
        contract = data.load_coco_contract(Path(dataset_root), approved_exhaustive=True)
        with patch.object(validation.objective.shared, "build_loss_functions", return_value=(FakeLoss(),)):
            metrics = validation.evaluate_validation(FakeModel(), contract, SyntheticDataset(contract),
                context=context, batch_size=2, num_workers=0, output_dir=output, step=90, amp=False)
        (Path(output) / f"returned-{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


class ValidationTests(unittest.TestCase):
    def test_tail_batches_and_empty_rank_preserve_exact_partition(self):
        self.assertEqual(validation.validation_batches(5, 2, 0, 2), [[0, 2], [4]])
        self.assertEqual(validation.validation_batches(5, 2, 1, 2), [[1, 3]])
        self.assertEqual(validation.validation_batches(2, 2, 2, 3), [])
        for world in (1, 2, 3, 8):
            all_indices = [index for rank in range(world)
                           for batch in validation.validation_batches(5, 2, rank, world) for index in batch]
            self.assertEqual(sorted(all_indices), list(range(5)))

    def test_original_dimensions_score_selection_and_global_loss_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = fixture(root / "data")
            monitor = RecordingMonitor()
            metrics = run_single(contract, root / "validation", monitor=monitor)
            self.assertEqual((metrics["images"], metrics["queries"], metrics["targets"]), (5, 10, 5))
            for index, key in enumerate(validation.objective.TARGET_COMPONENTS, 1):
                self.assertAlmostEqual(metrics[key], 2.4 * index)
            self.assertAlmostEqual(metrics["loss_ce"], 3.)
            self.assertAlmostEqual(metrics["presence_loss"], 1.5)
            self.assertAlmostEqual(metrics["total_loss"], 28.5)
            self.assertEqual(metrics["left_hand/positive_count"], 3)
            self.assertAlmostEqual(metrics["left_hand/candidate_dice"], 2 / 3)
            self.assertAlmostEqual(metrics["left_hand/miss_zero_dice"], 1 / 3)
            self.assertEqual(metrics["left_hand/false_negative_count"], 1)
            self.assertEqual(metrics["left_hand/absent_count"], 2)
            self.assertEqual(metrics["left_hand/false_positive_count"], 2)
            self.assertEqual(metrics["right_hand/false_positive_count"], 2)
            self.assertEqual(metrics["right_hand/candidate_boundary_iou_4px"], 1.)
            rows = [json.loads(line) for line in (root / "validation/rank-0.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            for row in rows:
                expected = contract.images[row["dataset_index"]]
                self.assertEqual(row["prediction_rle"]["size"], [expected["height"], expected["width"]])
                self.assertEqual(row["selected_decoder_query"], 0)
                self.assertEqual(row["dataset_role"], "validation")
            self.assertEqual(len(monitor.images), 2)
            self.assertTrue(all(len(images) == 5 for _, images in monitor.images))
            self.assertEqual(monitor.validations[0][0::2], (90, "dexycb_val"))
            summary = json.loads((root / "validation/summary.json").read_text())
            self.assertEqual(summary["verified_query_records"], 10)
            self.assertEqual(summary["metrics"], metrics)
            # Changing batch boundaries must not change any global statistic.
            self.assertEqual(run_single(contract, root / "one-per-batch", batch_size=1), metrics)

    def test_real_gloo_unequal_batches_and_a_rank_without_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size, world in ((5, 2), (2, 3)):
                with self.subTest(size=size, world=world):
                    contract = fixture(root / f"data-{size}", size=size)
                    expected = run_single(contract, root / f"serial-{size}")
                    output = root / f"distributed-{size}"
                    mp.spawn(distributed_worker,
                             args=(world, str(root / f"rendezvous-{size}"), str(contract.root), str(output)),
                             nprocs=world, join=True)
                    for rank in range(world):
                        actual = json.loads((output / f"returned-{rank}.json").read_text())
                        self.assertEqual(actual, expected)
                    if size < world:
                        self.assertEqual((output / "rank-2.jsonl").read_text(), "")

    def test_existing_output_realsense_scope_and_nonfinite_predictions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = fixture(root / "data")
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(RuntimeError, "FileExistsError"):
                run_single(contract, output)
            contract.images[0]["source"] = "realsense"
            with self.assertRaisesRegex(RuntimeError, "RealSense"):
                run_single(contract, root / "forbidden")
            self.assertFalse((root / "forbidden").exists())
            contract.images[0]["source"] = "dexycb"
            original_forward = FakeModel.forward

            def nonfinite_forward(model, batch):
                outputs = original_forward(model, batch)
                outputs[0]["pred_logits"][0, 0, 0] = float("nan")
                return outputs

            with patch.object(FakeModel, "forward", nonfinite_forward):
                with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                    run_single(contract, root / "nonfinite")
            self.assertFalse((root / "nonfinite/summary.json").exists())

    def test_record_union_rejects_duplicate_or_missing_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = fixture(root / "data", size=2)
            output = root / "validation"
            run_single(contract, output)
            path = output / "rank-0.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "cover every"):
                validation._verify_records(output, contract, 1)
            path.write_text("\n".join(lines + lines[:1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validation._verify_records(output, contract, 1)


if __name__ == "__main__":
    unittest.main()

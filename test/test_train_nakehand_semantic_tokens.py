from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

from scripts import cached_ve_text_features as cached
from scripts import train_nakehand_semantic_tokens as training


def make_encoder():
    padding = torch.ones(2, 32, dtype=torch.bool)
    padding[:, [0, 2, 5, 31]] = False
    features = torch.full((32, 2, 256), 100., dtype=torch.bfloat16)
    features[[0, 2, 5, 31], 0] = 2.
    features[[0, 2, 5, 31], 1] = 4.
    return cached.CachedVETextEncoder(padding, features, torch.ones(32, 2, 1024), mode="zero_delta",
                                      metadata={"base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64})


def coco_fixture(root):
    images, annotations = [], []
    for frame, categories in enumerate(((1, 2), (1,), (2,), ())):
        image_id = 100 + frame
        name = f"rgb-{frame}.png"
        Image.new("RGB", (8, 8), (10 + frame, 20, 30)).save(root / name)
        images.append({"id": image_id, "file_name": name, "width": 8, "height": 8,
                       "recording_id": "fake/recording", "frame_index": frame, "source_frame_index": frame})
        for category in categories:
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[1:4, (1 if category == 1 else 5):(3 if category == 1 else 7)] = 1
            rle = mask_utils.encode(np.asfortranarray(mask))
            bbox = mask_utils.toBbox(rle).tolist()
            area = int(mask_utils.area(rle))
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({"id": image_id * 2 + category - 1, "image_id": image_id,
                                "category_id": category, "segmentation": rle, "bbox": bbox, "area": area, "iscrowd": 0})
    return {"info": {"split": "train", "dataset_role": "train"}, "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "left_hand"}, {"id": 2, "name": "right_hand"}]}


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return training.evaluation.sha256(path)


def ready_fixture(parent):
    root = parent / "train"
    root.mkdir()
    data = coco_fixture(root)
    sources = []
    for index in range(60):
        path = parent / f"source-{index}.txt"
        path.write_text(f"immutable-{index}")
        sources.append({"path": str(path), "bytes": path.stat().st_size, "sha256": training.evaluation.sha256(path)})
    plan = {"output": str(parent), "sources": sources,
            "splits": {"train": {"recordings": ["fake/recording"], "images": 4, "coco_split": "train"}},
            "recordings": {"fake/recording": {"frame_count": 4, "global_image_id_offset": 100}}}
    plan_hash = save_json(parent / "frozen-plan.json", plan)
    data["info"]["frozen_plan_sha256"] = plan_hash
    annotation_hash = save_json(root / "annotations.json", data)
    counts = {"images": 4, "annotations": 4, "left_annotations": 2, "right_annotations": 2,
              "empty_images": 1, "one_hand_images": 2, "two_hand_images": 1}
    manifest = {"status": "complete", "split": "train", "annotations_sha256": annotation_hash,
                "frozen_plan_sha256": plan_hash, "sources": sources, "sources_unchanged": True, "counts": counts,
                "validation": {"all_png_sha256_and_pixels_checked": True, "all_rle_area_bbox_checked": True},
                "image_outputs": [{"image_id": image["id"], "files": {"rgb": {"path": image["file_name"],
                    "sha256": training.evaluation.sha256(root / image["file_name"])}}} for image in data["images"]]}
    manifest_hash = save_json(root / "manifest.json", manifest)
    ready = {"status": "complete", "annotations_sha256": annotation_hash, "manifest_sha256": manifest_hash,
             "frozen_plan_sha256": plan_hash, "counts": counts}
    ready_hash = save_json(root / "READY.json", ready)
    bound = {**ready, "ready_sha256": ready_hash, "directory": "train"}
    root_manifest = {"status": "complete", "frozen_plan_sha256": plan_hash, "sources_unchanged": True,
                     "sources": sources, "splits": {"train": bound}}
    root_manifest_hash = save_json(parent / "manifest.json", root_manifest)
    save_json(parent / "READY.json", {"status": "complete", "frozen_plan_sha256": plan_hash,
                                      "manifest_sha256": root_manifest_hash, "splits": {"train": bound}})
    return root


def checkpoint_fixture(weight=1., task_side=None):
    encoder = make_encoder()
    initial = training.shared.cpu_state(encoder.state_dict())
    optimizer = torch.optim.AdamW([encoder.delta], lr=.001, weight_decay=0.)
    features = encoder(list(cached.CLASS_NAMES))[1].float()
    task = (features if task_side is None else features[:, task_side]).sum()
    anchor, _ = training.anchor_penalty(encoder)
    task_value, anchor_value = float(task.detach()), float(anchor.detach())
    norms = training.apply_loss_gradients(task, anchor, encoder.delta, weight)
    optimizer.step()
    _, ratio = training.anchor_penalty(encoder)
    drift = ratio.detach().sqrt().tolist()
    core = {"sam3/fake.py": "d" * 64}
    config = {"learning_rate": .001, "anchor_weight": weight, "annotations_sha256": "e" * 64,
              "base_checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
              "initial_cache_sha256": "c" * 64, "core_sources_sha256": training.shared.json_hash(core),
              "data_provenance": {"dataset_role": "train", "root_ready_sha256": "f" * 64}}
    ids = list(range(20000, 29092))
    order = training.legacy.build_epoch_order(len(ids), 123)[:2000]
    state = training.make_checkpoint(encoder=encoder, optimizer=optimizer, config=config, initial_state=initial,
                                      annotation_summary={"images": 9092, "sha256": "e" * 64}, order=order,
                                      observed_ids=[ids[order[0]]], loss_history=[task_value + weight * anchor_value],
                                      gradient_counts=[int(value > 0) for value in norms["task"]],
                                      core_hashes=core, task_history=[task_value],
                                      anchor_history=[anchor_value], drift_history=[drift], task_grad_history=[norms["task"]])
    return state, config, order, ids, initial


class NakehandSemanticTrainingTest(unittest.TestCase):
    def test_normalized_anchor_uses_only_valid_positions_with_equal_side_weight(self):
        encoder = make_encoder()
        anchor, ratios = training.anchor_penalty(encoder)
        self.assertEqual(float(anchor.detach()), 0)
        self.assertTrue(torch.equal(torch.autograd.grad(anchor, encoder.delta)[0], torch.zeros_like(encoder.delta)))
        with torch.no_grad():
            encoder.delta.fill_(2.)
        anchor, ratios = training.anchor_penalty(encoder)
        torch.testing.assert_close(ratios, torch.tensor([1., .25]))
        self.assertAlmostEqual(float(anchor.detach()), .625)
        torch.testing.assert_close(ratios.sqrt(), torch.tensor([1., .5]))

    def test_combined_gradient_equals_task_plus_anchor_and_zero_anchor_gradient_is_allowed(self):
        for weight in (0., 1.):
            encoder = make_encoder()
            with torch.no_grad():
                encoder.delta.fill_(.125)
            task = encoder(list(cached.CLASS_NAMES))[1].float().sum()
            anchor, _ = training.anchor_penalty(encoder)
            expected = torch.autograd.grad(task + weight * anchor, encoder.delta, retain_graph=True)[0]
            norms = training.apply_loss_gradients(task, anchor, encoder.delta, weight)
            torch.testing.assert_close(encoder.delta.grad, expected)
            self.assertTrue(all(value > 0 for value in norms["task"]))
        encoder = make_encoder()
        task = encoder(list(cached.CLASS_NAMES))[1].float().sum()
        anchor, _ = training.anchor_penalty(encoder)
        norms = training.apply_loss_gradients(task, anchor, encoder.delta, 1.)
        self.assertEqual(norms["anchor"], [0., 0.])

    def test_finite_connected_zero_task_gradient_allowed_per_frame_not_hidden_by_anchor(self):
        encoder = make_encoder()
        with torch.no_grad():
            encoder.delta.fill_(1.)
        task = encoder.delta[0].sum()
        anchor, _ = training.anchor_penalty(encoder)
        norms = training.apply_loss_gradients(task, anchor, encoder.delta, 1.)
        self.assertGreater(norms["task"][0], 0)
        self.assertEqual(norms["task"][1], 0.)
        self.assertGreater(norms["total"][1], 0)
        self.assertEqual(training.validate_task_gradient_history([norms["task"]], 1), [1, 0])
        with self.assertRaisesRegex(ValueError, "task-gradient coverage"):
            training.validate_task_gradient_history([norms["task"]] * 20, 20)
        disconnected = torch.ones((), requires_grad=True)
        anchor, _ = training.anchor_penalty(encoder)
        with self.assertRaises(RuntimeError):
            training.apply_loss_gradients(disconnected, anchor, encoder.delta, 1.)

    def test_sparse_task_gradient_windows_and_exact_nonzero_counters(self):
        history = [[0., 0.] for _ in range(200)]
        history[0] = [1., 0.]
        history[19] = [0., 2.]
        history[100] = [3., 0.]
        history[199] = [0., 4.]
        self.assertEqual(training.validate_task_gradient_history(history, 200), [2, 2])
        self.assertEqual(training.validate_task_gradient_history(history[:19], 19), [1, 0])
        changed = deepcopy(history)
        changed[199] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "101..200"):
            training.validate_task_gradient_history(changed, 200)
        for bad in ([[float("nan"), 1.]], [[-1., 1.]], [[1.]], []):
            with self.assertRaises(ValueError):
                training.validate_task_gradient_history(bad, 1)

    def test_bilateral_real_cpu_loader_and_collator_accept_both_single_and_empty(self):
        from sam3.train.data.collator import collate_fn_api

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = coco_fixture(root)
            save_json(root / "annotations.json", data)
            images, refs, summary = training.validate_training_annotations(data, {"fake/recording": 4})
            self.assertEqual(summary["both_hand_images"], 1)
            dataset = training.evaluation.make_dataset(root)
            for index, image in enumerate(images):
                batch = collate_fn_api([dataset[index]], dict_key="train", with_seg_masks=True)["train"]
                self.assertEqual(training.validate_bilateral_batch(batch, image["id"], refs[image["id"]]), [image["id"]])
            batch = collate_fn_api([dataset[0]], dict_key="train", with_seg_masks=True)["train"]
            batch.find_targets[0].num_boxes[1] = 0
            with self.assertRaisesRegex(RuntimeError, "target count"):
                training.validate_bilateral_batch(batch, images[0]["id"], refs[images[0]["id"]])

    def test_task_loss_allows_two_positive_queries_not_three(self):
        parameter = torch.ones((), requires_grad=True)
        model = SimpleNamespace(back_convert=lambda target: {"num_boxes": target.num_boxes},
                                matcher=lambda prediction, targets: "matched")

        class CallableModel:
            back_convert = staticmethod(model.back_convert)
            matcher = staticmethod(model.matcher)
            def __call__(self, batch):
                return [{}]
        batch = SimpleNamespace(find_targets=[SimpleNamespace(num_boxes=torch.tensor([1, 1]))])
        functions = [lambda **kwargs: {"core_loss": parameter * kwargs["num_boxes"]}] * 3
        loss, _ = training.compute_task_loss(CallableModel(), batch, functions)
        self.assertEqual(float(loss.detach()), 6.)
        batch.find_targets[0].num_boxes = torch.tensor([1, 2])
        with self.assertRaises(RuntimeError):
            training.compute_task_loss(CallableModel(), batch, functions)

    def test_ready_binds_train_roles_counts_sources_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(training, "TRAIN_RECORDINGS", {"fake/recording": 4}):
            parent = Path(directory)
            root = ready_fixture(parent)
            images, _, _, manifest, provenance = training.validate_ready_dataset(root)
            self.assertEqual(len(images), 4)
            self.assertEqual(provenance["dataset_role"], "train")
            output = manifest["image_outputs"][0]
            training.verify_selected_rgb(root, images[0], output)
            (root / images[0]["file_name"]).write_bytes(b"changed-rgb")
            with self.assertRaisesRegex(RuntimeError, "RGB bytes changed"):
                training.verify_selected_rgb(root, images[0], output)
            (parent / "source-0.txt").write_text("changed-source")
            with self.assertRaisesRegex(ValueError, "Original source differs"):
                training.validate_ready_dataset(root)

    def test_changed_split_or_recording_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = coco_fixture(Path(directory))
            for field, value in (("split", "val"), ("dataset_role", "development_holdout")):
                changed = deepcopy(data)
                changed["info"][field] = value
                with self.assertRaises(ValueError):
                    training.validate_training_annotations(changed, {"fake/recording": 4})
            data["images"][0]["recording_id"] = "wrong/recording"
            with self.assertRaises(ValueError):
                training.validate_training_annotations(data, {"fake/recording": 4})

    def test_checkpoint_resume_strictly_checks_anchor_learning_rate_and_curves(self):
        state, config, order, ids, initial = checkpoint_fixture()
        self.assertEqual(training.validate_resume(state, config, order, ids, initial), 1)
        self.assertEqual(state["format"], training.FORMAT)
        self.assertNotIn("class_tokens", state)
        for mutation in ("anchor", "optimizer_lr", "task_history", "drift", "observed", "grad_counts", "grad_norms"):
            changed = deepcopy(state)
            if mutation == "anchor":
                changed["training_config"]["anchor_weight"] = 0.
            elif mutation == "optimizer_lr":
                changed["optimizer"]["param_groups"][0]["lr"] = .01
            elif mutation == "task_history":
                changed["task_loss_history"][0] += 10
            elif mutation == "drift":
                changed["relative_drift_history"][0][0] = float("nan")
            elif mutation == "grad_counts":
                changed["gradient_nonzero_steps"][0] = 0
            elif mutation == "grad_norms":
                changed["task_grad_norm_history"][0][0] = 0.
            else:
                changed["observed_image_ids"][0] = -1
            with self.assertRaises(ValueError):
                training.validate_resume(changed, config, order, ids, initial)

    def test_weights_only_resume_and_two_trials_share_initial_cache_and_order(self):
        unconstrained, _, order0, _, initial0 = checkpoint_fixture(0.)
        anchored, config, order1, ids, initial1 = checkpoint_fixture(1.)
        self.assertEqual(order0, order1)
        self.assertEqual(training.shared.cache_fingerprint(initial0), training.shared.cache_fingerprint(initial1))
        self.assertEqual(unconstrained["observed_image_ids"], anchored["observed_image_ids"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.pt"
            training.shared.atomic_save(path, anchored)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(training.validate_resume(loaded, config, order1, ids, initial1), 1)

    def test_resume_accepts_actual_sparse_task_counts_before_completed_window(self):
        state, config, order, ids, initial = checkpoint_fixture(task_side=0)
        self.assertEqual(state["gradient_nonzero_steps"], [1, 0])
        self.assertEqual(state["task_grad_norm_history"][0][1], 0.)
        self.assertEqual(training.validate_resume(state, config, order, ids, initial), 1)


if __name__ == "__main__":
    unittest.main()

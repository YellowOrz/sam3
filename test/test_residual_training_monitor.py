import builtins
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from scripts.residual_training_monitor import TrainingMonitor


class TrainingMonitorTests(unittest.TestCase):
    def test_actual_events_keep_steps_tags_samples_and_validation_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = TrainingMonitor(temporary)
            self.assertFalse(monitor.log_dir.exists())
            with monitor:
                monitor.log_scalars(120, {
                    "train/total_loss": torch.tensor(1.25, requires_grad=True),
                    "train/loss_mask": np.float32(.5), "train/lr": .001,
                    "perf/images_per_second": 4., "perf/data_wait_ms": 3.,
                }, samples_seen=480, wall_seconds=12.5)
                monitor.log_validation(120, {"dice": .875}, scope="recording_val")
                monitor.log_scalars(121, {"train/total_loss": 1.}, 484, 13.)
            events = EventAccumulator(str(monitor.log_dir)).Reload()
            loss = events.Scalars("train/total_loss")
            self.assertEqual([item.step for item in loss], [120, 121])
            self.assertEqual([item.value for item in loss], [1.25, 1.])
            self.assertEqual(events.Scalars("train/loss_mask")[0].value, .5)
            self.assertEqual(events.Scalars("perf/images_per_second")[0].value, 4.)
            self.assertEqual(events.Scalars("progress/samples_seen")[0].value, 480.)
            self.assertEqual(events.Scalars("validation/recording_val/dice")[0].step, 120)
            rows = [json.loads(line) for line in monitor.jsonl_path.read_text().splitlines()]
            self.assertEqual([row["global_step"] for row in rows], [120, 120, 121])
            self.assertEqual(rows[0]["samples_seen"], 480)
            self.assertEqual(rows[0]["wall_seconds"], 12.5)
            self.assertEqual(rows[1]["kind"], "validation")
            self.assertEqual(rows[1]["scope"], "recording_val")
            self.assertEqual(rows[1]["samples_seen"], 480)
            self.assertEqual(rows[2]["values"], {"train/total_loss": 1.})

    def test_rank_one_disabled_and_unused_monitors_never_write_or_import_writer(self):
        actual_import = builtins.__import__

        def without_tensorboard(name, *args, **kwargs):
            if name.startswith("torch.utils.tensorboard"):
                raise AssertionError("inactive monitor imported SummaryWriter")
            return actual_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            with patch("builtins.__import__", side_effect=without_tensorboard):
                for name, kwargs in (("rank1", {"rank": 1}), ("off", {"enabled": False})):
                    output = Path(temporary) / name
                    with TrainingMonitor(output, **kwargs) as monitor:
                        monitor.log_scalars(10, {"train/total_loss": 1.}, 20, 1.)
                        monitor.log_images(10, {"preview/rgb": np.zeros((8, 8, 3), np.uint8)})
                        monitor.log_validation(10, {"dice": .5}, "val")
                        monitor.flush()
                    self.assertFalse(output.exists())
                with TrainingMonitor(Path(temporary) / "unused"):
                    pass
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_nonfinite_validation_and_training_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with TrainingMonitor(temporary) as monitor:
                for value in (float("nan"), float("inf"), -float("inf")):
                    with self.subTest(value=value):
                        with self.assertRaises(ValueError):
                            monitor.log_scalars(1, {"train/loss": 1., "train/bad": value}, 2, 1.)
                        with self.assertRaises(ValueError):
                            monitor.log_validation(1, {"dice": value}, "val")
                        with self.assertRaises(ValueError):
                            monitor.log_scalars(1, {"train/loss": 1.}, 2, value)
                self.assertFalse(monitor.log_dir.exists())

    def test_steps_and_samples_do_not_rewind_and_bad_call_does_not_advance(self):
        with tempfile.TemporaryDirectory() as temporary:
            with TrainingMonitor(temporary) as monitor:
                monitor.log_scalars(50, {"train/total_loss": 1.}, 100, 2.)
                for operation in (
                    lambda: monitor.log_scalars(49, {"train/total_loss": 1.}, 101, 3.),
                    lambda: monitor.log_validation(49, {"dice": .5}, "val"),
                    lambda: monitor.log_images(49, {}),
                    lambda: monitor.log_scalars(51, {"train/total_loss": 1.}, 99, 3.),
                    lambda: monitor.log_scalars(52, {"train/total_loss": float("nan")}, 102, 3.),
                ):
                    with self.assertRaises(ValueError):
                        operation()
                monitor.log_validation(50, {"dice": .5}, "val")
                monitor.log_scalars(51, {"train/total_loss": .9}, 102, 3.)
            rows = [json.loads(line) for line in monitor.jsonl_path.read_text().splitlines()]
            self.assertEqual([row["global_step"] for row in rows], [50, 50, 51])

    def test_images_are_separate_and_preserve_rgb_and_binary_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            rgb = np.zeros((6, 8, 3), dtype=np.uint8)
            rgb[..., 0] = 90
            reference = np.zeros((6, 8), dtype=bool)
            reference[1:3, 2:5] = True
            predicted = torch.zeros((1, 6, 8), dtype=torch.bfloat16)
            predicted[:, 2:4, 3:6] = 1
            with TrainingMonitor(temporary) as monitor:
                monitor.log_images(7, {"preview/rgb": rgb, "preview/gt": reference,
                                       "preview/pred": predicted})
            events = EventAccumulator(str(monitor.log_dir)).Reload()
            self.assertEqual(set(events.Tags()["images"]), {"preview/rgb", "preview/gt", "preview/pred"})
            for tag in events.Tags()["images"]:
                self.assertEqual(events.Images(tag)[0].step, 7)
            decoded_rgb = np.asarray(Image.open(io.BytesIO(events.Images("preview/rgb")[0].encoded_image_string)))
            decoded_gt = np.asarray(Image.open(io.BytesIO(events.Images("preview/gt")[0].encoded_image_string)))
            np.testing.assert_array_equal(decoded_rgb, rgb)
            np.testing.assert_array_equal(decoded_gt[..., 0], reference.astype(np.uint8) * 255)

    def test_explicit_flush_and_context_exception_preserve_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            monitor = TrainingMonitor(temporary, flush_seconds=3600)
            with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                with monitor:
                    monitor.log_scalars(1, {"train/total_loss": .5}, 1, .1)
                    monitor.flush()
                    self.assertEqual(len(monitor.jsonl_path.read_text().splitlines()), 1)
                    live = EventAccumulator(str(monitor.log_dir)).Reload()
                    self.assertEqual(live.Scalars("train/total_loss")[0].step, 1)
                    monitor.log_scalars(2, {"train/total_loss": .4}, 2, .2)
                    raise RuntimeError("training interrupted")
            events = EventAccumulator(str(monitor.log_dir)).Reload()
            self.assertEqual([item.step for item in events.Scalars("train/total_loss")], [1, 2])
            self.assertEqual(len(monitor.jsonl_path.read_text().splitlines()), 2)
            monitor.close()
            with self.assertRaises(RuntimeError):
                monitor.log_scalars(3, {"train/total_loss": .3}, 3, .3)

    def test_bad_scalar_shapes_steps_tags_and_pixels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with TrainingMonitor(temporary) as monitor:
                for step in (-1, 1.5, True):
                    with self.assertRaises(ValueError):
                        monitor.log_scalars(step, {"train/loss": 1.}, 1, 1.)
                for value in (torch.ones(2), np.ones(1), "1", True):
                    with self.assertRaises(ValueError):
                        monitor.log_scalars(1, {"train/loss": value}, 1, 1.)
                for tag in ("", "train//loss", "../loss", " train/loss"):
                    with self.assertRaises(ValueError):
                        monitor.log_scalars(1, {tag: 1.}, 1, 1.)
                with self.assertRaises(ValueError):
                    monitor.log_scalars(1, {"progress/samples_seen": 1.}, 1, 1.)
                for value in (np.ones((2, 6, 8)), np.ones((6, 8)) * 2,
                              np.full((6, 8), float("nan")), np.zeros((0, 8))):
                    with self.assertRaises(ValueError):
                        monitor.log_images(1, {"preview/gt": value})
                self.assertFalse(monitor.log_dir.exists())


if __name__ == "__main__":
    unittest.main()

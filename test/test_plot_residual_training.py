"""CPU plotting contracts, real monitor/event roundtrips and optional real-run audit."""

from copy import deepcopy
import hashlib
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from scripts import plot_residual_training as plotting
from scripts.residual_training_monitor import TrainingMonitor


def train_record(step=1, loss=1.):
    return {"kind": "train", "global_step": step, "samples_seen": step * 6,
            "wall_seconds": step * 2., "values": {
                "train/total_loss": loss,
                **{f"train/{key}": loss / 6 for key in plotting.LOSS_KEYS},
                "perf/images_per_second": 3., "perf/data_wait_ms_per_step": 2.,
                "perf/h2d_ms_per_step": 1., "perf/compute_ms_per_step": 1900.}}


def validation_record(step=0, left=.3, right=.7):
    values = {}
    for side, dice in (("left_hand", left), ("right_hand", right)):
        values.update({f"{side}/miss_zero_dice": dice, f"{side}/candidate_dice": .8,
                       f"{side}/positive_count": 8, f"{side}/absent_count": 2,
                       f"{side}/false_negative_count": 1, f"{side}/false_positive_count": 1,
                       f"{side}/false_negative_rate": .125, f"{side}/false_positive_rate": .5,
                       f"{side}/miss_zero_boundary_iou_4px": .2,
                       f"{side}/candidate_boundary_iou_4px": .3})
    return {"kind": "validation", "scope": "dexycb_val", "global_step": step,
            "samples_seen": None if step == 0 else step * 6, "values": values}


def write_records(directory, records, tail=b""):
    directory.mkdir(parents=True)
    path = directory / "metrics.jsonl"
    path.write_bytes(b"".join((json.dumps(record) + "\n").encode() for record in records) + tail)
    return path


class LogReaderTests(unittest.TestCase):
    def test_cli_refuses_multiple_runs_instead_of_silently_using_last_argument(self):
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            plotting.main(["--log-dir", "/run/one", "--log-dir", "/run/two", "--output-dir", "/tmp/unused"])

    def test_reads_exact_double_precision_and_preserves_real_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tensorboard"
            path = write_records(root, [validation_record(), train_record(1, .123456789012345), train_record(20, .4)])
            before = path.read_bytes()
            result = plotting.load_log(root)
            self.assertEqual(result["series"]["train/total_loss"], {1: .123456789012345, 20: .4})
            self.assertEqual(result["source"]["captured_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(path.read_bytes(), before)

    def test_conflicting_duplicate_rejected_identical_duplicate_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            record = train_record()
            path = write_records(root, [record, deepcopy(record)])
            self.assertGreater(plotting.load_log(root)["source"]["identical_duplicate_scalars_deduplicated"], 0)
            changed = deepcopy(record)
            changed["values"]["train/total_loss"] += .01
            path.write_text(json.dumps(record) + "\n" + json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                plotting.load_log(root)

    def test_live_unfinished_tail_excluded_but_complete_malformed_line_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            path = write_records(root, [train_record()], b'{"kind":')
            result = plotting.load_log(root)
            self.assertEqual(result["source"]["complete_records"], 1)
            self.assertIn("unfinished", result["warnings"][0])
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "record 2"):
                plotting.load_log(root)

    def test_invalid_scalars_and_regressing_steps_counters_are_rejected(self):
        for key, replacement in (("global_step", True), ("samples_seen", -1),
                                 ("wall_seconds", float("nan")), ("kind", "test")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "logs"
                record = train_record()
                record[key] = replacement
                write_records(root, [record])
                with self.assertRaises(ValueError):
                    plotting.load_log(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "logs"
            write_records(root, [train_record(20), train_record(1)])
            with self.assertRaisesRegex(ValueError, "regressed"):
                plotting.load_log(root)

    def test_rolling_mean_uses_actual_records_not_missing_optimizer_steps(self):
        points = [(1, 1.), (20, 3.), (200, 5.)]
        self.assertEqual(plotting.rolling_mean(points), [(1, 1.), (20, 2.), (200, 3.)])
        points = [(index * 10, float(index)) for index in range(21)]
        result = plotting.rolling_mean(points)
        self.assertEqual(result[-1], (200, 10.5))
        self.assertEqual([step for step, _ in result], [step for step, _ in points])


class PlotExportTests(unittest.TestCase):
    def test_synthetic_export_pngs_exact_summary_and_no_interpolated_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "logs"
            source = write_records(root, [validation_record(0), train_record(1), train_record(20, .5), validation_record(20, .4, .8)])
            source_before = source.read_bytes()
            report = plotting.export_plots(root, directory / "plots")
            self.assertEqual(len(report["files"]), 6)
            self.assertFalse(report["validation_interpolation"])
            self.assertEqual(report["smoothing"]["window_records"], 20)
            self.assertEqual(report["smoothing"]["min_periods"], 1)
            self.assertIn("NOT epoch", report["loss_interpretation"])
            macro = report["series"]["validation/dexycb_val/derived_macro_miss_zero_dice"]
            self.assertEqual(macro["steps"], [0, 20])
            self.assertEqual(macro["values"][0], .5)
            self.assertAlmostEqual(macro["values"][1], .6)
            self.assertEqual(report["series"]["train/total_loss"]["steps"], [1, 20])
            self.assertEqual(report["observations"]["last_recorded_scalars"]["train/total_loss"],
                             {"step": 20, "value": .5})
            for file in report["files"]:
                with Image.open(file) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width, 1000)
                    self.assertGreater(image.height, 500)
            self.assertEqual(json.loads((directory / "plots/summary.json").read_text()), report)
            self.assertEqual(source.read_bytes(), source_before)
            with self.assertRaises(FileExistsError):
                plotting.export_plots(root, directory / "plots")

    def test_validation_panel_draws_no_data_connecting_lines(self):
        image = Image.new("RGB", (1440, 600), "white")
        draw = plotting.ImageDraw.Draw(image)
        with mock.patch.object(draw, "line", wraps=draw.line) as line:
            plotting._panel(draw, (0, 0, 1440, 500), "Validation", [
                ("left", [(0, .2), (20, .8)], "#2474b5")], validation=True, limits=(0, 1))
        # Six horizontal grids plus one axes polyline; no connection between
        # the measured validation points is drawn.
        self.assertEqual(line.call_count, 7)

    def test_missing_side_or_positive_count_does_not_fabricate_macro(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = validation_record()
            del record["values"]["right_hand/positive_count"]
            write_records(directory / "logs", [record])
            report = plotting.export_plots(directory / "logs", directory / "plots")
            self.assertNotIn("validation/dexycb_val/derived_macro_miss_zero_dice", report["series"])
            self.assertTrue(any("Macro Dice omitted" in warning for warning in report["warnings"]))
            self.assertFalse(any("training_" in Path(file).name for file in report["files"]))

    def test_conflicting_logged_macro_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = validation_record()
            record["values"]["derived_macro_miss_zero_dice"] = .99
            write_records(directory / "logs", [record])
            with self.assertRaisesRegex(ValueError, "macro conflicts"):
                plotting.export_plots(directory / "logs", directory / "plots")
            self.assertFalse((directory / "plots").exists())

    def test_invalid_bounded_metric_is_not_silently_clipped_out_of_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = validation_record()
            record["values"]["left_hand/candidate_dice"] = 1.1
            write_records(directory / "logs", [record])
            with self.assertRaisesRegex(ValueError, "plotting range"):
                plotting.export_plots(directory / "logs", directory / "plots")
            self.assertFalse((directory / "plots").exists())

    def test_rejects_repository_or_source_directory_outputs_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_records(directory / "logs", [train_record()])
            for output in (Path(plotting.__file__).resolve().parents[1] / "not-created-plot-output",
                           directory / "logs/plots"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    plotting.export_plots(directory / "logs", output)
                self.assertFalse(output.exists())

    def test_actual_training_monitor_jsonl_and_events_roundtrip_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with TrainingMonitor(directory / "actual-monitor") as monitor:
                monitor.log_validation(0, validation_record()["values"], "dexycb_val")
                for step in (1, 20):
                    record = train_record(step, .123456789012345)
                    monitor.log_scalars(step, record["values"], record["samples_seen"], record["wall_seconds"])
                monitor.log_validation(20, validation_record(20)["values"], "dexycb_val")
            log_dir = directory / "actual-monitor/tensorboard"
            jsonl = plotting.export_plots(log_dir, directory / "jsonl-plots")
            events = plotting.export_plots(log_dir, directory / "event-plots", source="events")
            self.assertEqual(jsonl["source"]["source_format"], "jsonl")
            self.assertEqual(events["source"]["source_format"], "events")
            self.assertEqual(jsonl["series"]["train/total_loss"]["steps"], [1, 20])
            self.assertEqual(events["series"]["train/total_loss"]["steps"], [1, 20])
            self.assertEqual(jsonl["series"]["train/total_loss"]["values"][0], .123456789012345)
            self.assertAlmostEqual(events["series"]["train/total_loss"]["values"][0], .123456789012345, places=7)

    def test_real_engineering_b2_jsonl_export_preserves_every_record(self):
        project = Path(plotting.__file__).resolve().parents[1]
        log_dir = project / "runs/residual-ddp-b2-benchmark-20260911/tensorboard"
        source = log_dir / "metrics.jsonl"
        if not source.is_file():
            self.skipTest("Optional local engineering run is not present in this checkout")
        raw = source.read_bytes()
        records = [json.loads(line) for line in raw.splitlines()]
        expected = [(record["global_step"], record["values"]["train/total_loss"])
                    for record in records if record["kind"] == "train"]
        with tempfile.TemporaryDirectory() as temporary:
            report = plotting.export_plots(log_dir, Path(temporary) / "engineering-plots")
            actual = report["series"]["train/total_loss"]
            self.assertEqual(list(zip(actual["steps"], actual["values"])), expected)
            self.assertEqual(report["source"]["captured_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(source.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import evaluate_nakehand_tokens as evaluation
from scripts import report_nakehand_evaluation as report


def fixture(root: Path, primary=True):
    data_root = root / "data"
    data_root.mkdir()
    Image.new("RGB", (8, 8)).save(data_root / "rgb.png")
    image = {"id": 4, "file_name": "rgb.png", "height": 8, "width": 8,
             "primary_test": primary, "diagnostic_id": "A", "recording_id": "ego/one", "view_type": "ego"}
    references = []
    annotations = []
    for index in range(2):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:4, (1 if index == 0 else 5):(3 if index == 0 else 7)] = 1
        references.append(mask.astype(bool))
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        annotations.append({"id": index, "image_id": 4, "category_id": index + 1, "segmentation": rle})
    data = {"images": [image], "annotations": annotations,
            "categories": [{"id": key, "name": value} for key, value in evaluation.SIDE_BY_CATEGORY.items()]}
    (data_root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")
    base = root / "base.pt"
    base.write_bytes(b"test-base-checkpoint")
    token = root / "tokens.pt"
    token.write_bytes(b"test-learned-checkpoint")
    records_dir = root / "records"
    records_dir.mkdir()
    models = {"epoch2": {"kind": "learned_class", "checkpoint": str(token),
                         "checkpoint_sha256": evaluation.shared.sha256(token), "completed_epochs_from_steps": 2},
              "ve-underscore": {"kind": "ve"}, "ve-natural": {"kind": "ve"}}
    all_records = []
    for label in models:
        rows = []
        for index, side in enumerate(evaluation.CLASS_NAMES):
            rows.append({"model": label, "image_id": 4, "dataset_index": 0, "prompt_key": side,
                         "primary_test": primary, "diagnostic_ids": ["A"], "recording_id": "ego/one",
                         "view_type": "ego", "file_name": "rgb.png", "identity_verified": True,
                         "observed_coco_image_id": 4, "top_class_probability": .8, "presence_probability": 1.,
                         **evaluation.measure_query(references[index], references[index], references[1 - index], .8)})
        all_records += rows
        (records_dir / f"{label}.json").write_text(json.dumps(rows), encoding="utf-8")
    summary = {"format": "nakehand-frozen-bilateral-evaluation-v1", "status": "completed",
               "data_root": str(data_root), "base_checkpoint": str(base),
               "base_checkpoint_sha256": evaluation.shared.sha256(base),
               "annotations_sha256": evaluation.shared.sha256(data_root / "annotations.json"),
               "detection_threshold": .5, "mask_threshold": .5, "thresholds_fitted_on_nakehand": False,
               "training_performed": False, "observed_identity_verified": True,
               "evaluated_images": 1, "evaluated_dataset_indices": [0], "evaluated_image_ids": [4],
               "primary_test_images": int(primary), "diagnostic_images": 1, "full_export_evaluated": True,
               "models": models, "metrics": evaluation.summarize(all_records), "visuals": []}
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, summary, all_records


class NakehandReportTest(unittest.TestCase):
    def test_three_model_report_revalidates_counts_hashes_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = fixture(Path(directory))
            summary, records, sources = report.validate_run(path)
            text = report.render_report(summary, records, sources)
            self.assertIn("主测试结果", text)
            self.assertIn("实际只有 1 张主样本", text)
            self.assertIn("SAM3 辅助生成参考", text)
            self.assertIn("不能仅凭路径证明训练时字节完全相同", text)
            self.assertIn("0/0（N/A）", text)
            self.assertEqual(len(sources["records"]), 3)

    def test_diagnostic_only_has_no_manufactured_primary_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = fixture(Path(directory), primary=False)
            summary, records, sources = report.validate_run(path)
            text = report.render_report(summary, records, sources)
            self.assertIn("仅诊断冒烟，不是主测试结论", text)
            self.assertNotIn("## 主测试结果", text)
            self.assertNotIn("主样本总体", text)
            self.assertIn("A/B/C 人工认可样例", text)

    def test_record_model_query_counts_and_pair_identity_are_checked(self):
        for mutation in ("missing", "duplicate", "model", "annotation"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _, _ = fixture(root)
                record_path = root / "records/epoch2.json"
                rows = json.loads(record_path.read_text())
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows[1] = rows[0]
                elif mutation == "model":
                    rows[0]["model"] = "other"
                else:
                    rows[0]["reference_pixels"] += 1
                record_path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    report.validate_run(path)

    def test_changed_annotation_or_base_hash_rejected(self):
        for target in ("base.pt", "data/annotations.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _, _ = fixture(root)
                changed = root / target
                changed.write_bytes(changed.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    report.validate_run(path)

    def test_unfinished_training_test_tuning_and_corrupted_summary_rejected(self):
        for mutation in ("status", "calibration", "metrics", "epochs"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path, summary, _ = fixture(Path(directory))
                if mutation == "status":
                    summary["status"] = "running"
                elif mutation == "calibration":
                    summary["thresholds_fitted_on_nakehand"] = True
                elif mutation == "metrics":
                    summary["metrics"]["epoch2"]["primary_test"]["overall"]["present_mean_candidate_dice"] = .123
                else:
                    summary["models"]["epoch2"]["completed_epochs_from_steps"] = 1
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    report.validate_run(path)

    def test_main_refuses_to_overwrite_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = fixture(root)
            output = root / "report.md"
            report.main(["--summary", str(path), "--output", str(output)])
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                report.main(["--summary", str(path), "--output", str(output)])
            self.assertEqual(output.read_bytes(), original)

    def test_low_score_swapped_candidates_remain_visible_with_detected_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, summary, rows = fixture(root)
            _, references, _ = evaluation.load_coco_index(root / "data")
            for row in rows:
                if row["model"] != "epoch2":
                    continue
                side = row["prompt_key"]
                other = evaluation.CLASS_NAMES[1 - evaluation.CLASS_NAMES.index(side)]
                row.update(evaluation.measure_query(references[4][other], references[4][side], references[4][other], .2))
                row["top_class_probability"] = .2
            (root / "records/epoch2.json").write_text(json.dumps([row for row in rows if row["model"] == "epoch2"]))
            summary["metrics"] = evaluation.summarize(rows)
            path.write_text(json.dumps(summary))
            checked, records, sources = report.validate_run(path)
            text = report.render_report(checked, records, sources)
            self.assertIn("不保证 mask 对应正确的物理手", text)
            self.assertIn("候选错侧/全部双手查询", text)
            self.assertIn("| epoch2 | 主样本双手图 | 2/2（100.00%） | 0/2（0.00%） | 0/0（N/A） |", text)
            self.assertIn("| epoch2 | A | 2/2（100.00%） | 0/2（0.00%） | 0/0（N/A） |", text)


if __name__ == "__main__":
    unittest.main()

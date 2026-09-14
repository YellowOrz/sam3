from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_residual_dex_only import prepare, select_dex, sha
from test_train_residual_ddp import _fixture


class DexOnlyTest(unittest.TestCase):
    def test_select_keeps_rows_and_rejects_wrong_split(self):
        data = {"images": [{"id": 1, "source_dataset": "dexycb", "source_split": "train",
                            "file_name": "images/1.png"},
                           {"id": 2, "source_dataset": "nakehand", "file_name": "images/2.png"}],
                "annotations": [{"id": 1, "image_id": 1}, {"id": 2, "image_id": 2}], "info": {}}
        old = deepcopy(data)
        selected = select_dex(data, split="train", expected_count=1)
        self.assertEqual(selected["images"], [data["images"][0]])
        self.assertEqual(selected["annotations"], [data["annotations"][0]])
        self.assertEqual(data, old)
        with self.assertRaises(ValueError):
            select_dex(data, split="val", expected_count=1)
        with self.assertRaises(ValueError):
            select_dex(data, split="train", expected_count=2)

    def fixture(self, source):
        args = _fixture(source)
        (source / "training-approval.json").write_bytes(args.approval.read_bytes())
        return args

    def test_publication_rows_pixels_val_unchanged_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "dex"
            source.mkdir()
            self.fixture(source)
            before = (source / "train/annotations.json").read_bytes()
            result = prepare(source, output, expected_train=7, expected_val=3)
            self.assertTrue(result["all_selected_rgb_hashes_verified"])
            self.assertEqual(before, (source / "train/annotations.json").read_bytes())
            self.assertTrue((output / "val").is_symlink())
            self.assertEqual(sha(output / "val/annotations.json"), sha(source / "val/annotations.json"))
            a = json.loads(before)
            b = json.loads((output / "train/annotations.json").read_bytes())
            self.assertEqual(a["images"], b["images"])
            self.assertEqual(a["annotations"], b["annotations"])
            with self.assertRaises(ValueError):
                prepare(source, output, expected_train=7, expected_val=3)

    def test_changed_rgb_or_annotations_fail_before_publication(self):
        for kind in ("rgb", "annotations"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                source, output = Path(temp) / "source", Path(temp) / "dex"
                source.mkdir()
                self.fixture(source)
                target = source / "train" / ("images/image-0.png" if kind == "rgb" else "annotations.json")
                target.write_bytes(b"modified")
                with self.assertRaises(ValueError):
                    prepare(source, output, expected_train=7, expected_val=3)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

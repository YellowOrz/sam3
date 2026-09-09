"""Dependency-free API/CLI checks, also runnable directly with Python."""

import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class LearnedPromptContractTests(unittest.TestCase):
    def test_existing_builder_positional_signatures_are_preserved(self):
        tree = ast.parse((ROOT / "sam3/model_builder.py").read_text(encoding="utf-8"))
        expected = {
            "build_sam3_image_model": [
                "bpe_path",
                "device",
                "eval_mode",
                "checkpoint_path",
                "load_from_HF",
                "enable_segmentation",
                "enable_inst_interactivity",
                "compile",
            ],
            "build_sam3_video_model": [
                "checkpoint_path",
                "load_from_HF",
                "bpe_path",
                "has_presence_token",
                "geo_encoder_use_img_cross_attn",
                "strict_state_dict_loading",
                "apply_temporal_disambiguation",
                "device",
                "compile",
            ],
        }
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in expected:
                self.assertEqual(
                    [arg.arg for arg in node.args.args],
                    expected[node.name] + ["learned_prompt_path"],
                )
                self.assertIsNone(node.args.defaults[-1].value)

    def test_prepare_cli_rejects_invalid_input_and_existing_output(self):
        spec = importlib.util.spec_from_file_location(
            "prepare_prompt", ROOT / "scripts/prepare_learned_prompt.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "feature.pt"
            arguments = [
                "--checkpoint",
                "sam3.pt",
                "--target-id",
                "left",
                "--reference-text",
                "left hand",
                "--output",
                str(output),
            ]
            self.assertEqual(module.parse_args(arguments).init, "text")
            with self.assertRaises(SystemExit):
                module.parse_args(arguments + ["--random-std", "0"])
            output.write_bytes(b"existing")
            with self.assertRaises(SystemExit):
                module.parse_args(arguments)
            self.assertEqual(output.read_bytes(), b"existing")

    def test_cuda_ids_parse_multi_gpu_device_string(self):
        tree = ast.parse((ROOT / "sam3/train/learned_prompt.py").read_text())
        node = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "_cuda_ids"
        )
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "cuda_ids", "exec"), namespace)
        parse = namespace["_cuda_ids"]
        self.assertEqual(parse("cuda:1,2,3"), [1, 2, 3])
        self.assertEqual(parse("cuda:1"), [1])
        self.assertEqual(parse("cpu"), None)
        # Execute the real native loader and TargetCOCO class with only tensor/RLE
        # helpers substituted; this check exercises their actual query generation.
        import collections

        native = ast.parse((ROOT / "sam3/train/data/coco_json_loaders.py").read_text())
        selected = ast.parse((ROOT / "sam3/train/learned_prompt.py").read_text())
        nodes = [
            node
            for node in native.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in ("load_coco_and_group_by_image", "COCO_FROM_JSON")
        ]
        nodes += [
            node
            for node in selected.body
            if isinstance(node, ast.ClassDef) and node.name == "TargetCOCO"
        ]

        class Box(list):
            def __mul__(self, value):
                return types.SimpleNamespace(item=lambda: float(self[0]) * value)

        class Bbox:
            def __getitem__(self, index):
                return Box([0.5]) if index == 2 else 0.5

        namespace = {
            "Dict": dict,
            "List": list,
            "Tuple": tuple,
            "json": json,
            "defaultdict": collections.defaultdict,
            "convert_boxlist_to_normalized_tensor": lambda *args: [Bbox()],
            "ann_to_rle": lambda seg, **kw: seg,
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "target_loader", "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instances.json"
            path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": i,
                                "file_name": f"{i}.jpg",
                                "width": 10,
                                "height": 10,
                            }
                            for i in (1, 2)
                        ],
                        "categories": [
                            {"id": 1, "name": "left"},
                            {"id": 2, "name": "right"},
                        ],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [0, 0, 5, 5],
                                "segmentation": [[0, 0, 5, 0, 5, 5]],
                            },
                            {"id": 2, "image_id": 2, "category_id": 2},
                        ],
                    }
                )
            )
            with patch.dict(
                sys.modules, {"pycocotools": types.SimpleNamespace(mask=None)}
            ):
                loader = namespace["TargetCOCO"](str(path), 1, "left_feature")
            positive, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)
            negative, absent_annotations = (
                loader.loadQueriesAndAnnotationsFromDatapoint(1)
            )
            self.assertEqual(len(annotations), 1)
            self.assertEqual(positive[0]["query_text"], "left_feature")
            self.assertEqual(negative[0]["original_cat_id"], 1)
            self.assertEqual(negative[0]["object_ids_output"], [])
            self.assertEqual(absent_annotations, [])
            self.assertEqual(loader.getDatapointIds(), [0, 1])


if __name__ == "__main__":
    unittest.main()

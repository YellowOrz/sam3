import unittest
import argparse
import ast
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from scripts.eval import mano_box_factorial as f


class FactorialProtocolTests(unittest.TestCase):
    def test_tapped_method_is_in_actual_model_source(self):
        source = Path(__file__).parents[1] / "sam3/model/sam3_video_base.py"
        methods = {node.name: node for node in ast.walk(ast.parse(source.read_text()))
                   if isinstance(node, ast.FunctionDef)}
        method = methods["run_backbone_and_detection"]
        names = {arg.arg for arg in method.args.args}
        self.assertTrue({"frame_idx", "geometric_prompt", "feature_cache"}.issubset(names))

    def test_source_sampling_keeps_start_and_tail(self):
        self.assertEqual(f.sampling_indices(8), [0, 3, 6])
        self.assertEqual(f.sampling_indices(1), [0])
        for value in (0, -1, True, 3.0):
            with self.assertRaises(ValueError):
                f.sampling_indices(value)

    def test_box_format_is_explicit(self):
        np.testing.assert_allclose(f.box_cxcywh([.1, .2, .4, .6]), [.3, .5, .4, .6])
        for box in ([0, 0, 0, .5], [0, 0, 2, 1], [.9, 0, .2, .1], [float('nan'), 0, 1, 1]):
            with self.assertRaises(ValueError):
                f.box_cxcywh(box)

    def test_empty_single_empty_and_unknown_distinct(self):
        zero = np.zeros((12, 14), bool)
        one = zero.copy(); one[3:7, 4:9] = True
        self.assertEqual(f.score(zero, zero)["dice"], 1)
        self.assertEqual(f.score(zero, one)["dice"], 0)
        self.assertTrue(f.score(zero, one)["missed_positive"])
        self.assertTrue(f.score(one, zero)["empty_false_positive"])
        self.assertIsNone(f.score(one, zero)["boundary_iou_4px"])
        self.assertIsNone(f.score(one, zero)["other_only_overlap_pixels"])

    def test_instance_union_not_top_candidate(self):
        masks = np.zeros((2, 10, 10), bool)
        masks[0, :3, :3] = True; masks[1, 6:, 6:] = True
        result = f.instance_record(masks, [.9, .8], [7, 12], (10, 10))
        np.testing.assert_array_equal(f.decode(result["union_rle"]), masks.any(0))
        self.assertEqual(result["prediction_pixels"], 25)
        # Scores already accepted by the video policy are not gated a second time.
        self.assertEqual(f.instance_record(masks[:1], [.2], [7], (10, 10))["prediction_pixels"], 9)

    def test_reject_invalid_instances(self):
        with self.assertRaises(ValueError):
            f.instance_record(np.zeros((2, 3, 3)), [.5, .5], [1, 1], (3, 3))
        with self.assertRaises(ValueError):
            f.instance_record(np.zeros((1, 3, 3)), [float('nan')], [1], (3, 3))

    def test_full_contiguous_coverage_before_sampling(self):
        rows = [dict(sequence="a", frame_index=i, method=m) for i in range(8)
                for m in ("frame_text", "video_text")]
        f.verify_rows(rows, {"a": 8}, ("frame_text", "video_text"))
        for bad in (rows[:-1], rows + rows[:1], [r for r in rows if r["frame_index"] % 3 == 0]):
            with self.assertRaises(ValueError):
                f.verify_rows(bad, {"a": 8}, ("frame_text", "video_text"))

    def test_boundary_and_wrong_side_diagnostic(self):
        target = np.zeros((20, 20), bool); target[2:8, 2:8] = True
        other = np.zeros_like(target); other[12:18, 12:18] = True
        result = f.score(target | other, target, other)
        self.assertAlmostEqual(result["dice"], 2/3)
        self.assertEqual(result["other_only_overlap_pixels"], 36)
        self.assertEqual(f.score(target, target)["boundary_iou_4px"], 1)

    def test_cpu_audit_and_four_condition_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "inputs"
            for sub in ("rgb", "right"):
                (root / "sample" / sub).mkdir(parents=True)
            refs = []
            for i in range(4):
                ref = np.zeros((16, 16), bool)
                if i == 3:
                    ref[3:8, 3:8] = True
                refs.append(ref)
                Image.fromarray(ref.astype(np.uint8)*255).save(root / "sample/right" / f"{i:06d}.png")
                Image.new("RGB", (16, 16)).save(root / "sample/rgb" / f"{i:06d}.png")
            seq = dict(name="sample", frame_count=4, sampled_indices=[0, 3], render_indices=[0, 3],
                prompts={}, file_sha256={str(p.relative_to(root)): f.sha(p) for p in root.rglob("*.png")})
            f.write_json(root / "plan.json", dict(contract=f.CONTRACT, sequences=[seq], scoring_frames=2))
            args = dict(root=root, output=Path(tmp)/"comparison")
            for mode in ("text", "box"):
                run = Path(tmp) / mode; run.mkdir()
                with (run / "sample.jsonl").open("w") as stream:
                    for i, ref in enumerate(refs):
                        for stage in ("frame", "video"):
                            row = dict(sequence="sample", frame_index=i, method=f"{stage}_{mode}",
                                geometry=[], geometry_available=False,
                                **f.instance_record(ref[None], [.9], [1], ref.shape))
                            stream.write(json.dumps(row)+"\n")
                summary = dict(status="complete", mode=mode, engineering=False, counts={"sample":4},
                    contract=f.CONTRACT, plan_sha256=f.sha(root/"plan.json"), runner_sha256="runner",
                    model_source_sha256={}, base_sha256="base", torch_version="test",
                    records_sha256={"sample":f.sha(run/"sample.jsonl")})
                f.write_json(run/"summary.json", summary)
                audit = Path(tmp) / (mode+"-audit")
                f.audit(argparse.Namespace(root=root, run=run, output=audit))
                args[mode+"_run"] = run; args[mode+"_audit"] = audit
            f.compare(argparse.Namespace(**args))
            result = json.loads((args["output"]/"metrics.json").read_text())
            self.assertEqual(result["scoring_frames_per_condition"], 2)
            self.assertEqual(set(result["overall"]), set(f.METHODS))
            self.assertTrue(all(v["mean_dice"] == 1 for v in result["overall"].values()))
            # Already published outputs are immutable, not silently overwritten.
            with self.assertRaises(ValueError):
                f.compare(argparse.Namespace(**args))
            (args["text_run"]/"sample.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                f.audit(argparse.Namespace(root=root, run=args["text_run"], output=Path(tmp)/"tampered-audit"))


if __name__ == "__main__":
    unittest.main()

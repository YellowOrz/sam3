import unittest
import argparse
import ast
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from scripts.eval import mano_box_factorial as f


class FactorialProtocolTests(unittest.TestCase):
    def test_sequence_partition_selects_complete_registered_video_only(self):
        plan = dict(sequences=[dict(name='a', frame_count=99), dict(name='b', frame_count=120)])
        self.assertEqual(f.select_sequences(plan, sequence='b'), [plan['sequences'][1]])
        self.assertEqual(f.select_sequences(plan), plan['sequences'])
        for name in ('missing', '', '../b'):
            with self.assertRaisesRegex(ValueError, 'exactly one registered'):
                f.select_sequences(plan, sequence=name)
        with self.assertRaisesRegex(ValueError, 'full original video'):
            f.select_sequences(plan, engineering=True, sequence='b')
        plan['sequences'].append(dict(name='b', frame_count=5))
        with self.assertRaisesRegex(ValueError, 'exactly one registered'):
            f.select_sequences(plan, sequence='b')

    def test_storage_trace_requires_actual_cpu_tensors_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = {d: dict(tensor_count=0) for d in ('cpu','cuda','other')}
            row = dict(sequence='a',frame_index=0,host_available_bytes=20*1024**3,
                storage=dict(storage_devices=['cpu'],cached_frame_outputs=copy.deepcopy(group),
                    large_tracker_outputs={'pred_masks':copy.deepcopy(group),'maskmem_features':copy.deepcopy(group)}))
            path = root/'storage.jsonl'
            summary = dict(storage_policy='cpu-tracker-and-forward-output-cache-v1',counts={'a':1})
            def save():
                path.write_text(json.dumps(row)+'\n')
                summary['storage_records_sha256']=f.sha(path)
            save(); f.verify_storage_trace(root,summary)
            row['storage']['large_tracker_outputs']['pred_masks']['cuda']['tensor_count']=1
            save()
            with self.assertRaisesRegex(ValueError,'not CPU-offloaded'): f.verify_storage_trace(root,summary)
            row['storage']['large_tracker_outputs']['pred_masks']['cuda']['tensor_count']=0
            save(); summary['counts']['a']=101
            with self.assertRaisesRegex(ValueError,'coverage'): f.verify_storage_trace(root,summary)
            summary['counts']['a']=1
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'telemetry changed'): f.verify_storage_trace(root,summary)

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

    def _nake_fixture(self, tmp):
        root = Path(tmp) / "inputs"
        for sub in ("rgb", "right"):
            (root / "ego_recording" / sub).mkdir(parents=True)
        refs = []
        for i in range(7):
            ref = np.zeros((16, 16), bool)
            if i == 3:
                ref[3:8, 3:8] = True
            refs.append(ref)
            Image.new("RGB", (16, 16), color=(i, i, i)).save(root / "ego_recording/rgb" / f"{i:06d}.png")
            if i not in (0, 4):
                Image.fromarray(ref.astype(np.uint8) * 255).save(root / "ego_recording/right" / f"{i:06d}.png")
        seq = dict(name="ego_recording", frame_count=7, height=16, width=16,
            sampled_indices=[3, 6], render_indices=[0, 3, 4, 6],
            reference_excluded_indices=[0, 4], prompts={},
            file_sha256={str(p.relative_to(root)): f.sha(p) for p in root.rglob("*.png")})
        plan = dict(contract=f.NAKE_CONTRACT, sequences=[seq], scoring_frames=2)
        f.write_json(root / "plan.json", plan)
        return root, plan, refs

    def _nake_run(self, tmp, root, refs, mode):
        run = Path(tmp) / mode
        run.mkdir()
        with (run / "ego_recording.jsonl").open("w") as stream:
            for i, ref in enumerate(refs):
                # Excluded context deliberately has a nonempty prediction. It must
                # not silently be scored against a fabricated empty reference.
                prediction = np.ones_like(ref) if i in (0, 4) else ref
                for stage in ("frame", "video"):
                    row = dict(sequence="ego_recording", frame_index=i, method=f"{stage}_{mode}",
                        geometry=[], geometry_available=False,
                        **f.instance_record(prediction[None], [.9], [1], ref.shape))
                    stream.write(json.dumps(row) + "\n")
        summary = dict(status="complete", mode=mode, engineering=False, counts={"ego_recording": 7},
            contract=f.NAKE_CONTRACT, plan_sha256=f.sha(root / "plan.json"), runner_sha256="runner",
            model_source_sha256={}, base_sha256="base", torch_version="test",
            records_sha256={"ego_recording": f.sha(run / "ego_recording.jsonl")})
        f.write_json(run / "summary.json", summary)
        return run

    def test_nake_exclusion_preserves_rgb_and_original_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan, _ = self._nake_fixture(tmp)
            f.check_inputs(root, plan)
            self.assertEqual(len(list((root / "ego_recording/rgb").glob("*.png"))), 7)
            self.assertFalse((root / "ego_recording/right/000000.png").exists())
            self.assertEqual(plan["sequences"][0]["sampled_indices"], [3, 6])
            bad = copy.deepcopy(plan)
            bad["sequences"][0]["sampled_indices"] = [1, 5]
            with self.assertRaisesRegex(ValueError, "Sampling anchor"):
                f.check_inputs(root, bad)
            missing_rgb = copy.deepcopy(plan)
            del missing_rgb["sequences"][0]["file_sha256"]["ego_recording/rgb/000000.png"]
            with self.assertRaisesRegex(ValueError, "contiguous RGB"):
                f.check_inputs(root, missing_rgb)

    def test_nake_requires_explicit_exclusions_and_known_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan, _ = self._nake_fixture(tmp)
            for excluded in (None, [0, 0], [0, 7], [True]):
                bad = copy.deepcopy(plan)
                bad["sequences"][0]["reference_excluded_indices"] = excluded
                with self.assertRaisesRegex(ValueError, "explicit, unique"):
                    f.check_inputs(root, bad)
            missing = copy.deepcopy(plan)
            del missing["sequences"][0]["file_sha256"]["ego_recording/right/000001.png"]
            with self.assertRaisesRegex(ValueError, "Missing unexcluded"):
                f.check_inputs(root, missing)
            Image.new("L", (16, 16)).save(root / "ego_recording/right/000000.png")
            with self.assertRaisesRegex(ValueError, "must not be synthesized"):
                f.check_inputs(root, plan)

    def test_nake_contract_tampering_rejected_and_legacy_unchanged(self):
        self.assertEqual(f.CONTRACT["format"], "sam3-mano-box-factorial-v1")
        self.assertNotIn("dataset", f.CONTRACT)
        self.assertNotIn("reference_exclusion_policy", f.CONTRACT)
        self.assertEqual(f.CONTRACT["box_source"], "MANO_wilor/right_hand/result_mano_1.npz:mesh")
        self.assertEqual(f.NAKE_CONTRACT["box_source"], "declared-unique-active-instance-mesh")
        for key in f.CONTRACT:
            if key not in ("format", "box_source"):
                self.assertEqual(f.NAKE_CONTRACT[key], f.CONTRACT[key])
        with tempfile.TemporaryDirectory() as tmp:
            root, plan, _ = self._nake_fixture(tmp)
            for key, value in (("scoring_stride", 2), ("detector_threshold", .4),
                               ("box_padding", .1), ("prompt", "hand")):
                bad = copy.deepcopy(plan)
                bad["contract"][key] = value
                with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
                    f.check_inputs(root, bad)

    def test_nake_engineering_uses_first_sequence_not_milk(self):
        seqs = [dict(name="ego_one"), dict(name="exo_two")]
        self.assertEqual(f.select_sequences(dict(contract=f.NAKE_CONTRACT, sequences=seqs), True), seqs[:1])
        self.assertEqual(f.select_sequences(dict(contract=f.NAKE_CONTRACT, sequences=seqs)), seqs)
        legacy = [dict(name="basket"), dict(name="milk")]
        self.assertEqual(f.select_sequences(dict(contract=f.CONTRACT, sequences=legacy), True), legacy[1:])

    def test_nake_audit_excluded_context_not_negative_and_compare_skips_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, plan, refs = self._nake_fixture(tmp)
            args = dict(root=root, output=Path(tmp) / "comparison")
            for mode in ("text", "box"):
                run = self._nake_run(tmp, root, refs, mode)
                audit = Path(tmp) / (mode + "-audit")
                f.audit(argparse.Namespace(root=root, run=run, output=audit))
                result = json.loads((audit / "metrics.json").read_text())
                self.assertEqual(result["all_frames_verified"], 14)
                self.assertEqual(result["reference_excluded_records"], 4)
                scored = [json.loads(line) for line in (audit / "scores.jsonl").read_text().splitlines()]
                self.assertEqual({r["frame_index"] for r in scored}, {1, 2, 3, 5, 6})
                excluded = [json.loads(line) for line in (audit / "reference-excluded.jsonl").read_text().splitlines()]
                self.assertTrue(all(r["reference_excluded"] and "dice" not in r for r in excluded))
                self.assertTrue(all(r["prediction_pixels"] == 256 for r in excluded))
                for metrics in result["methods"].values():
                    self.assertEqual(metrics["frames"], 2)
                    self.assertEqual(metrics["positive_frames"], 1)
                    self.assertEqual(metrics["empty_frames"], 1)
                    self.assertEqual(metrics["mean_dice"], 1.)
                    self.assertEqual(metrics["empty_false_positive"], 0)
                args[mode + "_run"] = run
                args[mode + "_audit"] = audit
            f.compare(argparse.Namespace(**args))
            result = json.loads((args["output"] / "metrics.json").read_text())
            self.assertEqual(result["contract"], f.NAKE_CONTRACT)
            self.assertEqual(result["raw_frames_per_condition"], 7)
            self.assertEqual(result["scoring_frames_per_condition"], 2)
            self.assertEqual(sorted(p.name for p in (args["output"] / "visualizations").iterdir()),
                             ["ego_recording-000003", "ego_recording-000006"])

    def test_nake_excluded_context_still_checks_prediction_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, refs = self._nake_fixture(tmp)
            run = self._nake_run(tmp, root, refs, "text")
            path = run / "ego_recording.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["prediction_pixels"] = 0
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary_path = run / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["records_sha256"]["ego_recording"] = f.sha(path)
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "saved area mismatch"):
                f.audit(argparse.Namespace(root=root, run=run, output=Path(tmp) / "audit"))


if __name__ == "__main__":
    unittest.main()

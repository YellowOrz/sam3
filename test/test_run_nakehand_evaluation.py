from pathlib import Path
import json
import tempfile
import unittest

import torch

from scripts.run_nakehand_evaluation import ExternalTest, core_source_hashes, parse_args, readable_summary, sha256, verify_epoch2, verify_published_data


class NakehandRunnerTest(unittest.TestCase):
    def setUp(self):
        self.state = {
            "format": "sam3-learnable-class-tokens-v2", "class_names": ["left_hand", "right_hand"],
            "class_tokens": torch.ones(2, 4, 256), "next_step": 6,
            "training_config": {"epochs": 2, "batch_size": 2},
            "annotation_summary": {"sha256": "a"},
        }
        self.summary = {"dataset_size": 5, "epochs": 2, "batch_size": 2,
                        "steps": 6, "samples_seen": 10, "annotation_summary": {"sha256": "a"}}

    def test_odd_batch_complete_old_checkpoint(self):
        self.assertEqual(verify_epoch2(self.state, self.summary), 6)

    def test_partial_or_mismatched_checkpoint_rejected(self):
        for key, value in (("next_step", 5), ("class_names", ["right_hand", "left_hand"]),
                           ("class_tokens", torch.ones(2, 1, 256)),
                           ("annotation_summary", {"sha256": "b"})):
            state = dict(self.state, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_epoch2(state, self.summary)

    def test_insufficient_actual_sample_count_rejected(self):
        with self.assertRaises(ValueError):
            verify_epoch2(self.state, dict(self.summary, samples_seen=9))

    def test_half_written_formal_summary_is_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            self.assertFalse(readable_summary(path))
            path.write_text('{"steps":')
            self.assertFalse(readable_summary(path))
            path.write_text('{"steps":6}')
            self.assertTrue(readable_summary(path))

    def test_no_budget_expansion(self):
        base = ["--data-root", "/tmp/data", "--base-checkpoint", "/tmp/base.pt",
                "--token-checkpoint", "/tmp/token.pt", "--formal-summary", "/tmp/formal.json",
                "--output-dir", "/tmp/new-external-result"]
        args = parse_args(base)
        self.assertEqual(args.child_timeout_seconds, 3600)
        with self.assertRaises(SystemExit):
            parse_args(base + ["--deadline", "2026-09-10T21:41:00+08:00"])
        with self.assertRaises(SystemExit):
            parse_args(base + ["--minimum-free-mib", "1000"])

    def test_source_change_invalidates_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root / "project"
            (project / "sam3").mkdir(parents=True)
            source = project / "sam3/core.py"
            source.write_text("# original\n")
            args = parse_args(["--data-root", "/tmp/data", "--base-checkpoint", "/tmp/base.pt",
                               "--token-checkpoint", "/tmp/token.pt", "--formal-summary", "/tmp/formal.json",
                               "--project-root", str(project), "--output-dir", str(root / "result")])
            args.output_dir.mkdir()
            runner = ExternalTest(args)
            runner.state.update(core_python_sources=core_source_hashes(project), comparison_valid=True)
            runner.check_core()
            source.write_text("# externally changed\n")
            with self.assertRaises(RuntimeError):
                runner.check_core()
            self.assertFalse(runner.state["comparison_valid"])

    def test_publication_requires_final_marker_and_matching_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "annotations.json").write_text("{}")
            digest = sha256(root / "annotations.json")
            publication = {"status": "complete", "annotations_sha256": digest}
            (root / "manifest.json").write_text(json.dumps(dict(publication, sources_unchanged=True)))
            (root / "export-in-progress.json").write_text(json.dumps(publication))
            with self.assertRaises(FileNotFoundError):
                verify_published_data(root)
            (root / "READY.json").write_text(json.dumps(dict(publication, manifest_sha256=sha256(root / "manifest.json"))))
            self.assertEqual(verify_published_data(root), digest)
            (root / "annotations.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_published_data(root)


if __name__ == "__main__":
    unittest.main()

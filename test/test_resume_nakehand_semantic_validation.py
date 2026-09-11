"""CPU-only package-entry regression and validation-only recovery safeguards."""
from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts import resume_nakehand_semantic_validation as recovery
from test_evaluate_ve_initialized_tokens import state as fixture_state


REPO = Path(__file__).resolve().parents[1]


class ResumeSemanticValidationTest(unittest.TestCase):
    def test_real_package_imports_reconstruct_install_and_restore_cpu_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "fixture.pt"
            torch.save(fixture_state(), checkpoint)
            result = recovery.package_identity_probe(checkpoint)
            self.assertEqual(result["entry_package"], "scripts")
            self.assertEqual(result["cached_module"], "scripts.cached_ve_text_features")
            self.assertTrue(result["strict_install_and_restore_passed"])
            self.assertFalse(result["cuda_initialized"])

    def test_direct_file_entry_reproduces_original_class_identity_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "fixture.pt"
            torch.save(fixture_state(), checkpoint)
            # Isolate the intentionally wrong top-level imports from this test
            # process, so subsequent tests retain coherent package identities.
            code = """
import runpy, sys, torch
from pathlib import Path
repo, cp = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / 'scripts'))
ns = runpy.run_path(str(repo / 'scripts/evaluate_nakehand_semantic_tokens.py'), run_name='__cpu_direct_file_probe__')
assert ns['cached'] is not ns['semantic'].cached
encoder = ns['semantic'].cache_from_state(torch.load(cp, map_location='cpu', weights_only=True)['initial_cache_state_dict'], frozen=True)
try:
    ns['cached'].install_cached_ve_text_encoder(None, encoder)
except TypeError as error:
    assert 'explicitly constructed CachedVETextEncoder' in str(error)
else:
    raise AssertionError('Expected original strict-class failure')
assert not torch.cuda.is_initialized()
print('original-direct-entry-failure-reproduced-on-CPU')
"""
            result = subprocess.run([sys.executable, "-c", code, str(REPO), str(checkpoint)],
                                    cwd=REPO, env=recovery.environment(REPO, 0),
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("failure-reproduced-on-CPU", result.stdout)

    def test_recovery_commands_never_train_subset_or_disable_guards(self):
        args = Namespace(python=sys.executable, project_root=REPO, output_dir=Path("/tmp/new-recovery"))
        evidence = {"data_root": "/tmp/data", "base_checkpoint": "/tmp/base.pt",
                    "checkpoints": {"unconstrained": {"checkpoint": "/tmp/un.pt"},
                                    "anchored": {"checkpoint": "/tmp/an.pt"}}}
        evaluation, reporting = recovery.commands(args, evidence)
        self.assertEqual(evaluation[:4], [sys.executable, "-u", "-m", recovery.EVALUATOR])
        self.assertIn("0.25", evaluation)
        self.assertEqual(recovery.required_argument(evaluation, "--variant"), "all")
        self.assertEqual(recovery.required_argument(evaluation, "--minimum-samples-seen"), "2000")
        self.assertNotIn("--indices", evaluation)
        self.assertFalse(any("train_" in arg for arg in evaluation + reporting))
        self.assertEqual(reporting[3], recovery.REPORTER)
        self.assertEqual(recovery.environment(REPO, 0)["PYTHONPATH"], str(REPO))

    def test_modified_frozen_source_and_changed_core_inventory_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sam3").mkdir()
            source = root / "sam3/model.py"
            source.write_text("original")
            rows = [{"path": str(source), "kind": "core", "sha256": recovery.file_hash(source)}]
            recovery.verify_sources(rows, root)
            source.write_text("changed")
            with self.assertRaises(RuntimeError):
                recovery.verify_sources(rows, root)
            source.write_text("original")
            (root / "sam3/new.py").write_text("added")
            with self.assertRaises(ValueError):
                recovery.verify_sources(rows, root)

    def test_singleton_prevents_duplicate_recovery_and_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "old-run"
            with recovery.singleton(old):
                with self.assertRaises(BlockingIOError):
                    with recovery.singleton(old):
                        self.fail("Second validation acquired the same singleton lock")
            with recovery.singleton(old):
                pass

    def test_ambiguous_historical_command_is_not_executed(self):
        with self.assertRaises(ValueError):
            recovery.required_argument(["--variant", "all", "--variant", "other"], "--variant")
        with self.assertRaises(ValueError):
            recovery.required_argument(["--variant"], "--variant")


if __name__ == "__main__":
    unittest.main()

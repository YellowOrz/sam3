"""CPU-only shell checks; a tiny interpreter stub prevents GPU/process launches."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/run_residual_ddp.sh"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stub = self.root / "python stub"
        self.stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'), "
            "'nccl': os.environ.get('NCCL_SOCKET_IFNAME'), 'cwd': os.getcwd(), "
            "'cuda_order': os.environ.get('CUDA_DEVICE_ORDER'), 'pythonpath': os.environ.get('PYTHONPATH')}))\n")
        self.stub.chmod(0o755)
        self.environment = {**os.environ, "SAM3_PYTHON": str(self.stub),
                            "CUDA_VISIBLE_DEVICES": "old-value", "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
                            "PYTHONPATH": "/existing path:relative-package-root",
                            "NCCL_SOCKET_IFNAME": "unchanged-interface"}

    def run_launcher(self, *arguments, environment=None):
        return subprocess.run(["bash", str(LAUNCHER), *arguments], cwd=self.root,
                              env=self.environment if environment is None else environment,
                              capture_output=True, text=True, timeout=10)

    def test_shell_syntax_and_help_need_no_training_arguments(self):
        syntax = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = self.run_launcher("--help")
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("SAM3_PYTHON", help_result.stdout)

    def test_three_and_four_gpus_forward_arguments_without_shell_evaluation(self):
        literal = "run with spaces; $(touch MUST_NOT_EXIST) `touch ALSO_NOT_EXIST`"
        for gpus, count in (("0,2,3", 3), ("0,1,2,3", 4)):
            with self.subTest(gpus=gpus):
                result = self.run_launcher("--gpus", gpus, "--", "--output-dir", literal, "--epochs", "2")
                self.assertEqual(result.returncode, 0, result.stderr)
                received = json.loads(result.stdout)
                self.assertEqual(received["gpus"], gpus)
                self.assertEqual(received["nccl"], "unchanged-interface")
                self.assertEqual(received["cuda_order"], "PCI_BUS_ID")
                self.assertEqual(received["cwd"], str(self.root))
                self.assertEqual(received["pythonpath"],
                                 f"{LAUNCHER.parent.parent}:/existing path:relative-package-root")
                self.assertEqual(received["argv"][:2], ["-m", "torch.distributed.run"])
                self.assertIn(f"--nproc_per_node={count}", received["argv"])
                self.assertIn("--max_restarts=0", received["argv"])
                self.assertIn("--standalone", received["argv"])
                self.assertIn("--nnodes=1", received["argv"])
                self.assertEqual(received["argv"][-4:], ["--output-dir", literal, "--epochs", "2"])
                self.assertEqual(received["argv"][-6:-4], ["--module", "scripts.train_residual_ddp"])
                self.assertFalse((self.root / "MUST_NOT_EXIST").exists())
                self.assertFalse((self.root / "ALSO_NOT_EXIST").exists())

    def test_explicit_python_overrides_environment_and_dry_run_does_not_execute(self):
        environment = {**self.environment, "SAM3_PYTHON": "/not/a/python"}
        result = self.run_launcher("--gpus", "0,1,2", "--python", str(self.stub), "--dry-run",
                                   "--", "--output-dir", "new run", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        self.assertEqual(command[:3], ["env", "CUDA_VISIBLE_DEVICES=0,1,2", "CUDA_DEVICE_ORDER=PCI_BUS_ID"])
        self.assertEqual(command[3], f"PYTHONPATH={LAUNCHER.parent.parent}:/existing path:relative-package-root")
        self.assertEqual(command[4], str(self.stub))
        self.assertEqual(command[-2:], ["--output-dir", "new run"])
        self.assertFalse((self.root / "new run").exists())

    def test_real_module_help_imports_from_outside_repository_without_gpu(self):
        environment = dict(self.environment)
        environment.pop("PYTHONPATH")
        result = self.run_launcher("--gpus", "0,1,2", "--python", sys.executable,
                                   "--dry-run", "--", "--help", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = shlex.split(result.stdout)
        injected = dict(item.split("=", 1) for item in command[1:4])
        self.assertEqual(injected["PYTHONPATH"], str(LAUNCHER.parent.parent))
        self.assertEqual(command[-3:], ["--module", "scripts.train_residual_ddp", "--help"])
        # Exercise the actual trainer module without launching torchrun workers
        # or its rendezvous server. This uses precisely the launcher's import path.
        child_environment = {**environment, **injected, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
        help_result = subprocess.run(
            [sys.executable, "-m", command[-2], "--help"], cwd=self.root,
            env=child_environment, capture_output=True, text=True, timeout=45)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--approval", help_result.stdout)
        self.assertIn("--batch-size-per-rank", help_result.stdout)
        self.assertIn("--tensorboard", help_result.stdout)
        self.assertEqual(list(self.root.iterdir()), [self.stub])

    def test_invalid_lists_missing_values_and_unseparated_trainer_options_fail(self):
        invalid = [[], ["--gpus"], ["--python"], ["--gpus", "0,1,2"],
                   ["--gpus", "0,1,2", "--"], ["--gpus", "0,1,2", "--epochs", "2"],
                   ["--gpus", "0,1,2", "--gpus", "1,2,3", "--", "--help"]]
        invalid += [["--gpus", value, "--", "--help"] for value in (
            "", "0", "0,1", "0,1,2,3,4", "0,1,1", "0,0,1,2", "0,1,-2",
            "0,1,02", "0, 1,2", "0,1,2,", "0,1,$(true)")]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                result = self.run_launcher(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("run_residual_ddp.sh:", result.stderr)


if __name__ == "__main__":
    unittest.main()

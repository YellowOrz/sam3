import unittest
import torch

from scripts.check_soft_ve_prompt import check_gradient, parse_args


class CheckSoftVEProbeTest(unittest.TestCase):
    def test_gradient_probe_rejects_disconnected_nonfinite_and_zero_role(self):
        self.assertEqual(check_gradient(torch.ones(2, 4), 2, "roles"), [2., 2.])
        for value in (None, torch.ones(1, 4), torch.full((2, 4), float("nan")),
                      torch.tensor([[1., 1.], [0., 0.]])):
            with self.assertRaises(RuntimeError):
                check_gradient(value, 2, "roles")

    def test_source_output_overlap_rejected_and_explicit_paths_kept(self):
        base = ["--data-root", "/tmp/soft-probe-test-data/train", "--base-checkpoint", "/tmp/base.pt"]
        with self.assertRaises(SystemExit):
            parse_args(base + ["--output-dir", "/tmp/soft-probe-test-data/train/probe"])
        args = parse_args(base + ["--output-dir", "/tmp/soft-probe-output-not-created"])
        self.assertEqual(str(args.data_root), "/tmp/soft-probe-test-data/train")

    def test_memory_fraction_default_and_authorized_upper_bound(self):
        base = ["--data-root", "/tmp/soft-probe-test-data/train", "--base-checkpoint", "/tmp/base.pt",
                "--output-dir", "/tmp/soft-probe-output-not-created"]
        self.assertEqual(parse_args(base).gpu_memory_fraction, .25)
        self.assertEqual(parse_args(base + ["--gpu-memory-fraction", ".35"]).gpu_memory_fraction, .35)

    def test_memory_fraction_rejects_nonfinite_and_outside_budget(self):
        base = ["--data-root", "/tmp/soft-probe-test-data/train", "--base-checkpoint", "/tmp/base.pt",
                "--output-dir", "/tmp/soft-probe-output-not-created"]
        for value in ("nan", "inf", "-inf", "0", "-0.1", "0.35000001", "1"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parse_args(base + [f"--gpu-memory-fraction={value}"])


if __name__ == "__main__":
    unittest.main()

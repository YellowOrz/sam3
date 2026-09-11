"""CPU-only distributed contracts, including real 3/4-rank DDP backward."""
from copy import deepcopy
import io
import multiprocessing
import os
from pathlib import Path
import queue
import random
import tempfile
import time
import traceback
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from scripts import residual_ddp_runtime as runtime


def _random_values():
    return random.random(), np.random.random(4).tolist(), torch.rand(4).tolist()


def _ddp_worker(rank, world_size, rendezvous, result_queue):
    """Keep process-group initialization and all collectives inside the child."""
    context = None
    try:
        torch.set_num_threads(1)
        environ_before = dict(os.environ)
        ranks = {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size)}
        context = runtime.initialize_distributed(
            "cpu", environ=ranks, init_method=rendezvous, timeout_seconds=25
        )
        assert dict(os.environ) == environ_before
        assert context.rank == rank and context.local_rank == rank
        assert context.world_size == world_size and context.backend == "gloo"
        assert context.owns_process_group
        reused = runtime.initialize_distributed("cpu", environ=ranks)
        assert not reused.owns_process_group
        runtime.cleanup_distributed(reused)
        assert torch.distributed.is_initialized()

        torch.manual_seed(734)
        model = nn.Linear(3, 2, dtype=torch.float64)
        # This frozen parameter also checks the wrapper does not switch modes or
        # invent gradients for a frozen backbone.
        model.register_parameter("frozen", nn.Parameter(torch.ones(2), requires_grad=False))
        model.eval()
        reference = deepcopy(model)
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        assert not model.training
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.007, weight_decay=0)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.007, weight_decay=0)
        batch_size = 2
        dataset_size = world_size * batch_size * 3 + 1
        sampler = runtime.DeterministicGlobalBatchSampler(
            dataset_size, batch_size, rank=rank, world_size=world_size, seed=29, epoch=2
        )
        x = torch.arange(dataset_size * 3, dtype=torch.float64).view(dataset_size, 3) / 41
        y = torch.stack((x[:, 0].sin(), x[:, 2].cos()), dim=1)
        all_indices = sampler.global_indices
        observed = []
        max_gradient_error = 0.0
        for step, local_indices in enumerate(sampler):
            observed.extend(local_indices)
            global_indices = all_indices[
                step * sampler.global_batch_size : (step + 1) * sampler.global_batch_size
            ]
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            ((wrapped(x[local_indices]) - y[local_indices]) ** 2).mean().backward()
            ((reference(x[global_indices]) - y[global_indices]) ** 2).mean().backward()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                if not actual.requires_grad:
                    assert actual.grad is None and expected.grad is None
                    continue
                difference = (actual.grad - expected.grad).abs().max().item()
                max_gradient_error = max(max_gradient_error, difference)
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
            optimizer.step()
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

        # Each rank contributes its own Python/NumPy/Torch streams. The aggregate
        # can be safely serialized by rank zero and restores each rank exactly.
        random.seed(1000 + rank)
        np.random.seed(2000 + rank)
        torch.manual_seed(3000 + rank)
        gathered = runtime.gather_rng_states("cpu")
        expected_draws = _random_values()
        buffer = io.BytesIO()
        torch.save({"rng_by_rank": gathered}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        assert len(saved["rng_by_rank"]) == world_size
        assert all(state["torch_cuda"] is None for state in saved["rng_by_rank"])
        assert len({bytes(state["torch_cpu"].tolist()) for state in gathered}) == world_size
        runtime.restore_rng_state(saved["rng_by_rank"][rank], "cpu")
        assert _random_values() == expected_draws

        # Every rank must observe the same exception, including successful peers.
        runtime.raise_if_distributed_error(None, context)
        try:
            runtime.raise_if_distributed_error(ValueError("sentinel") if rank == 1 else None, context)
        except RuntimeError as exc:
            gate_error = str(exc)
            assert "rank 1: ValueError: sentinel" in gate_error
        else:
            raise AssertionError("rank-consistent error gate did not raise")
        runtime.raise_if_distributed_error(None, context)
        result_queue.put({
            "rank": rank,
            "indices": observed,
            "global_indices": all_indices,
            "dropped_count": sampler.dropped_count,
            "steps": len(sampler),
            "max_gradient_error": max_gradient_error,
            "gate_error": gate_error,
        })
    except BaseException:
        result_queue.put({"rank": rank, "error": traceback.format_exc()})
        raise
    finally:
        if context is not None:
            runtime.cleanup_distributed(context)


class DistributedRuntimeTests(unittest.TestCase):
    def test_rank_environment_defaults_validation_and_no_mutation(self):
        self.assertEqual(runtime.torchrun_ranks({}), (0, 0, 1))
        environ = {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "4", "OTHER": "keep"}
        before = dict(environ)
        self.assertEqual(runtime.torchrun_ranks(environ), (2, 1, 4))
        self.assertEqual(environ, before)
        for bad in ({"RANK": "0"}, {**environ, "RANK": "4"}, {**environ, "LOCAL_RANK": "-1"},
                    {**environ, "WORLD_SIZE": "0"}, {**environ, "RANK": "bad"}):
            with self.subTest(environ=bad), self.assertRaises(ValueError):
                runtime.torchrun_ranks(bad)

    def test_single_process_cpu_lifecycle_does_not_initialize_cuda(self):
        environ_before = dict(os.environ)
        with mock.patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA touched")):
            context = runtime.initialize_distributed("cpu", environ={})
            self.assertEqual(context.device, torch.device("cpu"))
            self.assertTrue(context.is_main)
            self.assertFalse(context.distributed)
            self.assertFalse(context.owns_process_group)
            runtime.cleanup_distributed(context)
        self.assertEqual(dict(os.environ), environ_before)
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                runtime.initialize_distributed("cpu", environ={}, timeout_seconds=timeout)
        with self.assertRaises(ValueError):
            runtime.initialize_distributed("meta", environ={})

    def test_sampler_global_disjoint_order_drop_and_resume_three_four_ranks(self):
        for world_size in (3, 4):
            with self.subTest(world_size=world_size):
                samplers = [runtime.DeterministicGlobalBatchSampler(
                    53, 4, rank=rank, world_size=world_size, seed=123, epoch=5
                ) for rank in range(world_size)]
                batches = [list(sampler) for sampler in samplers]
                merged = [index for step in range(len(samplers[0]))
                          for rank in range(world_size) for index in batches[rank][step]]
                sampler = samplers[0]
                self.assertEqual(merged, sampler.global_indices)
                self.assertEqual(len(merged), len(set(merged)))
                self.assertEqual(len(merged), sampler.processed_count)
                self.assertEqual(sampler.dropped_count, 53 % (world_size * 4))
                self.assertEqual(set(merged) | set(sampler.dropped_indices), set(range(53)))
                self.assertFalse(set(merged) & set(sampler.dropped_indices))
                self.assertTrue(sampler.drop_last)
                for rank, sampler in enumerate(samplers):
                    original = list(sampler)
                    sampler.set_epoch(5, start_step=1)
                    self.assertEqual(list(sampler), original[1:])
                    self.assertEqual(len(sampler), sampler.total_steps - 1)
                    sampler.set_epoch(5, start_step=sampler.total_steps)
                    self.assertEqual(len(sampler), 0)
                    self.assertEqual(list(sampler), [])
                    sampler.set_epoch(6)
                    self.assertNotEqual(list(sampler), original)
                    sampler.set_epoch(5)
                    self.assertEqual(list(sampler), batches[rank])

    def test_sampler_uses_seed_plus_epoch_without_global_rng_side_effects(self):
        state = runtime.capture_rng_state("cpu")
        expected = _random_values()
        runtime.restore_rng_state(state, "cpu")
        sampler = runtime.DeterministicGlobalBatchSampler(57, 2, world_size=4, seed=99, epoch=3)
        actual_order = sampler.global_indices + sampler.dropped_indices
        generator = torch.Generator().manual_seed(102)
        self.assertEqual(actual_order, torch.randperm(57, generator=generator).tolist())
        self.assertEqual(_random_values(), expected)

    def test_sampler_empty_small_and_invalid_arguments(self):
        for size in (0, 1, 7):
            sampler = runtime.DeterministicGlobalBatchSampler(size, 2, world_size=4)
            self.assertEqual(len(sampler), 0)
            self.assertEqual(sampler.dropped_count, size)
            self.assertEqual(list(sampler), [])
        for kwargs in ({"dataset_size": -1}, {"batch_size_per_rank": 0}, {"rank": 4},
                       {"world_size": 0}, {"start_step": 4}, {"epoch": -1}, {"seed": True}):
            values = {"dataset_size": 24, "batch_size_per_rank": 2, "world_size": 4, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runtime.DeterministicGlobalBatchSampler(**values)

    def test_rng_weights_only_roundtrip_and_single_rank_gather(self):
        state_before = runtime.capture_rng_state("cpu")
        try:
            random.seed(128)
            np.random.seed(512)
            torch.manual_seed(1024)
            # Preserve NumPy's cached normal variate as well as the MT state.
            np.random.normal()
            saved = runtime.gather_rng_states("cpu")
            expected = (_random_values(), np.random.normal())
            buffer = io.BytesIO()
            torch.save(saved, buffer)
            buffer.seek(0)
            restored = torch.load(buffer, weights_only=True)
            self.assertEqual(len(restored), 1)
            runtime.restore_rng_state(restored[0], "cpu")
            self.assertEqual((_random_values(), np.random.normal()), expected)
        finally:
            runtime.restore_rng_state(state_before, "cpu")

    def test_single_rank_error_gate(self):
        runtime.raise_if_distributed_error(None)
        with self.assertRaisesRegex(RuntimeError, "Rank 0 failed: ValueError: example"):
            runtime.raise_if_distributed_error(ValueError("example"))

    def _assert_multi_rank_ddp(self, world_size):
        if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
            self.skipTest("This PyTorch build does not support Gloo")
        spawn = multiprocessing.get_context("spawn")
        result_queue = spawn.Queue()
        processes = []
        with tempfile.TemporaryDirectory(prefix=f"residual-ddp-{world_size}-") as directory:
            # A new file-store path for each test isolates concurrent invocations
            # without reserving/reusing a potentially contested TCP port.
            rendezvous = Path(directory, "rendezvous").as_uri()
            try:
                for rank in range(world_size):
                    process = spawn.Process(target=_ddp_worker, args=(rank, world_size, rendezvous, result_queue))
                    process.start()
                    processes.append(process)
                deadline = time.monotonic() + 55
                results = []
                for _ in range(world_size):
                    try:
                        results.append(result_queue.get(timeout=max(0.01, deadline - time.monotonic())))
                    except queue.Empty:
                        self.fail(f"{world_size}-rank DDP timed out; exit codes: {[p.exitcode for p in processes]}")
                for process in processes:
                    process.join(timeout=max(0.01, deadline - time.monotonic()))
                errors = [result["error"] for result in results if "error" in result]
                self.assertFalse(errors, "\n".join(errors))
                self.assertTrue(all(process.exitcode == 0 for process in processes))
                results.sort(key=lambda result: result["rank"])
                observed = [index for result in results for index in result["indices"]]
                self.assertEqual(len(observed), len(set(observed)))
                self.assertEqual(set(observed), set(results[0]["global_indices"]))
                self.assertEqual({result["steps"] for result in results}, {3})
                self.assertEqual({result["dropped_count"] for result in results}, {1})
                self.assertEqual(len({result["gate_error"] for result in results}), 1)
                self.assertLess(max(result["max_gradient_error"] for result in results), 1e-12)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3)
                result_queue.close()
                result_queue.join_thread()

    def test_three_rank_ddp_matches_single_process_global_batch(self):
        self._assert_multi_rank_ddp(3)

    def test_four_rank_ddp_matches_single_process_global_batch(self):
        self._assert_multi_rank_ddp(4)


if __name__ == "__main__":
    unittest.main()

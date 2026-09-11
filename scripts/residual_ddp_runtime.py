"""Small distributed primitives for frozen-backbone residual training.

The sampler never pads or repeats examples. Every participating rank receives
the same number of full local batches, so averaging rank-local mean losses in
DDP is equivalent to taking the mean over the corresponding global batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
import os
import random
from typing import Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    owns_process_group: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def torchrun_ranks(environ: Mapping[str, str] | None = None) -> tuple[int, int, int]:
    """Read ``(rank, local_rank, world_size)`` without changing the environment."""
    environ = os.environ if environ is None else environ
    names = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    present = [name in environ for name in names]
    if not any(present):
        return 0, 0, 1
    if not all(present):
        raise ValueError("torchrun requires RANK, LOCAL_RANK and WORLD_SIZE together")
    try:
        rank, local_rank, world_size = (int(environ[name]) for name in names)
    except (TypeError, ValueError) as exc:
        raise ValueError("torchrun ranks and world size must be integers") from exc
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun RANK, LOCAL_RANK or WORLD_SIZE")
    return rank, local_rank, world_size


def initialize_distributed(
    device: str | torch.device = "auto",
    *,
    timeout_seconds: float = 120,
    environ: Mapping[str, str] | None = None,
    init_method: str | None = None,
) -> DistributedContext:
    """Select the rank's device and initialize NCCL (CUDA) or Gloo (CPU).

    The default rendezvous is torchrun's ``env://``. ``environ`` overrides only
    rank discovery, never process environment variables; tests can combine it
    with an isolated ``file://`` init_method. An existing compatible group is
    reused, and only a group created here is destroyed by cleanup_distributed.
    """
    rank, local_rank, world_size = torchrun_ranks(environ)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = local_rank if selected.index is None else selected.index
        if world_size > 1 and index != local_rank:
            raise ValueError("A distributed CUDA device must match LOCAL_RANK")
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError("LOCAL_RANK selects a CUDA device that is not visible")
        selected = torch.device("cuda", index)
        torch.cuda.set_device(selected)
        backend = "nccl"
    elif selected.type == "cpu":
        selected = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("Residual DDP supports only CPU/Gloo and CUDA/NCCL")
    owns_group = False
    if dist.is_available() and dist.is_initialized():
        if (dist.get_rank(), dist.get_world_size(), dist.get_backend()) != (
            rank, world_size, backend
        ):
            raise RuntimeError("Existing process group conflicts with requested ranks/backend")
    elif world_size > 1:
        if not dist.is_available():
            raise RuntimeError("This PyTorch build has no distributed support")
        kwargs = {
            "backend": backend,
            "init_method": init_method or "env://",
            "rank": rank,
            "world_size": world_size,
            "timeout": timedelta(seconds=timeout_seconds),
        }
        if selected.type == "cuda":
            kwargs["device_id"] = selected
        dist.init_process_group(**kwargs)
        owns_group = True
    active_backend = backend if dist.is_available() and dist.is_initialized() else None
    return DistributedContext(rank, local_rank, world_size, selected, active_backend, owns_group)


def cleanup_distributed(context: DistributedContext) -> None:
    """Destroy our group without a barrier that could hang during error cleanup."""
    if context.owns_process_group and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class DeterministicGlobalBatchSampler(Sampler[list[int]]):
    """Shuffle once with seed + epoch, then split each full batch by rank.

    ``drop_last`` is always True: the final incomplete *global* batch is omitted
    in full, never padded. ``total_steps`` and ``processed_count`` describe the
    entire epoch; ``len(sampler)`` is the number of remaining local batches after
    ``start_step``. The index order uses an isolated generator and does not alter
    the model's Torch/Python/NumPy random streams.
    """

    drop_last = True

    def __init__(
        self,
        dataset_size: int,
        batch_size_per_rank: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        epoch: int = 0,
        start_step: int = 0,
    ) -> None:
        self.dataset_size = _integer("dataset_size", dataset_size)
        self.batch_size_per_rank = _integer("batch_size_per_rank", batch_size_per_rank, 1)
        self.world_size = _integer("world_size", world_size, 1)
        self.rank = _integer("rank", rank)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        self.seed = _integer("seed", seed)
        self.global_batch_size = self.world_size * self.batch_size_per_rank
        self.total_steps = self.dataset_size // self.global_batch_size
        self.processed_count = self.total_steps * self.global_batch_size
        self.dropped_count = self.dataset_size - self.processed_count
        self.set_epoch(epoch, start_step=start_step)

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        epoch = _integer("epoch", epoch)
        start_step = _integer("start_step", start_step)
        if start_step > self.total_steps:
            raise ValueError("start_step cannot exceed the epoch's total_steps")
        # Do not regenerate when only the resume offset changes.
        if getattr(self, "epoch", None) != epoch:
            generator = torch.Generator().manual_seed((self.seed + epoch) % (2**64))
            self._order = torch.randperm(self.dataset_size, generator=generator).tolist()
        self.epoch = epoch
        self.start_step = start_step

    @property
    def global_indices(self) -> list[int]:
        """All processed indices in global-batch order, including skipped steps."""
        return self._order[: self.processed_count]

    @property
    def dropped_indices(self) -> list[int]:
        return self._order[self.processed_count :]

    def __iter__(self) -> Iterator[list[int]]:
        local_offset = self.rank * self.batch_size_per_rank
        for step in range(self.start_step, self.total_steps):
            start = step * self.global_batch_size + local_offset
            yield self._order[start : start + self.batch_size_per_rank]

    def __len__(self) -> int:
        return self.total_steps - self.start_step


def _rng_cuda_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        if not torch.cuda.is_available():
            return None
        return torch.device("cuda", torch.cuda.current_device())
    selected = torch.device(device)
    if selected.type == "cpu":
        return None
    if selected.type != "cuda":
        raise ValueError("RNG state supports only CPU or CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("Cannot capture/restore CUDA RNG without CUDA")
    if selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    return selected


def capture_rng_state(device: str | torch.device | None = None) -> dict:
    """Capture Python, NumPy, CPU Torch and only this rank's CUDA RNG.

    Values contain only built-in types and tensors, allowing
    ``torch.load(..., weights_only=True)`` without NumPy allowlisting.
    """
    cuda_device = _rng_cuda_device(device)
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": torch.tensor(keys.astype(np.int64), dtype=torch.int64),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None if cuda_device is None else torch.cuda.get_rng_state(cuda_device).cpu().clone(),
    }


def restore_rng_state(state: dict, device: str | torch.device | None = None) -> None:
    """Restore a rank's checkpoint RNG to its current device (no other GPUs)."""
    cuda_device = _rng_cuda_device(device)
    cuda_state = state["torch_cuda"]
    if (cuda_device is None) != (cuda_state is None):
        raise ValueError("Checkpoint CUDA RNG presence does not match the requested device")
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((
        numpy_state["algorithm"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_device is not None:
        torch.cuda.set_rng_state(cuda_state.cpu(), device=cuda_device)


def gather_rng_states(device: str | torch.device | None = None) -> list[dict]:
    """Collect rank-ordered RNG states on every rank for rank-zero checkpointing.

    All ranks must call this function at the same checkpoint boundary. NCCL's
    object collectives use the current CUDA device selected during initialization.
    """
    state = capture_rng_state(device)
    if not dist.is_available() or not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def raise_if_distributed_error(
    error: BaseException | str | None,
    context: DistributedContext | None = None,
) -> None:
    """Raise on every rank when any rank reports an error at a shared boundary.

    Call on *every* rank, passing None for successful ranks, before entering the
    next DDP forward/backward. This cannot recover a collective that has already
    failed or a rank that has exited; process-group timeouts cover those cases.
    """
    message = None if error is None else f"{type(error).__name__}: {error}"
    if not dist.is_available() or not dist.is_initialized():
        if message is not None:
            raise RuntimeError(f"Rank 0 failed: {message}")
        return
    if context is not None and (context.rank, context.world_size) != (
        dist.get_rank(), dist.get_world_size()
    ):
        raise RuntimeError("Error gate context does not match the process group")
    messages = [None] * dist.get_world_size()
    dist.all_gather_object(messages, message)
    failures = [f"rank {rank}: {value}" for rank, value in enumerate(messages) if value is not None]
    if failures:
        raise RuntimeError("Distributed step failed: " + "; ".join(failures))

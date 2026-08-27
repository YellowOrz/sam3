# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Base predictor class shared by SAM3 and SAM3.1 (multiplex) video predictors.

Provides the common handle_request/handle_stream_request API and session management.
Subclasses only need to override methods where their behavior differs.
"""

import copy
import gc
import inspect
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from sam3.logger import get_logger

logger = get_logger(__name__)

# torch.cuda.empty_cache() forces a CUDA synchronization that stalls all
# streams in the process. Calling it on every close_session produces visible
# compute-utilization gaps when many sessions are active concurrently.
# Gate the call on device memory pressure: only fire when usage crosses the
# threshold. The allocator's caching pool already covers the common case
# where freed blocks get reused by the next session — empty_cache is only
# needed to keep memory from growing unbounded.
_CLEAR_CACHE_THRESHOLD = 80


@dataclass(frozen=True)
class _CpuTensorSnapshot:
    tensor: torch.Tensor
    device: torch.device


@dataclass(frozen=True)
class _TensorCacheEntry:
    source: weakref.ReferenceType
    snapshot: _CpuTensorSnapshot


def _tensor_cache_key(tensor: torch.Tensor) -> tuple:
    storage = tensor.untyped_storage()
    try:
        version = tensor._version
    except RuntimeError:
        # Tensors created under torch.inference_mode() intentionally have no
        # version counter. Their contents are compared after copying to CPU.
        version = None
    return (
        id(tensor),
        storage.data_ptr(),
        version,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _clone_state_to_cpu(value, tensor_cache, memo):
    """Clone mutable inference data while reusing unchanged tensor snapshots."""
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        key = _tensor_cache_key(value)
        entry = tensor_cache.get(key)
        has_version_counter = key[2] is not None
        cpu_tensor = None
        if not has_version_counter:
            cpu_tensor = value.detach().to(device="cpu", copy=True)
        can_reuse = entry is not None and entry.source() is value
        if can_reuse and cpu_tensor is not None:
            can_reuse = torch.equal(entry.snapshot.tensor, cpu_tensor)
        if not can_reuse:
            snapshot = _CpuTensorSnapshot(
                (
                    cpu_tensor
                    if cpu_tensor is not None
                    else value.detach().to(device="cpu", copy=True)
                ),
                value.device,
            )
            entry = _TensorCacheEntry(weakref.ref(value), snapshot)
            tensor_cache[key] = entry
        else:
            snapshot = entry.snapshot
        memo[value_id] = snapshot
        return snapshot
    if isinstance(value, np.ndarray):
        result = value.copy()
        memo[value_id] = result
        return result
    if isinstance(value, dict):
        result = type(value)()
        if hasattr(value, "default_factory"):
            result.default_factory = value.default_factory
        memo[value_id] = result
        for key, item in value.items():
            result[_clone_state_to_cpu(key, tensor_cache, memo)] = _clone_state_to_cpu(
                item, tensor_cache, memo
            )
        return result
    if isinstance(value, list):
        result = []
        memo[value_id] = result
        result.extend(_clone_state_to_cpu(item, tensor_cache, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_clone_state_to_cpu(item, tensor_cache, memo) for item in value)
        memo[value_id] = result
        return result
    if isinstance(value, set):
        result = {_clone_state_to_cpu(item, tensor_cache, memo) for item in value}
        memo[value_id] = result
        return result
    result = copy.deepcopy(value)
    memo[value_id] = result
    return result


def _restore_state_from_cpu(value, memo):
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, _CpuTensorSnapshot):
        result = value.tensor.to(device=value.device, copy=True)
        memo[value_id] = result
        return result
    if isinstance(value, np.ndarray):
        result = value.copy()
        memo[value_id] = result
        return result
    if isinstance(value, dict):
        result = type(value)()
        if hasattr(value, "default_factory"):
            result.default_factory = value.default_factory
        memo[value_id] = result
        for key, item in value.items():
            result[_restore_state_from_cpu(key, memo)] = _restore_state_from_cpu(
                item, memo
            )
        return result
    if isinstance(value, list):
        result = []
        memo[value_id] = result
        result.extend(_restore_state_from_cpu(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(_restore_state_from_cpu(item, memo) for item in value)
        memo[value_id] = result
        return result
    if isinstance(value, set):
        result = {_restore_state_from_cpu(item, memo) for item in value}
        memo[value_id] = result
        return result
    result = copy.deepcopy(value)
    memo[value_id] = result
    return result


class Sam3BasePredictor:
    """
    Base class for SAM3 video predictors. Provides:
    - Session management (start, reset, close)
    - Request dispatch (handle_request / handle_stream_request)
    - Common add_prompt / propagate_in_video / remove_object / reset_session / close_session

    Subclasses must set `self.model` and `self._all_inference_states` before use.
    """

    def __init__(self):
        # Subclasses must populate these
        self.model = None
        self._all_inference_states: Dict[str, dict] = {}

    # ── Request dispatch ──────────────────────────────────────────────

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id", None),
                offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                offload_state_to_cpu=request.get("offload_state_to_cpu", False),
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text", None),
                points=request.get("points", None),
                point_labels=request.get("point_labels", None),
                clear_old_points=request.get("clear_old_points", True),
                bounding_boxes=request.get("bounding_boxes", None),
                bounding_box_labels=request.get("bounding_box_labels", None),
                clear_old_boxes=request.get("clear_old_boxes", True),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
                obj_id=request.get("obj_id", None),
                rel_coordinates=request.get("rel_coordinates", True),
            )
        elif request_type == "remove_object":
            return self.remove_object(
                session_id=request["session_id"],
                frame_idx=request.get("frame_index", 0),
                obj_id=request["obj_id"],
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "save_checkpoint":
            return self.save_checkpoint(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
            )
        elif request_type == "restore_checkpoint":
            return self.restore_checkpoint(
                session_id=request["session_id"],
                target_frame_idx=request["frame_index"],
            )
        elif request_type == "clear_checkpoints":
            return self.clear_checkpoints(session_id=request["session_id"])
        elif request_type == "cancel_propagation":
            return self.cancel_propagation(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(
                session_id=request["session_id"],
                run_gc_collect=request.get("run_gc_collect", True),
                clear_cache_threshold=int(
                    request.get("clear_cache_threshold", _CLEAR_CACHE_THRESHOLD)
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_index", None),
                max_frame_num_to_track=request.get("max_frame_num_to_track", None),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    # ── Session management ────────────────────────────────────────────

    def start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        """Start a new inference session on a video directory or path."""
        init_kwargs = dict(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        # SAM 3 supports offloading the tracking state, while the SAM 3.1
        # multiplex tracker does not expose that option. Keep the shared
        # request API compatible with both implementations by forwarding the
        # option only when the selected model accepts it.
        init_state_parameters = inspect.signature(self.model.init_state).parameters
        accepts_extra_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in init_state_parameters.values()
        )
        if "offload_state_to_cpu" in init_state_parameters or accepts_extra_kwargs:
            init_kwargs["offload_state_to_cpu"] = offload_state_to_cpu
        elif offload_state_to_cpu:
            raise ValueError(
                f"{type(self.model).__name__} does not support "
                "offload_state_to_cpu=True"
            )
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(**init_kwargs)

        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
            "checkpoints": {},
            "checkpoint_tensor_cache": {},
        }
        logger.info(f"started new session {session_id}")
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        points=None,
        point_labels=None,
        clear_old_points: bool = True,
        bounding_boxes=None,
        bounding_box_labels=None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: Optional[int] = None,
        rel_coordinates: bool = True,
    ):
        """Add text, box and/or point prompt on a specific video frame."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        # Convert lists to tensors if needed
        if points is not None and not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if point_labels is not None and not isinstance(point_labels, torch.Tensor):
            point_labels = torch.tensor(point_labels, dtype=torch.int32)
        if bounding_boxes is not None and not isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        if bounding_box_labels is not None and not isinstance(
            bounding_box_labels, torch.Tensor
        ):
            bounding_box_labels = torch.tensor(bounding_box_labels, dtype=torch.int32)

        kwargs = dict(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            points=points,
            point_labels=point_labels,
            clear_old_points=clear_old_points,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
            clear_old_boxes=clear_old_boxes,
            output_prob_thresh=output_prob_thresh,
            rel_coordinates=rel_coordinates,
        )
        if obj_id is not None:
            kwargs["obj_id"] = obj_id

        # Filter kwargs to only pass what the model accepts
        # (SAM3 has a simpler add_prompt than SAM3.1)
        import inspect

        sig = inspect.signature(self.model.add_prompt)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            frame_idx, outputs = self.model.add_prompt(**filtered_kwargs)
        return {"frame_index": frame_idx, "outputs": outputs}

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ):
        """Remove an object from tracking."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        result = self.model.remove_object(
            inference_state, obj_id, frame_idx=frame_idx, is_user_action=is_user_action
        )
        # Handle both return conventions
        if result is None or (isinstance(result, tuple) and result[1] is None):
            import numpy as np

            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_binary_masks = torch.zeros(
                0,
                inference_state["orig_height"],
                inference_state["orig_width"],
                dtype=torch.bool,
            )
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
            outputs = {
                "out_obj_ids": out_obj_ids.cpu().numpy(),
                "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
                "out_binary_masks": out_binary_masks.cpu().numpy(),
            }
        elif isinstance(result, tuple):
            _, outputs = result
        else:
            outputs = result
        return {"frame_index": frame_idx, "outputs": outputs}

    def cancel_propagation(self, session_id):
        """Cancel any ongoing propagation. No-op if not supported by the model."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)
        if hasattr(self.model, "cancel_propagation"):
            self.model.cancel_propagation(inference_state)
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id,
        propagation_direction="both",
        start_frame_idx=None,
        max_frame_num_to_track=None,
        output_prob_thresh=0.5,
        **kwargs,
    ):
        """Propagate the added prompts to get results on all video frames."""
        try:
            session = self._get_session(session_id)
            inference_state = session["state"]
            self._extend_expiration_time(session)
            if propagation_direction not in ["both", "forward", "backward"]:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )

            propagate_kwargs = dict(
                inference_state=inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            )
            # Only pass output_prob_thresh / extra kwargs if the model supports them
            import inspect

            sig = inspect.signature(self.model.propagate_in_video)
            if "output_prob_thresh" in sig.parameters:
                propagate_kwargs["output_prob_thresh"] = output_prob_thresh
            for k, v in kwargs.items():
                if k in sig.parameters:
                    propagate_kwargs[k] = v

            # Forward propagation
            if propagation_direction in ["both", "forward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=False,
                ):
                    yield {"frame_index": frame_idx, "outputs": outputs}
            # Backward propagation
            if propagation_direction in ["both", "backward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=True,
                ):
                    yield {"frame_index": frame_idx, "outputs": outputs}
        finally:
            logger.info(f"propagation ended in session {session_id}")

    def reset_session(self, session_id):
        """Reset the session to its initial state."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)
        self.model.reset_state(inference_state)
        return {"is_success": True}

    def _checkpoint_frame(self, state, requested_frame_idx):
        processed = [
            index
            for index, output in enumerate(state.get("previous_stages_out", ()))
            if output is not None
        ]
        return max([int(requested_frame_idx), *processed])

    def _checkpoint_mutable_state(self, state, tensor_cache):
        # These entries are immutable or cheaply reproducible. In particular,
        # input_batch owns every decoded video frame and must never be copied per
        # checkpoint. feature_cache only keeps the current backbone frame.
        excluded = {"input_batch", "constants", "feature_cache"}
        memo = {}
        return {
            key: _clone_state_to_cpu(value, tensor_cache, memo)
            for key, value in state.items()
            if key not in excluded
        }

    def save_checkpoint(self, session_id, frame_idx):
        """Save a CPU-only snapshot of a session's mutable inference state."""
        session = self._get_session(session_id)
        self._extend_expiration_time(session)
        state = session["state"]
        checkpoint_frame = self._checkpoint_frame(state, frame_idx)
        checkpoints = session["checkpoints"]
        tensor_cache = session["checkpoint_tensor_cache"]
        dead_keys = [
            key for key, entry in tensor_cache.items() if entry.source() is None
        ]
        for key in dead_keys:
            tensor_cache.pop(key, None)
        created = checkpoint_frame not in checkpoints
        if created:
            checkpoints[checkpoint_frame] = self._checkpoint_mutable_state(
                state, tensor_cache
            )
        return {
            "is_success": True,
            "frame_index": checkpoint_frame,
            "created": created,
        }

    def restore_checkpoint(self, session_id, target_frame_idx):
        """Restore the newest checkpoint at or before ``target_frame_idx``."""
        session = self._get_session(session_id)
        self._extend_expiration_time(session)
        checkpoints = session["checkpoints"]
        candidates = [frame for frame in checkpoints if frame <= int(target_frame_idx)]
        if not candidates:
            return {"is_success": False, "frame_index": None}
        checkpoint_frame = max(candidates)
        state = session["state"]
        immutable = {
            key: state[key] for key in ("input_batch", "constants") if key in state
        }
        state.clear()
        gc.collect()
        restored = _restore_state_from_cpu(checkpoints[checkpoint_frame], {})
        state.update(immutable)
        state.update(restored)
        # Backbone features are valid for only the frame that produced them and
        # are intentionally recomputed after a restore.
        state["feature_cache"] = {}
        for tracker_state in state.get("tracker_inference_states", ()):
            if isinstance(tracker_state, dict) and "cached_features" in tracker_state:
                tracker_state["cached_features"] = state["feature_cache"]
        # A restored generator cannot be resumed. Removing the trailing
        # propagation marker makes the next bounded forward call recompute the
        # requested segment instead of taking the propagation_fetch path.
        history = state.get("action_history", [])
        while history and str(history[-1].get("type", "")).startswith("propagation"):
            history.pop()
        return {
            "is_success": True,
            "frame_index": checkpoint_frame,
        }

    def clear_checkpoints(self, session_id):
        session = self._get_session(session_id)
        self._extend_expiration_time(session)
        session["checkpoints"].clear()
        session["checkpoint_tensor_cache"].clear()
        return {"is_success": True}

    def close_session(
        self,
        session_id,
        run_gc_collect=True,
        clear_cache_threshold: int = _CLEAR_CACHE_THRESHOLD,
    ):
        """Close a session. Idempotent.

        ``run_gc_collect=True`` (the default) also returns the session's
        freed CUDA tensors back to the device by calling
        ``torch.cuda.empty_cache()`` after ``gc.collect()``. Without this,
        PyTorch's caching allocator retains the freed allocations in its
        per-process pool, so reserved memory keeps climbing across
        long-running workloads even though the Python-level objects are gone.

        ``empty_cache()`` itself triggers a CUDA sync, so it is gated on
        device memory pressure via the ``gpu_mem`` snapshot. Callers can
        override the threshold per-call via ``clear_cache_threshold``.

        When ``run_gc_collect=True``, the response includes a ``gpu_mem``
        snapshot (free / total / allocated / reserved bytes, plus active
        session count) so clients can decide whether the device has
        headroom for their next session — no separate admission RPC
        needed. See ``_gpu_mem_snapshot`` for the field shape. The
        first snapshot (after ``gc.collect()``) drives the cache-eviction
        decision; if ``empty_cache()`` fires, a second snapshot is taken
        so the response reflects the post-cleanup state and the freed
        bytes are logged. Old callers that only read ``is_success`` work
        unchanged — additional dict keys are ignored at the JSON layer.
        """
        session = self._all_inference_states.pop(session_id, None)
        result = {"is_success": True}
        if session is None:
            logger.warning(f"cannot close session {session_id} as it does not exist")
        else:
            # Explicitly clear the per-session dicts BEFORE ``del session``.
            #
            # ``inference_state`` (i.e., ``session["state"]``) is the dict
            # built by ``init_state`` / ``_construct_initial_input_batch``
            # and grown by ``add_prompt`` / propagation. It holds heavy
            # GPU-resident references — ``input_batch`` (the video frames
            # as a ``BatchedDatapoint`` on device),
            # ``constants["empty_geometric_prompt"]`` (zeroed device
            # tensors), and the per-inference accumulators
            # ``feature_cache``, ``cached_frame_outputs``,
            # ``tracker_inference_states``, ``tracker_metadata``.
            #
            # Empirically, relying on ``del session`` + ``gc.collect()``
            # alone has been insufficient in prod: across long-lived
            # IPNext replicas serving thousands of sessions, PyTorch's
            # *allocated* bucket monotonically climbs to ~93 GiB even
            # while ``empty_cache`` keeps *reserved-but-unallocated* at
            # ~600 MiB, leading to OOMs at concurrency=1 (SAM3 client
            # observation 2026-05-09 / 2026-05-10, jiids
            # 37154706936429064 and 41658306563594281). The fingerprint:
            #
            #     this process has 94.99 GiB memory in use.
            #     93.54 GiB allocated by PyTorch.
            #     577.70 MiB reserved by PyTorch but unallocated.
            #
            # which is the signature of dict-keyed references staying
            # alive (allocated bucket, not the cache). Calling ``clear()``
            # on the nested dict immediately drops all per-session tensor
            # refs the dict was keying, regardless of whether something
            # else (closure, asyncio task, metrics buffer, parent
            # container) is still holding the wrapper dict alive via a
            # cycle. Lists in the inference state hold ``None``s for
            # per-frame slots, so they don't need separate handling.
            state = session.get("state")
            if isinstance(state, dict):
                state.clear()
            session.clear()
            del session
            if run_gc_collect:
                gc.collect()
                gpu_mem = self._gpu_mem_snapshot()
                if (
                    torch.cuda.is_available()
                    and gpu_mem["total_bytes"] > 0
                    and (100.0 - gpu_mem["free_pct"]) >= clear_cache_threshold
                ):
                    torch.cuda.empty_cache()
                    post_gpu_mem = self._gpu_mem_snapshot()
                    logger.info(
                        f"empty_cache freed "
                        f"{post_gpu_mem['free_bytes'] - gpu_mem['free_bytes']} bytes "
                        f"(free_pct {gpu_mem['free_pct']:.1f}% -> "
                        f"{post_gpu_mem['free_pct']:.1f}%, reserved "
                        f"{gpu_mem['reserved_bytes']} -> "
                        f"{post_gpu_mem['reserved_bytes']} bytes)"
                    )
                    gpu_mem = post_gpu_mem
                # pyrefly: ignore [bad-assignment]
                result["gpu_mem"] = gpu_mem
            logger.info(f"removed session {session_id}")
        return result

    def _gpu_mem_snapshot(self) -> dict:
        """Snapshot of current GPU memory state for inclusion in
        session-close responses.

        Lets clients track free HBM across the fleet without a separate
        admission RPC: every ``close_session`` naturally exposes the
        post-cleanup state (taken AFTER any ``empty_cache`` call), which
        is exactly what the next session will face.

        Fields:
          - ``free_bytes`` / ``total_bytes`` — raw
            ``torch.cuda.mem_get_info()``.
          - ``allocated_bytes`` — ``torch.cuda.memory_allocated()``,
            live tensor footprint (no caching pool overhead).
          - ``reserved_bytes`` — ``torch.cuda.memory_reserved()``,
            caching-allocator pool size.
          - ``free_pct`` — ``free_bytes / total_bytes * 100`` for
            convenient % thresholds.
          - ``active_session_count`` — sessions still resident on this
            predictor instance after the close.

        Fail-open: any error reading the device returns zero stats so
        a broken CUDA context (e.g., CPU-only test env) NEVER breaks
        the session-close response.
        """
        active_count = len(self._all_inference_states)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            # No active CUDA context (e.g., CPU-only test env).
            return {
                "free_bytes": 0,
                "total_bytes": 0,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "free_pct": 0.0,
                "active_session_count": active_count,
            }
        free_pct = (free_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0
        # ``memory_allocated`` / ``memory_reserved`` are cheap host-side
        # bookkeeping reads (no CUDA sync) and complement
        # ``mem_get_info`` by exposing the caching-allocator pool size
        # vs the actual live tensor footprint.
        allocated_bytes = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        reserved_bytes = (
            torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        )
        return {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
            "free_pct": free_pct,
            "active_session_count": active_count,
        }

    def _get_session(self, session_id):
        session = self._all_inference_states.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _extend_expiration_time(self, session):
        """Update last-use time for session expiration tracking."""
        session["last_use_time"] = time.time()

    def shutdown(self):
        """Shutdown the predictor and clear all sessions."""
        self._all_inference_states.clear()

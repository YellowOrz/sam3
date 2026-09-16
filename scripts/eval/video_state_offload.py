"""Opt-in, instance-local use of SAM3's native CPU tracker-state storage.

This changes only the tracker ``init_state`` storage option.  It does not move
arbitrary caches, remove history, change dtypes, or change tracking decisions.
The enclosing video session must also be started with
``offload_state_to_cpu=True``.  Install before creating any tracker states.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping, Sequence
from typing import Any


def assert_tracker_state_cpu(state: Mapping[str, Any]) -> None:
    """Fail closed if native init_state did not honor the requested storage."""
    device = state.get("storage_device")
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type != "cpu" or state.get("offload_state_to_cpu") is not True:
        raise RuntimeError(
            "tracker init_state did not enable native CPU state storage: "
            f"storage_device={device!s}, "
            f"offload_state_to_cpu={state.get('offload_state_to_cpu')!r}"
        )


class TrackerCPUOffloadHandle:
    """Restoreable instance-method patch; construction explicitly enables it.

    Native callers which omit ``offload_state_to_cpu`` receive ``True``.
    Explicit contradictory values raise rather than silently overriding the
    caller.  All other positional/keyword arguments and defaults are unchanged.
    The original method is restored on context exit, including exception exit.
    """

    def __init__(self, tracker: Any):
        original = tracker.init_state
        if getattr(original, "_native_cpu_offload_wrapper", False):
            raise RuntimeError("CPU tracker-state offload is already installed")
        signature = inspect.signature(original)
        parameter = signature.parameters.get("offload_state_to_cpu")
        if parameter is None or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("tracker.init_state must expose offload_state_to_cpu")
        self._tracker = tracker
        self._had_override = "init_state" in vars(tracker)
        self._previous_override = vars(tracker).get("init_state")
        self._restored = False
        self.states_created = 0

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            if (
                "offload_state_to_cpu" in bound.arguments
                and bound.arguments["offload_state_to_cpu"] is not True
            ):
                raise ValueError(
                    "explicit offload_state_to_cpu conflicts with CPU-offload mode"
                )
            bound.arguments["offload_state_to_cpu"] = True
            state = original(*bound.args, **bound.kwargs)
            assert_tracker_state_cpu(state)
            self.states_created += 1
            return state

        wrapped._native_cpu_offload_wrapper = True
        self._wrapped = wrapped
        tracker.init_state = wrapped

    def restore(self) -> None:
        """Remove only this wrapper; never overwrite a later unrelated patch."""
        if self._restored:
            return
        if self._tracker.init_state is not self._wrapped:
            raise RuntimeError("tracker.init_state changed after CPU-offload installation")
        if self._had_override:
            self._tracker.init_state = self._previous_override
        else:
            del self._tracker.init_state
        self._restored = True

    def __enter__(self) -> "TrackerCPUOffloadHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def force_tracker_state_cpu(tracker: Any) -> TrackerCPUOffloadHandle:
    """Explicitly install the native-storage wrapper, returning its handle."""
    return TrackerCPUOffloadHandle(tracker)


class ForwardOutputCacheCPUHandle:
    """CPU-copy the auxiliary output cache for strictly forward-only evaluation.

    Original filtering runs first, and its immediate caller-owned CUDA tensors
    are left untouched.  This is not an interactive or reverse-tracking API:
    reading this cache through ``_build_tracker_output`` fails explicitly.
    """

    def __init__(self, model: Any):
        import torch

        names = ("_cache_frame_outputs", "_build_tracker_output")
        originals = {name: getattr(model, name) for name in names}
        if any(getattr(value, "_forward_cache_cpu_wrapper", False)
               for value in originals.values()):
            raise RuntimeError("forward output-cache CPU offload is already installed")
        signature = inspect.signature(originals["_cache_frame_outputs"])
        if not {"inference_state", "frame_idx"}.issubset(signature.parameters):
            raise TypeError("_cache_frame_outputs must expose inference_state and frame_idx")
        self._model = model
        self._previous = {
            name: (name in vars(model), vars(model).get(name)) for name in names
        }
        self._restored = False
        self.frames_cached = 0

        @functools.wraps(originals["_cache_frame_outputs"])
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            result = originals["_cache_frame_outputs"](*args, **kwargs)
            state = bound.arguments["inference_state"]
            index = bound.arguments["frame_idx"]
            current = state["cached_frame_outputs"][index]
            if not isinstance(current, Mapping) or any(
                not torch.is_tensor(value) for value in current.values()
            ):
                raise TypeError("cached frame must map object ids to mask tensors")
            # copy=True matters even when the input happens to be a CPU tensor.
            # Never detach_(), mutate the input mapping, or reuse its storages.
            state["cached_frame_outputs"][index] = {
                obj_id: mask.detach().to("cpu", copy=True)
                for obj_id, mask in current.items()
            }
            self.frames_cached += 1
            return result

        @functools.wraps(originals["_build_tracker_output"])
        def reject_interactive_cache_read(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "CPU output-cache mode is only for noninteractive forward evaluation; "
                "_build_tracker_output is unsupported"
            )

        wrapped._forward_cache_cpu_wrapper = True
        reject_interactive_cache_read._forward_cache_cpu_wrapper = True
        self._wrappers = {
            "_cache_frame_outputs": wrapped,
            "_build_tracker_output": reject_interactive_cache_read,
        }
        for name, value in self._wrappers.items():
            setattr(model, name, value)

    def restore(self) -> None:
        if self._restored:
            return
        if any(getattr(self._model, name) is not value
               for name, value in self._wrappers.items()):
            raise RuntimeError("video cache methods changed after CPU-offload installation")
        for name, (had_override, previous) in self._previous.items():
            if had_override:
                setattr(self._model, name, previous)
            else:
                delattr(self._model, name)
        self._restored = True

    def __enter__(self) -> "ForwardOutputCacheCPUHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.restore()


def force_forward_output_cache_cpu(model: Any) -> ForwardOutputCacheCPUHandle:
    """Explicitly enable CPU copies of forward-only auxiliary output masks."""
    return ForwardOutputCacheCPUHandle(model)


def _tensor_storage_inventory(values: Any) -> dict[str, dict[str, int]]:
    """Count tensor objects and unique backing storage without moving data."""
    import torch

    groups: dict[str, dict[str, int]] = {
        name: {"tensor_count": 0, "storage_count": 0, "storage_bytes": 0}
        for name in ("cpu", "cuda", "other")
    }
    seen_containers: set[int] = set()
    seen_tensors: set[int] = set()
    seen_storage: set[tuple[Any, ...]] = set()

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            if id(value) in seen_tensors:
                return
            seen_tensors.add(id(value))
            group = groups.get(value.device.type, groups["other"])
            group["tensor_count"] += 1
            if value.layout != torch.strided:
                raise TypeError("tracker output storage inventory expects strided tensors")
            storage = value.untyped_storage()
            size = storage.nbytes()
            key = (str(value.device), storage.data_ptr(), size)
            if key not in seen_storage:
                seen_storage.add(key)
                group["storage_count"] += 1
                group["storage_bytes"] += size
        elif isinstance(value, Mapping):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for child in value:
                visit(child)

    visit(values)
    return groups


def tracker_state_storage_snapshot(inference_state: Any) -> dict[str, Any]:
    """Read-only inventory of retained tracker output tensors and storage.

    Accepts a video state with ``tracker_inference_states``, one tracker state,
    or a list of tracker states.  Only known output/input-mask dictionaries are
    walked; image/model caches and arbitrary objects are deliberately excluded.
    Bytes describe *backing storage*, deduplicated across tensor views and all
    states; they are not CUDA allocator/reserved-memory or process-RSS readings.
    Small object pointers and scores are intentionally retained on CUDA by SAM3.

    The top-level cpu/cuda/other values count only tracker output dictionaries.
    Auxiliary cached_frame_outputs and mask_inputs_per_obj inventories are
    separate scopes: do not sum them, since they can share storage. ``combined``
    provides the storage-deduplicated total across these three known scopes.

    A snapshot scans the current history, so take periodic samples (e.g. every
    100 frames) rather than imposing a quadratic full-history walk every frame.
    No tensor values are read or copied and no CUDA synchronization is requested.
    """
    cached_outputs = {}
    if isinstance(inference_state, Mapping):
        if "tracker_inference_states" in inference_state:
            states = list(inference_state["tracker_inference_states"])
            cached_outputs = inference_state.get("cached_frame_outputs", {})
        elif "output_dict" in inference_state:
            states = [inference_state]
        else:
            raise ValueError("expected video state or tracker output state")
    elif isinstance(inference_state, Sequence) and not isinstance(
        inference_state, (str, bytes)
    ):
        states = list(inference_state)
    else:
        raise TypeError("expected video state, tracker state, or tracker-state list")

    # Repeated state references are not extra physical states or extra storage.
    states = list({id(state): state for state in states}.values())
    frames_per_state = []
    objects = 0
    storage_devices: set[str] = set()
    outputs = []
    input_masks = []
    large_outputs = {"maskmem_features": [], "pred_masks": []}
    for state in states:
        if not isinstance(state, Mapping):
            raise TypeError("tracker state must be a mapping")
        storage_devices.add(str(state.get("storage_device", "missing")))
        objects += len(state.get("obj_ids", ()))
        indices = set()
        output = state.get("output_dict", {})
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            indices.update(output.get(key, {}))
        frames_per_state.append(len(indices))
        for key in ("output_dict", "output_dict_per_obj", "temp_output_dict_per_obj"):
            outputs.append(state.get(key, {}))
        input_masks.append(state.get("mask_inputs_per_obj", {}))

    seen_containers = set()

    def collect_large_outputs(value: Any) -> None:
        if isinstance(value, Mapping):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for key, child in value.items():
                if key in large_outputs:
                    large_outputs[key].append(child)
                else:
                    collect_large_outputs(child)
        elif isinstance(value, (list, tuple)):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for child in value:
                collect_large_outputs(child)

    collect_large_outputs(outputs)
    return {
        "scope": "tracker_output_dicts_only_deduplicated_backing_storage",
        "tracker_state_count": len(states),
        "tracker_object_count": objects,
        "frame_outputs_per_state": frames_per_state,
        "frame_output_count": sum(frames_per_state),
        "storage_devices": sorted(storage_devices),
        **_tensor_storage_inventory(outputs),
        "cached_frame_outputs": {
            "frame_count": len(cached_outputs),
            **_tensor_storage_inventory(cached_outputs),
        },
        "mask_inputs_per_obj": _tensor_storage_inventory(input_masks),
        "large_tracker_outputs": {
            key: _tensor_storage_inventory(value) for key, value in large_outputs.items()
        },
        "combined": _tensor_storage_inventory([outputs, cached_outputs, input_masks]),
    }

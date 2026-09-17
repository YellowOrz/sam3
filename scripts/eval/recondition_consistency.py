"""Opt-in, reversible SAM3 single-GPU reconditioning consistency experiment.

No thresholds, matching, object lifecycle, weights or frozen source are changed.
The native add-mask/preflight path decides which corrections succeed. Only then
are their detection logits selected in the shared propagation buffer, BEFORE the
native occlusion and memory updates. Output uses that same selected buffer rather
than overriding it again with unsuppressed detections. This is not a SAM3.1 patch.
"""
from contextlib import AbstractContextManager
from functools import wraps
import hashlib
import inspect
import json


POLICY = "successful-recondition-selected-mask-v1"


def _bind(method, args, kwargs):
    values = inspect.signature(method).bind(*args, **kwargs)
    values.apply_defaults()
    return values


def _mask_digest(value):
    binary = value.detach().gt(0).to(device="cpu").contiguous().numpy()
    return dict(shape=list(binary.shape), pixels=int(binary.sum()),
                sha256=hashlib.sha256(binary.tobytes()).hexdigest())


class ConsistentReconditioning(AbstractContextManager):
    """One forward-only predictor instance; always restores methods on exit."""

    def __init__(self, model, stream=None):
        self.model, self.stream = model, stream
        self.undo = []
        self.context = None
        self.output_context = None
        self.counts = dict(planned_frames=0, correction_calls=0, corrected_frames=0,
                          corrected_objects=0, memory_writes_with_correction=0,
                          outputs_with_correction=0)

    def emit(self, event, **fields):
        if self.stream is not None:
            self.stream.write(json.dumps(dict(event=event, **fields), allow_nan=False)+"\n")
            self.stream.flush()

    def patch(self, obj, name, replacement):
        existed = name in obj.__dict__
        self.undo.append((obj, name, existed, obj.__dict__.get(name)))
        setattr(obj, name, replacement)

    def __enter__(self):
        model = self.model
        if getattr(model, "world_size", None) != 1 or getattr(model, "is_multiplex", False):
            raise ValueError("Consistency experiment supports only single-GPU non-multiplex SAM3")
        if getattr(model, "_recondition_consistency_active", False):
            raise ValueError("Reconditioning experiment is already installed")
        required = {
            "run_tracker_update_planning_phase": {"frame_idx", "reverse", "tracker_low_res_masks_global", "tracker_metadata_prev"},
            "_recondition_masklets": {"frame_idx", "det_out", "trk_id_to_max_iou_high_conf_det"},
            "_tracker_update_memories": {"frame_idx", "low_res_masks"},
            "build_outputs": {"frame_idx", "tracker_update_plan", "tracker_low_res_masks_global", "reconditioned_obj_ids"},
        }
        for name, keys in required.items():
            if not keys.issubset(inspect.signature(getattr(model, name)).parameters):
                raise ValueError(f"Unsupported native method contract: {name}")
        if not {"frame_idx", "obj_id", "mask"}.issubset(inspect.signature(model.tracker.add_new_mask).parameters):
            raise ValueError("Unsupported native add_new_mask contract")
        planning = model.run_tracker_update_planning_phase
        recondition = model._recondition_masklets
        add_mask = model.tracker.add_new_mask
        memory = model._tracker_update_memories
        build = model.build_outputs

        @wraps(planning)
        def plan_wrapper(*args, **kwargs):
            b = _bind(planning, args, kwargs).arguments
            if b["reverse"] or self.context is not None:
                raise ValueError("Consistency experiment requires non-reentrant forward-only inference")
            ids = [int(i) for i in b["tracker_metadata_prev"]["obj_ids_all_gpu"]]
            if len(set(ids)) != len(ids) or len(ids) != len(b["tracker_low_res_masks_global"]):
                raise ValueError("Invalid shared object/mask order")
            ctx = dict(frame=int(b["frame_idx"]), masks=b["tracker_low_res_masks_global"],
                       indices={id: i for i, id in enumerate(ids)}, selected=set(), adding=None)
            self.context = ctx
            self.output_context = None
            try:
                result = planning(*args, **kwargs)
                # Do not keep IDs that failed the native score/local-state gates.
                result[0]["reconditioned_obj_ids"] = set(ctx["selected"])
                self.output_context = ctx
                self.counts["planned_frames"] += 1
                self.counts["corrected_frames"] += bool(ctx["selected"])
                self.counts["corrected_objects"] += len(ctx["selected"])
                return result
            finally:
                self.context = None

        @wraps(add_mask)
        def add_wrapper(*args, **kwargs):
            ctx = self.context
            result = add_mask(*args, **kwargs)
            if ctx is not None and ctx["adding"] is not None:
                b = _bind(add_mask, args, kwargs).arguments
                if int(b["frame_idx"]) != ctx["frame"]:
                    raise ValueError("Correction frame mismatch")
                ctx["adding"].add(int(b["obj_id"]))
            return result

        @wraps(recondition)
        def recondition_wrapper(*args, **kwargs):
            import torch
            import torch.nn.functional as F
            ctx = self.context
            if ctx is None or ctx["adding"] is not None:
                raise ValueError("Reconditioning outside the native planning transaction")
            b = _bind(recondition, args, kwargs).arguments
            if int(b["frame_idx"]) != ctx["frame"]:
                raise ValueError("Reconditioning frame mismatch")
            ctx["adding"] = set()
            self.counts["correction_calls"] += 1
            try:
                result = recondition(*args, **kwargs)
                succeeded = sorted(ctx["adding"])
                replacements = []
                for id in succeeded:
                    idx = ctx["indices"][id]
                    det_idx = b["trk_id_to_max_iou_high_conf_det"][id]
                    det = b["det_out"]["mask"][det_idx:det_idx+1, None]
                    selected = F.interpolate(det.float(), size=ctx["masks"].shape[-2:],
                        mode="bilinear", align_corners=False)[0, 0].to(ctx["masks"])
                    if not torch.isfinite(selected).all():
                        raise ValueError("Nonfinite successful correction")
                    replacements.append((id, idx, selected))
                # Native add-mask AND all its preflights returned successfully.
                for id, idx, selected in replacements:
                    before = _mask_digest(ctx["masks"][idx]) if self.stream else None
                    ctx["masks"][idx].copy_(selected)
                    ctx["selected"].add(id)
                    self.emit("selected", frame=ctx["frame"], id=id, before=before,
                              selected=_mask_digest(selected) if self.stream else None)
                self.emit("recondition", frame=ctx["frame"],
                    candidates=sorted(int(i) for i in b["trk_id_to_max_iou_high_conf_det"]),
                    succeeded=succeeded)
                return result
            finally:
                ctx["adding"] = None

        @wraps(memory)
        def memory_wrapper(*args, **kwargs):
            b = _bind(memory, args, kwargs).arguments
            ctx = self.context
            if ctx is not None and ctx["selected"]:
                if int(b["frame_idx"]) != ctx["frame"]:
                    raise ValueError("Memory correction frame mismatch")
                self.counts["memory_writes_with_correction"] += 1
                self.emit("memory_input", frame=ctx["frame"], masks={str(id):
                    _mask_digest(b["low_res_masks"][ctx["indices"][id]]) for id in sorted(ctx["selected"])}
                    if self.stream else {})
            return memory(*args, **kwargs)

        @wraps(build)
        def build_wrapper(*args, **kwargs):
            b = _bind(build, args, kwargs)
            ctx = self.output_context
            if ctx is None or int(b.arguments["frame_idx"]) != ctx["frame"]:
                raise ValueError("Output without corresponding planning transaction")
            # Already selected BEFORE occlusion/memory. Never resurrect suppressed
            # pixels by applying the native raw-detector override a second time.
            b.arguments["reconditioned_obj_ids"] = set()
            if ctx["selected"]:
                self.counts["outputs_with_correction"] += 1
                self.emit("output_source", frame=ctx["frame"], masks={str(id):
                    _mask_digest(b.arguments["tracker_low_res_masks_global"][ctx["indices"][id]])
                    for id in sorted(ctx["selected"])} if self.stream else {})
            return build(*b.args, **b.kwargs)

        try:
            self.patch(model, "_recondition_consistency_active", True)
            self.patch(model, "run_tracker_update_planning_phase", plan_wrapper)
            self.patch(model, "_recondition_masklets", recondition_wrapper)
            self.patch(model.tracker, "add_new_mask", add_wrapper)
            self.patch(model, "_tracker_update_memories", memory_wrapper)
            self.patch(model, "build_outputs", build_wrapper)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for obj, name, existed, previous in reversed(self.undo):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.undo.clear()
        self.context = self.output_context = None
        return False

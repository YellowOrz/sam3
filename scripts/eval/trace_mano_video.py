"""Read-only lifecycle tracing of a frozen MANO-prompted SAM3 video run.

No thresholds, masks, associations, or model parameters are changed by hooks.
Reference masks are never opened by inference; evaluation is a separate pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from . import mano_box_factorial as f
except ImportError:
    import mano_box_factorial as f


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return [plain(v) for v in sorted(value)]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if type(value).__module__ == "torch" and hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported trace value: {type(value)}")


def metadata_snapshot(metadata, frame):
    rank = metadata.get("rank0_metadata", {})
    result = {k: metadata.get(k) for k in (
        "obj_ids_all_gpu", "max_obj_id", "obj_id_to_score", "obj_id_to_last_occluded")}
    result["tracker_scores"] = metadata.get("obj_id_to_tracker_score_frame_wise", {}).get(frame, {})
    result["rank0"] = {k: rank.get(k) for k in (
        "obj_first_frame_idx", "trk_keep_alive", "removed_obj_ids", "masklet_confirmation")}
    result["rank0"]["suppressed"] = rank.get("suppressed_obj_ids", {}).get(frame, set())
    result["rank0"]["unmatched_counts"] = {k: len(v) for k, v in rank.get("unmatched_frame_inds", {}).items()}
    result["rank0"]["duplicate_counts"] = {str(k): len(v) for k, v in rank.get("overlap_pair_to_frame_inds", {}).items()}
    return plain(result)


def mask_overlap_matrix(detections, tracks):
    if detections.ndim != 3 or tracks.ndim != 3 or detections.shape[1:] != tracks.shape[1:]:
        raise ValueError("Expected equal-spatial-size binary mask stacks")
    out = []
    for d in detections.astype(bool):
        row = []
        for t in tracks.astype(bool):
            union = int((d | t).sum())
            row.append(int((d & t).sum()) / union if union else 0.)
        out.append(row)
    return out


class LifecycleTrace:
    def __init__(self, model, stream):
        self.model, self.stream = model, stream
        self.frame = None
        self.out_frames = {}
        self.originals = {}
        self.events = 0

    def emit(self, event, frame=None, **data):
        record = dict(event=event, frame=self.frame if frame is None else frame, **data)
        self.stream.write(json.dumps(plain(record), allow_nan=False) + "\n")
        self.stream.flush()
        self.events += 1

    def install(self):
        names = ("_run_single_frame_inference", "_associate_det_trk", "_process_hotstart", "_postprocess_output")
        for name in names:
            original = getattr(self.model, name)
            self.originals[name] = original
            setattr(self.model, name, self._wrap(name, original))

    def restore(self):
        for name, original in self.originals.items():
            setattr(self.model, name, original)

    def _wrap(self, name, original):
        signature = inspect.signature(original)

        def wrapped(*args, **kwargs):
            b = signature.bind(*args, **kwargs)
            b.apply_defaults()
            values = b.arguments
            if name == "_run_single_frame_inference":
                self.frame = int(values["frame_idx"])
                state = values["inference_state"]
                self.emit("before", metadata=metadata_snapshot(state["tracker_metadata"], self.frame))
                out = original(*args, **kwargs)
                self.out_frames[id(out)] = self.frame
                masks = {str(k): f.encode(v.detach().cpu().numpy().reshape(v.shape[-2:]))
                         for k, v in out["obj_id_to_mask"].items()}
                self.emit("after", metadata=metadata_snapshot(state["tracker_metadata"], self.frame), masks=masks,
                          unconfirmed=out.get("unconfirmed_obj_ids", []))
                return out
            if name == "_associate_det_trk":
                # Copy inputs before the original call; never mutate tensors or lazy result objects.
                det = values["det_masks"].detach().gt(0).cpu().numpy()
                trk = values["trk_masks"].detach().gt(0).cpu().numpy()
                ids = np.asarray(values["trk_obj_ids"]).copy()
                scores = np.asarray(values["det_scores_np"]).copy()
                out = original(*args, **kwargs)
                self.emit("associate", tracker_ids=ids, detector_scores=scores,
                          iou=mask_overlap_matrix(det, trk),
                          track_masks=[f.encode(m) for m in trk],
                          new_detection_indices=out[0], unmatched=out[1], matches=out[2],
                          recondition_matches=out[3], empty=out[4])
                return out
            if name == "_process_hotstart":
                before = plain(values["rank0_metadata"])
                out = original(*args, **kwargs)
                self.emit("hotstart", frame=int(values["frame_idx"]),
                          new_ids=values["new_det_obj_ids"], newly_removed=out[0],
                          keep_before=before["trk_keep_alive"],
                          keep_after=out[1]["trk_keep_alive"],
                          removed=out[1]["removed_obj_ids"],
                          suppressed=out[1]["suppressed_obj_ids"].get(values["frame_idx"], set()))
                return out
            out = original(*args, **kwargs)
            source_frame = self.out_frames.pop(id(values["out"]), None)
            self.emit("postprocess", frame=source_frame,
                      computed_through_frame=self.frame,
                      removed=values.get("removed_obj_ids"), suppressed=values.get("suppressed_obj_ids"),
                      unconfirmed=values.get("unconfirmed_obj_ids"),
                      accepted_ids=out["out_obj_ids"], detector_scores=out["out_probs"],
                      tracker_scores=out["out_tracker_probs"])
            return out
        return wrapped


def run(a):
    import torch
    import torch.nn.functional as F
    plan = json.loads((a.root / "plan.json").read_text())
    seq = next(s for s in plan["sequences"] if s["name"] == a.sequence)
    f.check_inputs(a.root, dict(plan, sequences=[seq]), a.model_source, a.checkpoint)
    if f.sha(a.model_source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz") != plan["tokenizer_sha256"]:
        raise ValueError("Tokenizer mismatch")
    if a.output.exists():
        raise ValueError("Output must be new; partial runs are never overwritten")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available GPU")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.set_per_process_memory_fraction(.85)
    sys.path.insert(0, str(a.model_source))
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    from sam3.model_builder import build_sam3_predictor
    factory = Sam3VideoPredictor if a.entry == "direct" else build_sam3_predictor
    extra = {} if a.entry == "direct" else dict(version="sam3", gpus_to_use=[0])
    predictor = factory(checkpoint_path=str(a.checkpoint),
        bpe_path=str(a.model_source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        compile=False, async_loading_frames=True, strict_state_dict_loading=True, **extra)
    model = predictor.model.eval().requires_grad_(False)
    versions = [(p, p._version) for p in model.parameters()]
    n = min(a.limit or seq["frame_count"], seq["frame_count"])
    shape = (seq["height"], seq["width"])
    prompts = {int(k): v for k, v in seq["prompts"].items()}
    config_names = ("score_threshold_detection", "det_nms_thresh", "assoc_iou_thresh", "trk_assoc_iou_thresh",
        "new_det_thresh", "hotstart_delay", "hotstart_unmatch_thresh", "hotstart_dup_thresh",
        "suppress_unmatched_only_within_hotstart", "init_trk_keep_alive", "max_trk_keep_alive", "min_trk_keep_alive",
        "decrease_trk_keep_alive_for_empty_masklets", "masklet_confirmation_enable",
        "masklet_confirmation_consecutive_det_thresh", "recondition_every_nth_frame", "reconstruction_bbox_iou_thresh",
        "suppress_overlapping_based_on_recent_occlusion_threshold", "use_iom_recondition")
    a.output.mkdir(parents=True)
    runinfo = dict(sequence=a.sequence, frames=n, entry=a.entry, seed=a.seed,
        plan_sha256=f.sha(a.root / "plan.json"), model_source_sha256=plan["model_source_sha256"],
        checkpoint_sha256=plan["base_sha256"], runner_sha256=f.sha(__file__),
        helper_sha256=f.sha(f.__file__), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), model_config={k: getattr(model, k, None) for k in config_names},
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        started_at=datetime.now(timezone.utc).isoformat(), trace_enabled=not a.no_trace)
    f.write_json(a.output / "run.json", runinfo)
    stream = (a.output / "trace.jsonl").open("x")
    trace = LifecycleTrace(model, stream)
    if not a.no_trace:
        trace.install()
    original = model.run_backbone_and_detection
    signature = inspect.signature(original)
    frame_rows = {}

    def tapped(*args, **kwargs):
        b = signature.bind(*args, **kwargs).arguments
        i = int(b["frame_idx"])
        boxes = b["geometric_prompt"].box_embeddings
        actual = [] if boxes is None else boxes.detach().float().cpu().reshape(-1, 4).tolist()
        expected = [f.box_cxcywh(prompts[i]["boxes"][0])] if i in prompts else []
        if len(actual) != len(expected) or (actual and not np.allclose(actual, expected, atol=1e-6)):
            raise ValueError(f"Wrong geometry at source frame {i}")
        out = original(*args, **kwargs)
        scores = out["scores"].detach().float().cpu().numpy()
        if not torch.isfinite(out["mask"]).all() or (scores <= .5).any():
            raise ValueError("Invalid detector result")
        masks = F.interpolate(out["mask"][:, None].float(), shape, mode="bilinear", align_corners=False)[:, 0].gt(0).cpu().numpy() if len(scores) else np.zeros((0, *shape), bool)
        row = dict(sequence=a.sequence, frame_index=i, method="frame_box",
            **f.instance_record(masks, scores, np.arange(len(scores)), shape))
        if i in frame_rows and frame_rows[i] != row:
            raise ValueError("Repeated detector output changed")
        frame_rows[i] = row
        return out

    model.run_backbone_and_detection = tapped
    session = None
    started = time.monotonic()
    try:
        session = predictor.handle_request(dict(type="start_session", resource_path=str(a.root / a.sequence / "rgb"),
            offload_video_to_cpu=True, offload_state_to_cpu=False))["session_id"]
        predictor.handle_request(dict(type="add_geometry_prompts", session_id=session, frame_index=0,
            text="right hand", geometry_prompts=prompts))
        request = dict(type="propagate_in_video", session_id=session, propagation_direction="forward", start_frame_index=0)
        if a.limit:
            request["max_frame_num_to_track"] = n - 1
        seen = []
        with (a.output / "records.jsonl").open("x") as records:
            for item in predictor.handle_stream_request(request):
                i = int(item["frame_index"])
                if i != len(seen) or i >= n or i not in frame_rows:
                    raise ValueError("Noncontiguous outputs")
                out = item["outputs"]
                masks = np.asarray(out["out_binary_masks"]).reshape(-1, *shape)
                row = dict(sequence=a.sequence, frame_index=i, method="video_box",
                    **f.instance_record(masks, out["out_probs"], out["out_obj_ids"], shape))
                for r in (frame_rows[i], row):
                    records.write(json.dumps(r, allow_nan=False) + "\n")
                records.flush()
                seen.append(i)
                if len(seen) % 50 == 0 or len(seen) == n:
                    print(json.dumps(dict(frames=len(seen), total=n, elapsed=time.monotonic()-started)), flush=True)
                if time.monotonic()-started > a.max_seconds:
                    raise RuntimeError("Finite budget exceeded")
        if seen != list(range(n)) or any(p.requires_grad or p._version != v for p, v in versions):
            raise ValueError("Coverage or frozen weights violated")
        f.check_inputs(a.root, dict(plan, sequences=[seq]), a.model_source, a.checkpoint)
        f.write_json(a.output / "summary.json", dict(status="complete", **runinfo,
            events=trace.events, elapsed_seconds=time.monotonic()-started,
            records_sha256=f.sha(a.output / "records.jsonl"), trace_sha256=f.sha(a.output / "trace.jsonl")))
    except Exception as exc:
        f.write_json(a.output / "failure.json", dict(error=repr(exc)))
        raise
    finally:
        model.run_backbone_and_detection = original
        trace.restore()
        stream.close()
        if session is not None:
            predictor.handle_request(dict(type="close_session", session_id=session))
        predictor.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "model-source", "checkpoint", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--sequence", default="basket")
    p.add_argument("--entry", choices=("direct", "factory"), default="direct")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--no-trace", action="store_true")
    a = p.parse_args()
    if (a.limit is not None and a.limit < 1) or a.max_seconds < 1:
        p.error("limits must be positive")
    run(a)


if __name__ == "__main__":
    main()

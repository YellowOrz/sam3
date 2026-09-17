"""Frozen box/text x frame-detector/video diagnostic, with source-anchored scoring.

Run against an explicitly supplied, hashed SAM3 source snapshot that implements
per-frame geometry_prompts. No training, GT-dependent candidate selection or
runtime model-source edits. The frame baseline is the same detector's accepted
output before association/tracking; it is not the separate image-processor API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw


FORMAT = "sam3-mano-box-factorial-v1"
METHODS = ("frame_text", "frame_box", "video_text", "video_box")
CONTRACT = dict(
    format=FORMAT, prompt="right hand", inference_stride=1, scoring_stride=3,
    scoring_anchor=0, empty_empty_dice=1, mask_threshold=0.5,
    detector_threshold=0.5, nms_iou_threshold=0.1, boundary_pixels=4,
    precision="bfloat16", seed=123, training=False,
    frame_baseline="same-video-detector-after-NMS-before-tracking",
    frame_score="joint-presence-already-included-do-not-multiply-again",
    box_source="MANO_wilor/right_hand/result_mano_1.npz:mesh",
    box_padding=0.05, prompt_interval=1, missing_geometry="text-fallback",
    independent_ground_truth=False,
)
NAKE_CONTRACT = dict(CONTRACT, format="sam3-mano-box-factorial-nakehand-v1",
    dataset="nakehand",
    box_source="declared-unique-active-instance-mesh",
    reference_exclusion_policy="explicit-original-frame-indices-context-only")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def code_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / "sam3").rglob("*.py"))}


def sampling_indices(count):
    if type(count) is not int or count < 1:
        raise ValueError("Require positive source frame count")
    return list(range(0, count, 3))


def reference_exclusions(plan, seq):
    """Unknown references retain their original RGB frame, never become negatives."""
    if plan["contract"] == CONTRACT:
        return set()
    if plan["contract"] != NAKE_CONTRACT:
        raise ValueError("Protocol mismatch")
    values = seq.get("reference_excluded_indices")
    if (not isinstance(values, list) or any(type(i) is not int or not 0 <= i < seq["frame_count"] for i in values)
            or len(values) != len(set(values))):
        raise ValueError("Require explicit, unique original-frame reference exclusions")
    return set(values)


def select_sequences(plan, engineering=False, sequence=None):
    if sequence is not None:
        if engineering:
            raise ValueError("A sequence partition must run the full original video")
        selected = [s for s in plan["sequences"] if s["name"] == sequence]
        if len(selected) != 1:
            raise ValueError("Sequence partition must name exactly one registered video")
        return selected
    if not engineering:
        return plan["sequences"]
    if plan["contract"] == NAKE_CONTRACT:
        return plan["sequences"][:1]
    return [next(s for s in plan["sequences"] if s["name"] == "milk")]


def sampled_rows(rows, plan):
    indices = {s["name"]: set(s["sampled_indices"]) for s in plan["sequences"]}
    return [r for r in rows if r["frame_index"] in indices[r["sequence"]]]


def verify_scored_rows(rows, counts, methods, plan):
    exclusions = {s["name"]: reference_exclusions(plan, s) for s in plan["sequences"]}
    expected = {(s, i, m) for s, n in counts.items() for i in range(n)
                if i not in exclusions[s] for m in methods}
    keys = [(r["sequence"], r["frame_index"], r["method"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Missing/duplicate/unexpected known-reference score coverage")


def box_cxcywh(value):
    box = np.asarray(value, dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any()
            or (box[2:] <= 0).any() or (box[:2] + box[2:] > 1 + 1e-7).any()):
        raise ValueError("Require normalized xywh inside image")
    return np.concatenate((box[:2] + box[2:] / 2, box[2:])).tolist()


def encode(mask):
    from pycocotools import mask as mu
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Require 2D binary mask")
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=rle["size"], counts=rle["counts"].decode("ascii"))


def decode(rle):
    from pycocotools import mask as mu
    return mu.decode(rle).astype(bool)


def instance_record(masks, scores, ids, shape):
    masks, scores, ids = np.asarray(masks), np.asarray(scores), np.asarray(ids)
    if masks.shape != (len(ids), *shape) or scores.shape != (len(ids),):
        raise ValueError("Instance shape mismatch")
    if not np.isin(masks, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Nonbinary/nonfinite output")
    if ((scores < 0) | (scores > 1)).any() or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid scores or duplicate IDs")
    union = masks.astype(bool).any(0) if len(ids) else np.zeros(shape, bool)
    return dict(union_rle=encode(union), instances=[
        dict(id=int(i), score=float(s), rle=encode(m)) for i, s, m in zip(ids, scores, masks)
    ], prediction_pixels=int(union.sum()))


def verify_rows(rows, counts, methods):
    expected = {(s, i, m) for s, n in counts.items() for i in range(n) for m in methods}
    keys = [(r["sequence"], r["frame_index"], r["method"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Missing/duplicate/noncontiguous full-frame coverage")


def prepare(a):
    """Read original videos and actual producer-compatible projection implementation."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    cv2.setNumThreads(1)
    if a.output.exists():
        raise ValueError("Preparation output must be new")
    names = sorted(p.parent.parent.name for p in a.data_root.glob("*/masks_sam3/right_hand.mkv"))
    if not names:
        raise ValueError("No explicitly provided right-hand references")
    script = a.model_source / "scripts/process_mano_prompt_videos.py"
    # This is an explicitly selected trusted local implementation, never NPZ code.
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("frozen_mano_projection", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_hashes = code_hashes(a.model_source)
    producer_hashes = {str(p.relative_to(a.model_source)): sha(p) for p in [script,
        script.parent / "process_dataset_videos.py", script.parent / "common/video_utils.py"]}
    a.output.mkdir(parents=True)

    def sequence(name):
        root = a.data_root / name
        info = module.dataset.probe_video(root / "color.mp4")
        npz = module.find_mano(root / "color.mp4", "right", None)
        originals = [root / "color.mp4", root / "masks_sam3/right_hand.mkv", npz]
        left = root / "masks_sam3/left_hand.mkv"
        if left.exists():
            originals.append(left)
        originals_sha = {str(p): sha(p) for p in originals}
        args = argparse.Namespace(max_frames=None, focal_length=None, prompt_mode="box",
            box_source="mesh", hand_side="right", prompt_interval=1, box_padding=0.05)
        prompts, metadata = module.load_geometry(npz, info, args)
        for item in prompts.values():
            if set(item) != {"boxes"} or len(item["boxes"]) != 1:
                raise ValueError("Exactly one native box, no other prompt")
            box_cxcywh(item["boxes"][0])
        folder = a.output / name
        for sub in ("rgb", "right", "left"):
            (folder / sub).mkdir(parents=True)
        paths = {"rgb": root / "color.mp4", "right": originals[1]}
        if left.exists():
            paths["left"] = left
        caps = {k: cv2.VideoCapture(str(p)) for k, p in paths.items()}
        outputs = {}
        try:
            for cap in caps.values():
                if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != info["frame_count"]:
                    raise ValueError("Source stream coverage mismatch")
                if abs(cap.get(cv2.CAP_PROP_FPS) - info["fps"]) > 0.01:
                    raise ValueError("Source stream FPS mismatch")
            for i in range(info["frame_count"]):
                for key, cap in caps.items():
                    ok, frame = cap.read()
                    if not ok or frame.shape != (info["height"], info["width"], 3):
                        raise ValueError("Source frame missing or wrong size")
                    if key == "rgb":
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else:
                        if not np.array_equal(frame[..., 0], frame[..., 1]) or not np.array_equal(frame[..., 1], frame[..., 2]):
                            raise ValueError("Reference is not a grayscale label stream")
                        image = Image.fromarray((frame[..., 0] > 0).astype(np.uint8) * 255)
                    path = folder / key / f"{i:06d}.png"
                    image.save(path, compress_level=1)
                    # Exact decoded pixels verified before publishing.
                    if not np.array_equal(np.asarray(image), np.asarray(Image.open(path))):
                        raise ValueError("PNG roundtrip mismatch")
                    outputs[str(path.relative_to(a.output))] = sha(path)
            for cap in caps.values():
                if cap.read()[0]:
                    raise ValueError("Source has extra frames")
        finally:
            for cap in caps.values():
                cap.release()
        if originals_sha != {p: sha(p) for p in originals_sha}:
            raise ValueError("Source changed during export")
        result = dict(name=name, **info, prompts=prompts, mano=metadata,
            source_sha256=originals_sha, file_sha256=outputs,
            sampled_indices=sampling_indices(info["frame_count"]),
            render_indices=sorted(set([0, info["frame_count"] // 4,
                info["frame_count"] // 2, info["frame_count"] * 3 // 4, info["frame_count"] - 1])))
        print(json.dumps(dict(prepared=name, frames=info["frame_count"], boxes=len(prompts))), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        sequences = list(pool.map(sequence, names))
    if source_hashes != code_hashes(a.model_source):
        raise ValueError("Model source changed during preparation")
    write_json(a.output / "plan.json", dict(contract=CONTRACT, sequences=sequences,
        created_at=datetime.now(timezone.utc).isoformat(), model_source_sha256=source_hashes,
        projection_source_sha256=producer_hashes, base_sha256=a.base_sha256,
        tokenizer_sha256=sha(a.model_source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        raw_frames=sum(s["frame_count"] for s in sequences),
        scoring_frames=sum(len(s["sampled_indices"]) for s in sequences)))


def check_inputs(root, plan, source=None, checkpoint=None):
    if plan["contract"] not in (CONTRACT, NAKE_CONTRACT):
        raise ValueError("Protocol mismatch")
    if source is not None and code_hashes(source) != plan["model_source_sha256"]:
        raise ValueError("Frozen source mismatch")
    if checkpoint is not None and sha(checkpoint) != plan["base_sha256"]:
        raise ValueError("Base checkpoint mismatch")
    if plan["contract"] == NAKE_CONTRACT and plan["scoring_frames"] != sum(len(s["sampled_indices"]) for s in plan["sequences"]):
        raise ValueError("Scoring frame count differs from registered samples")
    for seq in plan["sequences"]:
        excluded = reference_exclusions(plan, seq)
        if seq["sampled_indices"] != [i for i in sampling_indices(seq["frame_count"]) if i not in excluded]:
            raise ValueError("Sampling anchor changed")
        if plan["contract"] == NAKE_CONTRACT:
            if any(type(seq.get(k)) is not int or seq[k] < 1 for k in ("height", "width")):
                raise ValueError("Require original image dimensions")
            for i in range(seq["frame_count"]):
                rgb = f"{seq['name']}/rgb/{i:06d}.png"
                ref = f"{seq['name']}/right/{i:06d}.png"
                if rgb not in seq["file_sha256"]:
                    raise ValueError("Require hashed contiguous RGB including excluded context")
                if i in excluded:
                    if ref in seq["file_sha256"] or (root / ref).exists():
                        raise ValueError("Excluded reference must not be synthesized or exported")
                elif ref not in seq["file_sha256"]:
                    raise ValueError("Missing unexcluded right-hand reference")
        for p, digest in seq["file_sha256"].items():
            if sha(root / p) != digest:
                raise ValueError(f"Input changed: {p}")


def run(a):
    import torch
    import torch.nn.functional as F
    from contextlib import ExitStack
    import resource
    cpu_storage = getattr(a, "cpu_tracker_state", False)
    equivalent_to = getattr(a, "equivalent_to", None)
    engineering_frames = getattr(a, "engineering_frames", 96)
    if engineering_frames < 1 or (not a.engineering and engineering_frames != 96):
        raise ValueError("Engineering length requires a positive explicit engineering run")
    if equivalent_to is not None and not cpu_storage:
        raise ValueError("Exact-prefix validation is reserved for explicit CPU-storage repair")
    helpers = {}
    if cpu_storage:
        if __package__:
            from scripts.eval import video_state_offload as storage
            from scripts.eval.video_output_equivalence import SavedOutputEquivalence
        else:
            import video_state_offload as storage
            from video_output_equivalence import SavedOutputEquivalence
        helpers = {p: sha(Path(__file__).parent / p) for p in
                   ("video_state_offload.py", "video_output_equivalence.py")}
    if a.output.exists():
        raise ValueError("Run output must be new; never append to partial videos")
    plan = json.loads((a.root / "plan.json").read_text())
    sequences = select_sequences(plan, a.engineering, getattr(a, "sequence", None))
    check_inputs(a.root, plan, a.model_source, a.checkpoint)
    tokenizer = a.model_source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    if sha(tokenizer) != plan["tokenizer_sha256"]:
        raise ValueError("Tokenizer mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available GPU to each process")
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.cuda.set_per_process_memory_fraction(0.85)
    sys.path.insert(0, str(a.model_source))
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor
    a.output.mkdir(parents=True)
    runinfo = dict(contract=plan["contract"], plan_sha256=sha(a.root / "plan.json"),
        runner_sha256=sha(__file__), mode=a.mode, engineering=a.engineering,
        model_source_sha256=plan["model_source_sha256"], base_sha256=plan["base_sha256"],
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        started_at=datetime.now(timezone.utc).isoformat(),
        storage_policy="cpu-tracker-and-forward-output-cache-v1" if cpu_storage else "native-gpu",
        storage_helpers_sha256=helpers, engineering_frames=engineering_frames if a.engineering else None,
        sequence_filter=getattr(a, "sequence", None))
    write_json(a.output / "run.json", runinfo)
    started = time.monotonic()
    predictor = Sam3VideoPredictor(checkpoint_path=str(a.checkpoint), bpe_path=str(tokenizer),
        compile=False, async_loading_frames=True, strict_state_dict_loading=True)
    model = predictor.model.eval().requires_grad_(False)
    if (model.score_threshold_detection != 0.5 or model.det_nms_thresh != 0.1
            or not model.detector.supervise_joint_box_scores):
        raise ValueError("Unexpected native detector protocol")
    versions = [(p, p._version) for p in model.parameters()]
    original = model.run_backbone_and_detection
    signature = inspect.signature(original)
    counts, summaries = {}, {}
    stack = ExitStack()
    verifier = SavedOutputEquivalence(equivalent_to, runinfo) if equivalent_to is not None else None

    try:
        if cpu_storage:
            stack.enter_context(storage.force_tracker_state_cpu(model.tracker))
            stack.enter_context(storage.force_forward_output_cache_cpu(model))
            memory_stream = stack.enter_context((a.output / "storage.jsonl").open("x"))
        for seq in sequences:
            name = seq["name"]
            n = seq["frame_count"] if not a.engineering else min(engineering_frames, seq["frame_count"])
            if verifier is not None:
                verifier.begin(name, n)
            shape = (seq["height"], seq["width"])
            counts[name] = n
            prompts = {int(k): v for k, v in seq["prompts"].items() if int(k) < n}
            frame_rows = {}
            stream = (a.output / f"{name}.jsonl").open("x")
            session_id = None
            geometry_seen = []

            def tapped(*args, **kwargs):
                bound = signature.bind(*args, **kwargs).arguments
                index = int(bound["frame_idx"])
                prompt = bound["geometric_prompt"]
                boxes = prompt.box_embeddings
                actual = [] if boxes is None else boxes.detach().float().cpu().reshape(-1, 4).tolist()
                expected = [box_cxcywh(prompts[index]["boxes"][0])] if a.mode == "box" and index in prompts else []
                if len(actual) != len(expected) or (actual and not np.allclose(actual, expected, atol=1e-6)):
                    raise ValueError(f"Geometry not delivered on correct source frame {index}")
                result = original(*args, **kwargs)
                logits = result["mask"]
                scores = result["scores"].detach().float().cpu().numpy()
                if not torch.isfinite(logits).all() or (scores <= 0.5).any():
                    raise ValueError("Invalid native detection masks/scores")
                if len(scores):
                    masks = F.interpolate(logits[:, None].float(), shape, mode="bilinear", align_corners=False)[:, 0].gt(0).cpu().numpy()
                else:
                    masks = np.zeros((0, *shape), bool)
                row = dict(sequence=name, frame_index=index, method=f"frame_{a.mode}",
                    geometry=actual, geometry_available=index in prompts,
                    **instance_record(masks, scores, np.arange(len(scores)), shape))
                if index in frame_rows and frame_rows[index] != row:
                    raise ValueError("Repeated frame detector output changed")
                frame_rows[index] = row
                geometry_seen.append(index)
                return result

            model.run_backbone_and_detection = tapped
            try:
                # Engineering limits only propagation, never remaps original frame indices.
                response = predictor.handle_request(dict(type="start_session", resource_path=str(a.root / name / "rgb"),
                    offload_video_to_cpu=True, offload_state_to_cpu=cpu_storage))
                session_id = response["session_id"]
                # Include full-video geometry so any one-frame lookahead has the correct prompt.
                full_prompts = {int(k): v for k, v in seq["prompts"].items()}
                if a.engineering:
                    prompts = full_prompts
                request = dict(type="add_prompt", session_id=session_id, frame_index=0, text="right hand")
                if a.mode == "box":
                    request.update(type="add_geometry_prompts", geometry_prompts=full_prompts)
                predictor.handle_request(request)
                seen = []
                request = dict(type="propagate_in_video", session_id=session_id,
                    propagation_direction="forward", start_frame_index=0, output_prob_thresh=0.5)
                if a.engineering:
                    request["max_frame_num_to_track"] = n - 1
                for item in predictor.handle_stream_request(request):
                    i = int(item["frame_index"])
                    if i != len(seen) or i >= n or i not in frame_rows:
                        raise ValueError("Noncontiguous video output or missing detector tap")
                    out = item["outputs"]
                    masks = np.asarray(out["out_binary_masks"])
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    ids = np.asarray(out["out_obj_ids"])
                    scores = np.asarray(out.get("out_probs", []))
                    row = dict(sequence=name, frame_index=i, method=f"video_{a.mode}",
                        geometry=frame_rows[i]["geometry"], geometry_available=i in prompts,
                        **instance_record(masks, scores, ids, shape))
                    for record in (frame_rows[i], row):
                        if verifier is not None:
                            verifier.check(record)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    seen.append(i)
                    if len(seen) % 50 == 0 or len(seen) == n:
                        print(json.dumps(dict(mode=a.mode, sequence=name, frames=len(seen), total=n,
                            elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
                    if cpu_storage and (i % 100 == 0 or i == n - 1):
                        state = predictor._all_inference_states[session_id]["state"]
                        # Native memory/low-resolution masks must really live on CPU.
                        for tracker_state in state["tracker_inference_states"]:
                            if str(tracker_state["storage_device"]) != "cpu":
                                raise ValueError("Tracker state did not adopt CPU storage")
                            for bucket in tracker_state["output_dict"].values():
                                for saved in bucket.values():
                                    for key in ("maskmem_features", "pred_masks"):
                                        value = saved.get(key)
                                        if value is not None and value.device.type != "cpu":
                                            raise ValueError(f"Persistent {key} unexpectedly remains on GPU")
                        for cached in state["cached_frame_outputs"].values():
                            if any(mask.device.type != "cpu" for mask in cached.values()):
                                raise ValueError("Forward output cache unexpectedly remains on GPU")
                        available = next(int(line.split()[1]) * 1024 for line in
                            Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
                        memory_stream.write(json.dumps(dict(sequence=name, frame_index=i,
                            cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            host_available_bytes=available,
                            storage=storage.tracker_state_storage_snapshot(state))) + "\n")
                        memory_stream.flush()
                        if available < 12 * 1024**3:
                            raise RuntimeError("Host RAM reserve below 12 GiB; stopping safely")
                    if time.monotonic() - started > a.max_seconds:
                        raise RuntimeError("Finite inference budget exceeded")
                if seen != list(range(n)):
                    raise ValueError("Incomplete propagation")
                if verifier is not None:
                    verifier.finish_sequence()
                summaries[name] = dict(frames=n, geometry_calls=len(geometry_seen),
                    distinct_detector_frames=len(set(geometry_seen)))
            finally:
                stream.close()
                model.run_backbone_and_detection = original
                if session_id is not None:
                    predictor.handle_request(dict(type="close_session", session_id=session_id))
            write_json(a.output / f"{name}-complete.json", dict(status="complete", **summaries[name],
                records_sha256=sha(a.output / f"{name}.jsonl")))
        if any(p.requires_grad or p._version != version for p, version in versions):
            raise ValueError("Frozen model weights mutated")
        check_inputs(a.root, plan, a.model_source, a.checkpoint)
        records = [json.loads(line) for s in counts for line in (a.output / f"{s}.jsonl").read_text().splitlines()]
        verify_rows(records, counts, (f"frame_{a.mode}", f"video_{a.mode}"))
        equivalence = verifier.summary() if verifier is not None else None
        if cpu_storage:
            memory_stream.flush()
            if any(sha(Path(__file__).parent / p) != digest for p, digest in helpers.items()):
                raise ValueError("Storage repair helper changed during inference")
        write_json(a.output / "summary.json", dict(status="complete", **runinfo, counts=counts,
            sequences=summaries, frozen_weights_verified=True, output_records=len(records),
            records_sha256={s: sha(a.output / f"{s}.jsonl") for s in counts},
            elapsed_seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            equivalence=equivalence,
            storage_records_sha256=sha(a.output / "storage.jsonl") if cpu_storage else None))
    except Exception as exc:
        write_json(a.output / "failure.json", dict(error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        model.run_backbone_and_detection = original
        try:
            stack.close()
        finally:
            predictor.shutdown()


def score(prediction, reference, other=None):
    from scipy.ndimage import binary_erosion
    if prediction.shape != reference.shape:
        raise ValueError("Prediction/reference shape mismatch")
    p, g = int(prediction.sum()), int(reference.sum())
    inter = int((prediction & reference).sum())
    def boundary(x):
        return x & ~binary_erosion(x, structure=np.ones((3, 3)), iterations=4, border_value=0)
    bp, bg = boundary(prediction), boundary(reference)
    denom = int((bp | bg).sum())
    return dict(prediction_pixels=p, reference_pixels=g, intersection=inter,
        dice=2*inter/(p+g) if p+g else 1., iou=inter/(p+g-inter) if p+g-inter else 1.,
        boundary_iou_4px=(int((bp & bg).sum())/denom if denom else 1.) if g else None,
        fp_pixels=p-inter, fn_pixels=g-inter, missed_positive=bool(g and not p),
        empty_false_positive=bool(not g and p),
        other_only_overlap_pixels=None if other is None else int((prediction & other & ~reference).sum()))


def aggregate(rows):
    if not rows:
        return dict(frames=0, mean_dice=None)
    positives = [r for r in rows if r["reference_pixels"]]
    return dict(frames=len(rows), mean_dice=float(np.mean([r["dice"] for r in rows])),
        positive_frames=len(positives), empty_frames=len(rows)-len(positives),
        positive_dice=float(np.mean([r["dice"] for r in positives])) if positives else None,
        positive_boundary_iou_4px=float(np.mean([r["boundary_iou_4px"] for r in positives])) if positives else None,
        **{k: sum(r[k] for r in rows) for k in ("missed_positive", "empty_false_positive", "fp_pixels", "fn_pixels")})


def verify_storage_trace(folder, summary):
    if summary.get("storage_policy") != "cpu-tracker-and-forward-output-cache-v1":
        return
    path = folder / "storage.jsonl"
    if sha(path) != summary["storage_records_sha256"]:
        raise ValueError("CPU storage telemetry changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {(name, i) for name, count in summary["counts"].items()
                for i in sorted(set(range(0, count, 100)) | {count - 1})}
    keys = [(r["sequence"], r["frame_index"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete CPU storage telemetry coverage")
    for row in rows:
        state = row["storage"]
        if state["storage_devices"] not in ([], ["cpu"]):
            raise ValueError("Tracker state did not remain on CPU")
        for values in [state["cached_frame_outputs"], *state["large_tracker_outputs"].values()]:
            if any(values[device]["tensor_count"] for device in ("cuda", "other")):
                raise ValueError("Large persistent output tensors were not CPU-offloaded")
        if row["host_available_bytes"] < 12 * 1024**3:
            raise ValueError("Host RAM reserve violated")


def audit(a):
    if a.output.exists():
        raise ValueError("Audit output must be new")
    plan = json.loads((a.root / "plan.json").read_text())
    check_inputs(a.root, plan)
    summary = json.loads((a.run / "summary.json").read_text())
    if (summary["status"] != "complete" or summary["plan_sha256"] != sha(a.root / "plan.json")
            or summary["contract"] != plan["contract"]):
        raise ValueError("Require complete run from exactly this plan")
    verify_storage_trace(a.run, summary)
    records = []
    for s in summary["counts"]:
        path = a.run / f"{s}.jsonl"
        if sha(path) != summary["records_sha256"][s]:
            raise ValueError("Output changed")
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    methods = (f"frame_{summary['mode']}", f"video_{summary['mode']}")
    verify_rows(records, summary["counts"], methods)
    scored, excluded_records = [], []
    by_name = {s["name"]: s for s in plan["sequences"]}
    exclusions = {name: reference_exclusions(plan, s) for name, s in by_name.items()}
    # Decode every mask (including nonscored context); never trust saved scalar metrics.
    for row in records:
        name, i = row["sequence"], row["frame_index"]
        available = str(i) in by_name[name]["prompts"]
        expected = [box_cxcywh(by_name[name]["prompts"][str(i)]["boxes"][0])] if summary["mode"] == "box" and available else []
        if (row["geometry_available"] != available or len(row["geometry"]) != len(expected)
                or (expected and not np.allclose(row["geometry"], expected, atol=1e-6))):
            raise ValueError("Saved prompt/absence differs from frozen source-frame plan")
        p = decode(row["union_rle"])
        if plan["contract"] == NAKE_CONTRACT and p.shape != (by_name[name]["height"], by_name[name]["width"]):
            raise ValueError("Prediction shape differs from source frame")
        union = np.zeros_like(p)
        for instance in row["instances"]:
            mask = decode(instance["rle"])
            if mask.shape != p.shape:
                raise ValueError("Instance shape mismatch")
            union |= mask
        if not np.array_equal(union, p) or int(p.sum()) != row["prediction_pixels"]:
            raise ValueError("Instance union or saved area mismatch")
        if i in exclusions[name]:
            excluded_records.append(dict(sequence=name, frame_index=i, method=row["method"],
                reference_excluded=True, reason="predeclared-reference-exclusion",
                geometry_available=row["geometry_available"], prediction_pixels=int(p.sum())))
            continue
        ref = np.asarray(Image.open(a.root / name / "right" / f"{i:06d}.png")) > 0
        other_path = a.root / name / "left" / f"{i:06d}.png"
        other = np.asarray(Image.open(other_path)) > 0 if other_path.exists() else None
        scored.append(dict(sequence=name, frame_index=i, method=row["method"],
            geometry_available=row["geometry_available"], **score(p, ref, other)))
    verify_scored_rows(scored, summary["counts"], methods, plan)
    sampled = sampled_rows(scored, plan)
    a.output.mkdir(parents=True)
    with (a.output / "scores.jsonl").open("x") as stream:
        for row in scored:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if plan["contract"] == NAKE_CONTRACT:
        with (a.output / "reference-excluded.jsonl").open("x") as stream:
            for row in excluded_records:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(a.output / "metrics.json", dict(status="complete", plan_sha256=summary["plan_sha256"],
        run_summary_sha256=sha(a.run / "summary.json"), engineering=summary["engineering"],
        counts=summary["counts"], methods={m: aggregate([r for r in sampled if r["method"] == m]) for m in methods},
        per_sequence={s: {m: aggregate([r for r in sampled if r["method"] == m and r["sequence"] == s])
            for m in methods} for s in summary["counts"]},
        all_frames_verified=len(records), scores_sha256=sha(a.output / "scores.jsonl"),
        **(dict(reference_excluded_records=len(excluded_records),
            reference_excluded_sha256=sha(a.output / "reference-excluded.jsonl"))
           if plan["contract"] == NAKE_CONTRACT else {})))


def compare(a):
    if a.output.exists():
        raise ValueError("Comparison output must be new")
    plan = json.loads((a.root / "plan.json").read_text())
    check_inputs(a.root, plan)
    runs, scored, records = [], [], {}
    for mode in ("text", "box"):
        run_path = getattr(a, mode + "_run")
        audit_path = getattr(a, mode + "_audit")
        runinfo = json.loads((run_path / "summary.json").read_text())
        proof = json.loads((audit_path / "metrics.json").read_text())
        if (proof["status"] != "complete" or runinfo["engineering"]
                or proof["run_summary_sha256"] != sha(run_path / "summary.json")
                or proof["scores_sha256"] != sha(audit_path / "scores.jsonl")
                or runinfo["mode"] != mode):
            raise ValueError("Require complete independently audited formal runs")
        if (plan["contract"] == NAKE_CONTRACT and
                proof["reference_excluded_sha256"] != sha(audit_path / "reference-excluded.jsonl")):
            raise ValueError("Reference exclusion audit changed")
        runs.append(runinfo)
        scored.extend(json.loads(line) for line in (audit_path / "scores.jsonl").read_text().splitlines())
        for name, digest in runinfo["records_sha256"].items():
            path = run_path / f"{name}.jsonl"
            if sha(path) != digest:
                raise ValueError("Run output changed after audit")
            for line in path.read_text().splitlines():
                row = json.loads(line)
                records[row["sequence"], row["frame_index"], row["method"]] = row
    for key in ("contract", "plan_sha256", "runner_sha256", "model_source_sha256", "base_sha256", "counts", "torch_version"):
        if runs[0][key] != runs[1][key]:
            raise ValueError(f"Unpaired experiment field: {key}")
    for key in ("storage_policy", "storage_helpers_sha256"):
        if runs[0].get(key) != runs[1].get(key):
            raise ValueError(f"Unpaired storage-only repair field: {key}")
    if runs[0]["plan_sha256"] != sha(a.root / "plan.json") or runs[0]["contract"] != plan["contract"]:
        raise ValueError("Different data plan")
    counts = {s["name"]: s["frame_count"] for s in plan["sequences"]}
    verify_scored_rows(scored, counts, METHODS, plan)
    verify_rows(list(records.values()), counts, METHODS)
    sampled = sampled_rows(scored, plan)
    a.output.mkdir(parents=True)
    report = dict(status="complete", contract=plan["contract"], plan_sha256=sha(a.root / "plan.json"),
        raw_frames_per_condition=sum(counts.values()), scoring_frames_per_condition=plan["scoring_frames"],
        overall={m: aggregate([r for r in sampled if r["method"] == m]) for m in METHODS},
        per_sequence={s: {m: aggregate([r for r in sampled if r["method"] == m and r["sequence"] == s]) for m in METHODS} for s in counts},
        geometry_coverage={str(flag): {m: aggregate([r for r in sampled if r["method"] == m and r["geometry_available"] == flag]) for m in METHODS} for flag in (True, False)},
        **(dict(reference_excluded_indices={s["name"]: sorted(reference_exclusions(plan, s)) for s in plan["sequences"]})
           if plan["contract"] == NAKE_CONTRACT else {}))
    write_json(a.output / "metrics.json", report)
    scores_by_key = {(r["sequence"], r["frame_index"], r["method"]): r for r in scored}
    panels = []
    known_diagnostics = {"blue_pen": [534], "milk": [432], "right_hand": [222]} if plan["contract"] == CONTRACT else {}
    for seq in plan["sequences"]:
        name = seq["name"]
        chosen = sorted(set(seq["render_indices"] + known_diagnostics.get(name, [])))
        for i in chosen:
            if i in reference_exclusions(plan, seq):
                continue
            folder = a.output / "visualizations" / f"{name}-{i:06d}"
            folder.mkdir(parents=True)
            rgb = Image.open(a.root / name / "rgb" / f"{i:06d}.png").convert("RGB")
            reference = Image.open(a.root / name / "right" / f"{i:06d}.png").convert("L")
            rgb.save(folder / "rgb.png"); reference.save(folder / "reference.png")
            tiles = [("RGB", rgb), ("Assisted right reference", reference)]
            for method in METHODS:
                row = records[name, i, method]
                image = Image.fromarray(decode(row["union_rle"]).astype(np.uint8) * 255)
                image.save(folder / f"{method}.png")
                tiles.append((f"{method} Dice={scores_by_key[name,i,method]['dice']:.4f}", image))
            canvas = Image.new("RGB", (1920, 285), "white")
            draw = ImageDraw.Draw(canvas)
            for col, (title, image) in enumerate(tiles):
                draw.text((col*320+4, 6), title, fill="black")
                canvas.paste(image.convert("RGB").resize((320, 240), Image.Resampling.NEAREST), (col*320, 45))
            canvas.save(folder / "comparison.png")
            panels.append((name, i, str((folder / "comparison.png").relative_to(a.output)),
                "预登记等间隔" if i in seq["render_indices"] else "先前已知错误/改善诊断，非随机样本"))
    protocol_text = (f"每条件完整推理 {sum(counts.values())} 帧；只对原始0::3的 {plan['scoring_frames']} 帧计分。参考为已查看的SAM3辅助标签，不是独立精标；没有训练或调阈值。"
        if plan["contract"] == CONTRACT else
        f"每条件完整推理 {sum(counts.values())} 帧；只对预登记原始0::3且参考有效的 {plan['scoring_frames']} 帧计分。未知/排除参考的原帧仍保留推理上下文，不当负样本。参考不是独立精标；没有训练或调阈值。")
    lines = ["# MANO box 四条件完整结果", "", "四条件共享同一原始模型、输入和MANO框。frame=同一次前向的Detector NMS/门控后、跟踪前输出；video=最终系统输出，不是单独图像API。",
        "", protocol_text,
        "", "| 条件 | mean Dice | 正参考Dice | 正参考边界IoU | 正参考全漏检 | 空参考有预测 |", "|---|---:|---:|---:|---:|---:|"]
    for m, r in report["overall"].items():
        lines.append(f"| {m} | {r['mean_dice']:.6f} | {r['positive_dice']:.6f} | {r['positive_boundary_iou_4px']:.6f} | {r['missed_positive']}/{r['positive_frames']} | {r['empty_false_positive']}/{r['empty_frames']} |")
    lines += ["", "## 各视频同口径结果", "", "| 视频 | frame_text | frame_box | video_text | video_box |", "|---|---:|---:|---:|---:|"]
    for name, result in report["per_sequence"].items():
        lines.append("| " + name + " | " + " | ".join(f"{result[m]['mean_dice']:.6f}" for m in METHODS) + " |")
    lines += ["", "## 解释边界", "", "先看frame_box−frame_text，再看video_box−video_text。后续视频阶段包含关联、门控、跟踪、重检测和后处理，不等同于只有memory这一变量；不可凭本表独断memory根因。逐帧、几何可用/缺失分组与对侧重叠诊断保留在原始记录和metrics.json。Detector实例编号是帧内编号，不能当跨帧身份；视频实例ID也未获独立身份标注认证。", "", "## 分离可视化", "", "下列等间隔图预先登记，另三个先前已知案例单独标记。未评分的上下文帧也可展示，但不加入0::3平均。原图/参考/四组预测均可单独打开。", ""]
    lines.extend(f"- [{name} 第{i}帧：{kind}]({path})" for name, i, path, kind in panels)
    with (a.output / "README.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for key in ("data-root", "model-source", "output"):
        prep.add_argument("--" + key, type=Path, required=True)
    prep.add_argument("--base-sha256", required=True)
    inference = sub.add_parser("run")
    for key in ("root", "model-source", "checkpoint", "output"):
        inference.add_argument("--" + key, type=Path, required=True)
    inference.add_argument("--mode", choices=("text", "box"), required=True)
    inference.add_argument("--engineering", action="store_true")
    inference.add_argument("--engineering-frames", type=int, default=96)
    inference.add_argument("--sequence", help="One exact registered video, complete from original frame zero; new process per video")
    inference.add_argument("--cpu-tracker-state", action="store_true",
        help="Explicit noninteractive forward-only CPU storage repair; no history deletion")
    inference.add_argument("--equivalent-to", type=Path,
        help="Original failed run whose saved prefix must match every RLE, ID and score exactly")
    inference.add_argument("--max-seconds", type=int, default=10800)
    cpu = sub.add_parser("audit")
    for key in ("root", "run", "output"):
        cpu.add_argument("--" + key, type=Path, required=True)
    comparison = sub.add_parser("compare")
    for key in ("root", "text-run", "box-run", "text-audit", "box-audit", "output"):
        comparison.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    elif args.command == "audit":
        audit(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

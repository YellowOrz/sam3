"""Read-only fixed RealSense test: actual natural VE versus a preselected delta.

This is intentionally separate from tuning validation. Never select epochs,
thresholds, or decoder candidates using test-reference overlap. Inference is
noninteractive; both masks and RGB are saved separately, including rejected
candidate masks. The caller owns GPU scheduling and the external time budget.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
import torch.nn.functional as F

from scripts import cached_ve_text_features as cached
from scripts import evaluate_bilateral_tokens as shared
from scripts import evaluate_nakehand_tokens as bilateral
from scripts import evaluate_ve_initialized_tokens as semantic
from scripts import prepare_realsense_test as preparation
from scripts import residual_ddp_checkpoint as checkpoint
from scripts import train_ve_initialized_tokens as initializer
from scripts.residual_ddp_validation import _boundary


FORMAT = "sam3-realsense-residual-test-evaluation-v1"
THRESHOLD = .5


def load_test(root):
    root = Path(root).resolve()
    paths = {name: root / name for name in ("READY.json", "manifest.json", "annotations.json", "frozen-plan.json")}
    documents = {name: json.loads(path.read_text()) for name, path in paths.items()}
    fingerprints = {str(path): shared.sha256(path) for path in paths.values()}
    ready, manifest, data, plan = (documents[name] for name in paths)
    if (ready.get("format") != preparation.FORMAT or ready.get("status") != "complete"
            or ready.get("dataset_role") != "external_test_only"
            or manifest.get("status") != "complete" or manifest.get("sources_unchanged") is not True):
        raise ValueError("Require completed RealSense TEST publication, never a validation alias")
    for key, name in (("manifest_sha256", "manifest.json"), ("annotations_sha256", "annotations.json"),
                      ("frozen_plan_sha256", "frozen-plan.json")):
        if ready[key] != fingerprints[str(paths[name])]:
            raise ValueError(f"Publication hash mismatch: {name}")
    plan_hash = ready["frozen_plan_sha256"]
    if (data.get("info", {}).get("dataset_role") != "external_test_only"
            or data["info"].get("frozen_plan_sha256") != plan_hash
            or manifest.get("frozen_plan_sha256") != plan_hash
            or plan.get("seed") != preparation.SEED
            or plan.get("samples_per_recording") != preparation.PER_RECORDING
            or plan.get("frame_counts") != preparation.FRAME_COUNTS
            or plan.get("selection_uses_predictions_or_mask_pixels") is not False):
        raise ValueError("Fixed test protocol changed")
    excluded = {name: {int(frame): reasons for frame, reasons in rows.items()}
                for name, rows in plan["exclusions"].items()}
    if preparation.select_indices(preparation.FRAME_COUNTS, excluded) != plan["selection"]:
        raise ValueError("Test selection is not the fixed seeded selection")
    if data.get("categories") != preparation.CATEGORIES:
        raise ValueError("Require exactly two anatomical side classes")
    images = sorted(data["images"], key=lambda row: int(row["id"]))
    if len(images) != 128 or [row["id"] for row in images] != list(range(1, 129)):
        raise ValueError("Require all 128 frozen test images exactly once")
    expected = [(name, frame) for name, frames in sorted(plan["selection"].items()) for frame in frames]
    if [(row["recording_id"], row["source_frame_index"]) for row in images] != expected:
        raise ValueError("COCO source identity differs from frozen test plan")
    output_rows = manifest["image_outputs"]
    outputs = {row["image_id"]: row for row in output_rows}
    if len(outputs) != len(output_rows) or set(outputs) != set(range(1, 129)):
        raise ValueError("Output manifest identity coverage differs")
    refs, annotations = {}, defaultdict(list)
    seen_ids = set()
    for row in data["annotations"]:
        if row["id"] in seen_ids or row["image_id"] not in outputs or row["category_id"] not in (1, 2):
            raise ValueError("Invalid/duplicate annotation identity")
        seen_ids.add(row["id"])
        annotations[row["image_id"]].append(row)
    for image in images:
        image_id = image["id"]
        if image.get("reference_provided") != {"left_hand": True, "right_hand": True}:
            raise ValueError("Missing side stream must not become a test negative")
        if (image["width"], image["height"]) != (640, 480) or image.get("source_dataset") != "realsense":
            raise ValueError("Invalid source dataset or image size")
        published = outputs[image_id]
        if (published["recording_id"], published["source_frame_index"]) != (
                image["recording_id"], image["source_frame_index"]):
            raise ValueError("Output file/source identity mismatch")
        refs[image_id] = {}
        files = published["files"]
        if files["rgb"]["path"] != image["file_name"]:
            raise ValueError("RGB manifest path differs from COCO")
        for key in ("rgb", "left_hand", "right_hand"):
            path = root / files[key]["path"]
            if (Path(files[key]["path"]).is_absolute() or ".." in Path(files[key]["path"]).parts
                    or root not in path.resolve().parents or path.is_symlink()
                    or shared.sha256(path) != files[key]["sha256"]):
                raise ValueError("Missing, relocated-unsafely or changed exported image")
            fingerprints[str(path)] = files[key]["sha256"]
            with Image.open(path) as decoded:
                array = np.asarray(decoded)
                if array.shape != ((480, 640, 3) if key == "rgb" else (480, 640)):
                    raise ValueError("Exported image has invalid dimensions/channels")
                if key != "rgb":
                    refs[image_id][key] = array > 0
        categories = [row["category_id"] for row in annotations[image_id]]
        if len(categories) != len(set(categories)):
            raise ValueError("Require one semantic union per side")
        for category, side in enumerate(shared.CLASS_NAMES, 1):
            matches = [row for row in annotations[image_id] if row["category_id"] == category]
            decoded = shared.decode_gt_mask(matches[0] if matches else None, 480, 640)
            if not np.array_equal(decoded, refs[image_id][side]):
                raise ValueError("RLE differs from unmodified source-side PNG positive union")
        expected_render = image["source_frame_index"] in plan["render_selection"][image["recording_id"]]
        if image.get("render_preselected") is not expected_render:
            raise ValueError("Visualization selection changed after test plan")
    return images, refs, plan, fingerprints


def load_residual(path, initial_cache, training_root, *, base_hash, tokenizer_hash):
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state.get("training_config", {})
    if (state.get("format") != checkpoint.FORMAT or config.get("base_sha256") != base_hash
            or config.get("tokenizer_sha256") != tokenizer_hash
            or config.get("initial_cache_file_sha256") != shared.sha256(initial_cache)):
        raise ValueError("Expected DDP output-delta checkpoint with matching original weights/cache/tokenizer")
    positions = config.get("residual_positions", "all")
    # Share the trainer's strict layout contract: historical v1 files default
    # to all positions; content files must explicitly bind mode/shape/count.
    shape = checkpoint.configured_delta_shape(config)
    mode = "content_delta" if positions == "content" else "zero_delta"
    parameter_count = int(np.prod(shape))
    initial = initializer.load_initial_cache(initial_cache, base_hash=base_hash, tokenizer_hash=tokenizer_hash,
                                             residual_positions=positions)
    annotation_path = Path(training_root) / "annotations.json"
    if shared.sha256(annotation_path) != config.get("annotations_sha256"):
        raise ValueError("Training annotations do not match checkpoint provenance")
    training = json.loads(annotation_path.read_text())
    image_ids = sorted(row["id"] for row in training["images"])
    if (len(image_ids) != config.get("dataset_size") or len(set(image_ids)) != len(image_ids)
            or checkpoint.canonical_hash(image_ids) != config.get("image_order_sha256")):
        raise ValueError("Training image identities changed")
    step = checkpoint.validate_resume(state, config, image_ids, initial.state_dict())
    if step < 1 or state.get("trainable_parameter_count") != parameter_count:
        raise ValueError("Require actual completed residual updates")
    encoder = semantic.cache_from_state(state["cache_state_dict"])
    if encoder.mode != mode or tuple(encoder.delta.shape) != tuple(shape):
        raise ValueError("Evaluated cache differs from trained residual positions")
    encoder.eval().requires_grad_(False)
    return encoder, {"checkpoint_sha256": shared.sha256(path), "progress": state["progress"],
        "training_config": config, "initial_cache_verified": True,
        "residual_positions": positions, "residual_mode": mode,
        "delta_shape": list(shape), "trained_parameter_count": parameter_count,
        "evaluation_encoder_frozen": True,
        "actual_training_identities_verified": True,
        "rank_cache_consistency_audit_present_and_verified": "rank_cache_audit" in state,
        "delta_l2_per_side": encoder.delta.detach().norm(dim=(1, 2)).tolist()}


def add_boundary(record, candidate, reference):
    record["candidate_boundary_iou_4px"] = None
    record["miss_zero_boundary_iou_4px"] = None
    if reference.any():
        pred, target = _boundary(candidate), _boundary(reference)
        value = int((pred & target).sum()) / int((pred | target).sum())
        record["candidate_boundary_iou_4px"] = value
        record["miss_zero_boundary_iou_4px"] = value if record["detected"] else 0.
    return record


def summarize(records):
    def group(rows):
        result = bilateral.aggregate(rows)
        for key in ("candidate_boundary_iou_4px", "miss_zero_boundary_iou_4px"):
            values = [row[key] for row in rows if row[key] is not None]
            result[key] = sum(values) / len(values) if values else None
        return result
    result = {"overall": group(records), "per_side": {}, "per_recording": {}}
    for side in shared.CLASS_NAMES:
        result["per_side"][side] = group([row for row in records if row["prompt_key"] == side])
    for name in sorted({row["recording_id"] for row in records}):
        result["per_recording"][name] = group([row for row in records if row["recording_id"] == name])
    macro = [row["present_mean_miss_zero_dice"] for row in result["per_recording"].values()]
    macro = [value for value in macro if value is not None]
    result["recording_macro_miss_zero_dice"] = sum(macro) / len(macro) if macro else None
    return result


def evaluate(model, label, prompts, dataset, images, refs, batch_size):
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api
    bilateral.validate_frozen_noninteractive_model(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Test must have no trainable parameters")
    records, masks = [], {}
    for number, indices in enumerate(shared.batches(list(range(len(images))), batch_size), 1):
        samples = [dataset[index] for index in indices]
        batch = collate_fn_api(samples, dict_key="test", with_seg_masks=True)["test"]
        bilateral.validate_batch_identity(batch, indices, images)
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        batch.find_text_batch = list(prompts)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)[0]
        for key in ("pred_logits", "presence_logit_dec", "pred_masks"):
            if not bool(torch.isfinite(output[key]).all()):
                raise RuntimeError("Nonfinite inference output")
        classes = output["pred_logits"].float().sigmoid().squeeze(-1)
        presence = output["presence_logit_dec"].float().sigmoid().reshape(len(classes), -1)[:, 0]
        confidence = classes * presence[:, None]
        best = confidence.argmax(dim=1)  # NO reference-dependent candidate selection.
        stage = batch.find_inputs[0]
        for row in range(len(best)):
            index = indices[int(stage.img_ids[row])]
            image = images[index]
            prompt_index = int(stage.text_ids[row])
            side, other = shared.CLASS_NAMES[prompt_index], shared.CLASS_NAMES[1 - prompt_index]
            logits = F.interpolate(output["pred_masks"][row, best[row]][None, None].float(),
                size=(image["height"], image["width"]), mode="bilinear", align_corners=False)[0, 0]
            candidate = logits.sigmoid().cpu().numpy() >= THRESHOLD
            reference = refs[image["id"]][side]
            rle = mask_utils.encode(np.asfortranarray(candidate.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            record = {"model": label, "dataset_role": "external_test_only", "dataset_index": index,
                "image_id": image["id"], "identity_verified": True, "file_name": image["file_name"],
                "recording_id": image["recording_id"], "source_frame_index": image["source_frame_index"],
                "source_mapping": image["source_mapping"], "prompt_key": side,
                "prompt_text": cached.NATURAL_PROMPTS[prompt_index], "primary_test": True,
                "diagnostic_ids": [], "reference_description": preparation.REFERENCE_DESCRIPTION,
                "top_class_probability": float(classes[row, best[row]]),
                "presence_probability": float(presence[row]), "selected_decoder_query": int(best[row]),
                "prediction_rle": rle, "detections_above_threshold": int((confidence[row] >= THRESHOLD).sum()),
                **bilateral.measure_query(candidate, reference, refs[image["id"]][other], float(confidence[row, best[row]]))}
            records.append(add_boundary(record, candidate, reference))
            if image["render_preselected"]:
                masks[(label, index, side)] = candidate
        if number == 1 or number % 20 == 0:
            print(f"{label}: {min(number * batch_size, len(images))}/{len(images)} fixed test images", flush=True)
    expected = {(image["id"], side) for image in images for side in shared.CLASS_NAMES}
    observed = [(row["image_id"], row["prompt_key"]) for row in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise RuntimeError("Test image/query coverage mismatch")
    return records, masks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path,
                        default=Path(__file__).resolve().parents[1] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--mode", choices=("ve", "residual", "both"), default="both")
    parser.add_argument("--delta-checkpoint", type=Path)
    parser.add_argument("--initial-cache", type=Path)
    parser.add_argument("--training-data-root", type=Path)
    parser.add_argument("--checkpoint-selection-note", help="Pre-test choice e.g. DexYCB-val best; never test-selected")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.6)
    parser.add_argument("--preflight-only", action="store_true", help="Verify dataset/checkpoint on CPU, no output writes")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Require positive batch size and allocator fraction within (0,1]")
    if args.mode != "ve" and not all((args.delta_checkpoint, args.initial_cache, args.training_data_root,
                                      args.checkpoint_selection_note)):
        parser.error("Residual mode requires checkpoint, original cache, training data root and pre-test selection note")
    images, refs, plan, fingerprints = load_test(args.data_root)
    base_hash, tokenizer_hash = shared.sha256(args.base_checkpoint), shared.sha256(args.tokenizer_path)
    fingerprints.update({str(args.base_checkpoint): base_hash, str(args.tokenizer_path): tokenizer_hash})
    encoder, metadata = None, None
    if args.mode != "ve":
        for path in (args.delta_checkpoint, args.initial_cache, args.training_data_root / "annotations.json"):
            fingerprints[str(path)] = shared.sha256(path)
        encoder, metadata = load_residual(args.delta_checkpoint, args.initial_cache, args.training_data_root,
                                         base_hash=base_hash, tokenizer_hash=tokenizer_hash)
    summary = {"format": FORMAT, "status": "preflight_only" if args.preflight_only else "running",
        "dataset_role": "external_test_only", "images": len(images), "queries_per_model": 2 * len(images),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_fingerprints": fingerprints,
        "reference_description": preparation.REFERENCE_DESCRIPTION, "protocol": plan,
        "checkpoint_selection_note": args.checkpoint_selection_note,
        "detection_threshold": THRESHOLD, "mask_threshold": THRESHOLD, "boundary_width_original_pixels": 4,
        "prediction_selection": "argmax(sigmoid(class)*sigmoid(presence)); no reference-dependent selection",
        "precision": "BF16 autocast; FP32 logits for sigmoid/interpolation; FP32 delta",
        "batch_size": args.batch_size, "models": {}, "metrics": {},
        "limitations": ["Auxiliary-reference agreement, not independent human GT accuracy",
                        "Test results must not select epochs, LR or thresholds",
                        "Restricted 8-recording subset; no whole-dataset or cross-subject accuracy claim",
                        "Opposite-overlap-dominant is a proxy, not an anatomical identity classifier"]}
    if args.preflight_only:
        print(json.dumps({"status": summary["status"], "images": len(images), "residual": metadata}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for inference; use --preflight-only for CPU verification")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(123)
    code_dir = args.output_dir / "code-snapshot"
    code_dir.mkdir()
    dependencies = {Path(__file__).resolve(), Path(inspect.getfile(_boundary)).resolve(),
        *(Path(inspect.getfile(module)).resolve() for module in
        (cached, shared, bilateral, semantic, preparation, checkpoint, initializer))}
    dependencies.update((Path(__file__).resolve().parents[1] / "sam3").rglob("*.py"))
    summary["implementation_sha256"] = {str(path): shared.sha256(path) for path in sorted(dependencies)}
    for path in sorted(dependencies):
        if path.parent.name == "scripts":
            shutil.copy2(path, code_dir / path.name)
    shared.atomic_write_json(args.output_dir / "progress.json", summary)
    started = time.monotonic()
    try:
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(checkpoint_path=str(args.base_checkpoint), bpe_path=str(args.tokenizer_path),
            load_from_HF=False, device="cuda", eval_mode=True, enable_segmentation=True,
            enable_inst_interactivity=False, text_encoder_type="ve")
        model.eval().requires_grad_(False)
        model.register_forward_hook(semantic.assert_finite_model_outputs)
        versions = [(parameter, parameter._version) for parameter in model.parameters()]
        raw = shared.make_dataset(args.data_root)
        if len(raw) != len(images):
            raise RuntimeError("Loader image count differs from test publication")
        dataset = semantic.IdentityCheckedDataset(raw, images)
        labels = (["ve-natural"] if args.mode in ("ve", "both") else [])
        labels += (["residual-preselected"] if args.mode in ("residual", "both") else [])
        records, masks = [], {}
        for label in labels:
            if label == "residual-preselected":
                original = cached.install_cached_ve_text_encoder(model, encoder.to("cuda"))
                cached.set_cached_ve_training_mode(model, train_delta=False)
                prompts = shared.CLASS_NAMES
                summary["models"][label] = metadata
            else:
                prompts = cached.NATURAL_PROMPTS
                summary["models"][label] = {"source": "actual original text encoder, natural prompts, no cached replacement"}
            before = len(dataset.observed_indices)
            rows, predictions = evaluate(model, label, prompts, dataset, images, refs, args.batch_size)
            if dataset.observed_indices[before:] != list(range(len(images))):
                raise RuntimeError("Actual loader order differs from fixed complete test selection")
            records.extend(rows)
            masks.update(predictions)
            shared.atomic_write_json(args.output_dir / f"records-{label}.json", rows)
            summary["metrics"][label] = summarize(rows)
            shared.atomic_write_json(args.output_dir / "progress.json", summary)
            if label == "residual-preselected":
                if initializer.cache_fingerprint(encoder.state_dict()) != initializer.cache_fingerprint(
                        torch.load(args.delta_checkpoint, map_location="cpu", weights_only=True)["cache_state_dict"]):
                    raise RuntimeError("Residual cache changed during inference")
                cached.restore_original_ve_text_encoder(model, original)
        if any(parameter._version != version for parameter, version in versions):
            raise RuntimeError("Original SAM3 parameters changed during inference")
        render = [index for index, image in enumerate(images) if image["render_preselected"]]
        summary["visualizations"] = bilateral.render_results(data_root=args.data_root, output_dir=args.output_dir / "visuals",
            images=images, references=refs, render_indices=render, records=records, masks=masks, labels=labels)
        for path, digest in {**fingerprints, **summary["implementation_sha256"]}.items():
            if shared.sha256(Path(path)) != digest:
                raise RuntimeError(f"Input/code changed during evaluation: {path}")
        summary.update(status="complete", elapsed_seconds=time.monotonic() - started,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), actual_complete_query_coverage_verified=True)
        shared.atomic_write_json(args.output_dir / "summary.json", summary)
        print(json.dumps({"status": "complete", "output": str(args.output_dir), "metrics": summary["metrics"]}, indent=2))
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started)
        shared.atomic_write_json(args.output_dir / "failure.json", summary)
        raise


if __name__ == "__main__":
    main()

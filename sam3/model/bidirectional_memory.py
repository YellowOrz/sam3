"""Inference-only memory export and joint reading for the standard SAM3 tracker.

No parameters are added and no source bank is modified during final decoding.
Imported lazily by the opt-in research path; ordinary SAM3 inference is unchanged.
"""

import math
from pathlib import Path

import torch


def export_memory_frame(inference_state, frame_idx):
    """Snapshot final encoded memory before trimming or hotstart output delays."""
    directory = Path(inference_state["memory_export_directory"])
    records = {}
    for state in inference_state["tracker_inference_states"]:
        outputs = state["output_dict"]
        conditioning = frame_idx in outputs["cond_frame_outputs"]
        output = outputs[
            "cond_frame_outputs" if conditioning else "non_cond_frame_outputs"
        ].get(frame_idx)
        if output is None or output.get("maskmem_features") is None:
            continue
        for obj_id, obj_idx in state["obj_id_to_idx"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            payload = {
                "maskmem_features": output["maskmem_features"][obj_slice]
                .detach()
                .cpu(),
                "maskmem_pos_enc": [
                    value[obj_slice].detach().cpu()
                    for value in output["maskmem_pos_enc"]
                ],
                "obj_ptr": output["obj_ptr"][obj_slice].detach().cpu(),
            }
            iou = output.get("iou_score")
            quality = float(iou[obj_slice].max()) if iou is not None else None
            if quality is not None and not math.isfinite(quality):
                quality = None
            presence = float(output["object_score_logits"][obj_slice].flatten()[0])
            path = directory / f"{frame_idx:06d}_{int(obj_id)}.pt"
            # torch.save completes before the next frame can mutate shared slices.
            torch.save(payload, path)
            records[int(obj_id)] = {
                "frame_index": frame_idx,
                "object_id": int(obj_id),
                "conditioning": conditioning,
                "quality": quality,
                "quality_source": "predicted_iou" if iou is not None else "unavailable",
                "presence_logit": presence if math.isfinite(presence) else None,
                "path": str(path),
                "spatial_tokens": payload["maskmem_features"].shape[-2]
                * payload["maskmem_features"].shape[-1],
                "pointer_values": payload["obj_ptr"].numel(),
            }
    inference_state["memory_export_records"][frame_idx] = records


@torch.inference_mode()
def read_memory(
    tracker, frame_idx, vision_feats, vision_pos, feat_sizes, spatial, pointers
):
    """Joint attention using existing distance encodings and a single decoder.

    Distances are unsigned in this zero-training baseline: no new direction
    embedding is introduced. Spatial distances saturate at existing slot limits.
    All pointer tokens remain at the end, as required by the RoPE exclusion rule.
    """
    if tracker.training:
        raise ValueError("Bidirectional memory decoding requires an eval-mode tracker")
    if tracker.num_maskmem < 2:
        raise ValueError("Bidirectional reading requires at least two memory slots")
    device = vision_feats[-1].device
    batch = vision_feats[-1].shape[1]
    if batch != 1:
        raise ValueError("Bidirectional memory decoding currently supports one object")
    channels = tracker.hidden_dim
    height, width = feat_sizes[-1]
    loaded = {}

    def load(entry):
        if entry["frame_index"] == frame_idx:
            raise ValueError("A frame cannot read its own intermediate memory")
        if entry["direction"] not in {"F", "B"} or (
            (entry["direction"] == "F") != (entry["frame_index"] < frame_idx)
        ):
            raise ValueError("Memory direction does not match its source time")
        key = entry["path"]
        if key not in loaded:
            loaded[key] = torch.load(key, map_location=device, weights_only=True)
        return loaded[key]

    tokens, positions = [], []
    for entry in spatial:
        payload = load(entry)
        memory = payload["maskmem_features"]
        if memory.shape[:2] != (1, tracker.mem_dim):
            raise ValueError("Memory shape does not match the current tracker")
        if memory.shape[-2:] != (height, width):
            raise ValueError(
                "Memory spatial resolution does not match the current frame"
            )
        tokens.append(memory.flatten(2).permute(2, 0, 1))
        pos = payload["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
        distance = abs(frame_idx - entry["frame_index"])
        if entry["conditioning"]:
            slot = tracker.num_maskmem - 1
            embedding = getattr(tracker, "cond_frame_spatial_embedding", None)
            if embedding is not None:
                pos = pos + embedding
        else:
            slot = min(distance - 1, tracker.num_maskmem - 2)
        positions.append(pos + tracker.maskmem_tpos_enc[slot])

    num_pointer_tokens = 0
    if pointers:
        ptr = torch.stack([load(entry)["obj_ptr"] for entry in pointers], dim=0)
        embedding = getattr(tracker, "cond_frame_obj_ptr_embedding", None)
        if embedding is not None:
            conditioning = torch.tensor(
                [entry["conditioning"] for entry in pointers], device=device
            )
            ptr = ptr + embedding * conditioning[:, None, None].float()
        # Cap to the pretrained normalization range rather than invent new codes.
        max_distance = max(1, tracker.max_obj_ptrs_in_encoder - 1)
        distances = [
            min(abs(frame_idx - entry["frame_index"]), max_distance)
            for entry in pointers
        ]
        pos = tracker._get_tpos_enc(
            distances, max_abs_pos=tracker.max_obj_ptrs_in_encoder, device=device
        ).unsqueeze(1)
        if tracker.mem_dim < channels:
            splits = channels // tracker.mem_dim
            ptr = ptr.reshape(-1, batch, splits, tracker.mem_dim)
            ptr = ptr.permute(0, 2, 1, 3).flatten(0, 1)
            pos = pos.repeat_interleave(splits, dim=0)
        tokens.append(ptr)
        positions.append(pos)
        num_pointer_tokens = ptr.shape[0]

    if not spatial:
        raise ValueError("Joint memory decoding requires spatial target evidence")
    encoded = tracker.transformer.encoder(
        src=vision_feats[-1:],
        src_key_padding_mask=[None],
        src_pos=vision_pos[-1:],
        prompt=torch.cat(tokens, dim=0),
        prompt_pos=torch.cat(positions, dim=0),
        prompt_key_padding_mask=None,
        feat_sizes=feat_sizes[-1:],
        num_obj_ptr_tokens=num_pointer_tokens,
    )["memory"]
    conditioned = encoded.permute(1, 2, 0).view(batch, channels, height, width)
    high_res = [
        value.permute(1, 2, 0).view(batch, value.size(2), *size)
        for value, size in zip(vision_feats[:-1], feat_sizes[:-1])
    ] or None
    outputs = tracker._forward_sam_heads(
        backbone_features=conditioned,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=high_res,
        multimask_output=tracker._use_multimask(False, None),
    )
    _, _, ious, low_res, _, _, presence = outputs
    if not all(
        torch.isfinite(value).all().item() for value in (low_res, ious, presence)
    ):
        raise RuntimeError("Non-finite output from bidirectional memory decoding")
    return {
        "logits": low_res,
        "predicted_iou": float(ious.max()),
        "presence_logit": float(presence.flatten()[0]),
        "spatial_tokens": sum(entry["spatial_tokens"] for entry in spatial),
        "pointer_tokens": num_pointer_tokens,
    }

"""Segmentation backend v1: headless video prediction and existing-mask editing.

The request/stream API is shared with Sam3VideoPredictor. No desktop UI,
checkpoint interception, global progress patches, or business identities.
"""

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from sam3.video_io import normalize_output_arrays

API_VERSION = 1
MODEL = "sam3"
CAPABILITIES = frozenset(
    {"text", "text_batch", "geometry", "points", "mask_tracking", "chunking"}
)


def create_predictor(
    *,
    checkpoint_path,
    device="cuda:0",
    compile=False,
    async_loading_frames=True,
    progress=False,
    clip_overseg=False,
):
    import torch
    from sam3 import build_sam3_predictor

    device = torch.device(device)
    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device unavailable: {device}; no fallback")
    torch.cuda.set_device(device)
    return build_sam3_predictor(
        version="sam3",
        checkpoint_path=checkpoint_path,
        compile=compile,
        async_loading_frames=async_loading_frames,
        progress=progress,
        clip_overseg=clip_overseg,
    )


def create_mask_editor(predictor, state, *, frame_path):
    from sam3.mask_editor import LoadedMaskPredictor

    return LoadedMaskPredictor(state, lambda: predictor, frame_path=frame_path)


def _segment_clip(
    predictor, video, info, frames, output, prompt, offset, geometry, progress
):
    from sam3.video_outputs import normalize_frame_outputs, write_interactive_outputs

    sid = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frames),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
        }
    )["session_id"]
    state = SimpleNamespace(
        cache={},
        probability_cache={},
        tracking_probability_cache={},
        frame_count=info.frame_count,
        video_info=info,
        video_path=video,
        frame_dir=frames,
        output_dir=output,
        frame_offset=offset,
        prompt=prompt,
        version="sam3",
        clip_overseg=False,
        confirmed_points=[],
        events=[],
        initial_propagation_mode="forward",
    )

    def cache(frame, outputs):
        if type(frame) is not int or not 0 <= frame < info.frame_count:
            raise ValueError("backend returned an invalid frame index")
        masks, detection, tracking = normalize_frame_outputs(outputs)
        if any(mask.shape != (info.height, info.width) for mask in masks.values()):
            raise ValueError("backend masks must use original image coordinates")
        (
            state.cache[frame],
            state.probability_cache[frame],
            state.tracking_probability_cache[frame],
        ) = masks, detection, tracking
        progress(len(state.cache))

    try:
        request = {
            "type": "add_geometry_prompts" if geometry else "add_prompt",
            "session_id": sid,
            "frame_index": 0,
            "text": prompt,
        }
        if geometry:
            request["geometry_prompts"] = geometry
        initial = predictor.handle_request(request)
        cache(0, initial.get("outputs"))
        for result in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": sid,
                "start_frame_index": 0,
                "max_frame_num_to_track": info.frame_count - 1,
                "propagation_direction": "forward",
            }
        ):
            cache(result["frame_index"], result["outputs"])
        if len(state.cache) != info.frame_count:
            raise ValueError("backend returned an incomplete video")
        write_interactive_outputs(state)
    finally:
        predictor.handle_request({"type": "close_session", "session_id": sid})


def segment_video(
    predictor,
    *,
    video,
    prompts,
    output_dirs,
    geometry_prompts=None,
    chunk_frames=0,
    progress=lambda *_: None,
    prompt_done=lambda *_: None,
):
    import torch
    from sam3.video_outputs import (
        extract_frames,
        frame_ranges,
        make_chunk_frame_dir,
        merge_chunk_outputs,
        probe_video,
        VideoInfo,
    )

    if (
        not prompts
        or len(prompts) != len(output_dirs)
        or len(set(prompts)) != len(prompts)
    ):
        raise ValueError("one distinct output directory is required per prompt")
    outputs = [Path(p).resolve() for p in output_dirs]
    if len(set(outputs)) != len(outputs) or any(
        p.exists() and any(p.iterdir()) for p in outputs
    ):
        raise ValueError("automatic output directories must be distinct and empty")
    if type(chunk_frames) is not int or chunk_frames < 0:
        raise ValueError("invalid chunk size")
    geometry_prompts = geometry_prompts or [None] * len(prompts)
    if len(geometry_prompts) != len(prompts) or chunk_frames and any(geometry_prompts):
        raise ValueError("geometry must align with prompts and cannot be chunked")
    video = Path(video).resolve()
    original = probe_video(video)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with tempfile.TemporaryDirectory(
        prefix="sam3_video_"
    ) as temporary, torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        workspace = Path(temporary)
        frames = workspace / "frames"
        count = extract_frames(video, frames)
        if count <= 0:
            raise ValueError("video contains no frames")
        info = VideoInfo(original.width, original.height, count, original.fps)
        total = count * len(prompts)
        for query, (prompt, output, geometry) in enumerate(
            zip(prompts, outputs, geometry_prompts)
        ):
            started = time.monotonic()
            if not chunk_frames:
                _segment_clip(
                    predictor,
                    video,
                    info,
                    frames,
                    output,
                    prompt,
                    0,
                    geometry,
                    lambda done: progress(query * count + done, total),
                )
            else:
                chunks = []
                for index, (start, end) in enumerate(frame_ranges(count, chunk_frames)):
                    chunk_dir = workspace / f"q{query}-c{index}"
                    chunk_frames_dir, chunk_output = (
                        chunk_dir / "frames",
                        chunk_dir / "output",
                    )
                    make_chunk_frame_dir(frames, chunk_frames_dir, start, end)
                    chunk_info = VideoInfo(
                        info.width, info.height, end - start, info.fps
                    )
                    _segment_clip(
                        predictor,
                        video,
                        chunk_info,
                        chunk_frames_dir,
                        chunk_output,
                        prompt,
                        start,
                        None,
                        lambda done: progress(query * count + start + done, total),
                    )
                    chunks.append((start, end, chunk_output))
                merge_chunk_outputs(
                    chunks,
                    output,
                    video,
                    info,
                    prompt,
                    "sam3",
                    chunk_frames,
                    frames,
                    confirmed=True,
                    clip_overseg=False,
                )
            prompt_done(prompt, output, count, time.monotonic() - started)
        return {"frames": count, "prompt_count": len(prompts)}

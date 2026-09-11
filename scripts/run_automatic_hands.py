"""Run left-hand and right-hand SAM3 prompts without user text input."""

import argparse
import os

import numpy as np

from sam3.model.class_token_checkpoint import (
    copy_class_tokens,
    load_class_token_checkpoint,
)
from sam3.model_builder import build_sam3_predictor


CLASS_COLORS = {
    "left_hand": np.array([255, 80, 80], dtype=np.uint8),
    "right_hand": np.array([80, 160, 255], dtype=np.uint8),
}


def merge_instance_masks(masks):
    masks = np.asarray(masks, dtype=bool)
    if masks.size == 0:
        return None
    masks = masks.reshape(-1, *masks.shape[-2:])
    return masks.any(axis=0)


def propagate_one_class(predictor, video_path, class_name):
    response = predictor.handle_request(
        request={"type": "start_session", "resource_path": video_path}
    )
    session_id = response["session_id"]
    try:
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": class_name,
            }
        )
        outputs = {}
        for response in predictor.handle_stream_request(
            request={"type": "propagate_in_video", "session_id": session_id}
        ):
            outputs[response["frame_index"]] = response["outputs"]
        return outputs
    finally:
        predictor.handle_request(
            request={"type": "close_session", "session_id": session_id}
        )


def draw_outputs(frame, outputs_by_class):
    import cv2

    result = frame.copy()
    for class_name, frame_outputs in outputs_by_class.items():
        output = frame_outputs.get("current")
        if output is None:
            continue
        mask = merge_instance_masks(output.get("out_binary_masks", []))
        if mask is None:
            continue
        color = CLASS_COLORS[class_name][::-1]
        result[mask] = (0.55 * result[mask] + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, tuple(int(x) for x in color), 2)
        cv2.putText(
            result,
            class_name,
            (20, 40 if class_name == "left_hand" else 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            tuple(int(x) for x in color),
            2,
            cv2.LINE_AA,
        )
    return result


def build_automatic_hands_predictor(
    checkpoint, *, token_checkpoint=None, tokens_per_class=None
):
    """Build the base video model and optionally overlay trained hand tokens."""
    if tokens_per_class is not None and (
        type(tokens_per_class) is not int or tokens_per_class < 1
    ):
        raise ValueError("--tokens-per-class must be a positive integer")
    tokens = None
    if token_checkpoint is not None:
        tokens = load_class_token_checkpoint(
            token_checkpoint, tokens_per_class=tokens_per_class
        )
        tokens_per_class = int(tokens.shape[1])
    elif tokens_per_class is None:
        tokens_per_class = 1

    predictor = build_sam3_predictor(
        version="sam3",
        checkpoint_path=checkpoint,
        text_encoder_type="learnable_class",
        tokens_per_class=tokens_per_class,
        use_fa3=False,
        async_loading_frames=False,
    )
    try:
        if tokens is not None:
            if getattr(predictor, "world_size", 1) != 1:
                raise ValueError("Token overlay currently requires a single-GPU predictor")
            encoder = predictor.model.detector.backbone.language_backbone
            copy_class_tokens(encoder, tokens)
        return predictor
    except BaseException:
        predictor.shutdown()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True, help="Base/full SAM3 checkpoint")
    parser.add_argument(
        "--token-checkpoint",
        help="Optional token-only training checkpoint; infer K from its class_tokens",
    )
    parser.add_argument(
        "--tokens-per-class",
        type=int,
        default=None,
        help="K (default: token checkpoint K, or 1 without a token checkpoint)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    import cv2

    predictor = build_automatic_hands_predictor(
        args.checkpoint,
        token_checkpoint=args.token_checkpoint,
        tokens_per_class=args.tokens_per_class,
    )
    outputs_by_class = {}
    try:
        for class_name in ("left_hand", "right_hand"):
            outputs_by_class[class_name] = propagate_one_class(
                predictor, args.video, class_name
            )

        cap = cv2.VideoCapture(args.video)
        writer = None
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot open video: {args.video}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            writer = cv2.VideoWriter(
                args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise OSError(f"Cannot create output video: {args.out}")
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                current = {
                    name: {"current": results.get(frame_index)}
                    for name, results in outputs_by_class.items()
                }
                writer.write(draw_outputs(frame, current))
                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        print(f"完成：{frame_index} 帧 → {args.out}")
    finally:
        predictor.shutdown()


if __name__ == "__main__":
    main()

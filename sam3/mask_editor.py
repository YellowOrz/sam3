"""Public tracker-only editor for existing masks (JPEG or lossless PNG frames)."""

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sam3.video_io import as_numpy


class LoadedMaskPredictor:
    """Tracker-only editing; never call the detector's video prefetch path."""

    def __init__(self, app: Any, get_predictor: Any, *, frame_path=None):
        self.app = app
        self.frame_path = frame_path or (
            lambda frame: app.frame_dir / f"{frame:05d}.jpg"
        )
        self.get_predictor = get_predictor
        self.model = None
        self.state = None
        self.features: Dict[int, Any] = {}
        self.seed_frame: Optional[int] = None
        self.seed_masks: Dict[int, np.ndarray] = {}

    def reset(self, frame: int, masks: Dict[int, np.ndarray]) -> None:
        self.state = None
        self.features.clear()
        self.seed_frame = frame
        self.seed_masks = {obj: mask.copy() for obj, mask in masks.items()}
        for point in self.app.confirmed_points:
            if point.frame_index == frame:
                self.seed_masks.setdefault(
                    point.obj_id,
                    np.zeros(
                        (self.app.video_info.height, self.app.video_info.width),
                        dtype=bool,
                    ),
                )

    @torch.inference_mode()
    def prepare_features(self, frame: int) -> None:
        if self.model is None:
            self.model = self.get_predictor().model
        if self.model.device.type == "cuda":
            torch.cuda.set_device(self.model.device)
        if frame in self.features:
            return
        from sam3.model.utils.sam2_utils import _load_img_as_tensor

        image, _, _ = _load_img_as_tensor(
            str(self.frame_path(frame)), self.model.image_size
        )
        image = image.to(self.model.device, dtype=torch.float32)
        mean = image.new_tensor(self.model.image_mean)[:, None, None]
        std = image.new_tensor(self.model.image_std)[:, None, None]
        image = ((image - mean) / std).unsqueeze(0)
        tracker = self.model.tracker
        if self.app.version == "sam3":
            backbone = self.model.detector.backbone.forward_image(image)[
                "sam2_backbone_out"
            ]
            backbone["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(
                backbone["backbone_fpn"][0]
            )
            backbone["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(
                backbone["backbone_fpn"][1]
            )
        else:
            from sam3.model.data_misc import NestedTensor

            backbone = self.model.detector.backbone.forward_image(
                NestedTensor(image, None),
                need_sam3_out=False,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            for key, decoder in (
                ("interactive", tracker.interactive_sam_mask_decoder),
                ("sam2_backbone_out", tracker.sam_mask_decoder),
            ):
                fpn = backbone[key]["backbone_fpn"]
                fpn[0].tensors = decoder.conv_s0(fpn[0].tensors)
                fpn[1].tensors = decoder.conv_s1(fpn[1].tensors)
        self.features.clear()
        self.features[frame] = (image, backbone)

    @torch.inference_mode()
    def initialize(self) -> None:
        frame = int(self.seed_frame)
        self.prepare_features(frame)
        tracker = self.model.tracker
        self.state = tracker.init_state(
            video_height=self.app.video_info.height,
            video_width=self.app.video_info.width,
            num_frames=self.app.frame_count,
            cached_features=self.features,
        )
        ids = sorted(self.seed_masks)
        if not ids:
            return
        masks = torch.as_tensor(
            np.stack([self.seed_masks[obj] for obj in ids]), device=self.model.device
        )
        if self.app.version == "sam3":
            for obj, mask in zip(ids, masks):
                tracker.add_new_mask(self.state, frame, obj, mask)
        else:
            tracker.add_new_masks(self.state, frame, ids, masks.float())
        tracker.propagate_in_video_preflight(self.state, run_mem_encoder=True)
        # Consolidate the seed into tracked outputs so clicks refine its mask,
        # rather than treating the first click as a fresh segmentation.
        self.track_frame(frame, reverse=False)
        if self.app.version == "sam3.1":
            self.state["user_refined_frames_per_obj"] = {obj: {frame} for obj in ids}

    @torch.inference_mode()
    def track_frame(self, frame: int, reverse: bool) -> Dict[str, Any]:
        self.prepare_features(frame)
        outputs = list(
            self.model.tracker.propagate_in_video(
                self.state,
                start_frame_idx=frame,
                max_frame_num_to_track=0,
                reverse=reverse,
                tqdm_disable=True,
                run_mem_encoder=True,
            )
        )
        if len(outputs) != 1 or int(outputs[0][0]) != frame:
            raise RuntimeError("tracker did not respect the requested single frame")
        _, ids, _, masks, scores = outputs[0]
        tracked = dict(zip(ids, as_numpy(masks[:, 0] > 0)))
        untracked = {
            obj: mask
            for obj, mask in self.app.cache[frame].items()
            if obj not in tracked
        }
        tracked.update(untracked)
        result = {
            "out_obj_ids": np.asarray(list(tracked)),
            "out_binary_masks": np.stack(list(tracked.values())),
        }
        score_by_id = dict(zip(ids, as_numpy(scores.float().sigmoid()).reshape(-1)))
        result["out_tracker_probs"] = np.asarray(
            [score_by_id.get(obj) for obj in tracked]
        )
        return result

    @torch.inference_mode()
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        kind = request["type"]
        if kind == "add_prompt":
            frame = int(request["frame_index"])
            obj_id = int(request["obj_id"])
            if self.state is not None and obj_id not in self.state["obj_ids"]:
                self.reset(frame, self.app.cache[frame])
            if self.state is None:
                # Register a new object before preflight locks the object slots.
                self.seed_masks.setdefault(
                    obj_id,
                    np.zeros(
                        (self.app.video_info.height, self.app.video_info.width),
                        dtype=bool,
                    ),
                )
                self.initialize()
            self.prepare_features(frame)
            tracker = self.model.tracker
            points = torch.tensor(request["points"], dtype=torch.float32)
            labels = torch.tensor(request["point_labels"], dtype=torch.int32)
            out = tracker.add_new_points_or_box(
                self.state,
                frame,
                request["obj_id"],
                points=points,
                labels=labels,
                rel_coordinates=True,
                clear_old_points=True,
            )
            ids, masks = out[1], out[3]
            return {
                "outputs": {
                    "out_obj_ids": np.asarray(ids),
                    "out_binary_masks": as_numpy(masks[:, 0] > 0),
                }
            }
        if kind in ("remove_object", "clear_checkpoints"):
            self.state = None
            self.features.clear()
        # The loaded branch rebuilds only its seed frame, not old checkpoints.
        if kind not in {
            "remove_object",
            "clear_checkpoints",
            "cancel_propagation",
            "save_checkpoint",
            "update_frame_checkpoints",
            "limit_checkpoint_spans",
        }:
            raise ValueError(f"unsupported loaded-result request: {kind}")
        return {"is_success": True, "created": False, "updated": 0}

    def handle_stream_request(
        self, request: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        start = int(request["start_frame_index"])
        limit = int(request["max_frame_num_to_track"])
        reverse = request["propagation_direction"] == "backward"
        end = start + (-limit if reverse else limit)
        if (
            limit < 0
            or not 0 <= min(start, end) <= max(start, end) < self.app.frame_count
        ):
            raise ValueError("invalid propagation interval")
        if limit == 0:
            return
        self.reset(start, self.app.cache[start])
        self.initialize()
        step = -1 if reverse else 1
        end = start + step * limit
        for frame in range(start + step, end + step, step):
            if frame in self.app.loaded_anchor_frames:
                self.reset(frame, self.app.cache[frame])
                self.initialize()
                # Keep the manually confirmed masks exactly, including the seed.
                masks = self.app.cache[frame]
                yield {
                    "frame_index": frame,
                    "outputs": {
                        "out_obj_ids": np.asarray(sorted(masks)),
                        "out_binary_masks": np.stack(
                            [masks[obj] for obj in sorted(masks)]
                        )
                        if masks
                        else np.empty(
                            (0, self.app.video_info.height, self.app.video_info.width),
                            dtype=bool,
                        ),
                    },
                }
                continue
            if not self.state["obj_ids"]:
                raise RuntimeError(
                    "no objects at propagation seed; add a positive point first"
                )
            yield {"frame_index": frame, "outputs": self.track_frame(frame, reverse)}

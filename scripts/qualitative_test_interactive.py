#!/usr/bin/env python3
"""交互式 SAM 3/3.1 视频分割工具。

使用方法
========

启动示例::

    uv run python scripts/qualitative_test_interactive.py \
        --version sam3 \
        --checkpoint ~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt \
        --video /path/to/color.mp4 \
        --text_prompt "human hand" \
        --device cuda:0 \
        --propagation-direction forward \
        --checkpoint-interval 20 \
        --output-dir ./outputs/interactive/example

如果输出目录中已经存在结果，请增加 ``--overwrite``。
``--propagation-direction`` 默认为 ``both``，会从编辑帧向前、向后传播；设为
``forward`` 时只向视频结尾传播，并保留编辑帧之前已有的结果。

交互操作
--------

* 程序会从第 0 帧应用文本提示，并在后台向整段视频传播 mask。
* 拖动 ``frame`` 时间轴可浏览已经处理完成的帧。
* 传播进行中按空格会同时暂停传播和画面播放；暂停后空格只控制画面播放，
  ``Enter`` 可从当前帧恢复传播。
* 鼠标中键点击 mask 可选择对象；重叠区域可重复点击以切换对象。
* 鼠标左键添加正点，右键添加负点；首次添加点时会暂停后台传播。
* 每个对象在每一帧最多使用 16 个点，窗口状态会显示当前数量；达到上限后
  不再接受更多点，本轮未确认的点可用 ``Backspace`` 撤销。
* ``P`` 根据当前编辑点刷新当前帧 mask，仅作为预览。
  每次预览都会恢复到最近的 CPU 检查点，并向前重放到编辑帧，因此相同点集
  使用相同的 tracker 基础状态。``--checkpoint-interval`` 控制检查点间隔，
  默认每 20 帧保存一次。
* ``Enter`` 确认当前编辑，并从当前帧重新开始传播。
* ``Backspace`` 撤销最近一次点击，``Esc`` 放弃当前未确认的编辑。
* ``C`` 清除所有已确认和未确认的点，恢复文本提示并从第 0 帧重新传播。
* 视频传播完成后窗口会保持打开，可继续浏览或点击任意已处理帧进行编辑。
* ``Q`` 结束操作；程序会补齐尚未完成的传播并写出结果。
* 每次鼠标按键点击和键盘输入都会以 ``INTERACTION`` 开头输出一行日志。

输出文件
--------

* ``result.mp4``：使用原视频帧率、叠加最终 mask 的完整视频。
* ``interactions.json``：文本提示、确认点和交互事件记录。
"""

import argparse
import getpass
import json
import math
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image as PIL_Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = "/tmp/sam3_qualitative_test"
WINDOW_NAME = "SAM 3 interactive video"
MAX_PROMPT_POINTS = 16
MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 64, 128),
    (128, 255, 0),
    (64, 128, 255),
    (255, 200, 0),
    (0, 200, 128),
    (200, 0, 128),
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 0),
    (0, 128, 128),
]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class PointEdit:
    sequence: int
    frame_index: int
    obj_id: int
    x: int
    y: int
    label: int

    def as_json(self) -> Dict[str, int]:
        return {
            "sequence": self.sequence,
            "frame_index": self.frame_index,
            "obj_id": self.obj_id,
            "x": self.x,
            "y": self.y,
            "label": self.label,
        }


@dataclass(frozen=True)
class CommitRecord:
    frame_index: int
    affected_keys: Tuple[Tuple[int, int], ...]
    point_sequences: Tuple[int, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def expand_path(value: str) -> Path:
    return Path(value).expanduser().absolute()


def parse_device(value: str) -> Tuple[str, int]:
    if value == "cuda":
        return "cuda:0", 0
    if not value.startswith("cuda:"):
        raise argparse.ArgumentTypeError("device must look like 'cuda:0'")
    try:
        index = int(value.split(":", 1)[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("device must look like 'cuda:0'") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("CUDA device index must be non-negative")
    return value, index


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def probe_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        info = VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        cap.release()
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError(f"invalid video dimensions: {video_path}")
    if not math.isfinite(info.fps) or info.fps <= 0:
        raise RuntimeError(f"invalid video FPS: {video_path}")
    return info


def extract_frames(video_path: Path, output_dir: Path) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video_path))
    index = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_path = output_dir / f"{index:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"failed to write extracted frame: {frame_path}")
            index += 1
    finally:
        cap.release()
    if index == 0:
        raise RuntimeError(f"video contains no readable frames: {video_path}")
    print(f"Extracted {index} frames to {output_dir}")
    return index


def synthesize_video(
    out_dir: Path,
    num_objects: int = 5,
    n_frames: int = 30,
    width: int = 1024,
    height: int = 1024,
) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    colors = [
        tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(num_objects)
    ]
    positions = [
        [
            float(np.random.randint(80, width - 80)),
            float(np.random.randint(80, height - 80)),
        ]
        for _ in range(num_objects)
    ]
    velocities = [
        [np.random.choice([-1, 1]) * 15, np.random.choice([-1, 1]) * 15]
        for _ in range(num_objects)
    ]
    for index in range(n_frames):
        image = PIL_Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for object_index in range(num_objects):
            x, y = positions[object_index]
            draw.ellipse(
                [(x - 50, y - 50), (x + 50, y + 50)], fill=colors[object_index]
            )
            vx, vy = velocities[object_index]
            positions[object_index] = [
                np.clip(x + vx, 50, width - 50),
                np.clip(y + vy, 50, height - 50),
            ]
            if x < 50 or x > width - 50:
                velocities[object_index][0] *= -1
            if y < 50 or y > height - 50:
                velocities[object_index][1] *= -1
        image.save(out_dir / f"{index:05d}.jpg")
    print(f"Generated {n_frames} synthetic frames with {num_objects} circles")
    return n_frames


def load_frame_bgr(frame_dir: Path, frame_index: int) -> np.ndarray:
    frame_path = frame_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read frame: {frame_path}")
    return frame


def load_frame(frame_dir: Path, frame_index: int) -> np.ndarray:
    return cv2.cvtColor(load_frame_bgr(frame_dir, frame_index), cv2.COLOR_BGR2RGB)


def as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_masks(outputs: Optional[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    if not outputs:
        return {}
    obj_ids = as_numpy(outputs.get("out_obj_ids")).reshape(-1)
    masks = as_numpy(outputs.get("out_binary_masks"))
    if masks.size == 0:
        return {}
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise RuntimeError(f"unexpected mask shape: {masks.shape}")
    if len(obj_ids) != len(masks):
        raise RuntimeError(
            f"object and mask counts differ: {len(obj_ids)} != {len(masks)}"
        )
    return {
        int(obj_id): np.asarray(mask, dtype=bool).copy()
        for obj_id, mask in zip(obj_ids, masks)
    }


def render_overlay(
    frame_rgb: np.ndarray, masks_by_obj: Dict[int, np.ndarray]
) -> np.ndarray:
    overlay = frame_rgb.copy().astype(np.float32)
    for obj_id, mask in sorted(masks_by_obj.items()):
        color = MASK_COLORS[obj_id % len(MASK_COLORS)]
        mask_bool = mask.astype(bool)
        for channel in range(3):
            overlay[:, :, channel] = np.where(
                mask_bool,
                overlay[:, :, channel] * 0.6 + color[channel] * 0.4,
                overlay[:, :, channel],
            )
    return overlay.astype(np.uint8)


def render_frame_bgr(
    frame: np.ndarray,
    masks_by_obj: Dict[int, np.ndarray],
    points: Sequence[PointEdit] = (),
    active_obj: Optional[int] = None,
    stale: bool = False,
    status: Optional[str] = None,
) -> np.ndarray:
    blended = frame.astype(np.float32)
    for obj_id, mask in sorted(masks_by_obj.items()):
        mask_bool = mask.astype(bool)
        if mask_bool.shape != frame.shape[:2]:
            raise RuntimeError(
                f"mask dimensions {mask_bool.shape} differ from frame {frame.shape[:2]}"
            )
        rgb = MASK_COLORS[obj_id % len(MASK_COLORS)]
        color = np.array((rgb[2], rgb[1], rgb[0]), dtype=np.float32)
        blended[mask_bool] = blended[mask_bool] * 0.55 + color * 0.45
    rendered = blended.astype(np.uint8)
    for obj_id, mask in sorted(masks_by_obj.items()):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rgb = MASK_COLORS[obj_id % len(MASK_COLORS)]
        color = (rgb[2], rgb[1], rgb[0])
        thickness = 4 if obj_id == active_obj else 2
        cv2.drawContours(rendered, contours, -1, (255, 255, 255), thickness + 2)
        cv2.drawContours(rendered, contours, -1, color, thickness)
        ys, xs = np.where(mask)
        if len(xs):
            center = (int(xs.mean()), int(ys.mean()))
            cv2.putText(
                rendered,
                str(obj_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered,
                str(obj_id),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                1,
                cv2.LINE_AA,
            )
    for point in points:
        if point.label == 1:
            cv2.line(
                rendered,
                (point.x - 7, point.y),
                (point.x + 7, point.y),
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                rendered,
                (point.x, point.y - 7),
                (point.x, point.y + 7),
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.line(
                rendered,
                (point.x - 6, point.y - 6),
                (point.x + 6, point.y + 6),
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                rendered,
                (point.x - 6, point.y + 6),
                (point.x + 6, point.y - 6),
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        cv2.circle(rendered, (point.x, point.y), 9, (255, 255, 255), 1, cv2.LINE_AA)
    labels = []
    if stale:
        labels.append("STALE")
    if active_obj is not None:
        labels.append(f"obj {active_obj}")
    if status:
        labels.append(status)
    if labels:
        text = " | ".join(labels)
        cv2.putText(
            rendered,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def save_overlay(
    frame_rgb: np.ndarray,
    masks_by_obj: Dict[int, np.ndarray],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    overlay = render_overlay(frame_rgb, masks_by_obj)
    figure, axis = plt.subplots(1, 1, figsize=(12, 7), dpi=100)
    axis.imshow(overlay)
    for obj_id, mask in sorted(masks_by_obj.items()):
        if mask.any():
            ys, xs = np.where(mask)
            color_rgb = MASK_COLORS[obj_id % len(MASK_COLORS)]
            axis.text(
                int(xs.mean()),
                int(ys.mean()),
                str(obj_id),
                color="white",
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=tuple(c / 255 for c in color_rgb),
                    alpha=0.8,
                ),
            )
    if title:
        axis.set_title(title, fontsize=12, fontweight="bold", pad=8)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def collect_propagation(
    model: Any, session_id: str
) -> Dict[int, Dict[int, np.ndarray]]:
    mask_dict = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for response in model.handle_stream_request(
            {"type": "propagate_in_video", "session_id": session_id}
        ):
            frame_index = response.get("frame_index")
            if frame_index is not None:
                mask_dict[int(frame_index)] = normalize_masks(
                    response.get("outputs", {})
                )
    torch.cuda.synchronize()
    return mask_dict


class PropagationRunner:
    def __init__(
        self,
        predictor: Any,
        event_queue: queue.Queue,
        propagation_direction: str,
        checkpoint_interval: int = 20,
    ):
        self.predictor = predictor
        self.event_queue = event_queue
        self.propagation_direction = propagation_direction
        self.checkpoint_interval = checkpoint_interval
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.generation = 0
        self.session_id = ""

    @property
    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, session_id: str, start_frame_index: int, generation: int) -> None:
        if self.is_alive:
            raise RuntimeError("propagation is already running")
        self.stop_event = threading.Event()
        self.generation = generation
        self.session_id = session_id
        cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None

        def run() -> None:
            stopped = False
            next_checkpoint_frame = start_frame_index
            max_forward_frame = start_frame_index - 1
            try:
                # CUDA's current device is thread-local. Bind this worker to the
                # caller's device before PyTorch or Triton launches any kernels.
                if cuda_device is not None:
                    torch.cuda.set_device(cuda_device)
                request = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": self.propagation_direction,
                    "start_frame_index": start_frame_index,
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for response in self.predictor.handle_stream_request(request):
                        if self.stop_event.is_set():
                            stopped = True
                            break
                        frame_index = int(response["frame_index"])
                        self.event_queue.put(
                            (
                                "frame",
                                generation,
                                frame_index,
                                normalize_masks(response.get("outputs", {})),
                            )
                        )
                        # The bidirectional stream yields its forward half first.
                        # Ignore decreasing frame indices so the backward half does
                        # not overwrite forward checkpoints with a different causal
                        # history.
                        if frame_index > max_forward_frame:
                            max_forward_frame = frame_index
                            if frame_index >= next_checkpoint_frame:
                                checkpoint = self.predictor.handle_request(
                                    {
                                        "type": "save_checkpoint",
                                        "session_id": session_id,
                                        "frame_index": frame_index,
                                    }
                                )
                                self.event_queue.put(
                                    (
                                        "checkpoint",
                                        generation,
                                        checkpoint,
                                        None,
                                    )
                                )
                                next_checkpoint_frame = (
                                    frame_index + self.checkpoint_interval
                                )
                stopped = stopped or self.stop_event.is_set()
                self.event_queue.put(("done", generation, stopped, None))
            except BaseException as exc:
                self.event_queue.put(("error", generation, exc, None))

        self.thread = threading.Thread(
            target=run, name=f"sam3-propagation-{generation}", daemon=True
        )
        self.thread.start()

    def request_stop(self, cancel_model: bool) -> None:
        if not self.is_alive:
            return
        self.stop_event.set()
        if cancel_model:
            self.predictor.handle_request(
                {"type": "cancel_propagation", "session_id": self.session_id}
            )

    def join(self) -> None:
        if self.thread is not None:
            self.thread.join()


class InteractiveApp:
    def __init__(
        self,
        predictor: Any,
        version: str,
        session_id: str,
        frame_dir: Path,
        video_path: Path,
        video_info: VideoInfo,
        frame_count: int,
        prompt: str,
        output_dir: Path,
        window_width: int,
        propagation_direction: str = "both",
        checkpoint_interval: int = 20,
    ):
        self.predictor = predictor
        self.version = version
        self.session_id = session_id
        self.frame_dir = frame_dir
        self.video_path = video_path
        self.video_info = video_info
        self.frame_count = frame_count
        self.prompt = prompt
        self.output_dir = output_dir
        self.propagation_direction = propagation_direction
        self.checkpoint_interval = checkpoint_interval
        self.event_queue: queue.Queue = queue.Queue()
        self.runner = PropagationRunner(
            predictor,
            self.event_queue,
            propagation_direction,
            checkpoint_interval,
        )
        self.generation = 0
        self.propagation_complete = False
        self.cache: Dict[int, Dict[int, np.ndarray]] = {}
        self.stale_frames: set[int] = set()
        self.display_index = 0
        self.follow_live = True
        self.playing = True
        self.last_play_time = time.monotonic()
        self.active_obj: Optional[int] = None
        self.status = "initializing"
        self.fatal_error: Optional[BaseException] = None
        self.editing = False
        self.edit_frame: Optional[int] = None
        self.draft_points: List[PointEdit] = []
        self.confirmed_points: List[PointEdit] = []
        self.commits: List[CommitRecord] = []
        self.preview_signature: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.preview_keys: set[Tuple[int, int]] = set()
        self.sequence = 0
        self.events: List[Dict[str, Any]] = []
        self.selection_position: Optional[Tuple[int, int, int]] = None
        self.selection_candidates: List[int] = []
        self.selection_offset = 0
        self.suppress_trackbar = False
        self.window_open = False
        self.display_width = min(video_info.width, window_width)
        self.display_scale = self.display_width / video_info.width
        self.display_height = max(1, round(video_info.height * self.display_scale))

    def record_event(self, event_type: str, **payload: Any) -> None:
        self.sequence += 1
        self.events.append(
            {
                "sequence": self.sequence,
                "timestamp": utc_now(),
                "type": event_type,
                **payload,
            }
        )

    def log_input(self, input_type: str, **payload: Any) -> None:
        print(
            "INTERACTION "
            + json.dumps(
                {
                    "timestamp": utc_now(),
                    "input": input_type,
                    "frame_index": self.display_index,
                    **payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def start(self, initial_masks: Dict[int, np.ndarray]) -> None:
        self.cache[0] = initial_masks
        self.record_event("initial_prompt", frame_index=0, text=self.prompt)
        self.save_checkpoint(0)
        self.start_propagation(0)

    def save_checkpoint(self, frame_index: int) -> Dict[str, Any]:
        response = self.predictor.handle_request(
            {
                "type": "save_checkpoint",
                "session_id": self.session_id,
                "frame_index": frame_index,
            }
        )
        cpu_bytes = int(response.get("cpu_bytes", 0))
        checkpoint_frame = int(response.get("frame_index", frame_index))
        checkpoint_count = int(response.get("checkpoint_count", 0))
        print(
            "CHECKPOINT "
            + json.dumps(
                {
                    "frame_index": checkpoint_frame,
                    "checkpoint_count": checkpoint_count,
                    "cpu_bytes": cpu_bytes,
                    "cpu_mib": round(cpu_bytes / (1024 * 1024), 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        self.record_event(
            "checkpoint_saved",
            frame_index=checkpoint_frame,
            checkpoint_count=checkpoint_count,
            cpu_bytes=cpu_bytes,
        )
        return response

    def start_propagation(self, frame_index: int) -> None:
        self.generation += 1
        self.propagation_complete = False
        if self.propagation_direction == "forward":
            self.stale_frames.update(
                index for index in self.cache if index >= frame_index
            )
        else:
            self.stale_frames.update(self.cache)
        self.runner.start(self.session_id, frame_index, self.generation)
        self.status = "propagating"
        self.record_event(
            "propagation_start", frame_index=frame_index, generation=self.generation
        )

    def stop_propagation(self) -> None:
        if self.runner.is_alive:
            self.runner.request_stop(cancel_model=self.version == "sam3.1")
            self.runner.join()
        self.drain_events()

    def drain_events(self) -> None:
        while True:
            try:
                event_type, generation, value, extra = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if generation != self.generation:
                continue
            if event_type == "frame":
                frame_index = int(value)
                self.cache[frame_index] = extra
                self.stale_frames.discard(frame_index)
                if self.follow_live and not self.editing:
                    self.set_display_index(frame_index, programmatic=True)
            elif event_type == "done":
                stopped = bool(value)
                self.propagation_complete = not stopped
                if stopped:
                    self.status = "paused"
                else:
                    self.follow_live = False
                    self.playing = False
                    self.status = "complete - ready for review"
                self.record_event(
                    "propagation_end", generation=generation, stopped=stopped
                )
            elif event_type == "checkpoint":
                checkpoint = value
                cpu_bytes = int(checkpoint.get("cpu_bytes", 0))
                checkpoint_frame = int(checkpoint["frame_index"])
                checkpoint_count = int(checkpoint.get("checkpoint_count", 0))
                print(
                    "CHECKPOINT "
                    + json.dumps(
                        {
                            "frame_index": checkpoint_frame,
                            "checkpoint_count": checkpoint_count,
                            "cpu_bytes": cpu_bytes,
                            "cpu_mib": round(cpu_bytes / (1024 * 1024), 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                self.record_event(
                    "checkpoint_saved",
                    frame_index=checkpoint_frame,
                    checkpoint_count=checkpoint_count,
                    cpu_bytes=cpu_bytes,
                )
            elif event_type == "error":
                self.fatal_error = value
                self.status = "error"

    def set_display_index(self, frame_index: int, programmatic: bool) -> None:
        if not self.cache:
            return
        frame_index = max(0, min(frame_index, self.frame_count - 1))
        if frame_index not in self.cache:
            frame_index = min(self.cache, key=lambda value: abs(value - frame_index))
        self.display_index = frame_index
        if programmatic and self.window_open:
            self.suppress_trackbar = True
            cv2.setTrackbarPos("frame", WINDOW_NAME, frame_index)
            self.suppress_trackbar = False

    def on_trackbar(self, frame_index: int) -> None:
        if self.suppress_trackbar:
            return
        if self.editing:
            self.set_display_index(int(self.edit_frame), programmatic=True)
            return
        self.follow_live = False
        self.playing = False
        self.set_display_index(frame_index, programmatic=True)

    def image_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        return (
            min(self.video_info.width - 1, max(0, round(x / self.display_scale))),
            min(self.video_info.height - 1, max(0, round(y / self.display_scale))),
        )

    def select_object(self, x: int, y: int) -> None:
        masks = self.cache.get(self.display_index, {})
        candidates = [
            obj_id
            for obj_id, mask in masks.items()
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])
        ]
        candidates.sort(key=lambda obj_id: int(masks[obj_id].sum()))
        position = (self.display_index, x, y)
        if not candidates:
            self.status = "no object at cursor"
            return
        if (
            position == self.selection_position
            and candidates == self.selection_candidates
        ):
            self.selection_offset = (self.selection_offset + 1) % len(candidates)
        else:
            self.selection_position = position
            self.selection_candidates = candidates
            self.selection_offset = 0
        self.active_obj = candidates[self.selection_offset]
        self.status = f"selected obj {self.active_obj}"
        self.record_event(
            "select_object",
            frame_index=self.display_index,
            obj_id=self.active_obj,
            x=x,
            y=y,
        )

    def add_point(self, x: int, y: int, label: int) -> None:
        if self.active_obj is None:
            self.status = "select an object first"
            return
        if self.editing and self.display_index != self.edit_frame:
            self.set_display_index(int(self.edit_frame), programmatic=True)
            return
        frame_index = int(self.edit_frame) if self.editing else self.display_index
        point_count = sum(
            1
            for point in [*self.confirmed_points, *self.draft_points]
            if (point.frame_index, point.obj_id)
            == (frame_index, int(self.active_obj))
        )
        if point_count >= MAX_PROMPT_POINTS:
            self.status = (
                f"point limit reached ({MAX_PROMPT_POINTS}) for obj {self.active_obj}"
            )
            return
        if not self.editing:
            self.stop_propagation()
            self.editing = True
            self.edit_frame = self.display_index
            self.follow_live = False
            self.playing = False
        self.sequence += 1
        point = PointEdit(
            self.sequence, int(self.edit_frame), int(self.active_obj), x, y, label
        )
        self.draft_points.append(point)
        self.events.append(
            {
                "sequence": point.sequence,
                "timestamp": utc_now(),
                "type": "add_point",
                **point.as_json(),
            }
        )
        self.status = (
            f"editing (unconfirmed) - obj {self.active_obj} points "
            f"{point_count + 1}/{MAX_PROMPT_POINTS}"
        )

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        buttons = {
            cv2.EVENT_LBUTTONDOWN: "left",
            cv2.EVENT_MBUTTONDOWN: "middle",
            cv2.EVENT_RBUTTONDOWN: "right",
        }
        if event not in buttons:
            return
        source_x, source_y = self.image_coordinates(x, y)
        self.log_input(
            "mouse",
            button=buttons[event],
            display_x=x,
            display_y=y,
            source_x=source_x,
            source_y=source_y,
            active_obj=self.active_obj,
        )
        if event == cv2.EVENT_MBUTTONDOWN:
            self.select_object(source_x, source_y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(source_x, source_y, 1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_point(source_x, source_y, 0)

    def points_for_key(
        self, key: Tuple[int, int], include_draft: bool
    ) -> List[PointEdit]:
        points = [
            point
            for point in self.confirmed_points
            if (point.frame_index, point.obj_id) == key
        ]
        if include_draft:
            points.extend(
                point
                for point in self.draft_points
                if (point.frame_index, point.obj_id) == key
            )
        return sorted(points, key=lambda point: point.sequence)

    def draft_signature(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            (p.sequence, p.frame_index, p.obj_id, p.x, p.y, p.label)
            for p in self.draft_points
        )

    def relative_point_coordinates(
        self, points: Sequence[PointEdit]
    ) -> List[List[float]]:
        return [
            [point.x / self.video_info.width, point.y / self.video_info.height]
            for point in points
        ]

    def apply_points(self, keys: Iterable[Tuple[int, int]]) -> None:
        for frame_index, obj_id in sorted(set(keys)):
            points = self.points_for_key((frame_index, obj_id), include_draft=True)
            if not points:
                raise RuntimeError("cannot preview an empty point set")
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": self.session_id,
                    "frame_index": frame_index,
                    "points": self.relative_point_coordinates(points),
                    "point_labels": [point.label for point in points],
                    "clear_old_points": True,
                    "obj_id": obj_id,
                    "rel_coordinates": True,
                }
            )
            self.cache[frame_index] = normalize_masks(response.get("outputs", {}))
            self.stale_frames.discard(frame_index)

    def preview(self) -> None:
        if not self.editing or not self.draft_points:
            self.status = "nothing to preview"
            return
        self.stop_propagation()
        signature = self.draft_signature()
        if signature == self.preview_signature:
            self.status = "preview"
            return
        keys = {(point.frame_index, point.obj_id) for point in self.draft_points}
        self.restore_preview_baseline(int(self.edit_frame))
        self.apply_points(keys)
        self.preview_signature = signature
        self.preview_keys = keys
        self.status = "preview"
        self.record_event(
            "preview",
            frame_index=self.edit_frame,
            point_sequences=[p.sequence for p in self.draft_points],
        )

    def undo(self) -> None:
        if not self.editing or not self.draft_points:
            self.status = "nothing to undo"
            return
        point = self.draft_points.pop()
        self.active_obj = point.obj_id
        self.set_display_index(point.frame_index, programmatic=True)
        self.status = "undo (preview is outdated)"
        self.record_event("undo", point_sequence=point.sequence)

    def confirm(self) -> None:
        if not self.editing:
            if not self.runner.is_alive and not self.propagation_complete:
                self.start_propagation(self.display_index)
            return
        self.stop_propagation()
        if not self.draft_points:
            if self.preview_signature is not None:
                self.restore_preview_baseline(int(self.edit_frame))
            frame_index = int(self.edit_frame)
            self.reset_edit_state()
            if not self.propagation_complete:
                self.start_propagation(frame_index)
            return
        if self.draft_signature() != self.preview_signature:
            self.preview()
        affected_keys = tuple(
            sorted({(p.frame_index, p.obj_id) for p in self.draft_points})
        )
        committed_now = tuple(point.sequence for point in self.draft_points)
        self.confirmed_points.extend(self.draft_points)
        frame_index = int(self.edit_frame)
        self.commits.append(CommitRecord(frame_index, affected_keys, committed_now))
        self.record_event(
            "confirm", frame_index=frame_index, point_sequences=list(committed_now)
        )
        self.reset_edit_state()
        self.predictor.handle_request(
            {"type": "clear_checkpoints", "session_id": self.session_id}
        )
        self.save_checkpoint(frame_index)
        self.start_propagation(frame_index)

    def reset_edit_state(self) -> None:
        self.editing = False
        self.edit_frame = None
        self.draft_points = []
        self.preview_signature = None
        self.preview_keys = set()

    def cancel_edit(self) -> None:
        if not self.editing:
            return
        frame_index = int(self.edit_frame)
        had_preview = self.preview_signature is not None
        self.record_event(
            "cancel_edit",
            frame_index=frame_index,
            point_sequences=[p.sequence for p in self.draft_points],
        )
        self.reset_edit_state()
        if had_preview:
            self.restore_preview_baseline(frame_index)
        self.start_propagation(frame_index)

    def clear_all_interactions(self) -> None:
        cleared_confirmed = len(self.confirmed_points)
        cleared_draft = len(self.draft_points)
        self.stop_propagation()
        self.confirmed_points = []
        self.commits = []
        self.reset_edit_state()
        self.active_obj = None
        self.selection_position = None
        self.selection_candidates = []
        self.selection_offset = 0
        self.predictor.handle_request(
            {"type": "clear_checkpoints", "session_id": self.session_id}
        )
        self.restore_confirmed_state()
        self.record_event(
            "clear_all_interactions",
            cleared_confirmed_points=cleared_confirmed,
            cleared_draft_points=cleared_draft,
        )
        self.save_checkpoint(0)
        self.start_propagation(0)

    def replay_commit_points(
        self, committed_sequences: set[int], keys: Iterable[Tuple[int, int]]
    ) -> None:
        for frame_index, obj_id in keys:
            points = [
                p
                for p in self.confirmed_points
                if p.sequence in committed_sequences
                and (p.frame_index, p.obj_id) == (frame_index, obj_id)
            ]
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": self.session_id,
                    "frame_index": frame_index,
                    "points": self.relative_point_coordinates(points),
                    "point_labels": [point.label for point in points],
                    "clear_old_points": True,
                    "obj_id": obj_id,
                    "rel_coordinates": True,
                }
            )
            self.cache[frame_index] = normalize_masks(response.get("outputs", {}))

    def synchronous_propagation(
        self,
        start_frame_index: int,
        propagation_direction: Optional[str] = None,
        max_frame_num_to_track: Optional[int] = None,
    ) -> None:
        request = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "propagation_direction": (
                self.propagation_direction
                if propagation_direction is None
                else propagation_direction
            ),
            "start_frame_index": start_frame_index,
        }
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = max_frame_num_to_track
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for response in self.predictor.handle_stream_request(request):
                frame_index = int(response["frame_index"])
                self.cache[frame_index] = normalize_masks(response.get("outputs", {}))
                self.stale_frames.discard(frame_index)
        self.propagation_complete = True

    def restore_preview_baseline(self, target_frame_index: int) -> None:
        self.stop_propagation()
        self.status = "restoring checkpoint"
        response = self.predictor.handle_request(
            {
                "type": "restore_checkpoint",
                "session_id": self.session_id,
                "frame_index": target_frame_index,
            }
        )
        if not response.get("is_success", False):
            raise RuntimeError(
                f"no checkpoint is available at or before frame {target_frame_index}"
            )
        checkpoint_frame = int(response["frame_index"])
        self.stale_frames.update(
            range(checkpoint_frame + 1, self.frame_count)
        )
        if checkpoint_frame < target_frame_index:
            replay_start = checkpoint_frame + 1
            self.status = (
                f"replaying {replay_start}-{target_frame_index} from checkpoint "
                f"{checkpoint_frame}"
            )
            # _get_processing_order uses an inclusive end point, hence N frames
            # from replay_start through target require max_frame_num_to_track=N-1.
            self.synchronous_propagation(
                replay_start,
                propagation_direction="forward",
                max_frame_num_to_track=target_frame_index - replay_start,
            )
        self.propagation_complete = False
        self.set_display_index(target_frame_index, programmatic=True)
        self.record_event(
            "checkpoint_restored",
            checkpoint_frame=checkpoint_frame,
            target_frame=target_frame_index,
            cpu_bytes=int(response.get("cpu_bytes", 0)),
        )

    def restore_confirmed_state(self) -> None:
        self.stop_propagation()
        self.status = "restoring"
        self.predictor.handle_request(
            {
                "type": "reset_session",
                "session_id": self.session_id,
            }
        )
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": 0,
                "text": self.prompt,
            }
        )
        self.cache[0] = normalize_masks(response.get("outputs", {}))
        self.stale_frames = set(range(self.frame_count))
        self.stale_frames.discard(0)
        committed_sequences: set[int] = set()
        for commit in self.commits:
            committed_sequences.update(commit.point_sequences)
            self.replay_commit_points(committed_sequences, commit.affected_keys)
        self.propagation_complete = False
        self.status = "restored"
        self.record_event("restore_complete", commits_replayed=len(self.commits))

    def current_points_for_display(self) -> List[PointEdit]:
        return [
            p
            for p in [*self.confirmed_points, *self.draft_points]
            if p.frame_index == self.display_index
        ]

    def render(self) -> np.ndarray:
        rendered = render_frame_bgr(
            load_frame_bgr(self.frame_dir, self.display_index),
            self.cache.get(self.display_index, {}),
            points=self.current_points_for_display(),
            active_obj=self.active_obj,
            stale=self.display_index in self.stale_frames,
            status=self.status,
        )
        if self.display_scale != 1.0:
            rendered = cv2.resize(
                rendered,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_AREA,
            )
        return rendered

    def advance_playback(self) -> None:
        if not self.playing or self.follow_live or self.editing:
            return
        now = time.monotonic()
        if now - self.last_play_time < 1.0 / self.video_info.fps:
            return
        self.last_play_time = now
        later = sorted(index for index in self.cache if index > self.display_index)
        if later:
            self.set_display_index(later[0], programmatic=True)

    def handle_key(self, key: int) -> bool:
        key_names = {
            8: "Backspace",
            10: "Enter",
            13: "Enter",
            27: "Esc",
            32: "Space",
            127: "Backspace",
        }
        key_name = key_names.get(key)
        if key_name is None:
            key_name = chr(key) if 32 <= key <= 126 else f"code-{key}"
        self.log_input("keyboard", key=key_name, key_code=key)
        if key in (ord("q"), ord("Q")):
            return False
        if key == 27:
            self.cancel_edit()
        elif key in (8, 127):
            self.undo()
        elif key in (ord("p"), ord("P")):
            self.preview()
        elif key in (ord("c"), ord("C")):
            self.clear_all_interactions()
        elif key in (10, 13):
            self.confirm()
        elif key == ord(" ") and not self.editing:
            self.follow_live = False
            if self.runner.is_alive:
                self.stop_propagation()
                self.playing = False
                self.status = "paused"
            else:
                self.playing = not self.playing
            self.last_play_time = time.monotonic()
        return True

    def finalize(self) -> None:
        if self.editing:
            had_preview = self.preview_signature is not None
            self.record_event(
                "quit_discard", point_sequences=[p.sequence for p in self.draft_points]
            )
            self.reset_edit_state()
            if had_preview:
                self.restore_confirmed_state()
        if self.runner.is_alive:
            self.runner.join()
            self.drain_events()
        if self.fatal_error is not None:
            raise RuntimeError("background propagation failed") from self.fatal_error
        if not self.propagation_complete or len(self.cache) < self.frame_count:
            start_frame = self.commits[-1].frame_index if self.commits else 0
            self.synchronous_propagation(start_frame)
        missing = sorted(set(range(self.frame_count)) - set(self.cache))
        if missing:
            raise RuntimeError(
                f"final propagation did not produce frames: {missing[:10]}"
            )
        self.record_event("finalize", frames=self.frame_count)
        write_interactive_outputs(self)

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        self.window_open = True
        cv2.createTrackbar(
            "frame", WINDOW_NAME, 0, max(0, self.frame_count - 1), self.on_trackbar
        )
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        try:
            running = True
            while running:
                self.drain_events()
                if self.fatal_error is not None:
                    raise RuntimeError(
                        "background propagation failed"
                    ) from self.fatal_error
                self.advance_playback()
                cv2.imshow(WINDOW_NAME, self.render())
                key = cv2.waitKey(15) & 0xFF
                if key != 255:
                    running = self.handle_key(key)
            self.finalize()
        finally:
            if self.runner.is_alive:
                self.runner.request_stop(cancel_model=False)
                self.runner.join()
            cv2.destroyWindow(WINDOW_NAME)
            self.window_open = False


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)


def write_interactive_outputs(app: InteractiveApp) -> None:
    app.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = app.output_dir / "result.mp4"
    temporary_result = app.output_dir / ".result.tmp.mp4"
    if temporary_result.exists():
        temporary_result.unlink()
    writer = cv2.VideoWriter(
        str(temporary_result),
        cv2.VideoWriter_fourcc(*"mp4v"),
        app.video_info.fps,
        (app.video_info.width, app.video_info.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"cannot create output video: {temporary_result}")
    try:
        for frame_index in range(app.frame_count):
            points = [p for p in app.confirmed_points if p.frame_index == frame_index]
            writer.write(
                render_frame_bgr(
                    load_frame_bgr(app.frame_dir, frame_index),
                    app.cache[frame_index],
                    points=points,
                )
            )
    except BaseException:
        writer.release()
        if temporary_result.exists():
            temporary_result.unlink()
        raise
    finally:
        writer.release()
    temporary_result.replace(result_path)
    atomic_write_json(
        app.output_dir / "interactions.json",
        {
            "status": "success",
            "created_at": utc_now(),
            "input_video": str(app.video_path),
            "model_version": app.version,
            "text_prompt": app.prompt,
            "propagation_direction": app.propagation_direction,
            "source": {
                "width": app.video_info.width,
                "height": app.video_info.height,
                "frame_count": app.frame_count,
                "fps": app.video_info.fps,
            },
            "outputs": {"visualization_video": "result.mp4"},
            "confirmed_points": [
                p.as_json()
                for p in sorted(app.confirmed_points, key=lambda point: point.sequence)
            ],
            "events": app.events,
        },
    )
    print(f"Saved interactive result to {result_path}")


def validate_interactive_outputs(output_dir: Path, overwrite: bool) -> None:
    existing = [
        p
        for p in (output_dir / "result.mp4", output_dir / "interactions.json")
        if p.exists()
    ]
    if existing and not overwrite:
        raise RuntimeError(
            "output already exists (use --overwrite): "
            + ", ".join(str(path) for path in existing)
        )


def run_interactive(
    predictor: Any,
    version: str,
    video_path: Path,
    video_info: VideoInfo,
    frame_dir: Path,
    frame_count: int,
    prompt: str,
    output_dir: Path,
    window_width: int,
    propagation_direction: str,
    checkpoint_interval: int,
) -> None:
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(frame_dir)}
    )
    session_id = response["session_id"]
    app: Optional[InteractiveApp] = None
    try:
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )
        app = InteractiveApp(
            predictor,
            version,
            session_id,
            frame_dir,
            video_path,
            video_info,
            frame_count,
            prompt,
            output_dir,
            window_width,
            propagation_direction,
            checkpoint_interval,
        )
        app.start(normalize_masks(response.get("outputs", {})))
        app.run()
    finally:
        if app is not None:
            session_id = app.session_id
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def run_batch(
    predictor: Any, version: str, frame_dir: Path, frame_count: int, text_prompt: str
) -> None:
    image = load_frame(frame_dir, 0)
    image_height, image_width = image.shape[:2]
    print(f"Video: {image_width}x{image_height}, {frame_count} frames")
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(frame_dir)}
    )
    session_id = response["session_id"]
    try:
        out_dir = Path(OUTPUT_DIR) / f"{version}_text_{text_prompt}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        print(f"\nTest: text prompt '{text_prompt}' -> propagate")
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
            }
        )
        masks_by_frame = collect_propagation(predictor, session_id)
        print(f"Propagated through {len(masks_by_frame)} frames")
        saved = 0
        for frame_index in sorted(masks_by_frame):
            if frame_index % 5 != 0 or not masks_by_frame[frame_index]:
                continue
            save_overlay(
                load_frame(frame_dir, frame_index),
                masks_by_frame[frame_index],
                out_dir / f"frame_{frame_index:05d}.png",
                title=f"{version} | frame {frame_index} | "
                f"{len(masks_by_frame[frame_index])} objects",
            )
            saved += 1
        frame_zero = masks_by_frame.get(0, {})
        print(f"\nDetected {len(frame_zero)} objects on frame 0:")
        for obj_id, mask in sorted(frame_zero.items()):
            if mask.any():
                ys, xs = np.where(mask)
                print(
                    f"  obj {obj_id}: centroid ({int(xs.mean())}, {int(ys.mean())}), "
                    f"{int(mask.sum())} pixels"
                )
        print(f"\nSaved {saved} overlay images to {out_dir}")
        print(
            "QUALITATIVE TEST PASSED" if frame_zero else "WARNING: No objects detected!"
        )
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SAM3 qualitative test")
    parser.add_argument("--version", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--checkpoint", help="Checkpoint path (auto-downloads if omitted)"
    )
    parser.add_argument(
        "--text_prompt", default="circle", help="Text prompt for detection"
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=("cuda:0", 0),
        metavar="cuda:N",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for the result video and metadata"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing interactive outputs"
    )
    parser.add_argument(
        "--propagation-direction",
        choices=("both", "forward"),
        default="both",
        help="Propagate from an edited frame in both directions or forward only",
    )
    parser.add_argument(
        "--window-width",
        type=positive_int,
        default=1280,
        help="Maximum interactive image width (default: 1280)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=positive_int,
        default=20,
        metavar="FRAMES",
        help="Save a CPU tracker checkpoint every FRAMES frames (default: 20)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    video_path = expand_path(args.video)
    checkpoint = expand_path(args.checkpoint) if args.checkpoint else None
    output_dir = expand_path(args.output_dir)
    if not video_path.is_file():
        parser.error(f"video does not exist: {video_path}")
    if checkpoint is not None and not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    try:
        validate_interactive_outputs(output_dir, args.overwrite)
    except RuntimeError as exc:
        parser.error(str(exc))
    device_name, device_index = args.device
    if not torch.cuda.is_available():
        parser.error("CUDA is required by the SAM 3 video predictor")
    if device_index >= torch.cuda.device_count():
        parser.error(
            f"Requested {device_name}, but only {torch.cuda.device_count()} "
            "CUDA device(s) are visible"
        )
    torch.cuda.set_device(device_index)
    username = getpass.getuser()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_cache_{username}"
    os.environ["USE_PERFLIB"] = "1"
    from sam3 import build_sam3_predictor

    print(f"Building {args.version} model on {device_name}...")
    build_kwargs = dict(version=args.version, compile=False, async_loading_frames=False)
    if checkpoint is not None:
        build_kwargs["checkpoint_path"] = str(checkpoint)
    predictor = build_sam3_predictor(**build_kwargs)
    try:
        video_info = probe_video(video_path)
        with tempfile.TemporaryDirectory(prefix="sam3_interactive_frames_") as temp:
            frame_dir = Path(temp)
            frame_count = extract_frames(video_path, frame_dir)
            run_interactive(
                predictor,
                args.version,
                video_path,
                video_info,
                frame_dir,
                frame_count,
                args.text_prompt,
                output_dir,
                args.window_width,
                args.propagation_direction,
                args.checkpoint_interval,
            )
    finally:
        predictor.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

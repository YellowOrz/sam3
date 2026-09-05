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
        --checkpoint-interval 20 \
        --output-dir ./outputs/interactive/example

如果输出目录中已经存在结果，请增加 ``--overwrite``。

交互操作
--------

* 程序从第 0 帧开始正向播放和正向传播 mask；播放和传播是两套互不影响的
  状态，后续传播方向由 UI 按钮控制。
* 下方时间轴的蓝色部分是从第 0 帧开始连续处理完成的可播放范围，灰色尾部
  只表示视频总长度；拖动位置不会超过蓝色范围，拖动会暂停画面播放，但不会
  影响传播。帧一旦处理过就保持可播放，即使后来因编辑被标记为待重新传播。
* 播放按钮依次为倒放、暂停、正放；传播按钮依次为暂停、反向、正向、先正向
  再反向、先反向再正向。每组只有一个按钮高亮；点击倒放或正放会取消当前
  mask 选择，但保留未提交点。
* 传播运行时只能点击传播暂停，其他传播方向按钮暂时禁用；自然完成后自动
  回到传播暂停。点击任一传播方向会确认未提交点，并从编辑帧（没有编辑时为
  当前帧）启动传播。
* 只有播放和传播都暂停后才能操作视频画面或使用编辑快捷键；运行期间除
  ``Q`` 外的键盘输入以及视频区域鼠标输入都会被忽略。
* 鼠标中键点击 mask 可选择对象；重叠区域从小到大循环，循环末尾取消选择；
  点击空白处也会取消选择。选中 mask 使用高亮填充和细青色轮廓。
* 鼠标左键添加正点，右键添加负点。未选择 mask 时，第一个正点会创建新的
  对象；第一个负点不会创建对象。
* 每个对象在每一帧最多使用 16 个点，窗口状态会显示当前数量；达到上限后
  不再接受更多点，本轮未确认的点可用 ``Backspace`` 撤销。
* ``P`` 根据当前编辑点刷新当前帧 mask，仅作为预览。
  每次预览都会恢复到最近的 CPU 检查点，并按对应方向重放到编辑帧，因此
  相同点集使用相同的 tracker 基础状态。正向和反向传播都会保存带方向的 CPU 检查点；
  ``--checkpoint-interval`` 控制检查点间隔，默认每 20 帧保存一次。
* ``Backspace`` 撤销最近一次点击，``Esc`` 放弃当前未确认的编辑。
* ``C`` 清除所有已确认和未确认的点，恢复文本提示并从第 0 帧重新传播。
* 未提交点始终绑定创建它们的编辑帧；播放或拖动时间轴不会提交或丢弃它们。
* 视频传播完成后窗口会保持打开，可继续浏览或点击任意已处理帧进行编辑。
* ``Enter`` 和空格不执行任何操作。
* ``Q`` 只检查所有帧是否都至少处理过一次：仍有未处理帧时提示并保持窗口
  打开；全部处理过才退出并写出结果。退出不会确认未提交点或自动补传播，
  终端会打印已处理帧数和可播放范围。
* 每次鼠标按键点击和键盘输入都会以 ``INTERACTION`` 开头输出一行日志。

输出文件
--------

* ``result.mp4``：使用原视频帧率、叠加最终 mask 的完整视频。
* ``masks.mkv``：FFV1 无损灰度标签视频；0 为背景，1--255 为对象标签。
* ``metadata.json``：输入、模型、对象标签映射和输出格式元数据。
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

if __package__:
    from scripts.video_utils import (
        as_numpy,
        color_for_label,
        COLORS,
        expand_path,
        lighter_color,
        parse_device,
        positive_int,
        utc_now as _utc_now,
    )
else:
    from video_utils import (  # type: ignore[no-redef]
        as_numpy,
        color_for_label,
        COLORS,
        expand_path,
        lighter_color,
        parse_device,
        positive_int,
        utc_now as _utc_now,
    )

WINDOW_NAME = "SAM 3 interactive video"
WINDOW_FLAGS = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL
MAX_PROMPT_POINTS = 16
MASK_ALPHA = 0.10
SELECTED_MASK_ALPHA = 0.20
SELECTED_COLOR = (255, 255, 0)
EDGE_HALO_THICKNESS = 2
EDGE_COLOR_THICKNESS = 1
SELECTED_HALO_THICKNESS = 4
SELECTED_EDGE_THICKNESS = 2
TIMELINE_HEIGHT = 72
BUTTON_ROW_HEIGHT = 82
CONTROL_HEIGHT = TIMELINE_HEIGHT + BUTTON_ROW_HEIGHT
PROPAGATION_MODES = (
    "backward",
    "forward",
    "forward_backward",
    "backward_forward",
)


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
    return _utc_now("milliseconds")


def window_width_int(value: str) -> int:
    number = positive_int(value)
    if number < 640:
        raise argparse.ArgumentTypeError("window width must be at least 640")
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


def load_frame_bgr(frame_dir: Path, frame_index: int) -> np.ndarray:
    frame_path = frame_dir / f"{frame_index:05d}.jpg"
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot read frame: {frame_path}")
    return frame


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


def normalize_frame_outputs(
    outputs: Optional[Dict[str, Any]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    masks = normalize_masks(outputs)
    if not outputs:
        return masks, {}
    obj_ids = as_numpy(outputs.get("out_obj_ids")).reshape(-1)
    probabilities = as_numpy(outputs.get("out_probs")).reshape(-1)
    if len(probabilities) not in (0, len(obj_ids)):
        raise RuntimeError(
            f"object and probability counts differ: "
            f"{len(obj_ids)} != {len(probabilities)}"
        )
    return masks, {
        int(obj_id): float(probability)
        for obj_id, probability in zip(obj_ids, probabilities)
    }


def render_frame_bgr(
    frame: np.ndarray,
    masks_by_obj: Dict[int, np.ndarray],
    probabilities_by_obj: Optional[Dict[int, float]] = None,
    object_to_label: Optional[Dict[int, int]] = None,
    frame_index: Optional[int] = None,
    prompt: Optional[str] = None,
    points: Sequence[PointEdit] = (),
    active_obj: Optional[int] = None,
    stale: bool = False,
    status: Optional[str] = None,
) -> np.ndarray:
    probabilities_by_obj = probabilities_by_obj or {}
    object_to_label = object_to_label or {}
    blended = frame.astype(np.float32)
    for obj_id, mask in masks_by_obj.items():
        mask_bool = mask.astype(bool)
        if mask_bool.shape != frame.shape[:2]:
            raise RuntimeError(
                f"mask dimensions {mask_bool.shape} differ from frame {frame.shape[:2]}"
            )
        label = object_to_label.get(obj_id, obj_id)
        selected = obj_id == active_obj
        color = np.asarray(
            SELECTED_COLOR if selected else color_for_label(label), dtype=np.float32
        )
        alpha = SELECTED_MASK_ALPHA if selected else MASK_ALPHA
        blended[mask_bool] = blended[mask_bool] * (1.0 - alpha) + color * alpha
    rendered = blended.astype(np.uint8)
    for obj_id, mask in masks_by_obj.items():
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        label = object_to_label.get(obj_id, obj_id)
        selected = obj_id == active_obj
        color = SELECTED_COLOR if selected else color_for_label(label)
        cv2.drawContours(
            rendered,
            contours,
            -1,
            (255, 255, 255) if selected else lighter_color(color),
            SELECTED_HALO_THICKNESS if selected else EDGE_HALO_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.drawContours(
            rendered,
            contours,
            -1,
            color,
            SELECTED_EDGE_THICKNESS if selected else EDGE_COLOR_THICKNESS,
            cv2.LINE_AA,
        )
        ys, xs = np.where(mask)
        if len(xs):
            center = (int(np.median(xs)), int(np.median(ys)))
            text = f"label={label} id={obj_id}"
            if obj_id in probabilities_by_obj:
                text += f" p={probabilities_by_obj[obj_id]:.2f}"
            cv2.putText(
                rendered,
                text,
                (max(0, center[0] - 40), max(18, center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    if frame_index is not None and prompt is not None:
        safe_prompt = prompt.encode("ascii", errors="replace").decode("ascii")
        cv2.putText(
            rendered,
            f"frame={frame_index} prompt={safe_prompt}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
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


def propagation_legs(mode: str) -> Tuple[str, ...]:
    legs = {
        "backward": ("backward",),
        "forward": ("forward",),
        "forward_backward": ("forward", "backward"),
        "backward_forward": ("backward", "forward"),
    }
    try:
        return legs[mode]
    except KeyError as exc:
        raise ValueError(f"unknown propagation mode: {mode}") from exc


def normalize_propagation_mode(mode: str) -> str:
    return "forward_backward" if mode == "both" else mode.replace("-", "_")


class PropagationRunner:
    def __init__(
        self,
        predictor: Any,
        event_queue: queue.Queue,
        checkpoint_interval: int = 20,
    ):
        self.predictor = predictor
        self.event_queue = event_queue
        self.checkpoint_interval = checkpoint_interval
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.generation = 0
        self.session_id = ""

    @property
    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(
        self,
        session_id: str,
        start_frame_index: int,
        generation: int,
        mode: str,
    ) -> None:
        if self.is_alive:
            raise RuntimeError("propagation is already running")
        self.stop_event = threading.Event()
        self.generation = generation
        self.session_id = session_id
        cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None

        def run() -> None:
            stopped = False
            try:
                # CUDA's current device is thread-local. Bind this worker to the
                # caller's device before PyTorch or Triton launches any kernels.
                if cuda_device is not None:
                    torch.cuda.set_device(cuda_device)
                for direction in propagation_legs(mode):
                    if self.stop_event.is_set():
                        stopped = True
                        break
                    self.event_queue.put(("direction", generation, direction, mode))
                    next_checkpoint_frame = start_frame_index
                    max_forward_frame = start_frame_index - 1
                    min_backward_frame = start_frame_index
                    if direction == "backward":
                        checkpoint = self.predictor.handle_request(
                            {
                                "type": "save_checkpoint",
                                "session_id": session_id,
                                "frame_index": start_frame_index,
                                "propagation_direction": direction,
                                "exact_frame": True,
                            }
                        )
                        self.event_queue.put(
                            ("checkpoint", generation, checkpoint, None)
                        )
                        next_checkpoint_frame -= self.checkpoint_interval
                    request = {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
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
                                    normalize_frame_outputs(
                                        response.get("outputs", {})
                                    ),
                                )
                            )
                            if (
                                direction == "forward"
                                and frame_index > max_forward_frame
                            ):
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
                            elif (
                                direction == "backward"
                                and frame_index < min_backward_frame
                            ):
                                min_backward_frame = frame_index
                                if frame_index <= next_checkpoint_frame:
                                    checkpoint = self.predictor.handle_request(
                                        {
                                            "type": "save_checkpoint",
                                            "session_id": session_id,
                                            "frame_index": frame_index,
                                            "propagation_direction": direction,
                                            "exact_frame": True,
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
                                        frame_index - self.checkpoint_interval
                                    )
                        if stopped:
                            break
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
        self.initial_propagation_mode = "forward"
        self.checkpoint_interval = checkpoint_interval
        self.event_queue: queue.Queue = queue.Queue()
        self.runner = PropagationRunner(
            predictor,
            self.event_queue,
            checkpoint_interval,
        )
        self.generation = 0
        self.propagation_complete = False
        self.cache: Dict[int, Dict[int, np.ndarray]] = {}
        self.probability_cache: Dict[int, Dict[int, float]] = {}
        self.stale_frames: set[int] = set()
        self.display_index = 0
        self.playing = True
        self.playback_direction = 1
        self.last_play_time = time.monotonic()
        self.propagation_mode: Optional[str] = None
        self.last_propagation_mode = self.initial_propagation_mode
        self.active_obj: Optional[int] = None
        self.status = "initializing"
        self.fatal_error: Optional[BaseException] = None
        self.editing = False
        self.edit_frame: Optional[int] = None
        self.draft_points: List[PointEdit] = []
        self.draft_new_obj_ids: set[int] = set()
        self.confirmed_points: List[PointEdit] = []
        self.commits: List[CommitRecord] = []
        self.preview_signature: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.preview_keys: set[Tuple[int, int]] = set()
        self.sequence = 0
        self.events: List[Dict[str, Any]] = []
        self.selection_position: Optional[Tuple[int, int, int]] = None
        self.selection_candidates: List[int] = []
        self.selection_offset = 0
        self.next_obj_id = 1
        self.hitboxes: Dict[str, Tuple[int, int, int, int]] = {}
        self.timeline_dragging = False
        self.window_open = False
        self.display_width = min(window_width, max(video_info.width, 640))
        self.display_scale = self.display_width / video_info.width
        self.display_height = max(1, round(video_info.height * self.display_scale))
        self.timeline_bounds = (20, max(20, self.display_width - 20))

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
        tqdm.write(
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
            file=sys.stderr,
        )

    def store_frame_outputs(
        self, frame_index: int, outputs: Optional[Dict[str, Any]]
    ) -> None:
        masks, probabilities = normalize_frame_outputs(outputs)
        self.cache[frame_index] = masks
        self.probability_cache[frame_index] = probabilities
        if masks:
            self.next_obj_id = max(self.next_obj_id, max(masks) + 1)

    def start(self, initial_outputs: Optional[Dict[str, Any]]) -> None:
        self.store_frame_outputs(0, initial_outputs)
        self.record_event("initial_prompt", frame_index=0, text=self.prompt)
        self.save_checkpoint(0)
        self.start_propagation(0, self.initial_propagation_mode)

    def save_checkpoint(self, frame_index: int) -> Dict[str, Any]:
        response = self.predictor.handle_request(
            {
                "type": "save_checkpoint",
                "session_id": self.session_id,
                "frame_index": frame_index,
                "propagation_direction": "both",
                "exact_frame": True,
            }
        )
        self.record_checkpoint(response, frame_index)
        return response

    def record_checkpoint(self, response: Dict[str, Any], fallback_frame: int) -> None:
        if not response.get("created", True):
            return
        frame_index = int(response.get("frame_index", fallback_frame))
        self.record_event("checkpoint_saved", frame_index=frame_index)
        tqdm.write(
            f"saved CPU checkpoint at frame {frame_index} in session {self.session_id}",
            file=sys.stderr,
        )

    def start_propagation(self, frame_index: int, mode: str) -> None:
        mode = normalize_propagation_mode(mode)
        if mode not in PROPAGATION_MODES:
            raise ValueError(f"unknown propagation mode: {mode}")
        self.generation += 1
        self.propagation_complete = False
        self.propagation_mode = mode
        self.last_propagation_mode = mode
        directions = set(propagation_legs(mode))
        if directions == {"forward"}:
            self.stale_frames.update(
                index for index in self.cache if index > frame_index
            )
        elif directions == {"backward"}:
            self.stale_frames.update(
                index for index in self.cache if index < frame_index
            )
        else:
            self.stale_frames.update(
                index for index in self.cache if index != frame_index
            )
        self.runner.start(self.session_id, frame_index, self.generation, mode)
        self.status = f"propagating {mode.replace('_', '-')}"
        self.record_event(
            "propagation_start",
            frame_index=frame_index,
            generation=self.generation,
            mode=mode,
        )

    def stop_propagation(self) -> None:
        if self.runner.is_alive:
            self.runner.request_stop(cancel_model=self.version == "sam3.1")
            self.runner.join()
        self.drain_events()
        self.propagation_mode = None

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
                masks, probabilities = extra
                self.cache[frame_index] = masks
                self.probability_cache[frame_index] = probabilities
                if masks:
                    self.next_obj_id = max(self.next_obj_id, max(masks) + 1)
                self.stale_frames.discard(frame_index)
            elif event_type == "done":
                stopped = bool(value)
                self.propagation_complete = not stopped
                finished_mode = self.propagation_mode
                self.propagation_mode = None
                self.status = (
                    "propagation paused"
                    if stopped
                    else f"propagation complete ({finished_mode})"
                )
                self.record_event(
                    "propagation_end",
                    generation=generation,
                    stopped=stopped,
                    mode=finished_mode,
                )
            elif event_type == "checkpoint":
                self.record_checkpoint(value, self.display_index)
            elif event_type == "direction":
                self.status = f"propagating {value}"
            elif event_type == "error":
                self.fatal_error = value
                self.status = "error"

    def playable_frontier(self) -> int:
        frontier = -1
        for frame_index in range(self.frame_count):
            if frame_index not in self.cache:
                break
            frontier = frame_index
        return frontier

    def unprocessed_frames(self) -> List[int]:
        return [
            frame_index
            for frame_index in range(self.frame_count)
            if frame_index not in self.cache
        ]

    def set_display_index(self, frame_index: int) -> None:
        if not self.cache:
            return
        frontier = self.playable_frontier()
        if frontier < 0:
            return
        frame_index = max(0, min(frame_index, frontier))
        self.display_index = frame_index

    def seek_timeline(self, x: int) -> None:
        frontier = self.playable_frontier()
        if frontier < 0:
            return
        left, right = self.timeline_bounds
        fraction = (min(right, max(left, x)) - left) / max(1, right - left)
        frame_index = round(fraction * frontier)
        self.playing = False
        self.set_display_index(frame_index)
        self.status = f"frame {self.display_index}"

    def image_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        return (
            min(self.video_info.width - 1, max(0, round(x / self.display_scale))),
            min(self.video_info.height - 1, max(0, round(y / self.display_scale))),
        )

    def clear_object_selection(self, update_status: bool = True) -> None:
        if self.active_obj is not None:
            self.record_event("deselect_object", frame_index=self.display_index)
        self.active_obj = None
        self.selection_position = None
        self.selection_candidates = []
        self.selection_offset = 0
        if update_status:
            self.status = "selection cleared"

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
            self.clear_object_selection()
            return
        if (
            position == self.selection_position
            and candidates == self.selection_candidates
        ):
            self.selection_offset = (self.selection_offset + 1) % (len(candidates) + 1)
        else:
            self.selection_position = position
            self.selection_candidates = candidates
            self.selection_offset = (
                len(candidates) if self.active_obj in candidates else 0
            )
        self.active_obj = (
            candidates[self.selection_offset]
            if self.selection_offset < len(candidates)
            else None
        )
        if self.active_obj is None:
            self.status = "selection cleared"
            self.record_event("deselect_object", frame_index=self.display_index)
            return
        self.status = f"selected obj {self.active_obj}"
        self.record_event(
            "select_object",
            frame_index=self.display_index,
            obj_id=self.active_obj,
            x=x,
            y=y,
        )

    def add_point(self, x: int, y: int, label: int) -> None:
        if self.playing or self.runner.is_alive:
            self.status = "pause playback and propagation before editing"
            return
        if self.active_obj is None:
            if label == 0:
                self.status = "add a positive point to create an object"
                return
            self.active_obj = self.next_obj_id
            self.next_obj_id += 1
            self.draft_new_obj_ids.add(self.active_obj)
            self.record_event(
                "create_object_draft",
                frame_index=self.display_index,
                obj_id=self.active_obj,
            )
        if self.editing and self.display_index != self.edit_frame:
            self.set_display_index(int(self.edit_frame))
            self.status = f"returned to draft frame {self.edit_frame}"
            return
        frame_index = int(self.edit_frame) if self.editing else self.display_index
        point_count = sum(
            1
            for point in [*self.confirmed_points, *self.draft_points]
            if (point.frame_index, point.obj_id) == (frame_index, int(self.active_obj))
        )
        if point_count >= MAX_PROMPT_POINTS:
            self.status = (
                f"point limit reached ({MAX_PROMPT_POINTS}) for obj {self.active_obj}"
            )
            return
        if not self.editing:
            self.editing = True
            self.edit_frame = self.display_index
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

    @staticmethod
    def point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    def handle_control_click(self, name: str) -> None:
        if name == "play_pause":
            self.playing = False
            self.status = "playback paused"
        elif name == "play_backward":
            self.clear_object_selection(update_status=False)
            if self.display_index == 0:
                self.playing = False
                self.status = "at first frame"
            else:
                self.playback_direction = -1
                self.playing = True
                self.last_play_time = time.monotonic()
                self.status = "playing backward"
        elif name == "play_forward":
            self.clear_object_selection(update_status=False)
            self.playback_direction = 1
            self.playing = True
            self.last_play_time = time.monotonic()
            self.status = "playing forward"
        elif name == "prop_pause":
            self.stop_propagation()
            self.status = "propagation paused"
        elif name.startswith("prop_"):
            if self.runner.is_alive:
                self.status = "pause propagation before changing direction"
                return
            self.confirm_and_propagate(name.removeprefix("prop_"))

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del param
        if event == cv2.EVENT_LBUTTONUP:
            self.timeline_dragging = False
            return
        if event == cv2.EVENT_MOUSEMOVE and self.timeline_dragging:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.seek_timeline(x)
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.display_height <= y < (
            self.display_height + TIMELINE_HEIGHT
        ):
            self.log_input("timeline", display_x=x)
            self.timeline_dragging = True
            self.seek_timeline(x)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, rect in self.hitboxes.items():
                if self.point_in_rect(x, y - self.display_height, rect):
                    self.log_input("control", control=name)
                    self.handle_control_click(name)
                    return
        buttons = {
            cv2.EVENT_LBUTTONDOWN: "left",
            cv2.EVENT_MBUTTONDOWN: "middle",
            cv2.EVENT_RBUTTONDOWN: "right",
        }
        if event not in buttons:
            return
        if y >= self.display_height:
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
        if self.playing or self.runner.is_alive:
            self.status = "pause playback and propagation before editing"
            return
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
            self.store_frame_outputs(frame_index, response.get("outputs", {}))
            self.stale_frames.discard(frame_index)

    def preview(self) -> None:
        if not self.editing or not self.draft_points:
            self.status = "nothing to preview"
            return
        if self.playing or self.runner.is_alive:
            self.status = "pause playback and propagation before preview"
            return
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
        self.set_display_index(point.frame_index)
        self.status = "undo (preview is outdated)"
        self.record_event("undo", point_sequence=point.sequence)
        if not self.draft_points:
            self.cancel_edit()

    def commit_draft(self) -> Optional[int]:
        if not self.editing:
            return None
        if not self.draft_points:
            if self.preview_signature is not None:
                self.restore_preview_baseline(int(self.edit_frame))
            frame_index = int(self.edit_frame)
            self.reset_edit_state()
            return frame_index
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
        return frame_index

    def confirm_and_propagate(self, mode: str) -> None:
        if self.runner.is_alive:
            self.status = "pause propagation before changing direction"
            return
        frame_index = int(self.edit_frame) if self.editing else self.display_index
        if self.editing:
            self.set_display_index(frame_index)
            self.commit_draft()
        self.start_propagation(frame_index, mode)

    def reset_edit_state(self) -> None:
        self.editing = False
        self.edit_frame = None
        self.draft_points = []
        self.draft_new_obj_ids = set()
        self.preview_signature = None
        self.preview_keys = set()

    def cancel_edit(self) -> None:
        if not self.editing:
            return
        frame_index = int(self.edit_frame)
        had_preview = self.preview_signature is not None
        cancelled_new_ids = set(self.draft_new_obj_ids)
        self.record_event(
            "cancel_edit",
            frame_index=frame_index,
            point_sequences=[p.sequence for p in self.draft_points],
        )
        self.reset_edit_state()
        if self.active_obj in cancelled_new_ids:
            self.active_obj = None
        if had_preview:
            self.restore_preview_baseline(frame_index)
        self.status = "edit cancelled"

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
        self.start_propagation(0, self.initial_propagation_mode)

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
            self.store_frame_outputs(frame_index, response.get("outputs", {}))

    def synchronous_propagation(
        self,
        start_frame_index: int,
        propagation_direction: Optional[str] = None,
        max_frame_num_to_track: Optional[int] = None,
    ) -> None:
        mode = normalize_propagation_mode(
            propagation_direction or self.last_propagation_mode
        )
        for direction in propagation_legs(mode):
            request = {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": direction,
                "start_frame_index": start_frame_index,
            }
            if max_frame_num_to_track is not None:
                request["max_frame_num_to_track"] = max_frame_num_to_track
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for response in self.predictor.handle_stream_request(request):
                    frame_index = int(response["frame_index"])
                    self.store_frame_outputs(frame_index, response.get("outputs", {}))
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
            self.status = f"rebuilding baseline for frame {target_frame_index}"
            self.restore_confirmed_state()
            if target_frame_index > 0:
                self.synchronous_propagation(
                    0,
                    propagation_direction="forward",
                    max_frame_num_to_track=target_frame_index,
                )
            self.propagation_complete = False
            self.set_display_index(target_frame_index)
            self.record_event(
                "preview_baseline_rebuilt",
                target_frame=target_frame_index,
            )
            return
        checkpoint_frame = int(response["frame_index"])
        checkpoint_direction = response.get("propagation_direction", "forward")
        if checkpoint_frame > target_frame_index:
            self.stale_frames.update(range(checkpoint_frame))
            self.status = (
                f"replaying {checkpoint_frame - 1}-{target_frame_index} backward "
                f"from checkpoint {checkpoint_frame}"
            )
            self.synchronous_propagation(
                checkpoint_frame,
                propagation_direction="backward",
                max_frame_num_to_track=checkpoint_frame - target_frame_index,
            )
        else:
            if (
                checkpoint_frame == target_frame_index
                and checkpoint_direction == "backward"
            ):
                self.stale_frames.update(range(checkpoint_frame))
            else:
                self.stale_frames.update(range(checkpoint_frame + 1, self.frame_count))
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
        self.set_display_index(target_frame_index)
        self.record_event(
            "checkpoint_restored",
            checkpoint_frame=checkpoint_frame,
            target_frame=target_frame_index,
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
        self.store_frame_outputs(0, response.get("outputs", {}))
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

    @staticmethod
    def draw_arrow_icon(
        image: np.ndarray,
        rect: Tuple[int, int, int, int],
        direction: int,
        color: Tuple[int, int, int],
        y_offset: int = 0,
    ) -> None:
        left, top, right, bottom = rect
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2 + y_offset
        half = max(6, min(12, (right - left) // 5))
        cv2.line(
            image,
            (center_x - direction * half, center_y),
            (center_x + direction * half, center_y),
            color,
            2,
            cv2.LINE_AA,
        )
        tip_x = center_x + direction * half
        points = np.array(
            [
                (tip_x, center_y),
                (tip_x - direction * 7, center_y - 6),
                (tip_x - direction * 7, center_y + 6),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, points, color, cv2.LINE_AA)

    def draw_button(
        self,
        image: np.ndarray,
        name: str,
        rect: Tuple[int, int, int, int],
        active: bool,
        enabled: bool,
    ) -> None:
        self.hitboxes[name] = rect
        background = (190, 125, 25) if active else (52, 52, 52)
        border = (255, 190, 70) if active else (95, 95, 95)
        if not enabled and not active:
            background, border = (35, 35, 35), (55, 55, 55)
        color = (255, 255, 255) if enabled or active else (105, 105, 105)
        cv2.rectangle(image, rect[:2], rect[2:], background, cv2.FILLED, cv2.LINE_AA)
        cv2.rectangle(image, rect[:2], rect[2:], border, 1, cv2.LINE_AA)
        if name.endswith("pause"):
            left, top, right, bottom = rect
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            cv2.rectangle(
                image,
                (center_x - 7, center_y - 11),
                (center_x - 3, center_y + 11),
                color,
                cv2.FILLED,
            )
            cv2.rectangle(
                image,
                (center_x + 3, center_y - 11),
                (center_x + 7, center_y + 11),
                color,
                cv2.FILLED,
            )
        elif name.endswith("forward_backward"):
            self.draw_arrow_icon(image, rect, 1, color, -8)
            self.draw_arrow_icon(image, rect, -1, color, 8)
        elif name.endswith("backward_forward"):
            self.draw_arrow_icon(image, rect, -1, color, -8)
            self.draw_arrow_icon(image, rect, 1, color, 8)
        elif name.endswith("backward"):
            self.draw_arrow_icon(image, rect, -1, color)
        else:
            self.draw_arrow_icon(image, rect, 1, color)

    def render_controls(self) -> np.ndarray:
        width = self.display_width
        panel = np.full((CONTROL_HEIGHT, width, 3), (27, 29, 32), dtype=np.uint8)
        self.hitboxes = {}
        frontier = self.playable_frontier()
        current = min(self.display_index, max(0, frontier))
        progress_text = f"frame {current} | processed to {max(0, frontier)} | total {self.frame_count}"
        status = self.status
        if self.editing and self.edit_frame is not None:
            status += f" | draft frame {self.edit_frame}"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        max_status_width = max(80, width - text_size[0] - 60)
        while (
            len(status) > 3
            and cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            > max_status_width
        ):
            status = status[:-4] + "..."
        cv2.putText(
            panel,
            status,
            (20, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            progress_text,
            (max(20, width - text_size[0] - 20), 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        track_left, track_right, track_y = 20, width - 20, 48
        cv2.line(
            panel,
            (track_left, track_y),
            (track_right, track_y),
            (70, 70, 70),
            10,
            cv2.LINE_AA,
        )
        if self.frame_count <= 1:
            available_right = track_right
            thumb_x = track_left
        else:
            available_right = track_left + round(
                (track_right - track_left) * max(0, frontier) / (self.frame_count - 1)
            )
            thumb_x = track_left + round(
                (track_right - track_left) * current / (self.frame_count - 1)
            )
        self.timeline_bounds = (track_left, max(track_left, available_right))
        if frontier >= 0:
            cv2.line(
                panel,
                (track_left, track_y),
                (available_right, track_y),
                (220, 145, 35),
                10,
                cv2.LINE_AA,
            )
            cv2.circle(panel, (thumb_x, track_y), 9, (255, 200, 80), cv2.FILLED)
        cv2.line(
            panel,
            (0, TIMELINE_HEIGHT),
            (width, TIMELINE_HEIGHT),
            (55, 55, 55),
            1,
        )

        row_top = TIMELINE_HEIGHT
        button_top = row_top + 17
        button_bottom = row_top + BUTTON_ROW_HEIGHT - 15
        play_right = round(width * 0.38)
        cv2.line(
            panel, (play_right, row_top), (play_right, CONTROL_HEIGHT), (55, 55, 55), 1
        )
        cv2.putText(
            panel,
            "PLAY",
            (18, row_top + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        play_names = ("play_backward", "play_pause", "play_forward")
        play_left = 78
        play_gap = 7
        play_width = max(34, (play_right - play_left - 20 - 2 * play_gap) // 3)
        active_play = (
            "play_pause"
            if not self.playing
            else ("play_forward" if self.playback_direction > 0 else "play_backward")
        )
        for index, name in enumerate(play_names):
            left = play_left + index * (play_width + play_gap)
            self.draw_button(
                panel,
                name,
                (left, button_top, left + play_width, button_bottom),
                name == active_play,
                True,
            )

        prop_left = play_right + 18
        cv2.putText(
            panel,
            "PROP",
            (prop_left, row_top + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        prop_names = (
            "prop_pause",
            "prop_backward",
            "prop_forward",
            "prop_forward_backward",
            "prop_backward_forward",
        )
        prop_button_left = prop_left + 66
        prop_gap = 7
        prop_width = max(
            34,
            (width - prop_button_left - 20 - 4 * prop_gap) // 5,
        )
        active_prop = (
            f"prop_{self.propagation_mode}"
            if self.runner.is_alive and self.propagation_mode
            else "prop_pause"
        )
        for index, name in enumerate(prop_names):
            left = prop_button_left + index * (prop_width + prop_gap)
            enabled = name == "prop_pause" or not self.runner.is_alive
            self.draw_button(
                panel,
                name,
                (left, button_top, left + prop_width, button_bottom),
                name == active_prop,
                enabled,
            )
        return panel

    def render(self) -> np.ndarray:
        rendered = render_frame_bgr(
            load_frame_bgr(self.frame_dir, self.display_index),
            self.cache.get(self.display_index, {}),
            probabilities_by_obj=self.probability_cache.get(self.display_index, {}),
            points=self.current_points_for_display(),
            active_obj=self.active_obj,
            stale=self.display_index in self.stale_frames,
            status=None,
        )
        if self.display_scale != 1.0:
            rendered = cv2.resize(
                rendered,
                (self.display_width, self.display_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.vstack((rendered, self.render_controls()))

    def advance_playback(self) -> None:
        if not self.playing:
            return
        now = time.monotonic()
        if now - self.last_play_time < 1.0 / self.video_info.fps:
            return
        self.last_play_time = now
        next_frame = self.display_index + self.playback_direction
        frontier = self.playable_frontier()
        if 0 <= next_frame <= frontier:
            self.set_display_index(next_frame)
            return
        if self.playback_direction < 0 or self.display_index >= self.frame_count - 1:
            self.playing = False
            self.status = "complete - ready for review"

    def handle_key(self, key: int) -> bool:
        key_names = {
            8: "Backspace",
            27: "Esc",
            127: "Backspace",
        }
        key_name = key_names.get(key)
        if key_name is None:
            key_name = chr(key) if 32 <= key <= 126 else f"code-{key}"
        self.log_input("keyboard", key=key_name, key_code=key)
        if key in (ord("q"), ord("Q")):
            missing = self.unprocessed_frames()
            if missing:
                preview = ", ".join(str(frame) for frame in missing[:10])
                suffix = "..." if len(missing) > 10 else ""
                self.status = (
                    f"cannot quit: {len(missing)} unprocessed frames "
                    f"({preview}{suffix})"
                )
                tqdm.write(
                    f"无法退出：还有 {len(missing)} 帧未处理" f"（{preview}{suffix}）",
                    file=sys.stderr,
                )
                return True
            return False
        if self.playing or self.runner.is_alive:
            self.status = "pause playback and propagation before editing"
            return True
        if key == 27:
            self.cancel_edit()
        elif key in (8, 127):
            self.undo()
        elif key in (ord("p"), ord("P")):
            self.preview()
        elif key in (ord("c"), ord("C")):
            self.clear_all_interactions()
        elif key in (10, 13, 32):
            self.status = "Enter and Space are disabled"
        return True

    def finalize(self) -> None:
        self.playing = False
        self.stop_propagation()
        if self.fatal_error is not None:
            raise RuntimeError("background propagation failed") from self.fatal_error
        missing = self.unprocessed_frames()
        if missing:
            raise RuntimeError(f"cannot finalize unprocessed frames: {missing[:10]}")
        self.record_event("finalize", frames=self.frame_count)
        write_interactive_outputs(self)
        frontier = self.playable_frontier()
        print(
            f"处理结果：已处理 {frontier + 1}/{self.frame_count} 帧"
            f"（可播放范围 0-{frontier}）"
        )

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
        self.window_open = True
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


def build_label_image(
    masks_by_obj: Dict[int, np.ndarray],
    object_to_label: Dict[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    label_image = np.zeros((height, width), dtype=np.uint8)
    for obj_id, mask in masks_by_obj.items():
        if obj_id not in object_to_label:
            label = len(object_to_label) + 1
            if label > 255:
                raise RuntimeError(
                    "more than 255 tracked instances; uint8 labels overflow"
                )
            object_to_label[obj_id] = label
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        label_image[mask.astype(bool)] = object_to_label[obj_id]
    return label_image


def write_interactive_outputs(app: InteractiveApp) -> None:
    app.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = app.output_dir / "result.mp4"
    masks_path = app.output_dir / "masks.mkv"
    temporary_result = app.output_dir / ".result.tmp.mp4"
    temporary_masks = app.output_dir / ".masks.tmp.mkv"
    for path in (temporary_result, temporary_masks):
        if path.exists():
            path.unlink()
    result_writer = cv2.VideoWriter(
        str(temporary_result),
        cv2.VideoWriter_fourcc(*"mp4v"),
        app.video_info.fps,
        (app.video_info.width, app.video_info.height),
    )
    if not result_writer.isOpened():
        result_writer.release()
        temporary_result.unlink(missing_ok=True)
        raise RuntimeError(f"cannot create output video: {temporary_result}")
    mask_writer = cv2.VideoWriter(
        str(temporary_masks),
        cv2.VideoWriter_fourcc(*"FFV1"),
        app.video_info.fps,
        (app.video_info.width, app.video_info.height),
        isColor=False,
    )
    if not mask_writer.isOpened():
        result_writer.release()
        mask_writer.release()
        for path in (temporary_result, temporary_masks):
            path.unlink(missing_ok=True)
        raise RuntimeError(f"cannot create lossless mask video: {temporary_masks}")
    object_to_label: Dict[int, int] = {}
    try:
        for frame_index in range(app.frame_count):
            masks_by_obj = app.cache[frame_index]
            label_image = build_label_image(
                masks_by_obj,
                object_to_label,
                app.video_info.width,
                app.video_info.height,
            )
            mask_writer.write(label_image)
            result_writer.write(
                render_frame_bgr(
                    load_frame_bgr(app.frame_dir, frame_index),
                    masks_by_obj,
                    probabilities_by_obj=app.probability_cache.get(frame_index, {}),
                    object_to_label=object_to_label,
                    frame_index=frame_index,
                    prompt=app.prompt,
                )
            )
    except BaseException:
        result_writer.release()
        mask_writer.release()
        for path in (temporary_result, temporary_masks):
            path.unlink(missing_ok=True)
        raise
    finally:
        result_writer.release()
        mask_writer.release()
    temporary_result.replace(result_path)
    temporary_masks.replace(masks_path)
    source = {
        "width": app.video_info.width,
        "height": app.video_info.height,
        "frame_count": app.frame_count,
        "fps": app.video_info.fps,
    }
    outputs = {
        "instance_masks_video": "masks.mkv",
        "instance_masks_codec": "FFV1",
        "instance_masks_pixel_format": "gray8",
        "visualization_video": "result.mp4",
        "label_dtype": "uint8",
        "background_label": 0,
    }
    atomic_write_json(
        app.output_dir / "metadata.json",
        {
            "status": "success",
            "completed_at": utc_now(),
            "input_video": str(app.video_path),
            "prompt": app.prompt,
            "model_version": app.version,
            "source": source,
            "frames_processed": app.frame_count,
            "object_id_to_label": {
                str(obj_id): label for obj_id, label in sorted(object_to_label.items())
            },
            "outputs": outputs,
        },
    )
    atomic_write_json(
        app.output_dir / "interactions.json",
        {
            "status": "success",
            "created_at": utc_now(),
            "input_video": str(app.video_path),
            "model_version": app.version,
            "text_prompt": app.prompt,
            "propagation_direction": app.initial_propagation_mode,
            "source": source,
            "outputs": outputs,
            "confirmed_points": [
                p.as_json()
                for p in sorted(app.confirmed_points, key=lambda point: point.sequence)
            ],
            "events": app.events,
        },
    )
    print(f"Saved interactive outputs to {app.output_dir}")


def validate_interactive_outputs(output_dir: Path, overwrite: bool) -> None:
    existing = [
        p
        for p in (
            output_dir / "result.mp4",
            output_dir / "masks.mkv",
            output_dir / "metadata.json",
            output_dir / "interactions.json",
        )
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
            checkpoint_interval,
        )
        app.start(response.get("outputs", {}))
        app.run()
    finally:
        if app is not None:
            session_id = app.session_id
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
        "--output-dir",
        required=True,
        help="Directory for the result video and metadata",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing interactive outputs"
    )
    parser.add_argument(
        "--window-width",
        type=window_width_int,
        default=1280,
        help="Maximum interactive image width, at least 640 (default: 1280)",
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
                args.checkpoint_interval,
            )
    finally:
        predictor.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

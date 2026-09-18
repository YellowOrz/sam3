# 交互式视频标注

在桌面窗口里用文本提示启动 SAM 3 / 3.1，逐帧检查、点选修正，再双向传播实例掩码。需要 **CUDA GPU** 和 **图形界面**（OpenCV 窗口）。

## 用 uv 安装环境

仓库根目录已有 `uv.lock`。Python 需 **3.12**（`>=3.9,<3.13`）。

```bash
cd /path/to/sam3
uv python install 3.12
uv sync --python 3.12
```

`uv sync` 会创建 `.venv` 并按锁文件安装本仓库（可编辑）及依赖。之后用 `uv run` 执行脚本，不必手动 `source .venv/bin/activate`。

确认 GPU 可用：

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

若 `torch.cuda.is_available()` 为 `False`，从 PyTorch 官方 CUDA 源重装 GPU 版（按本机 CUDA 改 index，例如 cu128）：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

SAM 3 的 `.pt` 权重可从 [ModelScope facebook/sam3](https://modelscope.cn/models/facebook/sam3) 下载：

```bash
pip install modelscope
modelscope download --model facebook/sam3
```

默认会落到 `~/.cache/modelscope/models/facebook--sam3/`（常见文件为 `snapshots/master/sam3.pt`）。运行时用 `--checkpoint` 指向该文件。不传 `--checkpoint` 时会按 Hugging Face 规则下载。

## 运行

```bash
uv run python scripts/qualitative_test_interactive.py \
    --version sam3 \
    --checkpoint ~/.cache/modelscope/models/facebook--sam3/snapshots/master/sam3.pt \
    --video /path/to/color.mp4 \
    --text-prompt "human hand" \
    --device cuda:0 \
    --checkpoint-interval 20 \
    --chunk-frames 0 \
    --output-dir ./outputs/interactive/example
```

输出目录里已有 `result.mp4` / `masks.mkv` / `metadata.json` / `interactions.json` 时必须加 `--overwrite`。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--version` | `sam3.1` | `sam3` 或 `sam3.1` |
| `--video` | 必填 | 输入视频 |
| `--checkpoint` | 省略则自动下载 | 权重文件 |
| `--text-prompt` | `circle` | 初始文本检测提示 |
| `--device` | `cuda:0` | CUDA 设备 |
| `--output-dir` | 必填 | 结果目录 |
| `--overwrite` | 关 | 覆盖已有结果 |
| `--window-width` | `1280` | 画面最大宽度，至少 640 |
| `--checkpoint-interval` | `20` | 每 N 帧存一次 CPU tracker 检查点（预览回放用） |
| `--chunk-frames` | `0` | `0` 不分段；大于 0 时必须 ≥ 100，按段独立 session，结束后终端复核 ID |

`--chunk-frames` 开启时，每段结束按剩余原 ID 升序紧凑编号为 0、1、2…。编辑过程中不改号。不分段则在整段导出时做同样整理。

## 界面

上半是视频；下半是控制条。

- 状态栏：当前操作提示；有未提交点时会标 `draft frame`。
- 进度：`frame 当前帧 | processed to 可播放末帧 | total 总帧数`。
- 时间轴：蓝色（偏橙）是从第 0 帧起**连续已处理、可播放**的范围；灰色是视频总长。拖动不能超出蓝色。已处理过的帧会一直可播放，即使后来因编辑变成待重传播。
- 进度条上方 `>|` / `|<`：传播窗口左右界（默认整段）。正向传到右界停、反向传到左界停，端点帧仍会写出。
- 进度条下方红色 ▴：该帧有已确认或未提交提示点；点击跳到该帧（仅限可播放范围）。
- 画面右上角：`CURRENT` 本次传播已更新；`STALE` 仍是旧结果、等重传播；`PREVIEW` 当前帧在预览未提交点。

**播放和传播是两套状态**，互不影响。程序从第 0 帧开始正向播放、正向传播。后续传播方向由 PROP 按钮决定。

**编辑前提**：播放和传播都必须暂停。运行中除 `Q` 外的键盘、以及视频区域鼠标都会被忽略。

## 按钮

左组 **PLAY**（连续播放三个里只有一个高亮）：

| 图标（从左到右） | 作用 |
| --- | --- |
| 左箭头 | 倒放。会取消当前 mask 选择，保留未提交点 |
| 竖条 + 左三角 | 上一帧。仅暂停时可点 |
| 暂停（两条竖线） | 暂停播放 |
| 右三角 + 竖条 | 下一帧。仅暂停时可点 |
| 右箭头 | 正放。会取消当前 mask 选择，保留未提交点 |

按帧步进不取消 mask 选择，也不提交点。步进范围与时间轴相同（0 到可播放末帧）。

右组 **PROP**：

| 图标（从左到右） | 作用 |
| --- | --- |
| 暂停 | 停止传播 |
| 左箭头 | 反向传播 |
| 右箭头 | 正向传播 |
| 上下两个箭头（上右下左） | 先正向再反向 |
| 上下两个箭头（上左下右） | 先反向再正向 |

传播进行中只能点传播暂停，其它方向按钮禁用；自然结束后回到传播暂停。点任一传播方向会**确认未提交点**，并从编辑帧启动（没有编辑时用当前显示帧）。起始帧若在传播窗口外，该方向只更新起始帧，不会把窗口外的帧纳入传播。

## 鼠标

**时间轴（控制条上半）**

- 拖动滑块：定位到可播放范围内的帧；会暂停画面播放，**不影响传播**。
- 拖动 `>|` / `|<`：改传播窗口，两界不能交叉。
- 点击 ▴：跳到对应提示点所在帧。

**视频画面（必须双暂停）**

| 操作 | 作用 |
| --- | --- |
| 左键 | 正点（绿十字）。未选中 mask 时，第一个正点创建新对象 |
| 右键 | 负点（红叉）。第一个负点不会创建对象 |
| 中键点在 mask 上 | 选中该对象（高亮填充 + 青色细轮廓）。重叠处从小到大循环，循环完取消选择 |
| 中键点空白 | 取消选择 |

每个对象在每一帧最多 16 个点，状态栏会显示数量；满了不再加点。未提交点绑在创建它们的那一帧，播放或拖时间轴不会提交也不会丢掉。

## 键盘

编辑快捷键同样要求播放和传播都暂停（`Q` 除外）。

| 键 | 作用 |
| --- | --- |
| `P` | 用当前编辑点刷新**当前帧** mask，只作预览。会回到最近 CPU 检查点再播到编辑帧 |
| `Backspace` | 撤销最近一次未确认点击 |
| `Esc` | 放弃当前未确认编辑 |
| `D` | 删除当前选中对象（tracker 和已缓存帧都去掉）。有未提交编辑会先取消。未选中时无效 |
| `[` / `]` | 把当前帧设为传播窗口左/右界；若交叉，另一侧跟着移到当前帧 |
| `C` | 清除全部已确认和未确认点，恢复文本提示，从第 0 帧重新传播。**不重置**传播窗口 |
| `Q` | 全部帧都至少处理过一次才退出并写出结果。有未处理帧则提示并保持窗口。**不**确认未提交点、**不**自动补传播 |
| `Enter` / 空格 | 无操作 |

每次鼠标按键和键盘输入都会在终端打一行以 `INTERACTION` 开头的日志。

## 分段复核（`--chunk-frames` ≥ 100）

各段交互结束后，默认相同局部 ID 视为同一物体，合成全视频并自动回放。画面里 `chunk` 从 1 计段号，`local` 是本段紧凑 ID，`id` 是全局 ID。

**回放窗口**

- 空格：暂停 / 继续
- 拖 `Frame` 滑条：定位（会暂停）
- `Q` / `Esc` / 关窗口：回到终端（不确认结果）

**终端**（只有这里的 `ok` 才算确认）

| 命令 | 作用 |
| --- | --- |
| `show` | 打印每段 `局部ID=全局ID` |
| `map 2 1=2` | 第 2 段局部 1 对应全局 2，作用于整段 |
| `map 2 1=2 2=1` | 同段交换；同段不能两个局部 ID 指向同一全局 ID |
| `replay` | 再回放 |
| `ok` | 确认：全局 ID 再按升序紧凑为 0、1、2… 并写出最终结果 |

可用未占用的全局 ID 表示新物体。全视频最多 255 个不同物体。非法命令不改结果。每次有效 `map` 都会重写视频、无损掩码和对应表。

复核期间结果为 `unconfirmed`。EOF 或 Ctrl+C 保留最近写出的文件并以非零退出。脚本**不能**在重启后接着复核。

## 输出

| 文件 | 内容 |
| --- | --- |
| `result.mp4` | 原帧率、叠加最终 mask |
| `masks.mkv` | FFV1 无损灰度标签；0 背景，1–255 对象。最终对象 ID 对应像素标签 **ID+1** |
| `metadata.json` | 输入、模型、ID 映射、输出格式；分段时含局部→全局对应及是否已确认 |
| `interactions.json` | 文本提示、确认点和交互事件。已删除对象导出 ID 为 null，仍保留原始 ID |

CUDA 推理用 AMP：优先 `bfloat16`，不支持则 `float16`。

# Repository Guidelines

## 项目结构与模块组织

`sam3/` 是主 Python 包：`model/` 包含图像、视频与跟踪模型，`train/` 包含训练器、损失、数据管道和 Hydra YAML 配置，`agent/` 提供智能体推理封装。`scripts/` 存放评测、数据处理和性能测试工具；`examples/` 为可运行 notebook；`assets/` 保存示例媒体和图表。新增自动化测试应放入 `tests/`；`test/` 是旧测试目录，不在默认 pytest 搜索路径中。生成结果写入 `outputs/`，不要提交大型模型权重或临时产物。

## 自定义视频脚本

共用代码位于 `scripts/common/`：`video_cli.py` 集中三个批量入口的数据输入输出、批量运行、模型加载和 GT 参数组，以及递归视频发现和目录校验；三者均用 `--rgb-name`（默认 `color.mp4`）递归查找根目录及所有子目录，输入输出根目录不得相同（含 `--list-only`）。`video_utils.py` 提供视频工具，`compare_gt_masks.py` 提供 GT 参数、评测和批次汇总。`process_learned_prompt_videos.py`、`process_mano_prompt_videos.py`、`process_dataset_videos.py` 统一通过 `--compare-gt --gt-mask-name NAME` 启用评测；`--gt-dir-name` 默认 `masks_sam3`，`--compare-skip-frames` 默认 `0`，GT 文件名不从提示词、目标 ID 或手侧推导。

- `process_dataset_videos.py`：递归查找 `color.mp4`，用统一文本提示执行 SAM 3/3.1 分割并保留目录结构。先用 `--list-only` 核对输入；示例：`python scripts/process_dataset_videos.py --input-root DATA --output-root OUT --version sam3.1 --text-prompt hand --device cuda:0`。
- `process_mano_prompt_videos.py`：同一目录约定，用 `<mano-dir-name>/<left|right>_hand/*.npz` 的全部 MANO 几何做 SAM 3 分割。多个 NPZ 的框/点按帧合并进一次检测；`--save-detector` 另写 `detector_masks.mkv` 和 `detector_result.mp4`。示例：`python scripts/process_mano_prompt_videos.py --input-root DATA --output-root OUT --text-prompt "left hand" --hand-side left --device cuda:0`。
- `process_learned_prompt_videos.py`：同一目录约定，用训练好的目标特征文件做无文本 SAM 3 分割（不支持 3.1）。`--rgb-name` 指定 RGB 文件名（默认 `color.mp4`）；`--compare-gt` 启用同级 GT 对比，`--gt-dir-name` 默认 `masks_sam3`，`--gt-mask-name` 在启用评测时必填，`--compare-skip-frames 2` 评测第 0、3、6…帧并生成对比视频及 IoU/Dice 汇总；完整预测可直接复用。详情见 `docs/learned_prompt_training.md`。示例：`python scripts/process_learned_prompt_videos.py --input-root DATA --output-root OUT --learned-prompt outputs/learned_left_hand/best.pt --checkpoint /path/to/sam3.pt --device cuda:0`。
- `process_bidirectional_videos.py`：基础 SAM3 单 GPU、严格零训练的双向 memory 融合。隔离正反向建库，再联合读取记忆并单次解码；源库不写回，不再使用 Viterbi。默认物理倒序；`--chunk-frames 120 --context-frames 30` 启用重叠分块，核心帧唯一归属。例：`python scripts/process_bidirectional_videos.py --input-root DATA --output-root OUT --text-prompt "left hand" --device cuda:0 --checkpoint /path/to/sam3.pt`。详细限制和消融见 `docs/bidirectional_hand_segmentation_research.md`。
- `process_hand_video_with_wilor_prompts.py`：处理单个目录中的 `color.mp4` 和 `MANO_wilor_occlusion/hand_joints_occlusion.jsonl`，用 WiLoR 关节迭代修正指定侧手的 SAM 分割。正点必须是落在上一轮目标掩码内的可见目标手关节；`--segmentation-passes` 默认为 2，设为 1 时仅运行文本分割。示例：`uv run scripts/process_hand_video_with_wilor_prompts.py --input-dir DATA --output-dir OUT --hand-side left --version sam3 --segmentation-passes 2 --device cuda:0`。
- `compare_dataset_videos.py`：按共同的 `result.mp4` 相对路径（可用 `--video-name` 指定其他文件名），将多个提示词输出拼成带标签的对比视频。至少传入两个输出根目录，并用 `--output-dir` 指定目标；自动布局不合适时传 `--grid 2x2`。
- `qualitative_test_interactive.py`：在桌面窗口中逐帧检查、修正并双向传播实例掩码，输出无损掩码、结果视频和元数据。运行：`python scripts/qualitative_test_interactive.py --video INPUT.mp4 --output-dir OUT --device cuda:0`；需图形环境与 CUDA。
- `mirror_color_videos.sh`：通过 FFmpeg 批量水平镜像 `color.mp4`，并在目标目录保留相对路径。运行：`./scripts/mirror_color_videos.sh INPUT_ROOT OUTPUT_ROOT`；输入与输出目录不可相同。

## 构建、测试与开发命令

- `pip install -e ".[dev,train]"`：以可编辑模式安装开发、测试和训练依赖。
- `pytest`：运行 `pyproject.toml` 配置的 `tests/` 测试集。
- `pytest tests/test_compare_dataset_videos.py -q`：快速运行单个测试模块。
- `ufmt format sam3 scripts tests`：使用 Black 24.2、usort 和 Ruff formatter 统一格式。
- `python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml`：启动示例训练；运行前确认数据、GPU 和配置路径。

基础安装仅需 `pip install -e .`；notebook 环境使用 `pip install -e ".[notebooks]"`。

## 编码风格与命名约定

Python 使用 4 空格缩进、类型注解，并遵循 `pyproject.toml` 中 Black 的 88 字符行宽。Markdown 不设行宽限制，不要为了满足代码行宽而机械换行段落或列表项。模块、函数及变量采用 `snake_case`，类采用 `PascalCase`，常量采用 `UPPER_SNAKE_CASE`。导入由 usort 排序；不要手工调整格式化工具生成的布局。保持模型逻辑、数据处理和 CLI 入口分离，优先复用现有工具函数。

## 测试规范

项目使用 pytest；文件命名为 `test_*.py`，测试类为 `Test*`，函数为 `test_*`。修复缺陷时添加可复现回归测试；新 CLI 应覆盖参数解析、错误输入和输出文件行为。测试应使用 `tmp_path`、`monkeypatch` 等 fixture，避免下载权重、依赖真实数据集或修改仓库文件。当前未规定覆盖率阈值，但应覆盖变更路径和边界情况。

## 提交与 Pull Request

近期历史通常采用 `feat(scope): ...`、`refactor: ...` 等 Conventional Commits，也保留简洁祈使句；每个提交聚焦一个逻辑变更。PR 从 `main` 分支创建，需说明动机、实现和验证命令，关联相关 issue；可视化或交互变化附截图或短视频。API 变化同步更新 README/notebook，并确保格式检查和测试通过。首次贡献须完成 Meta CLA。安全漏洞不要公开建 issue，应按 `CONTRIBUTING.md` 的披露流程处理。

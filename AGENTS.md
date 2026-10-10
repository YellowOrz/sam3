# Repository Guidelines

## 项目结构与模块组织

`sam3/` 是主 Python 包：`model/` 包含图像、视频与跟踪模型，`train/` 包含训练器、损失、数据管道和 Hydra YAML 配置，`agent/` 提供智能体推理封装。`scripts/` 存放评测、数据处理和性能测试工具；`examples/` 为可运行 notebook；`assets/` 保存示例媒体和图表。新增自动化测试应放入 `tests/`；`test/` 是旧测试目录，不在默认 pytest 搜索路径中。生成结果写入 `outputs/`，不要提交大型模型权重或临时产物。

## 自定义视频脚本

共用代码位于 `scripts/common/`：`video_cli.py` 集中三个批量入口的数据输入输出、批量运行、模型加载和 GT 参数组，以及递归视频发现和目录校验；三者均用 `--rgb-name`（默认 `color.mp4`）递归查找根目录及所有子目录，输入输出根目录不得相同（含 `--list-only`）。`--clip-overseg` 默认关闭，仅 SAM 3：高分检测几乎落在轨迹内部且轨迹明显更大时，用该检测替换轨迹掩码再写 memory；已有结果不含该标记时视为关闭。三个入口的 `--direction both` 从 `frame_count // 2` 帧初始化，在同一 session 中先正向、再反向传播，合为一套正序结果，不创建 `forward/`、`backward/` 子目录；GT 汇总按 `both` 统计，旧方向子目录不复用也不删除。`video_utils.py` 提供视频工具，`compare_gt_masks.py` 提供 GT 参数、评测和批次汇总。`process_learned_prompt_videos.py`、`process_mano_prompt_videos.py`、`process_dataset_videos.py` 统一通过 `--compare-gt --gt-mask-name NAME` 启用评测；`--gt-dir-name` 默认 `masks_sam3`，`--compare-skip-frames` 默认 `0`，GT 文件名不从提示词、目标 ID 或手侧推导。

- `process_dataset_videos.py`：递归查找 `color.mp4`，用文本提示执行 SAM 3/3.1 分割并保留目录结构。SAM 3 支持 `--text-prompt "left hand" "right hand"` 真正多文本检测 batch，共享逐帧图像 backbone/FPN，各提示词独立跟踪与输出；SAM 3.1 仅支持单提示词。多提示词写入 `OUT/<提示词目录>/<视频相对目录>`，单提示词路径不变；完全重复的提示词去重，非法/冲突目录名报错，完整结果按提示词分别复用。多提示词 GT 共用显式文件，各自生成评测与 `gt_summary.json`，不按文本选 GT 标签；OOM 不自动拆批或串行回退，失败继续下一视频。`--save-detector` 仅 `sam3`，另写 `detector_masks.mkv` 和 `detector_result.mp4`；与 `--compare-gt` 同开时对比视频增加 Detector 列和 `detector_*` 指标。先用 `--list-only` 核对输入；示例：`python scripts/process_dataset_videos.py --input-root DATA --output-root OUT --version sam3.1 --text-prompt hand --device cuda:0`。
- `process_mano_prompt_videos.py`：同一目录约定，用 `<mano-dir-name>/<left|right>_hand/*.npz` 的全部 MANO 几何做 SAM 3 分割。多个 NPZ 的框/点按帧合并进一次检测；`--start-from-mano-edge` 默认关闭，开启后覆盖 `--direction` 为双向：按每帧原始 MANO box 到四边的最小像素距离选最大值帧（不扩张、不裁剪，越界为负），同帧多框取最小值、并列取最早帧；评分采用 `--box-source`，不受提示模式/间隔限制，必要时仅给起点补一次当前模式提示；全无有效框时 warning 并从第 0 帧开始，metadata 记录策略、起点和距离用于复用校验。`--save-detector` 另写 `detector_masks.mkv` 和 `detector_result.mp4`。示例：`python scripts/process_mano_prompt_videos.py --input-root DATA --output-root OUT --text-prompt "left hand" --hand-side left --device cuda:0`。
- `process_learned_prompt_videos.py`：同一目录约定，用训练好的目标特征文件做无文本 SAM 3 分割（不支持 3.1）。`--rgb-name` 指定 RGB 文件名（默认 `color.mp4`）；`--compare-gt` 启用同级 GT 对比，`--gt-dir-name` 默认 `masks_sam3`，`--gt-mask-name` 在启用评测时必填，`--compare-skip-frames 2` 评测第 0、3、6…帧并生成对比视频及 IoU/Dice 汇总；完整预测可直接复用。详情见 `docs/learned_prompt_training.md`。示例：`python scripts/process_learned_prompt_videos.py --input-root DATA --output-root OUT --learned-prompt outputs/learned_left_hand/best.pt --checkpoint /path/to/sam3.pt --device cuda:0`。
- `process_bidirectional_videos.py`：基础 SAM3 单 GPU、严格零训练的双向 memory 融合。隔离正反向建库，再联合读取记忆并单次解码；源库不写回，不再使用 Viterbi。默认物理倒序；`--chunk-frames 120 --context-frames 30` 启用重叠分块，核心帧唯一归属。例：`python scripts/process_bidirectional_videos.py --input-root DATA --output-root OUT --text-prompt "left hand" --device cuda:0 --checkpoint /path/to/sam3.pt`。详细限制和消融见 `docs/bidirectional_hand_segmentation_research.md`。
- `process_hand_video_with_wilor_prompts.py`：处理单个目录中的 `color.mp4` 和 `MANO_wilor_occlusion/hand_joints_occlusion.jsonl`，用 WiLoR 关节迭代修正指定侧手的 SAM 分割。正点必须是落在上一轮目标掩码内的可见目标手关节；`--segmentation-passes` 默认为 2，设为 1 时仅运行文本分割。示例：`uv run scripts/process_hand_video_with_wilor_prompts.py --input-dir DATA --output-dir OUT --hand-side left --version sam3 --segmentation-passes 2 --device cuda:0`。
- `compare_dataset_videos.py`：按共同的 `result.mp4` 相对路径（可用 `--video-name` 指定其他文件名），将多个提示词输出拼成带标签的对比视频。至少传入两个输出根目录，并用 `--output-dir` 指定目标；自动布局不合适时传 `--grid 2x2`。
- `qualitative_test_interactive.py`：在桌面窗口中逐帧检查、修正并双向传播实例掩码，输出无损掩码、结果视频和元数据。运行：`python scripts/qualitative_test_interactive.py --video INPUT.mp4 --output-dir OUT --device cuda:0`；批量运行用 `--input-root DATA --output-root OUT --rgb-name color.mp4`，递归排序并逐个打开，输出去掉末尾 `raw` 层再追加 `--output-subdir`（默认 `masks_sam3`）和提示词目录（空格替换为下划线，如 `left hand` → `left_hand`），其他层级保留；单视频输出不受影响。整批复用一个模型，每视频/分块独立 session。批量跳过匹配的完整成功结果，不完整或未确认结果自动重跑；完整但设置不匹配的结果及单视频已有输出仍需 `--overwrite`。`--clip-overseg` 默认关闭且仅 SAM 3：高分检测几乎落在轨迹内部且轨迹明显更大时，用该检测替换轨迹掩码再写 memory；`metadata.json` 记录该开关，旧结果没有该字段时视为关闭，批量下开关不一致会重新推理；失败继续并在末尾汇总，`S` 跳过当前整条视频、不保存本次修改或复制加载结果，并继续下一个（单视频直接结束）；关闭编辑窗口或 Ctrl+C 停止整批，分块仍需终端 `ok` 确认。新增 `--load-existing`（与 `--overwrite` 互斥），批量模式即使带该参数或 `--existing-mask-dir`，也先跳过 output 中完整成功且设置匹配的结果以续审；input 侧预分割结果不因此跳过，未完成序列仍加载审查。重新审查已完成输出用单视频模式。单视频和批量均可加载完整且设置匹配的结果而不自动传播；无结果正常推理，残缺/不匹配结果报错且保留文件。加载时暂停并显示全视频突变曲线；编辑从当前帧初始化 tracker，手动传播严格遵守范围、双向各自使用同一起始掩码，已确认修正帧作为条件保留；分块结果按合并后的全视频编辑。优先加载 output 已有结果，output 无结果时再按同样的去末尾 `raw`、`--output-subdir` 与提示词目录规则查找 input 侧结果；单视频同样查找；输入侧还兼容 RGB 文件同级的 `raw/<output-subdir>/<提示词目录>`。支持无 `interactions.json` 的三文件批量推理结果（无人工交互），交互脚本自身的结果仍要求完整日志。从 output 加载后仅浏览按 `Q` 不写文件；从 input 加载后按 `Q` 无修改就原样复制已有结果文件到 output（三文件批量结果另补空交互日志），有修改就在 output 保存，input 原结果不变；关闭窗口或 Ctrl+C 不复制。修改后保存保留标签编号，未确认草稿取消；加载模式 `C` 只清点，不重传。不传 `--existing-mask-dir` 时找不到结果先打印查找路径再正常推理；显式传 `--existing-mask-dir DIR`（需 `--load-existing`）则只读每个 RGB 文件所在目录的 `DIR/<提示词目录>`，不优先读 output 或猜测 `raw` 上级；DIR 是无 `..` 的非空相对路径，与输出路径设置独立。显式路径缺失/结果不完整/设置不匹配时报错且不启动模型、不回退推理，已有输出仍受保护。失败立即打印并保留末尾汇总。加载浏览不需要 CUDA/权重；编辑仍需 CUDA，采用 tracker-only 推理，不恢复旧 memory 或重新文本检测。需图形环境。
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

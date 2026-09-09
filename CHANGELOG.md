# 改动日志：SAM3 → 无文字左右手分割第一版

## 基线

- 上游仓库：`https://github.com/YellowOrz/sam3`。
- 上游提交：`660a5e9 Fix B001: Replace bare except with except Exception`。
- 原始行为：字符串经 `SimpleTokenizer` 与 `VETextEncoder` 变为 `[L,B,256]`，视频推理通过 `add_prompt(text=...)` 接收用户文字。

## 1. 建立实验分支

- 分支：`sam3-learnable-tokens`。
- 原因：保留 `main` 基线，隔离左右手 token 实验。

## 2. 新增 LearnableClassTextEncoder

- 文件：`sam3/model/learnable_text_encoder.py`。
- 第一版类别：`left_hand`、`right_hand`。
- 默认每类 `K=1` 个 256 维 token；支持后续 `K=4/8`。
- 输入字符串只作为类别索引，不经过 tokenizer/Text Transformer。
- 输出保持 SAM3 接口：features `[K,B,256]`、mask `[B,K]`、embeds `[K,B,256]`。
- `visual` 与 `<text_placeholder>` 为零 token，仅兼容视频状态机占位。

## 3. 接入构建器

- 文件：`sam3/model_builder.py`。
- `build_sam3_image_model` 与 `build_sam3_video_model` 新增：
  - `text_encoder_type="ve" | "learnable_class"`；
  - `tokens_per_class=1`。
- 默认 `ve` 行为保持不变。
- learnable 模式加载 checkpoint 时忽略原 `language_backbone` 权重，视觉、decoder、tracker 等权重继续加载。
- 训练后 checkpoint 中的 `language_backbone.class_tokens` 会被保留并重新加载；只过滤旧 tokenizer/Text Transformer/resizer 权重。

## 4. 接入 Video Predictor

- 文件：`sam3/model/sam3_video_predictor.py`。
- 透传 `text_encoder_type` 和 `tokens_per_class` 到视频模型 builder。

## 5. 冻结策略

- `freeze_for_learnable_class_tokens(model)` 冻结全部旧参数，仅启用 `class_tokens`。
- 训练时保留梯度穿过冻结网络的路径，但 optimizer 只接收 class token。

## 6. 自动左右手视频脚本

- 文件：`scripts/run_automatic_hands.py`。
- 用户不输入文字，脚本内部依次运行 `left_hand` 和 `right_hand`。
- 复用 SAM3 原有视频 session、传播与 tracker。
- 输出左右手颜色叠加后的 MP4。
- 当前 token 是随机初始化，脚本验证的是流程，不是分割精度。

## 7. Loss 决策

- 未修改分类、框、GIoU、mask focal 和 dice loss。
- 训练时仍需提供左右手真值 mask，并为缺失左右标签设置 ignore，不能错误当成空目标。
- 后续训练 YAML 必须启用 segmentation head 与 `Masks` loss。

## 8. 测试与验证

- 12 项左右手 token、builder、冻结、梯度和 predictor 测试全部通过。
- 3 项自动视频可视化测试全部通过。
- 一帧 `1280×720` 临时视频端到端流程通过：两个 session 完成 start、add prompt、propagate、close，并生成非空 MP4。

## 下一步

1. 转换视频 RGB、left mask、right mask、valid 标志和 frame/instance ID。
2. 每帧构造左右手两个 `FindQuery`，缺失标签使用 ignore。
3. 创建启用实例 mask loss 的训练 YAML。
4. 在 4–16 个视频片段上完成 K=1 小样本过拟合。
5. 比较随机初始化、Text teacher 初始化和 K=1/4/8。
6. 第一版稳定后再增加 contact object 与 MANO geometry。

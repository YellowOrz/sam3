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

## 9. DexYCB right-hand 训练链路

- 新增 `scripts/unified_to_sam3.py`，将统一集单视角的 `rgb.mkv`、`mask.mkv`、`instances.json` 导出为 JPEG 与 COCO-RLE。
- 新增 `scripts/export_unified_manifest.py`，按 train/val/test manifest 合并导出右手正样本。
- 统一集类别名自动映射：`hand_left/right` → `left_hand/right_hand`。
- 历史导出统计：train 20275、val 2879、test 2402 个手可见帧。当时受统一转换器硬编码影响，全部误记为右手；“没有左手正样本”的旧结论作废，见第 10 节。
- K=1、1000 步试训：前 50 步平均 loss 0.4582，后 50 步 0.2395。
- 200 张验证图近似 Dice 0.8105；200 张测试图近似 Dice 0.7779。
- 修复 ViT MLP：训练时使用可反向传播的普通 Linear/activation 路径；推理 fused 路径在进入第二个 Linear 前恢复权重 dtype。

## 10. DexYCB 双手数据重建与两轮 token 训练（2026-09-10）

- 新增 `scripts/build_dexycb_manifests.py`：以 `sequence.json.extra.mano_sides` 为手别依据，保留原类别供溯源；不能从 MANO 标定名称的 `_right` 后缀推断物理手别。
- 当前覆盖 50 个序列、400 个视角，其中 30 个左手序列、20 个右手序列；沿用 subject-disjoint 划分。
- 更新导出器：固定 `left_hand=1`、`right_hand=2`，仅在内存中纠正旧手别；新增 `--include-empty` 和 `--min-mask-area`，不修改共享数据。
- 本次导出使用面积阈值 64：小正样本整帧跳过，不改成负样本；无可见手帧保留。train/val/test 分别为 23265/2909/2911 张图片、20212/2876/2401 条标注、3053/33/510 张负样本。
- 新增 `scripts/validate_sam3_coco.py`，检查类别、ID、图片存在性/尺寸、标注引用、面积/bbox 合法性、负样本数量及抽样 RLE 解码。
- 更新训练器：每图同时生成左右查询，仅优化 class tokens；正确侧使用 mask、box、classification 和 presence loss。`use_presence=True` 时，无 GT 查询的逐检测 query 分类损失被屏蔽，对侧和无手图通过 presence 负监督。
- 支持多 epoch、batch size、BF16、确定性逐轮遍历、左右梯度检查；原子保存定期/每轮 checkpoint，并校验恢复配置和训练标注 SHA256。
- 新增服务器启动脚本 `scripts/run_bilateral_training.sh`：K=4、batch=2、BF16、两轮、lr=0.01，每 250 batch 保存恢复点；脚本含本实验服务器路径，迁移时需调整。
- 新增 `scripts/evaluate_bilateral_tokens.py`：支持多个 token checkpoint 与原始 VE encoder 比较，左右及无手分组、按预测分数选 mask、Dice/IoU、presence/联合分数、误检率、阈值扫描、叠图和样本溯源。已修复 RLE 分支语法；端到端 GPU 验收与正式指标尚待完成。prompt 二分类 AP 不等于 COCO mask AP。
- 保留 `scripts/visualize_learnable_tokens.py` 作为历史右手可视化工具；它使用 GT matcher 选择 mask，其结果不能直接视作实际检出效果或双手评估结论。
- 新增/更新 manifest、类别映射及训练恢复相关测试。数据集说明保存在服务器派生数据目录 `sam3-dexycb-bilateral-v1/README.md`，不随本代码仓库提交。
- 源统一转换器的 side/track 修复补丁已备份，但受文件权限限制尚未应用；本次派生数据已在导出层纠正手别。训练权重、数据和临时备份不提交到 Git。

## 下一步

1. 完成双手评估脚本 GPU 冒烟，比较第一轮、第二轮与原始 VE encoder。
2. 用 val 选择模型与阈值，test 用于最终报告；重点检查对侧误检和正确侧漏检。
3. 在基线结果支持下比较学习率调度、按侧/序列均衡采样及安全的数据增强。
4. 后续再比较初始化方法与 K=1/4/8，并扩展视频、contact object 与 MANO geometry。

# SAM3 可学习目标特征模式

这是基础 SAM3 的可选模式。普通模式不传 `learned_prompt_path`，原来的文本编码器、训练入口、图片和视频 API 继续使用。新模式不构建 text encoder，只有替代文本编码结果的特征可以训练；视觉编码器、检测器、分割头和视频跟踪器的原有参数全部冻结。首版不包含 SAM3.1。

每个目标单独初始化、单独训练、保存为独立文件。训练左手时只用配置中左手类别的实例 GT，其他类别不参与损失；仍保留没有左手的图片作为负样本。图像里的其他物体不需要删除。掩码的语义和边界完全由数据集规定，代码没有手部专用规则。

## 1. 准备初始化特征

使用原有 SAM3 环境并安装训练依赖：`pip install -e ".[train,dev]"`。PyTorch/CUDA 的安装要求沿用 [原训练说明](../README_TRAIN.md)。以下命令在仓库根目录执行，路径替换为实际值。

```bash
python scripts/prepare_learned_prompt.py --checkpoint /models/sam3.pt --target-id left_hand --reference-text "left hand" --init text --device cuda --output outputs/left_hand_initial.pt
```

随机初始化：

```bash
python scripts/prepare_learned_prompt.py --checkpoint /models/sam3.pt --target-id left_hand --reference-text "left hand" --init random --random-std 0.02 --seed 0 --device cuda --output outputs/left_hand_random.pt
```

这一步单独构建并加载原始文本编码器。文本初始化保存其投影后的 `32 × 256` 特征和有效 token mask；随机初始化使用相同参考文本的 mask，将特征替换为给定标准差的高斯随机数。两种方式都不改变有效 token 布局。之后训练和推理只读取特征文件，不接收自然语言提示、不加载 text encoder。

只有有效 token 对应的数值是参数；padding 位置作为固定 buffer 保存，不受梯度或 weight decay 影响。因此可训练参数量是 `有效 token 数 × 256`，接口始终恢复为 `32 × B × 256`。文本初始化在训练前精确保留原始特征和 mask。

特征文件记录目标 ID、初始化信息和基础 checkpoint 的 SHA-256；加载时检查基础权重一致性。初始化工具不会覆盖已经存在的文件。不同目标使用不同输出路径。

## 2. 用原生数据格式训练

复制并编辑 [配置](../sam3/train/configs/learned_prompt.yaml)。数据使用 SAM3 自带 COCO 加载器支持的 `images / annotations / categories` JSON 和图像目录，mask 支持 polygon、压缩 RLE 和未压缩 RLE。

- `category_id`：数据集中的目标类别 ID，训练集和验证集应保持相同映射。
- `target_id`：初始化特征文件中的 ID，仅用于选择和校验，不送入文本编码器。
- `initial_feature`：上一步导出的特征文件。
- `train`、`val`：对应图像目录和标注 JSON。按原始视频划分数据，避免相邻帧跨训练和验证集。
- `checkpoint`、`device`、`output_dir`：基础模型、训练设备和输出位置。

选中类别的标注必须有实例 mask；如果没有 `bbox` 字段，会从 mask 计算。保留空目标图片，使用原生 crowd 过滤逻辑。数据在内部只生成选中类别的 query，不会把其他类别 GT 混入目标训练。

```bash
python -m sam3.train.learned_prompt --config sam3/train/configs/learned_prompt.yaml
```

这是独立的单设备训练入口，复用原有数据、变换、collator、实例匹配及 loss，不改变原 Trainer 和优化器的默认规则。图像缩放到基础 SAM3 的 1008 分辨率。损失包括 mask focal、Dice、框、分类和 presence；具体权重在 YAML 中可调。负样本通过原生 presence 损失监督目标不存在。

优化器只有当前目标的特征参数。训练保留检测器的训练输出和匹配逻辑，视觉塔及显式 Dropout/BatchNorm 模块设为 eval；不会用全模型 `no_grad()` 截断梯度。验证使用 eval 模式和相同 GT 损失，记录的是验证损失，不是分割准确率或视频遮挡指标。

输出：

- `learned_prompt.pt`：最后一轮的轻量特征，可用于推理。
- `best.pt`：验证损失最小的特征；没有验证集时不产生。
- `last.pt`：特征、优化器、完成轮数及随机状态，用于恢复训练。
- `config.yaml`、`metrics.jsonl`：运行配置和逐轮训练／验证损失。

恢复训练时，`epochs` 是希望达到的总轮数。恢复会沿用保存的优化器状态和学习率；如需新的学习率重新微调，可将 `initial_feature` 设为训练后的特征，使用新的输出目录，不传 `--resume`。

```bash
python -m sam3.train.learned_prompt --config sam3/train/configs/learned_prompt.yaml --resume outputs/learned_left_hand/last.pt
```

训练右手时，重新初始化一个目标文件，修改 `target_id`、`category_id`、`initial_feature` 和 `output_dir` 后独立运行。

## 3. 无文本图片推理

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path="/models/sam3.pt",
    learned_prompt_path="outputs/learned_left_hand/best.pt",
    device="cuda",
)
processor = Sam3Processor(model, device="cuda")
state = processor.set_image(Image.open("frame.jpg").convert("RGB"))
state = processor.set_learned_prompt(target_id="left_hand", state=state)
# state["masks"], state["boxes"], state["scores"] 与普通模式一致
```

## 4. 无文本视频推理

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(
    checkpoint_path="/models/sam3.pt",
    learned_prompt_path="outputs/learned_left_hand/best.pt",
    gpus_to_use=[0],
)
session = predictor.handle_request({"type": "start_session", "resource_path": "color.mp4"})
session_id = session["session_id"]
try:
    first = predictor.handle_request({
        "type": "add_learned_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "target_id": "left_hand",
    })
    for result in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
    }):
        pass  # 使用与原 API 相同的逐帧结果结构
finally:
    predictor.handle_request({"type": "close_session", "session_id": session_id})
    predictor.shutdown()
```

一个模型实例只加载一个目标文件；目标 ID 不匹配会报错。内部通过目标 ID 标记原有视频 query 和缓存，标识不经过 tokenizer/text encoder。不要在活跃会话中手工替换参数；切换目标请新建对应 predictor。已有会话的检测、关联、memory 和传播流程继续复用。

普通模式仍使用 `set_text_prompt` 或 `add_prompt` 的 `text` 字段。新模式显式使用 `set_learned_prompt` / `add_learned_prompt`，不会静默把自然语言当成另一个目标。

## 5. 验证与实验边界

无需权重的接口、梯度和恢复测试：

```bash
pytest tests/test_learned_prompt.py tests/test_learned_prompt_contract.py -q
```

服务器可运行真实 SAM3 的正样本、负样本、反向传播和恢复训练 smoke test（Linux shell）：

```bash
SAM3_TEST_CHECKPOINT=/models/sam3.pt pytest tests/test_learned_prompt_gpu.py -q
```

该测试使用合成图像，只验证训练流程；需要 CUDA 和完整训练依赖。视频质量应另用未参与训练的视频评估，在相同基础权重和跟踪设置下对比原始文本、未训练的文本初始化特征、训练后的特征和随机初始化训练后的特征，分别观察掩码边界、可见部分分割和遮挡后恢复。这里没有训练视频时序模块，不能据此宣称遮挡效果已经提升。

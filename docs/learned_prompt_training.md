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

## 2. 用 uni-hoi 统一集训练

复制并编辑 [配置](../sam3/train/configs/learned_prompt.yaml)。默认读取 `~/Datasets/uni-hoi-dataset`（目录布局见 uni-hoi-dataset 仓库的 `docs/hand-object-seg-spec.md`）：按 `metadata/split.json` 的 subject 划分取 `rgb.mkv` / `mask.mkv` / `instances.json`，不抽帧落盘。索引阶段用 ffmpeg 流式解码 16-bit `mask.mkv`（每个 split×kind 只扫一遍，结果写到 `metadata/uni_hoi_index-*.pkl`；多卡会抢同一把文件锁，其余进程直接读缓存）。训练时按帧读 H.264 `rgb.mkv`。需要本机 `ffmpeg`。

当前仓库中的统一集首期主要是 dex-ycb，手别全部为 `hand_right`。默认配置训练右手：初始化时用 `--target-id right_hand --reference-text "right hand"`。

- `kind`：`instances.json` 中的 `hand_left` 或 `hand_right`。只把该 kind 的实例 mask 当作正样本，其他物体留在图像里但不进损失。
- `target_id`：初始化特征文件中的 ID，仅用于选择和校验，不送入文本编码器。
- `initial_feature`：上一步导出的特征文件。
- `dataset_root`、`train.split`、`val.split`：统一集根目录和 `split.json` 中的划分名。同一 subject 的序列不会跨训练和验证集。
- `checkpoint`、`device`、`output_dir`：基础模型、训练设备（单卡如 `cuda:1`，多卡如 `cuda:1,2,3`）和输出位置。

无该侧手的帧保留为负样本（含 dex-ycb 序列开头无手帧）。掩码语义是 modal 可见像素：`frame_map` 覆盖但像素全空的实例会丢掉，该帧若没有其它正样本则按负样本处理。仍使用原生 crowd 过滤。数据在内部只生成选中 kind 的 query。

```bash
python -m sam3.train.learned_prompt --config sam3/train/configs/learned_prompt.yaml
```

若要改回 COCO JSON，在 `train` / `val` 里写 `images` 和 `annotations`（mask 支持 polygon、压缩 RLE 和未压缩 RLE），加载器会走原来的单类别 COCO 路径。

这是独立训练入口，复用原有数据、变换、collator、实例匹配及 loss，不改变原 Trainer 和优化器的默认规则。`device` 写一张卡（如 `cuda:1`）时单进程训练；写成 `cuda:1,2,3` 时会在这些物理 GPU 上自动 DDP，每卡 `batch_size` 仍为配置值。图像缩放到基础 SAM3 的 1008 分辨率。损失包括 mask focal、Dice、框、分类和 presence；具体权重在 YAML 中可调。负样本通过原生 presence 损失监督目标不存在。

优化器只有当前目标的特征参数。训练保留检测器的训练输出和匹配逻辑，视觉塔及显式 Dropout/BatchNorm 模块设为 eval。视觉塔在 `no_grad` 下前向：它没有可训练参数，且 ViT MLP 的融合核不可微；检测器与分割头仍走梯度，以便更新学习特征。验证使用 eval 模式和相同 GT 损失，记录的是验证损失，不是分割准确率或视频遮挡指标。

输出：

- `learned_prompt.pt`：最后一轮的轻量特征，可用于推理。
- `best.pt`：验证损失最小的特征；没有验证集时不产生。
- `last.pt`：特征、优化器、完成轮数及随机状态，用于恢复训练。
- `config.yaml`、`metrics.jsonl`：运行配置和逐轮训练／验证损失。
- `tensorboard/`：训练时每 `tensorboard_interval` 个 batch 写入 running train loss，每个 epoch 结束再写入全局聚合的 `train/*` 和 `val/*`。查看：`tensorboard --logdir outputs/learned_right_hand/tensorboard`。

TensorBoard 的 Images 页签中，`val/images/*` 按从左到右展示原图、GT 掩码叠加、预测掩码叠加（绿色，多个目标实例取并集）。图像使用验证输入的 1008 × 1008 尺寸，仅主进程写入，step 与该轮 loss 使用同一累计训练步数。YAML 中 `val_visualization_interval: 1` 表示每轮可视化，设为 n 则在第 n、2n…轮写入；`val_visualization_max_images: 8` 固定记录验证集前 8 张，设为 0 关闭；`val_visualization_threshold: 0.5` 为预测置信度阈值，沿用推理中的分类概率乘 presence 概率，掩码缩放后按概率 > 0.5 二值化。空 GT 或无预测时，对应面板保留原图。没有验证集时不写图像，验证 loss 仍按原有频率记录。

恢复训练时，`epochs` 是希望达到的总轮数。恢复会沿用保存的优化器状态和学习率；如需新的学习率重新微调，可将 `initial_feature` 设为训练后的特征，使用新的输出目录，不传 `--resume`。

```bash
python -m sam3.train.learned_prompt --config sam3/train/configs/learned_prompt.yaml --resume outputs/learned_right_hand/last.pt
```

训练左手时，重新初始化一个目标文件，把 `kind` 改为 `hand_left`，并修改 `target_id`、`initial_feature` 和 `output_dir` 后独立运行。当前 dex-ycb 没有左手正样本，那些帧会全部作为负样本。

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

批量处理 `color.mp4` 目录树时用与文本脚本相同的抽帧、传播和输出布局：

```bash
python scripts/process_learned_prompt_videos.py \
    --input-root DATA \
    --output-root OUT \
    --learned-prompt outputs/learned_left_hand/best.pt \
    --checkpoint /models/sam3.pt \
    --device cuda:0
```

省略 `--target-id` 时从特征文件读取；传入的 ID 必须与文件一致。`--direction both` 的独立正反向输出与 `process_dataset_videos.py` 相同。

一个模型实例只加载一个目标文件；目标 ID 不匹配会报错。内部通过目标 ID 标记原有视频 query 和缓存，标识不经过 tokenizer/text encoder。不要在活跃会话中手工替换参数；切换目标请新建对应 predictor。已有会话的检测、关联、memory 和传播流程继续复用。

普通模式仍使用 `set_text_prompt` 或 `add_prompt` 的 `text` 字段。新模式显式使用 `set_learned_prompt` / `add_learned_prompt`，不会静默把自然语言当成另一个目标。

## 5. 验证与实验边界

训练 YAML 支持 `save_every_n_epochs: 5`：每完成 5 个 epoch，额外保留 `epoch_0005.pt`、`epoch_0010.pt` 等快照，包含提示特征和优化器、epoch、随机状态，可传给 `--resume`。设为 `0` 或省略此项时不保留编号快照。`last.pt` 和 `learned_prompt.pt` 仍每个 epoch 更新，`best.pt` 仍在验证总 loss 改善时更新；最后不足一个间隔的训练结果保存在 `last.pt` 中。

无需权重的接口、梯度和恢复测试：

```bash
pytest tests/test_learned_prompt.py tests/test_learned_prompt_contract.py tests/test_uni_hoi_learned_prompt.py -q
```

服务器可运行真实 SAM3 的正样本、负样本、反向传播和恢复训练 smoke test（Linux shell）：

```bash
SAM3_TEST_CHECKPOINT=/models/sam3.pt pytest tests/test_learned_prompt_gpu.py -q
```

该测试使用合成图像，只验证训练流程；需要 CUDA 和完整训练依赖。视频质量应另用未参与训练的视频评估，在相同基础权重和跟踪设置下对比原始文本、未训练的文本初始化特征、训练后的特征和随机初始化训练后的特征，分别观察掩码边界、可见部分分割和遮挡后恢复。这里没有训练视频时序模块，不能据此宣称遮挡效果已经提升。

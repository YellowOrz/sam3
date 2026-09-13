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
python scripts/prepare_learned_prompt.py --checkpoint /models/sam3.pt --target-id left_hand --init random --num-tokens 4 --random-std 0.02 --seed 0 --device cuda --output outputs/left_hand_random.pt
```

文本初始化单独构建并加载原始文本编码器，保存其投影后的 `32 × 256` 特征和有效 token mask。随机初始化不构建文本编码器，不接收 `--reference-text`，必须用 `--num-tokens` 指定有效 token 总数（1–32），前 N 个位置有效，其余为 padding，特征使用给定标准差的高斯随机数。这里的数量不是单词数，也不会自动加起止 token；要匹配默认 tokenizer 的 `left hand` / `right hand` 布局，应设为 4（两个文本 token 加两个起止 token）。随机模式在 CPU 上生成特征，checkpoint 仅用于记录 SHA-256。之后训练和推理只读取特征文件，不接收自然语言提示、不加载 text encoder。

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

优化器只有当前目标的特征参数。训练保留检测器的训练输出和匹配逻辑，视觉塔及显式 Dropout/BatchNorm 模块设为 eval。视觉塔在 `no_grad` 下前向：它没有可训练参数，且 ViT MLP 的融合核不可微；检测器与分割头仍走梯度，以便更新学习特征。验证使用 eval 模式和相同 GT 损失，同时记录分类 F1 和目标存在判断准确率；这些指标不代表像素级分割准确率或视频遮挡效果。

输出：

- `learned_prompt.pt`：最后一轮的轻量特征，可用于推理。
- `best.pt`：验证损失最小的特征；没有验证集时不产生。
- `last.pt`：特征、优化器、完成轮数及随机状态，用于恢复训练。
- `config.yaml`、`metrics.jsonl`：运行配置和逐轮训练／验证损失。
- `tensorboard/`：训练时每 `tensorboard_interval` 个 batch 写入 running train loss，每个 epoch 结束再写入全局聚合的 `train/*` 和 `val/*`。查看：`tensorboard --logdir outputs/learned_right_hand/tensorboard`。

**TensorBoard：记录频率与横轴**

在 Scalars 页签查看 `train/<名称>` 和 `val/<名称>`。横轴 step 是累计训练 batch 数（每次优化器更新加 1），不是 epoch 或图片数；验证不增加 step，恢复训练时沿用保存的 step。

- 训练中每 `tensorboard_interval` 个 batch（默认 100）记录一次本轮开始至当前的样本数加权平均，不是最近 100 个 batch 的滑动平均。多卡时这些中途点只反映主进程的数据。
- 每轮结束在同一组 `train/*` tag 写入整轮统计，并写入 `val/*`；各 batch 的返回值先乘该 batch 图片数，累计后除以总图片数，多卡会汇总所有进程。验证集当前在每个进程完整遍历，汇总包含重复验证样本。
- `metrics.jsonl` 保存同一份逐轮统计。未配置验证集时没有 `val/*`；图像记录间隔不影响每轮验证标量的记录。

**Loss：越低越好**

下面名称均可加 `train/` 或 `val/` 前缀。各分项记录的是未乘 YAML 对应权重的值（`_o2m` 的额外缩放见后文），权重用于组成 `core_loss`。

| 名称 | 含义 | YAML 权重（当前配置） |
| --- | --- | --- |
| `core_loss` | 实际反向传播的总损失，合并主输出及存在的辅助／一对多分支；`best.pt` 按 `val/core_loss` 最小值保存。 | 各项加权求和后，另乘当前 find batch 大小的平方根。 |
| `loss_mask` | 匹配实例的像素级 sigmoid focal loss，监督前景／背景，当前使用 alpha=0.25、gamma=2。 | `loss.mask: 20.0` |
| `loss_dice` | 匹配实例的 soft Dice loss，衡量预测概率掩码与 GT 的重叠；不是二值化后的 Dice 指标。 | `loss.dice: 1.0` |
| `loss_bbox` | 匹配预测框与 GT 框的归一化中心坐标及宽高的 L1 误差。 | `loss.box: 5.0` |
| `loss_giou` | 匹配框的 `1 − GIoU`，监督框的重叠和几何位置；不是掩码 IoU。 | `loss.giou: 2.0` |
| `loss_ce` | 检测 query 的 IoU-aware 二分类损失：匹配正例使用由分类概率和框 IoU 构造的软标签，未匹配 query 为负例。无目标图片上的这项损失被屏蔽。 | `loss.classification: 20.0` |
| `presence_loss` | 图片中是否存在有效目标的损失；当前 alpha=0.5、gamma=0，相当于缩放后的二元交叉熵。没有目标的负样本主要由此监督。 | `loss.presence: 20.0` |

框和掩码损失按本 batch 的 GT 实例数归一化（分母至少为 1）。不同分项尺度和权重不同，不宜只凭曲线数值大小判断哪项更重要；`core_loss` 包含辅助分支和 batch 缩放，也不等于面板中六个主分项直接相加。训练与验证输出分支可能不同，应优先比较各自随训练的趋势。

**指标：越高越好，但不等于分割质量**

| 名称 | 含义与解读 |
| --- | --- |
| `ce_f1` | 将检测 query 分类概率按默认 0.5 阈值二值化，与实例匹配得到的正负标签比较，计算 `2TP / (2TP + FP + FN)`。使用分类概率本身，不乘 presence；不是像素 F1，也不是最终推理结果的实例评测。 |
| `presence_dec_acc` | 将 presence 概率按 `> 0.5` 判为存在目标，与 GT 是否有有效目标比较，记录正确比例。负样本较多时，仅预测不存在也可能取得较高准确率，需结合图像检查。 |

以上无后缀指标通常在 0–1 范围内，不参与 `core_loss` 加权。记录值是 batch 指标的样本数加权平均；尤其 `ce_f1` 并未先汇总全数据集 TP/FP/FN 再计算。当前入口不记录像素 IoU、像素 Dice、mAP、视频跟踪或遮挡恢复指标；视频 IoU/Dice 请使用第 4 节的 GT 对比流程。

**名称后缀：同一项来自哪个输出分支**

- 无后缀：最终主输出。
- `_aux_0`、`_aux_1` 等：中间 decoder 层的辅助输出，编号从 0 开始；`_fs` 表示 first-stage 输出（模型提供时才有）。当前 mask loss 不对辅助输出计算，因此不一定有对应的 mask／Dice 曲线。
- `_o2m`：一对多分支，允许一个 GT 匹配多个预测；可与前面的后缀组合，例如 `loss_ce_aux_0_o2m`。当前 wrapper 对该分支返回的所有值统一乘 `2.0`，包括 `ce_f1_o2m` 等诊断指标，因此这类 F1／准确率可能超过 1，不能直接当作比例或百分比。缺少 presence 输出的分支会记录为 0，不能据此判断 presence 学习失败。

具体 tag 随训练／验证实际提供的输出分支而定，不保证两侧完全一致。

**Images：原图、GT 与预测对照**

Images 页签的 `val/images/000`、`val/images/001` 等对应固定的验证集前 N 张图。每张记录从左到右是原图、GT 掩码叠加、预测掩码叠加；每个面板为验证输入的 1008 × 1008，整张三联图为 3024 × 1008。GT 和预测均将当前目标的多个实例取并集，再用 50% 透明度的绿色叠加，因此无法区分实例 ID。空 GT 或无保留预测时，对应面板保留原图。

| YAML 配置 | 作用 |
| --- | --- |
| `val_visualization_interval: 1` | 每轮写图；设为 n 则在第 n、2n…轮写入。 |
| `val_visualization_max_images: 8` | 最多记录固定的验证集前 8 张；设为 0 关闭图像记录。 |
| `val_visualization_threshold: 0.5` | 保留 `sigmoid(pred_logits) × sigmoid(presence_logit_dec) > 阈值` 的预测实例；仅影响可视化，不改变 loss 或上述指标。 |

保留实例的 mask logits 先双线性缩放到面板大小，再 sigmoid，并按概率 `> 0.5` 二值化，最后取并集。图像仅主进程写入，step 与该轮标量相同；没有验证集时不写图像。看图时重点比较漏分、误分和边界偏差；固定少量样本只能用于定性检查，不能代表全验证集质量。

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

用 `--rgb-name` 选择递归查找的 RGB 文件名，默认 `color.mp4`；例如另一类数据使用 `--rgb-name rgb.mkv`。文件名与 GT 目录名参数只接受单个名称，不接受路径或通配符。建议先加 `--list-only` 核对输入。

启用 GT 对比：

```bash
python scripts/process_learned_prompt_videos.py \
    --input-root DATA \
    --output-root OUT \
    --rgb-name color.mp4 \
    --learned-prompt outputs/learned_left_hand/best.pt \
    --checkpoint /models/sam3.pt \
    --device cuda:0 \
    --compare-gt \
    --gt-dir-name masks_sam3 \
    --gt-mask-name left_hand.mkv \
    --compare-skip-frames 2
```

每个 RGB 视频同级的 `masks_sam3/` 是默认 GT 目录，可通过 `--gt-dir-name` 修改。启用 `--compare-gt` 时必须显式提供 `--gt-mask-name`，例如 `--gt-mask-name left_hand.mkv`，不再根据目标 ID 推导。三个入口 `process_learned_prompt_videos.py`、`process_mano_prompt_videos.py`、`process_dataset_videos.py` 使用同一套 GT 参数和评测逻辑，具体命令见 [README](../README.md#gt-evaluation-for-video-scripts)。GT 是与 RGB 从第 0 帧开始对齐的无损 uint8 灰度标签视频；0 为背景，所有非零实例取并集。预测也取前景并集，不匹配实例 ID。

`--compare-skip-frames 2` 表示每取一帧跳过两帧，视频和指标均选择原始第 0、3、6…帧；默认 0，即全部比较。模型仍逐帧推理。CSV 保留原始帧号及时间，对比视频帧率为 RGB 帧率除以 3，维持采样帧间的原始播放速度（末帧可能因采样间隔多显示不足一个间隔）。`--max-frames N` 仅评测前 N 帧范围，GT 本身仍须与完整 RGB 的声明帧数一致。

每个预测输出目录新增：

- `comparison.mp4`：原图、预测叠加、GT 叠加三联视频，显示原始帧号及该帧 IoU、Dice。
- `gt_metrics.csv`：逐采样帧的 IoU、Dice、交集和双方前景像素数。
- `gt_metrics.json`：该视频的逐帧均值 `mean_iou` / `mean_dice`、累计像素指标 `pixel_iou` / `pixel_dice`、采样设置及来源。

输出根目录的 `gt_summary.json` 汇总本次选中序列，正反向分别统计；逐帧均值按所有成功序列的采样帧平均，累计像素指标先汇总交集和前景像素再计算。双方均为空时两项指标均为 1，仅一方为空时均为 0；没有成功评测帧的汇总指标为 `null`。`--direction both` 将对比产物分别放在 `forward/`、`backward/`，视频均按原始时间顺序播放。

已有符合当前目标、方向及帧数设置的完整预测时，直接读取 `masks.mkv` 补做对比，无需加载模型或使用 CUDA；仍需提供有效的特征文件和基础 checkpoint 路径。对比每次重新生成；`--overwrite` 会同时重新执行预测。更换同一目标的训练特征或基础 checkpoint 后，应使用新的输出目录或 `--overwrite`，已有预测完成检查不会校验权重内容。

GT 缺失、无法解码，或尺寸、帧率、帧数不匹配时，不缩放或截断对齐：记录该序列失败，删除过期对比视频和 CSV，保留预测结果，继续处理其他序列，最终返回非零退出码。批次汇总只将成功序列纳入指标，并列出失败原因。


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

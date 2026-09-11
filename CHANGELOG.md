# 改动日志：SAM3 手物分割适配

## 空间适配器训练预算扩展（2026-09-12）

- 新增显式、非覆盖的已完成空间 checkpoint 预算扩展工具；仅允许增加 epochs，保留优化器、RNG、步数和全部原训练配置，并记录源 SHA256。
- 必须完成原预算及轮末验证才能扩展；原训练器严格恢复检查不放宽，运行中的冻结源码不修改。
- 新增状态保留与拒绝不完整、未验证、缩减预算的测试。

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
- 初版 token 是随机初始化，早期视频冒烟只验证流程；2026-09-10 增加训练 token-only checkpoint 加载，见第 11 节。

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

评估后续修正（2026-09-10）：prompt 二分类 AP 按相同分数分组计算，消除并列分数下的输入顺序偏差；叠图只显示通过检出阈值的 mask，同时标记检出状态和阈值后 Dice。新增 RLE 往返、漏检 Dice 和并列 AP 回归测试。

共享 GPU 评估与审查修正（2026-09-10）：

- 新增 `--gpu-memory-fraction`，限制本评估进程的 PyTorch 显存分配并在摘要记录峰值；该限制不包含 CUDA 上下文等开销，启动时仍需检查余量。
- 推理前逐 query 校验实际图片 ID 和类别 ID，阻止数据加载器错误恢复后换图导致预测与 GT 错配；新增替代图片/错类别回归测试。
- 学习 token、原始 VE 下划线 prompt、原始 VE 空格 prompt 的三图 GPU 冒烟完成；batch=1、BF16，PyTorch 峰值分配约 4254 MiB。此小样本仅验证流程，不作为正式精度结论。
- 新增 `scripts/run_bilateral_evaluation.sh`：快照冻结轮次 checkpoint 和评估代码，在新目录保存全 val 对比日志/结果，batch=1、25% allocator 上限、单次最多两小时；支持指定 GPU，适用于 tmux 后台运行。

## 11. MANO 输入准备、视频 token 桥接与评估报告（2026-09-10）

- `scripts/run_automatic_hands.py` 新增 `--token-checkpoint`，在基础视频模型加载后只覆盖 detector 的 class tokens；自动从训练 checkpoint 推断 K，显式 `--tokens-per-class` 冲突时在模型创建前拒绝。无 token checkpoint 时保持默认 K=1。
- 新增 `sam3/model/class_token_checkpoint.py`：验证类别顺序、`(2,K,256)`、浮点有限值及元数据一致性，使用 `weights_only=True` 和 `no_grad()` 安全加载。真实第一轮 checkpoint 的 CPU 加载已验证；视频端到端效果尚未验证。
- 修复视频 prompt/传播异常时的 session 释放，并补充 predictor、视频读写器的异常清理和读取/写入失败检查；OpenCV 改为功能处导入，不影响不需要 OpenCV 的 CPU 测试。
- 新增 `scripts/prepare_mano_sidecar.py`：按 COCO provenance 精确关联当前统一集的序列、视角、帧与权威手别；检查 axis-angle、camera/米制声明、PCA 转换来源和参数有限值。保留物理身份与源可见段 ID，分别统计 mask 可见性与 MANO 有效性。
- sidecar 缺失/不合格 MANO 使用 `valid=false` 和 null 参数，不冒充零姿态。对实际解析的源文件保存 SHA256，发布前再次检查源是否变动，拒绝覆盖已有产物和写入共享统一集。此步骤只验证数据契约，尚未重新计算 PCA 或做网格投影验收。
- 新增独立 `ManoGeometryEncoder`：将规范化的全局旋转、局部姿态、形状、相机平移、手别分别编码为 5 个 256 维 token；无效输入预先清理并屏蔽，防止 NaN 污染梯度。冻结 helper 只开放该 adapter，已有文本 token 同样冻结。
- 新增 opt-in `ManoPrompt` / `ManoAugmentedGeometryEncoder`：完整保留点、框、mask 和 MANO 的 clone/梯度，在原 geometry 输出后附加 MANO tokens；未提供 MANO 时走原路径。必须先加载基座再安装 wrapper，默认 builder 和正在运行的 token 训练不变；这不等于已完成 MANO 分割微调。
- 新增 CPU 报告工具 `scripts/report_bilateral_evaluation.py`：从完整评估摘要生成 Markdown 对照，拒绝混合不同数据指纹、图片集合、阈值或冲突模型标签，明确 top-mask/阈值后 Dice 与 prompt 级无手误检的区别。
- 新增 `docs/finetuning-stages.md`：固定阶段验收和冻结策略；GT MANO 标作 oracle，规划仅手别提示对照，不按 GT 类别只给正确 query 发放 MANO；双向 memory 前先补时间连续的数据和视频基线。
- 上述组件补充 CPU 回归测试。当前运行环境缺少 OpenCV，绘图测试明确 skip，不能当作视频渲染验收；不向 Git 提交数据、模型权重或生成报告。
- 新增 `docs/figures/sam3-mano-architecture` 结构图（PNG、SVG 与可编辑 DOT）：对照原几何分支和新增 MANO adapter，标注张量维度、冻结范围及尚未正式微调的状态；图片可直接复制用于汇报，后续 memory 图沿用同一交付格式。
- 新增配套讲解 `docs/figures/sam3-mano-architecture-notes.md`：包含导师汇报口述稿、指图顺序、张量解释、PPT 三句摘要与常见追问；明确 GT MANO 的 oracle 条件、手别对照和“接口实现不等于精度提升”的边界。
- 讲解稿补充“改造前 / 当前原型”开场与对照表，明确从通用点/框/mask 几何提示扩展到 MANO tokens，保留原几何分支和输出定义，不将其表述为已重写整个 geometry encoder。

## 12. 几何改造教学文档与实验前置条件（2026-09-10）

- 新增 `docs/mano-geometry-explained.md`：从分割目标、token/encoder、原几何分支、MANO 参数与坐标，到逐 query 装配、冻结梯度、缺失输入、oracle 对照、验收和改进路线，提供完整理解材料与代码导航。
- 核对实际代码并澄清：默认 image geometry 未启用可选 mask encoder；无点框仍有图像条件化的 CLS；新增五个 MANO tokens 接在原 geometry 输出之后，不经过原分支内部三层交叉注意力；相同 256 维接口不代表已完成几何空间对齐或精度验证。已有汇报稿增加说明和详细文档入口。
- 根据用户最新约束，明确 learning token 有效性验证是 geometry 训练的前置条件；在结论确认前，不启动 MANO 参数更新任务，包括会更新 adapter 的 smoke。后续固定同一 base/token checkpoint，比较无 MANO、仅 side/valid、完整 MANO；不以 loss 下降或单项检出率上涨代替有效性结论。
- 新数据集搜索与下载分开：可继续官方资料筛选，任何下载须先汇报候选、用途和范围并取得用户同意。
- 新增 `docs/dataset-candidates-2026-09-10.md`：记录已实测可读的 EgoHOS/VISOR 官方资料，区分人工稀疏 GT 与自动插值，说明 MANO、许可和下载可用性未确认的部分；初次检索未下载，后续获准取样见下条。
- 经用户批准完成 VISOR 50 对原始 RGB/稀疏人工 GT 小样本检查，覆盖 10 视频/10 人；官方帧映射、ZIP 配对/长度/CRC32 与 494 项 SHA256 校验通过。实际响应正文 73,321,704 字节，含逐请求预留的保守计费 233,753,832 字节，低于 1,000,000,000 上限；未下载完整 ZIP/视频或启动新数据训练。初查多例 hand 标注包含前臂/衣袖，不能直接替换当前手部范围监督；Pillow 栅格预览不声称与官方 OpenCV 位级一致，原多边形保留。
- 分离可视化：新增 `scripts/render_separated_masks.py`，独立保存 RGB、黑白 GT、各模型/侧别的原始候选与阈值后 mask，以及带标签但不混色叠加的比较图。`evaluate_bilateral_tokens.py` 新增 `--visual-style separate|overlay`，默认 separate，保留显式 legacy overlay；保存样本溯源和独立文件路径。新增 CPU 测试及实际 GT RLE 解码核对。
- 3 个 val 诊断样本的 epoch1/VE 下划线/VE 自然语言实际推理与分离输出完成；独立核验 RGB 像素、GT RLE、二值预测和低分清空逻辑。样本用于展示问题，不代表总体分布。第一轮 2,909 张完整验证完成，不能据此宣称 token 全面改善；第二轮与阈值公平对照尚待完成。
- 评估报告补充可比性限制：旧摘要只记录基座路径，缺少基座 SHA256 与 AMP 字段，不能仅凭路径相同承诺所有数值条件完全一致。
- 补齐前序 CPU 数据准备记录：`build_temporal_manifest.py` 按同视角连续原帧号生成 L=8、stride=8 的 3,817 个 clips，覆盖 29,073 帧；12 个单帧段/尾部明确未覆盖，原 COCO 不删除。`docs/data-preparation-v2.md` 记录各 split 覆盖、负样本与缺口统计，清单不等于已开始时序训练。
- 新增 `scripts/finish_bilateral_validation.py` 与 11 项 CPU 调度测试：等待第二轮完整 checkpoint 和 GPU 1 至少 7,500 MiB 余量，只执行一次 token 验证，再与第一轮/既有 VE 合并报告。快照输入、代码与 renderer，独立输出/日志/状态，失败不重试；单次评估至多两小时且不超过 2026-09-10 21:40 的授权截止，不启动 geometry 或其他训练，不停止已有进程。
- 新增 `scripts/calibrate_bilateral_thresholds.py` 与 10 项 CPU 测试：只在 val、同图片/标注指纹上按 1%/5%/10% 对侧误检预算比较检出和漏检计零 Dice，同分整组处理，保留完整精度阈值与来源。第一轮结果说明 token 在 5%/10% 预算下更好，但在 1% 下不及自然语言 VE，候选 mask Dice 仍较低；分析见 `docs/learning-token-epoch1-results.md`，不自动调整推理阈值或放行 geometry 训练。
- 新增 nakehand 只读审计脚本、4 项 CPU 测试及 `docs/nakehand-dataset-audit-2026-09-10.md`：检查 6 段/18,498 帧和 42 帧同 PTS 抽样，确认左右 mask 是带人工提示的 SAM3 传播伪标签，保留 chunk/实例值 2 的语义；指出双手共现与当前单物理手训练/评估合同不兼容，以及 WiLoR 旋转矩阵、canonical shape、平移单位/离群值和独立 GT 的缺口。未改原始数据，未据此启动训练。
- 为独立短程学习率对照增加训练器可选 `--gpu-memory-fraction`（默认无上限，资源参数不改变恢复时的优化配置）；修正 `--max-steps` 短跑被误报为完整两轮的行为，保存 `_stepN_partial.pt` 并记录实际样本、完成轮次和完整训练状态。补充奇数末 batch/截断状态测试；未改运行中的基线进程、loss 或学习率。
- 根据用户新增 GPU 2/3 授权更新资源约定，截止仍为 21:40。GPU 3 完成 batch=1/BF16 的 token-only 20 步显存探测（含左右正样本与空图、恢复点），PyTorch 峰值分配 4,269.47 MiB，只有 2,048 个类别参数更新；这不是完整训练或精度验收，也不是 MANO 训练。
- 新增 `docs/learning-token-improvement-plan.md`：解释现有 AdamW/固定 LR、初始化尺度、学习率与检出阈值的区别，以及先单因素 LR 对照、再独立比较调度/初始化/采样的顺序。所有候选收益仍须验证；不提前放行 geometry。
- 训练器在前向/更新前核验实际 COCO 图片 ID、左右类别及每图双 query，拒绝底层读取失败后悄悄替换图片；checkpoint/summary 记录当次实际成功样本的 `observed_identity`，恢复时不伪造历史观测。新增可选初始 token SHA256 首步前校验，不改变随机数、loss 或旧恢复配置；修正短跑平均 loss 窗口文案。15 项训练器测试通过，并以真实 CPU collator 核验首 3 张训练图的 batch 1/2 身份。
- 新增 `run_token_lr_pilot.py` / `aggregate_token_lr_pilots.py` 及调度/聚合测试：支持显式单卡单 LR、GPU 2/3 提前试验、原始初值与实际 2,000 样本/4,000 queries 的指纹核验；训练/评估前后检查全部 `sam3/*.py` 文件集合和内容，核心变更即停止并标记不可比。各组完整 val 后按相同误检预算汇总，缺组/失败不宣称三组对照成功。
- 17:50 左右启动独立持久任务：GPU 2 运行 LR=0.003、GPU 3 运行 LR=0.001；GPU 0 的 LR=0.01 控制组排队等待原两轮完成，GPU 1 保留第二轮验证。三组均随机同初值、K=4、batch=1、BF16、2,000 步/样本、固定 loss、每 100 步恢复点；使用全新目录和代码快照，allocator 25%、启动前至少 7,000 MiB 空闲，训练单次至多 3,600 秒、评估至多 2,400 秒且全部受 21:40 截止限制。只是等预算短程试验，尚无新精度结论。
- 最终 CPU 测试共 171 项：170 项通过、1 项因缺少 OpenCV 明确跳过；编译与 `git diff --check` 通过。新增身份校验后的 20 步真实 GPU 测试也通过，初值哈希与原基线相同；未修改正在运行的两轮基线或共享源数据。

## 13. nakehand 人工反馈与外部双手测试（2026-09-10）

- 记录用户对 A（ego142020/575）、B（exo123926/1459）、C（ego142020/1724）现有 mask 的“可接受”反馈，以及 mask 视觉质量可能优于当前数据的判断；只标记这三张人工接受，不升级为全数据或 MANO/深度等字段已验收。
- 用户明确授权先以 nakehand 测试完整 epoch2 与原始 VE text encoder，不训练、不调阈值。新增 `docs/nakehand-external-test-protocol.md`，固定每录像均匀 100 帧共 600 主样本，额外人工认可诊断样本单列，保留 SAM3 辅助参考来源；双手两查询分别监督/计分，缺席侧与无手误检分母分开。
- 新增 `prepare_nakehand_test.py` 与 12 项 CPU 测试：按固定帧号、RGB/双侧 mask PTS 对齐导出，保留非零实例和 video/chunk 作用域；源文件前后 SHA、PNG/RLE 逐像素检查，全部通过后原子发布 `READY.json`。旧 schema 兼容修复前的失败目录保留、不使用，正式数据输出为 `nakehand-systematic600-20260910-v2`。
- 新增 `evaluate_nakehand_tokens.py` 与 14 项 CPU 测试（含真实 SAM3 1008 loader/collator 的双手/单侧/无手四图）；每图双 query 独立计分，固定两个 0.5 阈值，按模型分数选择候选，禁参考标注驱动的交互点/框。分别记录候选/漏检计零 Dice、分侧检出、缺席/空图误检分母和错手 overlap 代理，输出分离 RGB/参考/预测及完整文件指纹。终端仅打印进度和结果路径，完整细分指标留在 JSON。
- 新增 `run_nakehand_evaluation.py` 与 7 项 CPU 测试：等正式数据 READY 和原两轮真实 checkpoint/summary，快照依赖与权重、前后核验核心源码，失败不重试，单次最多 3,600 秒且不得超过 21:40。已启动持久会话 `sam3-nakehand-test`，在 GPU 0 的 LR 控制组真实启动后安排共享评估，额外要求至少 14,000 MiB 空闲与 25% allocator 上限；不改变原训练/LR 数学路径。
- A/B/C 新导出的 RGB 与左右参考 9/9 逐像素匹配已认可样本，三模型真实 GPU 冒烟通过（峰值分配约 4,191.44 MiB）。六个可见手查询中，epoch2 检出 3/6，原始自然语言 VE 检出 6/6；A/C 的 token 右查询候选更贴近左手，低分被过滤。这里只是三张诊断结果，不能外推为 600 帧主测试结论，主测试尚待完成。
- 原双侧 token 两轮基线已完成 23,266 步、46,530 张样本，epoch2 complete/final checkpoint 和训练摘要已保存；结束 loss 下降只作训练记录，不替代第二轮验证或外部测试结论。
- nakehand 固定 v2 数据已发布：602 张不同图片（600 主样本及额外 A/C）、1,103 条标注；600 主样本含 543 双手、13 单手、44 无手，1,099 个正查询及 101 个缺席侧查询。60 个来源文件前后哈希、3,010 个 PNG 哈希和全部 RLE 逐像素复核通过；B 自然处于主样本，不被额外重复计数。18:26 在 GPU 0 启动三模型正式推理，主结论等待完整结果。
- 新增 `report_nakehand_evaluation.py` 与 7 项 CPU 测试：从完整结果核验实际双查询覆盖、主/诊断分组、标注和模型哈希，重算指标一致后生成中文报告及分离图链接；主表及 ABC 同时列候选/高分错侧代理，防止低分抑制掩盖候选错侧。说明达阈不保证物理侧正确、无手只是双侧参考为空。只含 A/B/C 时明确拒绝总体外推，不训练、不调阈值、不覆盖旧报告；真实三图摘要复核通过。
- 18:35 完成原 DexYCB 2,909 张 val 第二轮验证：候选 Dice 0.8140、漏检计零 Dice 0.7024、检出率 85.05%、对侧误检 6.19%，相较第一轮与原自然语言 VE 均有改善。更新阶段与改进文档，不再把第一轮候选退化的结论套用到第二轮；保留旧摘要缺少基座 SHA/AMP 的限制，外部双手泛化与 geometry 放行仍单独验收。
- 18:42 完成 nakehand 三模型外测，比较有效性/核心哈希检查通过：600 主样本 epoch2 候选/漏检计零 Dice 为 0.7529/0.6267，原自然语言 VE 为 0.9701/0.9549；三模型缺席侧均 FP=0/101。token 双手候选错侧代理 239/1086，其中63个达阈输出，主要弱项在 ego/右手，不能只解释成置信度阈值问题。602 图共 3,612 条记录、26组分离可视化独立像素/指标验收通过，正式中文报告已保存到评估目录。
- 新增 `prepare_nakehand_manual_review.py` 与3项CPU测试，seed=20260910分层随机抽取10帧，排除已测试/审核618帧，不按mask或模型效果挑选；生成10张原图/左右参考分离图及2页总览。用户明确反馈“全都可接受 我觉得很不错”，单独保存 `human-review.json`，不改原始导出清单的历史状态；连同ABC共13张手mask获认可，不代表全量/MANO/物体标签验收。
- 人审RGB/mask及生成清单留在本地，新增定向gitignore规则避免作为代码误提交；保留可版本化的 `human-review.json` 回执。文件未删除，原图与独立可视化仍可查看。
- 新增 `docs/token-reliability-and-hand-object-plan.md` 与 `docs/nakehand-split-evidence.md`：澄清固定文字VE输出同样是静态特征、K4不等于模型只有2048参数；说明语义初始化/完整缓存/零增量的受控路线，以及手mask、物体实例和接触关系的不同标签要求。六录像采集身份/关联用户回答未知，录像级split只作提案，不称跨人独立测试，尚未训练nakehand。

## 14. 原 VE 几何基线与语义初始化路线（2026-09-10）

- 根据用户18:48后的明确决定，当前随机类别token退出主线、保留作对照；后续geometry使用原始自然语言VE。更新实验阶段、完整讲解和图稿版本说明，取消geometry依赖当前token有效性这一旧约束，但保留MANO单位/投影、真实接入与无MANO/side-only/full受控对照；不擅自启动正式几何训练。
- 新增3项 `test_mano_with_ve_contract.py` CPU合同测试：使用真实VE类的小型随机配置及真实geometry/prompt路径，验证wrapper/noMANO不改原提示、原文字padding保留、invalid MANO屏蔽及只开放adapter梯度。不等于预训练全SAM3 GPU验收，未执行optimizer step。
- 新增 opt-in `cached_ve_text_features.py` 与11项CPU测试：保留原自然文字特征完整三元输出、32位序列/padding及实际dtype，支持冻结cache和仅4有效位置的零增量（2048参数）。显式安装/恢复，不重建视觉decoder，禁止误用model.train导致冻结基座dropout变化；默认builder不变。
- 新增 `check_ve_prompt_equivalence.py` 与4项CPU测试，并于19:00–19:01完成真实GPU0检查：三图×无MANO wrapper/恢复原VE/冻结cache/zero-delta共12项比较，pred_logits、presence、boxes、全部200候选mask logits均逐元素相同（最大差值0），原输出二值mask全相同。保存基础权重/标注/RGB/tokenizer/源码指纹与缓存权重，峰值约4190.64MiB；没有参数更新，不等于训练后精度改善。
- 三组随机LR短程试验全部完成且初值/顺序/真实2000样本/156核心文件与数据哈希独立复核通过；本预算下LR0.01明显优于0.003/0.001，低LR低误检来自近乎不检出。新增 `docs/learning-token-lr-results.md`，不将早期短程结论外推为语义初始化或几何训练的最优LR。
- 新增独立 `train_ve_initialized_tokens.py` 与8项CPU测试：固定DexYCB train及首2000样本、batch1/BF16/LR0.01/原六项loss，完整VE语义起点仅更新2048个delta参数；独立checkpoint格式、每100步原子恢复、初始缓存/源码/配置/实际身份/AdamW/RNG严格校验，20步可恢复到总2000步，不能称两轮。
- 新增 `evaluate_ve_initialized_tokens.py` 与7项CPU测试：同基座、同新评估脚本比较冻结语义缓存与训练delta，默认只接受实际完成2000样本并跑完整val；诊断checkpoint/subset须显式标记，保留身份/源码/缓存/输出有限性检查，不在nakehand选择模型或阈值。
- 19:14完成语义增量20步真实GPU0训练冒烟，左右梯度20/20步均非零、loss有限、峰值约4009.42MiB，原子partial checkpoint保存；这不是完整2000样本试验或精度提升证据。完整CPU套件247项：246通过、1项OpenCV依赖跳过。
- 20步新checkpoint与冻结VE缓存的三图（val左/右/空）GPU配对评估通过，双query身份与冻结参数版本检查正常，生成分离输出；summary保留diagnostic checkpoint和非完整val标志，不将该冒烟用作精度结论。
- 按19:15–19:25报告约定新增 `docs/daily-progress-2026-09-10.md`，汇总数据纠错、原域/域外结果、LR排除性发现、13张人工认可、语义起点等价性，以及geometry/memory尚未训练等明确边界。
- 新增 `run_ve_delta_pilot.py` 与5项CPU测试：持久任务从真实20步checkpoint恢复，只有实际2000样本完成后才进入完整val冻结缓存/训练delta对照和中文报告；独立日志、源代码/输入快照与哈希、失败不重试，显存余量至少7500MiB、25% allocator，训练/验证阶段2700/3600秒上限并受21:40截止约束。19:21在服务器tmux `sam3-ve-delta` 启动新实验，尚无完整精度结论。
- 最后完整CPU回归252项：251通过、1项OpenCV缺失跳过；日报独立核对后更正时序clips为“最长8帧、含2–7帧短尾”，收紧样本独立性与三图等价检查的外推表述。
- 19:23实际GPU日志确认20步恢复后已继续到40/2000步，左右梯度正常、loss有限；完整val尚未开始，日报按约定在19:25前完成，不提前宣布语义增量精度改善。

## 15. nakehand 语义增量受控训练（2026-09-10，实施中）

- 用户批准将nakehand作为下一阶段训练数据并比较原VE、无约束语义delta和受约束delta。新增预注册 `docs/nakehand-semantic-ablation-v1.md`：落实3/1/2整录像开发划分、仅train更新、val完整对照、保留区不调参，明确人物/session未知与同源SAM3辅助标签限制。
- 固定两组各2000样本、同seed/初始化/顺序/batch1/BF16/LR0.001，唯一对照变量是归一化增量正则权重0/1；不宣称LR或正则最优，不新增随机初始化主线训练。数据/双手训练/评估实现分工进行，尚未据此宣布训练或效果完成；不修改旧运行依赖。
- 新增 `prepare_nakehand_training.py` 全帧录像级导出与7项CPU测试：全量保留零/单/双侧目标和原始实例值、逐帧PTS核对、5 PNG/帧及COCO RLE逐像素验证、60源文件/3实现快照/冻结plan哈希。train/val为9092/3449帧已发布；5957帧开发保留区原导出中断，总READY尚缺，不绕过训练门槛。
- 新增 `recover_nakehand_export.py` 与6项CPU测试：已发布train/val只读全检，独立staging重解码整个holdout，旧PNG仅与源视频像素完全一致才复制复用；旧partial整体改名备份，不删不覆盖源或已发布split。21:15在持久tmux启动CPU恢复，最终READY完成后才放行GPU训练。
- 新增 `train_nakehand_semantic_tokens.py`：严格train/全局帧身份/发布契约，支持每图0/1/2侧，冻结其余网络，仅FP32 delta更新；保存task/anchor/total loss、两侧漂移及真实task梯度历史，合法稀疏零梯度不误判，但首20步和完整100步窗口需两侧均有信号；每100步原子checkpoint，20→2000严格恢复。
- 新增 `evaluate_nakehand_semantic_tokens.py`、`run_nakehand_semantic_ablation.py`、`report_nakehand_semantic_ablation.py` 及CPU回归：三变体同完整3449图val、按自身分数选候选、独立mask与错侧代理，验证实际训练2000样本/同初值顺序/唯一anchor差异/源与数值指纹，未完成不生成正式结论。调度冻结实际执行快照，双GPU独立训练后才进入配对验证。
- 用户明确取消固定截止：新配对runner/evaluator的deadline改为可选且显式时区，保留有限阶段超时、恢复点、25% allocator和GPU余量检查，不修改旧实验历史。当前完整CPU套件298项，297通过、1项OpenCV依赖跳过；新的nakehand GPU冒烟尚待总READY完成。

## 16. 统一手物目标、失败分解与研究方向（2026-09-10）

- 新增 `docs/hand-object-segmentation-goal.md`：按用户确认把正确手侧/手腕边界/接触物体/时序身份作为共同验收目标；text、MANO geometry、memory是限定改动入口，分阶段对照、其余网络冻结，不以结构完成或置信度上涨当成功。
- 新增 `docs/hand-object-research-notes-2026-09-10.md`：基于SAM3、CoOp/CoCoOp/KgCoOp、Boundary IoU、MANO、VISOR/EgoHOS与SAM2Long原始资料，列语义保持、配对侧/条件prompt、边界监督、投影MANO、身份质量memory五项可证伪方案及最小对照/判退。含明天口述稿，未新增下载或geometry/memory训练。
- 20:04旧DexYCB语义初始化无anchor的2000样本实验完成：完整val候选Dice0.8078→0.7044，漏检计零0.4890→0.5796，检出率60.36%→81.36%。记录为检测改善但候选退化，不宣称整体成功；新摘要具备基座SHA/AMP，不照搬旧通用报告的历史缺失提示。
- 更新旧可靠性/划分提案的版本导航，保留历史语境并指向已批准的nakehand实施协议，不把过去“未执行”状态混作当前结论。
- 新增 `analyze_hand_segmentation_failures.py` 与14项CPU测试：将预测像素分为同侧参考、另一手独占区域、双手参考外，报告漏分/precision/recall及作者定义的mask级Boundary IoU（SciPy实现，无OpenCV）；不把参考外像素命名为前臂、不按GT重新选候选。真实26组分离图、312条记录及475个输入哈希前后核验通过，正式报告保存到独立 `nakehand-mask-error-decomposition-20260910-2120`；该诊断子集不是完整600帧总体结论。完整CPU回归更新为312项，311通过、1项依赖跳过。
- 调度审查修复取消固定deadline后的失败等待问题：任一训练组失败立即落盘并唤醒取消仍等GPU的同组任务，GPU等待独立上限1800秒；已实际运行的另组保留原有限阶段时限和checkpoint，不停止其他用户进程。新增4项CPU回归，训练/评估/报告/调度联合37项通过；未改变科学对照配置。
- nakehand恢复完成：18498帧/92490个正式PNG发布，总READY及正式trainer门槛复核通过；复用19265个源解码像素匹配PNG，新编码10520个，唯一中断RGB从源视频重建，原partial整体保留独立备份。记录根READY/三split标注SHA，并实测四种可见性CPU装配。21:33在GPU0/1启动anchor0/1各20步真实冒烟，尚不宣称完整2000步或精度完成。
- 新增 `diagnose_generic_hand_prompt.py` 与10项CPU测试、真实三query装配核验；完成固定8图原VE GPU诊断，同图hand/left hand/right hand三查询，generic保留所有过阈实例、左右各按自身top分数，GT不进入模型提示或候选选择。八图generic均有2个过阈候选，union Dice约0.940–0.995，左右查询也较好；未支持“这批样本原VE普遍不认识手”的假设，不能外推至腕部困难场景。分离PNG/JSON/报告保存在 `sam3-nakehand-experiments/generic-hand-diagnostic-20260910-2131`，无训练、无额外下载。
- 21:35两组nakehand真实20步完成并独立复核：仅2048个delta参数，两侧各20/20步非零task梯度、峰值4010.43MiB；配置仅anchor权重差异、初值和实际样本顺序一致。partial checkpoint保留，固定3图三变体GPU评估链路已启动，不把短程漂移/loss或冒烟指标当作新方法有效证据。最新完整CPU套件326项，325通过、1项OpenCV依赖跳过。
- 21:38三图三变体GPU链路完成，18条实际左右查询/样本前缀/源指纹核验并重算指标一致；显式保留diagnostic/非完整val标签。21:39启动持久tmux `nakehand-semantic-ablation-2139`，冻结202份代码/快照/输入指纹，在GPU0/1各自恢复20→2000步，之后自动跑3449图完整三变体验证与中文报告。实验目录 `sam3-nakehand-experiments/semantic-anchor-v1-20260910-2139`，尚无该完整实验精度结论。
- 21:42确认实际恢复继续到无约束280步/有约束240步，两个100步恢复文件独立CPU加载通过、task梯度两侧均100/100非零、loss有限；任务已越过旧21:40截止继续正常运行。后台自动评估/报告尚待完整训练，未把启动成功当作效果验收。

## 17. 夜间对照与文档便携包准备（2026-09-10）

- nakehand 两组各 2000 步实际训练于21:53/21:55完成并保留 checkpoint；随后完整 val 在安装encoder前因 snapshot直接启动与package导入混用导致类身份 TypeError。不是训练失败或模型质量结论，失败日志保留；没有重新训练。用户转入 RealSense 审计时尚未恢复完整 val。
- 新增 `soft_ve_prompt.py`、`check_soft_ve_prompt.py` 及11项CPU测试：保留原VE，给自然提示的两个词位置添加左右类别共享、零初始化的输入残差（2048参数），非手查询保持原分支；真实GPU等价/梯度probe尚未运行，不宣称该新架构有效。
- 新增 `train_nakehand_prompt_ablation.py` 及11项CPU测试：固定语义起点、样本顺序、LR与冻结范围，准备boundary权重0/4的单因素对照；每100步恢复、20→2000严格校验。尚未运行新GPU训练。
- 新增 `evaluate_hand_boundary_diagnostics.py` 及14项CPU测试：准备完整val三变体的4/8/16像素边带、错侧/参考外分解和预选分离图；新旧checkpoint分别校验，未开始此完整GPU边界评估。
- 新增 `package_review_docs.py`、`sync_review_docs_windows.ps1`、11项CPU测试和 `docs/windows-docs-sync.md`：分类、相对链接、SHA清单、安全新目录与Windows拉取流程；5MB试包已验证。尚无服务器到Windows接收连接，不宣称已拷入用户本地目录。

## 18. RealSense 手物数据只读调研与人工复核（2026-09-10）

- 按用户指定审计 `/data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh`，不执行源run.sh、不修改原始数据、不导出训练集、不启动新训练。
- 新增 `audit_realsense_hand_object.py`：105文件、10段/6204帧；所有26条mask全像素核验，RGB/mask尺寸/帧数一致、PTS量化差最大约0.000333秒；20个深度/预览文件受ACL限制，不把Permission denied报作损坏。原始DB和NPZ元数据只读，缺模态保持未知。
- 新增 `prepare_realsense_manual_review.py` 与4项CPU回归：事前固定分层随机10帧，RGB/左右/物体四列分离，原生实例PNG和正值并集PNG单独保存；缺文件为灰色未知，非负样本；源SHA前后不变。修正预览标题区高度，保证文字不进入原图/mask像素区域；预发布图保留在/tmp，不覆盖源数据。
- 用户对01–10整体反馈“很不错”，保存准确帧号与原话到 `docs/data-audits/realsense-20260910/manual-review/human-review.json`；仅记录整批目视认可，不扩成全量或MANO/深度验收。
- 新增 `docs/realsense-dataset-audit-2026-09-10.md` 和独立来源说明：26路mask与_only_label同SHA、9路进一步对应SAM3交互输出（6路有人工点），不一概称为全自动或独立GT；明确参考mask→bbox→WiLoR的geometry标签辅助风险、混合NPZ格式和无接触真值的边界。
- 新增 `audit_realsense_mano_numeric.py` 与6项CPU测试：18NPZ/11170按侧记录完整安全读取，有效9159条均数值有限，旋转与连续帧号检查通过，全部bbox与当前mask紧框逐帧一致；legacy无has_hand单独记录假设，不伪造字段。geometry仍待投影/标定，不启动optimizer。保留全量图层重叠统计，不强制互斥。
- 新增 `inspect_realsense_camera_metadata.py` 与5项CPU测试：只读10DB3的50条相机小消息，用已有Zstd及严格有界CDR字符串解析；提取原生Color640×480/Depth848×480内参与尺度，不装库、不碰ACL文件，不把内参可读等同深度已对齐。
- 新增 `compare_source_mask_versions.py` 与9项CPU测试：cup右手两版642帧同PTS全像素比较，465帧存在真实二值差异，非重编码/ID变化；保留双方SHA与前后stat，不自动选优或覆盖。
- 22:20完整CPU回归392项，391通过、1项OpenCV依赖跳过；此计数为当时已落盘脚本，不冒充后续新增测试已包含。
- 新增 `select_realsense_overlap_diagnostics.py` 与3项CPU测试、`render_realsense_diagnostics.py` 与4项CPU测试：固定D1–D4定向难例，显示当前左右/物体和另列历史版；不外推错误率、不自动修订。独立review修正未知重叠须为null、原生mask尺寸/位深/帧数检查、安全唯一输出ID；原尺寸PNG逐像素往返与前后源指纹通过，等待用户复核。
- 分类打包器增加 `data-audits/` 整体归入数据类、RealSense抽样/人工反馈小型凭据随包及显式白名单CHANGELOG（不递归收集仓库根其他文档）；新增2项回归（打包器共13项），不修改用户Windows已有文件。
- 收尾完整CPU回归410项：409通过、1项OpenCV依赖跳过；本次新增审计/提取/复核/版本比较均无GPU运行，编译与`git diff --check`通过。定向D1–D4仍待用户反馈，未把源图视觉认可等同训练效果。
- 随后收到D1–D4反馈：D1左层/D2右层混入对侧获用户确认；D3要求解释版本差异；D4保留不确定并要求前后帧。另存逐项human-review，不重绘、删除或覆盖源数据；继续定位D3与制作D4连续帧复核，不将随机十张认可外推全量正确。
- 新增 `render_realsense_temporal_review.py`，为D4提取basket570–645共76个连续源帧，五条10fps慢放含原图/右手/物体独立视频和局部放大；所有PTS与导出帧数检查通过，源/脚本SHA前后不变。保存原尺寸raw PNG，明确视频有显示重编码、缩放图不用于精确指标；未重新预测标签，D4仍待用户判断。
- 新增 `explain_cup_mask_versions.py` 与4项CPU测试：D3旧right的99.21%像素落当前left，连续171–186帧对应异常，171帧旧交互两个正点也落当前left。记录具体提示/身份线索但不把相关性冒充因果，当前版修订历史仍未知；不将历史/当前标注误称我们的训练前后效果。
- 用户看D4上下文后提出“编织缺口可能露出手”，仍不确定；另存后续反馈，保持未知而非坏标签，新增可见材质/填充轮廓孔洞约定的监督检查。分类包增加显式小型MP4与D4反馈，仍受单文件/总量上限，打包器新增视频预算回归（共14项）。
- RealSense问题登记区分D1/D2确认错误、D4时序复核后仍未知、D3历史版本关系；未确定替换像素或训练划分，不把465差异帧当作465当前错误。补充后完整CPU回归415项：414通过、1项OpenCV依赖跳过。

## 19. 旧训练完整验证恢复与输入残差真实GPU检查（2026-09-10夜间）

- 新增 `resume_nakehand_semantic_validation.py` 与6项CPU测试：207项旧来源/state/checkpoint指纹及两组实际2000样本契约核验通过；独立CPU进程复现直接文件入口的类身份错误，再验证一致包模式严格安装/恢复。保留旧源码/快照/checkpoint，只恢复评估，不弱化类型检查。
- 22:49在tmux `nakehand-val-recovery-20260910-2249` 实际启动GPU0完整三变体3449帧验证，supervisor3885281/evaluator3886948；第一组日志越过125/3449，确已越过原失败位置。25%allocator、5400秒验证上限，成功后自动报告，无重训练、无RealSense GPU推理。旧progress.json的初始3449/full_val_evaluated字段表示计划范围，不能当完成计数，以实际日志/最终records与summary为准。
- 新输入残差probe在GPU1实际执行：固定train0/61/66三图的全部logits/presence/boxes/200候选mask与原VE严格相等，非手cup/knife特征在手残差扰动后不变；反向阶段碰25%自设显存配额后OOM，保存失败摘要。没有optimizer step，未完成梯度验收，不能宣称可训练性或精度收益已通过。
- 新增受限 `--gpu-memory-fraction` probe参数及2项CPU回归（相关共13项）：重新确认GPU1余量后，在独立目录以35%配额、600秒上限执行一次真实probe，全部通过并退出。三图完整输出零差异、非手特征保持、两个词角色及左右输出特征梯度均非零；仅2048参数，0次optimizer更新，峰值allocated约5976MiB/reserved约6318MiB。保留25%失败及新35%条件的独立SHA记录，不把梯度路径通过当作精度改善。
- 新增 `docs/focal-gamma0-numerical-audit-2026-09-10.md`：只读核验真实presence配置确实走gamma=0的Triton分支，CPU极值对照确认本地反向算术有饱和非有限风险，而PyTorch fallback与alpha加权BCE一致。六源SHA前后不变，未运行GPU极值复现、未修改core/loss；不把潜在NaN风险归因为已知OOM，也不将正常样本probe通过当作所有数值角落验收。

## 20. DexYCB 转换保真复核与 SAM3 损失完整讲解（2026-09-10夜间）

- 新增 `docs/dexycb-conversion-trust-audit.md`：用官方论文/toolkit与当前本地代码追踪真实RGB、关键点/几何拟合渲染标签、整数实例映射、权威side修正、COCO RLE与训练预处理；区分标签来源可靠、转换忠实与边界适合任务，不将SAM3辅助本地参考或少量人审冒充全量独立GT。
- 新鲜有界抽查共46帧：原始NPZ→当前统一21帧（限已解压subject09）、原始→冻结COCO20帧、统一→COCO42帧，分别mismatch=0；4个小mask样本正确整帧过滤。保留第一份45帧JSON，另存补充h1样本及143个输入版本复核，明确首次哈希采集时点，不夸称所有首次解码前后校验。
- 当前源track分段与旧manifest/exporter单track假设不兼容，实际只读重建调用报错；源HEAD/文件SHA/时间已记录，冻结三split COCO SHA仍与既存说明相同。新增h1 frame51与旧COCO h0同为101像素且完全一致；仅报告兼容性问题，不覆盖或重新生成训练数据。
- 重新完整验证29,085张图的存在性/尺寸、25,489个RLE面积与bbox；完整RLE解码的校验器抽样仅9个，与原始像素审计范围分开。新增历史错侧frame52的原始RGB/官方手mask/统一手mask/COCO解码四列图及原生PNG，3394手像素完全一致，无模型预测、无彩色叠图。
- 新增10张固定面积分层正样本CPU预处理核验：实际API是uint8双线性缩放后取整至1008²，与显式参考10/10逐像素相同；原RLE、collator像素和源码/输入前后SHA通过。nearest/nearest-exact差异只记作算法差，不称标注误差；未改训练插值。
- 新增 `docs/sam3-loss-reference.md`：完整公式、目的、归一化、有效性门控、权重、semantic/association、论文tracker与注释实现边界、matcher/score/指标区别、当前六项及我们新增anchor/准备中的boundary。两轮独立公式review补清weak/padding优先级、IoU头监督范围与已实现但未训练的boundary状态；不把缺席侧instance分类写成负监督、不把box IoU分数当mask质量。
- 文档分类打包器增加loss会议类、转换/resize数据类、数值审计结果类及明确白名单小型Dex证据JSON；新增分类回归，打包相关15项通过。完整CPU套件424项：423通过、1项OpenCV依赖跳过；分类收尾改动另行复测，`git diff --check`通过。未修改core/loss/冻结训练评估依赖，Windows实际传输仍未确认。

## 21. 目标域语义结果复核与凌晨受控实验队列（2026-09-10/11）

- 旧 nakehand 三变体完整验证在 23:39:52 结束，23:40 生成报告；独立逐条复核 3×6,898 查询及来源指纹。原 VE / 无约束语义增量 / anchor 的漏检计零 Dice 约为 0.9443 / 0.9775 / 0.9772，新增缺席侧输出分别 0 / 30 / 36（分母496）；保留正例收益与误报代价，不将辅助参考的单录像验证等同独立泛化。
- 新增 `run_nakehand_boundary_ablation.py` 和 7 项 CPU 测试：两卡 weight0/4 同初值、同前2,000图的单因素对照，20步检查后恢复，每100步checkpoint，成功后旧/新两组完整空间验证；包模式执行、来源快照/指纹、真实记录完成检查、有限等待/时限、不无限重试、不操作他人进程。
- 队列7项、边界训练11项、空间评估14项定向CPU测试通过；此时只是运行前验收，不把脚本准备或测试通过当作GPU实验已经完成。
- 新增 `docs/overnight-experiment-plan-2026-09-11.md`：明确边界、输入VE共享零残差、缺席误报三条工作线，冻结范围与证据标准；没有新增 geometry/memory/RealSense 训练或覆盖封存资料包。

- 00:01:27 两组边界实验实际在 tmux `nakehand-boundary-20260911-0003`、GPU0/2启动；均通过20步GPU检查并自动恢复至2000预算，每100步checkpoint。此刻完整训练/评估尚未完成，不称已有边界收益。
- 新增 input-VE 独立训练/空间评估入口与21项CPU测试，另原型9项通过：2048共享输入词角色残差、零初始化、原六loss、完整冻结、相同2000前缀、20步恢复与独立格式。尚未启动该方案GPU训练；同LR不等于相同函数步幅，位置与跨侧共享结构是架构差异。
- 新增 nakehand 只读阈值诊断及18项CPU测试：完整3×6898查询复核，0/1/5%参考缺席输出预算、tie整组处理、双空图和单侧缺席分开。仅同val开发诊断，不改推理阈值；新增输出集中在0–19等少量片段。
- 新增完整 nakehand frame0 时序复核脚本及4项CPU测试：六个完整/慢放视频、13组源RGB/左右raw PNG、逐帧元数据、SHA与约17.75MiB ZIP；完整视频3449帧、慢放300帧逐一验收，13帧source与派生像素一致。首次RGB显示编码触180秒上限，失败目录保留；新目录veryfast编码和600秒上限完成，源数据无修改。
- 用户明确确认frame0/image4713是俯视的右手；助手此前按画面位置猜左手的说法撤回，原话另存人审JSON。该帧两参考空，但训练模型left_hand检出实际右手，确认参考漏标与模型错侧同时存在；修正全量/阈值报告解读，不覆盖旧COCO、checkpoint、原始统计或封存ZIP，不外推邻帧。
- 两组边界训练随后均实际完成2000步，checkpoint与严格配置/相同图序验收通过；旧/新两组完整三变体空间验证已启动。单独记录真实梯度量级：weight4的额外边界/任务梯度范数中位数约2.03%（均非零行），不是边界精度收益。
- 统一目标文档追加标签完整性合同：mask质量、解剖侧别、存在性完备性分别验收；明确当前空参考会产生presence负监督，单设is_exhaustive=False不能屏蔽该loss，三态监督尚未实现/启用。冻结训练数据不改，未来需单独版本与同预算对照。
- 新增train空参考审计和12项CPU测试：全train344双空/75单参考，旧两组实际2000图包含76双空/14单参考，166/4000参考缺席查询；24图固定首中末初检、120PNG发布SHA、48侧raw/binary/RLE一致。9双空抽样未见明显漏手，边缘腕臂样本待用户人审，不外推cleanGT或覆盖旧标签。
- 新增 `run_nakehand_input_ablation.py` 与16项CPU测试：前置队列终态和活跃子进程双重门控、物理0/2两学习率、20→4图执行检查→总2000→三模型全val、严格恢复与来源契约、自己的新进程组超时清理、无重试；小样本probe独立字段，不冒充完整报告。00:37:08已实际启动tmux等待，尚未进行input optimizer训练。
- 新增 `finish_overnight_review.py` 与16项CPU测试：真实完整记录/身份/SHA、boundary和input各必须两份报告、未知合同不整体认证；首中末最多3组分离图、有界等待及新目录分类打包，00:37:08已CPU-only持久启动，当前watching，尚未发布ZIP或传Windows。
- 分类打包器新增每份MD一行原文主题及索引回归（共17项）；夜间发布最多256资产、总100MiB、单8MiB，预算省略显式记录。所有旧封存ZIP保持不变。父supervisor禁用CUDA，实际验证未初始化CUDA，GPU子进程独立绑定授权卡。
- 收尾完整CPU回归520项：519通过、1项既有OpenCV依赖跳过；`git diff --check`通过。两份持续评估仍在GPU0/2，input等待和晨间发布为CPU任务；未做git commit/push或Windows实际写入。

## 22. 夜间结果验收、争议帧统一排除与实验报告（2026-09-11 上午）

- 新增 `docs/overnight-experiment-results-2026-09-11.md`：明确四组新 2,000 步训练、两组旧 checkpoint 的空间复评、四组全 val 验证，区分 7 个方案、数据划分、损失／冻结／恢复配置与实际完成量；不是完整 epoch，也未训练 MANO、memory 或 RealSense。
- 独立核对四份评估 82,776 条原始查询记录的身份、SHA 和指标重聚合；四组新最终 checkpoint 的真实训练图序、初值、有限性与冻结守卫通过。对已保存 40 组图、480 张预测 PNG 重新核算像素统计，并用独立重复腐蚀核对三个宽度的边界交并；不冒充全量人工验收。
- 当前目标域上，语义输出残差的候选／漏检计零 Dice 为 0.978554／0.977510，原 VE 为 0.973987／0.944291；参考正查询漏检 194→7。边界 weight4 不优于严格配对 weight0 的候选 Dice 与边界均值，anchor1 未增加分割收益。输入 LR=.0003 的正例指标接近输出残差，但单侧参考缺失时输出达 81/152（输出残差5/152），不按总体 Dice 宣称改进已通过。
- 按用户最新要求，仅将 val image4713 / source frame0 从主开发比较排除，保留原始数据、人审和历史报告；该帧不在 train/development_holdout，不重训、不扩大邻帧、不撤销历史证据。新增排除 manifest、过滤器及6项通过测试，严格源 SHA／图像身份／左右配对、拒绝覆盖、保留ID，生成3,448图筛选标注与12组各6,896条记录。
- 筛选产物位于 `runs/nakehand-review-exclusion-20260911/`，仅标注和记录副本，不是可冒充旧 READY 契约的独立数据根。另存精确前后复算 JSON：正例6,402及其Dice/BIoU不变；参考空查询496→494，E1输出30→29、输入.001为41→40、输入.0003为99→98。所有模型采用同一事后人工排除，明确这不是新的独立测试。
- 原始四份完整报告于凌晨01:27／01:32／02:59／03:04完成，03:04:58已发布历史晨间包；ZIP CRC、成员SHA和源核验通过。该旧包不包含上午新报告，15处源码链接离线不可用已记录；未声称已同步Windows，未做git commit/push。旧计划增加最终报告入口，保留历史进度正文。

## 23. 输出残差讲解与汇报示意图（2026-09-11）

- 新增 `docs/sam3-output-residual-explained.md` 与 `docs/figures/sam3-output-residual.{dot,png,svg}`，沿用静态科研图交付：原始文本路径、一次性缓存、有效位置加Δ、RGB独立视觉路径、冻结网络反传、仅更新2,048参数；附导师汇报讲稿与源码入口。
- 核对实际接口：残差在 resizer 后256维输出，完整长度仍32，只改每侧4个有效位置，padding及原始1024维查表embedding不变；不把raw_cache误称为Transformer上下文输出，不把缓存画成每帧重跑VE。
- 明确无anchor时是从原特征初始化的重参数化、并非语义硬约束或LoRA；每侧增量跨图片共享，当前固定手类缓存不支持任意物体prompt，后续需显式保留／路由原VE。说明阶段性收益与限制，不修改训练代码、数据、权重或启动新实验；未实际同步Windows。

## 24. 文档移出仓库与服务器 Git 交付流程（2026-09-11）

- 按用户要求将全部docs实际移到同级 `/home/zhengyuxi/projects/docs`；迁移前备份原目录、暂存区binary patch与index，并用tar compare逐项核对原目录。仓库内只留被忽略的本地兼容软链；仅撤销docs的暂存，不删除数据或取消其他代码暂存。
- 新增 `DOCS_LOCATION.md`：记录Linux/Windows同级布局和历史相对源码链接限制。交付遵循验证→更新CHANGELOG→commit/push；不添加Co-authored-by、不强推、不自动合并主分支、不上传数据/权重/凭据。个人账号运维说明随后移出仓库。
- 打包器支持显式 `--docs-root` 与同级目录默认发现，只允许既定旧docs直达新目录的兼容别名，不放开一般软链；仅在新阅读副本中映射旧链接，保持历史MD/JSON/哈希不变。晨间发布器透传并检查外置docs与输出不嵌套；Windows同步默认目的地改为paper_review下同级docs，仍以新版本目录交付。
- 新增7项迁移/路径安全测试，打包与发布定向40项通过。Git作者身份仅作仓库局部配置；账号认证和远端写权限另行验证，不能把作者配置完成等同已登录或已push。
- 迁移后的完整CPU回归533项：532通过、1项既有依赖跳过；所有训练数据、权重与参考源未修改。新增同级docs阅读包实际构建验证，包含上午输出残差说明和图，旧封存包不覆盖。GitHub登录尚待用户浏览器首次授权，提交与推送状态分别报告。

## 25. GitHub 配置限定在本用户目录（2026-09-11）

- Git作者与认证helper限定为仓库局部配置，工具及凭据限定为用户目录，不写system/global Git配置或shell启动文件，不影响其他用户。
- 认证、远端写权限与实际推送分别核验；个人账号、凭据位置及操作说明在仓库外单独管理，不随项目发布。
- 远端 `31cdd00` 与本地已有祖先 `f8260eb` 的完整 tree 均为 `f7010834d80653781f97a99e0134dbc33383ee87`，不存在额外远端代码差异。保留两条历史进行非强推合并，不回退服务器后续改动；合并仅补充本日志，代码与已完成533项CPU回归（532通过、1跳过）的本地版本相同，推送后再核对远端SHA。

## 26. 大 batch 输出残差 DDP、TensorBoard 与混合数据准备（2026-09-11）

- 用户选择 DexYCB 一半＋nakehand 全部训练、RealSense 测试：具体按原 Dex train 固定分层取半，保留原 Dex val；合并 nakehand 六录像并沿用争议 frame0 排除。新派生版本不覆盖原 split，nakehand 旧 val/holdout 此后不再作为独立验证。RealSense 不参与训练、学习率或阈值选择。
- 新增独立 `train_residual_ddp.py` 及 runtime/data/objective/checkpoint/validation 组件：torchrun 单机三/四卡、真正多图 batch、spawn workers、递归 pinned memory、严格图像/侧别身份校验，原 SAM3 与旧实验入口保持不变。
- DDP 使用标准 backward，仅更新 FP32 `delta[2,4,256]`，原主干仍 eval/frozen；六项原任务 loss 保持权重1。明确大 batch mask/box 按全局目标数归一化，先 clamp 总数再除卡数；CE/presence 按等量查询平均，不冒充旧逐图更新预算。
- 新版恢复点区分 optimizer steps、samples_seen、epoch 与 world size，保存各 rank RNG/实际图序、optimizer与冻结缓存，拒绝数据/代码/卡数/batch等配置悄悄改变；仅 rank0 原子写恢复点，失败不保存可能半更新的参数。
- 新增 rank0 TensorBoard＋原始JSONL，记录分项loss、学习率、左右梯度、数据等待/H2D/计算耗时、吞吐和峰值显存；验证记录loss、左右Dice/漏检/误报及分离RGB/GT/预测。训练loss下降不能代替验证效果。
- `prepare_residual_mixed_training.py` 提供新ID、保留源dataset/split/image/frame、批准源RGB软链及逐文件SHA、源READY/annotations绑定、末尾READY发布；不重画参考mask。派生版本已发布：Dex train 分层取11,632张＋nakehand18,497张＝train30,129张（45,299标注），val2,909张（2,876标注）。保留争议nakehand源frame0排除；train/val标注SHA分别为`6ed07c5010fe04b52c6d35f2c28288afd638b783d813112f6e9859199e8da866`、`dd6ed2af7448dc4541ea1901446fc0d6dd84a4828fa4b4eb41c75b7fa56c5ddd`。
- 完整CPU回归606项：605通过、1项因环境缺OpenCV跳过，无失败。覆盖三/四rank Gloo等价性、真实多图collator/原六项loss、跨rank缓存SHA审计、断点恢复、rank0 TensorBoard事件、仓库外launcher模块导入与数据准备契约。
- 101真实三卡NCCL默认初始化出现SIGSEGV；极小标量对照复现于NET/IB路径，仅在本进程设置`NCCL_IB_DISABLE=1`后通信通过。未改系统或为其他机器默认关闭IB。随后真实SAM3三卡（每卡batch1、BF16、lr0.001）两步短训练完成：loss有限，左右梯度非零，仅2048个残差参数可训练，冻结参数版本不变、三个rank缓存SHA一致，恢复点与TensorBoard事件均落盘。此为工程冒烟，不是两轮训练或分割精度提升证据；四卡GPU、大batch性能、正式训练及RealSense测试仍待验收。
- 真实三卡恢复进一步通过：保持原epochs2/每卡batch1/lr0.001配置，从step2完整恢复到step4（累计12张图）；三个rank完整缓存SHA均为`d0e5097ff312b142c93a4adc235b585418476e1c24dca52578976344890e2f3b`，左右delta L2分别约0.11198/0.08653。该项证明实际恢复与同步链路可执行；尚未声称GPU逐位等同不间断训练轨迹，也未运行正式精度验收。
- 三卡每卡batch2的受限显存短测在首步前触发allocator上限（25%），未执行优化更新。保留失败记录；不提高共享卡上限挤占其他进程，不把未通过的大batch配置作为正式训练配置。
- 任务指导清单和个人服务器/SSH/账号操作文档在仓库外独立维护，项目不保留副本；日志移除本轮个人认证信息。训练说明与指标阅读文档仍位于项目同级外置docs，不重写已有Git历史。

## 27. 残差验证选模、实验归档与正式训练准备（2026-09-11 下午）

- 新增纯状态机 `residual_selection.py`，基于固定Dex验证集的左右等权漏检计零Dice选择实际最高分best；patience/min_delta只控制显著改善与早停，不漏存小幅最高分。初始原VE基线也可成为best，不强行把训练过的checkpoint称为更好。
- DDP训练保存验证状态、不可变step-best与best入口；轮末保存后验证前发生中断，恢复时先补验证。各rank校验完整selection一致再保存/停止，partial smoke不消耗epoch patience。恢复绑定验证SHA/策略及历史，不冒充兼容旧实现哈希；迁移恢复需带历史best文件。
- 65项选模/训练器/恢复CPU测试和5项launcher测试通过，含真实TensorBoard选模事件、轮末故障恢复与Adam/delta一致、best/latest区别与终态恢复；修复符号链接启动的`__file__`指纹路径一致性。
- 足量显存单卡B2六步成功后，实际三卡B2/global6、workers4、28%allocator上限短测完成30步/180图，无OOM；仅2,048参数更新，冻结与rank一致性通过。后期短窗口约8–9.7图/秒，不将工程计划epochs3写成已训练三轮，也不据此声称精度改善。
- 实验总表与独立工程/失败报告已在仓库外建立，正式混合训练报告保持“准备/进行中”直到验证、best、真实曲线/分割图和结论完成。旧实验尚未全部迁入新归档；代码、私人运维信息与实验资料继续分离。
- 外置MANO更正说明明确：现有原型已接Detector而非Tracker，但只是原geometry输出后追加五组参数tokens；空间投影、多手输入和geometry内部融合尚未实现或训练。后续建议先核验MANO投影与原geometry点框基线，再训练空间适配器，不抢占残差主实验资源。
- 完整GPU验证闭环通过：三rank各自非重复分片，总2,909图/5,818侧查询，初始原VE及一步更新后均全量完成。初始候选Dice左0.81315/右0.79981、漏检计零Dice左0.42356/右0.59132；固定0.5阈值，结果不是正式残差增益。88项residual相关CPU回归与16项trainer复测通过。
- 新增独立 `prepare_realsense_test.py` / `evaluate_residual_test.py`：先冻结8段完整双侧流、每段16帧的seed抽样及已知争议排除；发布128RGB/211侧参考、逐文件SHA/RLE核验，禁止把缺文件当负例，不进入训练/选模验证入口。保留21帧参考重叠的限制，不声称全部经过独立人工精标。
- RealSense test使用真实原始VE自然提示与预先选定的DDP增量，同0.5置信度/像素阈值及4原图像素边界，输出候选/漏检计零Dice、FP/错侧代理、按录像指标与独立原图/参考/预测PNG，不以参考重叠挑decoder；12项新增CPU测试及完整128图/256query真实collator身份检查通过，GPU效果待推理。

## 28. 严格残差对比、真实曲线与版本化文档交付（2026-09-11 下午）

- 新增 `compare_residual_test_results.py`：固定128图/256侧查询、同源身份/阈值/代码指纹核对，从保存RLE重算区域、Boundary IoU、漏检/空侧误报和按录像指标，报告实际checkpoint进度与预先选点依据；15项CPU测试通过。工具完成不等于残差对比结果已产生。
- 新增 `plot_residual_training.py`：从实际JSONL或TensorBoard记录生成loss/验证/吞吐图与来源SHA；原始采样batch及20条滚动均值明确分开，验证只画实测点，不插值或虚构epoch均值。14项CPU测试通过，正式run与工程run真实曲线已保存仓库外。
- 原始VE固定RealSense128推理完成：候选Dice0.93236、漏检计零Dice0.89501、FN13/211、空侧FP2/45；保留SAM3辅助参考及21图左右参考重叠限制。原图/左右参考/预测独立PNG及16组固定五列总览随外部报告保存，不据该test调训练。
- 文档包增加“本次先读”及训练/TensorBoard分类，优先收录本次真实图像，保留历史文档时间与未打包路径提示。新增严格ZIP/清单复核后的原子`latest.json`发布器；旧版资料与原数据不覆盖。
- Windows单包接收器去除个人路径/服务器默认，新增显式RemoteRoot、严格主机校验、非交互连接及ZIP/manifest预期哈希校验；新增只读feed拉取器，已有版本须回执、ZIP及全部文件核验后才跳过。不安装计划任务、改执行策略、获取凭据或开放Windows入站端口。Windows实机接收仍待用户本地验证。
- 外部实验总表、独立工程报告、MANO初始geometry更正/CPU坐标契约预检与Windows长期同步说明已整理；未启动geometry训练，未把不可读MANO资产、canonical左镜像或虚拟相机约定写成已通过投影。私人运维/TODO和训练资料继续不进Git。

## 29. nakehand-only 数据对照与分离参考审查（2026-09-11 傍晚）

- 新增 `prepare_residual_nakehand_only.py`：从已批准的混合版本精确提取18,497张nakehand，保留原图、类别、mask、bbox、ID与溯源，沿用争议frame0排除；35,195标注逐行等价，全部RGB SHA及批准软链范围核验，Dex验证2,909图不变。源路径须显式传入，不含个人目录默认值；13项CPU测试及真实训练器preflight通过，不修改共享原始数据。
- 两轮数据对照保持原VE输出零增量初始化、lr0.001、anchor0、三卡每卡B2/global6，不从混合训练权重继续。预先指定实际第一/二轮完成点分别测试同一RealSense128；复用原训练器的stop/resume，在两轮计划不变的前提下顺序训练→测试→恢复，无新增后台巡检或训练核心改动。每轮3,082更新/18,492曝光，两轮不等同混合实验的更新预算；测试不用于改变后续训练参数。
- 新增 `visualize_residual_validation.py`：仅用已有验证RLE生成固定seed、预选12图的独立RGB/参考/原VE与残差候选及检出mask，不重新推理、不按收益挑图；9项CPU测试通过。混合第一轮已实际完成，Dex左右等权漏检计零Dice由0.50744至0.61974，但左手候选Dice下降，不能把检出改善当作边界整体改善。
- 新增 `audit_egohos_sample.py`：在获准小样本预算内对官方ZIP进行有限Range读取、明确下载计量、源版本绑定、配对及ZIP/CRC/SHA核验；16项CPU测试通过。已取得20对原RGB/label，原标签含前臂/衣袖，外层未提供contact，不据此训练或伪造手腕截断/接触真值；数据、预览与调研报告留在仓库外。
- 文档打包器将sample-review资料归入数据审查类，明确资产数量上限512，字节与安全边界不变；对应25项CPU测试通过。固定RealSense16图的人审材料只含RGB与左右辅助参考，保留全黑和重叠，不含预测，不修改测试集。
- CPU实测确认当前缓存使用自然语言空格提示，各侧4个有效位置；直接编码下划线提示是5个，但内部类别键不会重新分词。实际anchor权重0的梯度全零，2,048个增量元素均有更新；后续特殊token位置约束需独立对照，不在本次数据实验中混改。

## 30. RealSense全量四组评估与缺失参考处理（2026-09-11 晚）

- 新增 `prepare_realsense_full_test.py`，逐帧流式导出10录像全部6,204帧，不覆盖旧128测试。17,374个PNG、9,159个正参考RLE/面积/bbox、源视频与旧128像素对应经CPU核验；缺失侧不生成假空mask。13项准备器CPU测试通过。
- 固定保留两侧参考可用性与侧级质量标记：提供参考11,170查询，其中无既知问题侧11,076；已确认错误、不确定窗口、历史版本问题仍保留原像素并分别披露，普通已展示样本不视为质量错误。参考未知的1,238侧查询仍预测但不计真值指标；这是辅助参考开发对照，不宣称独立人工精标或盲测。
- 新增 `evaluate_realsense_full.py`：`ve-both/ve-left/ve-right/residual`、确定性整图分片、逐batch保存JSONL/RLE、固定阈值及非GT候选选择、只按批加载参考、缺侧指标null、分离RGB/参考/候选/检出图。原VE双侧共享一次前向后在报告中分成两组，另外评估两个用户指定的实际第一轮残差；15项CPU测试含真实loader/collator与缺侧语义通过。
- 新增 `compare_realsense_full.py`：32项CPU测试通过，严格检查四组所有分片身份、覆盖/重复、数据/推理源码/阈值/权重指纹，独立从RLE复算区域、4像素边界及漏检/误报；全部提供参考与侧级敏感性口径、旧128、部分参考和未知查询分组。真实全量loader核验峰值约139MiB，不把数GB解码mask常驻内存；缺组默认拒绝完整比较。
- 训练仍沿用冻结实现，两轮计划、优化器和恢复配置不变；全量评估另用冻结代码部署与独立显存受限卡，不修改原数据、旧结果或其他人的GPU进程。mixed/nake第一轮更新数及曝光量不同，不能声称等规模消融。数据、报告、权重和个人运维信息不进入Git；程序完成不等于四组全量推理已完成。

## 31. 显式第二轮全量评估完成点（2026-09-11 晚）

- `compare_realsense_full.py`新增`--nake-expected-epoch {1,2}`，默认仍为第一轮；第二轮必须显式指定，并核对实际step6164、完整epoch2、36,984次曝光及原每轮3,082步，不从best或测试得分自动选点。
- 保留第一轮默认拒绝第二轮权重的行为，拒绝部分轮次、错误曝光、无效epoch类型或超出原训练计划的完成点。报告轮次/曝光说明根据已核验元数据生成，不将第二轮误写成epoch1。
- 比较器35项CPU测试通过，包含第二轮完整路径、默认选点拒绝、部分轮次拒绝和训练计划限制；推理源码、阈值、精度及数据定义不变。新增测试通过`unittest discover`运行，避免同名标准库test包影响模块导入。
- 本次新增评估使用独立输出目录，复用冻结VE和mixed第一轮作为对照，不重训基线、不覆盖既有第一轮报告；数据、checkpoint、实验报告和运行脚本仍在Git之外。

## 32. 独立空间 mask 适配器工程试验

- 新增 opt-in `spatial_mask_adapter.py`：只在原实例分割投影前增加256→32→256的局部空间残差（16,992参数），末层零初始化；默认builder、原文本残差训练、geometry与tracker均不改。
- 新增有界单GPU `pilot_spatial_mask_adapter.py`，最多100步，读取已有哈希绑定训练选择，固定原自然VE，冻结全部基础参数，仅优化原mask focal和Dice；其他四项loss只记录。新输出目录、每10步独立adapter checkpoint、TensorBoard、真实SAM3零初始化与检测输出不变检查。显式拒绝DDP调用及恢复，不冒充正式多卡训练入口。
- 4项CPU测试通过。真实GPU B2/30步通过：零初始化mask/box/class/presence逐元素一致，冻结参数未更新，梯度有限；训练后mask变化，box/class/presence保持一致，峰值分配显存约3.66GiB。该结果仅证明工程可训练，不证明边界或泛化提升。
- 旧实验及代码已在项目外独立封存，新实验不覆盖历史。下一阶段需独立DDP恢复/验证契约和边界对照；不把空间权重当成旧token checkpoint读取。

## 33. 空间适配器独立DDP、恢复与固定评估

- 新增 `train_spatial_ddp.py` 和 `spatial_training_state.py`，与旧文本残差入口/权重格式分离。原VE与SAM3冻结，只优化空间mask focal＋Dice；global目标归一化、全rank有限性检查、每rank参数一致性、配置/数据/代码绑定及RNG/AdamW恢复。
- DDP初始化会同步冻结参数并改变版本计数，冻结审计基准放在同步之后；保留版本/梯度/可训练标志检查，不以关闭审核处理误报。
- 默认两轮及逐轮Dex验证；按左右候选Boundary4px宏平均保存best，不从RealSense选点；轮末待验证checkpoint恢复时补验证。原评分和未知参考规则不改。
- 新增 `evaluate_spatial_realsense.py`，显式核验完整且已验证的epoch，默认固定128、可显式全量6204；使用实际原VE，自然左右提示，空间权重独立加载。旧全量评估共享函数仅增加spatial报告标识，旧CLI行为不变。
- 33项相关CPU测试通过（空间模块/状态9、空间选点3、全量评估16、验证5）；Gloo通信测试需在允许本机通信环境执行。
- 实际双卡B3/global6两步保存、2→4步恢复和两次独立连续4步均完成。配置/步数/RNG精确一致；恢复参数最大差2.70e-7，连续重复最大差8.56e-8，优化器moment最大差2.68e-9，明确为工程数值容差通过，不声称逐位可复现。
- 正式新实验从零初始化重新开始，不继承30步pilot；TensorBoard/每250步恢复/两轮固定评估采用独立实验目录和有限工作流。尚无正式轮次效果，不把工程验证写成分割提升。

## 下一步

1. 原VE、完整缓存、零增量等价性及本轮完整val对照已完成；保留原VE和语义输出残差作为研究对照，优先检查单侧场景及左右条件区分，不把移除文字计算等同精度提高。
2. 保留现有原域和外部结果；用val选择模型/阈值，不将已看过的外部600或训练录像包装成未见最终test。
3. geometry固定原VE，先完成noMANO接入检查及可读MANO模型/投影/单位验收，再进行受控参数更新。
4. nakehand本轮四组新增2000步训练及四组全量验证已经完成，结果见第22节；RealSense只读审计及定向人工复核已记录，保留确认错误和不确定标注，未启动该数据集训练。人物/同步信息仍未知，明确开发划分限制。
5. 固定可用单帧/视频基线，之后再比较正向、反向、结果融合和真正双向memory；不把接口实现写成效果收益。

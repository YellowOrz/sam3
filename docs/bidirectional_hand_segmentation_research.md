# SAM 3 手部分割的双向推理与融合方案调研

> 调研日期：2026-09-04  
> 目标：提升 `left hand` / `right hand` 在手物交互视频中的身份正确性、
> 时序稳定性和掩码质量；优先零训练、离线处理；WiLoR 只作为可能出错的
> 弱证据。未来“被手接触物体”的分割只列为 TODO。

## 1. 结论先行

最值得先实现的不是修改 SAM 3 网络，而是一个**完全独立双向推理 +
序列级候选选择 + 保守区间重传播**的外部系统：

1. 将原视频按正常帧序运行一次 SAM 3，得到正向候选 `F`。
2. 将帧序真正反转，作为新视频、新 session 从头独立运行同一提示词，得到
   反向候选；再将结果索引翻回原时间轴，记为 `B`。
3. 先做实例/身份对齐，不能直接把两个结果视频中的 label ID 对齐。
4. 用动态规划（Viterbi）在每一帧的 `F`、`B`、必要时的“目标不可见”三种
   状态间选择一条全视频最优路径。评分同时考虑模型内部质量、运动补偿后的
   时序一致性、身份一致性和形状突变；**默认不做逐像素并集**。
5. 仅当正反向存在持续且严重的冲突，并且至少有另一类独立失败证据时，才把
   该段判为高不确定区间。在区间两侧寻找可信锚点，用新的 session 从两侧向
   区间内部重传播，再重新做序列选择。
6. WiLoR 通过可靠性门控后只能参与平局裁决和身份校验，不能单独触发修正，
   也不能覆盖强烈的 SAM/时序证据。

推荐优先级为：

| 优先级 | 方法 | 训练 | 预期收益 | 主要风险 |
|---|---|---:|---|---|
| P0 | 独立双向 + Viterbi 硬选择 | 无 | 高 | 质量分数需正确暴露与校准 |
| P0 | 保守区间检测 + 双锚点重传播 | 无 | 高 | 锚点若错会制造新漂移 |
| P1 | 在分歧边界带内融合 logits | 无 | 中 | logits 未校准、光流在遮挡处会错 |
| P1 | 多锚点/多假设束搜索 | 无 | 中到高 | 显存、工程复杂度和候选爆炸 |
| P1 | SAM2Long/SAMURAI 思路适配到 SAM 3 | 无训练但改内部 | 中到高 | SAM 3 已有部分记忆筛选，收益需实测 |
| P2 | 训练轻量候选质量预测器/时序 refiner | 少量训练 | 中到高 | 需要可靠标注和跨视频验证 |
| P3 | 把 SAM 3 主干改成原生双向网络 | 大量训练 | 不确定 | 分布偏移、训练和回归成本最高 |

在做完带标注的消融实验前，不能宣称融合一定优于两个方向。首先应测量
`oracle(F,B)` 上限：如果逐帧从 `F/B` 中挑真值更好者都没有明显提升，说明
仅靠候选选择不够，重点应转向多锚点重传播或产生新候选。

## 2. 任务定义与方向含义

本文的“反向”严格采用下面的定义：

```text
原视频： I0, I1, ..., I(T-1)  --全新 session--> F0, F1, ..., F(T-1)
反转后： I(T-1), ..., I1, I0  --全新 session--> B'(0), ..., B'(T-1)
索引复原：Bt = B'(T-1-t)
```

两次运行必须状态隔离。仓库接口的
`propagation_direction="backward"` 会从原视频末帧向前处理，理论上可作为避免
物理编码反转视频的快捷实现；但在采用前应做逐帧等价性测试。它不能和
“同一个中间锚点、同一 session 向前后两侧传播”混称为两次独立双向推理。

目标是指定左手或右手的单目标轨迹。若文本检测得到多个实例，需要先依据
目标手身份选择实例，再进行双向融合；不能把所有实例的前景并集当作目标手。

## 3. 为什么正序和倒序会不同

这不是随机噪声的简单表现，而是记忆式视频分割的自然结果。SAM 3 由图像级
检测器和基于 SAM 2 的记忆式 tracker 组成；官方说明其 detector 与 tracker
解耦并共享视觉 backbone。tracker 按处理顺序将历史预测编码为记忆，因此：

- 两个方向的初始化帧不同，文本检测质量和初始实例身份可能不同；
- 遮挡、运动模糊、手物粘连造成的一次错误会进入后续记忆并累积；
- 从一个方向看是“遮挡后恢复”，从另一个方向看可能是“清晰外观向遮挡传播”；
- 相似的另一只手、前臂或接触物会形成 distractor，方向不同会改变首次串轨时刻；
- 目标进入/离开画面时，presence、keep-alive 和重新检测启发式具有方向性。

这与 [SAM 3 的 detector–tracker 设计](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
和 [SAM 2 的 streaming memory 设计](https://ai.meta.com/research/sam2/)一致。
[SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)
也将错误记忆导致的级联漂移作为主要问题，并通过无需训练的多路径记忆树改善它。

因此，正反向差异本身是有用的诊断信号，但不是正确性的充分条件：两边可能
一致地漏掉手、选错同一实例，或一起把手中的物体吞进掩码。

## 4. 当前仓库能直接利用什么

### 4.1 已有能力

- [`process_dataset_videos.py`](../scripts/process_dataset_videos.py) 已支持从首帧
  forward、从末帧 backward，并能保存无损 `masks.mkv`。
- [`process_hand_video_with_wilor_prompts.py`](../scripts/process_hand_video_with_wilor_prompts.py)
  会选择一个可靠初始化帧，在同一 session 中从该帧向两侧传播，并用 WiLoR
  关节生成后续点提示。这适合“锚点向两侧传播”，但不是本文定义的两次独立
  双向候选。
- SAM 3 tracker 内部已有多 mask、预测 IoU、object presence logit 和 mask
  stability 机制；基础 SAM 3 默认还启用了基于 `presence × predicted IoU` 的
  记忆帧筛选，见 [`sam3_tracker_base.py`](../sam3/model/sam3_tracker_base.py) 和
  [`model_builder.py`](../sam3/model_builder.py)。这使 SAM2Long 思路具有适配基础。

### 4.2 一个必须先修正的分数误区

当前流式公开输出的 `out_probs` 来自 `obj_id_to_score`，它是实例首次检测时的
分数，并非逐帧掩码质量。内部的逐帧 `obj_id_to_tracker_score` 虽用于若干逻辑，
却没有出现在 `_postprocess_output()` 的最终返回值中；内部 tracker 的
`iou_score`、`object_score_logits`、`eff_iou_score` 也没有完整暴露。

因此不能直接写成“每帧选择 `out_probs` 较高的方向”。P0 实现应最小化修改
接口，额外输出下列诊断量，同时保持现有字段兼容：

- `out_tracker_probs`；
- predicted IoU / mask quality；
- object presence logit；
- mask stability；
- 最好保留阈值化前的低分辨率 mask logits。

SAM 3 和 SAM 3.1 的内部路径不完全相同，应分别验证这些量的语义与标定；不要
假设同一阈值可跨版本使用。

### 4.3 现有双向结果的无标注审计

对 `outputs/test_sam3/{left,right}_hand_{forward,backward}` 中可配对的 14 组结果
做前景 IoU 审计，共 8,378 帧：

- 每个序列平均正反向 IoU 的均值为 0.761，中位数为 0.903；
- 6/14 个序列的平均 IoU 低于 0.7；
- 多个冲突是连续区间而非孤立闪烁，`IoU < 0.5` 的最长连续区间为 447 帧；
- 有一组左右两边全程都为空，另有多组大量帧同时为空；这些帧会得到 IoU=1，
  再次说明“高一致性”不等于“正确”。

这个统计只是差异审计，不是精度评估。它还把同一帧的所有非零 label 合并成
了前景，而若干结果实际包含多个 object ID；正式实验必须先做实例匹配。现有
结果使用仓库 backward API，而非物理反转视频，后续还需验证两者是否等价。

## 5. 推荐方案 A：双向候选的序列级硬选择

### 5.1 为什么不用简单并集、交集或平均

- `F ∪ B` 提高 recall，却最容易把另一只手、前臂或接触物带入结果；这与既定
  优先级“避免串掩码 > 时序稳定 > 边界质量”冲突。
- `F ∩ B` 较保守，但在单侧漏分割时会切掉手指，严重时变空。
- 每帧独立选择高分方向会在 `F/B` 间快速跳变，形成新的 flicker。
- logits 平均只有在两方向分数可比且校准良好时才合理；二值 mask 无法恢复
  这种信息。

因此主结果先从完整候选 mask 中选择；只有候选身份已经稳定后，才在边界分歧
带内进行软融合。

### 5.2 候选状态与动态规划

每帧状态集合为 `S_t = {F, B, Ø}`。`Ø` 只在目标确实可能完全遮挡或离开画面时
允许，不能把“低分”自动解释为“不存在”。对方向 `d` 的帧级可靠度可写为：

```text
R_t(d) = wq * Qmodel
       + wt * IoU(M_t(d), Warp(M_(t-1)(chosen)))
       + wi * Qidentity
       + ws * Qshape
       + wd * Qdetector
```

其中：

- `Qmodel`：presence、predicted IoU、stability 和 tracker score 的校准组合；
- `Warp`：用光流把上一帧已选 mask 对齐到当前帧，只在前后向光流一致的像素
  上计分；
- `Qidentity`：目标手身份证据，WiLoR 可靠时才参与；
- `Qshape`：面积、质心速度、连通域数相对该轨迹近期稳健统计的异常程度；
- `Qdetector`：在稀疏关键帧上重新运行图像级概念检测得到的支持度。

转移代价同时惩罚运动补偿后不连续和无理由的方向切换。用 Viterbi 求全局代价
最低路径，而不是逐帧贪心。切换惩罚应有滞回：新方向需要连续多帧优于旧方向
才切换，但在当前方向目标消失、另一方向稳定存在时允许快速切换。

P0 第一版不必一次实现全部信号。建议按下面顺序消融：

1. `F/B/Ø + mask IoU + 面积/质心稳健统计`；
2. 加模型内部 predicted IoU、presence 和 stability；
3. 加前后向一致性过滤后的光流；
4. 最后加门控后的 WiLoR。

光流首选现有 PyTorch/torchvision 可用的 RAFT 权重；若希望无新增模型，OpenCV
光流可作为便宜基线，但不能默认它在高速手指、运动模糊和遮挡处可靠。
[RAFT](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf)
明确把快速运动、遮挡、模糊和无纹理区域列为困难点；
[MirrorFlow](https://openaccess.thecvf.com/content_iccv_2017/html/Hur_MirrorFlow_Exploiting_Symmetries_ICCV_2017_paper.html)
说明了用 forward–backward flow consistency 推断不可靠/遮挡区域的依据。

### 5.3 边界带软融合（P1，可选）

硬选择后，定义一致核心 `C = F ∩ B` 和分歧带 `D = F xor B`。只在 `D` 内融合
两方向 logits，并以模型质量和可靠光流加权；`C` 内保持前景，远离 `F ∪ B`
保持背景。融合结果还必须满足：

- 不明显增加到另一只手/物体的连通桥；
- 不降低可靠目标关节覆盖；
- 相对邻帧 warp 后边界没有恶化；
- 若任一检查失败，退回 Viterbi 选出的完整候选。

这样可吸收两边互补的手指边界，又避免全图平均造成身份混合。

## 6. 推荐方案 B：保守的不确定区间与双锚点重传播

用户偏好是宁可漏触发，也不要频繁自动标注。建议采用**多证据 AND 门控**：

### 6.1 区间起始条件

初始阈值仅用于验证，不是最终固定值：

- `IoU(F_t, B_t) < 0.35` 持续至少 5 帧；并且
- 至少满足以下一项：某方向模型质量显著降低、可靠 flow-warp IoU 突降、面积/
  质心/拓扑出现稳健异常、稀疏图像检测不支持该 mask、可靠 WiLoR 强烈反对；
- 用更宽松的结束阈值形成滞回，例如一致性恢复到 0.6 以上并持续若干帧才结束。

阈值应在独立验证视频上通过“低误触发率下的错误召回”选择。正反向同时为空、
同时选错实例或同时粘住物体时，冲突信号不会触发，所以还必须保留随机抽检和
稀疏语义重检测。

### 6.2 可信锚点

在区间左右搜索最近的可信帧，要求正反向高度一致、模型质量高、轨迹无突变，
并且身份证据无冲突。锚点 mask 的生成优先级是：

1. 两方向几乎一致时取稳定核心再用 SAM 3 单帧 mask/points refine；
2. 一方向得到多项独立证据支持时取该方向完整 mask；
3. 没有可信锚点则不自动重标，保留现有选择并写入 uncertainty metadata。

从左右锚点分别创建**新 session**，仅向区间内部传播，得到 `L/R` 新候选，再对
`F/B/L/R/Ø` 做一次区间级 Viterbi。仅当新路径改善综合一致性且不破坏锚点时
接受；否则回滚。这里“自动标注”本质是以高置信 mask 作为新视觉提示，不是把
未经验证的 WiLoR 点直接写进记忆。

这一设计和 [MiVOS 的双向独立传播及交互结果融合](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Modular_Interactive_Video_Object_Segmentation_Interaction-to-Mask_Propagation_and_Difference-Aware_Fusion_CVPR_2021_paper.html)
相近；MiVOS 会在新交互帧与旧交互帧之间双向传播并融合。区别是这里把自动
锚点门槛设得更保守，并要求新 session 隔离错误记忆。

## 7. WiLoR：只能是经过门控的弱专家

[WiLoR](https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf)
是逐图像的手检测和 3D 重建系统，论文明确其视频展示不使用 temporal component。
这意味着逐帧高检测分数不能保证轨迹身份、handedness 或遮挡状态连续正确。

建议先构造 `valid_wilor(t)`：

- detection confidence 和已有 `reliability_score_px` 通过阈值；
- handedness 在局部窗口内稳定；
- wrist/MCP 的速度和加速度不出现孤立尖峰（Hampel/MAD 检查）；
- 可见关节空间结构、手框尺度与近期轨迹相容；
- 不能与两方向 SAM 的共同稳定区域严重矛盾。

只有 `valid_wilor(t)` 为真时才计算：目标关节落入 mask 的比例、mask 到手框的
覆盖/泄漏、与反侧手关节的排斥。它只应：

- 在 `F/B` 其他证据接近时打破平局；
- 对明显覆盖反侧手的候选施加惩罚；
- 作为高不确定触发所需的“第二证据”之一。

它不应：单独新增正点、单独删掉 mask、单独声明左右手身份，或在严重遮挡时
强制 mask 包含预测关节。

当前 WiLoR 脚本要求正点必须已落在上一轮目标 mask 内，这很安全，却也意味着
它主要能维持已有区域，不能恢复被 baseline 完全漏掉的手指/手掌。区间重传播
可补这个缺口：只在可信锚点上从 mask 生成提示，而不是在失败帧盲目扩张。

## 8. 其他可选方法及评估

### 8.1 多锚点共识（P1，零训练）

不只从首尾两帧出发，而是先在全视频稀疏采样帧上做图像级检测，从中选取
外观清晰、模型质量高、彼此外观覆盖多样的 3–5 个锚点。每个锚点创建独立
session 向左右传播，在每帧形成多候选，再用束搜索/Viterbi 选路径。

优点是能产生首尾候选没有的正确 mask，尤其适合中段遮挡后重新出现；缺点是
锚点选择错误会引入更多假设。它与 [XMem++ 的多永久记忆帧和下一标注帧推荐](https://openaccess.thecvf.com/content/ICCV2023/html/Bekuzarov_XMem_Production-level_Video_Segmentation_From_Few_Annotated_Frames_ICCV_2023_paper.html)
有相同动机，但这里的锚点由严格自动门控产生。

### 8.2 SAM2Long 式多路径记忆树（P1，零训练但侵入内部）

[SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)
保留多个 mask/memory 路径，根据累计 predicted-IoU 分数做受限树搜索，并过滤
低 presence/低质量记忆。论文在其 12 个直接对比中报告平均约 3.7 J&F 提升，
最高约 5.3；这些数字来自 SAM 2 和通用 VOS，不能直接外推到 SAM 3 手部任务。

SAM 3 本地 tracker 已生成多 mask，并在基础版本默认含记忆选择，所以可先做
小型适配：只在低稳定/遮挡区间保留 2–3 条路径。SAM 3.1 multiplex 路径的实现
不同，当前 builder 中 `use_memory_selection=False`，应单独研究，不能直接复制
SAM 2 patch。

### 8.3 SAMURAI 式运动感知记忆（P1）

[SAMURAI](https://arxiv.org/abs/2411.11922) 在无需微调的前提下，把 Kalman 运动
分数与 mask affinity 结合，并筛选记忆帧；其报告的收益主要是 box-level VOT。
对手部，简单恒速 Kalman 容易被快速关节运动、相机运动和形变破坏，因此更适合
做候选/记忆否决信号，而非单一选择器。可用 flow 后的 mask 质心和尺度替代纯
box 运动模型。

### 8.4 DEVA 式 clip consensus（P1/P2）

[DEVA](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_Tracking_Anything_with_Decoupled_Video_Segmentation_ICCV_2023_paper.pdf)
先将邻近未来帧的图像分割反向对齐到当前帧形成 in-clip consensus，再和历史
传播结果融合；论文默认 clip size 3、每 5 帧融合一次，并显示其双向方案优于
若干单向/短轨迹基线。这个方向非常契合未来开放词汇接触物体，但对当前指定手
任务，先实现首尾双候选会更简单。DEVA 的实例关联和 consensus 机制可作为 P1
多锚点版本的参考。

### 8.5 多假设全视频优化（P1/P2）

[MHP-VOS](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_MHP-VOS_Multiple_Hypotheses_Propagation_for_Video_Object_Segmentation_CVPR_2019_paper.html)
主张延迟逐帧决定，保留多条跟踪假设，待获得全视频证据后再选择；这直接支持
本文采用 Viterbi/束搜索而非贪心融合。

### 8.6 替换 tracker（P2）

- [Cutie](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html)
  用 object-level memory 抑制相似背景/干扰物；
- [RMem](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_RMem_Restricted_Memory_Banks_Improve_Video_Object_Segmentation_CVPR_2024_paper.html)
  表明扩大记忆并不总是好，受限且高质量的记忆可能更准；
- [STM](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)
  是空间—时间记忆传播的经典方案。

可以用 SAM 3 文本检测生成高质量锚点，再交给 Cutie/XMem 类 tracker 双向传播，
作为“是否问题主要来自 SAM 3 tracker”的诊断基线。但这会引入新权重、依赖和
许可审查，不应先于 P0。

### 8.7 原生双向网络（P3，不推荐作为起点）

可设想让每帧同时 cross-attend 正向和反向 memory，再由门控器融合。但未经
训练直接拼接、平均或互写 hidden memory，会改变模型训练时的数据分布，并可能
让一个方向的错误污染另一个方向。安全实现至少需要：

- 冻结 SAM 3 backbone/tracker；
- 训练很小的双向门控/temporal refiner；
- 用方向 dropout、遮挡和错误 mask 增强训练；
- 与外部 Viterbi 融合作严格回归对比。

只有零训练方案达到 oracle 上限、仍明显不能满足需求时，才值得进入这一阶段。

一个更轻的已发表参照是
[DVIS](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_DVIS_Decoupled_Video_Instance_Segmentation_Framework_ICCV_2023_paper.html)：
它把分割、跟踪、时序 refinement 解耦，学习的 tracker/refiner 只占 segmenter
很小一部分计算量。若最终确实需要训练，优先训练这种外置轻量 refiner，而不是
重训 SAM 3。另需注意，[FSNet](https://openaccess.thecvf.com/content/ICCV2021/html/Ji_Full-Duplex_Strategy_for_Video_Object_Segmentation_ICCV_2021_paper.html)
所谓 full-duplex/bidirectional 主要是 appearance 与 motion 两种模态间的信息
交互，不等价于本文的时间正反向双流，不能只因名称相似就直接采用。

## 9. 评估设计

### 9.1 数据与标注

不要只标正反向冲突帧，否则会遗漏“两边一起错”。建议按视频划分 train-free
阈值验证集和最终测试集，不能按帧随机切分：

- 选择至少 8–12 段，覆盖左右手、双手交叉、手物遮挡、运动模糊、进出画面、
  快速运动、肤色/袖口相似和长时间遮挡；
- 对每个高分歧区间及其前后各密集标注一小段；
- 另从高一致区间随机采样至少同量帧，检查共同错误；
- 若资源允许，对 4–6 个最困难短 clip 做逐帧 dense mask；其余做稀疏标注。

公开基准可补充但不能替代内部手物数据：[DAVIS](https://davischallenge.org/)
覆盖遮挡、模糊和外观变化；[LVOS](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_LVOS_A_Benchmark_for_Long-term_Video_Object_Segmentation_ICCV_2023_paper.pdf)
强调长遮挡、重现和相似物；[EPIC-KITCHENS VISOR](https://epic-kitchens.github.io/VISOR/site)
提供手、active object、关系以及大量 dense interpolation，最接近未来 TODO；
[EgoHOS](https://arxiv.org/abs/2208.03826) 提供细粒度手—物接触边界。

### 9.2 必跑基线

1. 正向 `F`；
2. 反向 `B`；
3. 每视频选 `F/B` 中更好方向的 oracle；
4. 每帧选 `F/B` 中更好 mask 的 oracle；
5. `F ∪ B`、`F ∩ B`、可用时的 logits 平均；
6. Viterbi（逐项加入模型分数、flow、WiLoR）；
7. Viterbi + 保守区间重传播；
8. 多锚点或 SAM2Long 式多路径（进入 P1 后）。

oracle 只用于判断方法空间，不是可部署结果。还要单独对“物体可见”和“物体
完全不可见”帧计分，防止靠大量空帧虚增均值。

### 9.3 指标

主指标：

- region IoU (`J`) 与 boundary F-score (`F`)；
- 串到反侧手、前臂、接触物体的像素比例和 identity-switch 次数；
- 最差 10% 帧的 J&F、失败区间数量/长度，而不只看平均值；
- 运动补偿后的 mask IoU、面积/质心波动，作为时序指标；
- 目标存在/不存在的 precision、recall。

不确定性指标：

- 对“J 低于失败阈值”的 AUROC/AUPRC；
- 固定极低误触发率时能召回多少真实失败；
- selective risk：只自动接受低不确定帧时，保留比例与错误率的关系；
- 触发区间数、重传播次数和人工复核帧数。

效率指标记录 GPU 时间、峰值显存、session 数和中间存储，但当前不作为淘汰条件。

### 9.4 成功门槛

建议在实验前冻结门槛，避免看完结果再改标准：

- 相比 `max(mean(F), mean(B))`，测试集 J&F 有稳定提升；
- 串掩码与 identity switch 不能恶化；
- 最差 10% 帧和最长失败区间必须改善；
- 保守自动修正的 precision 优先于 recall；
- 每个消融在不同视频/人物上方向一致，而非由单个 clip 拉高。

## 10. 实施路线图

### P0-A：先证明候选互补

- 固化物理反转视频与 API backward 的等价性测试；
- 保存两个独立 session 的逐实例 mask、logits 和内部质量量；
- 做实例匹配及 `F/B`、union/intersection、两个 oracle 的报告；
- 标注少量冲突与一致区间，确认主要失败属于漏分、串物、串手还是消失。

若逐帧 oracle 相比最佳单方向提升很小，停止研究纯融合，直接进入多锚点候选生成。

### P0-B：部署级最小方案

- 实现 `F/B/Ø` Viterbi；
- 先用模型分数 + 几何稳健统计，不依赖 WiLoR 和重型光流；
- 输出每帧所选方向、分项分数、切换原因和 uncertainty；
- 与简单逐帧选高分对比，验证没有新增 flicker。

### P0-C：保守修复

- 加持续区间检测和滞回；
- 从可信双侧锚点创建新 session 向内传播；
- 新结果通过验收门控后才替换；否则保留并标记复核。

### P1：增强

- 加 RAFT 及 forward–backward flow reliability；
- 加 WiLoR 时序可靠性门控；
- 只在分歧边界带做 logits 融合；
- 比较多锚点、SAM2Long 式 2–3 路束搜索和替换 tracker。

### P2/P3：仅在上限不足时

- 用少量 GT 训练 mask-quality/calibration 模型；
- 冻结 SAM 3，只训练轻量时序 refiner；
- 最后才考虑原生双向 memory architecture。

## 11. 接触物体扩展 TODO

暂不改变当前手部分割范围，但保留以下接口和研究任务：

- [ ] 只给 `left/right hand` 时自动发现未知接触物体，并允许可选文本提示缩小候选；
- [ ] 将“物体身份轨迹”和“逐帧接触状态”分开，确认接触后仍跟踪完整可见时段；
- [ ] 输出手/物独立实例 mask、可派生联合区域、接触关系及各自 uncertainty；
- [ ] 不允许当前手 mask 融合把接触物永久吞入手实例；
- [ ] 用 VISOR/EgoHOS 定义关系评估，并研究 HOIST-Former 类“手持物身份持续”任务；
- [ ] 在多实例关联层保留 `hand_id -> object_id -> contact interval`，不要只保存前景视频。

## 12. 最终建议

立即修改主网络的性价比很低。当前最合理的研究假设是：正反向各自提供了不同
错误积累路径，而全视频优化可以在不训练的情况下利用其互补性；当两者共同失败
时，再由保守的可信锚点产生新候选。这个设计同时吸收了 MiVOS 的双向传播、
DEVA 的跨帧共识、MHP-VOS/SAM2Long 的延迟决策与多假设、SAMURAI 的运动感知
记忆思想，但把它们约束在现有 SAM 3 和手部身份任务上。

最重要的三个工程原则是：

1. 两个方向的 session、实例 ID 和记忆必须独立；
2. 不把首帧检测分数当逐帧质量，也不把正反向一致当正确；
3. WiLoR、光流和自动锚点都必须可被否决，最终替换要有回滚路径。

## 13. P0 实现说明

P0 已实现于 [`process_bidirectional_videos.py`](../scripts/process_bidirectional_videos.py)：

```bash
python scripts/process_bidirectional_videos.py \
  --input-root DATA \
  --output-root OUT \
  --prompt "left hand" \
  --version sam3 \
  --device cuda:0 \
  --backward-mode physical
```

默认 `--backward-mode verify` 会分别运行物理倒序帧和 SAM backward API；只有
逐帧 mask、存在状态及 tracker 分数达到等价门槛时才采用 API 结果，否则停止并
输出 `backward_equivalence.json` 和 `backward_verification/{physical,api}` 候选，
等待显式选择 `physical` 或 `api`。P0-C 默认关闭，完成验证集阈值校准后可传
`--repair-uncertain` 开启。

正序、物理倒序和 API backward 都保存 `add_prompt` 返回的锚点帧结果，再从相邻帧继续传播，保证锚点不丢失且每帧只处理一次。2026-09-04 修复前生成的 `backward_equivalence.json` 使用了不对称锚点协议，不应作为两种倒序实现是否等价的依据；需要重新运行真实 GPU 验证。

每个序列输出：

- `forward/masks.mkv`、`backward/masks.mkv`：两个独立 session 的全部实例标签；
- `masks.mkv`：Viterbi/P0-C 汇总后的目标手二值 mask；
- `result.mp4`：Forward、Backward、Fused、Difference 的 2×2 对比；
- `frames.jsonl`：逐帧方向、分数、分歧、切换和不确定性；
- `audit.json`：无需 GT 的双向分歧审计；
- `evaluation.json`：传入 `--ground-truth-root` 时的 J、F、J&F、尾部指标及
  oracle 报告；GT 直接读取交互工具生成的 FFV1 `masks.mkv`，不生成 PNG。

公开预测输出新增了统一的 `out_tracker_probs`。当前 P0 尚未从 tracker 深层
状态导出语义纯净的 presence、predicted IoU 与低分辨率 logits；几何 stability
proxy 会明确标注，不能冒充模型 stability。这些信号应在 P1 或发现现有分数
不足以区分候选时再接入。

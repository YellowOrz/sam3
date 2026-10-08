# SAM 3 改进与实验记录

本文件记录对 SAM 3 推理行为的改进，以及对应实验结果。新的改进写在旧的改进前面。指标来自各次运行的 `gt_summary.json`，评测第 0、3、6… 帧。平均 IoU 是逐帧平均，预测与 GT 都为空时该帧记 1。

## TODO

- 历史否决：检测和轨迹都相对上一张实际写入的掩码膨胀（候选几乎盖住上一张，面积约 2 倍）时，这一帧不编码。输出仍用当前掩码，手在动，不用上一帧的形状去裁。连续跳过超过 `num_maskmem`（7）再强制写入，否则近期记忆会空。`pred_masks` 存的是替换前的轨迹，不能当历史。pointer 对齐后如果胳膊仍按帧长回来，再做这一项。
- `cup`：高分检测贴近上一张已写入掩码、而轨迹不贴近时，直接采用检测。上一张本身是错的会被锁住。放在 pointer 实验之后，用同一套右手数据看 `cup` 会不会上来、`blue_pen` 会不会再掉。

## 2026-09-30 — 替换掩码后重算 obj_ptr

汇总提升几乎没有，实现未并入 `dev`。代码在分支 `experiment/clip-overseg-obj-ptr`。

上一节只换掩码：spatial memory 用替换后的掩码，`obj_ptr` 仍来自替换前的 tracker 解码。这次对被替换的对象，在 `_tracker_update_memories` 之前用这张掩码做 dense prompt，经 `Sam3TrackerBase._use_mask_as_output` 重算 `obj_ptr`。只取新的 pointer，decoder 重新预测的掩码丢弃。特征是当前帧原始图像特征，不经过旧 memory。未替换的对象不动。

### 实验

数据集、GT、`--compare-skip-frames 2` 与上一节相同。对照是上一节的 `--clip-overseg`（只换掩码）。本次 detector 与上一节在四位小数上逐项相同，下表不重复 detector 列。文本批次状态仍是 failed，`left_hand` 没有 `right_hand.mkv`；汇总只含成功序列，1852 帧。MANO 批次状态是 success，同样 1852 帧。

文本：

```bash
uv run python scripts/process_dataset_videos.py \
    --input-root /data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh \
    --output-root outputs/clip_overseg_obj_ptr/realsense_hand_object_with_seg_to_wjh/right_hand \
    --version sam3 --text-prompt "right hand" --device cuda:2 --clip-overseg \
    --compare-gt --gt-dir-name masks_sam3 --gt-mask-name right_hand.mkv \
    --compare-skip-frames 2 --save-detector
```

MANO 框，`--prompt-mode box`，`--mano-dir-name MANO_wilor_origin_conf0.6`：

```bash
uv run scripts/process_mano_prompt_videos.py \
    --input-root /data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh \
    --output-root outputs/clip_overseg_obj_ptr/realsense_hand_object_with_seg_to_wjh/right_hand_box_from_origin_wilor_conf0.6/ \
    --text-prompt "right hand" --hand-side right --prompt-mode box \
    --mano-dir-name MANO_wilor_origin_conf0.6 --device cuda:3 --clip-overseg \
    --compare-gt --gt-dir-name masks_sam3 --gt-mask-name right_hand.mkv \
    --compare-skip-frames 2 --save-detector
```

### 结果

`mean_*` 是逐帧平均，`pixel_*` 是全部采样帧的像素汇总。`clip` 是上一节只换掩码，`ptr` 是这次同时重算 `obj_ptr`。

| 设置 | mean_iou | mean_dice | pixel_iou | pixel_dice |
| --- | ---: | ---: | ---: | ---: |
| 文本 + clip | 0.8838 | 0.9029 | 0.8533 | 0.9208 |
| 文本 + ptr | 0.8874 | 0.9044 | 0.8613 | 0.9255 |
| MANO + clip | 0.9240 | 0.9447 | 0.9075 | 0.9515 |
| MANO + ptr | 0.9241 | 0.9449 | 0.9078 | 0.9517 |

逐序列 `mean_iou`：

| 序列 | 文本 clip | 文本 ptr | MANO clip | MANO ptr |
| --- | ---: | ---: | ---: | ---: |
| basket | 0.9568 | 0.9465 | 0.9370 | 0.9365 |
| black_pen | 0.9239 | 0.9668 | 0.9604 | 0.9612 |
| blue_pen | 0.9819 | 0.9838 | 0.9815 | 0.9803 |
| bottle | 0.9676 | 0.9697 | 0.9699 | 0.9694 |
| bowl | 0.9640 | 0.9638 | 0.9633 | 0.9644 |
| cup | 0.6760 | 0.6766 | 0.6837 | 0.6844 |
| milk | 0.9498 | 0.9495 | 0.9387 | 0.9404 |
| red_pen | 0.9488 | 0.9550 | 0.9543 | 0.9536 |
| right_hand | 0.5634 | 0.5634 | 0.9396 | 0.9396 |

逐序列 `mean_dice`：

| 序列 | 文本 clip | 文本 ptr | MANO clip | MANO ptr |
| --- | ---: | ---: | ---: | ---: |
| basket | 0.9722 | 0.9632 | 0.9574 | 0.9575 |
| black_pen | 0.9584 | 0.9825 | 0.9792 | 0.9798 |
| blue_pen | 0.9883 | 0.9892 | 0.9880 | 0.9874 |
| bottle | 0.9814 | 0.9825 | 0.9826 | 0.9824 |
| bowl | 0.9812 | 0.9811 | 0.9808 | 0.9814 |
| cup | 0.7180 | 0.7182 | 0.7234 | 0.7238 |
| milk | 0.9707 | 0.9706 | 0.9646 | 0.9656 |
| red_pen | 0.9694 | 0.9728 | 0.9728 | 0.9724 |
| right_hand | 0.5666 | 0.5666 | 0.9661 | 0.9661 |

逐序列 `pixel_iou`：

| 序列 | 文本 clip | 文本 ptr | MANO clip | MANO ptr |
| --- | ---: | ---: | ---: | ---: |
| basket | 0.9647 | 0.9580 | 0.9335 | 0.9318 |
| black_pen | 0.8830 | 0.9470 | 0.9379 | 0.9405 |
| blue_pen | 0.9854 | 0.9875 | 0.9847 | 0.9832 |
| bottle | 0.9597 | 0.9632 | 0.9640 | 0.9633 |
| bowl | 0.9340 | 0.9339 | 0.9330 | 0.9348 |
| cup | 0.7021 | 0.7025 | 0.7071 | 0.7078 |
| milk | 0.9478 | 0.9477 | 0.9325 | 0.9358 |
| red_pen | 0.9423 | 0.9566 | 0.9537 | 0.9523 |
| right_hand | 0.4419 | 0.4419 | 0.9309 | 0.9309 |

逐序列 `pixel_dice`：

| 序列 | 文本 clip | 文本 ptr | MANO clip | MANO ptr |
| --- | ---: | ---: | ---: | ---: |
| basket | 0.9820 | 0.9785 | 0.9656 | 0.9647 |
| black_pen | 0.9379 | 0.9728 | 0.9680 | 0.9694 |
| blue_pen | 0.9926 | 0.9937 | 0.9923 | 0.9915 |
| bottle | 0.9795 | 0.9812 | 0.9817 | 0.9813 |
| bowl | 0.9659 | 0.9658 | 0.9654 | 0.9663 |
| cup | 0.8250 | 0.8253 | 0.8285 | 0.8289 |
| milk | 0.9732 | 0.9732 | 0.9651 | 0.9668 |
| red_pen | 0.9703 | 0.9778 | 0.9763 | 0.9756 |
| right_hand | 0.6130 | 0.6130 | 0.9642 | 0.9642 |

文本 `black_pen` 四个指标从 0.9239 / 0.9584 / 0.8830 / 0.9379 升到 0.9668 / 0.9825 / 0.9470 / 0.9728。文本 `red_pen` 的 `pixel_iou` 从 0.9423 升到 0.9566。文本 `basket` 从 0.9568 / 0.9722 / 0.9647 / 0.9820 降到 0.9465 / 0.9632 / 0.9580 / 0.9785。文本 `blue_pen` 从 0.9819 / 0.9883 / 0.9854 / 0.9926 回到 0.9838 / 0.9892 / 0.9875 / 0.9937，仍低于基线 0.9842 / 0.9894 / 0.9878 / 0.9939。`right_hand` 两条路径都与只换掩码相同。`cup` 文本从 0.6760 / 0.7180 / 0.7021 / 0.8250 到 0.6766 / 0.7182 / 0.7025 / 0.8253，MANO 从 0.6837 / 0.7234 / 0.7071 / 0.8285 到 0.6844 / 0.7238 / 0.7078 / 0.8289，仍低于同次 detector（文本 0.8953 / 0.9144 / 0.8244 / 0.9037，MANO 0.8974 / 0.9319 / 0.8199 / 0.9010）。

### 结论

重算 `obj_ptr` 后，文本四个指标从 0.8838 / 0.9029 / 0.8533 / 0.9208 升到 0.8874 / 0.9044 / 0.8613 / 0.9255；MANO 从 0.9240 / 0.9447 / 0.9075 / 0.9515 升到 0.9241 / 0.9449 / 0.9078 / 0.9517。顺序都是 `mean_iou`、`mean_dice`、`pixel_iou`、`pixel_dice`。文本的 `pixel_iou` 加了 0.0080，MANO 四个指标的变化都不超过 0.0003。提升集中在文本 `black_pen` 和 `red_pen`；文本 `basket` 变差。`cup` 几乎不动，检测仍未被轨迹包住。历史否决和 `cup` 两条仍留在 TODO。

## 2026-09-29 — 过分割轨迹改用高分检测

开关 `--clip-overseg`，默认关闭，仅基础 SAM 3。实现是 `clip_oversegmented_tracker_masks`，在重叠抑制之后、写入 tracker memory 之前替换 `tracker_low_res_masks_global`。当前帧输出和随后写入的 spatial memory 使用同一张替换后的掩码。`obj_ptr` 仍来自替换前的 tracker 解码。

已有对象的可见掩码来自 tracker。原纠正要求检测与轨迹的 IoU 至少 0.8，且约每 16 帧才尝试。检测几乎落在轨迹内部、轨迹又明显更大时，IoU 落在关联阈值 0.1 和纠正阈值 0.8 之间：不会新建对象，也不会纠正，膨胀区域会写入 memory 并持续到后续帧。

### 替换条件

以下条件同时满足才替换。命中后用该检测的 mask logits 替换轨迹，不做 logit 平均，也不取交集。

- 检测分数 ≥ 0.8
- 覆盖率 `|D ∩ T| / |D|` ≥ 0.9，即检测几乎落在轨迹内部
- 面积比 `|T| / |D|` 在 1.25 到 10 之间
- 该检测只对应这一条轨迹，这条轨迹也只对应这一个检测

检测为空、一对多，或面积比大于 10 的小块检测，都不替换。规则不读取类别或文本；阈值来自这批右手视频。

`metadata.json` 记录 `clip_overseg`。旧结果没有该字段时视为关闭，不会因此重跑；开关与已有结果不一致时重新推理。

### 实验

数据集 `/data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh`，文本提示 `right hand`，GT 为同级 `masks_sam3/right_hand.mkv`，`--compare-skip-frames 2`。`left_hand` 没有该 GT，文本两组的批次状态都是 failed，下表汇总只含成功序列，共 1852 帧。两组 detector 指标分别与各自基线逐项相同。

文本，`--clip-overseg`：

```bash
uv run python scripts/process_dataset_videos.py \
    --input-root /data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh \
    --output-root outputs/clip_oversegmented_tracker_masks/realsense_hand_object_with_seg_to_wjh/right_hand \
    --version sam3 --text-prompt "right hand" --device cuda:2 --clip-overseg \
    --compare-gt --gt-dir-name masks_sam3 --gt-mask-name right_hand.mkv \
    --compare-skip-frames 2 --save-detector
```

文本基线，无 `--clip-overseg`：`outputs/origin_sam3/realsense_hand_object_with_seg_to_wjh`。

MANO 框，`--prompt-mode box`，`--mano-dir-name MANO_wilor_origin_conf0.6`，`--clip-overseg`：

```bash
uv run scripts/process_mano_prompt_videos.py \
    --input-root /data/xuzhefeng/Datasets/realsense_hand_object_with_seg_to_wjh \
    --output-root outputs/clip_oversegmented_tracker_masks/realsense_hand_object_with_seg_to_wjh/right_hand_box_from_origin_wilor_conf0.6/ \
    --text-prompt "right hand" --hand-side right --prompt-mode box \
    --mano-dir-name MANO_wilor_origin_conf0.6 --device cuda:3 --clip-overseg \
    --compare-gt --gt-dir-name masks_sam3 --gt-mask-name right_hand.mkv \
    --compare-skip-frames 2 --save-detector
```

MANO 基线：`outputs/mano_prompt/realsense_hand_object_with_seg_to_wjh/right_hand_box_from_origin_wilor_conf0.6`。

### 结果

`mean_*` 是逐帧平均，`pixel_*` 是全部采样帧的像素汇总。detector 与同一次运行的最终掩码使用同一批帧。


| 设置            | mean_iou | mean_dice | pixel_iou | pixel_dice |
| ------------- | -------- | --------- | --------- | ---------- |
| 文本基线          | 0.7683   | 0.8286    | 0.6806    | 0.8099     |
| 文本 + clip     | 0.8838   | 0.9029    | 0.8533    | 0.9208     |
| 文本 detector   | 0.8633   | 0.8798    | 0.8431    | 0.9149     |
| MANO 基线       | 0.8074   | 0.8700    | 0.7209    | 0.8378     |
| MANO + clip   | 0.9240   | 0.9447    | 0.9075    | 0.9515     |
| MANO detector | 0.9100   | 0.9383    | 0.8972    | 0.9458     |


逐序列 `mean_iou`：


| 序列         | 文本基线   | 文本 + clip | 文本 detector | MANO 基线 | MANO + clip | MANO detector |
| ---------- | ------ | --------- | ----------- | ------- | ----------- | ------------- |
| basket     | 0.9372 | 0.9568    | 0.8428      | 0.9359  | 0.9370      | 0.8370        |
| black_pen  | 0.8547 | 0.9239    | 0.9486      | 0.8541  | 0.9604      | 0.9397        |
| blue_pen   | 0.9842 | 0.9819    | 0.9561      | 0.9800  | 0.9815      | 0.9440        |
| bottle     | 0.7905 | 0.9676    | 0.9568      | 0.7900  | 0.9699      | 0.9299        |
| bowl       | 0.7379 | 0.9640    | 0.9726      | 0.7379  | 0.9633      | 0.9186        |
| cup        | 0.5906 | 0.6760    | 0.8953      | 0.5956  | 0.6837      | 0.8974        |
| milk       | 0.6625 | 0.9498    | 0.9300      | 0.6632  | 0.9387      | 0.9161        |
| red_pen    | 0.7928 | 0.9488    | 0.9345      | 0.7920  | 0.9543      | 0.9155        |
| right_hand | 0.5634 | 0.5634    | 0.3111      | 0.9396  | 0.9396      | 0.9084        |


逐序列 `mean_dice`：


| 序列         | 文本基线   | 文本 + clip | 文本 detector | MANO 基线 | MANO + clip | MANO detector |
| ---------- | ------ | --------- | ----------- | ------- | ----------- | ------------- |
| basket     | 0.9595 | 0.9722    | 0.8570      | 0.9570  | 0.9574      | 0.8560        |
| black_pen  | 0.9168 | 0.9584    | 0.9674      | 0.9165  | 0.9792      | 0.9628        |
| blue_pen   | 0.9894 | 0.9883    | 0.9749      | 0.9872  | 0.9880      | 0.9685        |
| bottle     | 0.8702 | 0.9814    | 0.9753      | 0.8700  | 0.9826      | 0.9574        |
| bowl       | 0.8227 | 0.9812    | 0.9835      | 0.8227  | 0.9808      | 0.9452        |
| cup        | 0.6681 | 0.7180    | 0.9144      | 0.6720  | 0.7234      | 0.9319        |
| milk       | 0.7829 | 0.9707    | 0.9588      | 0.7834  | 0.9646      | 0.9511        |
| red_pen    | 0.8765 | 0.9694    | 0.9494      | 0.8759  | 0.9728      | 0.9391        |
| right_hand | 0.5666 | 0.5666    | 0.3135      | 0.9661  | 0.9661      | 0.9490        |


逐序列 `pixel_iou`：


| 序列         | 文本基线   | 文本 + clip | 文本 detector | MANO 基线 | MANO + clip | MANO detector |
| ---------- | ------ | --------- | ----------- | ------- | ----------- | ------------- |
| basket     | 0.9210 | 0.9647    | 0.9148      | 0.9230  | 0.9335      | 0.9007        |
| black_pen  | 0.7702 | 0.8830    | 0.9324      | 0.7694  | 0.9379      | 0.9214        |
| blue_pen   | 0.9878 | 0.9854    | 0.9544      | 0.9826  | 0.9847      | 0.9390        |
| bottle     | 0.6993 | 0.9597    | 0.9429      | 0.6990  | 0.9640      | 0.9105        |
| bowl       | 0.4961 | 0.9340    | 0.9566      | 0.4961  | 0.9330      | 0.8768        |
| cup        | 0.6258 | 0.7021    | 0.8244      | 0.6281  | 0.7071      | 0.8199        |
| milk       | 0.5850 | 0.9478    | 0.9229      | 0.5858  | 0.9325      | 0.9039        |
| red_pen    | 0.7322 | 0.9423    | 0.9457      | 0.7309  | 0.9537      | 0.9194        |
| right_hand | 0.4419 | 0.4419    | 0.1907      | 0.9309  | 0.9309      | 0.8984        |


逐序列 `pixel_dice`：


| 序列         | 文本基线   | 文本 + clip | 文本 detector | MANO 基线 | MANO + clip | MANO detector |
| ---------- | ------ | --------- | ----------- | ------- | ----------- | ------------- |
| basket     | 0.9589 | 0.9820    | 0.9555      | 0.9599  | 0.9656      | 0.9477        |
| black_pen  | 0.8702 | 0.9379    | 0.9650      | 0.8696  | 0.9680      | 0.9591        |
| blue_pen   | 0.9939 | 0.9926    | 0.9767      | 0.9912  | 0.9923      | 0.9685        |
| bottle     | 0.8231 | 0.9795    | 0.9706      | 0.8228  | 0.9817      | 0.9532        |
| bowl       | 0.6632 | 0.9659    | 0.9778      | 0.6632  | 0.9654      | 0.9344        |
| cup        | 0.7698 | 0.8250    | 0.9037      | 0.7715  | 0.8285      | 0.9010        |
| milk       | 0.7382 | 0.9732    | 0.9599      | 0.7388  | 0.9651      | 0.9495        |
| red_pen    | 0.8454 | 0.9703    | 0.9721      | 0.8445  | 0.9763      | 0.9580        |
| right_hand | 0.6130 | 0.6130    | 0.3203      | 0.9642  | 0.9642      | 0.9465        |


文本 `blue_pen` 的 `mean_iou` 从 0.9842 降到 0.9819，`mean_dice` 从 0.9894 降到 0.9883，`pixel_iou` 从 0.9878 降到 0.9854，`pixel_dice` 从 0.9939 降到 0.9926。`right_hand` 两条路径的四个最终指标都与基线相同：文本上 detector 四个指标都更差，规则没有把轨迹收成那张检测；MANO 上轨迹已经接近检测。`cup` 四个指标都有提升，但仍低于 detector。

### 结论

打开 `--clip-overseg` 后，文本四个指标从 0.7683 / 0.8286 / 0.6806 / 0.8099 升到 0.8838 / 0.9029 / 0.8533 / 0.9208；MANO 从 0.8074 / 0.8700 / 0.7209 / 0.8378 升到 0.9240 / 0.9447 / 0.9075 / 0.9515。顺序都是 `mean_iou`、`mean_dice`、`pixel_iou`、`pixel_dice`。两组汇总上都高于同次 detector。`cup` 说明检测未被轨迹包住时，这条规则不会把结果拉到 detector。
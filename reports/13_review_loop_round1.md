# 两轮车闭环第一轮：人工复核回灌 → 真实指标 → 重训对比（含 3 个特殊情况处理）

- 日期：2026-08-16
- 分支：`feat/codex/elevator`
- 承接：报告 12（两轮车交叉标注闭环，120 图）；人工复核已完成并落盘 `outputs/elevator_review_tw/elevator_review_decisions.json`

---

## 0. 结论（TL;DR）

1. **人工复核完成**：120 图 / 124 教师框裁决 → **accept 77 / delete 43 / wrong_class 4**（教师提案 35% 被拒，佐证了"电瓶原始图混杂 + 教师同目标双类别"）。
2. **真实指标（vs 人工 GT 81 框）**：
   - **教师（LA）**：P 0.621 / R **0.951** / F1 **0.751** —— 召回极高，精度被 43 误检 + 4 类别翻转拖累；
   - **YOLO 批评者（detect_v2 训练）**：P 0.103 / R 0.074 / F1 **0.086**（碎片合并后）—— **在电梯两轮车场景几乎不可用**，这是真实跨域泛化结论（报告 09 的担忧得到证实）。
3. **3 个特殊情况已量化处理**：
   - Case 1 YOLO 碎片化：71 → 合并 58（~18% 碎片），真实指标计算已加同类合并；
   - Case 2 教师同目标双类别：人工已裁决（delete 其一），管线侧建议加"同目标跨类冲突"去重（TODO）；
   - Case 3 电瓶原始图混杂：**49/120（41%）图被全部删框**，其中 battery 17/30、wave2_battery 28/30 基本是噪声；已作为背景负样本纳入 v2 训练（val 含 3 个 background）。
4. **闭环重训对比（同用人工 GT val 12 图 / 9 实例）**：
   | 模型 | P | R | mAP50 | mAP50-95 |
   |---|---|---|---|---|
   | tw_run_v1（伪标签训练） | 0.852 | 0.676 | **0.757** | 0.576 |
   | tw_run_v2（人工 GT 训练） | 0.219 | 0.786 | **0.504** | 0.438 |
   - **诚实结论**：v2 未跑赢 v1。原因：v2 best@epoch1（几乎未微调）、训练数据太小（71 训练图 / 81 框）、val 仅 9 实例。**闭环机制已端到端跑通，但数据量不足以让 YOLO 从人工 GT 中获益** → 下一轮必须扩数据/调训练。
5. **下一步建议**：① 扩充两轮车正样本（wave2_ebike 30/30 + ebike_like 26/30 全部可复用，原始 ~1.7 万张），剔除 battery 组；② 调训练（更长 epochs/更低 LR/更强的锚定，避免 best@epoch1）；③ 管线加教师提案去重与 YOLO 碎片合并。

---

## 1. 人工复核统计

| 项 | 值 |
|---|---|
| 裁决文件 | `outputs/elevator_review_tw/elevator_review_decisions.json` |
| 覆盖 | 120 图 / 124 教师框（reviewed_images=120, boxes_total=124） |
| 决策分布 | **accept 77 / delete 43 / wrong_class 4**；漏检 missing=0 |
| 修正后 GT | 81 框：electric scooter 51 / bicycle 30 |
| 空图（全删） | **49/120（41%）**：battery 17/30、wave2_battery 28/30、ebike_like 4/30、wave2_ebike 0/30 |

> wrong_class 4 个：复核器未记录修正后类别，默认翻转为另一类（scooter↔bicycle），可在 `elevator_sample_tw_gt` 的 txt 中手工覆盖。

## 2. 真实指标（eval_review.py，IoU≥0.5 同类别匹配）

| 模型 | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 教师（原始 124 提案） | 77 | 47 | 4 | 0.621 | 0.951 | **0.751** |
| YOLO 批评者（detect_v2，合并后 58 框） | 6 | 52 | 75 | 0.103 | 0.074 | **0.086** |

- 教师 FN=4 = 4 个 wrong_class 框（位置对、类别翻转移位后教师旧类别不再匹配）——教师定位能力几乎全对；
- YOLO 批评者 F1 0.086：跨域不可用（detect_v2 场景=煤气罐/电动车照片，电梯场景差异大）。

## 3. 3 个特殊情况的量化与处理

### Case 1：YOLO 同一目标 2-3 个小框（碎片化）
- 量化：YOLO 两轮车框 71 → 同类 IoU>0.5 合并后 58（**13 个碎片，~18%**）；
- 处理：`eval_review.py` 真实指标计算前做同类合并（取并集）；
- TODO：在 `cross_check.py` 中也加合并，让复核工作表不再展示碎片框。

### Case 2：教师（LA）同目标同时检出 bicycle + scooter（高重叠）
- 现象：教师对同一目标输出两类框（类别置信分裂）；
- 处理：人工已裁决（保留其一 / 删除其一）；v2 GT 已去重；
- TODO：`annotate_yolo.py` 加"同目标跨类冲突"检测（IoU>0.5 且类别不同的框对），在 manifest/工作表打 conflict 标记，减少人工重复裁决。

### Case 3：数据集混杂电瓶原始图
- 量化：49 空图（battery 17 + wave2_battery 28 + ebike_like 4），即**电瓶类图像基本不是两轮车**；
- 处理：v2 训练把这些空图作为**背景负样本**（val 中 3 个 background），教会模型"电池≠两轮车"；
- 建议：后续采样**剔除 battery 组**，两轮车正样本主要来自 wave2_ebike（30/30）与 ebike_like（26/30）。

## 4. 闭环重训与对比

- 数据：`yolo_detect_tw_v2`（train 108 / val 12，人工 GT 81 框 + 49 空图负样本，与 v1 同 seed 同划分）
- 训练：`run_train_tw_v2.sh` → YOLO26s，early-stop@51（**best@epoch1**，小数据不稳定），`~/data/yolo_runs/tw_run_v2`
- 公平对比（两者均用人工 GT val 验证）：

| 模型 | 训练真值 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| tw_run_v1 | 教师伪标签（124 框） | 0.852 | 0.676 | **0.757** | 0.576 |
| tw_run_v2 | 人工 GT（81 框） | 0.219 | 0.786 | 0.504 | 0.438 |

- 判读：
  - v1 在干净人工 GT 上反而更高（伪标签 val 曾给 0.43 是因为把教师误检当 GT，人工 GT 更宽松且更真实）；
  - v2 best@epoch1 ≈ 未微调 → 说明**人工 GT 量太小 + 训练不稳定**，不是"人工标签有害"；
  - 结论：**闭环第一轮机制跑通（复核→GT→重训→对比），但数据量不足以支撑 YOLO 提升**；教师（LA）目前仍是该域更强的检测器（F1 0.751）。

## 5. 工具新增/修复

| 项 | 说明 |
|---|---|
| `scripts/elevator/apply_review.py` | 裁决 → 修正 GT（accept/delete/wrong_class 翻转 + missing），输出 `elevator_sample_tw_gt` |
| `scripts/elevator/eval_review.py` | 真实指标（教师/YOLO vs 人工 GT，YOLO 同类碎片合并） |
| `scripts/elevator/build_tw_dataset.py` | 支持空图负样本（v2 数据集） |
| `run_train_tw_v2.sh` / `run_val_compare.sh` | v2 训练 + 双模型公平对比 |

## 6. 下一步

1. **扩数据**：battery 组剔除；wave2_ebike + ebike_like 全量（~1.7 万图）分层采样 → 新一轮标注+复核（人工工作量可控：教师提案高召回，只需删误检）；
2. **调训练**：更长 epochs / 更低 LR / freeze 骨干，避免 best@epoch1；
3. **管线**：annotate 加跨类冲突标记；cross_check 加碎片合并；
4. **收敛判定**：用"教师修正率"（本批 43/124=34.7%）与"人工 GT 上 YOLO mAP"作为闭环指标，随迭代下降/上升即收敛。
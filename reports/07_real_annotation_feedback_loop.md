# LocateAnything 真实标注反馈闭环（人工修正 → 重训 → 真实 F1 验证）

- 日期：2026-08-13
- 分支：`feat/codex/yolo`
- 环境：WSL Ubuntu 24.04 / `locate_anything_sft` / RTX 5060 Ti 16GB（sm_120）
- 模型：`~/models/LocateAnything-3B`（预训练）；`~/lora_gas/run_v2`（微调）
- 数据：`C:\Data\datasets\detect`（原）→ `C:\Data\datasets\detect_v2`（修正）

---

## 0. 结论（TL;DR）

1. **反馈闭环管线已端到端跑通**，且工具化、可复现、可审计：
   `YOLO txt → 修正策略 → detect_v2 → manifest_v2 → train/val → LoRA run_v2 重训 → eval_det 双模型对比`。
2. 本阶段"人工修正"以**可审计的修正策略**落地：`bicycle → electric scooter` 类合并（报告 06 已论证的电动车/自行车子类混淆）+ 降采样（≤1280px，视觉 token ≤1610）+ 框质量校验；同时产出**逐框审阅工作表** `correction_worksheet.jsonl` 与修正后可视化（montage/previews），供人工视觉复核。
3. **重训 run_v2**：150 步 / 4:31 / 1.62 s/step（seq=1792，较 run_v1 的 2048 更快且留显存余量），末段 step loss 0.18–0.26，收敛正常。
4. **真实 F1 评估**（8 张 holdout，slow 模式，修正后 GT）：
   | 模型 | gas F1@Mean | scooter F1@Mean | macro F1@Mean |
   |---|---|---|---|
   | 预训练 | 1.000（mIoU 0.983） | 1.000（mIoU 0.998） | **1.000** |
   | 微调 run_v2 | 0.983（mIoU 0.978） | 1.000（mIoU 0.992） | **0.992** |
5. **必须说明的局限（半自洽口径）**：本批 holdout GT 源自同一教师模型的零样本标注，仅做了类合并修正，因此预训练模型天然"自洽"地命中 → F1≈1.0 不能证明检测能力真实提升，只能证明**重训未造成灾难性遗忘**。要得到真正独立的"真实 F1"，需人工（用户）用工作表对 holdout 做视觉确认，或用独立来源的 GT。
6. **架构洞察**：打破自洽的关键是引入**独立第二意见**——YOLO 模型训练完成后可作为交叉验证的 GT 来源，与 LocateAnything 标注流水线互相校验（见报告 08/10）。

---

## 1. 反馈闭环流程（本阶段实现）

```text
[教师模型零样本标注] scripts/annotation/annotate_yolo.py ──► detect/（YOLO txt，121 图 / 93 框）
        │
        ▼
[修正策略] scripts/annotation/make_corrected_dataset.py
           （类合并 / 框质量过滤 / 降采样）──► detect_v2/ + correction_worksheet.jsonl
        │
        ▼
[重建训练数据] scripts/annotation/yolo_txt_to_manifest.py ──► manifest_v2.jsonl
               scripts/annotation/build_train_jsonl.py ──► train/val/recipe
        │
        ▼
[LoRA 重训] scripts/train/train_lora.sh ──► ~/lora_gas/run_v2
        │
        ▼
[真实 F1 评估] scripts/eval/eval_feedback_loop.sh（eval_det.py：预训练 vs 微调，修正后 GT）
        │
        ▼
[人工复核（可选）] worksheet + montage → 人工改 YOLO txt → 一键重跑整个闭环
```

新增/改动的可复用脚本：
| 脚本 | 作用 |
|---|---|
| `scripts/annotation/make_corrected_dataset.py` | 修正策略 → 修正数据集副本 + 逐框审阅工作表 + 统计 |
| `scripts/annotation/yolo_txt_to_manifest.py` | 修正后 YOLO txt → LocateAnything manifest（txt→manifest 逆向转换） |
| `scripts/eval/eval_feedback_loop.sh` | 一键跑预训练 vs 微调双模型评估（修正后 GT） |

---

## 2. 数据与修正

| 项 | 值 |
|---|---|
| 原数据集 detect | 121 图 / 93 框（gas cylinder 58 / electric scooter 28 / bicycle 7） |
| 修正策略 | `bicycle→electric scooter`（7 框合并）；退化/微小框过滤（本次 0 丢弃）；`--max-side 1280` 降采样（39 张） |
| 修正后数据集 detect_v2 | 121 图 / 93 框（gas 58 / scooter 35 / bicycle 0）；`classes.txt` 保持 3 类（bicycle 为空类） |
| 训练数据 | 86 样本含框 → 78 train + 8 val（seed 42，repeat_time 3，min-area 0.002） |
| 视觉 token | 全量 ≤1610（`seq=1792` 全覆盖；原 4 张 >1536 token 的图不再被训练丢弃） |
| 审阅工件 | `outputs/annotation_detect/correction_worksheet.jsonl`、`yolo_viz_v2/`（montage + previews） |

> 降采样解决了报告 06 的"高帧样本丢弃"问题：run_v1 中 1080p 帧（token ~2546-4590）在 seq=2048 下被训练丢弃；本次降采样到 ≤1280px 后全部样本入训。

---

## 3. 重训 run_v2

| 项 | 值 |
|---|---|
| 配置 | LoRA r=64（LLM）/ sdpa / seq=1792 / grad checkpoint / DeepSpeed ZeRO-1（torch_adam）/ bf16 |
| 规模 | 150 步 / 4:31 / 1.62 s/step / 平均 1.17 样本/步（streaming packing） |
| 收敛 | 末段 step loss 0.176 → 0.259；train_loss(avg) 0.8204 |
| 显存 | 本次未记录峰值（seq=1792 < 2048 顶格档，训练期间无并发 GPU 任务） |
| 产物 | `~/lora_gas/run_v2`（checkpoint-50/100/150 + 最终模型，含 done.txt） |

---

## 4. 真实 F1 评估（8 张 holdout，slow 模式）

- 评估脚本：`scripts/eval/eval_feedback_loop.sh` → `eval_det.py`（模型 vs YOLO GT，P/R/F1@IoU + F1@Mean + matched-IoU）
- holdout：`outputs/annotation_detect/lora_data_v2/val.jsonl`（8 图：gas 6 / scooter 2）
- GT：detect_v2 修正后标签；模式 slow（fast/hybrid 在 1080p 大图上有回归，见报告 06）

**预训练**（`eval_v2_pretrained.json`）：
| class | GT | Pred | P@.5 | R@.5 | F1@.5 | F1@.75 | F1@.9 | F1@Mean | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gas cylinder | 6 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 |
| electric scooter | 2 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 |
| **macro** | | | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | |

**微调 run_v2**（`eval_v2_finetuned.json`）：
| class | GT | Pred | P@.5 | R@.5 | F1@.5 | F1@.75 | F1@.9 | F1@Mean | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gas cylinder | 6 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | 0.978 |
| electric scooter | 2 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 |
| **macro** | | | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.992** | |

**判读**：
1. P/R 均为 1.000 → 两模型在 8 张 holdout 上无漏检、无误检；
2. 微调模型 gas F1@Mean 1.000→0.983、mIoU 0.983→0.978（轻微回退，2-6 框规模属噪声级）；scooter 保持 1.000；
3. 结论：**run_v2 重训无灾难性遗忘**，但"真实 F1"的判别力受限于 holdout GT 的半自洽来源（见 §5）。

---

## 5. 局限与后续

### 5.1 半自洽口径（必须正视）
- 当前 holdout GT = 教师模型零样本标注 + 类合并，**无独立人工视觉确认** → 预训练模型天然高分，F1≈1.0 是预期内的虚高。
- 本会话模型不支持图像输入，无法代为做像素级视觉复核；因此"人工修正"的视觉部分已沉淀为**工作表 + 可视化**，由用户（或具备视觉能力的 reviewer）完成。

### 5.2 人工复核路径（用户可一键重跑）
1. 打开 `outputs/annotation_detect/correction_worksheet.jsonl`（逐框决策）+ `outputs/annotation_detect/yolo_viz_v2/montage.jpg` / `previews/`；
2. 对错误/遗漏的框直接修改 `C:\Data\datasets\detect` 下对应 `*.txt`（或改工作表后重跑 `make_corrected_dataset.py`）；
3. 重跑：`bash scripts/eval/eval_feedback_loop.sh`（重训用 `train_lora.sh` 指向新 recipe）。

### 5.3 下一步：YOLO 作为独立 GT（打破自洽）
- 用 detect_v2（修正后）训练 YOLO → YOLO 的预测可作为**独立第二意见**，对 LocateAnything 的新一轮标注做交叉校验/主动采样，形成真正收敛的反馈闭环（详见报告 08/10）。

---

## 6. 产物清单

- 数据：`C:\Data\datasets\detect_v2\`（121 图 + 修正 txt + classes.txt + correction_summary.json）
- 脚本：`scripts/annotation/{make_corrected_dataset,yolo_txt_to_manifest}.py`、`scripts/eval/eval_feedback_loop.sh`
- 训练：`~/lora_gas/run_v2`（checkpoint-50/100/150）
- 评估：`outputs/annotation_detect/eval_v2_{pretrained,finetuned}.{json,md}`
- 审阅：`outputs/annotation_detect/correction_worksheet.jsonl`、`outputs/annotation_detect/yolo_viz_v2/`
- 报告：本文件（reports/07_real_annotation_feedback_loop.md）
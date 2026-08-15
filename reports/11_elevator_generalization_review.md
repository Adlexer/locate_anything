# 电梯场景泛化验证 + 人工视觉复核编排（elevator_yolo_detect）

- 日期：2026-08-16
- 分支：`feat/codex/elevator`（自 `feat/codex/yolo@c1ef0a9` 签出）
- 数据：`C:\Data\datasets\elevator_yolo_detect`（约 2.5 万原始图像 + 2408 份人工标注 txt）
- 目的：用真实采集的大型数据集验证检测管线的**跨域泛化**，并把**人工视觉复核**工具化、可视化

---

## 0. 结论（TL;DR）

1. **编排已设计并工具化**：分层采样 → 教师标注 → YOLO 批评者交叉校验 → 人工复核工件（工作表 + 交互式 HTML 复核器）→ 人工裁决回灌 → 真实泛化 F1。本轮已跑通前 4 步并产出全部复核工件。
2. **数据审计发现一个阻断项**：`已标注/` 下 2408 份人工 txt（命名 `LBD_B_2411_*`，全部 class 0）在整棵树中**找不到同名图片**——标注与图片已失配，无法直接作为逐图 GT。已在编排中标注为"待用户确认配对"（若能找回原图/映射文件，即可直接做人工 GT 评估）。
3. **教师（LocateAnything base）零样本**：150 图采样标注出 **163 框**（gas 60 / scooter 60 / bicycle 43），约 0.9s/图。
4. **YOLO 批评者（detect_v2 训练的 YOLO26s）**：121 框（scooter 73 / gas 48，**从不输出 bicycle**）。
5. **教师-YOLO 交叉校验一致率仅 9.8%**：163 教师框中 141 个"教师独有"、99 个"YOLO 独有"；73% 教师框与任何 YOLO 框 best-IoU < 0.2 ——**不是阈值问题，是位置与语义层面的系统性分歧**。这正是跨域泛化的真实证据：detect_v2 训练的 YOLO 在电梯场景上表现与教师 VLM 显著不同；谁对谁错**必须由人工裁决**。
6. **人工视觉复核任务已就绪**：`outputs/elevator_review/reviewer.html`（交互式，双击打开，快捷键裁决）+ 工作表 CSV/MD + 总览拼图。你复核 150 图后导出裁决 JSON，我即可回灌得到该批的**人工真值 GT**，进而算出教师/YOLO 的**真实跨域 F1**。

> 版本注意：WSL 下模型与训练产物已迁移到 `~/data/`（`~/data/models/LocateAnything-3B`、`~/data/lora_gas/run_v2`、`~/data/yolo_runs/run_v1`）；本报告所有路径已按新位置执行，旧路径 `~/models/`、`~/yolo_runs/`、`~/lora_gas/` 已失效。

---

## 1. 任务编排设计

```text
[原始采集] elevator_yolo_detect（~2.5万图，5 个场景组）
    │  scripts/elevator/sample_dataset.py（分层随机 + 降采样 ≤1280px，seed=42）
    ▼
[elevator_sample] 150 图（5 组 × 30）
    │  scripts/annotation/annotate_yolo.py（LocateAnything base 零样本，hybrid）
    ▼
[教师提案] 163 框（txt + manifest + previews）
    │  scripts/loop/cross_check.py（YOLO26s 批评者同图推理 + 匹配）
    ▼
[交叉校验] 一致率 9.8%（agree 16 / mismatch 6 / teacher_only 141 / yolo_only 99）
    │  scripts/elevator/make_review_artifacts.py
    ▼
[人工复核工件] reviewer.html + review_worksheet.{csv,md} + 拼图 montage
    │  用户裁决 → decisions.json
    ▼
[真实 GT + 泛化 F1]（待人工完成后）：教师/YOLO vs 人工 GT → 真实跨域 F1 → （可选）用裁决数据训练电梯场景 YOLO
```

本轮状态：前 4 步完成；"人工裁决回灌"等待用户执行复核任务（见 §5）。

## 2. 数据审计

| 目录 | 内容 | 数量 |
|---|---|---|
| 第二波数据采集汇总/{电瓶,电瓶车} | 图像 | 10694 |
| 电瓶 | 图像 | 377 |
| 电瓶车类似物 | 图像 | 11399 |
| 煤气罐 | 图像 | 2824 |
| 已标注/{电瓶_0 label,电瓶_1 label} | 人工 txt（`LBD_B_2411_*`，全部 class 0）+ labelimg 统计工具 exe | 2408 |

- **配对阻断项**：`LBD_B_2411_*`（2406 个）与任何图像 basename 均不匹配（图像为时间戳/负数命名）；全树无 LBD 前缀图片。→ 该批人工标注无法直接作为逐图 GT。建议：若原始标注图片或映射文件可找回（如用户机器上的 `LBD_B_2411_*.jpg` 或 labelimg 项目），提供后即可打通"人工 GT 真值"评估。
- 类别语义：`classes.txt` 为占位（"0"/"1"）；结合目录（煤气罐/电瓶/电瓶车）判断业务类别 ≈ detect_v2 的 gas cylinder / electric scooter（两轮车）。

## 3. 采样与教师标注（已完成）

- 采样：`sample_dataset.py`，5 组 × 30 = **150 图**，seed=42，降采样 ≤1280px → `C:\Data\datasets\elevator_sample\`（manifest.json 可复现）
- 教师：`annotate_yolo.py`（base 模型，hybrid，max_new_tokens=2048），150 图 135s
  - **163 框**：gas cylinder 60 / electric scooter 60 / **bicycle 43**
  - 43 个 bicycle 是已知教师子类混淆的集中体现（报告 06/07），也是复核重点

## 4. 交叉校验与泛化基线（已完成）

- 批评者：`cross_check.py` + YOLO26s（detect_v2 训练，`~/data/yolo_runs/run_v1`），imgsz=640，conf=0.25，匹配 IoU≥0.5 且类别一致
- 结果（150 图）：

| 指标 | 值 |
|---|---|
| 教师框 / YOLO 框 | 163 / 121（scooter 73 + gas 48，无 bicycle） |
| agree / class_mismatch / teacher_only / yolo_only | 16 / 6 / 141 / 99 |
| **教师-YOLO 一致率（IoU≥0.5+同类别）** | **9.8%** |
| 教师框 best-IoU 直方图 | <0.2：119（73%）；0.2-0.35：14；0.35-0.5：8；≥0.5：22 |
| 图像级分布 | 双方均有框 65 / 仅教师 56 / 仅 YOLO 13 / 均无 16 |
| 每类 mean best-IoU | gas 0.133 / scooter 0.172 / bicycle 0.160 |

- imgsz=1280 复测一致率反而降至 4.9% → 分歧与输入分辨率无关。
- **判读（Facts vs Inference）**：
  - Fact：两模型在电梯场景的位置/类别层面系统性不一致（73% 教师框无 YOLO 邻居）。
  - Inference（待人工验证）：detect_v2 训练的 YOLO26s 在电梯场景（更杂乱、目标更小/更远、光照复杂）泛化差，或教师 VLM 在此场景过检/误检——需要人工 GT 判定真伪。
  - 结论：**报告 09 的"8 张 holdout 泛化结论"确实不可靠；本数据集是正确的大规模验证样本，但必须依赖人工复核建立真值。**

## 5. 人工视觉复核任务（你的任务）

### 5.1 打开方式
- 双击打开：`C:\Dev\locate_anything\outputs\elevator_review\reviewer.html`（Chrome/Edge，无需联网）
- 总览拼图（教师框标注）：`C:\Dev\locate_anything\outputs\elevator_viz\montage.jpg`
- 工作表：`outputs\elevator_review\review_worksheet.csv` / `.md`

### 5.2 界面与快捷键
- 教师框=实线（绿=gas cylinder / 蓝=electric scooter / 红=bicycle，带 #编号 与交叉校验标签）；YOLO 框=橙色虚线
- `←`/`→`/空格：切图；`1` 保留 / `2` 类别错 / `3` 删除 / `4` 漏检（在图上拖拽标待补框）；`R` 重置本图；`D` 下载裁决 JSON
- 右侧面板可点击按钮逐框裁决，进度条实时显示已裁决数

### 5.3 重点提示
- **teacher_only（141）**：教师检出而 YOLO 未检出 → 重点判断是否为误检（尤其 bicycle 43 个，多为"电瓶车 vs 自行车"类别裁决）
- **yolo_only（99）**：YOLO 检出而教师未检出 → 重点判断是否为漏检（真目标被教师漏掉）
- class_mismatch（6）：两模型类别不一致 → 人工定类别

### 5.4 交付
- 完成后点"下载裁决 JSON"得到 `elevator_review_decisions.json`，放回 `outputs/elevator_review/`（或直接发我路径）
- 我会：① 回灌生成该批**人工真值 GT**；② 计算教师/YOLO 的**真实跨域 P/R/F1**；③ 更新报告并（可选）用裁决数据训练电梯场景 YOLO

## 6. 工具清单（scripts/elevator/）

| 脚本 | 作用 |
|---|---|
| `sample_dataset.py` + `run_sample.sh` | 分层随机采样 + 降采样 + manifest |
| `run_annotate_elevator.sh` | 教师零样本标注（base 模型） |
| `run_crosscheck_elevator.sh` | YOLO 批评者 + 交叉校验（复用 scripts/loop/cross_check.py） |
| `analyze_agreement.py` | best-IoU 直方图 + 每类一致率分析 |
| `make_review_artifacts.py` + `run_make_review.sh` | 生成工作表 CSV/MD + reviewer.html + manifest |
| `reviewer_template.html` | 交互式复核器模板（无依赖单文件） |

## 7. 下一步

1. **你执行人工复核**（150 图，预计 30-60 分钟）→ 回灌 → 真实跨域 F1；
2. 若可找回 `LBD_B_2411_*` 原图/映射 → 打通 2408 张人工 GT 的直接评估；
3. 用裁决后的真值训练电梯场景 YOLO（detect_v2 权重微调 or 从 COCO 重训），对比教师；
4. 把复核裁决并入 `annotation-yolo-loop` skill 的 `review` 命令（闭环自动化）。
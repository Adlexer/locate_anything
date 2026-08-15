# 两轮车交叉标注与审核训练闭环（方向调整后重启）

- 日期：2026-08-16
- 分支：`feat/codex/elevator`
- 定位：承接报告 11，按用户要求**调整方向**——仅保留两轮车（电动车/单车），排除煤气罐（教师对煤气罐 vs 电瓶区分弱 + 多类别命名混淆），重启交叉标注 + 审核训练闭环

---

## 0. 结论（TL;DR）

1. **方向调整**：类别域收敛为 2 类——`electric scooter`（电动车：ebike/scooter/电瓶/电瓶车）与 `bicycle`（单车：bicycle/电瓶车类似物）；煤气罐与"电瓶(电池)"混淆域整体排除。
2. **两轮车子集**：`elevator_sample_tw`（120 图 = battery/ebike_like/wave2_battery/wave2_ebike × 30，seed 42 采样，≤1280px）。
3. **教师 2 类零样本标注**：124 框，**electric scooter 64 / bicycle 60**（类别均衡）。
4. **YOLO 批评者交叉校验**（含 model.names 修复）：一致率 **5.6%**（agree 7 / mismatch 5 / teacher_only 112 / yolo_only 92）；**bicycle 全部 60 框未被 YOLO 确认**（YOLO 只输出 scooter）→ 单车类别判定是本次人工复核的重点。
5. **复核工件已生成**：`outputs/elevator_review_tw/`（reviewer.html 120 图 + 工作表 + 拼图；YOLO 的煤气罐框已过滤，只保留两轮车）。
6. **训练腿（基线）**：教师伪标签 2 类 YOLO 数据集 `yolo_detect_tw`（train 108 / val 12）→ YOLO26s 基线训练 `tw_run_v1`（进行中）；复核回灌后再重训对比。
7. **工具修复**：`cross_check.py`/`analyze_agreement.py` 改用 `model.names` 映射 YOLO 类别（避免 2 类数据集下 YOLO 的 gas 被错标成 scooter）；复核器框渲染 bug 修复沿用（归一化坐标 × 画布宽高）。

---

## 1. 方向调整（为什么）

- 用户观察：多类别存在明显命名混淆（数据集目录"电瓶/电瓶车/电瓶车类似物/煤气罐"与模型类别语义不一致）；教师对**煤气罐 vs 电瓶**区分能力弱。
- 决策：本阶段**只做两轮车**，聚焦 `电动车 vs 单车` 的判别（这也是报告 06/07 已知的 scooter/bicycle 子类混淆点，正好是闭环要攻克的难点）。

| 业务类别 | 模型类别（本阶段） | 覆盖数据集目录 |
|---|---|---|
| 电动车 | electric scooter | 电瓶、电瓶车、第二波/电瓶、第二波/电瓶车 |
| 单车 | bicycle | 电瓶车类似物 |

## 2. 数据与标注（已完成）

| 项 | 值 |
|---|---|
| 子集 | `C:\Data\datasets\elevator_sample_tw`：120 图（4 组 × 30，seed 42，≤1280px，自 elevator_sample 排除 gas） |
| 教师标注 | `annotate_yolo.py`（base，hybrid，max 2048）：120 图 102.6s |
| 教师框 | **124 框：electric scooter 64 / bicycle 60** |
| 输出 | `outputs/annotation_elevator_tw/`（txt/manifest/previews/summary） |

## 3. 交叉校验（已完成）

- 批评者：YOLO26s（detect_v2 训练，`~/data/yolo_runs/run_v1`）；匹配 IoU≥0.5 且类别一致
- 结果（120 图）：

| 指标 | 值 |
|---|---|
| 教师框 / YOLO 框 | 124 / 104（其中两轮车类 71：electric scooter 71，bicycle 0） |
| agree / class_mismatch / teacher_only / yolo_only | 7 / 5 / 112 / 92 |
| 一致率 | **5.6%** |
| 教师框 best-IoU 直方图 | <0.2：89（72%）；0.2-0.35：13；0.35-0.5：8；≥0.5：14 |
| 每类 agree@0.5 / mean best-IoU | scooter 7/64（0.16）；bicycle 0/60（0.138） |

- 判读：
  - **bicycle 60 框全部 teacher_only**——批评者无法佐证"单车"类，类别真伪完全依赖人工；
  - scooter 也仅 7/64 与 YOLO 位置一致（其余位置分歧），说明教师与 YOLO 在电梯场景的目标定位差异大；
  - 与报告 11（3 类）结论一致：跨域泛化下教师-YOLO 分歧大，人工复核是唯一真值来源。

## 4. 人工复核任务（你的任务，120 图）

- 启动：`scripts\elevator\start_review_server.bat` 指向 `outputs\elevator_review`（旧 150 图）；**两轮车版**手动起服务：
  ```bat
  python -m http.server 8766 --bind 127.0.0.1 --directory C:\Dev\locate_anything\outputs\elevator_review_tw
  ```
  访问 http://127.0.0.1:8766/reviewer.html
- 界面：教师框=实线（蓝=electric scooter / 红=bicycle），YOLO 框=橙色虚线（已过滤煤气罐）
- 快捷键：`←`/`→` 切图；`1` 保留 / `2` 类别错 / `3` 删除 / `4` 漏检（拖拽）；`R` 重置；`D` 下载裁决 JSON
- 重点：**teacher_only 112**（尤其 bicycle 60 个：判断是"电动车"还是"单车"）；yolo_only 中两轮车框看是否教师漏检
- 交付：`elevator_review_decisions.json` 放回 `outputs\elevator_review_tw\` → 回灌得到 2 类人工 GT → 重训对比

## 5. 训练腿（基线，已完成）

- 数据集：`yolo_detect_tw`（train 108 / val 12，2 类，教师伪标签）
- 训练：`run_train_tw.sh` → YOLO26s（COCO 预训练迁移），early-stop @ epoch 96（最优 epoch 46，patience 50），峰值显存 ~8GB，产物 `~/data/yolo_runs/tw_run_v1`（best.pt）
- 验证（12 张 val / 14 实例，教师伪标签为 GT）：

| class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.845 | 0.354 | **0.430** | **0.279** |
| electric scooter | 0.691 | 0.375 | 0.506 | 0.255 |
| bicycle | 0.999 | 0.333 | 0.354 | 0.304 |

- 判读：基线偏弱（mAP50 0.43，R 仅 0.35）符合预期——训练真值=未审核教师伪标签（与 YOLO 批评者一致率仅 5.6%）+ 仅 108 图。**这正是闭环的意义**：人工复核回灌后以同一脚本重训 `tw_run_v2`，对比 R/mAP 是否提升、修正率是否下降。

## 6. 工具与修复清单

| 项 | 说明 |
|---|---|
| `scripts/elevator/run_annotate_elevator_tw.sh` | 2 类教师标注 runner |
| `scripts/elevator/run_crosscheck_elevator_tw.sh` | 两轮车交叉校验 runner |
| `scripts/elevator/run_make_review_tw.sh` | 两轮车复核工件 runner（YOLO 类别过滤） |
| `scripts/elevator/build_tw_dataset.py` + `run_build_tw_dataset.sh` | 两轮车 YOLO 数据集构建 |
| `scripts/elevator/run_train_tw.sh` | 两轮车 YOLO 基线训练 |
| `scripts/loop/cross_check.py` / `analyze_agreement.py` | **修复**：YOLO 类别用 `model.names`（防 2 类下类别错位） |
| `make_review_artifacts.py` | 新增：YOLO 框按数据集类别过滤（剔除煤气罐） |

## 7. 下一步

1. **你复核 120 图** → 回灌 2 类人工 GT；
2. 用人工 GT 重训 `tw_run_v2`，与 `tw_run_v1`（伪标签基线）对比 → 闭环第一轮收敛指标（修正率/一致率）；
3. （可选）扩充两轮车采样（当前 120 图，原始两轮车图 ~2.2 万张）或增加难例（teacher_only 高分 bicycle）；
4. 复核裁决并入 `annotation-yolo-loop` skill 的 review 命令。
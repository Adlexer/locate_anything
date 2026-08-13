---
name: annotation-yolo-loop
description: 在 LocateAnything → YOLO 自动标注流水线上执行"标注→训练→导出"闭环迭代。当用户要求对数据集做新一轮标注、交叉校验、YOLO 重训、导出，或要求执行/查看标注闭环（annotate/review/train/export/iterate/status）时使用。
---

# annotation-yolo-loop

把 LocateAnything（教师标注）+ 人工修正 + YOLO（批评者/部署模型）+ 导出的闭环固化为可重复的 Agent 工作流（设计见 reports/10）。

## 环境

- 教师：WSL conda `locate_anything_sft`，权重 `~/models/LocateAnything-3B`，LoRA 产物 `~/lora_gas/run_v{n}`
- YOLO：WSL conda `yolo`（ultralytics>=8.4），训练产物 `~/yolo_runs/run_v{n}`
- 数据：`C:\Data\datasets\detect`（原始）、`detect_v2`（修正版）、`yolo_detect`（YOLO 格式）；每轮新版本 `detect_v{n+1}`

## 命令

| 命令 | 动作 |
|---|---|
| `init` | 检查环境/权重/数据存在性；确认 reports/INDEX.md 与双 report 当前状态 |
| `annotate` | `scripts/annotation/annotate_yolo.py` 对未标注图/视频生成 YOLO txt + manifest |
| `review` | 生成工作表（correction_worksheet.jsonl）+ 可视化（montage）；跑 `scripts/loop/cross_check.py` 输出教师-YOLO 分歧清单 |
| `apply` | `scripts/annotation/make_corrected_dataset.py` 应用裁决 → 新版本 detect_v{n+1} |
| `train` | 重建数据（yolo_txt_to_manifest + build_train_jsonl + build_yolo_dataset）→ 教师 LoRA（train_lora.sh，可选）+ YOLO（train_yolo.sh）→ 评估（eval_feedback_loop.sh） |
| `export` | `scripts/yolo/export_yolo.py` 导出 ONNX/TRT FP16/INT8 + 精度回归 |
| `iterate` | 执行一轮完整闭环：annotate→review→apply→train→export，并统计收敛指标 |
| `status` | 读取双 report + INDEX，汇报当前版本/指标/下一步 |

## 收敛判定

每轮统计（写入 progress_report Timeline）：
- 人工修正率 = 被改框数 / 教师提案框数
- 教师-YOLO 一致率 = 一致框数 / 教师提案框数
- 稳定判据：修正率 < 5% 且一致率 > 95%，连续 2 轮（建议按累计 ≥500 框窗口计算）

## 维护规则（对齐 report skill）

- 每轮迭代在 `progress_report.md` 追加 Timeline 与指标；阶段成果追加 `result_report.md` Milestones；
- 新报告在 `reports/INDEX.md` 登记；代码/脚本在分支 `feat/codex/yolo`（或 `feat/codex/yolo-v{n}`）维护；
- 数据版本变更（类名/坐标体系）必须在 data.yaml / classes.txt 中锁定并记录；
- 禁止凭记忆臆测数据集状态，一律以磁盘 + 报告为准。
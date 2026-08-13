# 标注→训练→导出自动化闭环工作流（含 agent skill 化方案）

- 日期：2026-08-13
- 分支：`feat/codex/yolo`
- 定位：设计文档 + 落地路线；将 LocateAnything 自动标注、人工修正、YOLO 训推、导出串成可持续迭代的闭环，
  人工标注在其中的角色类比 RL 的"奖励/监督信号"，持续迭代标注质量直到稳定。

---

## 0. 结论（TL;DR）

1. 闭环 = **教师提案（LocateAnything）→ 批评者交叉校验（YOLO + 人工）→ 数据版本化 → 重训（教师 LoRA + YOLO）→ 导出**，人工只裁决"分歧样本"（主动学习式），随迭代人工工作量递减、标注质量收敛。
2. 本仓库已具备闭环的全部积木：annotate_yolo.py（教师）→ make_corrected_dataset.py（修正+工作表）→ yolo_txt_to_manifest/build_train_jsonl（数据重建）→ train_lora.sh / train_yolo.sh（双模型重训）→ eval_feedback_loop.sh / infer_yolo.py / export_yolo.py（评估/推理/导出）。缺的只是"编排器"与"收敛判定"。
3. **收敛准则（可量化）**：每轮迭代统计 (a) 人工修正率（被改框数/教师提案框数）与 (b) 教师-批评者（YOLO）一致率；当修正率 < 5% 且一致率 > 95% 连续 2 轮 → 判定稳定。当前基线（报告 07）：一致率因 GT 来源同教师而虚高，需人工复核首轮数据后才有真实基线。
4. **agent skill 化**：建议将编排逻辑固化为 Codex skill `annotation-yolo-loop`（仓库内 `skills/annotation-yolo-loop/SKILL.md`，可复制到 `~/.codex/skills`），提供 `init/annotate/review/train/export/iterate/status` 命令与双 report/INDEX 维护规则，使 Agent 可一键执行闭环并持续维护状态。

---

## 1. 目标与动机

- 业务目标：把"摄像头/无人机/现场照片 → YOLO 检测模型"的生产链路自动化，标注质量持续提升直至稳定。
- 动机（为什么需要闭环）：
  - 纯零样本（LocateAnything）标注质量受教师能力上限约束，且子类（scooter/bicycle）存在系统性混淆（报告 06/07）；
  - 纯人工标注成本高、不可扩展；
  - 单模型自洽评估会虚高（报告 07 的 F1≈1.0 教训）——需要**独立第二意见**打破自洽。

## 2. 已具备的组件（现状盘点）

| 环节 | 工具 | 状态 |
|---|---|---|
| 教师标注（零样本/微调） | `scripts/annotation/annotate_yolo.py` + `~/lora_gas/run_v2` | ✅ 可用 |
| 修正策略 + 人工工作表 | `scripts/annotation/make_corrected_dataset.py` | ✅ 可用（报告 07） |
| 数据重建（txt→manifest→JSONL/recipe） | `yolo_txt_to_manifest.py` + `build_train_jsonl.py` | ✅ 可用 |
| 教师重训（LoRA） | `scripts/train/train_lora.sh` | ✅ 可用（run_v2） |
| YOLO 重训 | `scripts/yolo/train_yolo.sh` | ✅ 可用（run_v1） |
| 双模型评估 | `scripts/eval/eval_feedback_loop.sh` | ✅ 可用 |
| YOLO 推理/导出 | `scripts/yolo/{infer_yolo,export_yolo}.py` | ✅ 可用 |
| **编排器 + 收敛判定** | （本报告设计，P0 待实现） | ⏳ 缺 |

## 3. 闭环架构

```text
                    ┌─────────────────────────────── 数据版本化（detect_v{n} + 报告/索引） ───────────────────────────────┐
                    │                                                                                                    │
   [新图/视频]      ▼            [教师提案]            [批评者交叉校验]                [人工裁决]              [重训]      [导出]
  inputs/ ──► annotate_yolo.py ──► detect/ (txt) ──► YOLO 同图推理 + 对比 ──► 一致→自动接受；分歧→人工(工作表) ──► 教师LoRA + YOLO ──► ONNX/TRT/INT8
                     ▲                                    │                     │                                      │            │
                     └────────────────────────────────────┴─────────────────────┴──────────────────────────────────────┴────────────┘
                                             每轮迭代统计：修正率↓、一致率↑ → 收敛判定（阈值见 §4）
```

组件职责（RL 类比）：
| 组件 | 类比 | 职责 |
|---|---|---|
| LocateAnything 教师 | Policy（提案生成） | 对未标注图生成检测提案（YOLO txt） |
| YOLO 批评者 | Critic / 第二意见 | 独立预测同一批图；与教师提案做 IoU+类别匹配，输出分歧清单 |
| 人工 | Reward / 真值 | 只裁决分歧样本；修正即"奖励信号" |
| 数据版本 + 报告 | Replay buffer / 训练分布 | 每个版本可追溯、可回滚；双 report + INDEX 记录每轮指标 |
| 重训（LoRA + YOLO） | Policy update | 用"已裁决真值"增量更新教师与 YOLO |
| 导出 | 部署 | 按目标设备导出并做量化精度回归 |

## 4. 迭代与收敛准则（可量化）

每轮迭代（对一批新图）：
1. 教师提案框数 N；
2. YOLO 批评者匹配：与教师提案按 IoU≥0.5 + 类别一致判"一致"；否则入分歧集；
3. 人工裁决分歧集（修正/删除/新增），记录**修正率 = 被人工改动的框数 / N**；
4. 重训 YOLO（增量，从上一版 best.pt 继续），可选重训教师 LoRA；
5. 记录指标到 reports + progress_report Timeline。

收敛判定：**修正率 < 5% 且 教师-YOLO 一致率 > 95%，连续 2 轮** → 标注质量稳定，进入纯增量维护。
（阈值可按业务调整；小数据集首轮需先人工复核建立真实基线，见报告 07 §5。）

## 5. agent skill 化方案

建议将编排逻辑固化为 Codex skill：`annotation-yolo-loop`（仓库内 `skills/annotation-yolo-loop/SKILL.md`，复制到 `~/.codex/skills` 即安装）。

Skill 提供的命令（示意）：
| 命令 | 动作 |
|---|---|
| `init` | 检查环境（locate_anything_sft / yolo）、权重、数据目录；初始化 report 与 INDEX |
| `annotate` | 对新图/视频跑教师标注 → detect/ txt + manifest |
| `review` | 生成工作表 + montage；输出分歧清单（YOLO 交叉校验）；等待人工裁决 |
| `apply` | 应用人工裁决（make_corrected_dataset）→ 新版本 detect_v{n} |
| `train` | 重建数据 → 教师 LoRA（可选）+ YOLO 增量训练 → 评估 |
| `export` | 导出 ONNX/TRT（FP16/INT8）+ 精度回归 |
| `iterate` | 执行一轮完整闭环并统计收敛指标 |
| `status` | 读取双 report + INDEX，汇报当前版本/指标/下一步 |

Skill 维护规则（对齐 report skill）：
- 每轮迭代在 progress_report.md 追加 Timeline 与指标；阶段成果（如某版本数据集验收、YOLO mAP）追加到 result_report.md Milestones；
- 新报告在 reports/INDEX.md 登记；
- 分支约定：迭代代码/脚本进 `feat/codex/yolo`（或按版本开 `feat/codex/yolo-v{n}`）。

## 6. 落地路线

| 优先级 | 任务 | 说明 |
|---|---|---|
| P0 | 编排脚本 `scripts/loop/run_loop.sh`（annotate→cross-check→worksheet→apply→train→report） | 先人工可执行闭环（半自动） |
| P0 | 交叉校验脚本 `scripts/loop/cross_check.py`（YOLO vs 教师提案 → 分歧清单） | 打破自洽的关键件 |
| P1 | 人工复核首轮 holdout（用户用工作表/可视化）→ 建立真实基线指标 | 依赖人工，报告 07 已备好工件 |
| P1 | 收敛判定逻辑（修正率/一致率）并入 run_loop + 报告 | 自动化收敛 |
| P2 | skill `annotation-yolo-loop` 完整化并安装到 `~/.codex/skills` | agent 一键闭环 |
| P2 | 目标设备导出流水线（Jetson TRT / RKNN / Hailo）+ 量化回归 | 场景 B 落地 |

## 7. 风险与权衡

1. **批评者偏差**：YOLO 与教师同源于 detect_v2 修正数据，初期批评者"独立性"有限；随着人工裁决数据积累，YOLO 逐步获得独立于教师的真值信号，独立性增强。权衡：早期需更多人工，后期自动化收益兑现。
2. **收敛指标被小样本噪声干扰**：每轮样本量小（几十框）时 5%/95% 阈值抖动大；建议按"累计 500+ 框窗口"计算指标。
3. **数据版本增长**：每轮新增版本要避免标注漂移（类名/坐标体系变更需在 data.yaml/classes.txt 中锁定并记录）。
4. **成本**：教师（VLM）标注慢于 YOLO，大图/视频抽帧成本高；可按需先 YOLO 预筛（粗检）再教师精修（精检），或反之。
5. **模型双轨维护**：教师 LoRA 与 YOLO 需同步版本；建议以 YOLO 为"部署真相"，教师为"标注引擎"，二者解耦升级。

---

## 8. 产物

- 设计文档：本文件（reports/10_closed_loop_pipeline.md）
- Skill 草案：`skills/annotation-yolo-loop/SKILL.md`（仓库内，可复制安装）
- 编排脚本：P0 待实现（见 §6）
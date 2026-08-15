# Reports Index

项目：LocateAnything → YOLO 自动标注流水线 + YOLO 训推闭环
规则：每份报告落盘时同步更新本索引；一句话结论只写关键产出/结论；状态取值 done / in-progress / pending

## 索引

| # | 文件 | 日期 | 主题 | 一句话结论 | 状态 |
|---|------|------|------|-----------|------|
| 01 | [01_initial_exploration_report.md](./01_initial_exploration_report.md) | 2026-08-04 | 源码/权重/环境初步探索 | LocateAnything = NVIDIA 通用视觉定位 VLM（PBD 并行框解码，~3.83B）；本地权重/环境跑通 | done |
| 02 | [02_todo_and_suggestions.md](./02_todo_and_suggestions.md) | 2026-08-04 | 待办与建议（预研收尾） | 列出环境/工程/推理/评估等后续待办与建议 | done |
| 03 | [03_environment_fa_laflash.md](./03_environment_fa_laflash.md) | 2026-08-04 | 环境巩固：权重本地化 + FA + la_flash + env 导出 | 权重加载 2.9s；Vision=FA2/LM=sdpa；la_flash batch≥4 省显存；environment.yml 导出 | done |
| 04 | [04_generation_mode_benchmark.md](./04_generation_mode_benchmark.md) | 2026-08-04 | 解码模式 × max_new_tokens 基准 | fast/hybrid/slow 速度-精度权衡 + 跨模式一致性；默认 hybrid/8192 | done |
| 05 | [05_lora_sft_research.md](./05_lora_sft_research.md) | 2026-08-04 | LoRA/SFT 预研 | 16GB 单卡可行方案 = LoRA r64 + sdpa + seq≤2048 + DS ZeRO-1/2；全参 SFT 不可行 | done |
| 06 | [06_zeroshot_annotation_and_lora_validation.md](./06_zeroshot_annotation_and_lora_validation.md) | 2026-08-05 | 零样本标注 + 小批量 LoRA 验证 | 122 txt / 93 框（gas 58/scooter 28/bike 7）；微调 holdout 8/8 一致、IoU 0.987（自洽口径，偏乐观） | done |
| 07 | [07_real_annotation_feedback_loop.md](./07_real_annotation_feedback_loop.md) | 2026-08-13 | 真实标注反馈闭环 | 闭环管线端到端跑通：修正策略（bicycle→scooter 合并+降采样）→ detect_v2 → run_v2 重训 → 双模型 eval（macro F1 1.000/0.992，半自洽口径已说明）；工作表+可视化供人工复核 | done |
| 08 | [08_yolo_selection_research.md](./08_yolo_selection_research.md) | 2026-08-13 | YOLO 选型调研（训练工作站 / 嵌入式小算力推理） | 选型结论：Ultralytics + YOLO26（n/s 边缘 / s/m 训练），双场景矩阵、TRT INT8 风险与落地计划 | done |
| 09 | [09_yolo_train_infer_framework.md](./09_yolo_train_infer_framework.md) | 2026-08-13 | YOLO 训推框架（WSL yolo env + 训练/导出/推理） | yolo env + yolo_detect 数据集；YOLO26s 微调 val mAP50/50-95=0.995；TRT FP16 4.1ms/帧；框架脚本交付 | done |
| 10 | [10_closed_loop_pipeline.md](./10_closed_loop_pipeline.md) | 2026-08-13 | 标注→训练→导出闭环工作流（含 agent skill 化） | 闭环设计落盘：教师-批评者-人工裁决-重训-导出；收敛准则（修正率<5%/一致率>95%）；skill 草案 + cross_check.py 已实现 | done |

## 维护规则

- 编号按时间顺序递增；文件名 `NN_<slug>.md`
- 每份报告落盘时在此新增/更新一行，并同步更新双 report（progress_report.md / result_report.md）
- 一句话结论保持一行，突出可验证的产出或结论
| 11 | [11_elevator_generalization_review.md](./11_elevator_generalization_review.md) | 2026-08-16 | 电梯场景泛化验证 + 人工视觉复核编排 | 编排工具化：采样150图→教师标注163框→YOLO交叉校验（一致率9.8%）→复核工件（reviewer.html/工作表/拼图）；审计发现LBD人工标注与图片失配 | done |

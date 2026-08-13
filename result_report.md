# Result Report

## Status

Not Finished（巩固阶段已完成，等待后续任务继续）

---

## Milestones

- [2026-08-04] WSL 环境就绪：Miniconda 26.5.3 + locate_anything conda 环境（python 3.10 / torch 2.9.0+cu130 / transformers 4.57.1 / deepspeed 0.15.4 / liger_kernel 0.3.1），全套 pyproject 依赖安装完成，系统 python 未污染
- [2026-08-04] 端到端冒烟验证通过：本地权重 C:\Data\LocateAnything-3B 在 WSL 内可加载（Magi→sdpa 回退），合成图检测推理输出格式正确
- [2026-08-04] 真实图片推理通过：bus.jpg（810×1080）detect 4 person + 1 bus + 1 car（2.0 BPS）、ground_multi 3 区域、point 命中车体；产物 outputs/smoke_real_*.{jpg,json}
- [2026-08-04] 预研报告落盘：reports/01_initial_exploration_report.md（源码/方法/环境/冒烟全景）、reports/02_todo_and_suggestions.md（待办与建议）
- [2026-08-04] 推理基线升级（巩固阶段）：权重本地化到 WSL ext4（加载 31-42s→2.9s）；使能 FlashAttention（Vision=flash_attention_2，LLM=sdpa），detect 提速至 1.45s / 4.1 BPS（约 2×）；la_flash 批处理验证通过（batch4 与 sdpa 速度相当并省 ~2.1GB 显存）；environment.yml 复现基线导出
- [2026-08-04] 生成模式对比落盘：reports/04_generation_mode_benchmark.md（fast/hybrid/slow × max_new_tokens 512/2048/8192，速度-精度 + 跨模式一致性；推荐 hybrid + 8192）
- [2026-08-04] 可复用推理 CLI 交付：scripts/infer.py（detect/ground/ground_multi/ground_text/detect_text/ground_gui/point → 像素坐标 boxes/points JSON + 标注图）
- [2026-08-04] 项目纳入 git 管理：git init（main 分支）+ 初始提交 74e3766（540 文件）；根 .gitignore（outputs/、缓存、Eagle LFS 占位二进制）+ .gitattributes（LF 归一化、二进制标记），工作区干净
- [2026-08-04] 推送 GitHub 远端 Adlexer/locate_anything 成功：合并远端初始提交（LICENSE/README.md），filter-branch 清理历史中 git-lfs 占位后 force push，main=200b6c4、541 文件、工作区干净

---

- [2026-08-04] LoRA/SFT 预研收尾：conda locate_anything_sft（clone 自 locate_anything）就绪；官方 LoRA 微调管线在 5060 Ti 16GB 端到端验证通过（LoRA r=64 + sdpa + seq=2048 + grad ckpt，13.05s/it，峰值显存 16,136MB，trainable 119,734,272），微调 checkpoint 可被 scripts/infer.py 直接加载并正确推理（5 框）；定位 DeepSpeed 0.15.4 Blackwell sm_120 JIT 编译 bug（compute_1.）并给出 torch_adam 绕行方案；报告 reports/05_lora_sft_research.md 落盘
- [2026-08-04] scripts 目录整理 + LoRA 训练启动/监控脚本交付：scripts/ 重组为 infer/bench/train/env/smoke 子目录（删除 .b64 残留、verify_*.py→.sh），新增 scripts/train/{train_lora.sh,train_watch.sh,ds_z1_torchadam.json,README.md} 与 scripts/README.md；train_lora.sh 端到端验证（1 步训练成功、断点续训检测、done.txt 提示），train_watch.sh 提供 tail/loss/gpu/tensorboard/checkpoints 监控
- [2026-08-05] 真实小数据集验证里程碑：零样本标注管线（scripts/annotation/ 三脚本 + README）交付，C:\Data\datasets\detect 产出 122 份 YOLO txt（93 框，0 错误）；LoRA 小批量训练验证通过（78 伪标签样本，150 步 loss 1.19→0.41，holdout 8/8 类别一致 IoU 0.987，无灾难性遗忘）；定位 16GB 显存约束导致的 3 个问题（高帧样本丢弃 / model_max_length 烤死 / fast-hybrid 大图回归）并改进 launcher（MAX_SEQ 默认 1536、三重警告、model_max_length 自动恢复、--warmup）；报告 reports/06_zeroshot_annotation_and_lora_validation.md 落盘
- [2026-08-05] 标准化检测评估交付：scripts/eval/eval_det.py（模型 vs YOLO GT，P/R/F1@IoU + F1@Mean + matched-IoU + 逐图明细，JSON+MD 报告），holdout 双模型评估跑通（F1@Mean=1.000 自洽口径，微调无回归）
- [2026-08-13] 新专项基础设施就绪：black 26.5.1 规范化 scripts/ 全部 13 个 py（Eagle/ 上游不动）+ root pyproject.toml（line-length 88），提交 e6aab29 推送 feat/codex/lora；自其签出并推送新分支 feat/codex/yolo；建立报告索引机制 reports/INDEX.md（01-06 登记 + 07-10 占位），双 report 同步更新

- [2026-08-13] 真实标注反馈闭环里程碑：闭环管线端到端跑通（修正策略 make_corrected_dataset.py：bicycle→scooter 合并 + ≤1280px 降采样 + 框质量过滤 → detect_v2；yolo_txt_to_manifest.py 逆向转换；eval_feedback_loop.sh 双模型评估）；LoRA run_v2 重训完成（150 步 / 4:31 / 1.62s/it，seq=1792 全覆盖无样本丢弃）；修正后 GT 上双模型 eval：预训练 macro F1@Mean 1.000、微调 0.992（gas 0.983），无灾难性遗忘；报告 07 落盘，半自洽口径与人工复核路径已说明

- [2026-08-13] YOLO 训推框架里程碑：WSL `yolo` conda 环境（ultralytics 8.4.118）+ `yolo_detect` 数据集（train/val 与 LoRA holdout 对齐）；YOLO26s 微调 run_v1 完成（200ep ≈ 6.4min，峰值显存 ~5.6GB，val mAP50=0.995 / mAP50-95=0.995，gas P0.991/R1.0、scooter P0.912/R1.0）；本地推理验证 val 8/8 正确；导出 ONNX（14s）+ TensorRT FP16 engine（~300s，4.1ms/帧@640 ≈ 4.2× 加速）；框架脚本 scripts/yolo/ 交付
- [2026-08-13] 闭环工作流设计里程碑：报告 08（YOLO 选型：Ultralytics + YOLO26 双场景矩阵）与报告 10（标注→训练→导出闭环 + 收敛准则 + skill 化方案）落盘；`scripts/loop/cross_check.py` 实现教师-YOLO 交叉校验（detect_v2 8 图一致率 1.000，YOLO 额外发现 1 框分歧样本）；skill 草案 skills/annotation-yolo-loop/SKILL.md 入库

## Summary

- 预研任务（探查 → 环境 → 真实图冒烟 → 报告）与巩固任务（权重本地化 + FA + la_flash + 环境导出 + 生成模式对比 + infer.py CLI）全部完成并交付；LocateAnything 在 WSL + RTX 5060 Ti 上形成「加载 2.9s、detect 1.45s、4.1 BPS」的最终推理基线

---

- 新一轮 LoRA/SFT 预研完成：从零为基础设施工程师普及 SFT/LoRA 原理、官方管线解读、显存账与 16GB 单卡可行方案，并用端到端 smoke 训练+推理验证可行性。
## Deliverables

- 报告：reports/01_initial_exploration_report.md、reports/02_todo_and_suggestions.md、reports/03_environment_fa_laflash.md、reports/04_generation_mode_benchmark.md
- 推理 CLI：scripts/infer.py（--image + --task/--query → boxes/points + 标注图）
- 基准脚本：scripts/bench_gen_mode.py、scripts/gen_mode_report.py、scripts/bench_la_flash.py、scripts/make_bench_assets.py
- 环境基线：environment.yml（含 flash-attn 2.8.3+cu.13.0.torch.2.9、cuda-toolkit 13.0.3 等）
- 推理产物：outputs/demo_detect_annotated.jpg、demo_ground_annotated.jpg、demo_point_annotated.jpg、demo_*.json、bench/gen_mode_results.json、bench/la_flash_results.json
- 权重：~/models/LocateAnything-3B（WSL ext4 本地副本，在仓库外不入库）；数据：data/bus.jpg、data/dense_text.png（+dense_text_gt.json 38 框 GT）
- git 仓库：C:\\Dev\\locate_anything（main 分支，origin=https://github.com/Adlexer/locate_anything，HEAD=200b6c4；outputs/ 与 Eagle LFS 占位不入库）
- WSL 环境：/home/xu/miniconda3 + locate_anything 环境（conda activate locate_anything）；~/.wslconfig memory=48GB

---

- 报告：reports/05_lora_sft_research.md（SFT/LoRA 原理 + 官方管线 + 显存估算与实测 + 本机方案 + 踩坑清单）
- 环境：WSL conda locate_anything_sft（clone 自 locate_anything，已补装 nvidia-ml-py/sortedcontainers/tensorboard）
- 冒烟产物：WSL ~/lora_smoke/（recipe.json / data.jsonl / ds_z1_torchadam.json / run.log / work_dirs 全量 checkpoint + 推理结果）
- 训练脚本：scripts/train/train_lora.sh（启动/续训/覆盖保护）、train_watch.sh（监控）、ds_z1_torchadam.json（绕 sm_120 bug）、train/README.md
- 目录索引：scripts/README.md
- 标注代码：scripts/annotation/{annotate_yolo.py, build_train_jsonl.py, eval_before_after.py, README.md}
- 数据集标注：C:\Data\datasets\detect 下 122 份 YOLO txt + classes.txt + _frames/（视频抽帧）
- 训练产物：WSL ~/lora_gas/run_v1（LoRA 微调，checkpoint-50/100/150 + 最终模型）
- 报告：reports/06_zeroshot_annotation_and_lora_validation.md
- 评测数据：outputs/annotation_detect/{summary.json, manifest.jsonl, eval_before/after*.json, previews/, probe/}
- 评估：scripts/eval/{eval_det.py, README.md}；产出 outputs/annotation_detect/eval_det_{pretrained,finetuned}.{json,md}
## Remaining Issues

- [已解决] /mnt/c 加载权重慢 → 已拷至 ~/models/LocateAnything-3B（2.9s）
- [已解决] flash-attn 未装 → 已装 2.8.3（Vision=FA2）；注意必须用 Astral cu130 预编译 wheel，源码编译会触发 WSL 全局 OOM / E_UNEXPECTED
- [已解决] LLM 注意力：magi 回退到 flash_attention_2 会 NotImplementedError → 已 patch 为回退 sdpa（modeling_qwen2.py 两处副本 + HF cache）
- la_flash：batch=1 + pipeline 调度器病态慢（170s），单行请用 eager 调度器；该后端仅在 batch≥4/长上下文时体现省显存价值
- 本机仅 sdpa 短上下文（~4K）；MagiAttention/长上下文需 Hopper/Blackwell
- 官方权重不支持 visual prompt 推理（需自行微调）
- NVIDIA 非商用 License，商用需注意合规

---

- [新增] 16GB 单卡 LoRA 训练 seq 仅支持 <=2048（峰值已顶格），8K/16K 长上下文或全参 SFT 需多卡/服务器 GPU（官方 8xH100）
- [新增] DeepSpeed 0.15.4 在 Blackwell sm_120 上 JIT FusedAdam 编译必失败（compute_1.），须 optimizer 加 torch_adam:true 或给 deepspeed 打补丁
- [新增] 训练数据 JSONL/recipe 需无 BOM UTF-8；annotation/root 建议绝对路径
- SFT 下一步：收集/标注业务数据（JSONL+recipe，先 200~1000 条）→ 小步验证（50~200 steps）→ 正式 LoRA 长跑（后台）→ evaluation 对比 → 服务化。
- [新增] 16GB 单卡 + seq=2048 训练已顶格，高分辨率帧（>2048 token）会被训练管线丢弃；需降采样或更大显存
- [新增] 微调产物 fast/hybrid 解码在 1080p 大图上可能截断/乱码，大图部署优先 slow 或把大图纳入训练
- [新增] 伪标签子类噪声（scooter/bicycle 互斥误判），类别纯度要求高时需合并或人工修正
- 后续：a) 降采样 1080p 帧重训一版对比电动车检出；b) 人工抽查 30-50 条伪标签后二轮训练；c) 真实业务评测集量化 F1。
## Suggestions

- **真实标注反馈闭环**：visualize_yolo.py 抽查修正伪标签（30-50 条，重点 scooter/bicycle）→ build_train_jsonl 重建 → LoRA 重训 → eval_det.py 出真实 F1（当前 1.000 为自洽口径，人工 GT 后才有意义）。
- **高分辨率帧降采样**（≤1280px）再入训，否则 seq=2048 下被丢弃（本次 1080p 帧即被丢）。
- **类别策略**：业务只要两轮车则合并 bicycle→electric scooter（--merge-two-wheeler）。
- **部署**：大图优先 slow；hybrid 待大图入训后再启用。
- **评估统一口径**：scripts/eval/eval_det.py（P/R/F1@IoU + F1@Mean，与论文口径一致），避免临时脚本各测各的。

### 待办

- [x] 推送 feat/codex/lora 至 origin（2026-08-06 用户手动推送，HEAD=aa30afa）
- [ ] 人工标注修正闭环
- [ ] run_v2（降采样）重训 + 真实 F1 对比
- [ ] fast/hybrid 大图回归复测
- [ ] 服务化/批量推理

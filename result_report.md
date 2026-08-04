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

---

## Summary

- 预研任务（探查 → 环境 → 真实图冒烟 → 报告）与巩固任务（权重本地化 + FA + la_flash + 环境导出 + 生成模式对比 + infer.py CLI）全部完成并交付；LocateAnything 在 WSL + RTX 5060 Ti 上形成「加载 2.9s、detect 1.45s、4.1 BPS」的最终推理基线

---

## Deliverables

- 报告：reports/01_initial_exploration_report.md、reports/02_todo_and_suggestions.md、reports/03_environment_fa_laflash.md、reports/04_generation_mode_benchmark.md
- 推理 CLI：scripts/infer.py（--image + --task/--query → boxes/points + 标注图）
- 基准脚本：scripts/bench_gen_mode.py、scripts/gen_mode_report.py、scripts/bench_la_flash.py、scripts/make_bench_assets.py
- 环境基线：environment.yml（含 flash-attn 2.8.3+cu.13.0.torch.2.9、cuda-toolkit 13.0.3 等）
- 推理产物：outputs/demo_detect_annotated.jpg、demo_ground_annotated.jpg、demo_point_annotated.jpg、demo_*.json、bench/gen_mode_results.json、bench/la_flash_results.json
- 权重：~/models/LocateAnything-3B（WSL ext4 本地副本，在仓库外不入库）；数据：data/bus.jpg、data/dense_text.png（+dense_text_gt.json 38 框 GT）
- git 仓库：C:\\Dev\\locate_anything（main 分支，初始提交 74e3766；outputs/ 与 Eagle LFS 占位不入库）
- WSL 环境：/home/xu/miniconda3 + locate_anything 环境（conda activate locate_anything）；~/.wslconfig memory=48GB

---

## Remaining Issues

- [已解决] /mnt/c 加载权重慢 → 已拷至 ~/models/LocateAnything-3B（2.9s）
- [已解决] flash-attn 未装 → 已装 2.8.3（Vision=FA2）；注意必须用 Astral cu130 预编译 wheel，源码编译会触发 WSL 全局 OOM / E_UNEXPECTED
- [已解决] LLM 注意力：magi 回退到 flash_attention_2 会 NotImplementedError → 已 patch 为回退 sdpa（modeling_qwen2.py 两处副本 + HF cache）
- la_flash：batch=1 + pipeline 调度器病态慢（170s），单行请用 eager 调度器；该后端仅在 batch≥4/长上下文时体现省显存价值
- 本机仅 sdpa 短上下文（~4K）；MagiAttention/长上下文需 Hopper/Blackwell
- 官方权重不支持 visual prompt 推理（需自行微调）
- NVIDIA 非商用 License，商用需注意合规

---

## Suggestions

- 下一步建议从「真实业务图批量验证 + 服务封装（FastAPI/locateanything_worker + infer.py）」或「LoRA 微调」或「评估复现（Rex-Omni-EvalData/ScreenSpot-Pro）」三选一推进；详细待办见 reports/02_todo_and_suggestions.md

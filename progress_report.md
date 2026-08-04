# Progress Report

## Task

LocateAnything LoRA / SFT 预研阶段：在 feat/codex/lora 分支上为不熟悉训练的基础设施工程师普及 SFT/LoRA 原理与技术方案；WSL conda 新建 locate_anything_sft（clone 自 locate_anything），端到端验证官方 LoRA 微调管线在本机（5060 Ti 16GB）可行性，并输出预研报告。

---

## Current Status

Completed（LoRA/SFT 预研阶段收尾：环境 clone + 端到端 smoke 验证 + 预研报告落盘；等待用户提供真实业务数据进入正式训练）

---

## Completed

- 源码探查：Embodied 目录、PBD/MTP 方法、特殊 token、推理/训练/评估管线（详见 reports/01_initial_exploration_report.md）
- WSL 环境：Miniconda 26.5.3 + locate_anything 环境（py3.10 / torch 2.9.0+cu130 / transformers 4.57.1 / deepspeed 0.15.4 / liger_kernel 0.3.1 等），系统 python 未污染
- 真实图片冒烟：ultralytics bus.jpg → detect 4 person + 1 bus + 1 car（6 boxes, 2.0 BPS）、ground_multi people 3 区域、point bus (499,437)；Magi→sdpa 回退正常
- 报告落盘：reports/01_initial_exploration_report.md、reports/02_todo_and_suggestions.md
- 产物：outputs/smoke_real_annotated.jpg、smoke_real_input.jpg、smoke_real_results.json；scripts/smoke_test.py、smoke_test_real.py、render_annotated.py
- [巩固] 权重拷贝：/mnt/c/Data/LocateAnything-3B → ~/models/LocateAnything-3B（48 文件、safetensors 字节一致），加载 31-42s → 2.9s
- [巩固] 使能 FA：安装 flash-attn 2.8.3+cu.13.0.torch.2.9（Astral cu130 预编译 wheel，规避源码编译的 WSL OOM/E_UNEXPECTED）；Vision=flash_attention_2、LLM=sdpa（patch modeling_qwen2.py magi→sdpa）；detect 1.45s / 4.1 BPS（约 2× 提速）
- [巩固] la_flash 验证：batch_infer.py --attn la_flash --scheduler pipeline 在 5060 Ti 上输出正确；batch=4 与 sdpa 速度相当（22.4 vs 23.3s）且省 ~2.1GB 显存；batch=1+pipeline 病态慢（170s），单行建议 eager
- [巩固] 环境导出：environment.yml（348 行，含 flash-attn/cuda-toolkit 13.0.3 等 157 个 pip 包）
- [巩固] generation_mode×max_new_tokens 对比：reports/04_generation_mode_benchmark.md（fast/hybrid/slow × 512/2048/8192，速度-精度+跨模式一致性）
- [巩固] 可复用 CLI：scripts/infer.py（detect/ground/ground_multi/ground_text/detect_text/ground_gui/point，输出 boxes/points 像素坐标 JSON + 标注图）
- [git] 项目纳入 git 管理：git init（main）；根 .gitignore（outputs/、缓存、Eagle LFS 占位）、.gitattributes（LF 归一化 + 二进制标记）
- [git] 推送 GitHub 远端：origin=https://github.com/Adlexer/locate_anything；filter-branch 清理历史中的 git-lfs 占位 prod_1.jpeg 后 force push 成功（main=200b6c4，541 文件）

---

- [LoRA/SFT] WSL conda 新建 locate_anything_sft（conda create --clone locate_anything，185 包 / python 3.10.20 / torch 2.9.0+cu130 / peft 0.12.0 / deepspeed 0.15.4 等），补装 nvidia-ml-py / sortedcontainers / tensorboard
- [LoRA/SFT] 源码研究：官方 LoRA 脚本 locate-anything-lora-visual-prompt.sh、locany_finetune_magi_stream.py、wrap_llm_lora/wrap_backbone_lora（modeling_locateanything.py）、MTP+stream packing+fused CE、TRAINING.md / DATA_PREPARATION.md
- [LoRA/SFT] 端到端 smoke 验证（5060 Ti 16GB）：LoRA r=64 + sdpa + seq=2048 + grad checkpoint + DS ZeRO-1(torch_adam)，2 steps 完成，~13s/it，峰值显存 16,136MB（顶格），checkpoint 全量 2 分片可被 scripts/infer.py 直接加载并正确推理（5 框）
- [LoRA/SFT] 定位并绕过 DeepSpeed 0.15.4 Blackwell sm_120 JIT 编译 bug（compute_capability_args 把 '12.0' 解析成 '1.' → nvcc compute_1. 失败；方案：optimizer torch_adam:true）
- [LoRA/SFT] 预研报告落盘：reports/05_lora_sft_research.md（原理普及 + 显存账 + 本机方案 + 踩坑清单）

## In Progress

- 无（巩固任务已收尾，等待用户后续指令）

---

## Next Step

- 待用户确认：a) 真实业务图/视频批量推理或服务封装（FastAPI）；b) LoRA/全参微调；c) 评估复现（详见 reports/02_todo_and_suggestions.md）

---

## User Requests

- [2026-08-04 15:16] 探查C:\Dev\locate_anything\Eagle\Embodied：locate anything源码。权重已经落盘本地：C:\Data\LocateAnything-3B
- [2026-08-04 15:22] [wsl-attach] 结合探查WSL中的环境，补齐WSL中的miniconda，conda创建locate_anything专用环境，禁止污染系统python，暂时不考虑使用本地(windows)python环境
- [2026-08-04 15:40] 做完上述工作之后，接下来任务编排：1.落盘初步探查报告 2.建立依赖环境，做一次真实图片推理冒烟测试 3.代办和建议项目落盘报告，等待继续 至此本预研任务结束
- [2026-08-04 18:01] 参考./reports/02_todo_and_suggestions.md，待办项目进一步巩固：1.权重拷贝+使能fa+la_flash验证+最终conda环境导出 2.对 generation_mode（fast/slow/hybrid）与 max_new_tokens（官方建议 8192）做速度-精度对比，精度时延报告落盘 3.基于上述工作，整理最终可复用的 infer.py CLI（输入图+query，输出 boxes/points + 标注图）
- [2026-08-04 18:2x] 用户更新 C:\Users\Adlexer Xu\.wslconfig：WSL 内存 48GB（给 Windows 留 16GB），以缓解 WSL 编译 OOM
- [2026-08-04 19:02] 用户要求整个项目工作目录纳入 git 管理

---

- [2026-08-04 19:30] 在当前的feat/codex/lora分支上，wsl机器conda新建locate_anything_sft clone from locate_anything，研究一下LoRA和SFT相关，因为我是infra这一块的，对训练不太了解，请预研同时给我输出报告，普及技术方案&原理。
## Timeline

- [2026-08-04 15:14] /Report generate 初始化 progress_report.md 与 result_report.md
- [2026-08-04 15:16] 用户发起 LocateAnything 源码探查任务
- [2026-08-04 15:30] 完成源码/权重/环境探查，产出分析
- [2026-08-04 15:22] 用户要求搭建 WSL conda 环境
- [2026-08-04 15:24] WSL 连接，安装 Miniconda 26.5.3
- [2026-08-04 15:30] 创建 locate_anything env（conda-forge，python 3.10）
- [2026-08-04 15:47] 安装 torch 2.9.0+cu130 及全部项目依赖，注册 eaglevl
- [2026-08-04 15:55] 合成图冒烟测试通过（模型加载 + 检测推理）
- [2026-08-04 15:40] 用户编排预研收尾：报告落盘 + 真实图冒烟 + 待办建议
- [2026-08-04 16:00] 真实图片（bus.jpg）冒烟通过（detect/ground/point），两份报告落盘，预研任务收尾
- [2026-08-04 18:01] 用户发起巩固任务（权重拷贝/FA/la_flash/环境导出 + gen-mode 对比 + infer.py CLI）
- [2026-08-04 18:10] 权重拷贝完成并校验（~/models/LocateAnything-3B，加载 2.9s）
- [2026-08-04 18:18] flash-attn 源码编译多次失败（WSL OOM / E_UNEXPECTED）→ 改用 Astral cu130 预编译 wheel 安装成功
- [2026-08-04 18:30] 用户将 .wslconfig 内存调至 48GB；patch modeling_qwen2.py magi→sdpa；验证 Vision=FA2/LLM=sdpa
- [2026-08-04 18:40] la_flash 批处理验证完成（4 组对比，输出正确，batch4 省 ~2.1GB）
- [2026-08-04 18:45] environment.yml 导出完成（含 flash-attn 2.8.3）
- [2026-08-04 18:50] generation_mode×max_new_tokens 基准完成（45 组），报告 04 落盘
- [2026-08-04 18:53] infer.py CLI 全任务类型验证通过（detect/ground_multi/point/detect_text），标注图+JSON 落盘
- [2026-08-04 19:02] 用户要求整个项目工作目录纳入 git 管理；git init + 初始提交 74e3766（540 文件）
- [2026-08-04 19:10] 用户创建 GitHub 远端 Adlexer/locate_anything；合并远端初始提交（LICENSE+README），filter-branch 移除历史 LFS 占位后 force push 成功

---

- [2026-08-04 19:30] 用户发起 LoRA/SFT 预研（clone env + 研究 + 报告）
- [2026-08-04 19:39] conda clone locate_anything → locate_anything_sft 完成（185 包，12G）
- [2026-08-04 19:5x] 端到端 LoRA smoke 训练通过（r=64/sdpa/seq2048，13.05s/it，峰值 16.1GB）；补装依赖并绕过 DS sm_120 bug
- [2026-08-04 19:58] 微调产物经 scripts/infer.py 验证推理正常（trainable 119,734,272，输出 5 框）
- [2026-08-04 20:0x] reports/05_lora_sft_research.md 落盘
- SFT 环境：conda activate locate_anything_sft；训练在 Eagle/Embodied 目录下用 torchrun --nproc_per_node=1 启动
- 本机 LoRA 方案：r=64 + sdpa + seq<=2048 + grad_checkpoint + DS ZeRO-1/2（optimizer 加 torch_adam:true）；全参 SFT 单卡不可行
- DeepSpeed 0.15.4 + sm_120：JIT FusedAdam 编译必失败（compute_1.），勿依赖 TORCH_CUDA_ARCH_LIST；用 torch_adam 或打补丁
- 数据注意：JSONL/recipe 需无 BOM UTF-8；annotation/root 相对路径按 cwd 解析，建议绝对路径；坐标用 <n> token（[0,1000]）
## Notes

- 报告：reports/03_environment_fa_laflash.md（环境/FA/la_flash）、reports/04_generation_mode_benchmark.md（速度-精度对比）
- WSL 激活：conda activate locate_anything；权重路径 ~/models/LocateAnything-3B（已本地化，勿再走 /mnt/c）
- 注意力基线：Vision=flash_attention_2（FA2）、LLM=sdpa（magi 回退已 patch 到 sdpa，两处 model 副本 + HF cache 已清理）
- 默认推理：generation_mode=hybrid, max_new_tokens=8192（官方建议；短任务 2048 亦够）
- la_flash：batch≥4 才有收益（省显存），单行用 eager 调度器
- 本机边界：5060 Ti（sm_120/16GB）仅 sdpa 短上下文；Magi/长上下文需 Hopper/Blackwell；flash-attn 必须用预编译 wheel（源码编译会 OOM 崩 WSL）
- flash-attn 安装：pip install "flash-attn==2.8.3+cu.13.0.torch.2.9" --index-url https://wheels.astral.sh/simple/cu130/ --no-deps

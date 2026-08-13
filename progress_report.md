# Progress Report

## Task

（阶段一·已完成）LocateAnything 真实小数据集验证：1) 预训练模型零样本标注能力（煤气罐/电动车/自行车，输出 YOLO txt 落数据集同目录，smoke 代码落 scripts/annotation/）；2) 小批量伪标签 LoRA 训练验证。已落盘报告 06，附显存事故复盘与 launcher 改进。

（阶段二·进行中）真实标注反馈闭环 + YOLO 训推专项：
1. 完善真实标注反馈闭环：人工修正伪标签 → 重建训练数据 → LoRA 重训 → eval_det 真实 F1；
2. YOLO 选型调研：训练 = AI 工作站（RTX5060Ti，甚至双卡）；导出与推理 = 嵌入式设备/摄像头/无人机等小 TOPS 算力设备；实验阶段允许推理本地验证；
3. YOLO 训推框架建立后，探索标注→训练→导出自动化闭环工作流（可 agent skill 化）；
4. 环境在 WSL 新建 yolo 专用 conda 隔离依赖；维持报告落盘格式 + reports/INDEX.md 索引；代码经 black 规范化；新任务代码入分支 feat/codex/yolo。

---

## Current Status

Running（阶段一完成；阶段二进行中：真实标注反馈闭环 → YOLO 选型调研与训推框架 → 闭环工作流）

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

- [整理] scripts 目录重组为 scripts/{infer,bench,train,env,smoke}/，删除遗留 .b64 临时文件；verify_*.py 实为 bash 包装，改名 verify_*.sh；同步更新 04/05 报告中脚本路径
- [整理] 新增 LoRA 训练启动/监控脚本：scripts/train/{train_lora.sh, train_watch.sh, ds_z1_torchadam.json, README.md} + scripts/README.md
- [整理] train_lora.sh 端到端验证：1 步训练完成（loss 0.6534，checkpoint 落盘）、断点续训检测与 done.txt 提示、监控脚本 tail/checkpoints 模式可用

- [标注] 零样本标注管线落盘 scripts/annotation/{annotate_yolo.py, build_train_jsonl.py, eval_before_after.py, README.md}；全量标注 49 图 + 9 视频（72 帧），0 错误，93 框（煤气罐 58/电动车 28/自行车 7），122 份 YOLO txt + classes.txt 落 C:\Data\datasets\detect
- [标注] 能力结论：煤气罐识别稳定；电动车/自行车子类判定有噪声（二选一互斥）；probe 输出留档 outputs/annotation_detect/probe/
- [训练] 78 条伪标签（train）+ 8 holdout，LoRA r=64 + seq=2048，150 步/9min，loss 1.19→0.41；微调产物 ~/lora_gas/run_v1（checkpoint-50/100/150）
- [训练] before/after（slow 模式）8/8 类别一致、平均 IoU 0.987——无灾难性遗忘；发现 fast/hybrid 在 1080p 大图上截断回归、tokenizer.model_max_length 被烤成 2048
- [教训] 显存事故复盘：seq=2048 训练 ~15GB 时并发推理差点搞崩宿主；launcher 默认 MAX_SEQ 2048→1536、seq>=2048 三重警告、训练后自动恢复 model_max_length、新增 --warmup；train/README 记录 4 条经验
- [报告] reports/06_zeroshot_annotation_and_lora_validation.md 落盘（结论/数据/能力表/训练对比/显存复盘/下一步）

- [eval] 标准化检测评估脚本落盘 scripts/eval/{eval_det.py, README.md}：模型 vs YOLO GT，逐类 P/R/F1@IoU(0.5/0.75/0.9) + F1@Mean + matched-IoU + 逐图明细，JSON+MD；预训练与微调模型在 8 张 holdout 上均为 macro F1@Mean=1.000（自洽口径），gas mIoU 0.992 vs 0.985
- [整理] black 26.5.1 规范化 scripts/ 全部 13 个 py（Eagle/ 上游保持不动），新增 root pyproject.toml 锁定格式配置；提交 e6aab29 并推送 feat/codex/lora
- [整理] 自 feat/codex/lora@e6aab29 签出新分支 feat/codex/yolo 并推送
- [索引] 新增 reports/INDEX.md 索引机制（01-06 登记 + 07-10 占位），双 report 同步体现

- [反馈闭环] 工具化闭环管线：make_corrected_dataset.py（修正策略+工作表）、yolo_txt_to_manifest.py（txt→manifest）、eval_feedback_loop.sh（双模型评估）；detect_v2 修正数据集（121 图/93 框，gas 58/scooter 35/bike 0）；run_v2 重训 + 真实 F1 评估落盘（reports/07）

## In Progress

- [完成] 真实标注反馈闭环（报告 07）：修正策略（bicycle→scooter 合并 + 降采样 ≤1280px）→ detect_v2 → run_v2 重训（150 步 / 1.62s/it）→ 双模型 eval（macro F1 预训练 1.000 / 微调 0.992，半自洽口径已说明）；工作表+可视化待用户视觉复核
- [规划] YOLO 选型调研：双场景（训练工作站 / 嵌入式小算力推理导出）模型与框架评估
- [规划] WSL 新建 yolo 专用 conda 环境 + YOLO 训推框架（训练/导出/本地推理验证）
- [规划] 标注→训练→导出自动化闭环工作流（agent skill 化方案）

---

## Next Step

### 建议（按优先级）

1. **真实标注反馈闭环**（本阶段首要）：抽检/人工修正伪标签（重点 scooter/bicycle 混淆）→ 重建 manifest/JSONL/recipe → LoRA run_v2 重训 → eval_det 出真实 F1；
2. **YOLO 选型调研**（报告 08）：训练 = AI 工作站 RTX5060Ti（单/双卡）；导出与推理 = 嵌入式/摄像头/无人机小 TOPS 设备；实验期本地推理验证；
3. **YOLO 训推框架**（报告 09）：WSL 新建 yolo conda 环境；训练/导出（ONNX·TensorRT·INT8）/本地推理验证；
4. **标注→训练→导出闭环工作流**（报告 10）：含 agent skill 化方案；
5. 报告 07 落盘真实 F1 结论；更新 INDEX + 双 report，提交推送 feat/codex/yolo。

### 待办清单

- [x] 推送 `feat/codex/lora` 至 origin（2026-08-06 用户手动推送，HEAD=aa30afa）
- [ ] 人工伪标签修正（visualize_yolo.py 抽查 → 改 YOLO txt → 重建 JSONL/recipe）
- [ ] 降采样脚本 + run_v2 重训 + eval_det 真实对比
- [ ] 大图入训后复测 fast/hybrid 解码回归
- [ ] 服务化 / 批量推理

- [ ] YOLO 选型调研落盘（报告 08）
- [ ] WSL yolo conda 环境 + 训推框架（报告 09）
- [ ] 标注→训练→导出闭环工作流（报告 10，含 skill 化）

- [2026-08-13 13:3x] 反馈闭环执行：detect_v2 修正数据集（类合并+降采样）、manifest/JSONL 重建（78+8）、run_v2 重训（150 步 4:31）、eval 双模型对比（macro F1 1.000/0.992）
- [2026-08-13 13:4x] 报告 07 落盘；INDEX.md 同步登记

## User Requests

- [2026-08-04 15:16] 探查C:\Dev\locate_anything\Eagle\Embodied：locate anything源码。权重已经落盘本地：C:\Data\LocateAnything-3B
- [2026-08-04 15:22] [wsl-attach] 结合探查WSL中的环境，补齐WSL中的miniconda，conda创建locate_anything专用环境，禁止污染系统python，暂时不考虑使用本地(windows)python环境
- [2026-08-04 15:40] 做完上述工作之后，接下来任务编排：1.落盘初步探查报告 2.建立依赖环境，做一次真实图片推理冒烟测试 3.代办和建议项目落盘报告，等待继续 至此本预研任务结束
- [2026-08-04 18:01] 参考./reports/02_todo_and_suggestions.md，待办项目进一步巩固：1.权重拷贝+使能fa+la_flash验证+最终conda环境导出 2.对 generation_mode（fast/slow/hybrid）与 max_new_tokens（官方建议 8192）做速度-精度对比，精度时延报告落盘 3.基于上述工作，整理最终可复用的 infer.py CLI（输入图+query，输出 boxes/points + 标注图）
- [2026-08-04 18:2x] 用户更新 C:\Users\Adlexer Xu\.wslconfig：WSL 内存 48GB（给 Windows 留 16GB），以缓解 WSL 编译 OOM
- [2026-08-04 19:02] 用户要求整个项目工作目录纳入 git 管理

---

- [2026-08-04 19:30] 在当前的feat/codex/lora分支上，wsl机器conda新建locate_anything_sft clone from locate_anything，研究一下LoRA和SFT相关，因为我是infra这一块的，对训练不太了解，请预研同时给我输出报告，普及技术方案&原理。
- [2026-08-04 20:05] 下一步任务：1.整理./scripts目录 2.落盘LoRA训练的启动监控脚本
- [2026-08-05 01:40] 巧了，我这正好有些数据集给你：C:\Data\datasets\detect，分别是煤气罐和电动车（包含少量自行车）图片和视频的小数据集，没有任何标注，现在我想验证：1.你的预训练模型的识别标注能力（对不常见物体），输出为YOLO格式的标注txt，放到数据集同目录下，smoke代码落盘：./scripts/annotation/ 2.小批量数据的LoRA训练验证。
- [2026-08-05 02:05] 这个训练配置的内存占用太极限了，差点把宿主windows都炸掉，先等待这次训练完成吧，下次记得吸取教训。
- [2026-08-13 13:0x] 这个专项任务我们将完善最后的真实标注反馈闭环，然后将重心转移到YOLO模型的训推中。首先需要明确几点：
  1.目前locate anything的最终设计目标为YOLO数据集的一种自动标注流水线，而且能够通过标注反馈闭环，人工标注也像是某种RL，持续迭代标注质量直到稳定。
  2.YOLO在历史上已经迭代了相当多的版本，在训推任务开始之前我希望先做足选型和调研工作，评估两种使用场景：训练是AI工作站（RTX5060Ti，甚至双卡），导出和推理是嵌入式设备/摄像头/无人机等小TOPS算力设备。当前实验阶段可以允许推理在本地验证。
  3.当YOLO完成训推框架的建立之后，探索某种和标注流水线一起闭环的工作流，形成标注->训练->导出的自动化流水线，你也可以以skill的方式建立在agent上的自动化。具体方案由你调研决定。
  4.具体项目细节注意：开发环境对标之前任务，在wsl上新建yolo专用conda环境以隔离依赖；注意维持报告落盘格式，并且要补充建立相应的索引机制：./reports/INDEX.md，双report中也要有所体现；任务开始之前对当前代码分支未提交上库部分做整理收尾上库，使用formatter对python代码进行自动规范化；新任务代码提交到新分支：feat/codex/yolo，由当前分支签出。

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
- [2026-08-04 20:05] 用户发起 scripts 整理 + LoRA 训练启动监控脚本任务
- [2026-08-04 20:10] scripts 重组为 infer/bench/train/env/smoke 子目录；删除 .b64 残留；verify_*.py 改名 .sh；更新 04/05 报告路径
- [2026-08-04 20:15] 新增 scripts/train/train_lora.sh + train_watch.sh + ds_z1_torchadam.json + README（含 scripts/README.md）
- [2026-08-04 20:18] train_lora.sh 端到端验证通过（1 步训练 + 断点续训检测 + done.txt 提示）；train_watch tail/checkpoints 模式可用
- [2026-08-05 01:40] 用户提供 C:\Data\datasets\detect 小数据集，发起零样本标注 + LoRA 训练验证
- [2026-08-05 01:5x] 数据盘点 + 词汇 probe（gas 稳 / 两轮车子类模糊）；annotate_yolo.py 全量标注完成（93 框，122 txt）
- [2026-08-05 02:00] build_train_jsonl 产出 78+8 样本；2 步 smoke 通过；150 步正式训练启动（seq=2048）
- [2026-08-05 02:05] 用户反馈显存极限险情（训练+并发推理差点搞崩宿主），要求等训练完成并吸取教训
- [2026-08-05 02:11] 训练完成：loss 1.19→0.41；before/after 评测（slow 8/8 一致 IoU 0.987）；发现 fast/hybrid 大图回归 + model_max_length 烤成 2048
- [2026-08-05 02:1x] launcher 改进（MAX_SEQ 默认 1536 + 警告 + model_max_length 自动恢复 + --warmup）并 1 步回归验证通过；报告 06 落盘
- 标注产物：C:\Data\datasets\detect 下 *.txt + classes.txt + _frames/；aux 在 outputs/annotation_detect/
- 训练/评测复现：reports/06 第 5 节
- 伪标签类别建议：业务只关心两轮车时用 --merge-two-wheeler；高分辨率帧需降采样入训
- [2026-08-05 02:2x] 用户提出缺标准化 eval 脚本；新增 scripts/eval/eval_det.py + README；holdout 上预训练/微调双模型标准评估通过（F1@Mean=1.000 自洽口径）
- [2026-08-06] 用户手动推送 feat/codex/lora 至 origin（HEAD=aa30afa），本地/远端一致
- [2026-08-13 13:0x] 用户发起新专项：完成真实标注反馈闭环 + 转入 YOLO 训推（选型调研、双场景评估、闭环工作流、yolo conda 隔离、INDEX 索引、black 规范化、分支 feat/codex/yolo）
- [2026-08-13 13:1x] black 26.5.1 规范化 scripts/ 13 个 py + root pyproject.toml；提交 e6aab29 推送 feat/codex/lora
- [2026-08-13 13:1x] 签出并推送 feat/codex/yolo；建立 reports/INDEX.md 索引机制

## Notes

- 报告：reports/03_environment_fa_laflash.md（环境/FA/la_flash）、reports/04_generation_mode_benchmark.md（速度-精度对比）
- WSL 激活：conda activate locate_anything；权重路径 ~/models/LocateAnything-3B（已本地化，勿再走 /mnt/c）
- 注意力基线：Vision=flash_attention_2（FA2）、LLM=sdpa（magi 回退已 patch 到 sdpa，两处 model 副本 + HF cache 已清理）
- 默认推理：generation_mode=hybrid, max_new_tokens=8192（官方建议；短任务 2048 亦够）
- la_flash：batch≥4 才有收益（省显存），单行用 eager 调度器
- 本机边界：5060 Ti（sm_120/16GB）仅 sdpa 短上下文；Magi/长上下文需 Hopper/Blackwell；flash-attn 必须用预编译 wheel（源码编译会 OOM 崩 WSL）
- flash-attn 安装：pip install "flash-attn==2.8.3+cu.13.0.torch.2.9" --index-url https://wheels.astral.sh/simple/cu130/ --no-deps

- 新阶段开发分支：feat/codex/yolo（自 feat/codex/lora@e6aab29 签出，已推送）
- 代码格式化：black 26.5.1（root pyproject.toml，line-length 88）；Eagle/ 上游不动
- 报告索引：reports/INDEX.md 为本项目报告唯一入口，新增报告必须登记
- 数据集现状：detect 121 图 / 93 框（gas 58 / scooter 28 / bike 7），含 _frames/ 视频帧；outputs/annotation_detect/manifest.jsonl 为过期文件，需由修正后 txt 重建

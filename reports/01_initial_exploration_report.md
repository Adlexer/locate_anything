# LocateAnything 初步探查报告（预研阶段）

- 日期：2026-08-04
- 状态：预研阶段完成
- 源码：`C:\Dev\locate_anything\Eagle\Embodied`（NVlabs/Eagle 的 Embodied 子目录，非 git 仓库快照）
- 权重：`C:\Data\LocateAnything-3B`（HuggingFace nvidia/LocateAnything-3B 本地副本，~7.7GB safetensors，含 batch_utils/kernel_utils）

---

## 1. 项目是什么

LocateAnything 是 NVIDIA 提出的通用视觉定位 VLM（ECCV 2026），核心贡献为 **Parallel Box Decoding (PBD)**：把每个 bounding box / point 视为"原子单元"，一次并行前向直接输出完整坐标 `(x1,y1,x2,y2)`，取代逐 token 自回归解码。

- 架构：MoonViT-SO-400M（视觉编码，27 层 / hidden 1152 / patch 14 / 2×2 merge）+ MLP 投影（2 层）+ Qwen2.5-3B-Instruct（语言模型，36 层 / hidden 2048，改造支持 MTP）
- 总参数量：约 **3.83B**（safetensors 头部实测 2,479,779,056 + 1,350,886,912）
- 官方指标：H100 12.7 BPS（约 10× Qwen3-VL、2.5× Rex-Omni）；LVIS F1 50.7、COCO 54.7、DocLayNet 76.8、M6Doc 70.1、ScreenSpot-Pro 60.3、RefCOCOg-val 76.7
- 训练数据 LocateAnything-Data：12M 图 / 138M queries / 785M boxes（检测 66.9%、GUI 16.5%、referring 7.3%、OCR 3.6%、layout 3.5%、point 2.2%）

## 2. 目录结构

```
Embodied/
├── README.md / pyproject.toml / locateanything_worker.py  # 说明 + 依赖 + 推理 Worker
├── eaglevl/
│   ├── model/locany/      # 训练态模型代码（Magi/SDPA 注意力、MTP 掩码）
│   ├── model/moon_vit/    # MoonViT 编码器
│   ├── utils/locany/      # 推理态处理器/模型副本（≈HF 发布版）
│   ├── train/             # locany_finetune_magi_stream.py（主训练脚本 69KB）、dataset、liger 融合损失、fastseek 视频抽帧
│   ├── patch/             # liger 融合算子、数据采样/打包、unsloth checkpoint 等训练补丁
│   ├── sp_utils/          # 序列并行（ring/ulysses、all-to-all）
│   └── conversation.py    # 会话/模板工具
├── evaluation/            # COCO/LVIS/grounding/SSPro DDP 推理 + fastevaluate 指标
├── shell/                 # streaming 全参微调 & LoRA visual-prompt 微调脚本
├── document/              # TRAINING / DATA_PREPARATION / RESULTS / STREAMING_PACKING
└── deepspeed_configs/     # ZeRO-1 / ZeRO-2
```

三份模型代码关系：`eaglevl/model/locany`（训练态）≈ `eaglevl/utils/locany`（推理态）≈ `C:\Data\LocateAnything-3B`（HF 发布版，做了小改动：LoRA clone 修复、attn 校验、LMDB 路径清理）。

## 3. 核心方法要点

### 3.1 三种生成模式（generate 的 generation_mode）
| 模式 | 说明 |
|---|---|
| fast | 纯 MTP，不回退，最快 |
| slow | 纯 AR（NTP），最稳 |
| hybrid（默认） | MTP 为主，`handle_pattern()` 检测到格式异常（error_box）回退 AR，遇 `</box>` 再切回 MTP |

### 3.2 MTP 机制
- 每次前向：`1 个真实 token + 5 个 <text_mask>`（n_future_tokens=6，block_size=6），位置 ID 对 mask 位置 −1
- `sample_tokens()` 中 `decode_bbox_avg()` 对 top-k 坐标 token 做概率加权平均得到原子 box；`handle_pattern()` 规整为 `<box><x1><y1><x2><y2></box>` / 点 / `<box>none</box>` / `<ref>…</ref>`
- 注意力为 block-diffusion 掩码：prefix 因果 + 扩散窗口内双向（causal_attn=False）
- 训练损失：`find_pred_pos_from_input_ids` 分桶 MTP 分段 loss（最多 4 步未来预测）

### 3.3 特殊 Token（vocab 152681）
`<IMG_CONTEXT>`(151665)、`<box>/</box>`(151668/9)、`<ref>/</ref>`(151672/3)、`<text_mask>`(151676)、`<0>..<1000>`(151677..152677，÷1000 归一化坐标)、`<null>`(152678)、`<switch>`(152679)、`</c>`(152680)

### 3.4 推理/批处理运行时
- `locateanything_worker.py`：detect / ground_single / ground_multi / ground_text / detect_text / ground_gui / point / predict + parse_boxes / parse_points
- `batch_infer.py` + `batch_utils/`（hybrid 调度器）+ `kernel_utils/`（la_flash：FlashAttention varlen 稀疏 range 计划，免 C++ 扩展，适合 A100/4090 等非 Hopper 卡）

## 4. 训练管线

- 数据：JSONL（ShareGPT 格式，`<100>` 式坐标 token）+ recipe JSON（annotation/root/repeat_time/data_augment/visual_prompt）
- Streaming Packing：在线 best-fit/big-rocks-first 打包，per_device_train_batch_size=1 + token 预算，状态化迭代器支持逐 bit 一致断点续训
- MTP 目标构建：有 `</box>/</ref>` 时按 box/ref 边界切块（box 原子性），无检测标记时随机切块
- 注意力后端：magi（Hopper/Blackwell，32K+）或 sdpa（任意卡，~4K）；损失用 Liger 融合 CE
- LoRA 微调：`USE_LLM_LORA=64` 默认冻结 LLM+backbone、只训 MLP；官方权重不支持 visual prompt 推理（需自行微调）
- 评估：COCO/LVIS 需从 Rex-Omni 拉 fastevaluate（C++ 模块）；数据源 Mountchicken/Rex-Omni-EvalData 与 likaixin/ScreenSpot-Pro

## 5. 环境与冒烟测试结果

### 5.1 WSL 环境（本预研阶段搭建）
- Ubuntu 24.04.4 LTS；Miniconda 26.5.3 → `/home/xu/miniconda3`（仅 conda-forge 渠道，default_channels=[]）
- 专用环境 `locate_anything`（python 3.10.20）：
  - torch 2.9.0+cu130 + torchvision 0.24.0（RTX 5060 Ti sm_120 可用，CUDA True）
  - transformers 4.57.1 / tokenizers 0.22.0 / numpy 1.26.4 / peft 0.12.0 / accelerate 1.5.2 / timm 1.0.28 / deepspeed 0.15.4 / liger_kernel 0.3.1 / gradio 3.35.2 / streamlit / decord 0.6.0 等全套
  - 环境内 cuda-toolkit（nvcc 12.8），`CUDA_HOME=$CONDA_PREFIX` 已通过 activate.d hook 固化
  - 项目以 editable 安装（`pip install -e . --no-deps`）
- 系统 python（3.12.3）未做任何改动；未使用 Windows python 环境

### 5.2 真实图片推理冒烟测试（ultralytics bus.jpg，810×1080）
| 任务 | 结果 |
|---|---|
| detect(person/bus/car) | 4 个 person + 1 个 bus + 1 个 car（6 boxes），2.0 BPS，无 AR 回退 |
| ground_multi("people") | 3 个 person 区域 |
| point("the bus") | 点 (499,437)/1000（车体中心附近） |

- 模型加载 31s（/mnt/c 盘），detect 推理约 3.0s（10 前向步 / 46 token）
- 注意力后端：magi 不可用自动回退 **sdpa**（符合 5060 Ti 预期）
- 产物：`outputs/smoke_real_annotated.jpg`、`outputs/smoke_real_input.jpg`、`outputs/smoke_real_results.json`

## 6. 结论

1. 源码可读、依赖可装、权重可用：端到端推理链路（加载 → 预处理 → MTP 生成 → 解析）已在 WSL+RTX 5060 Ti 上验证通过。
2. 本机定位：消费级 Blackwell 卡（sm_120、16GB）适合 **sdpa 短上下文（~4K）推理/小规模微调**；MagiAttention（32K+ 长上下文、高速训练）与官方 H100 基准需 Hopper/Blackwell。
3. 限制：官方 3B 权重不支持 visual prompt 推理；`/mnt/c` 加载偏慢（31-42s）。

## 7. 关键参考

- 论文：https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf（arXiv:2605.27365）
- 项目页：https://research.nvidia.com/labs/lpr/locate-anything/ ｜ GitHub：https://github.com/NVlabs/Eagle（Embodied）
- HF 模型：https://huggingface.co/nvidia/LocateAnything-3B
# LocateAnything LoRA / SFT 预研报告（面向基础设施工程师）

- 日期：2026-08-04
- 分支：`feat/codex/lora`（`af9031e`）
- 环境：WSL Ubuntu 24.04 / RTX 5060 Ti 16GB (sm_120) / conda `locate_anything_sft`（clone 自 `locate_anything`）
- 关键库：python 3.10.20 / torch 2.9.0+cu130 / transformers 4.57.1 / peft 0.12.0 / deepspeed 0.15.4 / flash_attn 2.8.3
- 模型：`~/models/LocateAnything-3B`（3.52B，本地化）
- 目的：为不熟悉训练的基础设施工程师普及 SFT 与 LoRA 的原理、技术方案，并结合本机（单卡 16GB）给出可落地的路径与实测依据。

---

## 0. TL;DR（一页结论）

1. **SFT（监督微调）** = 用「问题→标准答案」数据继续训练模型，让它学会某个领域/任务的输出格式与能力；**LoRA** = 只训练一小撮新增的低秩矩阵（约 3~4% 参数），冻结其余全部权重，是消费级显卡上做微调**唯一现实**的路线。
2. LocateAnything-3B 官方**原生支持 LoRA 微调**（`shell/locate-anything-lora-visual-prompt.sh`），默认配置：LLM 加 LoRA(r=64)、Vision/LLM 冻结、MLP 投影器可训练。
3. 本次已在 `locate_anything_sft` 环境**端到端跑通** LoRA 微调（2 步 smoke 测试）：
   - 训练可完成，~13 s/step（seq=2048），**峰值显存 ~16.1GB（已顶格 16GB）**；
   - 产出 checkpoint 能被现有 `scripts/infer/infer.py` 直接加载并正确推理（5 个框）；
   - 因此**本机可行方案 = LoRA r=64 + sdpa + seq≤2048 + grad checkpoint + DeepSpeed ZeRO-1/2**。
4. 全参 SFT（全部 3.52B 可训练）在 16GB 单卡**不可行**，需要多卡（官方用 8×H100 80GB）。
5. 踩了 4 个坑（见 §9），其中 **DeepSpeed 0.15.4 在 Blackwell sm_120 上的 `compute_1.` 编译 bug** 是本机特有、必须处理的问题。

---

## 1. 概念普及：SFT 是什么

### 1.1 训练三阶段（宏观）

| 阶段 | 做什么 | 数据 | 成本 |
|---|---|---|---|
| 预训练 (Pretrain) | 从零学语言/世界知识（next-token 预测） | 海量文本/图文（TB 级） | 极高（千卡月） |
| 指令微调 (SFT) | 学会「听指令、按格式作答」 | 任务问答（万~百万级） | 高（但远低于预训练） |
| 对齐 (RLHF/DPO 等) | 让回答更符合偏好 | 人类偏好/奖励数据 | 中 |

LocateAnything 已经完成前两个阶段（官方用 138M 样本做 SFT 得到现在的权重）。**我们说的「微调」= 在已有权重之上，再用自己的业务数据做一次 SFT**，让它学会我们的领域格式（例如特定行业的目标类别、GUI 元素、版面元素）。

### 1.2 SFT 的训练目标（一句话）

给定输入（图像 + 文本指令），让模型输出的 token 序列**尽量接近标注的标准答案**（交叉熵损失，`CE loss`）。训练只更新「需要计算梯度」的那部分参数。

### 1.3 为什么不能直接改 prompt 而要微调

- 模型权重里没有你的领域知识/输出格式时，prompt 再长也「学不会」；
- 微调能**固化**输出格式（`<ref>…</ref><box>…</box>`）、类别集合、专有名词，并提升精度；
- 代价是需要数据、算力与训练运维——这正是本报告要解决的部分。

---

## 2. 概念普及：LoRA 原理

### 2.1 动机

全参 SFT 需要为**每个参数**保存：权重(bf16) + 梯度 + 优化器状态（fp32 主副本 + 两个动量 ≈ 6× 参数量）。3.5B 模型单卡 16GB 根本放不下。

### 2.2 核心思想（低秩分解）

训练时**不更新原权重 W**，而是冻结它，只训练两个小矩阵 A、B：

```
W' = W + (α / r) · B · A
     ↑冻结      ↑新增的低秩适配器
```

- `W`：原始权重矩阵，形状 `d × k`（例如 attention 的 q_proj：2048×2048）；
- `A`：`r × k`，随机高斯初始化；
- `B`：`d × r`，全零初始化（保证训练开始时 W'=W，不破坏原模型）；
- `r`（rank，秩）：低秩的维度，官方默认 64；
- `α`（alpha）：缩放系数，官方取 `2r`，实际缩放 = α/r = 2；
- 训练只更新 A/B；推理时可以把 `B·A` 合并回 W（本仓库甚至不用合并，见 §4.4）。

### 2.3 为什么省显存

新增参数量 = `r × (d + k)`，与原矩阵 `d × k` 相比小几个数量级。以 q_proj(2048×2048) 为例：原 419 万参数，LoRA(r=64) 仅 26.2 万，**约 1/16**。

梯度与优化器状态**只对新增参数计算**，因此：

```
可训练参数 ≈ 4%  →  优化器/梯度显存 ≈ 全参 SFT 的 1/25
```

### 2.4 常见衍生概念（本仓库涉及到的）

| 概念 | 说明 | 本仓库 |
|---|---|---|
| 冻结 (freeze) | 让某模块 `requires_grad=False`，不参与训练 | `--freeze_llm/--freeze_backbone/--freeze_mlp` |
| target_modules | 给哪些线性层挂 LoRA | LLM: q/k/v/o_proj + gate/up/down_proj；Vision: q/k/v/o + fc1/fc2 |
| QLoRA | 再加 4-bit 量化进一步省显存 | 本仓库未启用（可用 bitsandbytes） |
| DoRA / AdaLoRA | LoRA 变体（权重归一化 / 自适应秩） | 未内置，需自行改代码 |
| 合并 (merge) | 把 B·A 写回 W，推理更快 | 本仓库 save 即全量 checkpoint，无需合并（见 §4.4） |

---

## 3. LocateAnything-3B 模型结构（决定我们调哪里）

`config.json` 显示为三部分，总计 **3,519,889,408 参数（≈3.52B）**：

| 组件 | 说明 | 参数规模（约） | 占比 |
|---|---|---|---|
| Vision 编码器 | MoonViT-SO-400M（27 层，hidden=1152，patch=14，2×2 merge） | ~400M | 11% |
| 连接器 MLP | LayerNorm(4608) + Linear(4608→2048) + GELU + Linear(2048→2048) | 13.6M | 0.4% |
| 语言模型 | Qwen2.5-3B-Instruct（36 层，hidden=2048，16 头，2 KV 头，intermediate=11008） | ~3.09B | 88% |

> 权重的 bf16 占用 ≈ 3.52B × 2B = **7.0 GB**（仅权重，不含激活/优化器），这决定了 16GB 卡静态就吃掉近一半。

微调时一般策略（官方 LoRA 默认）：
- **Vision 冻结**（视觉先验已够好，且 400M 全训显存压力大）；
- **LLM 加 LoRA**（核心学习能力所在）；
- **MLP 连接器可训练**（很小，13.6M，负责对齐视觉/语言两个空间，通常值得训）。

---

## 4. 官方训练管线解读（代码层面）

### 4.1 入口与脚本

| 文件 | 作用 |
|---|---|
| `Eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py` | 训练主程序（HF Trainer 定制版） |
| `Eagle/Embodied/shell/locate-anything-lora-visual-prompt.sh` | **LoRA 微调官方脚本**（含 visual prompt 数据转换） |
| `Eagle/Embodied/shell/locate-anything-streaming.sh` | 全参 SFT 官方脚本 |
| `Eagle/Embodied/document/TRAINING.md` / `DATA_PREPARATION.md` | 官方训练/数据文档 |

### 4.2 官方 LoRA 脚本的关键参数（默认值）

```bash
USE_LLM_LORA=64          # LLM LoRA rank=64（0 关闭）
USE_BACKBONE_LORA=0      # Vision LoRA rank=0（默认关闭）
FREEZE_LLM=True          # 冻结 LLM 原始权重
FREEZE_BACKBONE=True     # 冻结 Vision 原始权重
FREEZE_MLP=False         # MLP 连接器可训练
MAX_STEPS=5000  LR=2e-5  WARMUP_STEPS=500
MAX_SEQ_LENGTH=16384     # 官方目标长上下文（需要 Hopper/Blackwell 服务器卡）
ATTN=--attn_implementation magi   # MagiAttention（仅 H100/H20/B200 等）
DEEPSPEED=zero_stage1_config.json
```

### 4.3 代码里 LoRA 到底怎么挂的

`eaglevl/model/locany/modeling_locateanything.py`：

```python
def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
    lora_config = LoraConfig(
        r=r,
        target_modules=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
                        'mlp.gate_proj','mlp.down_proj','mlp.up_proj'],
        lora_alpha=lora_alpha, lora_dropout=lora_dropout, task_type='CAUSAL_LM')
    self.language_model = get_peft_model(self.language_model, lora_config)
    ...
```

- 用 HuggingFace **PEFT** 库实现，`lora_alpha=2*r`（缩放比恒为 2）；
- Vision 侧 `wrap_backbone_lora` 目标为 `self_attn.q/k/v/o_proj` + `mlp.fc1/fc2`；
- 训练冻结顺序（主程序 `main()`）：先 `freeze_*` 全部冻结 → 再 `wrap_*_lora`（PEFT 会把 LoRA 参数重新置为可训练）→ 最后可选 `freeze_mlp`/`unfreeze_vit_layers`。

### 4.4 训练三大特色（LocateAnything 特有）

1. **MTP（Multi-Token Prediction）**：`--block_size 6`。把 box 的 4 个坐标 token（加 `</box>` 等）作为**一个块并行预测**（用 `<text_mask>` 占位），而不是逐 token 生成——这是它推理快的根本原因，训练时同样生效；
2. **Stream（在线）Packing**：`per_device_train_batch_size=1`，用 `max_num_tokens`（token 预算）+ `packing_buffer_size` 把多个样本**拼进一个 batch**，避免 padding 浪费显存；
3. **Fused CE Loss**：Liger 融合 Linear+CrossEntropy，不物化超大 logits 矩阵，省显存。

另外注意：训练脚本 `trainer.save_model()` 保存的是**完整 checkpoint**（冻结权重 + LoRA 权重都在里面，约 7.3GB/份），config 里写入 `use_llm_lora=64` + `auto_map`，因此**微调产物可直接被 `scripts/infer/infer.py` / `locateanything_worker.py` 加载推理，无需额外 merge 步骤**（本次已实测验证）。

---

## 5. 数据准备（SFT 的输入）

### 5.1 两级配置

```
recipe.json（--meta_path 传入）
 └─ 数据集名 → { annotation: xxx.jsonl, root: 图片目录, repeat_time: 采样权重, data_augment: 是否增强, visual_prompt: 是否转视觉提示 }
```

### 5.2 JSONL 样本格式（ShareGPT 风格）

```jsonl
{"conversations": [
   {"from":"human","value":"Detect all objects in <image-1>."},
   {"from":"gpt","value":"<ref>car</ref><box><100><200><400><500></box><ref>person</ref><box><250><100><450><600></box>"}
 ], "image":"train/00001.jpg"}
```

关键约定：
- 图片用 `<image-1>`、`<image-2>` 占位；也可 `image_list`/`video`；
- **坐标是 [0,1000] 归一化整数，且必须写成 `<n>` 形式的特殊 token**（`<0>`~`<1000>`，模型输出也是这种 token）；
- 定位 token：`<ref>类别</ref><box><x1><y1><x2><y2></box>`；点：`<box><x><y></box>`；无目标：`<box>None</box>`；
- 多类别 prompt 用 `</c>` 分隔；
- 纯文本 QA 也支持（无 image 字段）。

> 踩坑提示：JSONL/recipe 必须是**无 BOM 的 UTF-8**（本次曾因 PowerShell 写文件带 BOM 导致 `Unexpected UTF-8 BOM` 报错）。

### 5.3 visual_prompt（官方新特性）

`visual_prompt: true` 的数据集会把「单类别检测」自动转成「图片裁剪作为查询」的样本（来源图仍是目标图，裁剪图作为额外图片占位）。注意：**当前公开权重不支持 visual prompt 推理**，需要自己微调出视觉提示能力。

---

## 6. 显存/成本估算（含实测）

### 6.1 理论账（LoRA r=64 + MLP 可训练，单卡）

| 项 | 计算 | 显存 |
|---|---|---|
| 冻结权重 bf16 | 3.52B × 2B | ~7.0 GB |
| LoRA 权重 bf16 | 119.73M × 2B | 0.22 GB |
| MLP 权重 bf16 | 13.64M × 2B | 0.03 GB |
| 梯度（仅可训练） | 133.4M × 2B | 0.25 GB |
| AdamW 状态（fp32 主副本 + 2 动量） | 133.4M × 4B × 3 | 1.49 GB |
| **静态合计** | | **≈ 9.0 GB** |
| 激活 + CUDA 上下文（seq=2048, grad checkpoint） | 实测 | **≈ 7.1 GB** |
| **峰值总计** | | **≈ 16.1 GB（顶格）** |

LoRA 参数精确值：每层 3,325,952（q/k/v/o=262k/147k/147k/262k，gate/up/down=835k 各）× 36 层 = **119,734,272**（训练日志实测打印一致）。若同时开 Vision LoRA r=64，再 +34.78M。

### 6.2 本次实测（5060 Ti 16GB，`locate_anything_sft`）

| 项 | 值 |
|---|---|
| 配置 | LoRA r=64（LLM）+ MLP 可训练，freeze LLM/Vision，sdpa，seq=2048，grad checkpoint，DeepSpeed ZeRO-1 + torch AdamW |
| 数据 | 2 个样本（1 张图，detect + ground） |
| 训练 | 2 steps 完成，loss 0.65→2.66（小数据过拟合，正常），**13.05 s/it** |
| 峰值显存 | **Step1 16,136 MB / Step2 15,988 MB**（温度 41→50°C，功耗 ~107W） |
| checkpoint | 全量 2 分片 ~7.3GB + DeepSpeed optimizer states + dataloader 状态 |
| 训练后推理 | `scripts/infer/infer.py --model <输出目录>` 4.5s 加载，输出 5 个合法框 ✓ |

### 6.3 对 16GB 单卡的结论

- **seq=2048 已经顶格**（16.1/16GB），正式训练建议 seq≤1536~2048、`max_num_tokens` 同值，必要时加 CPU offload；
- 8K/16K 长上下文训练在 16GB 单卡**不可行**（官方 16K 需 8×H100 或 Hopper/Blackwell 数据中心卡 + MagiAttention）；
- 速度参考：~13s/step@2048 → 1,000 steps ≈ 3.6h，5,000 steps ≈ 18h（后台可跑）；
- 全参 SFT 单卡不可行（静态就需要 3.52B×(2+2+12)B ≈ 17GB+，还没算激活）。

---

## 7. 本机（5060 Ti 16GB）推荐技术方案

```bash
# 激活 SFT 环境（已 clone 完成）
conda activate locate_anything_sft

# 训练（在 Eagle/Embodied 目录下执行）
LAUNCHER=pytorch python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=29511 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path /home/xu/models/LocateAnything-3B \
  --meta_path <你的 recipe.json> \
  --output_dir work_dirs/locany_lora_sft \
  --attn_implementation sdpa \          # 消费者 Blackwell 用 sdpa（magi 需 Hopper/数据中心）
  --block_size 6 --causal_attn False \
  --freeze_llm True --freeze_backbone True --freeze_mlp False \
  --use_llm_lora 64 --use_backbone_lora 0 \
  --max_seq_length 2048 --max_num_tokens_per_sample 2048 --max_num_tokens 2048 \
  --grad_checkpoint True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_steps 5000 --learning_rate 2e-5 --warmup_steps 200 --lr_scheduler_type cosine \
  --save_strategy steps --save_steps 200 --save_total_limit 3 \
  --deepspeed /home/xu/lora_smoke/ds_z1_torchadam.json \   # 见 §9.4 的 torch_adam 方案
  --bf16 True --report_to tensorboard
```

参数速查：`r=64` 官方默认；数据 <1k 条建议 `r=16~32` 防过拟合；`lr` 官方 2e-5；`MAX_STEPS` 按数据量决定（先小跑验证再放大）。

训练产物即 HF 格式完整模型目录，直接用现有 `scripts/infer/infer.py --model <输出目录>` 推理验证。

---

## 8. 建议路线图（下一步）

1. **数据**：收集/标注业务数据（detect/ground/point 均可），转 JSONL + recipe（无 BOM UTF-8）；建议先 200~1000 条验证闭环；
2. **冒烟**：小数据 50~200 steps 跑通，确认 loss 下降、推理输出格式正确（本次已验证 pipeline 可用）；
3. **正式训练**：后台长跑（参考 §7 命令），TensorBoard 监控；按需调 r/seq/lr；
4. **评估**：训练前后用同一评测集跑 `Eagle/Embodied/evaluation/` 的脚本（COCO/LVIS/Grounding/SSPro），对比 F1 等指标量化提升；
5. **上线**：微调产物 + 现有 `scripts/infer/infer.py`/批量推理/（可选 FastAPI 服务封装）。

---

## 9. 踩坑清单（本机特有，务必看）

1. **缺依赖**：clone 后训练还缺 `pynvml`(nvidia-ml-py)、`sortedcontainers`、`tensorboard`（已装进 `locate_anything_sft`）：
   ```bash
   pip install --no-cache-dir --timeout 30 --retries 2 nvidia-ml-py sortedcontainers tensorboard
   ```
2. **UTF-8 BOM**：Windows 侧 `Set-Content -Encoding UTF8` 写出的 JSON/JSONL 带 BOM，训练脚本 `json.loads` 直接报错——用无 BOM UTF-8 写，或在 WSL 内用 python 写。
3. **路径解析**：recipe 里 `annotation`/`root` 相对路径是相对**当前工作目录**解析，不是 recipe 所在目录——建议写绝对路径。
4. **DeepSpeed 0.15.4 + Blackwell (sm_120) JIT 编译 bug（关键）**：`compute_capability_args()` 里 `num = cc[0]+cc[2]` 把 `"12.0"` 解析成 `"1."`，导致 `nvcc fatal: Unsupported gpu architecture 'compute_1.'`，FusedAdam 编译失败。
   - 快速解法：deepspeed config 优化器加 `"torch_adam": true`（用 torch.optim.AdamW，跳过 FusedAdam JIT）；
   - 或给 deepspeed 打补丁修正 2 位主版本号解析；
   - 顺带：`TORCH_CUDA_ARCH_LIST` 在 JIT 模式下会被 deepspeed 主动 stash/清空，靠它救不了这个 bug。
5. **magi 注意力**：官方训练默认 `--attn_implementation magi`，仅支持 Hopper/Blackwell **数据中心**卡；本机（消费级 Blackwell sm_120）用 `sdpa`，且长上下文（>4K）训练不支持——与之前推理报告结论一致。

---

## 10. 参考

- 官方训练文档：`Eagle/Embodied/document/TRAINING.md`、`DATA_PREPARATION.md`
- LoRA 实现：`Eagle/Embodied/eaglevl/model/locany/modeling_locateanything.py`（`wrap_llm_lora`/`wrap_backbone_lora`）
- 训练主程序：`Eagle/Embodied/eaglevl/train/locany_finetune_magi_stream.py`
- 官方脚本：`Eagle/Embodied/shell/locate-anything-lora-visual-prompt.sh` / `locate-anything-streaming.sh`
- 复现本次 smoke：WSL `~/lora_smoke/`（recipe/data/ds_z1_torchadam.json/run.log/checkpoints）
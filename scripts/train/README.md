# LoRA / SFT 训练脚本

LocateAnything LoRA 微调启动与监控脚本，参数依据 `reports/05_lora_sft_research.md`（已在 RTX 5060 Ti 16GB 端到端验证）。

## 文件

| 文件 | 说明 |
|---|---|
| `train_lora.sh` | 训练启动：校验输入 → 激活 conda → 组装并执行 torchrun 训练命令，日志写入输出目录 |
| `train_watch.sh` | 监控：tail/loss/GPU/tensorboard/checkpoint 查看 |
| `ds_z1_torchadam.json` | DeepSpeed ZeRO-1 + `torch_adam:true`（绕开 DeepSpeed 0.15.4 在 Blackwell sm_120 上的 FusedAdam JIT 编译 bug） |

## 快速开始

```bash
# 1) 训练（recipe 的 annotation/root 建议用绝对路径；JSONL 需无 BOM UTF-8）
bash scripts/train/train_lora.sh     --meta /home/xu/data/recipe.json     --output /home/xu/work_dirs/locany_lora_v1     --steps 5000 --save-steps 200

# 2) 监控（另开终端）
bash scripts/train/train_watch.sh /home/xu/work_dirs/locany_lora_v1 --follow   # 日志
bash scripts/train/train_watch.sh /home/xu/work_dirs/locany_lora_v1 --gpu      # GPU
bash scripts/train/train_watch.sh /home/xu/work_dirs/locany_lora_v1 --tensorboard
bash scripts/train/train_watch.sh /home/xu/work_dirs/locany_lora_v1 --checkpoints

# 3) 训练产物即 HF 格式模型目录，直接用推理 CLI 验证
python scripts/infer/infer.py --model /home/xu/work_dirs/locany_lora_v1     --image test.jpg --task detect --query "cat"
```

## 默认配置（可覆盖）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rank` / `RANK` | 64 | LLM LoRA rank（0=关闭） |
| `--backbone-rank` / `BACKBONE_RANK` | 0 | Vision LoRA rank |
| `--freeze-mlp` / `FREEZE_MLP` | False | 是否冻结 MLP 连接器 |
| `--seq` / `MAX_SEQ` | 1536 | max_seq_length（默认 1536 留显存余量；2048 为实测 16GB 顶格，无余量） |
| `--steps` / `MAX_STEPS` | 5000 | 总训练步数 |
| `--lr` / `LR` | 2e-5 | 学习率 |
| `--save-steps` / `SAVE_STEPS` | 200 | checkpoint 间隔 |
| `--attn` / `ATTN` | sdpa | 消费级卡用 sdpa；magi 需 Hopper/数据中心 Blackwell |
| `--gpus` / `GPUS` | 1 | GPU 数 |
| `MODEL_PATH` | /home/xu/models/LocateAnything-3B | 基座模型 |
| `CONDA_ENV` | locate_anything_sft | conda 环境 |

## 断点续训

输出目录存在 `checkpoint-*` 时 `train_lora.sh` 自动续训（streaming 数据顺序 bit-wise 恢复）。若目录非空但无 checkpoint，脚本会拒绝启动（防误覆盖），用 `--overwrite` 强制重建。

> 注意：训练脚本在输出目录写入 `done.txt` 标记完成；若存在 `done.txt`，再次启动会直接退出。需要继续训练时先 `rm <out>/done.txt`（launcher 会给出提示）。

## 2026-08-05 经验教训（务必遵守）

1. **显存余量**：seq=2048 训练峰值 ~15-16GB，已顶格 16GB，会挤压 Windows 宿主显存（并发任何 GPU 进程可能把宿主搞崩）。默认已调低到 `MAX_SEQ=1536`；若必须 2048，训练期间**禁止并发其他 GPU 任务**。
2. **样本丢弃**：单样本视觉 token 超过 `MAX_SEQ` 的图片会被训练管线静默丢弃（日志出现 `idx N failed: image token mismatch`）。1080p 帧需要 ~2500-4600 token，seq=2048 时基本进不了训练。要对高分辨率图做 LoRA，先降采样（如 ≤1280px）再训练，或提高 seq（需更大显存）。
3. **推理回归自动修复**：训练会把 `tokenizer.model_max_length` 烤成训练 seq（如 2048），导致微调产物在 1080p 大图上推理截断。`train_lora.sh` 训练成功后会自动把 `model_max_length` 恢复为基座模型的值。
4. **fast/hybrid 解码回归**：在 seq=2048 且没见到高分辨率两轮车数据的情况下，微调模型的 MTP 路径（fast/hybrid）在 1080p 大图上可能截断/乱码；`slow`（纯 AR）模式输出正常。涉及大图部署时优先 slow 或把大图纳入训练。

## 常见问题

- **DeepSpeed `compute_1.` 编译错误**：DeepSpeed 0.15.4 对两位数算力（sm_120）解析有 bug，本仓库用 `ds_z1_torchadam.json`（`torch_adam:true`）规避；不要依赖 `TORCH_CUDA_ARCH_LIST`。
- **UTF-8 BOM**：JSONL/recipe 必须无 BOM（Windows `Set-Content -Encoding UTF8` 会带 BOM）。
- **OOM**：降低 `--seq`（如 1536/1024）、`--save-total-limit`，或把 `packing_buffer_size` 调小。
- **数据格式**：坐标用 `<n>` token（[0,1000] 归一化），详见 `Eagle/Embodied/document/DATA_PREPARATION.md`。

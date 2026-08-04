# LocateAnything 环境巩固：权重本地化 + FlashAttention + la_flash 验证 + 环境导出

- 日期：2026-08-04
- 目标：将预研阶段的「可选」待办落实为可复现的最终推理基线（对应 reports/02 待办 A 节）

---

## 1. 权重拷贝到 WSL ext4

| 项 | 值 |
|---|---|
| 源 | `/mnt/c/Data/LocateAnything-3B`（HF 下载副本，7.3 GB，48 文件） |
| 目标 | `~/models/LocateAnything-3B`（WSL ext4） |
| 校验 | 文件数 48=48；safetensors 两个分片字节数一致（4,959,632,160 / 2,701,795,216） |
| 效果 | 模型加载 **31-42s → 2.9s**（约 10-14×），规避 drvfs 小文件 IO 开销 |

## 2. 使能 FlashAttention（Vision=FA2，LLM=sdpa）

### 2.1 安装方式（关键决策：不再源码编译）

- 安装 `flash-attn==2.8.3+cu.13.0.torch.2.9`（cp310 / manylinux_2_24_x86_64），来自 **Astral cu130 预编译 wheel 索引**：
  `pip install "flash-attn==2.8.3+cu.13.0.torch.2.9" --index-url https://wheels.astral.sh/simple/cu130/ --no-deps`
- 配套：conda 环境内 `cuda-toolkit=13.0.3`（nvcc 13.0，与 torch 2.9.0+cu130 匹配）
- 内核验证：`flash_attn_func` 在 RTX 5060 Ti（sm_120）上正常返回 `(2,256,8,64) fp16`

### 2.2 源码编译的坑（记录，避免重蹈）

源码编译 `flash_attn-2.8.3.post1` 在本机 WSL2 反复失败：
1. nvcc 12.8 与 torch cu130 不匹配（`CUDA_MISMATCH`）→ 升级 cuda-toolkit 13.0；
2. 即便 nvcc 13.0，编译时大量 `cicc` 子进程（每个 ~1GB RSS）触发 WSL **全局 OOM**（dmesg 可见 `oom-kill task=cicc global_oom`），并连带 WSL 服务进入 `Wsl/Service/E_UNEXPECTED` 状态，需 `wsl --shutdown` 恢复；
3. 结论：本机（WSL2 + 16GB 显卡 + 48GB VM 上限）**直接使用预编译 wheel**，不要源码编译 flash-attn。

### 2.3 模型侧注意力配置（重要修复）

- MoonViT 视觉编码器：`modeling_locateanything.py` 默认 `vision_attn_impl = ... or 'flash_attention_2'`，装好 flash-attn 后自动走 **FA2**；
- Qwen2.5 LLM：config 请求 `magi`，原回退链为 `magi → flash_attention_2`，但自定义 `modeling_qwen2.py` 的 `forward()` 只实现 magi/sdpa（FA2 会 `NotImplementedError`）；
- **修复**：将回退改为 `magi → sdpa`（已同步 patch `~/models/LocateAnything-3B/modeling_qwen2.py` 与 `/mnt/c/Data/LocateAnything-3B/modeling_qwen2.py`，并清理 HF remote-code 缓存）。
- 最终运行态：`LLM=sdpa`，`Vision=flash_attention_2`。

### 2.4 提速效果（bus.jpg detect person/bus/car）

| 指标 | 之前（纯 sdpa，/mnt/c 加载） | 之后（Vision=FA2，ext4 加载） |
|---|---:|---:|
| 模型加载 | 31-42 s | 2.9 s |
| detect 生成时延 | ~3.0 s | 1.45 s |
| BPS（框/秒） | 2.0 | 4.1 |
| prefill | ~2.46 s | 1.01 s |

> 速度提升主要来自 Vision 编码器 FA2（prefill 减半）与权重本地化（加载 10× 以上）。

## 3. la_flash 批处理验证（batch_utils 运行时）

命令：`batch_infer.py --model ~/models/LocateAnything-3B --attn {sdpa|la_flash} --scheduler {eager|pipeline} --batch-size {1|4} --max-new-tokens 8192`
请求集 4 条：bus.jpg ×3（detect/ground/7 类多目标）+ dense_text.png ×1（OCR）。

| case | attn | scheduler | batch | 墙钟(s) | 峰值显存(MB) | 结果 |
|---|---:|---|---:|---:|---:|---|
| sdpa_b1 | sdpa | eager | 1 | 17.5 | 11679 | 4 行有效 |
| laflash_b1 | la_flash | pipeline | 1 | 170.6 | 11231 | 4 行有效 |
| laflash_b1e | la_flash | eager | 1 | 34.5 | — | 4 行有效 |
| laflash_b4 | la_flash | pipeline | 4 | 22.4 | 12426 | 4 行有效 |
| sdpa_b4 | sdpa | eager | 4 | 23.3 | 14552 | 4 行有效 |

- **正确性**：la_flash 输出与 sdpa 一致（同数量同近似坐标，仅采样噪声）；
- **显存**：batch=4 时 la_flash 比 sdpa 省 ~2.1GB（12426 vs 14552MB，≈14%）；batch=1 省 ~0.4GB；
- **速度**：batch≥4 时 la_flash ≈ sdpa（22.4 vs 23.3s）；**batch=1 + pipeline 调度器病态慢**（170.6s，range-plan 每行开销大），单行请用 eager 调度器（34.5s，仍比 sdpa 慢 ~2×）；
- **结论**：本机（4K token 级短上下文、16GB 显存）默认 sdpa 足够；la_flash 的价值在**大 batch / 长上下文**时压低显存（README 声称 A100 上 sdpa 35GB→la_flash 11.7GB），适合批量服务化场景。

## 4. 最终 conda 环境导出

- 输出：`environment.yml`（项目根目录，348 行，含 pip 段 157 个包）
- 关键版本：`python 3.10.20`、`torch 2.9.0+cu130`、`transformers 4.57.1`、`deepspeed 0.15.4`、`liger-kernel 0.3.1`、`flash-attn 2.8.3+cu.13.0.torch.2.9`、`cuda-toolkit 13.0.3`（nvcc 13.0）
- 复现：`conda env create -f environment.yml`（CUDA 侧依赖 torch 自带 cu130 wheel；nvcc 仅编译/训练需要）

## 5. WSL 资源与稳定性备注

- `~/.wslconfig`：`[wsl2] memory=48GB`（用户调整，Windows 保留 ~16GB）+ 默认 12GB swap；
- flash-attn 源码编译曾触发 WSL 全局 OOM 与 `Wsl/Service/E_UNEXPECTED`；改用预编译 wheel 后无需重型编译，环境稳定。

## 6. 结论

1. 权重本地化 + Vision FA2 已构成**最终推理基线**：加载 2.9s、detect 1.45s、4.1 BPS（相对预研阶段 ~2× 提速）。
2. la_flash 路径验证通过，作为大 batch/长上下文的可选后端；单行用 eager 调度器。
3. `environment.yml` 已留存复现基线；本报告对应待办 A 节全部闭环。

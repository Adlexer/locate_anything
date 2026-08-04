# LocateAnything 待办与建议（预研阶段收尾）

- 日期：2026-08-04
- 状态：等待后续任务指令

---

## 一、待办（TODO）

### A. 环境/工程
- [ ] （建议）把权重从 `/mnt/c/Data/LocateAnything-3B` 拷贝到 WSL ext4（如 `~/models/LocateAnything-3B`），加载时间可从 31-42s 降到 ~10s 内；同时规避 drvfs 对大量小文件（tokenizer/json）的 IO 开销
- [ ] （可选）安装 flash-attn（需适配 torch 2.9 / sm_120 的版本），让 MoonViT 走 FA2 而非 sdpa；不装则保持 sdpa 回退
- [ ] （可选）验证 `la_flash` 批处理路径（`batch_infer.py --attn la_flash --scheduler pipeline`）在 5060 Ti 上的显存/吞吐表现（README 声称 A100 上 sdpa 35GB→la_flash 11.7GB）
- [ ] （可选）固定 conda 环境导出：`conda env export > environment.yml` 留存复现基线
- [ ] （可选）把 `scripts/` 下的冒烟脚本（smoke_test.py / smoke_test_real.py / render_annotated.py）整理为可复用的 `infer.py` CLI（输入图+query，输出 boxes/points + 标注图）

### B. 推理/部署
- [ ] 用真实业务图/视频做一轮更广的验证：dense detection、OCR（detect_text）、GUI grounding（ground_gui box/point）、layout（doc）、多类别 `</c>` 长 query
- [ ] 对 `generation_mode`（fast/slow/hybrid）与 `max_new_tokens`（官方建议 8192）做速度-精度对比
- [ ] （可选）基于 `locateanything_worker.py` 封装 FastAPI/HTTP 服务（README 已指明可嵌入任何 serving 框架）

### C. 微调（如需）
- [ ] 准备 JSONL + recipe：按 `document/DATA_PREPARATION.md` 构造检测/grounding/OCR 数据（坐标用 `<0>..<1000>` token）
- [ ] 小规模 LoRA 验证：`shell/locate-anything-lora-visual-prompt.sh`（MODEL_PATH 指向本地权重；USE_LLM_LORA=64，冻结 LLM+backbone 只训 MLP）
- [ ] 注意：本机只能用 `--attn_implementation sdpa --max_seq_length 4096`（Magi 需 Hopper/Blackwell）；大 batch 需降低 max_num_tokens 防 OOM（16GB 显存）
- [ ] visual prompt 能力官方权重不支持，需用 `visual_prompt=true` 数据自行微调

### D. 评估复现（如需）
- [ ] 下载 Rex-Omni-EvalData / ScreenSpot-Pro，装 fastevaluate（`evaluation/fastevaluate` C++ 模块）
- [ ] 跑 `eval_coco.sh / eval_lvis.sh / eval_grounding.sh / eval_sspro.sh`，与官方 RESULTS 表对比

---

### E. Infra 自研优化（可选，需改模型代码）
- [ ] **MTP 适配 FA2（LLM 侧启用 Flash Attention）**：接通 `Qwen2Model.forward` 的 `flash_attention_2` 分支，让 LLM 解码（含 MTP 多 token 预测）走 FA2，替代当前 sdpa 回退
  - 背景：官方 `modeling_qwen2.py` 的 `QWEN2_ATTENTION_CLASSES` 已映射 `"flash_attention_2" -> Qwen2FlashAttention2`，但 `Qwen2Model.forward`（~L1321-1335）只实现 `magi`/`sdpa`，其余直接 `raise NotImplementedError`。装上 flash-attn 后，magi 回退会落到 FA2 并在推理时崩溃；当前已把 LLM 回退固定为 sdpa（Vision 保持 FA2，已验证 2x 提速）。
  - 改动点（modeling_qwen2.py）：
    - `Qwen2Model.forward` 增加 `flash_attention_2` 分支，把 MTP 的 block-diffusion 掩码（prefix 因果 + 扩散窗口内双向，`causal_attn=False`）与 `position_ids`（mask 位置 -1）正确喂给 `Qwen2FlashAttention2`
    - 核对 `prepare_inputs_for_generation` 的 KV 缓存裁剪（每步截到 `generated.shape[1]`），确保 FA2 与 sdpa 的缓存语义一致
    - 掩码可用 FA2 支持的 4D `attention_mask` / `is_causal` 承载；参考 `mask_sdpa_utils.py` 的等价掩码构造
    - 本机已具备 `-gencode arch=compute_120,code=sm_120`（flash-attn 2.8.3+cu13 轮子含 Blackwell kernel）
  - 验收标准：
    - 加载后 `model.language_model.model._attn_implementation == "flash_attention_2"` 且无 NotImplementedError
    - bus.jpg detect / ground / point 输出与 sdpa 参考一致（坐标逐项相等或 IoU>=0.99）；dense_text OCR 长输出 box 数量一致
    - 速度可测提升：重点看 prefill 与长上下文/大 batch（batch=1 短上下文下 FA2 收益有限）
    - 回归：generation_mode fast/slow/hybrid x max_new_tokens 512/2048/8192 全跑通
  - 风险：FA2 对自定义 MTP 掩码的支持需自行验证；若 FA2 数值路径与 sdpa 有微小差异，需确认不影响 PBD 坐标解码；改动为本地 patch，建议 `git init` 后用 diff 管理

---

## 二、建议（Suggestions）

1. **环境优先级**：先做"权重拷入 WSL ext4"（收益最大、零风险），再决定是否上 flash-attn / la_flash。
2. **本机能力边界**：RTX 5060 Ti（sm_120, 16GB, 无 NVLink）：
   - 推理：单卡 sdpa 短上下文完全够用；长图（in_token_limit 25600 上限）要注意显存，建议先用 4K token 级图片。
   - 训练：仅支持 sdpa ~4K 上下文的小规模 SFT/LoRA；全参 25K 步的大规模训练不现实（官方 8×H100）。
3. **首选路线**（若目标是"尽快用起来"）：真实数据推理 → 需要领域适配时做 LoRA（batch 1、max_num_tokens 4-8K、ZeRO-1）→ 用 la_flash 批处理提吞吐。
4. **风险提示**：
   - `transformers==4.57.1` 是硬性版本（新版本 API 变化可能导致 remote code 失效）；升级前先验证。
   - deepspeed JIT 编译算子依赖环境内 nvcc（已装 12.8）；若以后换机器/环境，需重装 cuda-toolkit 或设 DS_BUILD_OPS=0。
   - 官方权重为 NVIDIA 非商用 License（学术/非营利研究可用，商用禁止）；Qwen2.5 为 Qwen 研究许可、MoonViT 为 MIT，注意合规。
5. **代码整洁**：当前 Embodied 非 git 仓库，建议 `git init` 纳入版本管理，便于后续微调 diff 追踪。
6. **复现基线**：训练脚本 `trainer_state.json` 显示官方 3B 权重 = 5000 步 streaming SFT（loss ~0.0228，8×H100 约 59 分钟），可作为下游微调起点参考。

---

## 三、下一步建议的默认动作

在用户未给出新指令前，预研阶段到此收尾；后续若继续，建议从 **B 节（真实业务图批量验证 + 服务封装）** 或 **C 节（LoRA 微调）** 二选一推进。
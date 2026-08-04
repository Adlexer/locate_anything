# scripts 目录说明

按用途整理的可复用脚本集（2026-08-04 整理）。运行环境：WSL + `locate_anything` / `locate_anything_sft` conda 环境。

## 目录结构

| 目录 | 用途 | 内容 |
|---|---|---|
| `scripts/infer/` | 推理 | `infer.py`（主力 CLI，7 类任务 → boxes/points JSON + 标注图）、`render_annotated.py`（标注图渲染） |
| `scripts/bench/` | 基准测试 | `bench_gen_mode.py`（generation_mode×max_new_tokens）、`bench_la_flash.py`（la_flash）、`gen_mode_report.py`（生成报告 04）、`make_bench_assets.py`（基准素材） |
| `scripts/train/` | LoRA/SFT 训练 | `train_lora.sh`（训练启动）、`train_watch.sh`（训练监控）、`ds_z1_torchadam.json`（DeepSpeed ZeRO-1 + torch AdamW，绕 sm_120 JIT bug）、`README.md` |
| `scripts/env/` | 环境/FlashAttention | `build_fa.sh`、`install_fa_wheel.sh`、`check_fa_wheels.sh`、`list_hf_wheels.sh`、`verify_fa.sh`、`verify_fa_model.sh`、`verify_sdpa_llm.sh` |
| `scripts/smoke/` | 早期冒烟测试（历史保留） | `smoke_test.py`、`smoke_test_real.py` |

> 说明：早期遗留的 `smoke_test.b64` / `smoke_test_real.b64`（base64 传输临时产物）已删除，git 历史仍可追溯。

## 快速上手

```bash
# 推理（WSL，任意 conda 环境，需能 import eaglevl）
python /mnt/c/Dev/locate_anything/scripts/infer/infer.py     --image /path/to/img.jpg --task detect --query "person</c>car"     --out /mnt/c/Dev/locate_anything/outputs --generation-mode hybrid

# LoRA 训练 + 监控
bash /mnt/c/Dev/locate_anything/scripts/train/train_lora.sh     --meta <recipe.json> --output <out_dir> --steps 5000 --save-steps 200
bash /mnt/c/Dev/locate_anything/scripts/train/train_watch.sh <out_dir> --follow
```

## 路径迁移记录

- 2026-08-04 整理：`scripts/infer.py` → `scripts/infer/infer.py`；bench 四件套 → `scripts/bench/`；FA 工具 → `scripts/env/`；smoke → `scripts/smoke/`；删除 `.b64` 临时文件。
- 旧路径在 reports 04/05 中的引用已同步更新；progress/result_report 历史条目保持原样（append 新记录说明）。

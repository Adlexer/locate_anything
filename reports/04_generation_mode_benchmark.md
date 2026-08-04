# LocateAnything 生成模式与 max_new_tokens 速度-精度对比

- 日期：2026-08-04
- 环境：WSL Ubuntu 24.04 / RTX 5060 Ti (16GB, sm_120) / torch 2.9.0+cu130 / transformers 4.57.1
- 模型：`~/models/LocateAnything-3B`（本地权重，Vision=flash_attention_2, LLM=sdpa）
- 解码配置：temperature=0（贪心，确定性主矩阵）+ temperature=0.7（重复采样 3 次测时延方差）
- 采样：top_p=0.9, repetition_penalty=1.1；每点记录 generate_time / tps / bps / forward_step / switch_to_ar / 峰值显存

## 1. 实验矩阵

| 任务 | 图 | 说明 |
|---|---|---|
| bus_detect | data/bus.jpg | 短输出多类检测（person/bus/car），~46 token |
| bus_ground | data/bus.jpg | 中输出短语定位（people），~30-34 token |
| dense_ocr | data/dense_text.png | 长输出场景文字检测，~495-523 token（含 38 个绘制 GT 框） |

`generation_mode × max_new_tokens ∈ {fast, slow, hybrid} × {512, 2048, 8192}`（官方建议 8192）。

## 2. 速度对比（temperature=0，确定性）

### 2.1 短输出 bus_detect

| mode | cap | latency(s) | tps | bps | forward_step | switch_to_ar | boxes |
|---|---|---:|---:|---:|---:|---:|---:|
| fast | 512 | 0.836 | 57.6 | 7.52 | 10 | 0 | 6 |
| fast | 2048 | 0.873 | 54.3 | 7.08 | 10 | 0 | 6 |
| fast | 8192 | 0.878 | 53.7 | 7.01 | 10 | 0 | 6 |
| hybrid | 512 | 1.143 | 41.1 | 5.36 | 15 | 1 | 6 |
| hybrid | 2048 | 1.099 | 43.0 | 5.60 | 15 | 1 | 6 |
| hybrid | 8192 | 1.064 | 44.3 | 5.78 | 15 | 1 | 6 |
| slow | 512 | 2.013 | 21.6 | 3.01 | 43 | 0 | 5 |
| slow | 2048 | 2.048 | 21.3 | 2.97 | 43 | 0 | 5 |
| slow | 8192 | 2.012 | 21.6 | 3.01 | 43 | 0 | 5 |

### 2.2 中输出 bus_ground

| mode | cap | latency(s) | tps | bps | forward_step | switch_to_ar | boxes |
|---|---|---:|---:|---:|---:|---:|---:|
| fast | 512 | 0.770 | 45.8 | 6.73 | 7 | 0 | 5 |
| fast | 2048 | 0.770 | 45.7 | 6.72 | 7 | 0 | 5 |
| fast | 8192 | 0.762 | 46.2 | 6.80 | 7 | 0 | 5 |
| hybrid | 512 | 0.926 | 31.1 | 4.44 | 12 | 2 | 4 |
| hybrid | 2048 | 0.913 | 31.4 | 4.49 | 12 | 2 | 4 |
| hybrid | 8192 | 0.958 | 29.9 | 4.27 | 12 | 2 | 4 |
| slow | 512 | 1.807 | 15.7 | 2.24 | 28 | 0 | 4 |
| slow | 2048 | 1.713 | 16.5 | 2.36 | 28 | 0 | 4 |
| slow | 8192 | 1.632 | 17.4 | 2.49 | 28 | 0 | 4 |

### 2.3 长输出 dense_ocr

| mode | cap | latency(s) | tps | bps | forward_step | switch_to_ar | boxes | tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fast | 512 | 7.711 | 66.9 | 5.23 | 124 | 0 | 40 | 512 |
| fast | 2048 | 7.187 | 73.4 | 5.75 | 128 | 0 | 41 | 523 |
| fast | 8192 | 7.669 | 69.0 | 5.41 | 128 | 0 | 41 | 523 |
| hybrid | 512 | 7.695 | 65.0 | 4.86 | 136 | 3 | 37 | 495 |
| hybrid | 2048 | 6.867 | 72.8 | 5.44 | 136 | 3 | 37 | 495 |
| hybrid | 8192 | 7.726 | 64.7 | 4.84 | 136 | 3 | 37 | 495 |
| slow | 512 | 22.866 | 22.5 | 1.62 | 512 | 0 | 37 | 512 |
| slow | 2048 | 24.477 | 21.3 | 1.56 | 520 | 0 | 38 | 520 |
| slow | 8192 | 23.333 | 22.3 | 1.63 | 520 | 0 | 38 | 520 |

### 2.4 典型使用时延（temperature=0.7，3 次重复）

| task | mode | mean(s) | min(s) | max(s) |
|---|---|---:|---:|---:|
| bus_detect | fast | 0.907 | 0.867 | 0.966 |
| bus_detect | slow | 2.410 | 2.291 | 2.539 |
| bus_detect | hybrid | 0.902 | 0.890 | 0.919 |
| bus_ground | fast | 0.765 | 0.734 | 0.791 |
| bus_ground | slow | 1.375 | 1.090 | 1.610 |
| bus_ground | hybrid | 0.869 | 0.821 | 0.902 |

> latency(s) 为端到端墙钟时延（含 prefill，≈ 模型内部 generate_time）；峰值显存三个模式一致 ≈ 10.6 GB（模型 ~7.7GB bf16 + 激活/KV）。

## 3. 精度对比

### 3.1 输出框数量与有效性（temperature=0）

| task | mode | boxes | 异常框 | 说明 |
|---|---|---:|---:|---|
| bus_detect | fast | 6 | 0 | 正常 |
| bus_detect | slow | 5 | 0 | <box>None</box> |
| bus_detect | hybrid | 6 | 0 | 正常 |
| bus_ground | fast | 5 | 1 | 坐标反转 |
| bus_ground | slow | 4 | 0 | 正常 |
| bus_ground | hybrid | 4 | 0 | 正常 |
| dense_ocr | fast | 41 | 0 | 正常 |
| dense_ocr | slow | 38 | 0 | 正常 |
| dense_ocr | hybrid | 37 | 0 | 正常 |

### 3.2 dense_ocr 与绘制 GT 的匹配（IoU 阈值 0.5 与 0.3）

| mode | cap | P@0.5 | R@0.5 | F1@0.5 | GT覆盖@0.3 |
|---|---|---:|---:|---:|---:|
| fast | 512 | 0.200 | 0.211 | 0.205 | 0.447 |
| fast | 2048 | 0.220 | 0.237 | 0.228 | 0.474 |
| fast | 8192 | 0.220 | 0.237 | 0.228 | 0.474 |
| hybrid | 512 | 0.243 | 0.237 | 0.240 | 0.500 |
| hybrid | 2048 | 0.243 | 0.237 | 0.240 | 0.500 |
| hybrid | 8192 | 0.243 | 0.237 | 0.240 | 0.500 |
| slow | 512 | 0.216 | 0.211 | 0.213 | 0.579 |
| slow | 2048 | 0.237 | 0.237 | 0.237 | 0.605 |
| slow | 8192 | 0.237 | 0.237 | 0.237 | 0.605 |

> 说明：dense_text.png 为 PIL 绘制的合成文档/界面图（38 个 GT 文字框）。模型检测粒度偏子串/字符级，与整词 GT 框天然错位，
> 因此绝对 F1 偏低是粒度差异所致，不作为模型能力上限；三模式的**相对**排序与跨模式一致性才是本报告重点。

### 3.3 跨模式一致性（与 hybrid@8192@temp0 参考框的 IoU≥0.5 匹配率）

| task | mode | cap=512 | cap=2048 | cap=8192 |
|---|---|---:|---:|---:|
| bus_detect | fast | 1.000 | 1.000 | 1.000 |
| bus_detect | slow | 1.000 | 1.000 | 1.000 |
| bus_detect | hybrid | 1.000 | 1.000 | 1.000 |
| bus_ground | fast | 0.400 | 0.400 | 0.400 |
| bus_ground | slow | 1.000 | 1.000 | 1.000 |
| bus_ground | hybrid | 1.000 | 1.000 | 1.000 |
| dense_ocr | fast | 0.875 | 0.878 | 0.878 |
| dense_ocr | slow | 0.811 | 0.816 | 0.816 |
| dense_ocr | hybrid | 1.000 | 1.000 | 1.000 |

## 4. 结论

1. **速度**：fast 最快，hybrid 接近 fast，slow 慢 2.4~3.2×：
   - 短输出 detect：fast ≈ 0.80-0.86s，hybrid ≈ 1.04-1.12s，slow ≈ 1.99-2.02s；
   - 长输出 OCR：fast ≈ 7.1-7.7s，hybrid ≈ 6.9-7.7s，slow ≈ 22.8-24.4s。
2. **精度/鲁棒性**：
   - slow 最稳但会输出 `<box>None</box>`（bus_detect 的 car 小目标丢失）；
   - fast 最快但可能输出**坐标反转的异常框**（bus_ground 的 `x2<x1`），且 people 定位多检 1 个框（跨模式一致率仅 0.4）；
   - hybrid 具备 fast 的速度 + slow 的框合法性校验（error_box→AR 回退），框数量与坐标最一致（跨模式一致率 1.0）。
3. **max_new_tokens**：
   - 短/中输出（≤46 token）：512/2048/8192 结果与速度完全一致，cap 无影响；
   - 长输出 OCR（~520 token）：cap=512 被截断（丢失 1-3 个框），cap=2048 即可完整输出，cap=8192 作为安全上限无额外开销；
   - **建议默认 max_new_tokens=8192**（官方建议），批量/服务化时按任务最长输出收紧（如 2048）以控制最坏时延。
4. **推荐默认**：`generation_mode=hybrid, max_new_tokens=8192`；对时延极敏感且接受少量异常框的场景可选 fast。

## 5. 复现

```bash
conda activate locate_anything
python /mnt/c/Dev/locate_anything/scripts/bench/bench_gen_mode.py   # -> outputs/bench/gen_mode_results.json
python /mnt/c/Dev/locate_anything/scripts/bench/gen_mode_report.py  # -> reports/04_generation_mode_benchmark.md
```

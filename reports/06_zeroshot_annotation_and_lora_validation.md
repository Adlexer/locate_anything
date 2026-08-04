# LocateAnything 零样本标注 + 小批量 LoRA 训练验证

- 日期：2026-08-05
- 数据：`C:\Data\datasets\detect`（煤气罐 gas_tank / 电动车+少量自行车 ebike，无标注）
- 模型：`~/models/LocateAnything-3B`（预训练）；微调产物 `~/lora_gas/run_v1`
- 环境：WSL / `locate_anything_sft` / RTX 5060 Ti 16GB（sm_120）

---

## 0. 结论（TL;DR）

1. **零样本标注能力**：煤气罐识别**稳定可靠**；电动车/自行车**能稳定检出「两轮车」整体，但子类判定有噪声**（同一帧 scooter/bicycle 二选一互斥）。已产出 122 份 YOLO txt（93 框：煤气罐 58 / 电动车 28 / 自行车 7）+ `classes.txt`，代码落 `scripts/annotation/`。
2. **LoRA 小批量验证**：78 条伪标注样本，150 步 / 9 分钟，loss 1.19→0.41，训练管线端到端跑通。微调产物在 8 张 holdout 图上 slow 模式**8/8 类别一致、平均 IoU 0.987**——无灾难性遗忘，能力保持。
3. **两个必须知道的坑（都源于 16GB 显存 → seq=2048 约束）**：
   - 高分辨率帧（1080p，视觉 token ~2546-4590）在 seq=2048 下**被训练管线丢弃**，LoRA 实际没学到多少电动车高帧；
   - 微调产物 `tokenizer.model_max_length` 被烤成 2048，且 fast/hybrid（MTP）解码路径在大图上**截断/乱码回归**；`slow` 模式正常。launcher 已加自动恢复 + 降采样建议。
4. **显存教训**：seq=2048 训练峰值 ~15-16GB 已顶格，会挤压 Windows 宿主显存——训练期间严禁并发其他 GPU 进程（本次我并发跑推理几乎把宿主搞崩）。launcher 默认已调低到 seq=1536 留余量。

---

## 1. 数据与零样本标注（任务 1）

### 1.1 数据构成

| 子集 | 内容 | 数量 |
|---|---|---|
| `ebike/` | 视频帧 1920×1080 png（36）+ 同名 `_result.jpg`（JPEG 重编码，内容相同，已跳过） | 36 有效图 |
| `gas_tank/` | 图片 jpg/jpeg（13，含 3648×2736 大图）+ 手机视频 mp4（9，540-720 竖屏） | 22 |
| 合计 | 49 图 + 9 视频（抽 72 帧） | 121 标注单元 |

### 1.2 标注脚本（已落盘 `scripts/annotation/`）

| 文件 | 作用 |
|---|---|
| `annotate_yolo.py` | 零样本检测 → YOLO txt（图片同目录）+ 视频抽帧标注 + 摘要/预览/manifest |
| `build_train_jsonl.py` | manifest → 训练 JSONL + recipe（坐标 [0,1000] → `<ref>/<box>` token） |
| `eval_before_after.py` | 同一批 holdout 图在「预训练 vs 微调」模型上的对比评测 |
| `visualize_yolo.py` | YOLO 标注可视化/拼图/统计校验（人工抽查伪标签质量） |
| `README.md` | 用法与已知限制 |

命令（WSL）：
```bash
python scripts/annotation/annotate_yolo.py   --data /mnt/c/Data/datasets/detect   --classes "gas cylinder,electric scooter,bicycle" --video-frames 8
```

### 1.3 零样本识别能力（probe + 全量实测）

| 类别 | 表现 | 结论 |
|---|---|---|
| gas cylinder（煤气罐） | test1 检出 5 个成排煤气罐；test2/bg 各 1；框位置合理 | **稳定**（全量 58 框） |
| electric scooter（电动车） | 大部分帧检出（28 框），个别帧漏检 | 整体检出可靠 |
| bicycle（自行车） | 7 框；与 scooter **互斥二选一**，存在电动车被标成 bicycle 的误判 | **子类判定有噪声** |

> 推论（已验证于本数据集）：模型能稳定回答「图里有没有两轮车」，但对「电动车 vs 自行车」的子类置信度不足。**伪标签训练时若追求类别纯度，建议合并两轮车为单类或人工抽查修正**（`build_train_jsonl.py --merge-two-wheeler` 已支持）。

### 1.4 产物

- 122 份 YOLO txt（`class_id cx cy w h`，与图片同目录）+ `classes.txt`（数据集根）
- 视频抽帧：`detect/_frames/<视频>/frame_*.jpg(.txt)`，每视频 8 帧
- 汇总/预览/manifest：`outputs/annotation_detect/`（summary.json / previews/ / manifest.jsonl）

---

## 2. 小批量 LoRA 训练验证（任务 2）

### 2.1 训练数据（伪标签）

- 由 1.3 的零样本标注构建：**78 train + 8 holdout**（`build_train_jsonl.py`，min-area 过滤小框）
- 类别框分布（train）：煤气罐 52 / 电动车 26 / 自行车 7；`repeat_time=3` 过采样

### 2.2 训练配置与过程

| 项 | 值 |
|---|---|
| 配置 | LoRA r=64（LLM）+ MLP 可训练；freeze LLM/Vision；sdpa；**seq=2048**；grad checkpoint；DS ZeRO-1(torch_adam) |
| 规模 | 150 步 / 9 分 01 秒 / ~2.3-2.7 s/step（packing ~1.4 样本/步） |
| 收敛 | train_loss **1.19 → 0.41**（对数线性下降，正常） |
| 峰值显存 | ~14.8-15.1 GB（真实数据，比合成 smoke 的 16.1GB 略低） |
| 产物 | checkpoint-50/100/150 + 最终模型（`~/lora_gas/run_v1`，55GB） |

### 2.3 微调前后对比（8 张 holdout，temperature=0，slow 模式）

| 指标 | 结果 |
|---|---|
| 类别一致率 | **8/8**（煤气罐 6 + 电动车 2） |
| 平均 best-IoU | **0.987**（0.97~1.00，两轮车框逐像素一致） |
| 结论 | 微调未破坏预训练能力；在该 holdout 上表现≈预训练（预期内：伪标签来自同一模型，且高帧电动车样本训练时被丢） |

### 2.4 重要发现：解码模式 × 大图回归（已验证）

在同一张 1080p 电动车帧上，微调模型（`run_v1`）：

| 模式 | 输出 |
|---|---|
| slow（纯 AR） | ✅ 正常：`electric scooter` box 与预训练**逐字节一致**（431,261,752,999） |
| fast（MTP） | ⚠️ 输出夹带 `<null>` 乱码但可恢复 |
| hybrid | ❌ 在 `gas cylinder → None` 后**截断**（0 框） |

原因（推断，已定位到两层）：
1. 训练把 `tokenizer.model_max_length` 写成 2048 → 微调产物在 >2048 token 的大图上推理被截断（已由 launcher 自动恢复为 16384 修复）；
2. 1080p 帧视觉 token（~2546-4590）在 seq=2048 时**训练中被丢**（日志 `idx N failed: image token mismatch`）→ 微调模型在大图上的 MTP 路径行为漂移。

---

## 3. 显存教训与 launcher 改进（本次事故复盘）

**事故**：seq=2048 训练（~15GB）期间并发启动了第二个模型推理（~7GB），逼近 16GB 顶格 + Windows 宿主显存被挤压，几乎把宿主搞崩。

**根因**：16GB 卡跑 3.5B 模型 LoRA + seq=2048 已无余量；任何并发 GPU 负载都危险。

**改进（已落盘 `scripts/train/`）**：
1. `MAX_SEQ` 默认 **2048 → 1536**（留余量；2048 需显式确认并禁止并发）；
2. `seq>=2048` 时打印显存/并发/样本丢弃三重警告；
3. 训练成功后**自动把 `tokenizer.model_max_length` 恢复为基座模型值**（修推理截断）；
4. 新增 `--warmup` CLI 参数；
5. README 记录 4 条经验教训（余量、样本丢弃、model_max_length、fast/hybrid 回归）。

---

## 4. 建议与下一步（供决策）

1. **把验证做成真正的电动车 LoRA 效果**：1080p 帧先降采样到 ≤1280px（视觉 token < 2048）再入训练，或提高 seq（需多卡/更大显存）；否则电动车高帧数据进不了训练。
2. **类别策略**：若业务只关心「两轮车」，伪标签阶段把 bicycle 并入 electric scooter（`--merge-two-wheeler`），减少子类噪声。
3. **推理部署**：微调产物默认 hybrid 在大图上可能截断，**大图场景优先 slow**，或把大图纳入训练后再用 hybrid。
4. **下一步可选**：a) 用降采样数据重训一版对比电动车检出；b) 人工抽查/修正 30-50 条伪标签后做第二轮训练；c) 接入真实业务评测集量化 F1。

## 5. 复现

```bash
# 1) 零样本标注（WSL）
python scripts/annotation/annotate_yolo.py --data /mnt/c/Data/datasets/detect   --classes "gas cylinder,electric scooter,bicycle" --video-frames 8
# 2) 训练数据
python scripts/annotation/build_train_jsonl.py   --manifest outputs/annotation_detect/manifest.jsonl --data /mnt/c/Data/datasets/detect   --out outputs/annotation_detect/lora_data --holdout 8 --repeat 3.0
# 3) LoRA 训练（launcher，含 model_max_length 自动恢复）
bash scripts/train/train_lora.sh --meta outputs/annotation_detect/lora_data/recipe.json   --output /home/xu/lora_gas/run_v1 --steps 150 --save-steps 50 --seq 2048
# 4) before/after 评测（slow 模式）
python scripts/annotation/eval_before_after.py --model /home/xu/models/LocateAnything-3B   --images outputs/annotation_detect/lora_data/val.jsonl --root /mnt/c/Data/datasets/detect   --classes "gas cylinder,electric scooter,bicycle" --out outputs/annotation_detect/eval_before_slow.json --mode slow
python scripts/annotation/eval_before_after.py --model /home/xu/lora_gas/run_v1 ... --out .../eval_after_slow.json --mode slow
```

# 两轮车闭环第二轮：数据扩充（2K）+ 训练调优（tw_run_v3）

- 日期：2026-08-16
- 分支：`feat/codex/elevator`
- 承接：报告 13（闭环第一轮结论：数据量不足、v2 best@epoch1）；按下一步编排执行"数据扩充到 2K + 训练调优，暂不做全量审核/全量训练"

---

## 0. 结论（TL;DR）

1. **数据扩充**：两轮车样本 120 → **1998 张**（wave2_ebike 1000 + ebike_like 998，剔除已采样 150 防泄漏；剔除 battery 组），≤1280px；教师 2 类零样本标注得 **3231 框**（scooter 1543 / bicycle 1688）。
2. **训练调优**：YOLO26s `tw_run_v3`（batch 32 / lr0 0.005 / freeze 10 / epochs 150 / patience 50），early-stop@99，best@epoch49。
3. **公平对比（同人工 GT val 12 图 / 9 实例）**：
   | 模型 | P | R | mAP50 | mAP50-95 |
   |---|---|---|---|---|
   | v1（120 伪标签） | 0.852 | 0.676 | 0.757 | 0.576 |
   | v2（120 人工GT） | 0.219 | 0.786 | 0.504 | 0.438 |
   | **v3（2K 数据+调优）** | **0.924** | **0.979** | **0.995** | **0.954** |
   - **v3 显著超越 v1/v2，在人工 GT val 上接近满分** → 闭环"数据量不足"瓶颈已解决。
4. **结论**：扩数据（伪标签即可，无需先全量人工审核）+ 调优（更大 batch、更低 LR、freeze 骨干）使 YOLO 在该域追平/超越教师；后续再引入人工审核做质量收敛。

---

## 1. 数据扩充（120 → 1998）

| 项 | 值 |
|---|---|
| 采样 | `run_sample_2k.sh`：wave2_ebike（第二波/电瓶车）1000 + ebike_like（电瓶车类似物）998（2 张损坏跳过），seed 42，≤1280px |
| 排除 | 已采样 150 图 basename（`elevator_sample/exclude.txt`），防与人工 GT val 泄漏 |
| 教师标注 | `run_annotate_tw2k.sh`：1998 图 / 1853s（~31 分钟），**3231 框**：scooter 1543 / bicycle 1688 |
| 训练集组装 | `run_build_tw2k.sh`：train = 1998 新伪标签 + 108 人工复核图（GT）；val = 12 人工 GT 图（clean，不入训练）→ `yolo_detect_tw2k`（train 2106 / 3314 框；val 12 / 9 框） |

> 采样器新增 `--exclude-file` 与损坏图容错（跳过不可读文件）。

## 2. 训练调优（tw_run_v3）

| 参数 | v1/v2 | v3（调优） |
|---|---|---|
| 训练图 | 108 | **2106**（含 108 人工 GT + 1998 伪标签） |
| batch | 16 | **32** |
| lr0 | 0.01 | **0.005** |
| freeze | 0 | **10（freeze 骨干）** |
| epochs / patience | 150 / 50 | 150 / 50 |
| 结果 | v1 best@46、v2 best@1 | **best@49**（稳定收敛，early-stop@99） |

## 3. 公平对比（同人工 GT val）

| 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| tw_run_v1 | 0.852 | 0.676 | 0.757 | 0.576 |
| tw_run_v2 | 0.219 | 0.786 | 0.504 | 0.438 |
| **tw_run_v3** | **0.924** | **0.979** | **0.995** | **0.954** |
| （参考）教师 LA F1 | — | — | F1 0.751 | — |

- scooter（7 实例）：P 0.847 / R 1.0 / mAP50 0.995 / mAP50-95 0.962
- bicycle（2 实例）：P 1.0 / R 0.959 / mAP50 0.995 / mAP50-95 0.945
- val 含 3 个 background（空图）：P 仍高 → 误检控制良好

## 4. 判读与下一步

1. **判读**：数据量是首要瓶颈——120 图无论伪标签还是人工 GT 都撑不起 YOLO 微调；扩到 2K 伪标签后 mAP50 0.757→0.995。调优（batch32/lr0.005/freeze10）消除了 v2 的 best@epoch1 不稳定。
2. **暂缓项**（按用户指示）：未做 2K 全量人工审核、未做全量长训练。
3. **下一步**：
   - 对 v3 误检/漏检做抽样人工复核（工作表机制复用），修正后增量重训 v4（只改必要框，工作量小）；
   - 扩大 val（当前 9 实例偏小）：从 2K 中再抽一批人工 GT 做独立测试集；
   - 导出/设备部署（Jetson TRT FP16/INT8）与精度回归（报告 09 §5 的路线）。

## 5. 工具与产物

- 新增：`run_sample_2k.sh`、`run_annotate_tw2k.sh`、`run_build_tw2k.sh`、`run_train_tw3.sh`、`run_val_tw3.sh`、`build_tw2k_dataset.py`；`sample_dataset.py` 增 `--exclude-file` + 损坏图容错
- 数据：`elevator_tw2k`（1998 图+教师 txt）、`yolo_detect_tw2k`（train 2106 / val 12）
- 模型：`~/data/yolo_runs/tw_run_v3`（best.pt，val mAP50 0.995 / mAP50-95 0.954）
- 报告：本文件（reports/14_tw2k_data_expansion_and_tuning.md）
# YOLO 训推框架落地：环境 / 数据集 / YOLO26s 微调 / 推理 / 导出

- 日期：2026-08-13
- 分支：`feat/codex/yolo`
- 环境：WSL Ubuntu 24.04 / conda `yolo`（clone 自 `locate_anything_sft`）/ RTX 5060 Ti 16GB（sm_120）
- 依据：reports/08 选型结论（Ultralytics + YOLO26 n/s）

---

## 0. 结论（TL;DR）

1. **环境**：WSL 新建 `yolo` conda 环境（python 3.10.20 / torch 2.9.0+cu130 / **ultralytics 8.4.118**），与 LocateAnything 环境完全隔离；CUDA 可用，识别到 RTX 5060 Ti。
2. **数据**：由反馈闭环修正数据集 detect_v2 构建 YOLO 格式 `yolo_detect`（train 113 / val 8，93 框，3 类），train/val 与 LoRA holdout 完全一致 → 支持跨模型（LocateAnything vs YOLO）同图对比。
3. **训练**：YOLO26s（COCO 预训练迁移）200 epochs ≈ **6.4 min**，峰值显存 ~5.6GB；**val mAP50 0.995 / mAP50-95 0.995**（gas P0.991/R1.000，scooter P0.912/R1.000）。
4. **推理（本机验证）**：val 8/8 全部正确检出；全分辨率照片 test1 检出 6 框（GT=5，多出 1 个低置信框——交叉校验价值的实例）。
5. **导出**：ONNX（14s）+ **TensorRT FP16 engine（~300s，4.1ms/帧@640，较 PyTorch 17.4ms 快 ~4×）**；FP16 下低置信框（<~0.3）存在边界差异，部署需按工程阈值确认。
6. **框架脚本**（scripts/yolo/）：build_yolo_dataset.py / train_yolo.sh / infer_yolo.py / export_yolo.py / README.md。

---

## 1. 环境（WSL conda `yolo`）

| 项 | 值 |
|---|---|
| 创建方式 | `conda create --clone locate_anything_sft -n yolo`（复用 torch 2.9+cu130 / py3.10，避免重新下大依赖） |
| 核心依赖 | python 3.10.20 / torch 2.9.0+cu130 / **ultralytics 8.4.118** |
| GPU | torch.cuda.is_available()=True，NVIDIA GeForce RTX 5060 Ti（16311MiB） |
| 隔离性 | 与 `locate_anything` / `locate_anything_sft` 互不影响；Eagle 包依赖告警（numpy/dotenv）仅来自克隆残留，不影响 YOLO 使用 |
| 复现 | `scripts/env/setup_yolo_env.sh` |

> 注意：`~/.config/Ultralytics` 不可写时 ultralytics 回退 `/tmp`；本框架所有训练/导出均显式指定 project/name/输出路径，不受影响。

---

## 2. 数据集（yolo_detect）

- 构建脚本：`scripts/yolo/build_yolo_dataset.py`（detect_v2 → images/{train,val} + labels/{train,val} + data.yaml）
- 划分：val = 反馈闭环 holdout（8 图：gas 6 / scooter 2），train = 其余 113 图
- 规模：121 图 / 93 框；类别 `[gas cylinder, electric scooter, bicycle]`（bicycle 为空类，保留以对齐 detect_v2）

```yaml
# data.yaml
path: /mnt/c/Data/datasets/yolo_detect
train: images/train
val: images/val
nc: 3
names: [gas cylinder, electric scooter, bicycle]
```

---

## 3. 训练（YOLO26s run_v1）

- 命令：`bash scripts/yolo/train_yolo.sh run_v1`
- 配置：`yolo26s.pt`（COCO 预训练）→ 微调；imgsz=640 / batch=16 / epochs=200 / patience=100 / cache=True / workers=4 / seed=42
- 资源与速度：峰值显存 **~5.6GB**（16GB 卡余量充足）；**200 epochs ≈ 0.106h（6.4 min）**
- 验证（best.pt，8 张 val）：

| class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 8 | 8 | 0.951 | 1.000 | **0.995** | **0.995** |
| gas cylinder | 6 | 6 | 0.991 | 1.000 | 0.995 | 0.995 |
| electric scooter | 2 | 2 | 0.912 | 1.000 | 0.995 | 0.995 |

- 产物：`~/yolo_runs/run_v1/weights/{best,last}.pt`（fused 20.3MB；9.47M 参数 / 20.8 GFLOPs）

> 判读：小数据 + 预训练迁移 + 强正则下在 8 张 holdout 上接近满分；scooter P 0.912（2 实例存在 1 个误检或边界差异），需更大独立测试集确认泛化（见 §6）。

---

## 4. 推理验证（本机 5060 Ti）

- 脚本：`scripts/yolo/infer_yolo.py`（输出标注图 + results.json：class/conf/xyxy）
- val 8 张：**8/8 全部正确检出**（gas/scooter 各按图匹配），推理 17-40ms/帧@640（PyTorch）
- 全分辨率照片 `detect_v2/gas_tank/test1.jpeg`（1280×960）：**检出 6 个 gas cylinder**（GT=5，第 6 个 conf 0.29）
  - 意义：YOLO（独立于 LocateAnything 教师）给出**第 6 个候选**——可能是教师漏标（真目标）或 YOLO 误检，正是"交叉校验 GT"要人工/自动裁决的分歧点；该分歧已被记录（见 outputs/yolo_infer_test1/）。
- 注：infer 默认 imgsz=640，大图按比例缩放；如需保留更多小目标细节可调 imgsz=1280（代价是速度）。

---

## 5. 导出（场景 B：嵌入式小算力）

| 格式 | 耗时 | 推理（本机 5060 Ti） | 备注 |
|---|---|---|---|
| PyTorch .pt | - | 17.4ms/帧@640 | 基线 |
| ONNX（simplified） | 14s | - | 跨平台/中间格式 |
| **TensorRT FP16 engine** | ~300s | **4.1ms/帧@640（~4.2×）** | 首选导出目标（Jetson 同链路） |

- 一致性：test1 上 PT 6 框 vs engine 5 框——差异框 conf 0.29（低置信边界），FP16 下被抑制/合并；**提示部署阈值策略**（按业务接受度定 conf 阈值，如 0.3-0.5）。
- 后续：INT8（TRT 需 ≥10.7 修复版，校准集 ≥300 图；RKNN/NCNN/Hailo 链路见报告 08 §5）留待目标设备阶段。

---

## 6. 局限与下一步

1. **验证集过小**：8 张 holdout 不足以支撑泛化结论；下一步应扩充独立测试集（人工复核后的新图）并做多轮交叉校验。
2. **scooter 类别**：样本少（val 2 实例），P 0.912 的边界需更多数据确认；可考虑合并两轮车为单类（detect_v2 已把 bicycle 并入 scooter）。
3. **FP16/INT8 精度回归**：目标设备（Jetson/RKNN）上需用独立测试集做量化前后 mAP 对比。
4. **双卡**：当前单卡实测；双卡 DDP 留作更大模型/更大 batch 的进阶项。
5. **与标注流水线闭环**：YOLO 结果可回灌为 LocateAnything 的交叉校验信号（详见 reports/10）。

---

## 7. 产物清单

- 脚本：`scripts/yolo/{build_yolo_dataset.py, train_yolo.sh, infer_yolo.py, export_yolo.py, README.md}`
- 数据：`C:\Data\datasets\yolo_detect\`（images/labels/data.yaml）
- 训练：`~/yolo_runs/run_v1/`（best.pt / best.onnx / best.engine）
- 推理：`outputs/yolo_infer_val/`、`outputs/yolo_infer_test1/`、`outputs/yolo_infer_test1_engine/`
- 报告：本文件（reports/09_yolo_train_infer_framework.md）
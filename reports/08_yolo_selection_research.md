# YOLO 选型调研：双场景评估（训练 AI 工作站 / 嵌入式小算力推理导出）

- 日期：2026-08-13
- 分支：`feat/codex/yolo`
- 场景 A（训练）：AI 工作站 RTX 5060 Ti 16GB（Blackwell sm_120），甚至双卡
- 场景 B（导出/推理）：嵌入式设备 / 摄像头 / 无人机等小 TOPS 算力设备；实验阶段允许本地（5060 Ti）推理验证
- 数据：`C:\Data\datasets\detect_v2`（121 图 / 93 框，gas cylinder 58 / two-wheeler 35，修正后）

---

## 0. 结论（TL;DR）

1. **训练框架：Ultralytics（`ultralytics` pip 包）**。单一框架覆盖训练/验证/推理/导出，是当前 YOLO 生态的事实标准，对 sm_120（Blackwell）与双卡（DDP）支持成熟，导出链路（ONNX/TensorRT/LiteRT/CoreML/OpenVINO）最全。
2. **检测模型：YOLO26（n/s/m/l/x）**，即 2026-01 发布的 Ultralytics 最新代际（v8.4.0+）。理由：
   - **原生端到端 NMS-free**（默认 one-to-one head，输出 N×300×6）：去掉 NMS 后处理，边缘推理延迟方差小、集成简单；
   - **DFL-free** 检测头：导出兼容性最好（对 RKNN/NCNN/LiteRT 等低算力后端友好）；
   - **精度-延迟 Pareto 全面优于 RT-DETRv2**（YOLO26x 57.5 mAP / 11.8 ms T4 TRT vs RT-DETRv2-x 54.3 / 15.03 ms，且参数量更小）；
   - 官方明确面向"cloud + edge"，CPU ONNX 推理较 YOLO11n 快至 43%，对无 GPU 的嵌入式设备友好。
3. **推荐档位**：场景 B 首推 **YOLO26n/s**（n：2.4M 参数 / 5.4 GFLOPs / T4 TRT 1.7ms；s：9.5M / 20.7 / 2.5ms）；场景 A 训练可用 **YOLO26s/m**，按需上 m 提精度。当前任务仅 2-3 类、数据量小，从 COCO 预训练权重微调即可，n/s 足够。
4. **导出精度基线**：FP16 为主（Jetson TensorRT FP16 稳定、收益大），INT8 作为可选优化（见 §5 的 TensorRT 10.x INT8 坑与解法）。
5. **与标注流水线的耦合（重要）**：YOLO26 家族含 **YOLOE-26 开放词汇检测**（文本/视觉提示零样本），可作为 LocateAnything 标注的"独立第二意见"与交叉校验来源（衔接报告 07 §5.3 / 报告 10 闭环设计）。

> 事实来源标注：本节除"当前数据/环境"外，均来自 Ultralytics 官方文档/论文（2026-01 发布）与 2026 年公开基准，非本机实测；本机实测将在报告 09 中进行。

---

## 1. 评估框架（决策维度）

| 维度 | 场景 A（训练，5060 Ti 单/双卡） | 场景 B（推理，小 TOPS 设备） |
|---|---|---|
| 硬件约束 | 16GB/卡 显存；sm_120（Blackwell）；双卡 DDP | 1-26 TOPS（Jetson Orin Nano 67 TOPS*、RK3588 6 TOPS、Hailo-8 26 TOPS、Coral 4 TOPS、树莓派 5 CPU） |
| 关键指标 | 训练吞吐、显存占用、收敛稳定性、易用性 | 延迟/吞吐、后处理复杂度、量化（FP16/INT8）友好度、导出格式覆盖 |
| 约束项 | 数据量小（~90 框）→ 需要强正则/预训练迁移 | NMS/DFL 等后处理在低算力后端是主要开销与兼容性风险 |

*注：Jetson Orin Nano 官方标称 40 TOPS（稀疏）/ 67 TOPS（INT8 稀疏），实际以 FP16/INT8 推理为准。

---

## 2. 候选模型盘点（2026-08 视角）

| 模型 | 发布 | 框架 | 架构要点 | 对本项目的相关性 |
|---|---|---|---|---|
| YOLOv5 | 2020-2023 | 独立 repo | CNN，成熟但停止主线迭代 | 生态成熟但非新起点 |
| YOLOv8 | 2023 | Ultralytics | 统一框架起点 | 仍是"企业稳妥基线"，但已非最新 |
| YOLOv9 | 2024 | Ultralytics | GELAN/PGI | 社区采用度一般 |
| YOLOv10 | 2024 | Ultralytics | NMS-free（端到端） | NMS-free 先驱，但 RKNN 上自定义算子兼容差 |
| YOLO11 | 2024 | Ultralytics | C3k2/C2PSA | 上代主流；精度-速度均衡 |
| YOLO12 | 2025 初 | 社区（attention-centric） | 注意力中心架构 | 内存/CPU 吞吐劣于 YOLO11/26，边缘不占优 |
| **YOLO26** | **2026-01** | **Ultralytics** | **NMS-free e2e + DFL-free + Progressive Loss/STAL + MuSGD** | **首选（见 §3/§4）** |
| RT-DETRv2 | 2024 | Baidu/Ultralytics | Transformer 检测头 | 精度高但训练显存大、边缘算子复杂，本项目不选 |
| YOLOE-26 | 2026-01 | Ultralytics | 开放词汇（文本/视觉提示） | 与标注流水线耦合的交叉校验候选 |

---

## 3. YOLO26 关键设计（为什么适合本项目）

1. **原生端到端（NMS-free）**：默认 one-to-one head 直接输出最终检测（最多 300 个/图），去掉 NMS 后处理 → 推理延迟确定性高、边缘部署逻辑简单（这正是无人机/摄像头逐帧处理的痛点）。如需更高精度可切 one-to-many head（`end2end=False`）。
2. **DFL-free**：移除 Distribution Focal Loss，检测头更轻、导出更简单 → 对 RKNN/NCNN/LiteRT 等算子集受限的后端大幅降低"算子回退 CPU"风险（YOLOv10 在 RK3588 上因自定义算子回退而慢 10× 的教训）。
3. **小目标友好**：Progressive Loss + STAL 提升小目标正样本覆盖——无人机俯拍、监控远距场景直接受益。
4. **训练资源友好**：官方/第三方评测指出其训练显存显著低于 RT-DETR（attention 机制），消费级显卡（16GB）可用更大 batch；MuSGD（Muon+SGD）提升收敛稳定性。
5. **导出覆盖**：TensorRT / ONNX / CoreML / LiteRT / OpenVINO，与场景 B 全链路对齐。

---

## 4. 双场景评估矩阵

### 4.1 模型档位（COCO，官方数据）

| 模型 | mAPval 50-95 | mAPval(e2e) | CPU ONNX(ms) | T4 TRT10(ms) | params(M) | FLOPs(B) |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26n | 40.9 | 40.1 | 38.9 | 1.7 | 2.4 | 5.4 |
| YOLO26s | 48.6 | 47.8 | 87.2 | 2.5 | 9.5 | 20.7 |
| YOLO26m | 53.1 | 52.5 | 220.0 | 4.7 | 20.4 | 68.2 |
| YOLO26l | 55.0 | 54.4 | 286.2 | 6.2 | 24.8 | 86.4 |
| YOLO26x | 57.5 | 56.9 | 525.8 | 11.8 | 55.7 | 193.9 |
| RT-DETRv2-s/m/l/x | 48.1/51.9/53.4/54.3 | - | - | 5.03/7.51/9.76/15.03 | 20/36/42/76 | 60/100/136/259 |

### 4.2 场景适配结论

| 场景 | 推荐 | 备选 | 说明 |
|---|---|---|---|
| A 训练基线 | YOLO26s（m 可选） | YOLO26m（重精度时） | 16GB 单卡可训 s/m；双卡 DDP 上 m 更从容 |
| B 推理导出 | YOLO26n（超低算力）/ s（均衡） | YOLO11n（保守兼容） | n 面向树莓派/RK3588/手机；s 面向 Jetson/带 NPU 设备 |
| 标注流水线耦合 | YOLOE-26（开放词汇） | - | 文本/视觉提示零样本，作交叉校验 GT 来源 |

---

## 5. 导出与量化要点（场景 B 关键风险）

1. **Jetson（NVIDIA 设备）**：首选 TensorRT。FP16 为稳定基线（第三方实测 Jetson Orin Nano 上 TRT FP16 较 PyTorch 提速 ~8×，YOLO26s FP16 可达 ~50 FPS 级）。**INT8 坑**：TensorRT 10.x 早期（10.3）存在 YOLO26 INT8 校准构建失败问题，需升级 TensorRT ≥10.7（JetPack 6 对应修复，ultralytics v8.4.44 亦做了 INT8/end2end 兼容处理）；INT8 校准需 ≥300 张代表性图片，且建议用修正后数据集（detect_v2 + 扩充）而非 4 张示例。
2. **RK3588（Rockchip NPU 6 TOPS INT8）**：走 RKNN 工具链（ONNX→RKNN）。YOLO26 的 DFL-free + NMS-free 显著降低算子兼容风险；仍需验证每个算子在 RKNN 的映射（经验：自定义算子会回退 CPU，慢 ~10×）。INT8 量化精度损失需用校正集评估。
3. **Hailo-8 / Coral / 树莓派**：Hailo 走 HEF（INT8）；Coral 走 TFLite/LiteRT INT8；树莓派 5 走 LiteRT/ONNX CPU。YOLO26n 的 CPU ONNX 39ms（Xeon 2.0GHz）量级在小 ARM CPU 上可预期到 100-300ms 级（推理级可用，非实时），实时需 NPU。
4. **通用建议**：先在本地（5060 Ti）用 `model.export(format=...)` 验证各格式输出一致性（mAP 对比），再上目标设备；量化前后用 `scripts/eval` 思路做精度回归。

---

## 6. 风险与未决问题

1. **数据量过小**：~90 框 / 121 图，YOLO 微调易过拟合。对策：COCO 预训练迁移 + 数据增强（ultralytics 内建）+ 小模型（n/s）+ 早停；如需提升精度，先扩充标注（走反馈闭环 + 人工复核）。
2. **sm_120 兼容性**：训练需 torch 版本支持 Blackwell（本机已有 cu130 环境可复用经验）；ultralytics 对新 GPU 的支持以官方 issue 为准，落地时先跑 1 epoch smoke。
3. **INT8 精度**：小目标（远距气罐/两轮车）在 INT8 下易掉点，需按目标设备实测后再决定是否用 INT8 或保留 FP16。
4. **双卡训练**：DDP 收益在小数据上有限（数据量小、单卡 16GB 已够 n/s），双卡主要利好更大 batch/更大模型（m/l）；落地时以单卡 n/s 为主、双卡 m 为进阶。
5. **YOLOE-26 与固定类别权衡**：开放词汇适合"标注流水线"侧；正式部署的检测器建议用固定类 YOLO26（更小更快更稳），两者各司其职。

---

## 7. 落地计划（衔接报告 09/10）

1. 报告 09：WSL 新建 `yolo` conda 环境（python 3.10 + torch 2.9+cu130 + ultralytics≥8.4）；用 detect_v2 构建 YOLO 数据集（images+labels+data.yaml）；YOLO26s 微调 + 本地（5060 Ti）验证；导出 ONNX/TensorRT（FP16）并做精度回归。
2. 报告 10：标注→训练→导出闭环工作流设计（agent skill 化），YOLOE-26 作为交叉校验 GT 来源。

---

## 8. 主要来源

- Ultralytics YOLO26 官方文档/论文：https://docs.ultralytics.com/models/yolo26/ 、arXiv:2606.03748
- Ultralytics YOLO26 vs RTDETRv2：https://docs.ultralytics.com/compare/yolo26-vs-rtdetr
- ultralytics v8.4.0（YOLO26/YOLOE-26 发布）、v8.4.44（Jetson TRT 10.7 INT8 修复）：GitHub releases
- Jetson/TensorRT 部署指南（INT8 校准要求）：https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/
- 2026 年公开基准（YOLO26s TRT FP16 ~51 FPS、Orin Nano 部署对比、RK3588 RKNN 自定义算子回退教训等）：网络检索，未在本机复现
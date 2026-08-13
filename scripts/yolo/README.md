# scripts/yolo — YOLO 训推框架（Ultralytics + YOLO26）

面向双场景：训练在 AI 工作站（RTX 5060 Ti 16GB，sm_120，可双卡）；导出/推理面向嵌入式小算力设备
（Jetson Orin / RK3588 / Hailo / 树莓派等），实验阶段在本机（5060 Ti）验证。

## 环境

WSL conda env `yolo`（clone 自 `locate_anything_sft`，隔离依赖）：
- python 3.10.20 / torch 2.9.0+cu130 / ultralytics >= 8.4（YOLO26 支持）
- 激活：`conda activate yolo`；脚本直接调用 `~/miniconda3/envs/yolo/bin/python`
- 若 `~/.config/Ultralytics` 不可写，设置 `YOLO_CONFIG_DIR` 或使用 `/tmp` 回退（已兼容）

## 数据

- 输入：`C:\Data\datasets\detect_v2`（反馈闭环修正后的 LocateAnything 标注，121 图 / 93 框）
- 构建：`python scripts/yolo/build_yolo_dataset.py --data .../detect_v2 --out .../yolo_detect --val <feedback-loop val.jsonl>`
  - 产出 `images/{train,val}` + `labels/{train,val}` + `data.yaml`
  - train/val 划分与 LoRA 反馈闭环的 8 张 holdout 一致，便于跨模型对比

## 训练

```bash
bash scripts/yolo/train_yolo.sh run_v1        # YOLO26s COCO 预训练迁移
```
- 默认：imgsz=640, batch=16, epochs=200, patience=100, cache=True, workers=4, seed=42
- 产物：`~/yolo_runs/run_v1/weights/{best,last}.pt`（约 20MB）+ 日志/曲线
- 本机实测（2026-08-13）：200 epochs ≈ 6.4 min；val（8 图）mAP50 0.995 / mAP50-95 0.995；
  峰值显存 ~5.6GB；推理 2.9ms/帧@640（PyTorch）

## 推理

```bash
python scripts/yolo/infer_yolo.py --model <best.pt|engine|onnx> --source <img|dir> --out <dir>
```
- 输出：标注图（`<out>/annotated/`）+ `results.json`（class/conf/xyxy）

## 导出

```bash
python scripts/yolo/export_yolo.py --model best.pt --format onnx --imgsz 640 --simplify
python scripts/yolo/export_yolo.py --model best.pt --format engine --half --imgsz 640   # TRT FP16
python scripts/yolo/export_yolo.py --model best.pt --format engine --int8 --data data.yaml  # TRT INT8（需校准集）
```
- 本机实测：ONNX 导出 14s；TRT FP16 engine 构建 ~300s；engine 推理 4.1ms/帧@640（约 4× 快于 PyTorch）
- 已知边界：FP16 下低置信框（<~0.3）可能被抑制/合并（test1 上 PT 检出 6 框、engine 5 框），部署以工程阈值为准

## 双卡（DDP）

小数据下单卡已足够；如需更大模型/更大 batch，用 ultralytics `device=[0,1]` 或 `--device 0,1` 自动 DDP。
（本机当前为单卡 5060 Ti，未实测双卡。）

## 与标注流水线的衔接

- detect_v2 的标注来自 LocateAnything 反馈闭环；YOLO 训练/推理结果可反向校验 LocateAnything 的新一轮标注
  （交叉验证 GT），形成标注→训练→导出闭环（见 reports/10）。
- YOLOE-26（开放词汇）可作为动态类别的标注交叉校验候选（见 reports/08 §4.2）。
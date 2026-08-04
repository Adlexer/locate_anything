# scripts/eval — 标准化检测评估（模型 vs YOLO GT）

把 LocateAnything（预训练或微调）在带 YOLO 标注的数据集上做标准检测评估：逐类 P/R/F1@IoU、F1@Mean、matched-IoU，逐图明细，输出 JSON + Markdown。

## 为什么需要它

- `scripts/annotation/eval_before_after.py`：两个模型的**定性对比**（无 GT、无指标）；
- `Eagle/Embodied/evaluation/`：官方 benchmark 格式（COCO/LVIS），与 YOLO-format 数据集不通用；
- 本脚本补上「**模型 vs 自己的 YOLO 真值**」的定量评估，且不依赖官方 benchmark 数据。

## 用法（WSL `locate_anything_sft`）

```bash
# 评估预训练模型（holdout 子集）
python /mnt/c/Dev/locate_anything/scripts/eval/eval_det.py     --model /home/xu/models/LocateAnything-3B     --data /mnt/c/Data/datasets/detect     --classes "gas cylinder,electric scooter,bicycle"     --split /mnt/c/Dev/locate_anything/outputs/annotation_detect/lora_data/val.jsonl     --out /mnt/c/Dev/locate_anything/outputs/annotation_detect/eval_det_pretrained.json     --report-md /mnt/c/Dev/locate_anything/outputs/annotation_detect/eval_det_pretrained.md

# 评估微调模型，对比同一份 holdout
python /mnt/c/Dev/locate_anything/scripts/eval/eval_det.py     --model /home/xu/lora_gas/run_v1     --data /mnt/c/Data/datasets/detect     --classes "gas cylinder,electric scooter,bicycle"     --split .../val.jsonl     --out .../eval_det_finetuned.json --report-md .../eval_det_finetuned.md
```

不带 `--split` 时对整个数据目录递归评估（GT txt 与图片同目录，自动跳过 `_result.` 派生文件与辅助目录）。

## 指标说明

| 指标 | 定义 | 说明 |
|---|---|---|
| P / R / F1 @ IoU | 阈值 IoU 下的精确率/召回率/F1 | 逐类 + macro 平均 |
| F1@Mean | IoU 0.5~0.95（步长 0.05）F1 的均值 | 与 LocateAnything 论文 `F1@Mean` 同口径 |
| matched-IoU | IoU=0.5 匹配到的 GT-Pred 对平均 IoU | 定位质量代理指标 |
| macro | 有 GT 的类别的算术平均 | 无 GT 的类不参与 |

**关于 mAP**：LocateAnything 输出不带置信度分数，排序型 mAP 无法良定义；因此采用阈值型 P/R/F1 + F1@Mean（该模型家族的诚实口径）。若后续拿到带分数的检测头再补 mAP。

## 匹配规则（已实现）

- 逐类、贪心一对一（GT 依次取最大 IoU 未占用预测）；预测框先做逐类 NMS（默认 IoU 0.5，`--nms 0` 关闭）；
- 类别错配（如 scooter↔bicycle）在指标上表现为该类的 FN + 另一类的 FP——正好量化子类混淆；
- 未知 label 的预测计入 `unknown_label_preds`（通常为 0，说明检测标签符合 classes 词表）。

## 已知口径提醒

- 用「模型自己生成的伪标签」做 GT 评估是**自洽性测试**（circular），数字偏乐观；拿到人工真实标注后同脚本即可得到真实精度。
- 视频帧评估：`--split` 可指向含 `_frames/...` 的 val.jsonl（GT txt 已随抽帧生成）。

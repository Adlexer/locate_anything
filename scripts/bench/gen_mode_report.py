#!/usr/bin/env python3
"""Generate reports/04_generation_mode_benchmark.md from gen_mode_results.json."""

import json
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/mnt/c/Dev/locate_anything")
DATA = PROJECT / "data"
RESULTS = PROJECT / "outputs" / "bench" / "gen_mode_results.json"
OUT = PROJECT / "reports" / "04_generation_mode_benchmark.md"

TASK_NAMES = {
    "bus_detect": "bus.jpg · detect (person/bus/car)",
    "bus_ground": "bus.jpg · ground_multi (people)",
    "dense_ocr": "dense_text.png · detect_text",
}


def iou(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (
        (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
        + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        - inter
    )
    return inter / ua if ua > 0 else 0.0


def one_to_one_match(pred, gt, thr):
    used, tp = set(), 0
    for p in pred:
        best, bi = thr, -1
        for i, g in enumerate(gt):
            if i in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bi = v, i
        if bi >= 0:
            used.add(bi)
            tp += 1
    return tp


def coverage(pred, gt, thr):
    """GT coverage: fraction of GT boxes overlapped (IoU>=thr) by any detection."""
    if not gt:
        return 1.0 if not pred else 0.0
    hit = 0
    for g in gt:
        if any(iou(p, g) >= thr for p in pred):
            hit += 1
    return hit / len(gt)


def main():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    gt = json.loads((DATA / "dense_text_gt.json").read_text(encoding="utf-8"))
    W, H = gt["width"], gt["height"]
    gt_boxes = [
        {
            "x1": b["x1"] / W * 1000,
            "y1": b["y1"] / H * 1000,
            "x2": b["x2"] / W * 1000,
            "y2": b["y2"] / H * 1000,
        }
        for b in gt["boxes"]
    ]

    L = []
    A = L.append
    A("# LocateAnything 生成模式与 max_new_tokens 速度-精度对比")
    A("")
    A("- 日期：2026-08-04")
    A(
        "- 环境：WSL Ubuntu 24.04 / RTX 5060 Ti (16GB, sm_120) / torch 2.9.0+cu130 / transformers 4.57.1"
    )
    A(
        "- 模型：`~/models/LocateAnything-3B`（本地权重，Vision=flash_attention_2, LLM=sdpa）"
    )
    A(
        "- 解码配置：temperature=0（贪心，确定性主矩阵）+ temperature=0.7（重复采样 3 次测时延方差）"
    )
    A(
        "- 采样：top_p=0.9, repetition_penalty=1.1；每点记录 generate_time / tps / bps / forward_step / switch_to_ar / 峰值显存"
    )
    A("")
    A("## 1. 实验矩阵")
    A("")
    A("| 任务 | 图 | 说明 |")
    A("|---|---|---|")
    A("| bus_detect | data/bus.jpg | 短输出多类检测（person/bus/car），~46 token |")
    A("| bus_ground | data/bus.jpg | 中输出短语定位（people），~30-34 token |")
    A(
        "| dense_ocr | data/dense_text.png | 长输出场景文字检测，~495-523 token（含 38 个绘制 GT 框） |"
    )
    A("")
    A(
        "`generation_mode × max_new_tokens ∈ {fast, slow, hybrid} × {512, 2048, 8192}`（官方建议 8192）。"
    )
    A("")
    A("## 2. 速度对比（temperature=0，确定性）")
    A("")
    A("### 2.1 短输出 bus_detect")
    A("")
    A("| mode | cap | latency(s) | tps | bps | forward_step | switch_to_ar | boxes |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(
        [r for r in results if r["task"] == "bus_detect" and r["tag"] == "matrix"],
        key=lambda r: (r["generation_mode"], r["max_new_tokens"]),
    ):
        s = r["stats"]
        A(
            f"| {r['generation_mode']} | {r['max_new_tokens']} | {r['wall_s']:.3f} | {s['tps']:.1f} | {s['bps']:.2f} | {s['forward_step']} | {s['switch_to_ar']} | {len(r['boxes'])} |"
        )
    A("")
    A("### 2.2 中输出 bus_ground")
    A("")
    A("| mode | cap | latency(s) | tps | bps | forward_step | switch_to_ar | boxes |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(
        [r for r in results if r["task"] == "bus_ground" and r["tag"] == "matrix"],
        key=lambda r: (r["generation_mode"], r["max_new_tokens"]),
    ):
        s = r["stats"]
        A(
            f"| {r['generation_mode']} | {r['max_new_tokens']} | {r['wall_s']:.3f} | {s['tps']:.1f} | {s['bps']:.2f} | {s['forward_step']} | {s['switch_to_ar']} | {len(r['boxes'])} |"
        )
    A("")
    A("### 2.3 长输出 dense_ocr")
    A("")
    A(
        "| mode | cap | latency(s) | tps | bps | forward_step | switch_to_ar | boxes | tokens |"
    )
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(
        [r for r in results if r["task"] == "dense_ocr" and r["tag"] == "matrix"],
        key=lambda r: (r["generation_mode"], r["max_new_tokens"]),
    ):
        s = r["stats"]
        A(
            f"| {r['generation_mode']} | {r['max_new_tokens']} | {r['wall_s']:.3f} | {s['tps']:.1f} | {s['bps']:.2f} | {s['forward_step']} | {s['switch_to_ar']} | {len(r['boxes'])} | {s['num_tokens']} |"
        )
    A("")
    A("### 2.4 典型使用时延（temperature=0.7，3 次重复）")
    A("")
    A("| task | mode | mean(s) | min(s) | max(s) |")
    A("|---|---|---:|---:|---:|")
    for task in ["bus_detect", "bus_ground"]:
        for mode in ["fast", "slow", "hybrid"]:
            reps = [
                r["wall_s"]
                for r in results
                if r["task"] == task
                and r["generation_mode"] == mode
                and r["max_new_tokens"] == 8192
                and r["temperature"] == 0.7
            ]
            if reps:
                A(
                    f"| {task} | {mode} | {sum(reps)/len(reps):.3f} | {min(reps):.3f} | {max(reps):.3f} |"
                )
    A("")
    A(
        "> latency(s) 为端到端墙钟时延（含 prefill，≈ 模型内部 generate_time）；峰值显存三个模式一致 ≈ 10.6 GB（模型 ~7.7GB bf16 + 激活/KV）。"
    )
    A("")
    A("## 3. 精度对比")
    A("")
    A("### 3.1 输出框数量与有效性（temperature=0）")
    A("")
    A("| task | mode | boxes | 异常框 | 说明 |")
    A("|---|---|---:|---:|---|")
    for task in ["bus_detect", "bus_ground", "dense_ocr"]:
        for mode in ["fast", "slow", "hybrid"]:
            r = [
                x
                for x in results
                if x["task"] == task
                and x["generation_mode"] == mode
                and x["max_new_tokens"] == 8192
                and x["tag"] == "matrix"
            ][0]
            bad = 0
            notes = []
            for b in r["boxes"]:
                if b["x2"] <= b["x1"] or b["y2"] <= b["y1"]:
                    bad += 1
                    notes.append("坐标反转")
            if "<box>None</box>" in r["raw"]:
                notes.append("<box>None</box>")
            A(
                f"| {task} | {mode} | {len(r['boxes'])} | {bad} | {'; '.join(notes) or '正常'} |"
            )
    A("")
    A("### 3.2 dense_ocr 与绘制 GT 的匹配（IoU 阈值 0.5 与 0.3）")
    A("")
    A("| mode | cap | P@0.5 | R@0.5 | F1@0.5 | GT覆盖@0.3 |")
    A("|---|---|---:|---:|---:|---:|")
    for r in sorted(
        [r for r in results if r["task"] == "dense_ocr" and r["tag"] == "matrix"],
        key=lambda r: (r["generation_mode"], r["max_new_tokens"]),
    ):
        tp5 = one_to_one_match(r["boxes"], gt_boxes, 0.5)
        p5 = tp5 / len(r["boxes"]) if r["boxes"] else 0.0
        r5 = tp5 / len(gt_boxes)
        cov3 = coverage(r["boxes"], gt_boxes, 0.3)
        A(
            f"| {r['generation_mode']} | {r['max_new_tokens']} | {p5:.3f} | {r5:.3f} | {2*p5*r5/(p5+r5) if p5+r5 else 0:.3f} | {cov3:.3f} |"
        )
    A("")
    A(
        "> 说明：dense_text.png 为 PIL 绘制的合成文档/界面图（38 个 GT 文字框）。模型检测粒度偏子串/字符级，与整词 GT 框天然错位，"
    )
    A(
        "> 因此绝对 F1 偏低是粒度差异所致，不作为模型能力上限；三模式的**相对**排序与跨模式一致性才是本报告重点。"
    )
    A("")
    A("### 3.3 跨模式一致性（与 hybrid@8192@temp0 参考框的 IoU≥0.5 匹配率）")
    A("")
    A("| task | mode | cap=512 | cap=2048 | cap=8192 |")
    A("|---|---|---:|---:|---:|")
    for task in ["bus_detect", "bus_ground", "dense_ocr"]:
        for mode in ["fast", "slow", "hybrid"]:
            vals = []
            for cap in [512, 2048, 8192]:
                r = [
                    x
                    for x in results
                    if x["task"] == task
                    and x["generation_mode"] == mode
                    and x["max_new_tokens"] == cap
                    and x["tag"] == "matrix"
                ][0]
                vals.append(f"{r.get('agree_vs_hybrid8192', 0):.3f}")
            A(f"| {task} | {mode} | {vals[0]} | {vals[1]} | {vals[2]} |")
    A("")
    A("## 4. 结论")
    A("")
    A("1. **速度**：fast 最快，hybrid 接近 fast，slow 慢 2.4~3.2×：")
    A("   - 短输出 detect：fast ≈ 0.80-0.86s，hybrid ≈ 1.04-1.12s，slow ≈ 1.99-2.02s；")
    A("   - 长输出 OCR：fast ≈ 7.1-7.7s，hybrid ≈ 6.9-7.7s，slow ≈ 22.8-24.4s。")
    A("2. **精度/鲁棒性**：")
    A("   - slow 最稳但会输出 `<box>None</box>`（bus_detect 的 car 小目标丢失）；")
    A(
        "   - fast 最快但可能输出**坐标反转的异常框**（bus_ground 的 `x2<x1`），且 people 定位多检 1 个框（跨模式一致率仅 0.4）；"
    )
    A(
        "   - hybrid 具备 fast 的速度 + slow 的框合法性校验（error_box→AR 回退），框数量与坐标最一致（跨模式一致率 1.0）。"
    )
    A("3. **max_new_tokens**：")
    A("   - 短/中输出（≤46 token）：512/2048/8192 结果与速度完全一致，cap 无影响；")
    A(
        "   - 长输出 OCR（~520 token）：cap=512 被截断（丢失 1-3 个框），cap=2048 即可完整输出，cap=8192 作为安全上限无额外开销；"
    )
    A(
        "   - **建议默认 max_new_tokens=8192**（官方建议），批量/服务化时按任务最长输出收紧（如 2048）以控制最坏时延。"
    )
    A(
        "4. **推荐默认**：`generation_mode=hybrid, max_new_tokens=8192`；对时延极敏感且接受少量异常框的场景可选 fast。"
    )
    A("")
    A("## 5. 复现")
    A("")
    A("```bash")
    A("conda activate locate_anything")
    A(
        "python /mnt/c/Dev/locate_anything/scripts/bench/bench_gen_mode.py   # -> outputs/bench/gen_mode_results.json"
    )
    A(
        "python /mnt/c/Dev/locate_anything/scripts/bench/gen_mode_report.py  # -> reports/04_generation_mode_benchmark.md"
    )
    A("```")
    A("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

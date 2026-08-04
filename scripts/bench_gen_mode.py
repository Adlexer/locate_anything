#!/usr/bin/env python3
"""Speed-precision benchmark: generation_mode (fast/slow/hybrid) x max_new_tokens.

Run inside the WSL `locate_anything` conda env:
    python /mnt/c/Dev/locate_anything/scripts/bench_gen_mode.py

Outputs:
  outputs/bench/gen_mode_results.json   (raw per-run records)
  reports/03_gen_mode_speed_precision.md (human-readable report)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path("/mnt/c/Dev/locate_anything")
SCRIPTS = PROJECT / "scripts"
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
REPORTS = PROJECT / "reports"
MODEL = "/home/xu/models/LocateAnything-3B"

sys.path.insert(0, str(PROJECT / "Eagle" / "Embodied"))
from locateanything_worker import LocateAnythingWorker  # noqa: E402

TASKS = [
    {
        "name": "bus_detect",
        "image": str(DATA / "bus.jpg"),
        "kind": "detect",
        "categories": ["person", "bus", "car"],
        "note": "short-output multi-category detection",
    },
    {
        "name": "bus_ground",
        "image": str(DATA / "bus.jpg"),
        "kind": "ground_multi",
        "phrase": "people",
        "note": "medium-output phrase grounding",
    },
    {
        "name": "dense_ocr",
        "image": str(DATA / "dense_text.png"),
        "kind": "detect_text",
        "note": "long-output scene-text detection (GT available)",
    },
]

MODES = ["fast", "slow", "hybrid"]
CAPS = [512, 2048, 8192]

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
PAT = re.compile(
    r"<ref>(.*?)</ref>"
    r"|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    r"|<box><(\d+)><(\d+)></box>"
    r"|<box>none</box>"
)


def parse_boxes(answer):
    """Return boxes (normalized 0-1000) with optional labels, plus points."""
    items, current_label = [], None
    for m in PAT.finditer(answer):
        if m.group(1) is not None:
            current_label = m.group(1)
        elif m.group(2) is not None:
            items.append({
                "label": current_label, "kind": "box",
                "x1": int(m.group(2)), "y1": int(m.group(3)),
                "x2": int(m.group(4)), "y2": int(m.group(5)),
            })
        elif m.group(6) is not None:
            items.append({
                "label": current_label, "kind": "point",
                "x": int(m.group(6)), "y": int(m.group(7)),
            })
    boxes = [i for i in items if i["kind"] == "box"]
    points = [i for i in items if i["kind"] == "point"]
    return boxes, points


def parse_stats(stats_str):
    out = {}
    if stats_str:
        for m in re.finditer(r"([A-Za-z_]+)(?:\(s\))?=([\d.]+)", stats_str):
            key, val = m.group(1), m.group(2)
            out[key] = float(val) if "." in val else int(val)
    return out


def iou(a, b):
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def match_stats(pred, gt, thr=0.5):
    """One-to-one IoU matching of normalized boxes; return (precision, recall, f1, tp)."""
    if not gt:
        return (1.0 if not pred else 0.0), 0.0, 0.0, 0
    used = set(); tp = 0
    for p in pred:
        best, bi = thr, -1
        for i, g in enumerate(gt):
            if i in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bi = v, i
        if bi >= 0:
            used.add(bi); tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp


def agreement(pred, ref, thr=0.5):
    """Fraction of pred boxes matched to ref boxes (one-to-one)."""
    if not ref:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    used = set(); hits = 0
    for p in pred:
        for i, g in enumerate(ref):
            if i in used:
                continue
            if iou(p, g) >= thr:
                used.add(i); hits += 1
                break
    return hits / len(pred)


def run_one(worker, task, generation_mode, max_new_tokens, temperature, tag):
    img = Image.open(task["image"]).convert("RGB")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    if task["kind"] == "detect":
        res = worker.detect(img, task["categories"], generation_mode=generation_mode,
                            max_new_tokens=max_new_tokens, temperature=temperature,
                            verbose=True)
    elif task["kind"] == "ground_multi":
        res = worker.ground_multi(img, task["phrase"], generation_mode=generation_mode,
                                  max_new_tokens=max_new_tokens, temperature=temperature,
                                  verbose=True)
    elif task["kind"] == "detect_text":
        res = worker.detect_text(img, generation_mode=generation_mode,
                                 max_new_tokens=max_new_tokens, temperature=temperature,
                                 verbose=True)
    else:
        raise ValueError(task["kind"])
    wall = time.time() - t0
    answer = res["answer"]
    boxes, points = parse_boxes(answer)
    stats = parse_stats(res.get("stats", ""))
    return {
        "tag": tag, "task": task["name"], "generation_mode": generation_mode,
        "max_new_tokens": max_new_tokens, "temperature": temperature,
        "wall_s": round(wall, 3), "raw": answer, "stats": stats,
        "boxes": boxes, "points": points,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def load_gt(task_name):
    if task_name != "dense_ocr":
        return None
    with open(DATA / "dense_text_gt.json", "r", encoding="utf-8") as f:
        gt = json.load(f)
    W, H = gt["width"], gt["height"]
    return [{"x1": b["x1"] / W * 1000, "y1": b["y1"] / H * 1000,
             "x2": b["x2"] / W * 1000, "y2": b["y2"] / H * 1000} for b in gt["boxes"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--no-repeats", action="store_true", help="skip temp=0.7 repeats")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[bench] loading model {args.model} ...", flush=True)
    worker = LocateAnythingWorker(args.model)
    print(f"[bench] loaded in {time.time() - t0:.1f}s; attn={worker.model.language_model.model._attn_implementation} "
          f"vision_attn={getattr(worker.model.vision_model.config, '_attn_implementation', None)}", flush=True)

    results = []
    # 1) deterministic matrix (temperature=0)
    for task in TASKS:
        gt = load_gt(task["name"])
        for mode in MODES:
            for cap in CAPS:
                print(f"[bench] {task['name']} mode={mode} cap={cap} temp=0 ...", flush=True)
                rec = run_one(worker, task, mode, cap, 0.0, "matrix")
                if gt is not None:
                    prec, rec_, f1, tp = match_stats(rec["boxes"], gt)
                    rec.update({"gt_boxes": len(gt), "precision": round(prec, 4),
                                "recall": round(rec_, 4), "f1": round(f1, 4), "tp": tp})
                results.append(rec)

    # 2) repeat sampling runs (default temp=0.7) for latency variance on key configs
    if not args.no_repeats:
        for task in TASKS[:2]:
            for mode in ["hybrid", "fast", "slow"]:
                for rep in range(3):
                    print(f"[bench] {task['name']} mode={mode} cap=8192 temp=0.7 rep={rep} ...", flush=True)
                    rec = run_one(worker, task, mode, 8192, 0.7, f"repeat{rep}")
                    results.append(rec)

    # 3) post-process: cross-mode agreement on real-image tasks
    # reference = hybrid @ 8192 @ temp=0
    for task in TASKS:
        refs = [r for r in results if r["task"] == task["name"]
                and r["generation_mode"] == "hybrid" and r["max_new_tokens"] == 8192
                and r["temperature"] == 0.0]
        if not refs:
            continue
        ref_boxes = refs[0]["boxes"]
        for r in results:
            if r["task"] == task["name"] and r["tag"] == "matrix":
                r["agree_vs_hybrid8192"] = round(agreement(r["boxes"], ref_boxes), 4)

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "bench").mkdir(exist_ok=True)
    with open(OUTPUTS / "bench" / "gen_mode_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[bench] done in {time.time() - t0:.1f}s; wrote {OUTPUTS / 'bench' / 'gen_mode_results.json'}", flush=True)


if __name__ == "__main__":
    main()

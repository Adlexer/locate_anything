#!/usr/bin/env python3
"""Analyze teacher-YOLO agreement on the elevator sample (IoU histogram + per-class)."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    data = Path(args.data)
    names = [
        ln.strip()
        for ln in (data / "classes.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    images = sorted(
        p
        for p in data.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and "_result" not in p.name
    )
    model = YOLO(args.model)

    hist = {
        "0-0.20": 0,
        "0.20-0.35": 0,
        "0.35-0.50": 0,
        "0.50-0.65": 0,
        "0.65-0.80": 0,
        "0.80-1.00": 0,
    }
    cls_agree = {
        c: {"t": 0, "agree": 0, "best_iou_sum": 0.0, "n_iou": 0} for c in names
    }
    per_img = []
    for p in images:
        tboxes = []
        txt = p.with_suffix(".txt")
        if txt.exists():
            for raw in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw.split()
                if len(parts) == 5:
                    try:
                        cid = int(float(parts[0]))
                        cx, cy, w, h = (float(x) for x in parts[1:5])
                    except ValueError:
                        continue
                    if 0 <= cid < len(names):
                        tboxes.append(
                            {
                                "cls": names[cid],
                                "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                            }
                        )
        res = model.predict(str(p), imgsz=args.imgsz, verbose=False)[0]
        yboxes = [
            {
                "cls": model.names[int(b.cls)],
                "box": [float(v) for v in b.xyxyn[0].tolist()],
            }
            for b in res.boxes
        ]
        n = {
            "image": str(p.relative_to(data)).replace("\\", "/"),
            "t": len(tboxes),
            "y": len(yboxes),
            "best_iou": [],
        }
        for t in tboxes:
            best, bestc = -1.0, None
            for y in yboxes:
                v = iou(t["box"], y["box"])
                if v > best:
                    best, bestc = v, y["cls"]
            n["best_iou"].append(round(best, 3))
            cls_agree[t["cls"]]["t"] += 1
            if best >= 0:
                cls_agree[t["cls"]]["best_iou_sum"] += best
                cls_agree[t["cls"]]["n_iou"] += 1
            if best >= 0.5 and bestc == t["cls"]:
                cls_agree[t["cls"]]["agree"] += 1
            if best < 0.20:
                hist["0-0.20"] += 1
            elif best < 0.35:
                hist["0.20-0.35"] += 1
            elif best < 0.50:
                hist["0.35-0.50"] += 1
            elif best < 0.65:
                hist["0.50-0.65"] += 1
            elif best < 0.80:
                hist["0.65-0.80"] += 1
            else:
                hist["0.80-1.00"] += 1
        per_img.append(n)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "images": len(images),
        "teacher_boxes": sum(len(x["best_iou"]) for x in per_img),
        "yolo_boxes": sum(x["y"] for x in per_img),
        "best_iou_hist": hist,
        "per_class": {
            c: {
                "teacher_boxes": v["t"],
                "agree_at_0.5": v["agree"],
                "mean_best_iou": (
                    round(v["best_iou_sum"] / v["t"], 3) if v["t"] else None
                ),
            }
            for c, v in cls_agree.items()
        },
        "per_image": per_img,
    }
    (out / "agreement_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[agreement] images={result['images']} teacher={result['teacher_boxes']} yolo={result['yolo_boxes']}"
    )
    print(f"[agreement] best-IoU hist: {hist}")
    print(f"[agreement] per-class: {result['per_class']}")
    print(f"[agreement] -> {out}/agreement_analysis.json")


if __name__ == "__main__":
    main()

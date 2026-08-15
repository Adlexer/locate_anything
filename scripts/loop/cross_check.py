#!/usr/bin/env python3
"""Cross-check LocateAnything teacher proposals vs YOLO critic predictions.

This is the "second opinion" component of the annotation loop (reports/10):
it runs a trained YOLO model over the same images the teacher annotated, matches
boxes (IoU>=thr + same class), and outputs:
  - agreement stats (agree / class_mismatch / teacher_only / yolo_only)
  - a disagreement worksheet JSONL for human adjudication

Usage (WSL yolo env):
    python scripts/loop/cross_check.py \
        --data /mnt/c/Data/datasets/detect_v2 \
        --model /home/xu/yolo_runs/run_v1/weights/best.pt \
        --out outputs/loop_crosscheck \
        --limit 8
"""

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


def read_teacher_txt(txt_path, names):
    boxes = []
    if not txt_path.exists():
        return boxes
    for raw in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = raw.split()
        if len(p) != 5:
            continue
        try:
            cid = int(float(p[0]))
            cx, cy, w, h = (float(x) for x in p[1:5])
        except ValueError:
            continue
        if cid < 0 or cid >= len(names):
            continue
        boxes.append(
            {
                "class": names[cid],
                "xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            }
        )
    return boxes


def match(teacher, yolo_boxes, thr):
    used = [False] * len(yolo_boxes)
    verdicts = []
    for t in teacher:
        best, bi = -1.0, -1
        for j, y in enumerate(yolo_boxes):
            if used[j]:
                continue
            v = iou(t["xyxy"], y["xyxy"])
            if v > best:
                best, bi = v, j
        if best >= thr:
            used[bi] = True
            if yolo_boxes[bi]["class"] == t["class"]:
                verdicts.append(
                    {
                        "teacher": t,
                        "yolo": yolo_boxes[bi],
                        "iou": round(best, 3),
                        "verdict": "agree",
                    }
                )
            else:
                verdicts.append(
                    {
                        "teacher": t,
                        "yolo": yolo_boxes[bi],
                        "iou": round(best, 3),
                        "verdict": "class_mismatch",
                    }
                )
        else:
            verdicts.append(
                {"teacher": t, "yolo": None, "iou": None, "verdict": "teacher_only"}
            )
    for j, y in enumerate(yolo_boxes):
        if not used[j]:
            verdicts.append(
                {"teacher": None, "yolo": y, "iou": None, "verdict": "yolo_only"}
            )
    return verdicts


def main():
    ap = argparse.ArgumentParser(description="Cross-check teacher vs YOLO")
    ap.add_argument(
        "--data", required=True, help="dataset root with images + teacher txt"
    )
    ap.add_argument("--model", required=True, help="YOLO model (.pt/engine/onnx)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5, help="matching IoU threshold")
    ap.add_argument(
        "--limit", type=int, default=0, help="limit number of images (0=all)"
    )
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
        if p.suffix.lower() in IMAGE_EXTS
        and "_result" not in p.name
        and "_frames" not in p.parts
    )
    if args.limit:
        images = images[: args.limit]

    model = YOLO(args.model)
    stats = {
        "images": 0,
        "teacher_boxes": 0,
        "yolo_boxes": 0,
        "agree": 0,
        "class_mismatch": 0,
        "teacher_only": 0,
        "yolo_only": 0,
    }
    worksheet = []
    for p in images:
        teacher = read_teacher_txt(p.with_suffix(".txt"), names)
        res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        yolo_boxes = [
            {
                "class": model.names[int(b.cls)],
                "xyxy": [float(v) for v in b.xyxyn[0].tolist()],
                "conf": round(float(b.conf), 4),
            }
            for b in res.boxes
        ]
        verdicts = match(teacher, yolo_boxes, args.iou_thr)
        rel = str(p.relative_to(data)).replace("\\", "/")
        for v in verdicts:
            stats[v["verdict"]] += 1
        stats["teacher_boxes"] += len(teacher)
        stats["yolo_boxes"] += len(yolo_boxes)
        stats["images"] += 1
        worksheet.append(
            {
                "image": rel,
                "teacher_boxes": len(teacher),
                "yolo_boxes": len(yolo_boxes),
                "verdicts": verdicts,
            }
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crosscheck.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(out / "worksheet.jsonl", "w", encoding="utf-8") as f:
        for m in worksheet:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    agree_rate = (
        stats["agree"] / stats["teacher_boxes"] if stats["teacher_boxes"] else 0.0
    )
    print(
        f"[cross-check] images={stats['images']} teacher_boxes={stats['teacher_boxes']} yolo_boxes={stats['yolo_boxes']}"
    )
    print(
        f"[cross-check] agree={stats['agree']} class_mismatch={stats['class_mismatch']} teacher_only={stats['teacher_only']} yolo_only={stats['yolo_only']}"
    )
    print(f"[cross-check] teacher-YOLO agreement rate = {agree_rate:.3f}")
    print(f"[cross-check] -> {out}/crosscheck.json + worksheet.jsonl")


if __name__ == "__main__":
    main()

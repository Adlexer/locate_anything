#!/usr/bin/env python3
"""Real metrics: teacher & YOLO vs human-reviewed GT (two-wheeler, 2 classes).

- GT        : human decisions applied to teacher proposals (accept keep / wrong_class flip / delete drop)
- Teacher   : original LocateAnything proposals (the 124 reviewed boxes)
- YOLO      : critic boxes (two-wheeler classes only), with same-class fragmentation merge (Case 1)

Matching: greedy one-to-one IoU>=thr + same class -> TP; else FP/FN. P/R/F1@IoU.
Also reports YOLO merge statistics (boxes before/after) to quantify fragmentation.

Usage:
    python scripts/elevator/eval_review.py \
        --manifest outputs/elevator_review_tw/review_manifest.json \
        --decisions outputs/elevator_review_tw/elevator_review_decisions.json \
        --crosscheck outputs/elevator_crosscheck_tw/worksheet.jsonl \
        --out outputs/elevator_review_tw/eval_review.json
"""

import argparse
import json
from pathlib import Path

CLASSES = ["electric scooter", "bicycle"]
IOU_THR = 0.5


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def flip(c):
    return "bicycle" if c == "electric scooter" else "electric scooter"


def merge_same_class(boxes, thr=0.5):
    """Greedily merge same-class boxes with IoU>thr into union (fragmentation fix)."""
    out = []
    remaining = [dict(b) for b in boxes]
    while remaining:
        base = remaining.pop(0)
        merged = True
        while merged:
            merged = False
            for i, b in list(enumerate(remaining)):
                if b["class"] == base["class"] and iou(base["xyxy"], b["xyxy"]) > thr:
                    x1 = min(base["xyxy"][0], b["xyxy"][0])
                    y1 = min(base["xyxy"][1], b["xyxy"][1])
                    x2 = max(base["xyxy"][2], b["xyxy"][2])
                    y2 = max(base["xyxy"][3], b["xyxy"][3])
                    base = {
                        "class": base["class"],
                        "xyxy": [x1, y1, x2, y2],
                        "conf": max(base.get("conf", 0), b.get("conf", 0)),
                    }
                    remaining.pop(i)
                    merged = True
                    break
        out.append(base)
    return out


def match(gts, preds):
    used = [False] * len(preds)
    tp = fp = fn = 0
    for g in gts:
        best, bi = -1.0, -1
        for j, p in enumerate(preds):
            if used[j]:
                continue
            if p["class"] != g["class"]:
                continue
            v = iou(g["xyxy"], p["xyxy"])
            if v > best:
                best, bi = v, j
        if best >= IOU_THR:
            used[bi] = True
            tp += 1
        else:
            fn += 1
    fp = sum(1 for u in used if not u)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="Real review metrics")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--crosscheck", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    dec = json.load(open(args.decisions, encoding="utf-8"))
    decisions = dec.get("decisions", {})
    cross = {
        json.loads(l)["image"]: json.loads(l)
        for l in open(args.crosscheck, encoding="utf-8")
        if l.strip()
    }

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    agg = {
        "gt": 0,
        "teacher_tp": 0,
        "teacher_fp": 0,
        "teacher_fn": 0,
        "yolo_tp": 0,
        "yolo_fp": 0,
        "yolo_fn": 0,
        "yolo_boxes_raw": 0,
        "yolo_boxes_merged": 0,
    }
    cls_gt = {"electric scooter": 0, "bicycle": 0}
    per_image = []
    for im in manifest["images"]:
        rel = im["image"]
        gts = []
        for b in im["boxes"]:
            d = decisions.get(str(b["id"]), "accept")
            if d == "delete":
                continue
            cls = b["class"] if d == "accept" else flip(b["class"])
            gts.append({"class": cls, "xyxy": b["xyxy"]})
        teacher = [{"class": b["class"], "xyxy": b["xyxy"]} for b in im["boxes"]]
        yolo_raw = []
        for v in cross.get(rel, {}).get("verdicts", []):
            y = v.get("yolo")
            if y and y["class"] in CLASSES:
                yolo_raw.append(
                    {"class": y["class"], "xyxy": y["xyxy"], "conf": y.get("conf")}
                )
        yolo = merge_same_class(yolo_raw, 0.5)

        tt, tfp, tfn = match(gts, teacher)
        yt, yfp, yfn = match(gts, yolo)
        agg["gt"] += len(gts)
        agg["teacher_tp"] += tt
        agg["teacher_fp"] += tfp
        agg["teacher_fn"] += tfn
        agg["yolo_tp"] += yt
        agg["yolo_fp"] += yfp
        agg["yolo_fn"] += yfn
        agg["yolo_boxes_raw"] += len(yolo_raw)
        agg["yolo_boxes_merged"] += len(yolo)
        for g in gts:
            cls_gt[g["class"]] += 1
        per_image.append(
            {"image": rel, "gt": len(gts), "teacher_tp": tt, "yolo_tp": yt}
        )

    def report(name, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {
            "model": name,
            "gt": agg["gt"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1(p, r), 4),
        }

    result = {
        "iou_thr": IOU_THR,
        "classes": CLASSES,
        "gt_boxes": agg["gt"],
        "gt_class_counts": cls_gt,
        "teacher": report(
            "teacher", agg["teacher_tp"], agg["teacher_fp"], agg["teacher_fn"]
        ),
        "yolo": report("yolo(merged)", agg["yolo_tp"], agg["yolo_fp"], agg["yolo_fn"]),
        "yolo_fragmentation": {
            "raw_boxes": agg["yolo_boxes_raw"],
            "merged_boxes": agg["yolo_boxes_merged"],
        },
        "per_image": per_image,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-review] GT={result['gt_boxes']} class_counts={cls_gt}")
    for m in (result["teacher"], result["yolo"]):
        print(
            f"[eval-review] {m['model']}: P={m['precision']} R={m['recall']} F1={m['f1']} (tp={m['tp']} fp={m['fp']} fn={m['fn']})"
        )
    print(
        f"[eval-review] yolo fragmentation: raw={agg['yolo_boxes_raw']} merged={agg['yolo_boxes_merged']}"
    )
    print(f"[eval-review] -> {out}")


if __name__ == "__main__":
    main()

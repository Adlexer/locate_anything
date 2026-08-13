#!/usr/bin/env python3
"""Standard object-detection evaluation: LocateAnything vs YOLO-format ground truth.

Metrics (per class + macro avg):
  - P / R / F1 @ IoU = 0.5, 0.75, 0.9
  - F1@Mean : mean F1 over IoU 0.5..0.95 step 0.05  (same convention as LocateAnything paper "F1@Mean")
  - matched-IoU : mean best-IoU of matched GT-pred pairs @ IoU 0.5
  - count deltas (GT vs pred) and confusion (scooter/bicycle mix shows as FN+FP)

Note on scores: LocateAnything boxes carry no confidence, so ranking-based mAP is not
well-defined; we report thresholded P/R/F1 + F1@Mean (honest for this model family).

Usage (WSL `locate_anything_sft` env):

    python /mnt/c/Dev/locate_anything/scripts/eval/eval_det.py \
        --model /home/xu/models/LocateAnything-3B \
        --data /mnt/c/Data/datasets/detect \
        --classes "gas cylinder,electric scooter,bicycle" \
        --split outputs/annotation_detect/lora_data/val.jsonl \
        --out outputs/annotation_detect/eval_det_pretrained.json

Output: <out> JSON (per-image + per-class + overall) and a console table.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Eagle" / "Embodied"))
from locateanything_worker import LocateAnythingWorker  # noqa: E402

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SKIP_DIR_NAMES = {
    "_annotated",
    "_probe",
    "_frames_aux",
    "__pycache__",
    "previews",
    "outputs",
}
SKIP_FILE_RE = re.compile(r"_result\.|\.json$|\.jsonl$|\.txt$")

BOX_RE = re.compile(
    r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>|<box>None</box>"
)
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def iou(a, b):
    """a,b = (x1,y1,x2,y2) in normalized [0,1] coords."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def parse_gt(txt_path, n_classes):
    """YOLO txt -> list of (class_id, box[0,1]). Returns (boxes, issues)."""
    boxes, issues = [], []
    if not txt_path.exists():
        return boxes, ["missing_label"]
    for i, raw in enumerate(
        txt_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        p = raw.split()
        if len(p) != 5:
            issues.append(f"line{i}:fields={len(p)}")
            continue
        try:
            cid = int(float(p[0]))
            cx, cy, w, h = (float(x) for x in p[1:5])
        except ValueError:
            issues.append(f"line{i}:non-numeric")
            continue
        if cid < 0 or cid >= n_classes:
            issues.append(f"line{i}:class_id={cid}")
        boxes.append((cid, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes, issues


def parse_pred(answer, classes):
    """Model answer -> list of (class_id, box[0,1]); unknown labels skipped."""
    items, cur, unknown = [], None, 0
    for m in BOX_RE.finditer(answer):
        if m.group(1) is not None:
            cur = m.group(1)
        elif m.group(2) is not None:
            label = (cur or "").strip()
            if label not in classes:
                unknown += 1
                continue
            box = tuple(v / 1000.0 for v in map(int, m.groups()[1:5]))
            items.append((classes[label], box))
    return items, unknown


def nms_per_class(preds, thr):
    """Class-wise greedy NMS on (class_id, box)."""
    if thr <= 0:
        return preds
    by_cls = {}
    for cid, box in preds:
        by_cls.setdefault(cid, []).append(box)
    out = []
    for cid, boxes in by_cls.items():
        boxes = sorted(
            boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])
        )  # largest first
        keep = []
        for b in boxes:
            if all(iou(b, k) < thr for k in keep):
                keep.append(b)
        out.extend((cid, b) for b in keep)
    return out


def match(gts, preds, thr):
    """Greedy one-to-one matching. Returns (tp, fp, fn, matched_ious)."""
    used = [False] * len(preds)
    tp = fp = fn = 0
    matched_ious = []
    for g in gts:
        best, bi = -1.0, -1
        for j, p in enumerate(preds):
            if used[j]:
                continue
            v = iou(g[1], p[1])
            if v > best:
                best, bi = v, j
        if best >= thr:
            used[bi] = True
            tp += 1
            matched_ious.append(best)
        else:
            fn += 1
    fp = sum(1 for u in used if not u)
    return tp, fp, fn, matched_ious


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def collect_images(data_dir, split_jsonl):
    if split_jsonl:
        imgs = []
        for line in open(split_jsonl, encoding="utf-8"):
            line = line.strip()
            if line:
                imgs.append(Path(data_dir) / json.loads(line)["image"])
        return imgs
    imgs = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for f in filenames:
            if SKIP_FILE_RE.search(f):
                continue
            if Path(f).suffix.lower() in IMAGE_EXTS:
                imgs.append(Path(dirpath) / f)
    return sorted(imgs)


def main():
    ap = argparse.ArgumentParser(description="LocateAnything vs YOLO-GT detection eval")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--data", required=True, help="dataset root (GT txt sits next to images)"
    )
    ap.add_argument(
        "--classes",
        required=True,
        help="comma-separated class names; index = YOLO class id",
    )
    ap.add_argument(
        "--split",
        default=None,
        help='optional JSONL with "image" fields (e.g. val.jsonl)',
    )
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--mode", default="slow", choices=["fast", "slow", "hybrid"])
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument(
        "--nms", type=float, default=0.5, help="class-wise NMS IoU (0 = off)"
    )
    ap.add_argument("--report-md", default=None, help="optional markdown report path")
    args = ap.parse_args()

    classes = {}
    for i, name in enumerate(c.strip() for c in args.classes.split(",") if c.strip()):
        classes[name] = i
    if not classes:
        sys.exit("no classes")
    data = Path(args.data)
    images = collect_images(data, args.split)
    print(f"[eval] classes={list(classes)} images={len(images)}")

    t0 = time.time()
    worker = LocateAnythingWorker(args.model, device="cuda")
    print(f"[eval] model loaded in {time.time()-t0:.1f}s")

    # accumulators: per class -> {gt_count, per-threshold tp/fp/fn, matched_ious}
    acc = {
        cid: {
            "gt": 0,
            "pred": 0,
            "t": {thr: [0, 0, 0] for thr in IOU_THRESHOLDS},
            "ious": [],
        }
        for cid in classes.values()
    }
    per_image = {}
    unknown_total = 0
    gt_issues = {}

    for img_path in images:
        rel = str(img_path.relative_to(data))
        gts, issues = parse_gt(img_path.with_suffix(".txt"), len(classes))
        if issues:
            gt_issues[rel] = issues
        img = Image.open(img_path).convert("RGB")
        cats = list(classes)
        r = worker.detect(
            img,
            cats,
            generation_mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.1,
            verbose=False,
        )
        preds, unknown = parse_pred(r["answer"], classes)
        unknown_total += unknown
        preds = nms_per_class(preds, args.nms)

        # per-class matching at each threshold
        row = {"gt": len(gts), "pred": len(preds)}
        for thr in IOU_THRESHOLDS:
            row[f"tp@{thr}"] = row[f"fp@{thr}"] = row[f"fn@{thr}"] = 0
        for cid in classes.values():
            cgts = [g for g in gts if g[0] == cid]
            cpreds = [p for p in preds if p[0] == cid]
            acc[cid]["gt"] += len(cgts)
            acc[cid]["pred"] += len(cpreds)
            for thr in IOU_THRESHOLDS:
                tp, fp, fn, ious = match(cgts, cpreds, thr)
                acc[cid]["t"][thr][0] += tp
                acc[cid]["t"][thr][1] += fp
                acc[cid]["t"][thr][2] += fn
                row[f"tp@{thr}"] += tp
                row[f"fp@{thr}"] += fp
                row[f"fn@{thr}"] += fn
            _, _, _, ious = match(cgts, cpreds, 0.5)
            acc[cid]["ious"].extend(ious)
        per_image[rel] = {
            "gt_boxes": len(gts),
            "pred_boxes": len(preds),
            "tp@0.5": row["tp@0.5"],
            "fp@0.5": row["fp@0.5"],
            "fn@0.5": row["fn@0.5"],
        }

    # ---- aggregate metrics ----
    def f1_at(thr):
        out = {}
        for cid, a in acc.items():
            tp, fp, fn = a["t"][thr]
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            out[cid] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1(p, r), 4),
            }
        return out

    per_class = {}
    for name, cid in sorted(classes.items(), key=lambda kv: kv[1]):
        a = acc[cid]
        f1s = []
        row = {
            "gt": a["gt"],
            "pred": a["pred"],
            "matched_ious": (
                round(sum(a["ious"]) / len(a["ious"]), 4) if a["ious"] else None
            ),
        }
        for thr in [0.5, 0.75, 0.9]:
            tp, fp, fn = a["t"][thr]
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            row[f"P@{thr}"] = round(p, 4)
            row[f"R@{thr}"] = round(r, 4)
            row[f"F1@{thr}"] = round(f1(p, r), 4)
        for thr in IOU_THRESHOLDS:
            tp, fp, fn = a["t"][thr]
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1s.append(f1(p, r))
        row["F1@Mean"] = round(sum(f1s) / len(f1s), 4) if f1s else None
        per_class[name] = row

    # macro avg over classes with gt > 0
    macro_classes = [name for name, row in per_class.items() if row["gt"] > 0]
    macro = {"classes_in_macro": macro_classes}
    for key in ["F1@Mean", "F1@0.5", "F1@0.75", "F1@0.9", "P@0.5", "R@0.5"]:
        vals = [
            per_class[n][key]
            for n in macro_classes
            if per_class[n].get(key) is not None
        ]
        macro[key] = round(sum(vals) / len(vals), 4) if vals else None

    result = {
        "model": args.model,
        "data": str(data),
        "classes": list(classes),
        "mode": args.mode,
        "nms_iou": args.nms,
        "images": len(images),
        "unknown_label_preds": unknown_total,
        "gt_issues": gt_issues,
        "macro_avg": macro,
        "per_class": per_class,
        "per_image": per_image,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # console table
    hdr = f"{'class':22s} {'GT':>3s} {'Pred':>4s} {'P@.5':>6s} {'R@.5':>6s} {'F1@.5':>6s} {'F1@.75':>6s} {'F1@.9':>6s} {'F1@Mean':>8s} {'mIoU':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for name, row in per_class.items():
        miou = row["matched_ious"] if row["matched_ious"] is not None else 0.0
        print(
            f"{name:22s} {row['gt']:3d} {row['pred']:4d} {row['P@0.5']:6.3f} {row['R@0.5']:6.3f} "
            f"{row['F1@0.5']:6.3f} {row['F1@0.75']:6.3f} {row['F1@0.9']:6.3f} {row['F1@Mean']:8.3f} {miou:6.3f}"
        )
    m = macro
    print("-" * len(hdr))
    print(
        f"{'macro':22s} {'':3s} {'':4s} {m['P@0.5']:6.3f} {m['R@0.5']:6.3f} "
        f"{m['F1@0.5']:6.3f} {m['F1@0.75']:6.3f} {m['F1@0.9']:6.3f} {m['F1@Mean']:8.3f} {'':6s}"
    )
    print(f"[eval] wrote {out}  ({time.time()-t0:.0f}s)")

    if args.report_md:
        lines = [
            f"# Detection Eval: {Path(args.model).name}",
            "",
            f"- data: {args.data} | images: {len(images)} | mode: {args.mode} | nms: {args.nms} | classes: {list(classes)}",
            "",
            "| class | GT | Pred | P@.5 | R@.5 | F1@.5 | F1@.75 | F1@.9 | F1@Mean | mIoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in per_class.items():
            miou = row["matched_ious"] if row["matched_ious"] is not None else 0.0
            lines.append(
                f"| {name} | {row['gt']} | {row['pred']} | {row['P@0.5']:.3f} | {row['R@0.5']:.3f} | "
                f"{row['F1@0.5']:.3f} | {row['F1@0.75']:.3f} | {row['F1@0.9']:.3f} | {row['F1@Mean']:.3f} | {miou:.3f} |"
            )
        lines.append(
            f"| **macro** |  |  | **{m['P@0.5']:.3f}** | **{m['R@0.5']:.3f}** | **{m['F1@0.5']:.3f}** | "
            f"**{m['F1@0.75']:.3f}** | **{m['F1@0.9']:.3f}** | **{m['F1@Mean']:.3f}** |  |"
        )
        Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[eval] wrote {args.report_md}")


if __name__ == "__main__":
    main()

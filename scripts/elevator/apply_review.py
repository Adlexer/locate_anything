#!/usr/bin/env python3
"""Apply human review decisions -> corrected two-wheeler GT (YOLO txt).

Reads review_manifest.json (box_id -> image/class/xyxy) + decisions JSON and
writes corrected per-image YOLO txt into --out (mirroring --data structure, 2 classes).

Decision semantics:
  accept      -> keep box as-is
  wrong_class -> flip to the other two-wheeler class (scooter <-> bicycle)
  delete      -> drop box
  (missing boxes in decisions are also written back if provided)

Images that end up with zero boxes are reported (likely raw-battery noise) and can
be excluded from training with --drop-empty.

Usage:
    python scripts/elevator/apply_review.py \
        --manifest outputs/elevator_review_tw/review_manifest.json \
        --decisions outputs/elevator_review_tw/elevator_review_decisions.json \
        --data C:/Data/datasets/elevator_sample_tw \
        --out C:/Data/datasets/elevator_sample_tw_gt \
        --drop-empty
"""

import argparse
import json
from pathlib import Path

CLASSES = ["electric scooter", "bicycle"]  # class order for YOLO txt ids


def flip(c):
    return "bicycle" if c == "electric scooter" else "electric scooter"


def main():
    ap = argparse.ArgumentParser(description="Apply review decisions -> corrected GT")
    ap.add_argument("--manifest", required=True, help="review_manifest.json")
    ap.add_argument("--decisions", required=True, help="elevator_review_decisions.json")
    ap.add_argument("--data", required=True, help="dataset dir (images + teacher txt)")
    ap.add_argument("--out", required=True, help="output corrected dataset dir")
    ap.add_argument(
        "--drop-empty", action="store_true", help="skip images with 0 GT boxes"
    )
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    dec = json.load(open(args.decisions, encoding="utf-8"))
    decisions = dec.get("decisions", {})
    missing = dec.get("missing", [])

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = {
        "accept": 0,
        "wrong_class": 0,
        "delete": 0,
        "missing_added": 0,
        "images_total": 0,
        "images_empty": 0,
        "gt_boxes": 0,
    }
    class_counts = {"electric scooter": 0, "bicycle": 0}
    empty_images = []
    missing_by_image = {}
    for m in missing:
        missing_by_image.setdefault(m["image"], []).append(m)

    for im in manifest["images"]:
        rel = im["image"]
        src_img = data / rel
        if not src_img.exists():
            continue
        stats["images_total"] += 1
        kept = []
        for b in im["boxes"]:
            d = decisions.get(str(b["id"]), "accept")
            if d == "delete":
                stats["delete"] += 1
                continue
            cls = b["class"] if d == "accept" else flip(b["class"])
            stats[d] = stats.get(d, 0) + 1
            x1, y1, x2, y2 = b["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            cid = CLASSES.index(cls)
            kept.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            class_counts[cls] += 1
            stats["gt_boxes"] += 1
        for m in missing_by_image.get(rel, []):
            x1, y1, x2, y2 = m["x1"], m["y1"], m["x2"], m["y2"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cid = CLASSES.index(
                "electric scooter"
            )  # missing default class; override via note
            kept.append(f"{cid} {cx:.6f} {cy:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            stats["missing_added"] += 1
            stats["gt_boxes"] += 1
            class_counts["electric scooter"] += 1
        if not kept:
            stats["images_empty"] += 1
            empty_images.append(rel)
            if args.drop_empty:
                continue
        dst_img = out / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(src_img, dst_img)
        txt = dst_img.with_suffix(".txt")
        txt.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    summary = {**stats, "class_counts": class_counts, "empty_images": empty_images}
    (out / "gt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[apply-review] {stats}")
    print(
        f"[apply-review] class_counts={class_counts} empty_images={len(empty_images)}"
    )
    if empty_images:
        print("[apply-review] first empty images:", empty_images[:8])
    print(f"[apply-review] -> {out} (gt_summary.json written)")


if __name__ == "__main__":
    main()

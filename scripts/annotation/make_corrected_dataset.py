#!/usr/bin/env python3
"""Build a corrected dataset copy from YOLO labels + an auditable correction policy.

Feedback-loop step 1 (label correction). Reads a YOLO-annotated dataset, applies
a policy (class merge, degenerate/tiny-box filter, optional downscale), copies
images into a new root, and writes a per-box review worksheet so a human can
audit every decision and re-run with edited inputs.

Policy (default, first pass):
  - merge: bicycle -> electric scooter  (documented class confusion, see reports/06)
  - drop boxes with normalized area < --min-area (tiny boxes = noise)
  - drop degenerate boxes (x2<=x1 or y2<=y1 after normalization)
  - optional --max-side downscale (longest side cap, aspect preserved)

Outputs (under --out):
  <out>/classes.txt            corrected class list
  <out>/<rel image>            copied (and optionally downscaled) images
  <out>/<rel image>.txt        corrected YOLO labels
  <worksheet>                  JSONL per-box decisions (for human review)
  <summary>                    JSON stats (boxes kept/merged/dropped)

Usage (WSL env with PIL, e.g. locate_anything_sft):

    python /mnt/c/Dev/locate_anything/scripts/annotation/make_corrected_dataset.py \
        --data /mnt/c/Data/datasets/detect \
        --out /mnt/c/Data/datasets/detect_v2 \
        --max-side 1280 \
        --worksheet /mnt/c/Dev/locate_anything/outputs/annotation_detect/correction_worksheet.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIR_NAMES = {
    "_annotated",
    "_probe",
    "__pycache__",
    "previews",
    "outputs",
    "_yolo_viz",
}
SKIP_FILE_RE = re.compile(r"_result\.")


def load_classes(data_dir, cli_classes):
    if cli_classes:
        return [c.strip() for c in cli_classes.split(",") if c.strip()]
    cls_file = Path(data_dir) / "classes.txt"
    if cls_file.exists():
        return [
            ln.strip()
            for ln in cls_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    raise SystemExit(f"[correct] no classes.txt under {data_dir}; use --classes")


def parse_txt(txt_path, n_classes):
    boxes = []
    if not txt_path.exists():
        return boxes
    for i, raw in enumerate(
        txt_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{txt_path}: line {i} fields={len(parts)}")
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError as e:
            raise ValueError(f"{txt_path}: line {i} non-numeric: {e}") from e
        if cid < 0 or cid >= n_classes:
            raise ValueError(f"{txt_path}: line {i} class_id={cid} out of range")
        boxes.append({"cid": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def _to_rgb(img):
    return img.convert("RGB") if img.mode in ("RGBA", "LA", "P") else img


def maybe_downscale(src, dst, max_side):
    img = _to_rgb(Image.open(src))
    w, h = img.size
    longest = max(w, h)
    if max_side and longest > max_side:
        scale = max_side / longest
        img = img.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )
        img.save(dst, quality=92)
        return {"resized": [w, h, img.size[0], img.size[1]]}
    img.save(dst)
    return {"resized": None}


def main():
    ap = argparse.ArgumentParser(description="Build corrected YOLO dataset copy")
    ap.add_argument("--data", required=True, help="source dataset root")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument(
        "--classes",
        default=None,
        help="comma-separated class names (default: <data>/classes.txt)",
    )
    ap.add_argument(
        "--merge",
        default="bicycle:electric scooter",
        help="class merge as 'old:new[,old:new...]' (default: bicycle:electric scooter; '' to disable)",
    )
    ap.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="drop boxes with normalized area < value",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="downscale images so longest side <= value (0 = copy as-is)",
    )
    ap.add_argument(
        "--worksheet", default=None, help="write per-box review worksheet JSONL"
    )
    ap.add_argument(
        "--summary",
        default=None,
        help="write stats JSON (default: <out>/correction_summary.json)",
    )
    ap.add_argument("--dry-run", action="store_true", help="stats only, no writes")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    classes = load_classes(data, args.classes)
    merge = {}
    if args.merge:
        for pair in args.merge.split(","):
            old, _, new = pair.partition(":")
            old, new = old.strip(), new.strip()
            if old and new:
                merge[old] = new
    for old, new in merge.items():
        if old not in classes:
            raise SystemExit(f"[correct] merge source '{old}' not in classes {classes}")
        if new not in classes:
            raise SystemExit(f"[correct] merge target '{new}' not in classes {classes}")
    print(
        f"[correct] classes={classes} merge={merge} min_area={args.min_area} max_side={args.max_side}"
    )

    stats = {
        "images": 0,
        "boxes_in": 0,
        "boxes_kept": 0,
        "boxes_merged": 0,
        "boxes_dropped": 0,
        "resized": 0,
    }
    worksheet = []

    for dirpath, dirnames, filenames in __import__("os").walk(data):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix.lower() not in IMAGE_EXTS or SKIP_FILE_RE.search(p.name):
                continue
            rel = str(p.relative_to(data)).replace("\\", "/")
            boxes = parse_txt(p.with_suffix(".txt"), len(classes))
            stats["images"] += 1
            stats["boxes_in"] += len(boxes)
            kept, decisions = [], []
            for b in boxes:
                name = classes[b["cid"]]
                merged = merge.get(name)
                if merged:
                    name = merged
                    b["merged_from"] = classes[b["cid"]]
                area = b["w"] * b["h"]
                if area < args.min_area or b["w"] <= 0 or b["h"] <= 0:
                    stats["boxes_dropped"] += 1
                    decisions.append(
                        {
                            "class": name,
                            "action": "drop",
                            "reason": "tiny/degenerate",
                            **b,
                        }
                    )
                    continue
                cid = classes.index(name)
                kept.append(
                    f"{cid} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
                )
                action = "merge" if merged else "keep"
                stats["boxes_merged" if merged else "boxes_kept"] += 1
                decisions.append({"class": name, "action": action, "reason": None, **b})
            worksheet.append({"image": rel, "boxes": decisions})
            if args.dry_run:
                continue
            dst_img = out / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            res = maybe_downscale(p, dst_img, args.max_side)
            if res["resized"]:
                stats["resized"] += 1
            txt = dst_img.with_suffix(".txt")
            txt.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    if not args.dry_run:
        (out / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
        if args.worksheet:
            ws = Path(args.worksheet)
            ws.parent.mkdir(parents=True, exist_ok=True)
            with open(ws, "w", encoding="utf-8") as fh:
                for m in worksheet:
                    fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(f"[correct] worksheet -> {ws}")
    summary_path = (
        Path(args.summary) if args.summary else out / "correction_summary.json"
    )
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[correct] summary -> {summary_path}")
    print(f"[correct] stats={stats}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a YOLO-format annotated dataset (images + .txt) back to a LocateAnything manifest.

This is the reverse of annotate_yolo.py's txt output: it turns (possibly
human-corrected) YOLO label files into the manifest.jsonl format consumed by
build_train_jsonl.py, so the annotation feedback loop can rebuild LoRA training
data from corrected labels without re-running the teacher model.

Manifest line:
  {"image": "rel/path.png", "boxes": [{"class": "...", "x1":..,"y1":..,"x2":..,"y2":..}], "raw": null}

Coordinates are converted from normalized YOLO (cx cy w h) to LocateAnything
[0,1000] pixel space (rounded to int).

Usage:
    python scripts/annotation/yolo_txt_to_manifest.py \
        --data /mnt/c/Data/datasets/detect \
        --out outputs/annotation_detect/manifest_corrected.jsonl
"""

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIR_NAMES = {
    "_annotated",
    "_probe",
    "__pycache__",
    "previews",
    "outputs",
    "_yolo_viz",
}
SKIP_FILE_RE = None  # _result files are excluded by extension already; keep hook


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
    raise SystemExit(f"[yolo2manifest] no classes.txt under {data_dir}; use --classes")


def parse_txt(txt_path, n_classes):
    """Return list of (class_id, cx, cy, w, h) with validation."""
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
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            raise ValueError(f"{txt_path}: line {i} coords out of [0,1]: {parts[1:]}")
        boxes.append((cid, cx, cy, w, h))
    return boxes


def to_manifest_box(cid, cx, cy, w, h, classes):
    """YOLO normalized -> LocateAnything [0,1000] box dict."""
    x1 = max(0.0, min(1000.0, (cx - w / 2) * 1000.0))
    x2 = max(0.0, min(1000.0, (cx + w / 2) * 1000.0))
    y1 = max(0.0, min(1000.0, (cy - h / 2) * 1000.0))
    y2 = max(0.0, min(1000.0, (cy + h / 2) * 1000.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "class": classes[cid],
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
    }


def main():
    ap = argparse.ArgumentParser(description="YOLO txt -> LocateAnything manifest")
    ap.add_argument("--data", required=True, help="dataset root (walked recursively)")
    ap.add_argument("--out", required=True, help="manifest.jsonl output path")
    ap.add_argument(
        "--classes",
        default=None,
        help="comma-separated class names (default: <data>/classes.txt)",
    )
    ap.add_argument(
        "--include-frames",
        action="store_true",
        help="also include _frames/ video frame images (default: skip)",
    )
    args = ap.parse_args()

    data = Path(args.data)
    classes = load_classes(data, args.classes)
    n_classes = len(classes)
    print(f"[yolo2manifest] classes={classes}")

    manifest = []
    n_imgs = 0
    n_boxes = 0
    for dirpath, dirnames, filenames in __import__("os").walk(data):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        if not args.include_frames and Path(dirpath).name == "_frames":
            dirnames[:] = []
            continue
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix.lower() not in IMAGE_EXTS or "_result" in p.stem:
                continue
            txt_path = p.with_suffix(".txt")
            if not txt_path.exists():
                continue
            rel = str(p.relative_to(data)).replace("\\", "/")
            boxes = []
            for cid, cx, cy, w, h in parse_txt(txt_path, n_classes):
                b = to_manifest_box(cid, cx, cy, w, h, classes)
                if b is not None:
                    boxes.append(b)
            manifest.append({"image": rel, "boxes": boxes, "raw": None})
            n_imgs += 1
            n_boxes += len(boxes)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[yolo2manifest] images={n_imgs} boxes={n_boxes} -> {out}")


if __name__ == "__main__":
    main()

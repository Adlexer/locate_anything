#!/usr/bin/env python3
"""Visualize a YOLO-format annotated dataset (images + .txt labels).

Scans a folder recursively, parses sibling YOLO label files
(class_id cx cy w h, normalized 0-1), renders boxes onto images,
optionally builds a contact-sheet montage, and writes a stats/validation
summary (JSON) + prints a concise table.

Usage (WSL `locate_anything_sft` env, or any env with PIL):

    python /mnt/c/Dev/locate_anything/scripts/annotation/visualize_yolo.py \
        --data /mnt/c/Data/datasets/detect \
        --out /mnt/c/Dev/locate_anything/outputs/yolo_viz \
        --montage

Outputs:
  <out>/previews/<rel>.jpg     annotated preview per image
  <out>/montage.jpg            contact sheet (--montage)
  <out>/summary.json           per-image stats + class distribution + issues
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None
    HAVE_PIL = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIR_NAMES = {"_annotated", "_probe", "_frames", "__pycache__", "previews", "outputs"}
SKIP_FILE_RE = re.compile(r"_result\.|^classes\.txt$")
PALETTE = [
    (220, 20, 60), (30, 144, 255), (34, 139, 34), (255, 140, 0),
    (138, 43, 226), (0, 206, 209), (255, 20, 147), (60, 179, 113),
    (70, 130, 180), (255, 215, 0), (199, 21, 133), (0, 128, 128),
]


def load_classes(data_dir, cli_classes):
    if cli_classes:
        names = [c.strip() for c in cli_classes.split(",") if c.strip()]
    else:
        cls_file = Path(data_dir) / "classes.txt"
        if cls_file.exists():
            names = [ln.strip() for ln in cls_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            names = []
    return names


def parse_txt(txt_path, n_classes):
    """Return (boxes, issues) where boxes = list of (class_id, cx, cy, w, h)."""
    boxes, issues = [], []
    if not txt_path.exists():
        return boxes, ["missing_label"]
    try:
        lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:  # pragma: no cover
        return boxes, [f"unreadable:{e}"]
    for i, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line{i}:fields={len(parts)}")
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            issues.append(f"line{i}:non-numeric")
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            issues.append(f"line{i}:out-of-range")
        if cid < 0 or (n_classes and cid >= n_classes):
            issues.append(f"line{i}:class_id={cid}")
        boxes.append((cid, cx, cy, w, h))
    return boxes, issues


def boxes_to_pixels(boxes, w, h):
    # Convert normalized YOLO boxes to pixel coords, clamped to the image.
    # Boxes fully outside (e.g. cx > 1 from label noise) become degenerate after
    # clamping and are skipped, so draw_boxes never receives x1 > x2.
    px = []
    for cid, cx, cy, bw, bh in boxes:
        x1 = min(max((cx - bw / 2) * w, 0.0), w)
        x2 = min(max((cx + bw / 2) * w, 0.0), w)
        y1 = min(max((cy - bh / 2) * h, 0.0), h)
        y2 = min(max((cy + bh / 2) * h, 0.0), h)
        if x2 <= x1 or y2 <= y1:
            continue  # fully outside / degenerate; still counted in stats, flagged in issues
        px.append((cid, x1, y1, x2, y2))
    return px


def draw_boxes(img, px_boxes, classes, title=None):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=20)
    except Exception:
        font = ImageFont.load_default()
    W, H = img.size
    for cid, x1, y1, x2, y2 in px_boxes:
        color = PALETTE[cid % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = classes[cid] if cid < len(classes) else f"class_{cid}"
        draw.text((x1 + 2, max(0, y1 - 24)), label, fill=color, font=font)
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    return img


def main():
    ap = argparse.ArgumentParser(description="YOLO dataset visualizer")
    ap.add_argument("--data", required=True, help="dataset root (walked recursively)")
    ap.add_argument("--classes", default=None, help="comma-separated class names (default: <data>/classes.txt)")
    ap.add_argument("--out", default=None, help="output dir (default: <data>/_yolo_viz)")
    ap.add_argument("--max-side", type=int, default=0, help="downscale so longest side <= N px (0 = keep)")
    ap.add_argument("--min-area", type=float, default=0.0, help="drop boxes with normalized area < min-area (0 = keep all)")
    ap.add_argument("--montage", action="store_true", help="also build a contact sheet")
    ap.add_argument("--cols", type=int, default=4, help="montage columns")
    ap.add_argument("--thumb", type=int, default=360, help="montage thumbnail long side")
    ap.add_argument("--only-annotated", action="store_true", help="skip images without a txt label")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing previews")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not HAVE_PIL:
        sys.exit("PIL not available; install pillow (pip install pillow)")

    data = Path(args.data)
    if not data.is_dir():
        sys.exit(f"data dir not found: {data}")
    out_root = Path(args.out) if args.out else (data / "_yolo_viz")
    preview_dir = out_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # skip aux dirs: explicit names plus the "_" convention (own outputs, frames, etc.)
    skip_dirs = set(SKIP_DIR_NAMES)
    if out_root.is_relative_to(data):
        skip_dirs.add(out_root.name)

    classes = load_classes(data, args.classes)
    if not classes:
        sys.exit("no classes: pass --classes or provide <data>/classes.txt")

    # collect images
    images = []
    for dirpath, dirnames, filenames in os.walk(data):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith("_")]
        for f in filenames:
            if SKIP_FILE_RE.search(f):
                continue
            if Path(f).suffix.lower() in IMAGE_EXTS:
                images.append(Path(dirpath) / f)
    images.sort()

    per_image, class_counts, issues_all = {}, {c: 0 for c in classes}, {}
    annotated, empty, missing, rendered, unknown_boxes = 0, 0, 0, 0, 0
    montage_tiles = []

    for img_path in images:
        rel = str(img_path.relative_to(data))
        txt_path = img_path.with_suffix(".txt")
        boxes, issues = parse_txt(txt_path, len(classes))
        if args.min_area > 0:
            boxes = [b for b in boxes if b[3] * b[4] >= args.min_area]
        if issues:
            issues_all[rel] = issues
        if not txt_path.exists():
            missing += 1
        if not boxes:
            empty += 1
        else:
            annotated += 1
            for cid, *_ in boxes:
                if cid < len(classes):
                    class_counts[classes[cid]] = class_counts.get(classes[cid], 0) + 1
                else:
                    unknown_boxes += 1
        per_image[rel] = {"boxes": len(boxes),
                          "classes": sorted({classes[cid] for cid, *_ in boxes if cid < len(classes)}),
                          "unknown_class_boxes": sum(1 for cid, *_ in boxes if cid >= len(classes))}

        if args.only_annotated and not boxes:
            continue
        img = Image.open(img_path).convert("RGB")
        if args.max_side:
            img.thumbnail((args.max_side, args.max_side))
        px = boxes_to_pixels(boxes, *img.size)
        title = f"{rel} | {len(boxes)} obj"
        annotated_img = draw_boxes(img.copy(), px, classes, title=title)

        out_png = preview_dir / f"{img_path.stem}.jpg"
        if args.overwrite or not out_png.exists():
            annotated_img.save(out_png, quality=90)
            rendered += 1

        if args.montage:
            t = annotated_img.copy()
            t.thumbnail((args.thumb, args.thumb))
            montage_tiles.append((rel, t))

    # summary
    summary = {
        "data_dir": str(data),
        "classes": classes,
        "n_images": len(images),
        "annotated": annotated,
        "empty_labels": empty,
        "missing_labels": missing,
        "total_boxes": sum(class_counts.values()) + unknown_boxes,
        "unknown_class_boxes": unknown_boxes,
        "class_counts": class_counts,
        "per_image": per_image,
        "issues": issues_all,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"[yolo-viz] images={len(images)} annotated={annotated} empty={empty} missing={missing}")
        print(f"[yolo-viz] total_boxes={summary['total_boxes']} class_counts={class_counts}")
        if issues_all:
            print(f"[yolo-viz] files with label issues: {len(issues_all)} (see summary.json)")
        print(f"[yolo-viz] previews -> {preview_dir} (rendered={rendered})")

    if args.montage and montage_tiles:
        cols = max(1, args.cols)
        rows = (len(montage_tiles) + cols - 1) // cols
        tw, th = montage_tiles[0][1].size
        cell_h = th + 26
        sheet = Image.new("RGB", (cols * tw, rows * cell_h), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default(size=16)
        except Exception:
            font = ImageFont.load_default()
        for idx, (rel, tile) in enumerate(montage_tiles):
            r, c = divmod(idx, cols)
            x, y = c * tw, r * cell_h
            sheet.paste(tile, (x, y))
            draw.text((x + 4, y + th + 4), Path(rel).name[:40], fill=(255, 255, 255), font=font)
        montage_path = out_root / "montage.jpg"
        sheet.save(montage_path, quality=88)
        if not args.quiet:
            print(f"[yolo-viz] montage -> {montage_path} ({len(montage_tiles)} tiles)")


if __name__ == "__main__":
    main()
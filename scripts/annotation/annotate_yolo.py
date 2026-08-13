#!/usr/bin/env python3
"""LocateAnything zero-shot -> YOLO format annotation tool.

Runs the pretrained LocateAnything-3B in detect mode over a folder of images
(and optionally sampled video frames), writes YOLO-format label txt files
next to each image, plus a summary/manifest for inspection & training reuse.

Run inside the WSL `locate_anything_sft` conda env:

    python /mnt/c/Dev/locate_anything/scripts/annotation/annotate_yolo.py \
        --data /mnt/c/Data/datasets/detect \
        --classes "gas cylinder,electric scooter,bicycle" \
        --video-frames 8

Outputs:
  <data>/<image>.txt                 YOLO label (class_id cx cy w h, normalized)
  <data>/classes.txt                 class names (one per line, YOLO convention)
  <data>/_frames/<video>/frame_*.jpg sampled video frames + matching .txt
  outputs/annotation_detect/summary.json      per-image stats + class distribution
  outputs/annotation_detect/manifest.jsonl    image path + boxes (for training JSONL)
  outputs/annotation_detect/previews/         rendered annotated previews
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

import cv2
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm"}
# derived/aux artifacts to skip while walking
SKIP_DIR_NAMES = {"_annotated", "_probe", "_frames", "__pycache__"}
SKIP_FILE_RE = re.compile(r"_result\.|\.txt$|\.json$|\.jsonl$|classes\.txt$")

# <ref>label</ref><box><x1><y1><x2><y2></box>  |  <box><x><y></box>  |  <box>None</box>
BOX_RE = re.compile(
    r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>|<box>None</box>"
)


def parse_answer(answer):
    """Return list of (label, x1, y1, x2, y2) in normalized [0,1000] coords."""
    items, cur_label = [], None
    for m in BOX_RE.finditer(answer):
        if m.group(1) is not None:
            cur_label = m.group(1)
        elif m.group(2) is not None:
            items.append(
                (
                    cur_label,
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                )
            )
    return items


def to_yolo(label, box, classes):
    """Map (label, [0,1000] box) -> YOLO line or None (unknown label / degenerate)."""
    label = (label or "").strip()
    if label not in classes:
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2 / 1000.0
    cy = (y1 + y2) / 2 / 1000.0
    w = (x2 - x1) / 1000.0
    h = (y2 - y1) / 1000.0
    # clamp into (0,1]
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
    if w <= 0 or h <= 0:
        return None
    cid = classes[label]
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def draw_preview(img, items, classes, out_path):
    """items: (label, x1,y1,x2,y2) in [0,1000]; render overlay to out_path."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    palette = [
        (220, 20, 60),
        (30, 144, 255),
        (34, 139, 34),
        (255, 140, 0),
        (138, 43, 226),
        (0, 206, 209),
    ]
    try:
        font = ImageFont.load_default(size=22)
    except Exception:
        font = ImageFont.load_default()
    w, h = img.size
    for i, (label, x1, y1, x2, y2) in enumerate(items):
        color = palette[(classes.get(label, 0)) % len(palette)]
        px1, py1, px2, py2 = x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        draw.text((px1, max(0, py1 - 24)), f"{label}", fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)


def sample_video_frames(video_path, out_dir, n_frames, step):
    """Extract up to n_frames evenly sampled frames; return list of frame paths."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] cannot open video: {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if total <= 0:
        cap.release()
        return []
    if step and step > 0:
        idxs = list(range(0, total, step))
    else:
        # evenly spread n_frames across the video
        idxs = [
            int(round(i * (total - 1) / max(1, n_frames - 1)))
            for i in range(min(n_frames, total))
        ]
        idxs = sorted(set(idxs))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(p), frame)
        frames.append(p)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(
        description="LocateAnything zero-shot -> YOLO annotation"
    )
    ap.add_argument("--data", required=True, help="dataset root (walked recursively)")
    ap.add_argument(
        "--classes",
        required=True,
        help='comma-separated class names, e.g. "gas cylinder,electric scooter,bicycle"',
    )
    ap.add_argument("--model", default="/home/xu/models/LocateAnything-3B")
    ap.add_argument(
        "--out",
        default=None,
        help="aux output dir (summary/previews/manifest); default <repo>/outputs/annotation_detect",
    )
    ap.add_argument(
        "--video-frames", type=int, default=8, help="max sampled frames per video"
    )
    ap.add_argument(
        "--video-step",
        type=int,
        default=0,
        help="sample every N-th frame (0 = evenly spread)",
    )
    ap.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument(
        "--overwrite", action="store_true", help="re-annotate even if txt exists"
    )
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    classes = {}
    for i, name in enumerate([c.strip() for c in args.classes.split(",") if c.strip()]):
        classes[name] = i
    if not classes:
        sys.exit("no classes given")

    data = Path(args.data)
    if not data.is_dir():
        sys.exit(f"data dir not found: {data}")
    out_root = Path(args.out) if args.out else (REPO / "outputs" / "annotation_detect")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- collect media ----
    images, videos = [], []
    for dirpath, dirnames, filenames in os.walk(data):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for f in filenames:
            if SKIP_FILE_RE.search(f):
                continue
            p = Path(dirpath) / f
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                images.append(p)
            elif ext in VIDEO_EXTS:
                videos.append(p)
    images.sort()
    videos.sort()
    print(
        f"[annotate] classes={list(classes)} images={len(images)} videos={len(videos)}"
    )

    # write classes.txt into dataset root
    (data / "classes.txt").write_text(
        "\n".join(classes.keys()) + "\n", encoding="utf-8"
    )

    # ---- load model ----
    t0 = time.time()
    worker = LocateAnythingWorker(args.model, device=args.device)
    print(f"[annotate] model loaded in {time.time()-t0:.1f}s")

    summary = {
        "classes": list(classes),
        "images": {},
        "videos": {},
        "total_boxes": 0,
        "class_counts": {c: 0 for c in classes},
    }
    manifest = []
    errors = []

    def annotate_image(img_path, manifest_rel):
        """img_path: absolute; manifest_rel: path used in manifest/training."""
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            return "skip"
        img = Image.open(img_path).convert("RGB")
        cats = list(classes)
        r = worker.detect(
            img,
            cats,
            generation_mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            verbose=False,
        )
        items = parse_answer(r["answer"])
        lines, boxes = [], []
        for label, x1, y1, x2, y2 in items:
            line = to_yolo(label, (x1, y1, x2, y2), classes)
            if line is None:
                continue
            lines.append(line)
            boxes.append({"class": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        txt_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        if not args.no_preview:
            draw_preview(
                img.copy(),
                items,
                classes,
                out_root / "previews" / f"{img_path.stem}.jpg",
            )
        manifest.append({"image": manifest_rel, "boxes": boxes, "raw": r["answer"]})
        return len(lines)

    # images
    for p in images:
        rel = str(p.relative_to(data))
        t1 = time.time()
        try:
            n = annotate_image(p, rel)
            summary["images"][rel] = {
                "boxes": 0 if n in ("skip", None) else n,
                "time_s": round(time.time() - t1, 2),
                "status": "skip" if n == "skip" else "ok",
            }
            if n not in ("skip", None):
                summary["total_boxes"] += n
        except Exception as e:
            summary["images"][rel] = {"status": "error", "msg": str(e)}
            errors.append((rel, str(e)))
            print(f"[warn] {rel}: {e}")

    # videos -> sampled frames
    for vp in videos:
        rel = str(vp.relative_to(data))
        frames_dir = data / "_frames" / vp.stem
        try:
            frames = sample_video_frames(
                vp, frames_dir, args.video_frames, args.video_step
            )
            summary["videos"][rel] = {"sampled_frames": len(frames)}
            for fp in frames:
                frel = str(fp.relative_to(data))
                t1 = time.time()
                n = annotate_image(fp, frel)
                if n not in ("skip", None):
                    summary["total_boxes"] += n
                    summary["videos"][rel]["frames_with_boxes"] = summary["videos"][
                        rel
                    ].get("frames_with_boxes", 0) + (1 if n else 0)
        except Exception as e:
            summary["videos"][rel] = {"status": "error", "msg": str(e)}
            errors.append((rel, str(e)))
            print(f"[warn] {rel}: {e}")

    # class distribution from manifest boxes
    for m in manifest:
        for b in m["boxes"]:
            summary["class_counts"][b["class"]] += 1
    summary["errors"] = errors

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(out_root / "manifest.jsonl", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(
        f"[annotate] done in {time.time()-t0:.1f}s total_boxes={summary['total_boxes']}"
    )
    print(f"[annotate] class_counts={summary['class_counts']}")
    print(f"[annotate] summary: {out_root / 'summary.json'}")
    print(f"[annotate] classes.txt: {data / 'classes.txt'}")


if __name__ == "__main__":
    main()

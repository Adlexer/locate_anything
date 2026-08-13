#!/usr/bin/env python3
"""Build a YOLO (ultralytics) dataset from the corrected detect_v2 dataset.

Layout produced (under --out):
  images/train/*, images/val/*     (copied or symlinked images)
  labels/train/*.txt, labels/val/*.txt
  data.yaml                        (path/train/val/nc/names)

The train/val split mirrors the LoRA feedback-loop holdout (lora_data_v2/val.jsonl)
so LocateAnything and YOLO can be compared on the same images.

Usage (WSL yolo env, any python with PIL):
    python scripts/yolo/build_yolo_dataset.py \
        --data /mnt/c/Data/datasets/detect_v2 \
        --out /mnt/c/Data/datasets/yolo_detect \
        --val /mnt/c/Dev/locate_anything/outputs/annotation_detect/lora_data_v2/val.jsonl
"""

import argparse
import json
import shutil
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


def main():
    ap = argparse.ArgumentParser(description="Build YOLO dataset from detect_v2")
    ap.add_argument("--data", required=True, help="source dataset root (detect_v2)")
    ap.add_argument("--out", required=True, help="output YOLO dataset root")
    ap.add_argument(
        "--val", required=True, help="val.jsonl from LoRA feedback loop (holdout set)"
    )
    ap.add_argument(
        "--link", action="store_true", help="symlink images instead of copying"
    )
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    val_images = {
        json.loads(line)["image"].replace("\\", "/")
        for line in open(args.val, encoding="utf-8")
        if line.strip()
    }
    print(f"[yolo-ds] val images: {len(val_images)}")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = n_val = n_boxes = 0
    for dirpath, dirnames, filenames in __import__("os").walk(data):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix.lower() not in IMAGE_EXTS or "_result" in p.stem:
                continue
            txt = p.with_suffix(".txt")
            rel = str(p.relative_to(data)).replace("\\", "/")
            split = "val" if rel in val_images else "train"
            dst_img = out / "images" / split / rel.replace("/", "_")
            dst_txt = out / "labels" / split / (dst_img.stem + ".txt")
            if args.link:
                dst_img.symlink_to(p)
            else:
                shutil.copy2(p, dst_img)
            if txt.exists():
                shutil.copy2(txt, dst_txt)
                n_boxes += sum(
                    1
                    for ln in txt.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                )
            if split == "val":
                n_val += 1
            else:
                n_train += 1

    names = [
        ln.strip()
        for ln in (data / "classes.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    yaml = "\n".join(
        [
            f"# YOLO dataset built from {data} (corrected, feedback-loop v2)",
            f"path: {out.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(names)}",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(names)],
            "",
        ]
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[yolo-ds] train={n_train} val={n_val} boxes={n_boxes} classes={names}")
    print(f"[yolo-ds] -> {out} (data.yaml written)")


if __name__ == "__main__":
    main()

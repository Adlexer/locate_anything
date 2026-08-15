#!/usr/bin/env python3
"""Assemble the 2K two-wheeler YOLO dataset.

train = (a) newly sampled ~1998 images with teacher 2-class pseudo-labels
        (b) 108 human-reviewed images (GT) from the round-1 sample (val excluded)
val   = 12 human-GT images from round-1 (clean, never in train)

Usage:
    python scripts/elevator/build_tw2k_dataset.py \
        --new /mnt/c/Data/datasets/elevator_tw2k \
        --reviewed /mnt/c/Data/datasets/elevator_sample_tw_gt \
        --val-list /mnt/c/Data/datasets/yolo_detect_tw_v2/images/val \
        --out /mnt/c/Data/datasets/yolo_detect_tw2k
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["electric scooter", "bicycle"]


def main():
    ap = argparse.ArgumentParser(description="Assemble 2K two-wheeler YOLO dataset")
    ap.add_argument(
        "--new", required=True, help="newly sampled dir (group subdirs + teacher txt)"
    )
    ap.add_argument(
        "--reviewed", required=True, help="human-reviewed GT dir (group subdirs)"
    )
    ap.add_argument(
        "--val-list", required=True, help="dir containing the 12 val basenames"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    new_dir = Path(args.new)
    rev_dir = Path(args.reviewed)
    out = Path(args.out)
    val_names = {
        p.stem for p in Path(args.val_list).iterdir() if p.suffix.lower() in IMAGE_EXTS
    }
    print(f"[tw2k] val basenames: {len(val_names)}")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_new = n_rev_train = n_rev_val = 0
    # (a) new sample
    for p in new_dir.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or "_result" in p.name:
            continue
        group = p.parent.name
        dst = out / "images" / "train" / f"{group}_{p.name}"
        shutil.copy2(p, dst)
        txt = p.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, out / "labels" / "train" / (dst.stem + ".txt"))
        n_new += 1
    # (b) reviewed GT
    for p in rev_dir.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or "_result" in p.name:
            continue
        split = "val" if p.stem in val_names else "train"
        dst = out / "images" / split / p.name
        shutil.copy2(p, dst)
        txt = p.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, out / "labels" / split / (dst.stem + ".txt"))
        if split == "val":
            n_rev_val += 1
        else:
            n_rev_train += 1

    yaml = "\n".join(
        [
            "# Two-wheeler 2K dataset (1998 teacher pseudo + 108 human GT train; 12 human GT val)",
            f"path: {out.resolve()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(CLASSES)}",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(CLASSES)],
            "",
        ]
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"[tw2k] new_train={n_new} rev_train={n_rev_train} rev_val={n_rev_val}")
    print(f"[tw2k] -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a two-wheeler YOLO dataset from the teacher-annotated subset.

Splits elevator_sample_tw images (teacher txt, classes: electric scooter / bicycle)
into train/val by ratio, writes yolo_detect_tw/{images,labels}/{train,val} + data.yaml.

Usage (WSL yolo env):
    python scripts/elevator/build_tw_dataset.py \
        --data /mnt/c/Data/datasets/elevator_sample_tw \
        --out /mnt/c/Data/datasets/yolo_detect_tw \
        --val-ratio 0.1 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Build two-wheeler YOLO dataset")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    images = sorted(
        p
        for p in data.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and "_result" not in p.name
    )
    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_val = max(1, round(len(images) * args.val_ratio))
    val_set = set(images[:n_val])

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_t = n_v = 0
    for p in images:
        split = "val" if p in val_set else "train"
        dst = out / "images" / split / p.name
        shutil.copy2(p, dst)
        txt = p.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, out / "labels" / split / (dst.stem + ".txt"))
        if split == "val":
            n_v += 1
        else:
            n_t += 1

    names = [
        ln.strip()
        for ln in (data / "classes.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    yaml = "\n".join(
        [
            f"# Two-wheeler YOLO dataset (teacher 2-class pseudo-labels)",
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
    print(f"[tw-ds] train={n_t} val={n_v} classes={names}")
    print(f"[tw-ds] -> {out} (data.yaml written)")


if __name__ == "__main__":
    main()

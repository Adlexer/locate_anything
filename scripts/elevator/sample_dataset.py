#!/usr/bin/env python3
"""Stratified random sampling from a raw collected dataset into a work set.

Samples N images per scene group (deterministic seed), optionally downscales to
--max-side (longest side), copies into <out>/<group_slug>/<basename>.<ext>, and
writes a manifest.json recording the sample (source path, group, size) for
reproducibility.

Usage (WSL, env with PIL e.g. locate_anything_sft):
    python scripts/elevator/sample_dataset.py \
        --data /mnt/c/Data/datasets/elevator_yolo_detect \
        --groups gas:煤气罐 battery:电瓶 ebike_like:电瓶车类似物 \
        --per-group 30 --seed 42 --max-side 1280 \
        --out /mnt/c/Data/datasets/elevator_sample
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _to_rgb(img):
    return img.convert("RGB") if img.mode in ("RGBA", "LA", "P") else img


def sample_group(data, rel_dir, n, rng, exclude=None):
    base = data / rel_dir
    if not base.exists():
        raise SystemExit(f"[sample] group dir not found: {base}")
    files = sorted(
        p
        for p in base.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and "_result" not in p.name
    )
    if len(files) < n:
        print(f"[sample] WARN group {rel_dir}: only {len(files)} images (< {n})")
    files = [p for p in files if p.name not in (exclude or set())]
    if len(files) < n:
        print(
            f"[sample] WARN group {rel_dir}: only {len(files)} images after exclude (< {n})"
        )
    chosen = rng.sample(files, min(n, len(files)))
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Stratified dataset sampler")
    ap.add_argument("--data", required=True, help="raw dataset root")
    ap.add_argument("--groups", required=True, help="slug:rel_dir[,slug:rel_dir...]")
    ap.add_argument("--per-group", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-side", type=int, default=1280, help="downscale longest side (0=keep)"
    )
    ap.add_argument("--out", required=True, help="output work set root")
    ap.add_argument(
        "--exclude-file",
        default=None,
        help="file with source basenames to exclude (one per line)",
    )
    args = ap.parse_args()

    exclude = set()
    if args.exclude_file:
        exclude = {
            ln.strip()
            for ln in Path(args.exclude_file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    data = Path(args.data)
    out = Path(args.out)
    groups = {}
    for pair in args.groups.split(","):
        slug, _, rel = pair.partition(":")
        groups[slug.strip()] = rel.strip()
    rng = random.Random(args.seed)

    manifest = []
    for slug, rel in groups.items():
        chosen = sample_group(data, rel, args.per_group, rng, exclude)
        gdir = out / slug
        gdir.mkdir(parents=True, exist_ok=True)
        for src in chosen:
            try:
                img = _to_rgb(Image.open(src))
                img.load()
            except Exception as e:
                print(f"[sample] WARN skip unreadable {src}: {e}")
                continue
            w, h = img.size
            if args.max_side and max(w, h) > args.max_side:
                scale = args.max_side / max(w, h)
                img = img.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
                )
            dst = gdir / src.name
            img.save(dst)
            manifest.append(
                {
                    "group": slug,
                    "source": str(src).replace("\\", "/"),
                    "file": str(dst.relative_to(out)).replace("\\", "/"),
                    "orig_size": [w, h],
                    "size": list(img.size),
                }
            )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = len(manifest)
    by_group = {}
    for m in manifest:
        by_group[m["group"]] = by_group.get(m["group"], 0) + 1
    print(f"[sample] total={total} by_group={by_group}")
    print(f"[sample] -> {out} (manifest.json written)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build LoRA training JSONL + recipe from a zero-shot annotation manifest.

Input : outputs/annotation_detect/manifest.jsonl (image rel path + boxes in [0,1000])
Output: <out>/train.jsonl, <out>/val.jsonl (holdout), <out>/recipe.json

Samples use LocateAnything training format:
  {"conversations":[{"from":"human","value":"Detect all objects in <image-1>."},
                    {"from":"gpt","value":"<ref>label</ref><box><x1><y1><x2><y2></box>..."}],
   "image":"relative/path.jpg"}
"""
import argparse
import os
import json
import random
from pathlib import Path

BOX_ORDER = ["x1", "y1", "x2", "y2"]


def boxes_to_answer(boxes, merge_map=None):
    """boxes: list of {'class','x1','y1','x2','y2'} in [0,1000] -> gpt answer string."""
    parts = []
    for b in boxes:
        label = merge_map.get(b["class"], b["class"]) if merge_map else b["class"]
        x1, y1, x2, y2 = (int(b[k]) for k in BOX_ORDER)
        if x2 <= x1 or y2 <= y1:
            continue
        parts.append(f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>")
    return "".join(parts)


def to_wsl_path(path):
    """Convert a Windows path (C:\...) to a WSL /mnt/c/... path for training in WSL."""
    p = str(path)
    if os.name == "nt" and len(p) > 2 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p.replace("\\", "/")


def main():
    ap = argparse.ArgumentParser(description="Build LoRA training JSONL/recipe from annotation manifest")
    ap.add_argument("--manifest", required=True, help="manifest.jsonl path")
    ap.add_argument("--data", required=True, help="dataset root (recipe root)")
    ap.add_argument("--out", required=True, help="output dir for jsonl + recipe")
    ap.add_argument("--holdout", type=int, default=8, help="number of holdout images (val, for before/after)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=float, default=3.0, help="recipe repeat_time (oversample small data)")
    ap.add_argument("--min-area", type=float, default=0.002,
                    help="drop boxes with normalized area < min-area (tiny boxes are noise)")
    ap.add_argument("--merge-two-wheeler", action="store_true",
                    help="merge bicycle -> electric scooter (teacher subtype is noisy)")
    args = ap.parse_args()

    manifest = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))

    # keep only samples with >=1 box; drop tiny boxes
    keep = []
    for m in manifest:
        boxes = [b for b in m["boxes"] if (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) / 1e6 >= args.min_area]
        if boxes:
            keep.append({"image": m["image"], "boxes": boxes})
    print(f"[build] manifest={len(manifest)} with_boxes={len(keep)}")

    rng = random.Random(args.seed)
    rng.shuffle(keep)
    val = keep[: args.holdout]
    train = keep[args.holdout:]
    print(f"[build] train={len(train)} val={len(val)}")

    merge_map = {"bicycle": "electric scooter"} if args.merge_two_wheeler else None

    def write_jsonl(path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                answer = boxes_to_answer(s["boxes"], merge_map)
                if not answer:
                    continue
                rec = {
                    "conversations": [
                        {"from": "human", "value": "Detect all objects in <image-1>."},
                        {"from": "gpt", "value": answer},
                    ],
                    "image": s["image"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    recipe = {
        "locany_pseudo_detect": {
            "annotation": to_wsl_path(train_path.resolve()),
            "root": to_wsl_path(Path(args.data).resolve()) + "/",
            "repeat_time": args.repeat,
            "data_augment": True,
        }
    }
    with open(out / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)
    print(f"[build] wrote {train_path} {val_path} {out / 'recipe.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run detect with a given LocateAnything checkpoint over a set of images, dump results.

Used for before/after LoRA comparison on holdout images.

    python scripts/annotation/eval_before_after.py \
        --model /home/xu/models/LocateAnything-3B \
        --images outputs/annotation_detect/lora_data/val.jsonl \
        --classes "gas cylinder,electric scooter,bicycle" \
        --out outputs/annotation_detect/eval_before.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Eagle" / "Embodied"))
from locateanything_worker import LocateAnythingWorker  # noqa: E402

from PIL import Image

BOX_RE = re.compile(
    r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>|<box>None</box>"
)


def parse_answer(answer):
    items, cur = [], None
    for m in BOX_RE.finditer(answer):
        if m.group(1) is not None:
            cur = m.group(1)
        elif m.group(2) is not None:
            items.append(
                {
                    "class": cur,
                    "x1": int(m.group(2)),
                    "y1": int(m.group(3)),
                    "x2": int(m.group(4)),
                    "y2": int(m.group(5)),
                }
            )
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True, help="val.jsonl with image fields")
    ap.add_argument(
        "--root", required=True, help="dataset root for resolving image paths"
    )
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    imgs = []
    with open(args.images, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                imgs.append(json.loads(line)["image"])

    worker = LocateAnythingWorker(args.model, device="cuda")
    results = {}
    for rel in imgs:
        t0 = time.time()
        img = Image.open(Path(args.root) / rel).convert("RGB")
        r = worker.detect(
            img,
            classes,
            generation_mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.1,
            verbose=False,
        )
        results[rel] = {
            "raw": r["answer"],
            "boxes": parse_answer(r["answer"]),
            "time_s": round(time.time() - t0, 2),
        }
        print(f"  {rel}: {len(results[rel]['boxes'])} boxes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"model": args.model, "classes": classes, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

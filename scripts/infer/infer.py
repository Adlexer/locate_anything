#!/usr/bin/env python3
"""LocateAnything reusable inference CLI.

Input : an image + a query (or a task preset)
Output: JSON with boxes/points (normalized 0-1000 and pixel) + an annotated image.

Run inside the WSL `locate_anything` conda env:

    python /mnt/c/Dev/locate_anything/scripts/infer/infer.py \
        --image /path/to/img.jpg --task detect --query "person</c>car" \
        --out /mnt/c/Dev/locate_anything/outputs --generation-mode hybrid \
        --max-new-tokens 8192

Tasks:
  detect       : object detection for --query (split on `</c>` or `,`)
  ground       : phrase grounding (single instance)
  ground_multi : phrase grounding (all instances)
  ground_text  : scene text grounding ("Please locate the text referred as ...")
  detect_text  : scene text detection (all text boxes)
  ground_gui   : GUI grounding (box by default, use --output-type point for a point)
  point        : pointing (single point)

Output files (prefix defaults to the input image stem):
  <out>/<prefix>.json        structured result (boxes/points + raw answer + stats)
  <out>/<prefix>_annotated.jpg  annotated image
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("/mnt/c/Dev/locate_anything")
DEFAULT_MODEL = "/home/xu/models/LocateAnything-3B"

sys.path.insert(0, str(PROJECT / "Eagle" / "Embodied"))
from locateanything_worker import LocateAnythingWorker  # noqa: E402

PAT = re.compile(
    r"<ref>(.*?)</ref>"
    r"|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    r"|<box><(\d+)><(\d+)></box>"
    r"|<box>none</box>"
)

PALETTE = [(220, 20, 60), (30, 144, 255), (34, 139, 34), (255, 140, 0),
           (138, 43, 226), (0, 206, 209), (255, 20, 147), (60, 179, 113)]


def parse_items(answer):
    """Parse model output into box/point items (normalized 0-1000)."""
    items, current_label = [], None
    for m in PAT.finditer(answer):
        if m.group(1) is not None:
            current_label = m.group(1)
        elif m.group(2) is not None:
            items.append({
                "kind": "box", "label": current_label,
                "x1": int(m.group(2)), "y1": int(m.group(3)),
                "x2": int(m.group(4)), "y2": int(m.group(5)),
            })
        elif m.group(6) is not None:
            items.append({
                "kind": "point", "label": current_label,
                "x": int(m.group(6)), "y": int(m.group(7)),
            })
    return items


def to_pixel(item, w, h):
    if item["kind"] == "box":
        return {
            "kind": "box", "label": item["label"],
            "x1": round(item["x1"] / 1000 * w, 2), "y1": round(item["y1"] / 1000 * h, 2),
            "x2": round(item["x2"] / 1000 * w, 2), "y2": round(item["y2"] / 1000 * h, 2),
        }
    return {
        "kind": "point", "label": item["label"],
        "x": round(item["x"] / 1000 * w, 2), "y": round(item["y"] / 1000 * h, 2),
    }


def annotate(img, items, title=""):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
    except Exception:
        font = ImageFont.load_default()
    w, h = img.size
    for i, it in enumerate(items):
        color = PALETTE[i % len(PALETTE)]
        if it["kind"] == "box":
            x1, y1, x2, y2 = it["x1"] / 1000 * w, it["y1"] / 1000 * h, it["x2"] / 1000 * w, it["y2"] / 1000 * h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f'{it.get("label") or ""} [{it["x1"]},{it["y1"]},{it["x2"]},{it["y2"]}]'.strip()
            draw.text((x1 + 2, max(0, y1 - 24)), label, fill=color, font=font)
        else:
            x, y = it["x"] / 1000 * w, it["y"] / 1000 * h
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
            label = f'{it.get("label") or ""} [{it["x"]},{it["y"]}]'.strip()
            draw.text((x + 8, max(0, y - 24)), label, fill=color, font=font)
    if title:
        draw.text((12, 12), title, fill=(0, 0, 0), font=font)
    return img


def run_task(worker, image, args):
    if args.task == "detect":
        cats = re.split(r"</c>|,", args.query) if args.query else []
        cats = [c.strip() for c in cats if c.strip()]
        if not cats:
            raise SystemExit("--task detect requires --query (categories separated by ',' or '</c>')")
        return worker.detect(image, cats, generation_mode=args.generation_mode,
                             max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                             top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                             verbose=True)
    if args.task == "ground":
        return worker.ground_single(image, args.query, generation_mode=args.generation_mode,
                                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                    top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                                    verbose=True)
    if args.task == "ground_multi":
        return worker.ground_multi(image, args.query, generation_mode=args.generation_mode,
                                   max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                   top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                                   verbose=True)
    if args.task == "ground_text":
        return worker.ground_text(image, args.query, generation_mode=args.generation_mode,
                                  max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                  top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                                  verbose=True)
    if args.task == "detect_text":
        return worker.detect_text(image, generation_mode=args.generation_mode,
                                  max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                  top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                                  verbose=True)
    if args.task == "ground_gui":
        return worker.ground_gui(image, args.query, output_type=args.output_type,
                                 generation_mode=args.generation_mode, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                 repetition_penalty=args.repetition_penalty, verbose=True)
    if args.task == "point":
        return worker.point(image, args.query, generation_mode=args.generation_mode,
                            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                            verbose=True)
    raise SystemExit(f"unknown task: {args.task}")


def main():
    ap = argparse.ArgumentParser(description="LocateAnything inference CLI")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model directory (HF format)")
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--task", default="detect", choices=[
        "detect", "ground", "ground_multi", "ground_text", "detect_text", "ground_gui", "point"])
    ap.add_argument("--query", default="", help="query/categories (e.g. 'person</c>car' or 'the bus')")
    ap.add_argument("--output-type", default="box", choices=["box", "point"], help="ground_gui output type")
    ap.add_argument("--out", default=str(PROJECT / "outputs"), help="output directory")
    ap.add_argument("--prefix", default="", help="output file prefix (default: input image stem)")
    ap.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    ap.add_argument("--max-new-tokens", type=int, default=8192, help="official suggestion: 8192")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.quiet:
        print(f"[infer] loading model {args.model} ...", flush=True)
    worker = LocateAnythingWorker(args.model, device=args.device)
    if not args.quiet:
        print(f"[infer] model loaded in {time.time() - t0:.1f}s", flush=True)

    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    t1 = time.time()
    result = run_task(worker, image, args)
    wall = time.time() - t1

    answer = result["answer"]
    items = parse_items(answer)
    boxes = [to_pixel(it, w, h) for it in items if it["kind"] == "box"]
    points = [to_pixel(it, w, h) for it in items if it["kind"] == "point"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.image).stem
    json_path = out_dir / f"{prefix}.json"
    ann_path = out_dir / f"{prefix}_annotated.jpg"

    payload = {
        "model": args.model,
        "image": str(Path(args.image)),
        "image_size": {"width": w, "height": h},
        "task": args.task,
        "query": args.query or None,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "latency_s": round(wall, 3),
        "raw_answer": answer,
        "stats": result.get("stats"),
        "boxes": boxes,
        "points": points,
        "num_boxes": len(boxes),
        "num_points": len(points),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not args.no_annotate:
        annotate(image.copy(), items, title=f"{args.task} | {args.generation_mode} | {args.query}")
        image.save(ann_path, quality=92)

    if not args.quiet:
        print(f"[infer] task={args.task} mode={args.generation_mode} max_new_tokens={args.max_new_tokens}")
        print(f"[infer] latency={wall:.2f}s  boxes={len(boxes)}  points={len(points)}")
        print(f"[infer] raw: {answer}")
        print(f"[infer] wrote {json_path}")
        if not args.no_annotate:
            print(f"[infer] wrote {ann_path}")


if __name__ == "__main__":
    main()

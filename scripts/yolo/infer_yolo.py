#!/usr/bin/env python3
"""YOLO inference CLI: image(s) -> predictions JSON + annotated images.

Usage (WSL yolo env):
    python scripts/yolo/infer_yolo.py --model /home/xu/yolo_runs/run_v1/weights/best.pt \
        --source /mnt/c/Data/datasets/detect_v2/gas_tank/test1.jpeg \
        --out outputs/yolo_infer
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="YOLO inference CLI")
    ap.add_argument("--model", required=True, help="path to .pt or exported engine")
    ap.add_argument("--source", required=True, help="image path or folder or glob")
    ap.add_argument(
        "--out", required=True, help="output dir (annotated images + results.json)"
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--name", default=None, help="results.json filename")
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=True,
        project=str(Path(args.out) / "annotated"),
        name=".",
        exist_ok=True,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or "results.json"
    payload = []
    for r in results:
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                boxes.append(
                    {
                        "class": r.names[int(b.cls)],
                        "conf": round(float(b.conf), 4),
                        "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
                    }
                )
        payload.append({"source": str(Path(r.path)), "boxes": boxes})
    (out / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[yolo-infer] {len(results)} images -> {out / name}")
    for p in payload:
        print(f"  {Path(p['source']).name}: {len(p['boxes'])} boxes")


if __name__ == "__main__":
    main()

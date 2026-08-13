#!/usr/bin/env python3
"""YOLO export CLI: .pt -> ONNX / TensorRT (FP16) with optional consistency check.

Usage (WSL yolo env; TensorRT export needs the local TensorRT libs from torch build):
    python scripts/yolo/export_yolo.py --model /home/xu/yolo_runs/run_v1/weights/best.pt \
        --format onnx --imgsz 640
    python scripts/yolo/export_yolo.py --model .../best.pt --format engine --half
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="YOLO export CLI")
    ap.add_argument("--model", required=True, help="source .pt path")
    ap.add_argument(
        "--format", required=True, choices=["onnx", "engine"], help="target format"
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (TensorRT engine)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action="store_true", help="onnxsim")
    ap.add_argument(
        "--int8", action="store_true", help="INT8 (TensorRT engine, needs --data)"
    )
    ap.add_argument("--data", default=None, help="dataset yaml for INT8 calibration")
    args = ap.parse_args()

    model = YOLO(args.model)
    fmt = args.format
    kwargs = dict(
        imgsz=args.imgsz, dynamic=False, simplify=args.simplify, opset=args.opset
    )
    if fmt == "engine":
        kwargs["half"] = args.half
        if args.int8:
            kwargs["int8"] = True
            kwargs["data"] = args.data
            kwargs["half"] = False
    exported = model.export(format=fmt, **kwargs)
    print(f"[yolo-export] -> {exported}")


if __name__ == "__main__":
    main()

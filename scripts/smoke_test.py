#!/usr/bin/env python3
"""LocateAnything WSL smoke test: load local weights and run one detection."""
import sys, time
import torch
from PIL import Image, ImageDraw

MODEL_PATH = "/mnt/c/Data/LocateAnything-3B"

t0 = time.time()
print("[smoke] loading worker ...", flush=True)
sys.path.insert(0, "/mnt/c/Dev/locate_anything/Eagle/Embodied")
from locateanything_worker import LocateAnythingWorker
worker = LocateAnythingWorker(MODEL_PATH)
print(f"[smoke] model loaded in {time.time()-t0:.1f}s", flush=True)
print(f"[smoke] effective LLM attn: {worker.model.language_model.model._attn_implementation}", flush=True)

# Build a tiny synthetic scene: two colored boxes on white
img = Image.new("RGB", (768, 512), "white")
d = ImageDraw.Draw(img)
d.rectangle([60, 60, 320, 240], fill="red", outline="black", width=4)
d.rectangle([420, 280, 700, 460], fill="blue", outline="black", width=4)
img.save("/tmp/smoke_test.png")

t1 = time.time()
res = worker.detect(img, ["car", "person"], generation_mode="hybrid", max_new_tokens=256, verbose=True)
print("[smoke] detect(car,person):")
print(res.get("answer"))
print("[smoke] stats:", res.get("stats"))
print(f"[smoke] inference took {time.time()-t1:.1f}s", flush=True)

w, h = img.size
boxes = worker.parse_boxes(res["answer"], w, h)
print("[smoke] parsed boxes:", boxes)
print("[smoke] SUCCESS", flush=True)
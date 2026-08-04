#!/usr/bin/env python3
"""LocateAnything real-image smoke test in WSL."""
import sys, time, json, re
import torch
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = "/mnt/c/Data/LocateAnything-3B"
IMG_PATH = "/mnt/c/Dev/locate_anything/data/bus.jpg"
OUT_DIR = "/mnt/c/Dev/locate_anything/outputs"

sys.path.insert(0, "/mnt/c/Dev/locate_anything/Eagle/Embodied")
from locateanything_worker import LocateAnythingWorker

BOX_RE = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")

def parse_ref_boxes(answer):
    out = []
    for m in BOX_RE.finditer(answer):
        label = m.group(1)
        x1, y1, x2, y2 = [int(g) for g in m.groups()[1:]]
        out.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out

def draw_boxes(img, items, w, h, title=""):
    draw = ImageDraw.Draw(img)
    palette = [(220,20,60),(30,144,255),(34,139,34),(255,140,0),(138,43,226),(0,206,209)]
    for i, it in enumerate(items):
        x1 = it["x1"]/1000*w; y1 = it["y1"]/1000*h
        x2 = it["x2"]/1000*w; y2 = it["y2"]/1000*h
        color = palette[i % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'{it.get("label","")} {it["x1"]},{it["y1"]},{it["x2"]},{it["y2"]}'
        try:
            font = ImageFont.load_default(size=18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x1, max(0, y1-20)), label, fill=color, font=font)
    if title:
        draw.text((10, 10), title, fill="black")
    return img

t0 = time.time()
print("[smoke] loading worker ...", flush=True)
worker = LocateAnythingWorker(MODEL_PATH)
print(f"[smoke] model loaded in {time.time()-t0:.1f}s; attn={worker.model.language_model.model._attn_implementation}", flush=True)

img = Image.open(IMG_PATH).convert("RGB")
w, h = img.size
print(f"[smoke] image size: {w}x{h}", flush=True)
results = {}

# Task 1: multi-category detection
t1 = time.time()
res = worker.detect(img, ["person", "bus", "car"], generation_mode="hybrid", max_new_tokens=4096, verbose=True)
ans = res["answer"]; stats = res.get("stats", "")
boxes = parse_ref_boxes(ans)
results["detect"] = {"raw": ans, "stats": stats, "boxes": boxes}
print("[smoke] detect person/bus/car raw:", ans, flush=True)
print("[smoke] parsed boxes:", boxes, flush=True)

# Task 2: phrase grounding
t2 = time.time()
res2 = worker.ground_multi(img, "people", generation_mode="hybrid", max_new_tokens=2048, verbose=False)
boxes2 = parse_ref_boxes(res2["answer"])
results["ground_people"] = {"raw": res2["answer"], "boxes": boxes2}
print("[smoke] ground 'people':", res2["answer"], flush=True)

# Task 3: point
t3 = time.time()
res3 = worker.point(img, "the bus", generation_mode="hybrid", max_new_tokens=512, verbose=False)
pts = []
for m in POINT_RE.finditer(res3["answer"]):
    pts.append({"x": int(m.group(1)), "y": int(m.group(2))})
results["point_bus"] = {"raw": res3["answer"], "points": pts}
print("[smoke] point 'the bus':", res3["answer"], flush=True)

# Annotated image
ann = draw_boxes(img.copy(), boxes, w, h, title="detect: person/bus/car")
ann.save(f"{OUT_DIR}/smoke_real_annotated.jpg", quality=92)
img.save(f"{OUT_DIR}/smoke_real_input.jpg", quality=92)

with open(f"{OUT_DIR}/smoke_real_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[smoke] times: total={time.time()-t0:.1f}s detect={t2-t1:.1f}s ground={t3-t2:.1f}s", flush=True)
print("[smoke] outputs:", f"{OUT_DIR}/smoke_real_annotated.jpg", f"{OUT_DIR}/smoke_real_results.json", flush=True)
print("[smoke] SUCCESS", flush=True)
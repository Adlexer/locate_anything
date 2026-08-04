#!/usr/bin/env python3
"""Re-render annotated image from saved JSON with improved ref+multi-box parser."""
import json, re
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/mnt/c/Dev/locate_anything/outputs"
with open(f"{OUT_DIR}/smoke_real_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

def parse_ref_boxes(answer):
    items, current_label = [], None
    pat = re.compile(r"<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>|<box>none</box>")
    for m in pat.finditer(answer):
        if m.group(1) is not None:
            current_label = m.group(1)
        elif m.group(2) is not None:
            items.append({"label": current_label, "x1": int(m.group(2)), "y1": int(m.group(3)),
                          "x2": int(m.group(4)), "y2": int(m.group(5))})
    return items

def draw_boxes(img, items, w, h, title=""):
    draw = ImageDraw.Draw(img)
    palette = [(220,20,60),(30,144,255),(34,139,34),(255,140,0),(138,43,226),(0,206,209)]
    try:
        font = ImageFont.load_default(size=20)
    except Exception:
        font = ImageFont.load_default()
    for i, it in enumerate(items):
        x1 = it["x1"]/1000*w; y1 = it["y1"]/1000*h
        x2 = it["x2"]/1000*w; y2 = it["y2"]/1000*h
        color = palette[i % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'{it.get("label","")} {it["x1"]},{it["y1"]},{it["x2"]},{it["y2"]}'
        draw.text((x1, max(0, y1-22)), label, fill=color, font=font)
    if title:
        draw.text((10, 10), title, fill="black", font=font)
    return img

img = Image.open(f"{OUT_DIR}/smoke_real_input.jpg").convert("RGB")
w, h = img.size
det_boxes = parse_ref_boxes(results["detect"]["raw"])
g_boxes = parse_ref_boxes(results["ground_people"]["raw"])
print("detect parsed boxes:", det_boxes)
print("ground parsed boxes:", g_boxes)
ann = draw_boxes(img.copy(), det_boxes, w, h, title="detect: person/bus/car")
ann.save(f"{OUT_DIR}/smoke_real_annotated.jpg", quality=92)
print("saved annotated:", f"{OUT_DIR}/smoke_real_annotated.jpg", "count=", len(det_boxes))
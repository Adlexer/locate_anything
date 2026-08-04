#!/usr/bin/env python3
"""Create a synthetic dense-text image (screenshot-like) with known ground-truth
text boxes, for LocateAnything OCR / detect_text precision evaluation."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path("/mnt/c/Dev/locate_anything/data")
IMG_PATH = OUT_DIR / "dense_text.png"
GT_PATH = OUT_DIR / "dense_text_gt.json"

W, H = 1280, 1700


def F(size):
    return ImageFont.load_default(size=size)


def text_bbox(draw, xy, text, font):
    return draw.textbbox(xy, text, font=font)


def main():
    img = Image.new("RGB", (W, H), (245, 246, 248))
    draw = ImageDraw.Draw(img)
    gt = []

    title_font = F(34)
    body_font = F(24)
    small_font = F(20)

    # Top navigation bar
    draw.rectangle([0, 0, W, 96], fill=(40, 44, 63))
    nav_items = ["Home", "Products", "Solutions", "Pricing", "Docs", "Support"]
    x = 40
    for it in nav_items:
        draw.text((x, 34), it, fill=(240, 240, 245), font=title_font)
        bb = text_bbox(draw, (x, 34), it, title_font)
        gt.append({"text": it, "x1": x - 8, "y1": 28, "x2": bb[2] + 8, "y2": bb[3] + 8})
        x = bb[2] + 44

    # Hero heading
    hero = "Build once, locate anything."
    draw.text((60, 150), hero, fill=(25, 28, 40), font=F(52))
    bb = text_bbox(draw, (60, 150), hero, F(52))
    gt.append({"text": hero, "x1": 52, "y1": 144, "x2": bb[2] + 8, "y2": bb[3] + 8})

    sub = "A universal visual grounding model for detection, OCR and GUI."
    draw.text((60, 222), sub, fill=(90, 95, 110), font=body_font)
    bb = text_bbox(draw, (60, 222), sub, body_font)
    gt.append({"text": sub, "x1": 52, "y1": 216, "x2": bb[2] + 8, "y2": bb[3] + 8})

    # Sidebar menu
    draw.rectangle([40, 320, 320, 1580], fill=(255, 255, 255), outline=(220, 220, 225), width=1)
    menu = ["Dashboard", "Analytics", "Reports", "Settings", "Billing", "API Keys", "Team", "Activity", "Storage"]
    y = 350
    for m in menu:
        draw.text((64, y), m, fill=(60, 65, 80), font=body_font)
        bb = text_bbox(draw, (64, y), m, body_font)
        gt.append({"text": m, "x1": 48, "y1": y - 6, "x2": 312, "y2": bb[3] + 6})
        y += 66

    # Main content cards
    cards = [
        ("Revenue overview", ["Monthly recurring revenue grew 18%.", "Annual plan conversion rate: 24.5%.",
                              "Net revenue retention reached 131%.", "Churn dropped to 2.1% this quarter."]),
        ("Deployment status", ["All 42 regions are healthy.", "Latest model v3.2 deployed at 06:00 UTC.",
                               "Average inference latency: 31 ms.", "Zero failed requests in the last hour."]),
        ("Customer highlights", ["Acme Corp signed a 3-year enterprise deal.", "Globex migrated 12M records.",
                                 "Initech doubled their API quota.", "Umbrella adopted the on-prem plan."]),
        ("Recent activity", ["User alice created workspace 'vision-lab'.", "API key rotated by admin.",
                             "New dataset 'streetview-2026' indexed.", "Alert threshold updated for p99 latency."]),
    ]
    cx, cw = 380, 840
    cy = 330
    for title, lines in cards:
        draw.rectangle([cx, cy, cx + cw, cy + 300], fill=(255, 255, 255), outline=(220, 220, 225), width=1)
        draw.text((cx + 20, cy + 18), title, fill=(30, 32, 42), font=title_font)
        bb = text_bbox(draw, (cx + 20, cy + 18), title, title_font)
        gt.append({"text": title, "x1": cx + 12, "y1": cy + 12, "x2": cx + cw - 12, "y2": bb[3] + 12})
        ty = bb[3] + 26
        for ln in lines:
            draw.text((cx + 20, ty), ln, fill=(75, 78, 90), font=body_font)
            bb = text_bbox(draw, (cx + 20, ty), ln, body_font)
            gt.append({"text": ln, "x1": cx + 12, "y1": ty - 4, "x2": cx + cw - 12, "y2": bb[3] + 4})
            ty += 58
        cy += 320

    # Footer
    draw.rectangle([0, 1620, W, H], fill=(40, 44, 63))
    foot = "Copyright 2026 LocateAnything Inc. | Privacy Policy | Terms of Service"
    draw.text((40, 1642), foot, fill=(220, 222, 228), font=small_font)
    bb = text_bbox(draw, (40, 1642), foot, small_font)
    gt.append({"text": foot, "x1": 32, "y1": 1636, "x2": bb[2] + 8, "y2": bb[3] + 8})

    img.save(IMG_PATH, quality=95)
    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump({"image": str(IMG_PATH), "width": W, "height": H, "boxes": gt}, f, ensure_ascii=False, indent=2)
    print(f"wrote {IMG_PATH} ({W}x{H}) with {len(gt)} GT boxes -> {GT_PATH}")


if __name__ == "__main__":
    main()

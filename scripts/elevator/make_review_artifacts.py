#!/usr/bin/env python3
"""Generate human visual-review artifacts for the elevator annotation loop.

Reads a dataset dir (images + teacher YOLO txt) plus the cross-check worksheet
(teacher vs YOLO), and emits:
  - <out>/review_worksheet.csv   flat per-box review table (human_decision column)
  - <out>/review_worksheet.md    summary + instructions for the human reviewer
  - <out>/reviewer.html          single-file interactive reviewer (browser, no deps)
  - <out>/review_manifest.json   box payload embedded by reviewer.html

Usage (any python):
    python scripts/elevator/make_review_artifacts.py \
        --data /mnt/c/Data/datasets/elevator_sample \
        --crosscheck /mnt/c/Dev/locate_anything/outputs/elevator_crosscheck/worksheet.jsonl \
        --out /mnt/c/Dev/locate_anything/outputs/elevator_review
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_teacher(data):
    classes = [
        ln.strip()
        for ln in (Path(data) / "classes.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    boxes = {}
    for p in Path(data).rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or "_result" in p.name:
            continue
        txt = p.with_suffix(".txt")
        rel = str(p.relative_to(data)).replace("\\", "/")
        bl = []
        if txt.exists():
            for raw in txt.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw.split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                if 0 <= cid < len(classes):
                    bl.append(
                        {
                            "class": classes[cid],
                            "xyxy": [
                                round(cx - w / 2, 4),
                                round(cy - h / 2, 4),
                                round(cx + w / 2, 4),
                                round(cy + h / 2, 4),
                            ],
                        }
                    )
        boxes[rel] = bl
    return classes, boxes


def load_crosscheck(path):
    rows = {}
    if path and Path(path).exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            rows[m["image"]] = m["verdicts"]
    return rows


def main():
    ap = argparse.ArgumentParser(description="Generate elevator review artifacts")
    ap.add_argument("--data", required=True, help="dataset dir (images + teacher txt)")
    ap.add_argument(
        "--crosscheck", default=None, help="cross_check.py worksheet.jsonl (optional)"
    )
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--title", default="Elevator Sample Review")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes, teacher = load_teacher(data)
    cross = load_crosscheck(args.crosscheck)

    images = sorted(teacher)
    rows = []
    box_id = 0
    payload = []
    for rel in images:
        t_boxes = teacher[rel]
        v = cross.get(rel)
        y_boxes = []
        if v:
            y_boxes = [
                x["yolo"]
                for x in v
                if x["yolo"] is not None and x["yolo"]["class"] in classes
            ]
        box_meta = []
        for i, b in enumerate(t_boxes):
            verdict = None
            iou = None
            if v and i < len(v):
                verdict = v[i]["verdict"]
                iou = v[i].get("iou")
            box_id += 1
            rows.append(
                {
                    "box_id": box_id,
                    "image": rel,
                    "teacher_class": b["class"],
                    "teacher_xyxy": b["xyxy"],
                    "yolo_verdict": verdict or "",
                    "iou": iou or "",
                    "human_decision": "",
                }
            )
            box_meta.append(
                {
                    "id": box_id,
                    "class": b["class"],
                    "xyxy": b["xyxy"],
                    "yolo_verdict": verdict or "",
                }
            )
        img_out = out / "images" / rel.replace("/", "_")
        img_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((data / rel).resolve(), img_out)
        payload.append(
            {
                "image": rel,
                "src": "images/" + rel.replace("/", "_"),
                "boxes": box_meta,
                "yolo_boxes": [
                    {
                        "class": y["class"],
                        "xyxy": [round(x, 4) for x in y["xyxy"]],
                        "conf": y.get("conf"),
                    }
                    for y in y_boxes
                ],
            }
        )

    # CSV
    with open(out / "review_worksheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    # MD
    n_agree = sum(1 for r in rows if r["yolo_verdict"] == "agree")
    n_mismatch = sum(1 for r in rows if r["yolo_verdict"] == "class_mismatch")
    n_teacher_only = sum(1 for r in rows if r["yolo_verdict"] == "teacher_only")
    n_yolo_only = sum(1 for r in rows if r["yolo_verdict"] == "yolo_only")
    md = "\n".join(
        [
            f"# {args.title} — 人工视觉复核任务",
            "",
            f"- 图片数：{len(images)}；教师框数：{len(rows)}",
            f"- 教师-YOLO 交叉校验：agree={n_agree} class_mismatch={n_mismatch} teacher_only={n_teacher_only} yolo_only={n_yolo_only}",
            "",
            "## 操作步骤",
            "1. 用浏览器打开 `reviewer.html`（推荐 Chrome/Edge，双击即可，无需联网）；",
            "2. 逐图审阅：教师框为实线（绿=gas cylinder，蓝=electric scooter，红=bicycle），YOLO 框为橙色虚线；",
            "3. 对每个教师框按快捷键裁决：`1`=正确保留，`2`=类别错误，`3`=框不准/应删除，`4`=漏检（画面还有目标未标）——`4` 会在图片上标注待补区域（可拖拽）",
            "   - 也可点击框上按钮；`←`/`→` 切换图片，`R` 重置当前图裁决，`D` 下载裁决 JSON；",
            "4. 完成（或分批）后点 `下载裁决`，把 `decisions.json` 交给 Agent（或放回 `outputs/elevator_review/`）；",
            "5. Agent 用裁决结果更新 txt → 得到该批图片的**人工真值 GT**，再算教师/YOLO 的**真实泛化 F1**。",
            "",
            "## 裁决字段说明（CSV/JSON）",
            "| 字段 | 含义 |",
            "|---|---|",
            "| box_id | 教师框编号 |",
            "| image | 图片相对路径 |",
            "| teacher_class | 教师类别（gas cylinder / electric scooter / bicycle） |",
            "| teacher_xyxy | 教师框坐标 [x1,y1,x2,y2]（归一化 0-1） |",
            "| yolo_verdict | 交叉校验结论（agree/class_mismatch/teacher_only/yolo_only/空=无YOLO框） |",
            "| human_decision | 你填写的裁决（accept / wrong_class / delete / missing） |",
            "",
            "## 技巧",
            "- teacher_only：教师检出而 YOLO 未检出 → 重点看是否为误检；",
            "- yolo_only：YOLO 检出而教师未检出 → 重点看是否为漏检（真目标）；",
            "- class_mismatch：两模型类别不一致 → 人工定类别。",
        ]
    )
    (out / "review_worksheet.md").write_text(md + "\n", encoding="utf-8")

    # HTML payload
    manifest = {
        "title": args.title,
        "classes": classes,
        "images": payload,
        "notes": "decisions.json: {box_id: 'accept'|'wrong_class'|'delete'|'missing', missing: [{image, x1,y1,x2,y2}]}",
    }
    (out / "review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tpl = Path(__file__).with_name("reviewer_template.html").read_text(encoding="utf-8")
    html = tpl.replace("__TITLE__", args.title).replace(
        "__PAYLOAD__", json.dumps(manifest, ensure_ascii=False)
    )
    (out / "reviewer.html").write_text(html, encoding="utf-8")

    print(f"[review] images={len(images)} teacher_boxes={len(rows)}")
    print(
        f"[review] agree={n_agree} mismatch={n_mismatch} teacher_only={n_teacher_only} yolo_only={n_yolo_only}"
    )
    print(
        f"[review] -> {out}/review_worksheet.csv .md reviewer.html review_manifest.json"
    )


if __name__ == "__main__":
    main()

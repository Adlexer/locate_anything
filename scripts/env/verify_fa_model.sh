#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locate_anything
python - <<'PY'
import sys, time, re
sys.path.insert(0, "/mnt/c/Dev/locate_anything/Eagle/Embodied")
from locateanything_worker import LocateAnythingWorker
from PIL import Image

t0 = time.time()
worker = LocateAnythingWorker("/home/xu/models/LocateAnything-3B")
print(f"[verify] model loaded in {time.time()-t0:.1f}s")
print("[verify] LLM attn =", worker.model.language_model.model._attn_implementation)
print("[verify] vision config attn =", getattr(worker.model.vision_model.config, "_attn_implementation", None))

img = Image.open("/mnt/c/Dev/locate_anything/data/bus.jpg").convert("RGB")
t1 = time.time()
res = worker.detect(img, ["person", "bus", "car"], generation_mode="hybrid", max_new_tokens=2048, verbose=True)
ans = res["answer"]
boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", ans)
print(f"[verify] detect wall={time.time()-t1:.2f}s boxes={len(boxes)}")
print("[verify] raw:", ans)
print("MODEL_FA_VERIFY_OK")
PY
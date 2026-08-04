#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locate_anything
python - <<'PY'
import sys, time, re
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig
sys.path.insert(0, "/mnt/c/Dev/locate_anything/Eagle/Embodied")

MP = "/home/xu/models/LocateAnything-3B"
t0 = time.time()
config = AutoConfig.from_pretrained(MP, trust_remote_code=True)
print("[t] config._attn_implementation (before) =", config._attn_implementation)
config._attn_implementation = "sdpa"  # LLM: magi -> sdpa (FA2 not implemented in custom Qwen2 forward)
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
model = AutoModel.from_pretrained(MP, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
print(f"[t] loaded in {time.time()-t0:.1f}s")
print("[t] LLM attn =", model.language_model.model._attn_implementation)
print("[t] vision config attn =", getattr(model.vision_model.config, "_attn_implementation", None))

from PIL import Image
img = Image.open("/mnt/c/Dev/locate_anything/data/bus.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "Locate all the instances that matches the following description: person</c>bus</c>car."},
]}]
text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = proc.process_vision_info(messages)
inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
t1 = time.time()
resp = model.generate(
    pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws", None),
    tokenizer=tok, max_new_tokens=2048, use_cache=True,
    generation_mode="hybrid", temperature=0.7, do_sample=True,
    top_p=0.9, top_k=None, repetition_penalty=1.1, verbose=True,
)
ans = resp[0] if isinstance(resp, tuple) else resp
boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", ans)
print(f"[t] detect wall={time.time()-t1:.2f}s boxes={len(boxes)}")
print("[t] raw:", ans)
print("VERIFY_OK")
PY
#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate locate_anything
python - <<'PY'
import torch
from flash_attn import flash_attn_func
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
q = torch.randn(2, 256, 8, 64, dtype=torch.float16, device="cuda")
k = torch.randn_like(q); v = torch.randn_like(q)
out = flash_attn_func(q, k, v, causal=True)
print("flash_attn_func out", tuple(out.shape), out.dtype)
print("FA2_KERNEL_OK")
PY
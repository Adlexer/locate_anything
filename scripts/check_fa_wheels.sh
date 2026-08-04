#!/bin/bash
echo "=== astral cu130 index (cp310) ==="
curl -s --max-time 40 "https://wheels.astral.sh/simple/cu130/flash-attn/" -H "User-Agent: Mozilla/5.0" \
  | grep -oE "flash_attn-[^\"< ]+\.whl" | grep cp310 | head -20
echo "=== conda-forge flash-attn ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda search -c conda-forge flash-attn 2>&1 | tail -20
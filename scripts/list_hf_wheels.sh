#!/bin/bash
# List prebuilt flash-attn wheels on HF wheelhouse repos (cp310 / linux).
for repo in \
  "strangertoolshf/flash_attention_2_wheelhouse" \
  "minhnguyent546/flash-attn-v2.8.3-glibc-2.31" ; do
  echo "===== $repo ====="
  curl -s "https://huggingface.co/api/models/$repo" -H 'User-Agent: Mozilla/5.0' \
    | grep -o '"rfilename":"[^"]*"' | cut -d'"' -f4 \
    | grep -E 'cp310|linux' | head -50
done

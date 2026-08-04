#!/usr/bin/env python3
"""Validate the batch runtime (la_flash vs sdpa) on RTX 5060 Ti.

Measures wall time, decode stats and peak GPU memory for:
  - batch_infer.py --attn sdpa   (baseline)
  - batch_infer.py --attn la_flash --scheduler pipeline   (sparse range plan)
at batch sizes 1 and 4, max_new_tokens 8192.

Run inside the WSL `locate_anything` conda env:
    python /mnt/c/Dev/locate_anything/scripts/bench/bench_la_flash.py
"""
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

PROJECT = Path("/mnt/c/Dev/locate_anything")
MODEL = "/home/xu/models/LocateAnything-3B"
BATCH_INFER = f"{MODEL}/batch_infer.py"
REQS = "/home/xu/la_flash_reqs.jsonl"
OUT = PROJECT / "outputs" / "bench"

REQUESTS = [
    {"image": "/mnt/c/Dev/locate_anything/data/bus.jpg", "query": "person</c>bus</c>car"},
    {"image": "/mnt/c/Dev/locate_anything/data/bus.jpg", "query": "people"},
    {"image": "/mnt/c/Dev/locate_anything/data/bus.jpg",
     "query": "person</c>bus</c>car</c>traffic light</c>tree</c>building</c>sidewalk"},
    {"image": "/mnt/c/Dev/locate_anything/data/dense_text.png",
     "query": "Detect all the text in box format."},
]

CASES = [
    {"name": "sdpa_b1", "attn": "sdpa", "scheduler": "eager", "batch_size": 1},
    {"name": "laflash_b1", "attn": "la_flash", "scheduler": "pipeline", "batch_size": 1},
    {"name": "laflash_b4", "attn": "la_flash", "scheduler": "pipeline", "batch_size": 4},
    {"name": "sdpa_b4", "attn": "sdpa", "scheduler": "eager", "batch_size": 4},
]


def build_requests():
    with open(REQS, "w", encoding="utf-8") as f:
        for r in REQUESTS:
            f.write(json.dumps(r) + "\n")


class GpuSampler(threading.Thread):
    def __init__(self, log_path):
        super().__init__(daemon=True)
        self.log_path = log_path
        self._stop_event = threading.Event()

    def run(self):
        cmd = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
               "--format=csv,noheader,nounits", "-l", "1"]
        with open(self.log_path, "w") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            while not self._stop_event.is_set() and p.poll() is None:
                time.sleep(0.2)
            if p.poll() is None:
                p.terminate()

    def stop(self):
        self._stop_event.set()


def peak_from_log(log_path):
    peak_mem = 0
    peak_util = 0
    try:
        for line in open(log_path):
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    mem = int(parts[0].strip())
                    util = int(parts[1].strip())
                    peak_mem = max(peak_mem, mem)
                    peak_util = max(peak_util, util)
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return peak_mem, peak_util


def run_case(case, env):
    OUT.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT / f"{case['name']}_out.jsonl"
    gpu_log = OUT / f"{case['name']}_gpu.log"
    cmd = [
        "python", BATCH_INFER,
        "--model", MODEL,
        "--attn", case["attn"],
        "--scheduler", case["scheduler"],
        "--batch-size", str(case["batch_size"]),
        "--max-new-tokens", "8192",
        "--requests", REQS,
        "--out", str(out_jsonl),
    ]
    sampler = GpuSampler(gpu_log)
    sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        return {"name": case["name"], "error": "timeout"}
    finally:
        sampler.stop()
        sampler.join(timeout=3)
    wall = time.time() - t0
    peak_mem, peak_util = peak_from_log(gpu_log)
    # parse stats from output jsonl
    rows = []
    if out_jsonl.exists():
        for line in open(out_jsonl, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {
        "name": case["name"],
        "attn": case["attn"],
        "scheduler": case["scheduler"],
        "batch_size": case["batch_size"],
        "n_requests": len(rows),
        "wall_s": round(wall, 3),
        "peak_vram_mb": peak_mem,
        "peak_gpu_util": peak_util,
        "rc": proc.returncode,
        "stderr_tail": (proc.stderr or "")[-500:],
        "sample_rows": rows[:2],
    }


def main():
    build_requests()
    env = dict(__import__("os").environ)
    env["LA_FLASH_MODEL"] = MODEL
    env["LA_FLASH_ATTN"] = "la_flash"
    env["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"
    results = []
    for case in CASES:
        print(f"[bench] running {case['name']} ...", flush=True)
        res = run_case(case, env)
        results.append(res)
        print(f"[bench] {case['name']}: wall={res.get('wall_s')}s peak_vram={res.get('peak_vram_mb')}MB "
              f"rc={res.get('rc')} rows={res.get('n_requests')}", flush=True)
    with open(OUT / "la_flash_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[bench] wrote {OUT / 'la_flash_results.json'}")


if __name__ == "__main__":
    main()

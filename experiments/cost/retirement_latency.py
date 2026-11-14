"""Cache-retirement (zeroisation) latency, measured on its own.

Retirement is the only deadline-path work of a ready shadow: select the shadow handle and zero every tensor of
the live cache. Measured with CUDA events on a persistent cache (no allocation or clone in the timed region),
for the per-tensor loop used by the runtime and for a single grouped call. Also reports cache bytes and the
number of tensors. Run alone on the GPU.
"""

from __future__ import annotations

from shadowveil.hostload import snapshot
from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import MambaCache


def ev(fn, reps=60, warm=10):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return {"median_ms": statistics.median(out), "sd_ms": statistics.pstdev(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="state-spaces/mamba-130m-hf,state-spaces/mamba-370m-hf,state-spaces/mamba-1.4b-hf")
    ap.add_argument("--output", default="results/retirement_latency.json")
    args = ap.parse_args()
    h0 = snapshot()
    rows = []
    for name in args.models.split(","):
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
        cache = MambaCache(model.config, max_batch_size=1, dtype=torch.float16, device="cuda")
        tensors = cache.ssm_states + cache.conv_states
        for t in tensors:
            t.normal_()
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        loop = ev(lambda: [t.zero_() for t in tensors])
        grouped = ev(lambda: torch._foreach_zero_(tensors))
        verify = ev(lambda: max(float(t.float().abs().max()) for t in tensors), reps=20, warm=3)
        rows.append({"model": name, "tensors": len(tensors), "cache_bytes": nbytes,
                     "zero_loop": loop, "zero_grouped": grouped, "verify_max_abs": verify})
        print(json.dumps(rows[-1]), flush=True)
        del model, cache, tensors
        torch.cuda.empty_cache()
    write_json(args.output, {"status": "retire_latency", "rows": rows, "host_load": {"before": h0, "after": snapshot(1.0)}})


if __name__ == "__main__":
    main()

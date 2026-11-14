"""Selective-scan kernel latency of the fused path, isolated from projections, convolution and norms.

Times the Triton scan kernel on random inputs at the layer shapes of the Mamba-1 checkpoints and at several chunk lengths
(n = 1 is a decode step, larger n a burst). The same shapes are timed with the official `mamba_ssm` kernel by
scan_bench_official.py; scan_bench_compare.py joins the two files. Batch 1, fp16 activations, fp32 state accumulation.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json
from shadowveil.hostload import snapshot

import argparse
import statistics
from pathlib import Path

import torch
import triton

from shadowveil.fused import _ssm_scan_kernel

SHAPES = {"130m": (1536, 16), "370m": (2048, 16), "1p4b": (4096, 16)}   # (d_inner, d_state) of one layer


def ev_time(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="1,16,64,256,1024")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--output", default="results/scan_bench_ours.json")
    args = ap.parse_args()
    h0 = snapshot()
    torch.manual_seed(0)
    rows = []
    for tag, (I, N) in SHAPES.items():
        A = -torch.rand(I, N, device="cuda", dtype=torch.float32) - 0.1
        D = torch.rand(I, device="cuda", dtype=torch.float32)
        for n in [int(x) for x in args.lengths.split(",")]:
            x = torch.randn(n, I, device="cuda", dtype=torch.float16)
            dt = torch.randn(n, I, device="cuda", dtype=torch.float16)
            z = torch.randn(n, I, device="cuda", dtype=torch.float16)
            Bm = torch.randn(n, N, device="cuda", dtype=torch.float16)
            Cm = torch.randn(n, N, device="cuda", dtype=torch.float16)
            y = torch.empty_like(x)
            state = torch.zeros(1, I, N, device="cuda", dtype=torch.float16)
            BLOCK_D = 32

            def run():
                _ssm_scan_kernel[(triton.cdiv(I, BLOCK_D), 1)](state, x, dt, A, Bm, Cm, D, z, y, n, I, N=N, BLOCK_D=BLOCK_D)

            for _ in range(5):
                run()
            ts = []
            for _ in range(args.repeats):
                ts.append(ev_time(run))
            rows.append({"scale": tag, "d_inner": I, "d_state": N, "tokens": n,
                         "median_ms": statistics.median(ts), "sd_ms": statistics.pstdev(ts)})
            print(tag, n, round(rows[-1]["median_ms"], 4), "ms", flush=True)
    write_json(Path(args.output), {"status": "scan_kernel_latency", "kernel": "shadowveil Triton selective scan (sequential over tokens)",
                                   "gpu": torch.cuda.get_device_name(0), "rows": rows,
                                   "host_load": {"before": h0, "after": snapshot(1.0)}})


if __name__ == "__main__":
    main()

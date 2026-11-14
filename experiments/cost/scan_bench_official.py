"""Selective-scan kernel latency of the official `mamba_ssm` CUDA kernel at the same shapes as scan_bench_ours.py.

The official package is installed in a WSL environment (Ubuntu-22.04, ~/sv-wsl). Run from Windows this script starts itself
inside WSL; inside WSL it times `selective_scan_fn` (zero initial state; the official Mamba-1 scan has no initial-state
argument). Same GPU, same shapes, sequential runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

SHAPES = {"130m": (1536, 16), "370m": (2048, 16), "1p4b": (4096, 16)}


def inner(args):
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    def ev_time(fn):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    torch.manual_seed(0)
    rows = []
    for tag, (I, N) in SHAPES.items():
        A = -torch.rand(I, N, device="cuda", dtype=torch.float32) - 0.1
        D = torch.rand(I, device="cuda", dtype=torch.float32)
        for n in [int(x) for x in args.lengths.split(",")]:
            u = torch.randn(1, I, n, device="cuda", dtype=torch.float16)
            dt = torch.randn(1, I, n, device="cuda", dtype=torch.float16)
            z = torch.randn(1, I, n, device="cuda", dtype=torch.float16)
            Bm = torch.randn(1, N, n, device="cuda", dtype=torch.float16)
            Cm = torch.randn(1, N, n, device="cuda", dtype=torch.float16)

            def run():
                selective_scan_fn(u, dt, A, Bm, Cm, D, z=z, delta_softplus=True)

            for _ in range(5):
                run()
            ts = [ev_time(run) for _ in range(args.repeats)]
            rows.append({"scale": tag, "d_inner": I, "d_state": N, "tokens": n,
                         "median_ms": statistics.median(ts), "sd_ms": statistics.pstdev(ts)})
            print(tag, n, round(rows[-1]["median_ms"], 4), "ms", flush=True)
    out = {"status": "scan_kernel_latency", "kernel": "official mamba_ssm selective_scan_fn (zero initial state)",
           "gpu": torch.cuda.get_device_name(0), "mamba_ssm": __import__("mamba_ssm").__version__, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.output) + ".tmp")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    tmp.replace(args.output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="1,16,64,256,1024")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--output", default="results/scan_bench_official.json")
    ap.add_argument("--inner", action="store_true")
    args = ap.parse_args()
    if args.inner or platform.system() != "Windows":
        return inner(args)
    here = Path(__file__).resolve()
    root = here.parents[2]
    wsl_root = "/mnt/" + root.drive[0].lower() + str(root.as_posix())[2:]
    cmd = (f"source ~/sv-wsl/bin/activate && cd {wsl_root} && python experiments/cost/scan_bench_official.py --inner "
           f"--lengths {args.lengths} --repeats {args.repeats} --output {Path(args.output).as_posix()}")
    from shadowveil.hostload import snapshot
    h0 = snapshot()
    code = subprocess.call(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", cmd], cwd=root)
    if code:
        sys.exit(code)
    out = root / args.output
    d = json.loads(out.read_text(encoding="utf-8"))
    d["host_load"] = {"before": h0, "after": snapshot(1.0)}
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(out)


if __name__ == "__main__":
    main()

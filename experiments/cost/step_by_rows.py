"""Decode-step latency of the fused CUDA-graph path as a function of the number of batch rows.

The scheduler simulation needs the cost of a batched decode step at the row counts a server actually runs
(tens to hundreds of rows), so the step is timed at every row count in --rows, not extrapolated from small batches.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json
from shadowveil.hostload import snapshot

import argparse
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

from shadowveil.chunked import clone_cache
from shadowveil.fused import FusedMamba

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mechanism"))
from fused_lookahead_sweep import empty, ev_time  # noqa: E402
from snapshot_and_batch_tests import build_cases, stack_caches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--rows", default="1,2,4,8,16,32,48,64,96,128,192,256,384,512")
    ap.add_argument("--max-cache-gb", type=float, default=1.5, help="skip row counts whose stacked cache exceeds this")
    ap.add_argument("--repeats", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    h0 = snapshot()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    fm = FusedMamba(model)
    case = build_cases(tok, 1, 64, args.seed)[0]
    base = empty(model)
    with torch.inference_mode():
        fm.forward(base, torch.tensor([case["P"]], device="cuda"))
        one = sum(t.numel() * t.element_size() for t in base.ssm_states + base.conv_states)
        out = {}
        for B in [int(x) for x in args.rows.split(",")]:
            if 2 * one * B > args.max_cache_gb * 2**30:
                print(f"skipping {B} rows: cache would exceed {args.max_cache_gb} GB", flush=True)
                continue
            cache = stack_caches(model, [clone_cache(base, model.config) for _ in range(B)], "cuda")
            rep, ibuf, _ = fm.capture_step(cache, B)
            tok_b = torch.tensor([[case["A"][0]]] * B, device="cuda")
            for _ in range(5):
                rep(tok_b)
            ts = [ev_time(lambda: rep(tok_b)) for _ in range(args.repeats)]
            out[B] = {"median": statistics.median(ts), "sd": statistics.pstdev(ts)}
            print(B, round(out[B]["median"], 3), "ms", flush=True)
            del cache, rep, ibuf
            torch.cuda.empty_cache()
    write_json(Path(args.out_dir) / f"step_rows_{args.tag}.json",
               {"status": "decode_step_by_rows", "model": args.model, "gpu": torch.cuda.get_device_name(0),
                "backend": "shadowveil.fused (Triton selective scan + CUDA graph)", "step_ms_by_rows": {str(k): v for k, v in out.items()},
                "host_load": {"before": h0, "after": snapshot(1.0)}})


if __name__ == "__main__":
    main()

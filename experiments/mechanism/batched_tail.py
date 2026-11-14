"""Measured dual-decode tail with the shadow as a BATCHED row.

In a serving system the shadow's tail steps share the live sequence's batched decode call. Here the tail is
executed exactly that way: rows = [live, shadow] in one CUDA-graph step (B=2) versus the live row alone
(B=1). Marginal cost of the shadow = t(B=2) - t(B=1). Exactness: the shadow row is compared bitwise with a
reference built by the same schedule in the same batch geometry (with a different companion row), which
the HF-path K2 test showed to be the condition for bitwise equality.
"""

from __future__ import annotations

from shadowveil.hostload import snapshot
from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

from shadowveil.chunked import cache_tensors
from shadowveil.fused import FusedMamba
from fused_lookahead_sweep import copy_cache_into, empty, ev_time, graph_chunk
from snapshot_and_batch_tests import build_cases, stack_caches


@torch.inference_mode()
def set_row(stacked, row, src):
    for a, b in zip(cache_tensors(stacked), cache_tensors(src)):
        a[row].copy_(b[0])


@torch.inference_mode()
def rows_equal(a, ra, b, rb):
    return all(torch.equal(x[ra], y[rb]) for x, y in zip(cache_tensors(a), cache_tensors(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--tag", default="130m")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--deltas", default="16,32,64,128,256")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    h0 = snapshot()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    fm = FusedMamba(model)
    L = args.window
    case = build_cases(tok, 1, L, args.seed)[0]
    P, X, gX, A = case["P"], case["X"], case["gX"], case["A"]
    T = lambda ids: torch.tensor([ids], dtype=torch.long, device="cuda")
    s0 = empty(model)
    fm.forward(s0, T(P))
    rows = []
    for delta in [int(x) for x in args.deltas.split(",")]:
        burst, tail = gX + A[: L - delta], A[L - delta:]
        live_pre = empty(model)
        fm.forward(live_pre, T(P + X))
        if L - delta > 0:
            fm.forward(live_pre, T(A[: L - delta]))
        shadow, ref, dummy = empty(model), empty(model), empty(model)
        copy_cache_into(shadow, s0)
        copy_cache_into(ref, s0)
        fm.forward(shadow, T(burst))
        fm.forward(ref, T(burst))
        # B=1 live-only tail (graph), B=2 [live, shadow] tail (graph), B=2 [dummy, reference] tail (graph)
        live1 = empty(model)
        copy_cache_into(live1, live_pre)
        rep1 = fm.capture_step(live1, 1)[0]
        st2 = stack_caches(model, [live_pre, shadow], "cuda")
        rep2 = fm.capture_step(st2, 2)[0]
        rf2 = stack_caches(model, [dummy, ref], "cuda")
        repr_ = fm.capture_step(rf2, 2)[0]
        t1, t2, exact = [], [], True
        for _ in range(args.repeats):
            copy_cache_into(live1, live_pre)
            set_row(st2, 0, live_pre)
            # rebuild the shadow state after the burst for this repeat
            tmp = empty(model)
            copy_cache_into(tmp, s0)
            fm.forward(tmp, T(burst))
            set_row(st2, 1, tmp)
            set_row(rf2, 0, dummy)
            set_row(rf2, 1, tmp)
            toks = [torch.tensor([[t]], device="cuda") for t in tail]
            t1.append(ev_time(lambda: [rep1(t) for t in toks]))
            t2.append(ev_time(lambda: [rep2(torch.cat([t, t])) for t in toks]))
            for t in toks:
                repr_(torch.cat([t, t]))
            torch.cuda.synchronize()
            exact &= rows_equal(st2, 1, rf2, 1)
        row = {"delta": delta, "tail_tokens": len(tail), "tail_b1_ms": statistics.median(t1),
               "tail_b2_ms": statistics.median(t2), "marginal_shadow_row_ms": statistics.median(t2) - statistics.median(t1),
               "marginal_fraction_of_live": (statistics.median(t2) - statistics.median(t1)) / statistics.median(t1),
               "shadow_row_bitwise_equal_to_reference_row": bool(exact)}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    write_json(Path(args.out_dir) / f"batched_tail_{args.tag}.json",
               {"status": "fused_batched_tail", "model": args.model, "window": L, "rows": rows,
                "host_load": {"before": h0, "after": snapshot(1.0)}})


if __name__ == "__main__":
    main()

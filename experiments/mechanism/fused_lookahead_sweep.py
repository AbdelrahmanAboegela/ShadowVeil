"""Measured (not projected) lookahead sweep on the fused Triton path.

For each delta the actual sequence is executed and timed with CUDA events:
  live branch   : prefill(P + X) then L allowed tokens, one CUDA-graph decode step per token
  shadow branch : copy of the s0 snapshot (state after a real non-zero prefix P), ONE fused burst chunk
                  over g(X) + A[:L-delta], then delta graph decode steps
  reference     : an independently rebuilt shadow by the same schedule -> bitwise comparison
  retirement    : zero the live cache, check max |.| = 0
Also measured: graph decode-step latency for batch rows B = 1,2,4,8 (does a shadow row share the batched
step for free?) and fused chunk latency vs tokens. Writes a cost profile in the schema consumed by
scheduler_sim.py so the simulation can run on measured FUSED costs.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

from shadowveil.hostload import snapshot

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

from shadowveil.chunked import cache_bytes, cache_max_diff, cache_tensors, caches_equal, clone_cache
from shadowveil.fused import FusedMamba
from snapshot_and_batch_tests import build_cases, stack_caches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--tag", default="130m")
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--deltas", default="0,16,32,64,128,256")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--lengths", default="8,16,32,64,128,256,512,1024,2048,4096,8192")
    p.add_argument("--exact-cases", type=int, default=8)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def ev_time(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def empty(model, B=1):
    return MambaCache(model.config, max_batch_size=B, dtype=next(model.parameters()).dtype, device="cuda")


def copy_cache_into(dst, src):
    for a, b in zip(cache_tensors(dst), cache_tensors(src)):
        a.copy_(b)


@torch.inference_mode()
def graph_chunk(fm, cache, n):
    """Capture a fused chunk of n tokens bound to `cache`. Returns (replay(ids_list), logits_buffer)."""
    ids = torch.zeros((1, n), dtype=torch.long, device="cuda")
    snapshot = [t.clone() for t in cache_tensors(cache)]
    for _ in range(2):
        fm.forward(cache, ids)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fm.forward(cache, ids)
    for t, s in zip(cache_tensors(cache), snapshot):
        t.copy_(s)

    def replay(token_ids):
        with torch.inference_mode():
            ids.copy_(torch.tensor([token_ids], device="cuda"))
            g.replay()
        return out
    return replay


def main():
    args = parse_args()
    h0 = snapshot()
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    fm = FusedMamba(model)
    L = args.window
    case = build_cases(tok, 1, L, args.seed)[0]
    P, X, gX, A = case["P"], case["X"], case["gX"], case["A"]

    def T(ids):
        return torch.tensor([ids], dtype=torch.long, device="cuda")

    # ---- profile: step latency vs rows, chunk latency vs tokens (graph-replayed) -------------------------
    base = empty(model)
    fm.forward(base, T(P))
    step_ms = {}
    for B in [int(x) for x in args.batches.split(",")]:
        cache = stack_caches(model, [clone_cache(base, model.config) for _ in range(B)], "cuda")
        rep, ibuf, _ = fm.capture_step(cache, B)
        tok_b = torch.tensor([[A[0]]] * B, device="cuda")
        for _ in range(5):
            rep(tok_b)
        ts = [ev_time(lambda: rep(tok_b)) for _ in range(40)]
        step_ms[B] = {"median": statistics.median(ts), "sd": statistics.pstdev(ts)}
    chunk_ms = {}
    for n in [int(x) for x in args.lengths.split(",")]:
        c = clone_cache(base, model.config)
        rep = graph_chunk(fm, c, n)
        ids = (A * (n // len(A) + 1))[:n]
        for _ in range(3):
            rep(ids)
        ts = [ev_time(lambda: rep(ids)) for _ in range(20)]
        chunk_ms[n] = {"median": statistics.median(ts), "sd": statistics.pstdev(ts)}
    ns = np.array(sorted(chunk_ms), dtype=float)
    cm = np.array([chunk_ms[int(n)]["median"] for n in ns])
    b_fit, a_fit = np.polyfit(ns, cm, 1)
    Bs = np.array(sorted(step_ms), dtype=float)
    c1, c0 = np.polyfit(Bs, np.array([step_ms[int(b)]["median"] for b in Bs]), 1)
    _rc = clone_cache(base, model.config)
    retire = [ev_time(lambda: [t.zero_() for t in cache_tensors(_rc)]) for _ in range(20)]
    nbytes = cache_bytes(base)

    # ---- delta sweep: actual sequence, timed ---------------------------------------------------------------
    s0 = empty(model)
    fm.forward(s0, T(P))                                              # snapshot: non-zero state after prefix
    live = empty(model)
    shadow = empty(model)
    ref = empty(model)
    rows = []
    for delta in [int(x) for x in args.deltas.split(",")]:
        burst = gX + A[: L - delta]
        tail = A[L - delta:]
        rep_sh = graph_chunk(fm, shadow, len(burst))
        rep_rf = graph_chunk(fm, ref, len(burst))
        step_sh = fm.capture_step(shadow, 1)[0] if tail else None
        step_rf = fm.capture_step(ref, 1)[0] if tail else None
        step_lv = fm.capture_step(live, 1)[0]
        copy_cache_into(live, empty(model))
        fm.forward(live, T(P + X))
        reps = {"burst_ms": [], "tail_ms": [], "live_ms": [], "total_extra_ms": []}
        exact_all, retired = True, 0.0
        for r in range(args.repeats):
            copy_cache_into(shadow, s0)
            copy_cache_into(ref, s0)
            copy_cache_into(live, empty(model))
            fm.forward(live, T(P + X))
            def run_tail(rep_step, cache_tokens):
                for t in cache_tokens:
                    rep_step(torch.tensor([[t]], device="cuda"))
            tb = ev_time(lambda: rep_sh(burst))
            tt = ev_time(lambda: run_tail(step_sh, tail)) if tail else 0.0
            tl = ev_time(lambda: run_tail(step_lv, A))
            rep_rf(burst)
            if tail:
                run_tail(step_rf, tail)
            torch.cuda.synchronize()
            exact_all &= caches_equal(shadow, ref)
            live_copy = clone_cache(live, model.config)
            for t in cache_tensors(live):
                t.zero_()
            torch.cuda.synchronize()
            retired = max(retired, max(float(t.float().abs().max()) for t in cache_tensors(live)))
            reps["burst_ms"].append(tb)
            reps["tail_ms"].append(tt)
            reps["live_ms"].append(tl)
            reps["total_extra_ms"].append(tb + tt)
        row = {"delta": delta, "burst_tokens": len(burst), "tail_tokens": len(tail), "exact_bitwise": bool(exact_all),
               "retired_max_abs": retired,
               **{k: {"median": statistics.median(v), "sd": statistics.pstdev(v)} for k, v in reps.items()}}
        rows.append(row)
        print(json.dumps({"delta": delta, "burst_ms": round(row["burst_ms"]["median"], 2),
                          "tail_ms": round(row["tail_ms"]["median"], 2), "exact": exact_all}), flush=True)

    # exactness over additional documents (every delta), eager fused path, same schedule twice
    extra_rows = []
    deltas_all = [int(x) for x in args.deltas.split(",")]
    for ci, cs in enumerate(build_cases(tok, args.exact_cases, L, args.seed)[1:], start=1):
        P2, gX2, A2 = cs["P"], cs["gX"], cs["A"]
        s0b = empty(model)
        fm.forward(s0b, T(P2))
        for delta in deltas_all:
            burst2, tail2 = gX2 + A2[: L - delta], A2[L - delta:]
            a_c, b_c = empty(model), empty(model)
            copy_cache_into(a_c, s0b)
            copy_cache_into(b_c, s0b)
            before = fm.token_transitions
            fm.forward(a_c, T(burst2))
            for t in tail2:
                fm.forward(a_c, T([t]))
            shadow_transitions = fm.token_transitions - before
            fm.forward(b_c, T(burst2))
            for t in tail2:
                fm.forward(b_c, T([t]))
            extra_rows.append({"case": ci, "delta": delta, "exact_bitwise": bool(caches_equal(a_c, b_c)),
                               "shadow_token_transitions_counted": shadow_transitions,
                               "expected_g_plus_window": len(gX2) + L})
    print(json.dumps({"extra_cases_checked": len(extra_rows), "all_exact": all(r["exact_bitwise"] for r in extra_rows),
                      "transitions_conserved": all(r["shadow_token_transitions_counted"] == r["expected_g_plus_window"] for r in extra_rows)}), flush=True)

    # numerical agreement of the fused path with the HF path (documented, not bitwise)
    from shadowveil.chunked import prefill
    hf, _ = prefill(model, P + gX + A[:64], "cuda")
    fz = empty(model)
    fm.forward(fz, T(P + gX + A[:64]))
    num = cache_max_diff(fz, hf) / max(float(t.float().abs().max()) for t in cache_tensors(hf))

    out = {
        "status": "fused_sweep", "model": args.model, "tag": args.tag, "window": L,
        "gpu": torch.cuda.get_device_name(0), "backend": "Triton selective-scan + CUDA-graph decode step (shadowveil.fused)",
        "fused_vs_hf_relative_state_difference": num, "cache_bytes": nbytes,
        "step_ms_by_batch_rows": {str(k): v for k, v in step_ms.items()},
        "chunk_ms_by_tokens": {str(k): v for k, v in chunk_ms.items()},
        "retire_ms": {"median": statistics.median(retire), "sd": statistics.pstdev(retire)},
        "fit": {"chunk_a_ms": float(a_fit), "chunk_b_ms_per_token": float(b_fit),
                "step_c0_ms": float(c0), "step_c1_ms_per_extra_row": float(c1)},
        "delta_sweep": rows,
        "exactness_additional_documents": extra_rows,
    }
    out["host_load"] = {"before": h0, "after": snapshot(1.0)}
    write_json(Path(args.out_dir) / f"fused_sweep_{args.tag}.json", out)
    prof = {k: out[k] for k in ("status", "model", "gpu", "backend", "cache_bytes", "step_ms_by_batch_rows",
                                "chunk_ms_by_tokens", "retire_ms", "fit", "host_load")}
    write_json(Path(args.out_dir) / f"cost_profile_fused_{args.tag}.json", prof)
    print(json.dumps({"fit": out["fit"], "step": {k: round(v["median"], 3) for k, v in step_ms.items()},
                      "chunk": {k: round(v["median"], 2) for k, v in chunk_ms.items()}, "num_vs_hf": num}, indent=1))


if __name__ == "__main__":
    main()

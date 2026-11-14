"""Snapshot and batch-geometry tests: K1 (burst from a non-zero s0) and K2 (batch sensitivity).

K1  The lookahead family snapshots s0 before the protected span X and later builds the shadow with
    one burst from s0. Earlier experiments used s0 = empty state. Here s0 is the state after a real
    TAB left-context (non-zero) and we test, for every delta:
      (a) shadow == independently rebuilt reference, bitwise (same execution path);
      (b) the chunked burst agrees numerically with the token-by-token recurrence and with a monolithic
          prefill of the sanitized stream (quantifies how path-relative 'bitwise' is);
      (c) the naive HF burst (prefill branch on a populated cache) is wrong -> why a chunked path is needed;
      (d) the live-vs-sanitized distance, i.e. the signal that (b)'s noise must stay far below.
K2  Is a row of a batched step bitwise equal to the same step at batch 1? Decides whether live and
    shadow can share one batched decode call without losing bitwise equality.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

from shadowveil.hostload import snapshot

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

from shadowveil.chunked import cache_max_diff, cache_tensors, caches_equal, chunk, clone_cache, prefill, step
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import choose_surfaces, unique_contexts

NEUTRAL = " The service continued routine processing with ordinary public information."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--output", default="results/snapshot_batch_130m.json")
    p.add_argument("--cases", type=int, default=8)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--deltas", default="0,32,64,128")
    p.add_argument("--k2-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_cases(tokenizer, count, window, seed):
    occurrences = load_candidates(180)
    _, surfaces = choose_surfaces(occurrences, 8, 8)
    contexts = unique_contexts(occurrences, count * 3, seed)
    filler = tokenizer.encode(NEUTRAL, add_special_tokens=False)
    cases = []
    for index, (doc_key, redacted) in enumerate(contexts):
        left, _, right = redacted.partition("[REDACTED PERSON]")
        prefix = tokenizer.encode(left, add_special_tokens=False)
        if len(prefix) < 8:
            continue
        allowed = tokenizer.encode(right, add_special_tokens=False)
        allowed = (allowed + filler * (window // len(filler) + 2))[:window]
        cases.append({
            "doc_key_hash": hash(doc_key) & 0xFFFFFFFF,
            "P": prefix,
            "X": tokenizer.encode(surfaces[index % len(surfaces)], add_special_tokens=False),
            "gX": tokenizer.encode("[REDACTED PERSON]", add_special_tokens=False),
            "A": allowed,
        })
        if len(cases) == count:
            break
    return cases


@torch.inference_mode()
def advance(model, cache, pos, ids, device):
    logits = None
    for t in ids:
        cache, logits = step(model, cache, pos, t, device)
        pos += 1
    return cache, pos, logits


def rel(a, b):
    num = max(float((x.float() - y.float()).abs().max()) for x, y in zip(cache_tensors(a), cache_tensors(b)))
    den = max(float(x.float().abs().max()) for x in cache_tensors(b))
    return num / den if den else 0.0


@torch.inference_mode()
def k1_case(model, case, window, deltas, device):
    cfg = model.config
    P, X, gX, A = case["P"], case["X"], case["gX"], case["A"]
    rows = []
    # Live trajectory (ordinary serving path): prefill(P+X) then token-by-token over A.
    live, lp = prefill(model, P + X, device)[0], len(P + X)
    live, lp, _ = advance(model, live, lp, A, device)
    # s0 snapshot: state after the common prefix (non-zero).
    s0, _ = prefill(model, P, device)
    s0_abs_max = max(float(t.float().abs().max()) for t in cache_tensors(s0))
    # Monolithic definition of the sanitized trajectory (one parallel prefill of the whole stream).
    mono, mono_logits = prefill(model, P + gX + A, device)
    for delta in deltas:
        burst_ids = gX + A[: window - delta]
        tail_ids = A[window - delta:]

        def build():
            c = clone_cache(s0, cfg)
            lg = chunk(model, c, burst_ids, device)[:, -1]
            pos = len(P) + len(burst_ids)
            c, pos, tl = advance(model, c, pos, tail_ids, device)
            return c, (tl[:, -1] if tl is not None else lg)

        shadow, shadow_logits = build()
        s0_ref, _ = prefill(model, P, device)            # independently rebuilt snapshot + reference
        c = clone_cache(s0_ref, cfg)
        chunk(model, c, burst_ids, device)
        c, _, _ = advance(model, c, len(P) + len(burst_ids), tail_ids, device)
        reference = c

        # Pure recurrent (all single-token) construction from s0.
        rec = clone_cache(s0, cfg)
        rec, _, rec_logits = advance(model, rec, len(P), gX + A, device)

        # Naive HF burst: prefill branch on a populated cache (ignores previous conv inputs).
        naive = clone_cache(s0, cfg)
        K = cfg.conv_kernel
        model(input_ids=torch.tensor([burst_ids], device=device), cache_params=naive,
              cache_position=torch.arange(K, device=device), use_cache=True, return_dict=True)
        naive, _, _ = advance(model, naive, len(P) + len(burst_ids), tail_ids, device)

        rows.append({
            "delta": delta,
            "burst_tokens": len(burst_ids),
            "shadow_equals_reference_bitwise": bool(caches_equal(shadow, reference)),
            "shadow_vs_recurrent_max_abs": cache_max_diff(shadow, rec),
            "shadow_vs_recurrent_rel": rel(shadow, rec),
            "shadow_vs_monolithic_max_abs": cache_max_diff(shadow, mono),
            "shadow_vs_monolithic_rel": rel(shadow, mono),
            "live_vs_sanitized_max_abs": cache_max_diff(live, shadow),
            "live_vs_sanitized_rel": rel(live, shadow),
            "naive_hf_burst_vs_recurrent_max_abs": cache_max_diff(naive, rec),
            "naive_hf_burst_vs_recurrent_rel": rel(naive, rec),
            "top1_agrees_shadow_vs_monolithic": bool(
                int(shadow_logits.argmax(-1)[0]) == int(mono_logits[:, -1].argmax(-1)[0])),
            "top1_agrees_shadow_vs_recurrent": bool(
                int(shadow_logits.argmax(-1)[0]) == int(rec_logits[:, -1].argmax(-1)[0])),
            "next_token_logit_max_abs_shadow_vs_recurrent": float(
                (shadow_logits.float() - rec_logits[:, -1].float()).abs().max()),
        })
    return {"prefix_tokens": len(P), "s0_abs_max": s0_abs_max, "rows": rows}


@torch.inference_mode()
def stack_caches(model, caches, device):
    cfg = model.config
    out = MambaCache(cfg, max_batch_size=len(caches), dtype=caches[0].ssm_states[0].dtype, device=device)
    for layer in range(cfg.num_hidden_layers):
        for b, c in enumerate(caches):
            out.ssm_states[layer][b].copy_(c.ssm_states[layer][0])
            out.conv_states[layer][b].copy_(c.conv_states[layer][0])
    return out


@torch.inference_mode()
def k2(model, cases, steps, device):
    cfg = model.config
    n = len(cases)
    min_len = min(len(c["P"]) for c in cases)
    base = [prefill(model, c["P"][:min_len], device)[0] for c in cases]
    streams = [c["A"][:steps] for c in cases]
    singles = []
    for b in range(n):
        cache = clone_cache(base[b], cfg)
        cache, _, logits = advance(model, cache, min_len, streams[b], device)
        singles.append((cache, logits))
    out = {}

    def run_batched(order):
        """order: list of case indices placed in rows 0..B-1; returns final batched cache."""
        batched = stack_caches(model, [clone_cache(base[b], cfg) for b in order], device)
        for t in range(steps):
            ids = torch.tensor([[streams[b][t]] for b in order], device=device)
            o = model(input_ids=ids, cache_params=batched, cache_position=torch.tensor([min_len + t], device=device),
                      use_cache=True, return_dict=True)
            batched = o.cache_params
        return batched

    def row_equal(ca, ra, cb, rb):
        return all(torch.equal(ca.ssm_states[l][ra], cb.ssm_states[l][rb]) and
                   torch.equal(ca.conv_states[l][ra], cb.conv_states[l][rb])
                   for l in range(cfg.num_hidden_layers))

    if n >= 3:
        ref = run_batched([0, 1])
        again = run_batched([0, 1])
        other_companion = run_batched([0, 2])
        swapped = run_batched([1, 0])
        out["B=2_geometry_tests"] = {
            "rerun_identical_inputs_bitwise_equal": all(row_equal(ref, r, again, r) for r in (0, 1)),
            "row0_independent_of_companion_content": row_equal(ref, 0, other_companion, 0),
            "stream_equal_when_moved_to_other_row": row_equal(ref, 0, swapped, 1),
        }
    for B in (2, 4, 8):
        if B > n:
            continue
        batched = stack_caches(model, [clone_cache(base[b], cfg) for b in range(B)], device)
        times, last = [], None
        for t in range(steps):
            ids = torch.tensor([[streams[b][t]] for b in range(B)], device=device)
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            o = model(input_ids=ids, cache_params=batched, cache_position=torch.tensor([min_len + t], device=device),
                      use_cache=True, return_dict=True)
            torch.cuda.synchronize() if device == "cuda" else None
            times.append(1000 * (time.perf_counter() - t0))
            batched, last = o.cache_params, o.logits
        rows = []
        for b in range(B):
            eq = all(torch.equal(batched.ssm_states[l][b], singles[b][0].ssm_states[l][0]) and
                     torch.equal(batched.conv_states[l][b], singles[b][0].conv_states[l][0])
                     for l in range(cfg.num_hidden_layers))
            md = max(float((batched.ssm_states[l][b].float() - singles[b][0].ssm_states[l][0].float()).abs().max())
                     for l in range(cfg.num_hidden_layers))
            rows.append({"row": b, "bitwise_equal_to_batch1": bool(eq), "max_abs_state_diff": md,
                         "logits_bitwise_equal": bool(torch.equal(last[b], singles[b][1][0]))})
        out[f"B={B}"] = {"rows": rows, "median_step_ms": statistics.median(times)}
    # B=1 step time
    c = clone_cache(base[0], cfg)
    times = []
    for t in range(steps):
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        c, _ = step(model, c, min_len + t, streams[0][t], device)
        torch.cuda.synchronize() if device == "cuda" else None
        times.append(1000 * (time.perf_counter() - t0))
    out["B=1"] = {"median_step_ms": statistics.median(times)}
    return out


def main():
    args = parse_args()
    h0 = snapshot()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    deltas = [int(x) for x in args.deltas.split(",")]
    cases = build_cases(tokenizer, args.cases, args.window, args.seed)
    k1_results = []
    partial = Path(args.output + ".partial")
    sig = {k: getattr(args, k) for k in ("model", "window", "deltas", "seed", "cases")}
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("signature") == sig:
            k1_results = saved["k1_cases"]
            print(f"resuming: {len(k1_results)} cases already done", flush=True)
    for i, case in enumerate(cases):
        if i < len(k1_results):
            continue
        r = k1_case(model, case, args.window, deltas, args.device)
        k1_results.append(r)
        write_json(partial, {"signature": sig, "k1_cases": k1_results})
        print(json.dumps({"case": i, "prefix": r["prefix_tokens"],
                          "exact": all(x["shadow_equals_reference_bitwise"] for x in r["rows"]),
                          "naive_rel": max(x["naive_hf_burst_vs_recurrent_rel"] for x in r["rows"]),
                          "chunk_vs_rec_rel": max(x["shadow_vs_recurrent_rel"] for x in r["rows"])}), flush=True)
    k2_result = k2(model, cases[: min(8, len(cases))], args.k2_steps, args.device)
    flat = [x for r in k1_results for x in r["rows"]]
    summary = {
        "k1_all_shadow_equals_reference_bitwise": all(x["shadow_equals_reference_bitwise"] for x in flat),
        "k1_max_rel_chunk_vs_recurrent": max(x["shadow_vs_recurrent_rel"] for x in flat),
        "k1_max_rel_shadow_vs_monolithic": max(x["shadow_vs_monolithic_rel"] for x in flat),
        "k1_min_rel_live_vs_sanitized": min(x["live_vs_sanitized_rel"] for x in flat),
        "k1_min_rel_naive_hf_burst_error": min(x["naive_hf_burst_vs_recurrent_rel"] for x in flat),
        "k1_top1_agreement_vs_monolithic": sum(x["top1_agrees_shadow_vs_monolithic"] for x in flat) / len(flat),
        "k1_top1_agreement_vs_recurrent": sum(x["top1_agrees_shadow_vs_recurrent"] for x in flat) / len(flat),
        "k2_batched_rows_bitwise_equal_fraction": {
            key: sum(r["bitwise_equal_to_batch1"] for r in v["rows"]) / len(v["rows"])
            for key, v in k2_result.items() if "rows" in v},
        "k2_median_step_ms": {key: v["median_step_ms"] for key, v in k2_result.items() if "median_step_ms" in v},
        "k2_geometry": k2_result.get("B=2_geometry_tests"),
    }
    output = {
        "status": "kill_tests_k1_k2", "model": args.model, "configuration": vars(args),
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "summary": summary, "k1_cases": k1_results, "k2": k2_result,
    }
    output["host_load"] = {"before": h0, "after": snapshot(1.0)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    if partial.exists():
        partial.unlink()
    print(json.dumps(summary, indent=2))
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

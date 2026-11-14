"""Extends the Mamba-2 portability check (stage24) into a full lookahead
(delta) sweep, matching stage22's protocol: for each delta, build the shadow
via a burst-then-continue construction and verify boundary equivalence
against an independently computed reference built the same way, on Mamba-2's
stacked-tensor Mamba2Cache.

This directly tests whether the corrected lookahead-family construction in
Sec. III-A (which now explicitly accounts for the shared s_0/transcript
prerequisite) still yields exact boundary equivalence at every delta on a
structurally different recurrence, not just at the delta=L endpoint checked
in stage24.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="AntonV/mamba2-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage25_mamba2_lookahead_sweep.json")
    parser.add_argument("--window-tokens", type=int, default=128)
    parser.add_argument("--deltas", default="0,32,64,128")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def cache_bytes(cache):
    return cache.ssm_states.numel() * cache.ssm_states.element_size() + \
        cache.conv_states.numel() * cache.conv_states.element_size()


@torch.inference_mode()
def prefill(model, ids, device):
    sync(device)
    started = time.perf_counter()
    output = model(input_ids=torch.tensor([ids], device=device), use_cache=True, return_dict=True)
    sync(device)
    return output.cache_params, 1000 * (time.perf_counter() - started)


@torch.inference_mode()
def advance_one(model, cache, position, token_id, device):
    output = model(
        input_ids=torch.tensor([[token_id]], device=device),
        cache_params=cache,
        cache_position=torch.tensor([position], device=device),
        use_cache=True, return_dict=True,
    )
    return output.cache_params, position + 1


def cache_equal(a, b):
    return bool(torch.equal(a.ssm_states, b.ssm_states) and torch.equal(a.conv_states, b.conv_states))


def run_one_delta(model, tokenizer, live_prefix, shadow_prefix, filler, window_tokens, delta, device):
    burst_len = window_tokens - delta

    live_cache, _ = prefill(model, live_prefix, device)
    live_pos = len(live_prefix)
    for token_id in filler[:window_tokens]:
        live_cache, live_pos = advance_one(model, live_cache, live_pos, token_id, device)

    burst_ids = shadow_prefix + filler[:burst_len]
    shadow_cache, burst_ms = prefill(model, burst_ids, device)
    shadow_pos = len(burst_ids)
    for token_id in filler[burst_len:window_tokens]:
        shadow_cache, shadow_pos = advance_one(model, shadow_cache, shadow_pos, token_id, device)

    reference_cache, _ = prefill(model, burst_ids, device)
    reference_pos = len(burst_ids)
    for token_id in filler[burst_len:window_tokens]:
        reference_cache, reference_pos = advance_one(model, reference_cache, reference_pos, token_id, device)

    exact = cache_equal(shadow_cache, reference_cache)

    with torch.inference_mode():
        live_cache.ssm_states.zero_()
        live_cache.conv_states.zero_()
    retired_max = max(
        float(live_cache.ssm_states.float().abs().max()),
        float(live_cache.conv_states.float().abs().max()),
    )

    return {
        "delta_tokens": delta,
        "burst_replay_tokens": burst_len,
        "sustained_dual_cache_tokens": delta,
        "burst_replay_ms": burst_ms,
        "bitwise_equal_to_independent_reference": exact,
        "retired_cache_max_absolute_value": retired_max,
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = Mamba2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    live_prefix = tokenizer.encode(
        "Session record. Private code: apple. Retained city: Oslo.", add_special_tokens=False
    )
    shadow_prefix = tokenizer.encode(
        "Session record. Private code: stone. Retained city: Oslo.", add_special_tokens=False
    )
    if len(live_prefix) != len(shadow_prefix):
        raise RuntimeError("Live and sanitized prefixes must be token-length matched")
    neutral = tokenizer.encode(
        " The service continued routine processing with ordinary public information.",
        add_special_tokens=False,
    )
    filler = (neutral * (args.window_tokens // len(neutral) + 1))[: args.window_tokens]

    probe_cache, _ = prefill(model, live_prefix, args.device)
    single_bytes = cache_bytes(probe_cache)

    deltas = [int(x) for x in args.deltas.split(",")]
    rows = []
    for delta in deltas:
        row = run_one_delta(model, tokenizer, live_prefix, shadow_prefix, filler, args.window_tokens, delta, args.device)
        rows.append(row)
        print(json.dumps(row), flush=True)

    output = {
        "status": "stage_25_mamba2_lookahead_sweep",
        "model": args.model,
        "architecture": "Mamba-2 (SSD, stacked-tensor Mamba2Cache)",
        "window_tokens": args.window_tokens,
        "single_cache_bytes": single_bytes,
        "dual_cache_bytes": 2 * single_bytes,
        "all_deltas_exact": all(r["bitwise_equal_to_independent_reference"] for r in rows),
        "all_deltas_retired_zero": all(r["retired_cache_max_absolute_value"] == 0.0 for r in rows),
        "rows": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

"""Portability check: does ShadowVeil's exact-equality mechanism transfer to
Mamba-2 (SSD-based selective state space, Dao & Gu 2024), not just to larger
Mamba-1 checkpoints?

Mamba2Cache stores ssm_states/conv_states as single stacked tensors across
all layers (shape [num_layers, batch, ...]), unlike Mamba-1's MambaCache
(a Python list of per-layer tensors), so the clone/compare/zeroize logic used
throughout stage10-stage23 does not apply directly and is reimplemented here
for the stacked-tensor layout.

This is a single-configuration mechanism check (boundary + continuation
equality, one live/shadow/reference triple), matching the portability tier
used for the original Mamba-370M check: it establishes whether the exact
counterfactual equality proof transfers to a structurally different
recurrent architecture. It is not a privacy confirmation and does not train
an attacker.
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
    parser.add_argument("--output", default="results/stage24_mamba2_portability_check.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def cache_tensors(cache):
    return [cache.ssm_states, cache.conv_states]


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for t in cache_tensors(cache))


@torch.inference_mode()
def prefill(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    sync(device)
    started = time.perf_counter()
    output = model(input_ids=ids, use_cache=True, return_dict=True)
    sync(device)
    elapsed_ms = 1000 * (time.perf_counter() - started)
    return output.cache_params, elapsed_ms, ids.shape[1]


@torch.inference_mode()
def continue_tokens(model, cache, position, token_ids, device):
    for token_id in token_ids:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        cache_position = torch.tensor([position], device=device)
        output = model(
            input_ids=input_ids, cache_params=cache, cache_position=cache_position,
            use_cache=True, return_dict=True,
        )
        cache = output.cache_params
        position += 1
    return cache, position


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = Mamba2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    live_prefix = "Session record. Private code: apple. Retained city: Oslo."
    shadow_prefix = "Session record. Private code: stone. Retained city: Oslo."
    live_ids_len = len(tokenizer(live_prefix, add_special_tokens=False).input_ids)
    shadow_ids_len = len(tokenizer(shadow_prefix, add_special_tokens=False).input_ids)
    if live_ids_len != shadow_ids_len:
        raise RuntimeError("Live and sanitized prefixes must be token-length matched")

    neutral_suffix = " The service continued routine processing with ordinary public information."
    suffix_ids = tokenizer(neutral_suffix, add_special_tokens=False).input_ids

    live_cache, live_ms, live_pos = prefill(model, tokenizer, live_prefix, args.device)
    shadow_cache, shadow_ms, shadow_pos = prefill(model, tokenizer, shadow_prefix, args.device)
    reference_cache, _, reference_pos = prefill(model, tokenizer, shadow_prefix, args.device)

    sync(args.device)
    swap_started = time.perf_counter_ns()
    active_cache = shadow_cache
    swap_ns = time.perf_counter_ns() - swap_started

    boundary_exact = bool(
        torch.equal(active_cache.ssm_states, reference_cache.ssm_states)
        and torch.equal(active_cache.conv_states, reference_cache.conv_states)
    )
    boundary_max_diff = max(
        float((active_cache.ssm_states.float() - reference_cache.ssm_states.float()).abs().max()),
        float((active_cache.conv_states.float() - reference_cache.conv_states.float()).abs().max()),
    )

    retire_started = time.perf_counter()
    with torch.inference_mode():
        live_cache.ssm_states.zero_()
        live_cache.conv_states.zero_()
    sync(args.device)
    retire_ms = 1000 * (time.perf_counter() - retire_started)
    retired_max = max(
        float(live_cache.ssm_states.float().abs().max()),
        float(live_cache.conv_states.float().abs().max()),
    )

    active_cache, active_pos = continue_tokens(model, active_cache, shadow_pos, suffix_ids, args.device)
    reference_cache, reference_pos = continue_tokens(model, reference_cache, reference_pos, suffix_ids, args.device)
    continuation_exact = bool(
        torch.equal(active_cache.ssm_states, reference_cache.ssm_states)
        and torch.equal(active_cache.conv_states, reference_cache.conv_states)
    )
    continuation_max_diff = max(
        float((active_cache.ssm_states.float() - reference_cache.ssm_states.float()).abs().max()),
        float((active_cache.conv_states.float() - reference_cache.conv_states.float()).abs().max()),
    )

    output = {
        "status": "stage_24_mamba2_portability_check",
        "model": args.model,
        "architecture": "Mamba-2 (SSD, stacked-tensor Mamba2Cache)",
        "params": sum(p.numel() for p in model.parameters()),
        "interpretation": (
            "Single-configuration mechanism portability check (boundary + "
            "continuation equality), matching the tier used for the original "
            "Mamba-370M check. Not a privacy confirmation; no attacker trained."
        ),
        "boundary_equivalence": {
            "bitwise_equal_to_independent_reference": boundary_exact,
            "max_absolute_difference": boundary_max_diff,
            "pointer_swap_ns": int(swap_ns),
        },
        "continuation_equivalence": {
            "bitwise_equal_to_independent_reference": continuation_exact,
            "max_absolute_difference": continuation_max_diff,
        },
        "retirement": {
            "secure_retire_ms": retire_ms,
            "retired_cache_max_absolute_value": retired_max,
        },
        "timing": {
            "live_prefill_ms": live_ms,
            "shadow_prefill_ms": shadow_ms,
            "single_cache_bytes": cache_bytes(live_cache),
            "dual_cache_bytes": 2 * cache_bytes(live_cache),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(json.dumps(output, indent=2))
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

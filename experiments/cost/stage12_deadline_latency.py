"""Measure the compute/latency trade-off of prospective shadow revocation.

Replay recomputes the sanitized trajectory when deletion is requested. Shadow
revocation computes that trajectory prospectively and pays only a state-handle
swap plus explicit retirement of the contaminated fixed-size cache at expiry.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage12_deadline_latency.json")
    parser.add_argument("--delay-tokens", default="0,16,64,256")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def prefill(model, ids, device):
    sync(device)
    started = time.perf_counter()
    output = model(input_ids=ids.to(device), use_cache=True, return_dict=True)
    sync(device)
    return output.cache_params, 1000 * (time.perf_counter() - started)


def cache_tensors(cache):
    return cache.ssm_states + cache.conv_states


def cache_bytes(cache):
    return sum(tensor.numel() * tensor.element_size() for tensor in cache_tensors(cache))


@torch.inference_mode()
def retire_and_swap(live, shadow, device):
    sync(device)
    swap_started = time.perf_counter_ns()
    active = live
    active = shadow
    swap_ns = time.perf_counter_ns() - swap_started
    retire_started = time.perf_counter()
    for tensor in cache_tensors(live):
        tensor.zero_()
    sync(device)
    retire_ms = 1000 * (time.perf_counter() - retire_started)
    retired_max = max(float(tensor.float().abs().max()) for tensor in cache_tensors(live))
    return active, swap_ns, retire_ms, retired_max


def mean_sd(values):
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    live_prefix = tokenizer.encode(
        "Session record. Private code: apple. Retained city: Oslo.",
        add_special_tokens=False,
    )
    shadow_prefix = tokenizer.encode(
        "Session record. Private code: stone. Retained city: Oslo.",
        add_special_tokens=False,
    )
    if len(live_prefix) != len(shadow_prefix):
        raise RuntimeError("Live and sanitized prefixes must be token-length matched")
    neutral = tokenizer.encode(
        " The service continued routine processing with ordinary public information.",
        add_special_tokens=False,
    )
    expiry = tokenizer.encode(
        " The private code reaches its deletion deadline now.", add_special_tokens=False
    )
    rows = []
    for delay in [int(item) for item in args.delay_tokens.split(",")]:
        repeated = (neutral * (delay // len(neutral) + 1))[:delay]
        live_ids = torch.tensor([live_prefix + repeated + expiry])
        shadow_ids = torch.tensor([shadow_prefix + repeated + expiry])
        # Warm kernels and allocator before measurements.
        prefill(model, live_ids, args.device)
        prefill(model, shadow_ids, args.device)
        live_times, shadow_times, swap_times, retire_times = [], [], [], []
        exact_checks = []
        bytes_per_cache = None
        retired_maxima = []
        for _ in range(args.repeats):
            live, live_ms = prefill(model, live_ids, args.device)
            shadow, shadow_ms = prefill(model, shadow_ids, args.device)
            reference, _ = prefill(model, shadow_ids, args.device)
            active, swap_ns, retire_ms, retired_max = retire_and_swap(
                live, shadow, args.device
            )
            exact_checks.append(all(
                torch.equal(actual, expected)
                for actual, expected in zip(cache_tensors(active), cache_tensors(reference))
            ))
            bytes_per_cache = cache_bytes(active)
            live_times.append(live_ms)
            shadow_times.append(shadow_ms)
            swap_times.append(swap_ns)
            retire_times.append(retire_ms)
            retired_maxima.append(retired_max)
        rows.append({
            "delay_tokens": delay,
            "trajectory_tokens": int(live_ids.shape[1]),
            "live_prefill_ms": mean_sd(live_times),
            "replay_at_deadline_ms": mean_sd(shadow_times),
            "prospective_dual_compute_ms": mean_sd([
                live + shadow for live, shadow in zip(live_times, shadow_times)
            ]),
            "pointer_swap_ns": mean_sd(swap_times),
            "secure_retire_ms": mean_sd(retire_times),
            "revocation_event_ms": mean_sd([
                swap / 1e6 + retire for swap, retire in zip(swap_times, retire_times)
            ]),
            "bitwise_equal_to_redacted_reference": bool(all(exact_checks)),
            "retired_cache_max_absolute_value": float(max(retired_maxima)),
            "single_cache_bytes": int(bytes_per_cache),
            "dual_cache_bytes": int(2 * bytes_per_cache),
        })
        print(json.dumps(rows[-1]), flush=True)
    output = {
        "status": "stage_12_deadline_latency_tradeoff",
        "configuration": vars(args),
        "scope": "single active revocation domain; deterministic input-side state",
        "results": rows,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

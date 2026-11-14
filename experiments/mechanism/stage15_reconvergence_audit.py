"""Test whether divergent Mamba caches reconverge under a shared continuation.

This is a falsification experiment: early retirement is supported only if the
full cache becomes exactly equal, or if a predeclared approximate threshold is
crossed and remains crossed. Aggregate norms alone cannot justify retirement.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, MambaForCausalLM

from shadowveil.runtime import ShadowMambaSession, cache_equal
from stage14_online_runtime_validation import build_cases, longest_common_prefix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--horizons", default="0,8,16,32,64,128,256")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--approx-relative-rms", type=float, default=1e-3)
    parser.add_argument("--output", default="results/stage15_reconvergence_audit.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cache_metrics(live, shadow) -> dict:
    layers = []
    all_sq_diff = 0.0
    all_sq_reference = 0.0
    all_count = 0
    global_max = 0.0
    for layer, (live_ssm, shadow_ssm, live_conv, shadow_conv) in enumerate(
        zip(live.ssm_states, shadow.ssm_states, live.conv_states, shadow.conv_states)
    ):
        layer_sq_diff = 0.0
        layer_sq_reference = 0.0
        layer_count = 0
        layer_max = 0.0
        for left, right in ((live_ssm, shadow_ssm), (live_conv, shadow_conv)):
            difference = left.float() - right.float()
            layer_sq_diff += float(torch.sum(difference * difference))
            layer_sq_reference += float(torch.sum(right.float() * right.float()))
            layer_count += difference.numel()
            layer_max = max(layer_max, float(difference.abs().max()))
        rms = (layer_sq_diff / layer_count) ** 0.5
        reference_rms = (layer_sq_reference / layer_count) ** 0.5
        layers.append({
            "layer": layer,
            "rms_difference": rms,
            "relative_rms_difference": rms / max(reference_rms, 1e-12),
            "max_absolute_difference": layer_max,
        })
        all_sq_diff += layer_sq_diff
        all_sq_reference += layer_sq_reference
        all_count += layer_count
        global_max = max(global_max, layer_max)
    rms = (all_sq_diff / all_count) ** 0.5
    reference_rms = (all_sq_reference / all_count) ** 0.5
    return {
        "bitwise_equal": cache_equal(live, shadow),
        "rms_difference": rms,
        "relative_rms_difference": rms / max(reference_rms, 1e-12),
        "max_absolute_difference": global_max,
        "layers": layers,
    }


def main() -> None:
    args = parse_args()
    horizons = sorted({int(item) for item in args.horizons.split(",")})
    if not horizons or horizons[0] != 0 or any(item < 0 for item in horizons):
        raise ValueError("horizons must be nonnegative and include zero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    continuation_text = (
        " The court then considered the public procedural record, the applicable standard, "
        "and the remaining evidence before issuing its decision."
    )
    continuation_seed = tokenizer.encode(continuation_text, add_special_tokens=False)
    continuation = [continuation_seed[i % len(continuation_seed)] for i in range(max(horizons))]

    rows = []
    for case_index, (live_text, shadow_text) in enumerate(build_cases(args.examples, args.seed)):
        live_ids = tokenizer.encode(live_text, add_special_tokens=False)
        shadow_ids = tokenizer.encode(shadow_text, add_special_tokens=False)
        split = longest_common_prefix(live_ids, shadow_ids)
        session = ShadowMambaSession(model, live_ids[:split], args.device)
        session.advance_live(live_ids[split:])
        session.advance_shadow(shadow_ids[split:])
        measurements = [{"horizon": 0, **cache_metrics(session.live.cache, session.shadow.cache)}]
        previous = 0
        for horizon in horizons[1:]:
            segment = continuation[previous:horizon]
            session.advance_live(segment)
            session.advance_shadow(segment)
            measurements.append({
                "horizon": horizon,
                **cache_metrics(session.live.cache, session.shadow.cache),
            })
            previous = horizon
        rows.append({
            "case": case_index,
            "live_tokens_at_divergence_end": len(live_ids),
            "shadow_tokens_at_divergence_end": len(shadow_ids),
            "measurements": measurements,
        })

    exact_horizons = [
        measurement["horizon"]
        for row in rows
        for measurement in row["measurements"]
        if measurement["bitwise_equal"]
    ]
    approximately_reconverged = []
    for row in rows:
        candidates = [
            measurement["horizon"]
            for measurement in row["measurements"]
            if measurement["relative_rms_difference"] <= args.approx_relative_rms
        ]
        approximately_reconverged.append(min(candidates) if candidates else None)
    initial = [row["measurements"][0]["relative_rms_difference"] for row in rows]
    final = [row["measurements"][-1]["relative_rms_difference"] for row in rows]
    output = {
        "status": "stage_15_reconvergence_falsification_audit",
        "configuration": {**vars(args), "horizons": horizons},
        "data_protocol": "real TAB legal contexts; counterfactual identities; no text emitted",
        "retirement_rule": (
            "exact retirement requires full-cache bitwise equality; approximate threshold is "
            "diagnostic only and is not treated as exact revocation"
        ),
        "any_exact_reconvergence": bool(exact_horizons),
        "exact_reconvergence_horizons": exact_horizons,
        "approximate_threshold": args.approx_relative_rms,
        "approximate_reconvergence_horizon_by_case": approximately_reconverged,
        "median_initial_relative_rms": float(np.median(initial)),
        "median_final_relative_rms": float(np.median(final)),
        "median_relative_rms_ratio_final_to_initial": float(np.median(np.array(final) / np.array(initial))),
        "rows": rows,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2) if args.device == "cuda" else None,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, output, newline=True)
    print(json.dumps({
        "output": str(destination),
        "any_exact_reconvergence": output["any_exact_reconvergence"],
        "approximate_reconvergence_horizon_by_case": approximately_reconverged,
        "median_relative_rms_ratio_final_to_initial": output["median_relative_rms_ratio_final_to_initial"],
        "peak_gpu_memory_mb": output["peak_gpu_memory_mb"],
    }, indent=2))


if __name__ == "__main__":
    main()

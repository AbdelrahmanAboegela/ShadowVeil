"""Validate an actual online ShadowVeil dual-cache session on real text."""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, MambaForCausalLM

from shadowveil.runtime import ShadowMambaSession, cache_equal, cache_max_difference
from stage1_leakage_probe import RETAINED_CITIES
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import choose_surfaces, unique_contexts
from stage10_shadow_revocation_tab import TEST_PREFIX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", default="results/stage14_online_runtime_validation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    size = min(len(left), len(right))
    for index in range(size):
        if left[index] != right[index]:
            return index
    return size


def build_cases(examples: int, seed: int):
    occurrences = load_candidates(180)
    _, surfaces = choose_surfaces(occurrences, 8, 8)
    contexts = unique_contexts(occurrences, examples, seed)
    rng = random.Random(seed)
    cases = []
    for index, (_, redacted_context) in enumerate(contexts):
        identity = surfaces[index % len(surfaces)]
        retained = RETAINED_CITIES[rng.randrange(len(RETAINED_CITIES))]
        live_context = redacted_context.replace("[REDACTED PERSON]", identity, 1)
        live_text = TEST_PREFIX.format(context=live_context, retained=retained)
        shadow_text = TEST_PREFIX.format(context=redacted_context, retained=retained)
        cases.append((live_text, shadow_text))
    return cases


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    continuation = tokenizer.encode(
        " The public routing category remains available for the next request.",
        add_special_tokens=False,
    )

    rows = []
    for live_text, shadow_text in build_cases(args.examples, args.seed):
        live_ids = tokenizer.encode(live_text, add_special_tokens=False)
        shadow_ids = tokenizer.encode(shadow_text, add_special_tokens=False)
        split = longest_common_prefix(live_ids, shadow_ids)
        if split == 0 or split == min(len(live_ids), len(shadow_ids)):
            raise RuntimeError("expected a nonempty shared prefix and a protected divergence")

        session = ShadowMambaSession(model, live_ids[:split], args.device)
        session.advance_live(live_ids[split:])
        session.advance_shadow(shadow_ids[split:])

        reference = ShadowMambaSession(model, shadow_ids[:split], args.device)
        reference.advance_shadow(shadow_ids[split:])
        boundary_equal = cache_equal(session.shadow.cache, reference.shadow.cache)
        boundary_max_difference = cache_max_difference(session.shadow.cache, reference.shadow.cache)

        expiry = session.expire()
        session.advance_active(continuation)
        reference.active = reference.shadow
        reference.advance_active(continuation)
        continuation_equal = cache_equal(session.active.cache, reference.active.cache)
        continuation_max_difference = cache_max_difference(session.active.cache, reference.active.cache)
        rows.append({
            "live_tokens": len(live_ids),
            "shadow_tokens": len(shadow_ids),
            "shared_prefix_tokens": split,
            "boundary_bitwise_equal": boundary_equal,
            "boundary_max_absolute_difference": boundary_max_difference,
            "continuation_bitwise_equal": continuation_equal,
            "continuation_max_absolute_difference": continuation_max_difference,
            **expiry,
        })

    output = {
        "status": "stage_14_online_dual_cache_runtime_validation",
        "configuration": vars(args),
        "data_protocol": "real TAB legal contexts; no names or excerpts emitted",
        "examples": len(rows),
        "all_boundary_bitwise_equal": all(row["boundary_bitwise_equal"] for row in rows),
        "all_continuation_bitwise_equal": all(row["continuation_bitwise_equal"] for row in rows),
        "all_retired_cache_zero": all(row["retired_cache_max_absolute_value"] == 0.0 for row in rows),
        "rows": rows,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2) if args.device == "cuda" else None,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, output, newline=True)
    print(json.dumps({
        "output": str(destination),
        "examples": len(rows),
        "boundary_exact": output["all_boundary_bitwise_equal"],
        "continuation_exact": output["all_continuation_bitwise_equal"],
        "retirement_zero": output["all_retired_cache_zero"],
    }, indent=2))


if __name__ == "__main__":
    main()

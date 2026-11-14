"""Multi-item revocation scaling: k independently revocable items, measured.

Generalises stage20 (k=2) to k = 1..6 and adds the *ambiguity window* w: the number of items whose next
revocation order is not yet known. With instantaneous revocation and no replay interval, every
present/sanitised combination of the w ambiguous items must be materialised (2^w branches); items whose
order is known contribute nothing, so the cost is governed by w, not by the total k.

For every (k, w) we build the 2^w branches, verify each against an independently rebuilt reference
bitwise, reveal a revocation, and check the surviving branch count collapses by half (2^(w-1)).
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

from shadowveil.chunked import cache_bytes, caches_equal, prefill

REAL = ["Diane Castellano", "Mateusz Wrobel", "Ingrid Haugen", "Tomas Ferreira", "Noor Rahman", "Pavel Dvorak"]
SURR = ["Registrant Alpha", "Registrant Beta", "Registrant Gamma", "Registrant Delta", "Registrant Epsilon", "Registrant Zeta"]
HEAD = "Case log entry."
SUFFIX = " Meeting notes: the quarterly review covered budget items only."


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def text_for(sanitized):
    parts = [HEAD]
    for i, s in enumerate(sanitized):
        parts.append(f" Attendee {i + 1}: {SURR[i] if s else REAL[i]}.")
    return "".join(parts) + SUFFIX


@torch.inference_mode()
def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    def build(sanitized):
        ids = tok.encode(text_for(sanitized), add_special_tokens=False)
        return prefill(model, ids, args.device)[0]

    rows = []
    for k in range(1, 7):
        for w in range(1, k + 1):
            t0 = time.perf_counter()
            # items [0, k-w) are already revoked (known order); the next w are ambiguous; none beyond.
            fixed = [True] * (k - w)
            combos = list(itertools.product([False, True], repeat=w))
            branches = {c: build(fixed + list(c)) for c in combos}
            peak_bytes = sum(cache_bytes(b) for b in branches.values())
            # reveal: the first ambiguous item is revoked -> branches with it still present are unreachable
            observed = tuple([True] + [False] * (w - 1))
            active = branches[observed]
            exact = caches_equal(active, build(fixed + list(observed)))
            survivors = [c for c in combos if c[0]]
            # all branches must equal an independent rebuild (bitwise)
            all_exact = all(caches_equal(b, build(fixed + list(c))) for c, b in branches.items())
            rows.append({
                "k_items": k, "ambiguity_window_w": w, "branches_required": 2 ** w,
                "branches_if_all_k_ambiguous": 2 ** k, "peak_cache_bytes": peak_bytes,
                "single_cache_bytes": cache_bytes(active), "peak_over_single": peak_bytes / cache_bytes(active),
                "exact_on_reveal": bool(exact), "all_branches_exact": bool(all_exact),
                "branches_after_first_reveal": len(survivors), "seconds": time.perf_counter() - t0,
            })
            print(json.dumps({k_: rows[-1][k_] for k_ in ("k_items", "ambiguity_window_w", "branches_required",
                                                          "all_branches_exact", "branches_after_first_reveal")}), flush=True)
    out = {"status": "multi_item_scaling", "model": args.model, "rows": rows,
           "known_order_chain": {"branches": 2, "note": "w=1: live + one shadow, chained after each expiry"},
           "interpretation": "branch count is 2^w in the number of order-ambiguous items, not 2^k in all items"}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

"""Measured cost profile of the serving primitives (feeds the scheduler simulation).

Measures on the actual GPU, per model: decode step latency vs batch rows B, chunked-continuation
(burst) latency vs token count n from a non-zero state, HF prefill latency, cache retirement
latency and cache size. Run ALONE on the GPU (concurrent jobs corrupt timings).
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

from shadowveil.chunked import cache_bytes, cache_tensors, chunk, clone_cache, prefill
from snapshot_and_batch_tests import stack_caches

TEXT = ("The Court notes that the applicant lodged the complaint under the relevant provisions and that the "
        "Government submitted observations on the admissibility and merits of the case in due course. ") * 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--output", default="results/cost_profile_hf_130m.json")
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--lengths", default="8,16,32,64,128,256")
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def timed(fn, device, repeats, warm=3):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), statistics.pstdev(out)


@torch.inference_mode()
def main():
    args = parse_args()
    h0 = snapshot()
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    cfg = model.config
    ids = tok.encode(TEXT, add_special_tokens=False)
    base, _ = prefill(model, ids[:48], dev)
    nbytes = cache_bytes(base)

    step_ms = {}
    for B in [int(x) for x in args.batches.split(",")]:
        cache = stack_caches(model, [clone_cache(base, cfg) for _ in range(B)], dev)
        pos = {"p": 48}
        tokens = torch.tensor([[ids[50]]] * B, device=dev)

        def one():
            o = model(input_ids=tokens, cache_params=cache, cache_position=torch.tensor([pos["p"]], device=dev),
                      use_cache=True, return_dict=True)
            pos["p"] += 1
        m, sd = timed(one, dev, args.repeats)
        step_ms[B] = {"median": m, "sd": sd}

    chunk_ms, prefill_ms = {}, {}
    for n in [int(x) for x in args.lengths.split(",")]:
        seq = (ids * 3)[48:48 + n]

        def run_chunk():
            c = clone_cache(base, cfg)
            chunk(model, c, seq, dev)
        chunk_ms[n] = dict(zip(("median", "sd"), timed(run_chunk, dev, args.repeats)))

        def run_prefill():
            prefill(model, (ids * 3)[:n], dev)
        prefill_ms[n] = dict(zip(("median", "sd"), timed(run_prefill, dev, args.repeats)))

    _rc = clone_cache(base, cfg)

    def retire():
        for t in cache_tensors(_rc):
            t.zero_()
    retire_ms = dict(zip(("median", "sd"), timed(retire, dev, args.repeats * 2)))
    clone_ms = dict(zip(("median", "sd"), timed(lambda: clone_cache(base, cfg), dev, args.repeats * 2)))

    ns = np.array(sorted(chunk_ms), dtype=float)
    cm = np.array([chunk_ms[int(n)]["median"] for n in ns])
    b_fit, a_fit = np.polyfit(ns, cm, 1)
    Bs = np.array(sorted(step_ms), dtype=float)
    sm = np.array([step_ms[int(b)]["median"] for b in Bs])
    c1, c0 = np.polyfit(Bs, sm, 1)
    out = {
        "status": "cost_profile", "model": args.model, "configuration": vars(args),
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "backend": "HF Transformers 4.51.3 unfused (sequential) Mamba path; chunk = shadowveil.chunked.chunk",
        "cache_bytes": nbytes,
        "step_ms_by_batch_rows": {str(k): v for k, v in step_ms.items()},
        "chunk_ms_by_tokens": {str(k): v for k, v in chunk_ms.items()},
        "hf_prefill_ms_by_tokens": {str(k): v for k, v in prefill_ms.items()},
        "retire_ms": retire_ms, "clone_ms": clone_ms,
        "fit": {"chunk_a_ms": float(a_fit), "chunk_b_ms_per_token": float(b_fit),
                "step_c0_ms": float(c0), "step_c1_ms_per_extra_row": float(c1)},
    }
    out["host_load"] = {"before": h0, "after": snapshot(1.0)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(json.dumps({"fit": out["fit"], "step": {k: round(v["median"], 2) for k, v in step_ms.items()},
                      "chunk": {k: round(v["median"], 1) for k, v in chunk_ms.items()},
                      "retire_ms": retire_ms["median"]}, indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

"""Mamba-2 attacker evaluation under the primary TAB protocol.

Same data construction, seeds, probe (standardised ridge, alpha 10, 1,024 seeded SSM coordinates per
layer, features after the fixed neutral suffix), conditions (untreated / shadow swap / full reset) and
exactness check as stage10, with the Mamba-2 stacked-tensor cache (`ssm_states`, `conv_states` of shape
[layers, batch, ...]). Same probe, seeds and conditions as the Mamba-1 evaluation.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Mamba2ForCausalLM

from stage4_live_cache_scrubber import SUFFIX
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import balanced_rows, choose_surfaces, unique_contexts
from stage10_shadow_revocation_tab import TEST_PREFIX, TRAIN_PREFIX, fit_predict, labels, render


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="AntonV/mamba2-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--attacker-per-class", type=int, default=12)
    p.add_argument("--test-per-class", type=int, default=6)
    p.add_argument("--context-characters", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=1, help="the Hugging Face Mamba-2 prefill allocates batch x chunk^2 x heads x state; keep at 1")
    p.add_argument("--coordinates-per-layer", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sample_indices(width, layers, count, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {layer: torch.randperm(width, generator=g)[:count].to(device) for layer in range(layers)}


def features(cache, indices):
    chunks = []
    for layer, idx in indices.items():
        chunks.append(cache.ssm_states[layer].float().flatten(1).index_select(1, idx))
    return torch.cat(chunks, dim=1)


@torch.inference_mode()
def run_streams(model, tok, texts, reset, indices, batch_size, device):
    suffix = tok.encode(SUFFIX, add_special_tokens=False)
    out, tot_nll, tot_tok = [], 0.0, 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        cache = model(**enc, use_cache=True, return_dict=True).cache_params
        if reset:
            cache.ssm_states.zero_()
            cache.conv_states.zero_()
        suf = torch.tensor([suffix] * len(batch), device=device)
        logits = []
        for off in range(suf.shape[1]):
            pos = torch.tensor([enc["input_ids"].shape[1] + off], device=device)
            o = model(input_ids=suf[:, off:off + 1], cache_params=cache, cache_position=pos,
                      use_cache=True, return_dict=True)
            cache = o.cache_params
            logits.append(o.logits)
        logits = torch.cat(logits, dim=1)[:, :-1].float()
        tot_nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), suf[:, 1:].reshape(-1), reduction="sum").item()
        tot_tok += suf[:, 1:].numel()
        out.append(features(cache, indices).cpu().numpy())
    return np.concatenate(out), tot_nll / tot_tok


@torch.inference_mode()
def exactness(model, tok, present, redacted, device):
    def cache_for(text):
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        return model(input_ids=enc.input_ids.to(device), use_cache=True, return_dict=True).cache_params
    live, shadow, ref = cache_for(present), cache_for(redacted), cache_for(redacted)
    eq = torch.equal(shadow.ssm_states, ref.ssm_states) and torch.equal(shadow.conv_states, ref.conv_states)
    diff = max(float((shadow.ssm_states.float() - ref.ssm_states.float()).abs().max()),
               float((shadow.conv_states.float() - ref.conv_states.float()).abs().max()))
    live.ssm_states.zero_()
    live.conv_states.zero_()
    zero = max(float(live.ssm_states.float().abs().max()), float(live.conv_states.float().abs().max()))
    return {"bitwise_equal_to_independent_redacted_reference": bool(eq), "max_absolute_cache_difference": diff,
            "retired_live_cache_max_absolute_value": zero,
            "single_cache_bytes": int(sum(t.numel() * t.element_size() for t in (shadow.ssm_states, shadow.conv_states)))}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    occ = load_candidates(args.context_characters)
    _, surfaces = choose_surfaces(occ, 8, args.classes)
    n_att, n_test = args.classes * args.attacker_per_class, args.classes * args.test_per_class
    contexts = unique_contexts(occ, n_att + n_test, args.seed)
    att_rows = balanced_rows(contexts[:n_att], surfaces, args.attacker_per_class, args.seed + 17)
    test_rows = balanced_rows(contexts[n_att:], surfaces, args.test_per_class, args.seed + 23)
    att_y, att_ret = labels(att_rows)
    test_y, test_ret = labels(test_rows)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = Mamba2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    cfg = model.config
    width = cfg.num_heads * cfg.head_dim * cfg.state_size
    indices = sample_indices(width, cfg.num_hidden_layers, args.coordinates_per_layer, args.seed, args.device)

    ex = exactness(model, tok, render(test_rows, TEST_PREFIX, "present")[0],
                   render(test_rows, TEST_PREFIX, "redacted")[0], args.device)
    conds = {"untreated": ("present", False), "shadow_swap": ("redacted", False), "full_reset": ("present", True)}
    results = []
    for name, (field, reset) in conds.items():
        t0 = time.perf_counter()
        xa, _ = run_streams(model, tok, render(att_rows, TRAIN_PREFIX, field), reset, indices, args.batch_size, args.device)
        xt, nll = run_streams(model, tok, render(test_rows, TEST_PREFIX, field), reset, indices, args.batch_size, args.device)
        ident, ret = fit_predict(xa, att_y, xt, test_y), fit_predict(xa, att_ret, xt, test_ret)
        results.append({"condition": name, "identity": ident, "retained_fact": ret, "suffix_nll": nll,
                        "runtime_seconds": time.perf_counter() - t0})
        print(json.dumps({"condition": name, "identity": ident["accuracy"], "retained": ret["accuracy"], "nll": nll}), flush=True)
    out = {"status": "mamba2_attacker_audit", "model": args.model, "configuration": vars(args),
           "chance_accuracy": 1.0 / args.classes, "document_overlap_attacker_test": 0,
           "sampled_ssm_coordinates": cfg.num_hidden_layers * args.coordinates_per_layer,
           "exactness": ex, "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

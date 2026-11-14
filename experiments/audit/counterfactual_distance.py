"""String-free distance to the counterfactual (sanitized) execution, and matched-seed generation identity.

For every test document the reference state is the cache of the sanitized stream. Methods are compared by
  * relative state distance ||S - R||_F / ||R||_F over all cache tensors,
  * KL( softmax(R logits) || softmax(S logits) ) of the next-token distribution after the neutral continuation,
  * top-1 agreement, and
  * matched-seed sampling: k samples of `gen_tokens` tokens at temperature 0.7 from the method's state and from the
    reference with identical random streams; the rate at which the generation is token-for-token identical.
Methods: untreated live state, full reset, learned subspace editing (sensitive and random control, optionally all
coordinates), and the shadow (an independently built sanitized cache, expected to be at distance 0).
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
from transformers import AutoTokenizer, MambaForCausalLM

from shadowveil.chunked import cache_tensors, clone_cache, prefill, step
from shadowveil.hostload import snapshot
from stage10_shadow_revocation_tab import TEST_PREFIX, TRAIN_PREFIX, labels, render, sample_indices
from stage3_feature_scrubber import sensitive_rows
from stage4_live_cache_scrubber import SUFFIX, prefix_features, transform_cache
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import balanced_rows, choose_surfaces, unique_contexts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--defender-per-class", type=int, default=12)
    p.add_argument("--test-per-class", type=int, default=6)
    p.add_argument("--context-characters", type=int, default=180)
    p.add_argument("--coordinates-per-layer", type=int, default=1024)
    p.add_argument("--full-layer", action="store_true", help="edit all coordinates of every layer")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--gen-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def flat(cache):
    return torch.cat([t.float().flatten() for t in cache_tensors(cache)])


@torch.inference_mode()
def advance(model, cache, pos, ids, device):
    logits = None
    for t in ids:
        cache, logits = step(model, cache, pos, t, device)
        pos += 1
    return cache, pos, logits[0, -1].float()


@torch.inference_mode()
def sample(model, cache, pos, logits, n_tokens, temperature, seed, device):
    cache = clone_cache(cache, model.config)
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for _ in range(n_tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        tok = int(torch.multinomial(probs, 1, generator=g))
        out.append(tok)
        cache, logits_all = step(model, cache, pos, tok, device)
        pos += 1
        logits = logits_all[0, -1].float()
    return out


def main():
    args = parse_args()
    h0 = snapshot()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device
    occ = load_candidates(args.context_characters)
    defender_surfaces, eval_surfaces = choose_surfaces(occ, args.classes, args.classes)
    n_def, n_test = args.classes * args.defender_per_class, args.classes * args.test_per_class
    contexts = unique_contexts(occ, n_def + n_test, args.seed)
    test_rows = balanced_rows(contexts[:n_test], eval_surfaces, args.test_per_class, args.seed + 23)
    def_rows = balanced_rows(contexts[n_test:], defender_surfaces, args.defender_per_class, args.seed + 29)
    def_y, _ = labels(def_rows)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(dev).eval()
    width = model.config.intermediate_size * model.config.state_size
    layers, indices = sample_indices(model, width if args.full_layer else args.coordinates_per_layer, args.seed, dev)

    boundary = prefix_features(model, tok, render(def_rows, TRAIN_PREFIX, "present"), layers, indices, args.batch_size, dev)
    mean_np = boundary.mean(axis=0, keepdims=True)
    std_np = boundary.std(axis=0, keepdims=True) + 1e-5
    basis_np, _ = sensitive_rows((boundary - mean_np) / std_np, def_y[:, None])
    rng = np.random.default_rng(args.seed)
    rand = np.linalg.qr(rng.standard_normal((boundary.shape[1], basis_np.shape[0])).astype(np.float32), mode="reduced")[0].T.astype(np.float32)
    mean, std = torch.from_numpy(mean_np).to(dev), torch.from_numpy(std_np).to(dev)
    basis, rbasis = torch.from_numpy(basis_np).to(dev), torch.from_numpy(rand).to(dev)

    suffix = tok.encode(SUFFIX, add_special_tokens=False)
    methods = ["untreated", "full_reset", "sensitive_subspace", "random_subspace", "shadow"]
    acc = {m: {"state_dist": [], "kl": [], "top1": [], "gen_identical": [], "first_token": []} for m in methods}
    present = render(test_rows, TEST_PREFIX, "present")
    redacted = render(test_rows, TEST_PREFIX, "redacted")
    t0 = time.perf_counter()
    for i, (txt_p, txt_r) in enumerate(zip(present, redacted)):
        ids_p, ids_r = tok.encode(txt_p, add_special_tokens=False), tok.encode(txt_r, add_special_tokens=False)
        live, _ = prefill(model, ids_p, dev)
        ref, _ = prefill(model, ids_r, dev)
        shadow, _ = prefill(model, ids_r, dev)                    # independently built sanitized cache
        caches = {"untreated": clone_cache(live, model.config), "shadow": shadow}
        for m in ("full_reset", "sensitive_subspace", "random_subspace"):
            c = clone_cache(live, model.config)
            transform_cache(c, m, mean, std, basis, rbasis, layers, indices, args.seed)
            caches[m] = c
        ref_c, ref_pos, ref_logits = advance(model, ref, len(ids_r), suffix, dev)
        ref_state = flat(ref_c)
        ref_p = torch.softmax(ref_logits, dim=-1)
        ref_samples = [sample(model, ref_c, ref_pos, ref_logits, args.gen_tokens, args.temperature, 1000 * i + k, dev) for k in range(args.samples)]
        for m in methods:
            c, pos, logits = advance(model, caches[m], len(ids_p) if m != "shadow" else len(ids_r), suffix, dev)
            dist = float((flat(c) - ref_state).norm() / ref_state.norm())
            kl = float((ref_p * (torch.log(ref_p.clamp_min(1e-12)) - torch.log_softmax(logits, -1))).sum())
            acc[m]["state_dist"].append(dist)
            acc[m]["kl"].append(kl)
            acc[m]["top1"].append(int(logits.argmax()) == int(ref_logits.argmax()))
            for k in range(args.samples):
                s = sample(model, c, pos, logits, args.gen_tokens, args.temperature, 1000 * i + k, dev)
                acc[m]["gen_identical"].append(s == ref_samples[k])
                acc[m]["first_token"].append(s[0] == ref_samples[k][0])
        if i % 8 == 0:
            print(json.dumps({"doc": i, "elapsed_s": round(time.perf_counter() - t0)}), flush=True)
    summary = {}
    for m in methods:
        a = acc[m]
        summary[m] = {"state_dist_mean": float(np.mean(a["state_dist"])), "state_dist_max": float(np.max(a["state_dist"])),
                      "kl_mean": float(np.mean(a["kl"])), "kl_max": float(np.max(a["kl"])),
                      "top1_agreement": float(np.mean(a["top1"])),
                      "first_token_identical": float(np.mean(a["first_token"])),
                      "generation_identical": float(np.mean(a["gen_identical"])), "draws": len(a["gen_identical"])}
        print(m, {k: round(v, 4) if isinstance(v, float) else v for k, v in summary[m].items()}, flush=True)
    out = {"status": "counterfactual_distance", "model": args.model, "configuration": vars(args), "subspace_rank": int(basis.shape[0]),
           "edited_coordinates_per_layer": int(width if args.full_layer else args.coordinates_per_layer),
           "documents": len(present), "summary": summary, "host_load": {"before": h0, "after": snapshot(1.0)}}
    write_json(Path(args.output), out)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

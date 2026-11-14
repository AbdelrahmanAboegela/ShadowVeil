"""Audit controls for the sanitizer and privacy evidence.

Same TAB data construction, probe, seeds and splits as stage10, plus:
  * leaky sanitizer (partial masking: first 3 characters of the name survive). Equality to the
    *declared* sanitized stream still holds bit for bit, yet identity stays recoverable: the guarantee is
    state equality to the sanitizer's output, privacy is only as good as the sanitizer, and this audit
    detects a leaky one (positive control for the audit itself);
  * stronger-access attackers: ridge on the FULL cache (all SSM + conv state), and a training-free
    logit-access attacker that scores each candidate name by its log-likelihood under the cache;
  * a label-permutation null for the ridge probe: accuracy of probes trained on shuffled labels, the
    correct reference distribution for 'a probe trained on label-free features'.
No raw names or passages are written to the result file.
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
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, MambaForCausalLM

from shadowveil.chunked import clone_cache
from shadowveil.fused import FusedMamba
from stage1_leakage_probe import RETAINED_CITIES  # noqa: F401  (import side effects of stage modules)
from stage4_live_cache_scrubber import SUFFIX, selected_features
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import balanced_rows, choose_surfaces, unique_contexts
from stage10_shadow_revocation_tab import (
    TEST_PREFIX, TRAIN_PREFIX, fit_predict, labels, render, sample_indices, verify_exact_boundary,
)

QUERIES = (
    " Question: who is the person mentioned in this passage? Answer:",                     # task
    " Question: the person named in this passage is",                                       # direct
    " Let me think about who the passage is about. The person is",                         # think
    " Question: what name was removed from this passage? The name is",                     # introspect
    " Question: list the names of people in this passage. Names:",                         # enumerate
    " The passage concerns a person whose full name is",                                   # cued completion
)
QUERY = QUERIES[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="state-spaces/mamba-130m-hf")
    p.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--corpus", choices=("tab", "enron"), default="tab")
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--attacker-per-class", type=int, default=12)
    p.add_argument("--test-per-class", type=int, default=6)
    p.add_argument("--context-characters", type=int, default=180)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--coordinates-per-layer", type=int, default=1024)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--skip-full-cache", action="store_true")
    p.add_argument("--skip-logit", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def partial_mask(surface: str) -> str:
    return surface[:3] + "[REDACTED]"


@torch.inference_mode()
def streams(model, tokenizer, texts, layers, indices, batch_size, device, want_full):
    """Features after the fixed neutral suffix: seeded SSM coordinates (primary protocol) and optionally
    the full cache (all SSM + conv state)."""
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    sel, full = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        cache = model(**enc, use_cache=True, return_dict=True).cache_params
        suf = torch.tensor([suffix] * len(batch), device=device)
        for off in range(suf.shape[1]):
            pos = torch.tensor([enc["input_ids"].shape[1] + off], device=device)
            cache = model(input_ids=suf[:, off:off + 1], cache_params=cache, cache_position=pos,
                          use_cache=True, return_dict=True).cache_params
        sel.append(selected_features(cache, layers, indices).cpu().numpy())
        if want_full:
            parts = [s.float().flatten(1) for s in cache.ssm_states] + [c.float().flatten(1) for c in cache.conv_states]
            full.append(torch.cat(parts, dim=1).half().cpu().numpy())
    return np.concatenate(sel), (np.concatenate(full) if want_full else None)


@torch.inference_mode()
def logit_scores(model, tokenizer, texts, candidates, device, query_text=QUERY):
    """Training-free logit-access attacker: log-likelihood of each candidate name given the cache
    (fused Triton continuation; same arithmetic for every condition)."""
    from transformers.cache_utils import MambaCache
    fm = FusedMamba(model)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    query = tokenizer.encode(query_text, add_special_tokens=False)
    cand_ids = [tokenizer.encode(" " + c, add_special_tokens=False) for c in candidates]
    dtype = next(model.parameters()).dtype
    scores = np.zeros((len(texts), len(candidates)), dtype=np.float64)
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False) + suffix
        cache = MambaCache(model.config, max_batch_size=1, dtype=dtype, device=device)
        fm.forward(cache, torch.tensor([ids], dtype=torch.long, device=device))
        for j, cid in enumerate(cand_ids):
            c = clone_cache(cache, model.config)
            logits = fm.forward(c, torch.tensor([query + cid], dtype=torch.long, device=device))[0]
            logp = torch.log_softmax(logits, dim=-1)
            pos = torch.arange(len(query) - 1, len(query) - 1 + len(cid), device=device)
            scores[i, j] = float(logp[pos, torch.tensor(cid, device=device)].sum())
    return scores


def ridge_full_dual(x_attack, y_attack, x_test, y_test, alpha=10.0, chunk=131072, classes=8):
    """Standardised ridge classifier in dual form, column-chunked (memory-light for multi-million-dim caches).
    Same classifier family as the primary probe: per-column standardisation on the attacker set, ridge alpha=10,
    one-vs-rest targets in {-1,+1}, argmax decision."""
    na, nt = x_attack.shape[0], x_test.shape[0]
    gaa = np.zeros((na, na), dtype=np.float64)
    gta = np.zeros((nt, na), dtype=np.float64)
    for start in range(0, x_attack.shape[1], chunk):
        xa = x_attack[:, start:start + chunk].astype(np.float32)
        xt = x_test[:, start:start + chunk].astype(np.float32)
        mu, sd = xa.mean(0), xa.std(0)
        sd[sd == 0] = 1.0
        xa = (xa - mu) / sd
        xt = (xt - mu) / sd
        gaa += xa @ xa.T
        gta += xt @ xa.T
    target = -np.ones((na, classes))
    target[np.arange(na), y_attack] = 1.0
    ybar = target.mean(0, keepdims=True)
    coef = np.linalg.solve(gaa + alpha * np.eye(na), target - ybar)
    pred = (gta @ coef + ybar).argmax(1)
    correct = (pred == y_test)
    from stage10_shadow_revocation_tab import wilson
    return {"accuracy": float(correct.mean()), "wilson_95": wilson(int(correct.sum()), len(y_test))}


@torch.inference_mode()
def generation_attack(model, tokenizer, texts, candidates, labels_true, device, new_tokens=8, query_text=QUERY):
    """Behavioural attacker: greedily generate a short continuation of the query from the cache and check
    whether the text begins with a candidate name. Reports the rate at which the TRUE name is produced and the
    rate at which ANY candidate name is produced (the latter is the attacker's raw output rate)."""
    from transformers.cache_utils import MambaCache
    fm = FusedMamba(model)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    query = tokenizer.encode(query_text, add_special_tokens=False)
    dtype = next(model.parameters()).dtype
    cand_norm = [c.lower() for c in candidates]
    hit_true = hit_any = 0
    for text, y in zip(texts, labels_true):
        ids = tokenizer.encode(text, add_special_tokens=False) + suffix + query
        cache = MambaCache(model.config, max_batch_size=1, dtype=dtype, device=device)
        logits = fm.forward(cache, torch.tensor([ids], dtype=torch.long, device=device))[:, -1]
        out = []
        for _ in range(new_tokens):
            nxt = int(logits.argmax(-1)[0])
            out.append(nxt)
            logits = fm.forward(cache, torch.tensor([[nxt]], dtype=torch.long, device=device))[:, -1]
        gen = tokenizer.decode(out).strip().lower()
        matches = [i for i, c in enumerate(cand_norm) if gen.startswith(c)]
        hit_any += bool(matches)
        hit_true += bool(matches) and matches[0] == int(y)
    n = len(texts)
    return {"true_name_rate": hit_true / n, "any_candidate_rate": hit_any / n}


def ridge_acc(x_attack, y_attack, x_test, y_test):
    return fit_predict(x_attack, y_attack, x_test, y_test)


def permutation_null(x_attack, y_attack, x_test, y_test, n, seed):
    scaler = StandardScaler().fit(x_attack)
    xa, xt = scaler.transform(x_attack), scaler.transform(x_test)
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n):
        probe = RidgeClassifier(alpha=10.0).fit(xa, rng.permutation(y_attack))
        accs.append(float((probe.predict(xt) == y_test).mean()))
    true = RidgeClassifier(alpha=10.0).fit(xa, y_attack)
    obs = float((true.predict(xt) == y_test).mean())
    accs = np.array(accs)
    return {"observed": obs, "null_mean": float(accs.mean()), "null_sd": float(accs.std()),
            "null_p95": float(np.percentile(accs, 95)),
            "p_value_ge_observed": float((1 + (accs >= obs).sum()) / (1 + len(accs)))}


def enron_rows(enron, records, surfaces, per_class, seed, surrogates):
    """Enron rows with the class-independent pseudonym (primary) and a partial-mask (leaky) sanitizer."""
    needed = len(surfaces) * per_class
    by_doc = {}
    for row in records:
        by_doc.setdefault(row["doc_key"], row)
    contexts = list(by_doc.values())
    rng = random.Random(seed)
    rng.shuffle(contexts)
    if len(contexts) < needed:
        raise RuntimeError(f"need {needed} unique Enron messages, found {len(contexts)}")
    lab = [label for label in range(len(surfaces)) for _ in range(per_class)]
    ret = [i % len(enron.RETAINED_CITIES) for i in range(needed)]
    rng.shuffle(lab)
    rng.shuffle(ret)
    sid = [i % len(surrogates) for i in range(needed)]
    rng.shuffle(sid)
    out = []
    for src, label, rl, sg in zip(contexts[:needed], lab, ret, sid):
        out.append({"doc_key": src["doc_key"], "label": label, "retained_label": rl,
                    "present": src["redacted"].replace("[REDACTED PERSON]", surfaces[label], 1),
                    "redacted": src["redacted"].replace("[REDACTED PERSON]", surrogates[sg], 1),
                    "leaky": src["redacted"].replace("[REDACTED PERSON]", partial_mask(surfaces[label]), 1)})
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
    if args.corpus == "tab":
        occ = load_candidates(args.context_characters)
        _, surfaces = choose_surfaces(occ, 8, args.classes)
        n_att, n_test = args.classes * args.attacker_per_class, args.classes * args.test_per_class
        contexts = unique_contexts(occ, n_att + n_test, args.seed)
        att_rows = balanced_rows(contexts[:n_att], surfaces, args.attacker_per_class, args.seed + 17)
        test_rows = balanced_rows(contexts[n_att:], surfaces, args.test_per_class, args.seed + 23)
        for rows in (att_rows, test_rows):
            for r in rows:
                r["leaky"] = r["redacted"].replace("[REDACTED PERSON]", partial_mask(surfaces[r["label"]]), 1)
        train_tpl, test_tpl = TRAIN_PREFIX, TEST_PREFIX
    else:
        import stage11_shadow_revocation_enron as enron
        records, counts = enron.load_enron(args.context_characters)
        surfaces, surrogates, _ = enron.choose_length_matched_surfaces(counts, tokenizer, args.classes, 1)
        att_rows = enron_rows(enron, records["train"], surfaces, args.attacker_per_class, args.seed + 17, surrogates)
        test_rows = enron_rows(enron, records["test"], surfaces, args.test_per_class, args.seed + 23, surrogates)
        train_tpl, test_tpl = enron.TRAIN_PREFIX, enron.TEST_PREFIX
    att_y, _ = labels(att_rows)
    test_y, _ = labels(test_rows)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers, indices = sample_indices(model, args.coordinates_per_layer, args.seed, args.device)

    test_present = render(test_rows, test_tpl, "present")
    exactness = {
        "primary_sanitizer": verify_exact_boundary(model, tokenizer, test_present[0],
                                                  render(test_rows, test_tpl, "redacted")[0], args.device),
        "leaky_sanitizer": verify_exact_boundary(model, tokenizer, test_present[0],
                                                 render(test_rows, test_tpl, "leaky")[0], args.device),
    }
    conditions = {"untreated": "present", "shadow_primary_sanitizer": "redacted", "shadow_leaky_sanitizer": "leaky"}
    cand = list(surfaces)
    out_rows = []
    for name, field in conditions.items():
        t0 = time.perf_counter()
        att_texts, test_texts = render(att_rows, train_tpl, field), render(test_rows, test_tpl, field)
        xa, fa = streams(model, tokenizer, att_texts, layers, indices, args.batch_size, args.device, not args.skip_full_cache)
        xt, ft = streams(model, tokenizer, test_texts, layers, indices, args.batch_size, args.device, not args.skip_full_cache)
        row = {"condition": name, "ridge_seeded_coordinates": {k: v for k, v in ridge_acc(xa, att_y, xt, test_y).items()
                                                                if k in ("accuracy", "wilson_95")}}
        row["permutation_null_ridge"] = permutation_null(xa, att_y, xt, test_y, args.permutations, args.seed)
        if not args.skip_full_cache:
            row["ridge_full_cache"] = ridge_full_dual(fa, att_y, ft, test_y, classes=args.classes)
            row["full_cache_dim"] = int(fa.shape[1])
            del fa, ft
        if not args.skip_logit:
            per_probe, gen_true, gen_any = [], [], []
            hit_any_doc = np.zeros(len(test_texts), dtype=bool)
            for q in QUERIES:
                sc = logit_scores(model, tokenizer, test_texts, cand, args.device, query_text=q)
                per_probe.append({"raw_accuracy": float((sc.argmax(1) == test_y).mean()),
                                  "calibrated_accuracy": float(((sc - sc.mean(0, keepdims=True)).argmax(1) == test_y).mean())})
                g = generation_attack(model, tokenizer, test_texts, cand, test_y, args.device, query_text=q)
                gen_true.append(g["true_name_rate"])
                gen_any.append(g["any_candidate_rate"])
            row["logit_candidate_attacker"] = dict(per_probe[1])                 # direct probe (primary)
            row["logit_candidate_attacker"]["per_probe"] = per_probe
            row["logit_candidate_attacker"]["mean_calibrated_over_probes"] = float(np.mean([x["calibrated_accuracy"] for x in per_probe]))
            row["logit_candidate_attacker"]["best_calibrated_over_probes"] = float(max(x["calibrated_accuracy"] for x in per_probe))
            row["generation_attacker"] = {"true_name_rate": gen_true[1], "per_probe_true_name_rate": gen_true,
                                          "best_over_probes": float(max(gen_true)), "any_candidate_rate_per_probe": gen_any}
        row["seconds"] = time.perf_counter() - t0
        out_rows.append(row)
        print(json.dumps({"condition": name, **{k: (v.get("accuracy") if isinstance(v, dict) and "accuracy" in v else v)
                                                for k, v in row.items() if k != "permutation_null_ridge"}}, default=str), flush=True)

    out = {
        "status": "audit_controls", "model": args.model, "configuration": vars(args),
        "chance_accuracy": 1.0 / args.classes,
        "note": "additional controls beyond the primary probe",
        "exactness": {k: {"bitwise_equal": v["bitwise_equal_to_independent_redacted_reference"],
                          "max_abs_diff": v["max_absolute_cache_difference"]} for k, v in exactness.items()},
        "rows": out_rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, out)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

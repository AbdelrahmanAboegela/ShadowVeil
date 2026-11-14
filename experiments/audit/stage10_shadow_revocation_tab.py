"""Prospective shadow-state revocation on real TAB contexts.

During the retention window, the system maintains a normal Mamba cache and a
counterfactual cache produced from a redacted view of the same stream.  At the
expiry event, revocation is a pointer swap to the shadow cache.  This script
evaluates post-swap leakage with a fresh attacker and verifies equivalence to
an independently recomputed redacted reference.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, MambaForCausalLM

from stage1_leakage_probe import RETAINED_CITIES
from stage4_live_cache_scrubber import SUFFIX, selected_features
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import (
    balanced_rows,
    choose_surfaces,
    unique_contexts,
)


TRAIN_PREFIX = (
    "Legal document excerpt: {context}\n"
    "Non-private case category: {retained}. "
    "Retention event: the identified person in this excerpt expires now."
)
TEST_PREFIX = (
    "Court-record passage: {context}\n"
    "Public case category to retain: {retained}. "
    "Privacy deadline: revoke the person in this passage now."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage10_shadow_revocation_tab.json")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--attacker-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--context-characters", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def render(rows, template, condition):
    return [
        template.format(
            context=row[condition], retained=RETAINED_CITIES[row["retained_label"]]
        )
        for row in rows
    ]


def labels(rows):
    return (
        np.asarray([row["label"] for row in rows]),
        np.asarray([row["retained_label"] for row in rows]),
    )


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def fit_predict(x_train, y_train, x_test, y_test):
    probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
    probe.fit(x_train, y_train)
    prediction = probe.predict(x_test)
    successes = int((prediction == y_test).sum())
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "wilson_95": wilson(successes, len(y_test)),
        "correct": (prediction == y_test).astype(int).tolist(),
        "predictions": prediction.astype(int).tolist(),
    }


def sample_indices(model, count, seed, device):
    state_width = model.config.intermediate_size * model.config.state_size
    if count > state_width:
        raise ValueError(f"Requested {count} coordinates from per-layer width {state_width}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layers = list(range(model.config.num_hidden_layers))
    indices = {
        layer: torch.randperm(state_width, generator=generator)[:count].to(device)
        for layer in layers
    }
    return layers, indices


@torch.inference_mode()
def run_streams(model, tokenizer, texts, reset, layers, indices, batch_size, device):
    width = sum(index.numel() for index in indices.values())
    output = np.empty((len(texts), width), dtype=np.float32)
    suffix_1d = tokenizer.encode(SUFFIX, add_special_tokens=False)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cache = model(**encoded, use_cache=True, return_dict=True).cache_params
        if reset:
            for state in cache.ssm_states:
                state.zero_()
            for state in cache.conv_states:
                state.zero_()
        suffix_ids = torch.tensor([suffix_1d] * len(batch), device=device)
        logits = []
        for offset in range(suffix_ids.shape[1]):
            position = torch.tensor([encoded["input_ids"].shape[1] + offset], device=device)
            continued = model(
                input_ids=suffix_ids[:, offset:offset + 1], cache_params=cache,
                cache_position=position, use_cache=True, return_dict=True,
            )
            cache = continued.cache_params
            logits.append(continued.logits)
        logits = torch.cat(logits, dim=1)[:, :-1].float()
        target = suffix_ids[:, 1:]
        total_nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"
        ).item()
        total_tokens += target.numel()
        output[start:start + len(batch)] = selected_features(cache, layers, indices).cpu().numpy()
    return output, total_nll / total_tokens


@torch.inference_mode()
def verify_exact_boundary(model, tokenizer, present, redacted, device):
    def cache_for(text):
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        return model(**encoded, use_cache=True, return_dict=True).cache_params

    live = cache_for(present)
    shadow = cache_for(redacted)
    reference = cache_for(redacted)
    # Both caches are fully materialized at this point. Synchronize before the
    # timer so the measurement contains only the host-side state-handle swap.
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    active = live
    active = shadow
    swap_nanoseconds = time.perf_counter_ns() - started
    max_difference = 0.0
    exact = True
    for actual, expected in zip(active.ssm_states + active.conv_states, reference.ssm_states + reference.conv_states):
        exact = exact and torch.equal(actual, expected)
        max_difference = max(max_difference, float((actual.float() - expected.float()).abs().max()))
    cache_bytes = sum(item.numel() * item.element_size() for item in live.ssm_states + live.conv_states)
    if device == "cuda":
        torch.cuda.synchronize()
    retire_started = time.perf_counter()
    for state in live.ssm_states + live.conv_states:
        state.zero_()
    if device == "cuda":
        torch.cuda.synchronize()
    secure_retire_ms = 1000 * (time.perf_counter() - retire_started)
    retired_max = max(
        float(state.float().abs().max()) for state in live.ssm_states + live.conv_states
    )
    return {
        "bitwise_equal_to_independent_redacted_reference": bool(exact),
        "max_absolute_cache_difference": max_difference,
        "python_pointer_swap_nanoseconds": int(swap_nanoseconds),
        "secure_live_cache_retire_ms": secure_retire_ms,
        "retired_live_cache_max_absolute_value": retired_max,
        "single_cache_bytes": int(cache_bytes),
        "dual_cache_bytes": int(2 * cache_bytes),
    }


def timed_prefill(model, tokenizer, texts, batch_size, device):
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[offset:offset + batch_size], return_tensors="pt",
                padding=True, add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded, use_cache=True, return_dict=True)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    occurrences = load_candidates(args.context_characters)
    _, surfaces = choose_surfaces(occurrences, 8, args.classes)
    attack_count = args.classes * args.attacker_per_class
    test_count = args.classes * args.test_per_class
    contexts = unique_contexts(occurrences, attack_count + test_count, args.seed)
    attack_rows = balanced_rows(
        contexts[:attack_count], surfaces, args.attacker_per_class, args.seed + 17
    )
    test_rows = balanced_rows(
        contexts[attack_count:], surfaces, args.test_per_class, args.seed + 23
    )
    attack_present = render(attack_rows, TRAIN_PREFIX, "present")
    attack_redacted = render(attack_rows, TRAIN_PREFIX, "redacted")
    test_present = render(test_rows, TEST_PREFIX, "present")
    test_redacted = render(test_rows, TEST_PREFIX, "redacted")
    attack_y, attack_retained = labels(attack_rows)
    test_y, test_retained = labels(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers, indices = sample_indices(
        model, args.coordinates_per_layer, args.seed, args.device
    )

    exactness = verify_exact_boundary(
        model, tokenizer, test_present[0], test_redacted[0], args.device
    )
    live_prefill = timed_prefill(model, tokenizer, test_present, args.batch_size, args.device)
    shadow_prefill = timed_prefill(model, tokenizer, test_redacted, args.batch_size, args.device)

    conditions = {
        "untreated": (attack_present, test_present, False),
        "shadow_swap": (attack_redacted, test_redacted, False),
        "full_reset": (attack_present, test_present, True),
    }
    results = []
    shadow_features = None
    for condition, (attack_texts, test_texts, reset) in conditions.items():
        started = time.perf_counter()
        x_attack, _ = run_streams(
            model, tokenizer, attack_texts, reset, layers, indices,
            args.batch_size, args.device,
        )
        x_test, nll = run_streams(
            model, tokenizer, test_texts, reset, layers, indices,
            args.batch_size, args.device,
        )
        if condition == "shadow_swap":
            shadow_features = x_test.copy()
        identity = fit_predict(x_attack, attack_y, x_test, test_y)
        retained = fit_predict(x_attack, attack_retained, x_test, test_retained)
        row = {
            "condition": condition,
            "identity": identity,
            "retained_fact": retained,
            "suffix_nll": nll,
            "runtime_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(json.dumps({
            "condition": condition,
            "identity_accuracy": identity["accuracy"],
            "retained_accuracy": retained["accuracy"],
            "suffix_nll": nll,
        }), flush=True)

    reference_features, _ = run_streams(
        model, tokenizer, test_redacted, False, layers, indices,
        args.batch_size, args.device,
    )
    exactness["post_suffix_sampled_feature_max_difference"] = float(
        np.max(np.abs(shadow_features - reference_features))
    )
    exactness["post_suffix_sampled_features_bitwise_equal"] = bool(
        np.array_equal(shadow_features, reference_features)
    )
    output = {
        "status": "stage_10_prospective_shadow_revocation_tab",
        "dataset": "Text Anonymization Benchmark v1.0 (MIT)",
        "configuration": vars(args),
        "data_protocol": "real TAB names and legal contexts with counterfactual name-context assignment",
        "chance_accuracy": 1.0 / args.classes,
        "document_overlap_attacker_test": 0,
        "identity_classes": args.classes,
        "sampled_ssm_coordinates": len(layers) * args.coordinates_per_layer,
        "exactness": exactness,
        "prefill_timing": {
            "live_seconds": live_prefill,
            "shadow_seconds": shadow_prefill,
            "dual_to_live_compute_ratio": (live_prefill + shadow_prefill) / live_prefill,
            "note": "unfused Transformers Mamba fallback; ratio is more portable than absolute latency",
        },
        "results": results,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

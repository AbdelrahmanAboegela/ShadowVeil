"""Nonlinear and sparse-dictionary attackers against shadow-revoked TAB state.

The primary confirmation protocol (stage10) checks a linear RidgeClassifier
attacker. This script reruns the same TAB protocol (same data construction,
same conditions, same exactness checks) and additionally scores the
shadow-revoked cache against:

  - an MLP (nonlinear) attacker on the same sampled coordinates;
  - a sparse-autoencoder attacker: an SAE trained on a defender-only,
    document-disjoint, identity-disjoint split of *untreated* cache states,
    then a linear probe on the resulting sparse code.

This does not replace the primary confirmation results; it is an
additional robustness check answering "does a stronger attacker change the
conclusion," run separately and reported alongside stage10's output.
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

from attacker_variants import fit_predict_linear, fit_predict_mlp, fit_predict_sae, train_sae
from stage5_tab_context_validation import load_candidates
from stage9_unseen_identity_scrubber import balanced_rows, choose_surfaces, unique_contexts
from stage10_shadow_revocation_tab import (
    TEST_PREFIX,
    TRAIN_PREFIX,
    labels,
    render,
    run_streams,
    sample_indices,
    verify_exact_boundary,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage19_nonlinear_attacker_tab.json")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--attacker-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--defender-per-class", type=int, default=12)
    parser.add_argument("--context-characters", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    occurrences = load_candidates(args.context_characters)
    defender_surfaces, eval_surfaces = choose_surfaces(occurrences, args.classes, args.classes)

    attack_count = args.classes * args.attacker_per_class
    test_count = args.classes * args.test_per_class
    defender_count = args.classes * args.defender_per_class
    contexts = unique_contexts(occurrences, attack_count + test_count + defender_count, args.seed)

    attack_rows = balanced_rows(contexts[:attack_count], eval_surfaces, args.attacker_per_class, args.seed + 17)
    test_rows = balanced_rows(
        contexts[attack_count:attack_count + test_count], eval_surfaces, args.test_per_class, args.seed + 23
    )
    defender_rows = balanced_rows(
        contexts[attack_count + test_count:], defender_surfaces, args.defender_per_class, args.seed + 29
    )

    attack_present = render(attack_rows, TRAIN_PREFIX, "present")
    attack_redacted = render(attack_rows, TRAIN_PREFIX, "redacted")
    test_present = render(test_rows, TEST_PREFIX, "present")
    test_redacted = render(test_rows, TEST_PREFIX, "redacted")
    defender_present = render(defender_rows, TRAIN_PREFIX, "present")
    attack_y, attack_retained = labels(attack_rows)
    test_y, test_retained = labels(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers, indices = sample_indices(model, args.coordinates_per_layer, args.seed, args.device)

    exactness = verify_exact_boundary(model, tokenizer, test_present[0], test_redacted[0], args.device)

    defender_features, _ = run_streams(
        model, tokenizer, defender_present, False, layers, indices, args.batch_size, args.device
    )
    # SAE features live on GPU-sized float32 arrays; the SAE itself trains on CPU
    # to avoid contending with the Mamba forward passes' VRAM. Trained once here
    # and reused across every condition below -- the SAE only depends on the
    # defender split, not on which condition is being scored, so retraining it
    # per condition would be redundant compute.
    defender_features_cpu = defender_features.astype(np.float32)
    sae_bundle = train_sae(defender_features_cpu, seed=args.seed, device="cpu")

    conditions = {
        "untreated": (attack_present, test_present, False),
        "shadow_swap": (attack_redacted, test_redacted, False),
        "full_reset": (attack_present, test_present, True),
    }
    results = []
    for condition, (attack_texts, test_texts, reset) in conditions.items():
        started = time.perf_counter()
        x_attack, _ = run_streams(model, tokenizer, attack_texts, reset, layers, indices, args.batch_size, args.device)
        x_test, nll = run_streams(model, tokenizer, test_texts, reset, layers, indices, args.batch_size, args.device)

        linear = fit_predict_linear(x_attack, attack_y, x_test, test_y)
        mlp = fit_predict_mlp(x_attack, attack_y, x_test, test_y, seed=args.seed)
        sae = fit_predict_sae(
            sae_bundle, defender_features_cpu, x_attack, attack_y, x_test, test_y, device="cpu"
        )
        row = {
            "condition": condition,
            "linear_attacker": linear,
            "mlp_attacker": mlp,
            "sae_attacker": sae,
            "suffix_nll": nll,
            "runtime_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(json.dumps({
            "condition": condition,
            "linear_accuracy": linear["accuracy"],
            "mlp_accuracy": mlp["accuracy"],
            "sae_accuracy": sae["accuracy"],
        }), flush=True)

    output = {
        "status": "stage_19_nonlinear_and_sae_attacker_tab",
        "dataset": "Text Anonymization Benchmark v1.0 (MIT)",
        "configuration": vars(args),
        "note": "Robustness check against a stronger attacker; does not modify the stage10 confirmation protocol or results.",
        "chance_accuracy": 1.0 / args.classes,
        "defender_split_disjoint_from_attack_test": True,
        "exactness": exactness,
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

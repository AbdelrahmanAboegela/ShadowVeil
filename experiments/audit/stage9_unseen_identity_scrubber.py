"""Generic StateVeil defense evaluated on identities unseen during training.

The scrubber is learned from paired boundary states for real TAB contexts with
a person name present versus redacted.  Defender names, adaptive-attacker
names, documents, and prompt templates are separated.  The intervention edits
the complete SSM state of selected layers rather than sampled coordinates.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from transformers import AutoTokenizer, MambaForCausalLM

from stage1_leakage_probe import RETAINED_CITIES, fit_score
from stage4_live_cache_scrubber import prefix_features, run_streams
from stage5_tab_context_validation import load_candidates
from stage6_tab_live_defense import TRAIN_PREFIX, TEST_PREFIX


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage9_unseen_identity_scrubber.json")
    parser.add_argument("--defender-names", type=int, default=16)
    parser.add_argument("--evaluation-names", type=int, default=8)
    parser.add_argument("--defense-per-name", type=int, default=8)
    parser.add_argument("--attacker-per-name", type=int, default=12)
    parser.add_argument("--test-per-name", type=int, default=6)
    parser.add_argument("--context-characters", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--layers", default="12,15,17,20,23")
    parser.add_argument("--subspace-rank", type=int, default=32)
    parser.add_argument("--methods", default="none,sensitive_subspace,random_subspace,gaussian,full_reset")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def valid_surface(surface):
    return (
        3 <= len(surface) <= 50
        and 1 <= len(surface.split()) <= 4
        and re.fullmatch(r"[^\W\d_]+(?:[ '\-][^\W\d_]+)*", surface, flags=re.UNICODE) is not None
    )


def choose_surfaces(occurrences, defender_count, evaluation_count):
    ranked = [
        surface for surface, rows in sorted(occurrences.items(), key=lambda item: len(item[1]), reverse=True)
        if valid_surface(surface)
    ]
    needed = defender_count + evaluation_count
    if len(ranked) < needed:
        raise RuntimeError(f"Need {needed} valid person surfaces, found {len(ranked)}")
    return ranked[:defender_count], ranked[defender_count:needed]


def unique_contexts(occurrences, needed, seed):
    by_doc = {}
    for rows in occurrences.values():
        for row in rows:
            if "[REDACTED PERSON]" in row["redacted_context"]:
                by_doc.setdefault(row["doc_key"], row["redacted_context"])
    contexts = list(by_doc.items())
    random.Random(seed).shuffle(contexts)
    if len(contexts) < needed:
        raise RuntimeError(f"Need {needed} unique documents, found {len(contexts)}")
    return contexts[:needed]


def balanced_rows(contexts, surfaces, per_name, seed):
    labels = [label for label in range(len(surfaces)) for _ in range(per_name)]
    retained = [label for label in range(len(RETAINED_CITIES)) for _ in range(len(labels) // len(RETAINED_CITIES))]
    retained += list(range(len(labels) - len(retained)))
    rng = random.Random(seed)
    rng.shuffle(labels)
    rng.shuffle(retained)
    rows = []
    for (doc_key, redacted_context), label, retained_label in zip(contexts, labels, retained):
        present = redacted_context.replace("[REDACTED PERSON]", surfaces[label], 1)
        rows.append({
            "doc_key": doc_key,
            "present": present,
            "redacted": redacted_context,
            "label": label,
            "retained_label": retained_label,
        })
    return rows


def render(rows, template, condition):
    texts = [
        template.format(
            context=row[condition], retained=RETAINED_CITIES[row["retained_label"]]
        )
        for row in rows
    ]
    labels = np.asarray([row["label"] for row in rows])
    retained = np.asarray([row["retained_label"] for row in rows])
    return texts, labels, retained


def paired_basis(present, redacted, rank, seed):
    pooled = np.concatenate([present, redacted], axis=0)
    mean = redacted.mean(axis=0, keepdims=True).astype(np.float32)
    std = (pooled.std(axis=0, keepdims=True) + 1e-5).astype(np.float32)
    differences = ((present - redacted) / std).astype(np.float32)
    differences -= differences.mean(axis=0, keepdims=True)
    rank = min(rank, differences.shape[0] - 1, differences.shape[1])
    _, singular, vt = randomized_svd(
        differences, n_components=rank, n_iter=5, random_state=seed
    )
    return mean, std, vt.astype(np.float32), singular.astype(np.float32)


def digest_surfaces(surfaces):
    return hashlib.sha256("\n".join(surfaces).encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    occurrences = load_candidates(args.context_characters)
    defender_surfaces, evaluation_surfaces = choose_surfaces(
        occurrences, args.defender_names, args.evaluation_names
    )
    defense_count = args.defender_names * args.defense_per_name
    attack_count = args.evaluation_names * args.attacker_per_name
    test_count = args.evaluation_names * args.test_per_name
    contexts = unique_contexts(occurrences, defense_count + attack_count + test_count, args.seed)
    defense_rows = balanced_rows(
        contexts[:defense_count], defender_surfaces, args.defense_per_name, args.seed + 11
    )
    attack_rows = balanced_rows(
        contexts[defense_count:defense_count + attack_count], evaluation_surfaces,
        args.attacker_per_name, args.seed + 17,
    )
    test_rows = balanced_rows(
        contexts[defense_count + attack_count:], evaluation_surfaces,
        args.test_per_name, args.seed + 23,
    )
    defense_present, _, _ = render(defense_rows, TRAIN_PREFIX, "present")
    defense_redacted, _, _ = render(defense_rows, TRAIN_PREFIX, "redacted")
    attack_texts, attack_y, attack_retained = render(attack_rows, TRAIN_PREFIX, "present")
    test_texts, test_y, test_retained = render(test_rows, TEST_PREFIX, "present")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers = [int(item) for item in args.layers.split(",")]
    state_width = model.config.intermediate_size * model.config.state_size
    full_indices = {
        layer: torch.arange(state_width, device=args.device) for layer in layers
    }

    present_boundary = prefix_features(
        model, tokenizer, defense_present, layers, full_indices, args.batch_size, args.device
    )
    redacted_boundary = prefix_features(
        model, tokenizer, defense_redacted, layers, full_indices, args.batch_size, args.device
    )
    mean_np, std_np, basis_np, singular = paired_basis(
        present_boundary, redacted_boundary, args.subspace_rank, args.seed
    )
    rng = np.random.default_rng(args.seed)
    random_matrix = rng.standard_normal((basis_np.shape[1], basis_np.shape[0])).astype(np.float32)
    random_basis_np = np.linalg.qr(random_matrix, mode="reduced")[0].T.astype(np.float32)
    mean = torch.from_numpy(mean_np).to(args.device)
    std = torch.from_numpy(std_np).to(args.device)
    basis = torch.from_numpy(basis_np).to(args.device)
    random_basis = torch.from_numpy(random_basis_np).to(args.device)

    results = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        started = time.perf_counter()
        x_attack, _, attack_event = run_streams(
            model, tokenizer, attack_texts, method, mean, std, basis, random_basis,
            layers, full_indices, args.batch_size, args.device, args.seed,
        )
        x_test, test_nll, test_event = run_streams(
            model, tokenizer, test_texts, method, mean, std, basis, random_basis,
            layers, full_indices, args.batch_size, args.device, args.seed + 1,
        )
        identity_accuracy, identity_balanced = fit_score(x_attack, attack_y, x_test, test_y)
        retained_accuracy, retained_balanced = fit_score(
            x_attack, attack_retained, x_test, test_retained
        )
        row = {
            "method": method,
            "adaptive_identity_accuracy": identity_accuracy,
            "adaptive_identity_balanced_accuracy": identity_balanced,
            "retained_fact_accuracy": retained_accuracy,
            "retained_fact_balanced_accuracy": retained_balanced,
            "suffix_nll": test_nll,
            "mean_boundary_event_ms": 1000 * (attack_event + test_event) / 2,
            "runtime_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    doc_sets = {
        "defense": {row["doc_key"] for row in defense_rows},
        "attacker": {row["doc_key"] for row in attack_rows},
        "test": {row["doc_key"] for row in test_rows},
    }
    output = {
        "status": "stage_9_generic_unseen_identity_live_defense",
        "dataset": "Text Anonymization Benchmark v1.0 (MIT)",
        "configuration": vars(args),
        "chance_accuracy": 1.0 / args.evaluation_names,
        "effective_state_coordinates_per_layer": state_width,
        "defender_evaluation_identity_overlap": len(set(defender_surfaces) & set(evaluation_surfaces)),
        "defender_surface_sha256": digest_surfaces(defender_surfaces),
        "evaluation_surface_sha256": digest_surfaces(evaluation_surfaces),
        "document_overlap_defense_attacker": len(doc_sets["defense"] & doc_sets["attacker"]),
        "document_overlap_defense_test": len(doc_sets["defense"] & doc_sets["test"]),
        "document_overlap_attacker_test": len(doc_sets["attacker"] & doc_sets["test"]),
        "sensitive_subspace_rank": int(basis.shape[0]),
        "largest_singular_values": singular[:10].tolist(),
        "results": results,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

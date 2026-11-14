"""Live-cache defense transfer to cue-controlled real TAB person names."""

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

from stage1_leakage_probe import RETAINED_CITIES, fit_score
from stage3_feature_scrubber import sensitive_rows
from stage4_live_cache_scrubber import (
    prefix_features,
    run_streams,
)
from stage5_tab_context_validation import load_candidates


TRAIN_PREFIX = (
    "Legal document excerpt: {context}\n"
    "Non-private case category: {retained}. "
    "Retention event: the identified person in this excerpt has expired and must not be retained."
)
TEST_PREFIX = (
    "Court-record passage: {context}\n"
    "Public case category to retain: {retained}. "
    "Privacy deadline: forget the person named in this passage now."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage6_tab_live_defense.json")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--defense-per-class", type=int, default=12)
    parser.add_argument("--attacker-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--context-characters", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--layers", default="12,15,17,20,23")
    parser.add_argument("--methods", default="none,sensitive_subspace,random_subspace,gaussian,full_reset")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def prepare_splits(occurrences, args):
    surfaces = [
        surface for surface, rows in sorted(occurrences.items(), key=lambda item: len(item[1]), reverse=True)
        if len(rows) >= 10
    ][:args.classes]
    needed = args.classes * (args.defense_per_class + args.attacker_per_class + args.test_per_class)
    by_doc = {}
    for rows in occurrences.values():
        for row in rows:
            if "[REDACTED PERSON]" in row["redacted_context"]:
                by_doc.setdefault(row["doc_key"], row["redacted_context"])
    rng = random.Random(args.seed)
    contexts = list(by_doc.items())
    rng.shuffle(contexts)
    if len(contexts) < needed:
        raise RuntimeError(f"Need {needed} unique documents, found {len(contexts)}")
    contexts = contexts[:needed]

    split_sizes = {
        "defense": args.classes * args.defense_per_class,
        "attacker": args.classes * args.attacker_per_class,
        "test": args.classes * args.test_per_class,
    }
    splits = {}
    cursor = 0
    for split_name, size in split_sizes.items():
        split_contexts = contexts[cursor:cursor+size]
        cursor += size
        per_class = size // args.classes
        labels = [label for label in range(args.classes) for _ in range(per_class)]
        retained_labels = [label for label in range(args.classes) for _ in range(per_class)]
        rng.shuffle(labels)
        rng.shuffle(retained_labels)
        rows = []
        for (doc_key, context), label, retained_label in zip(split_contexts, labels, retained_labels):
            present = context.replace("[REDACTED PERSON]", surfaces[label], 1)
            rows.append({
                "doc_key": doc_key, "text": present, "label": label,
                "retained_label": retained_label,
            })
        splits[split_name] = rows
    return surfaces, splits


def render(rows, template):
    return (
        [template.format(context=row["text"], retained=RETAINED_CITIES[row["retained_label"]]) for row in rows],
        np.asarray([row["label"] for row in rows]),
        np.asarray([row["retained_label"] for row in rows]),
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    occurrences = load_candidates(args.context_characters)
    surfaces, splits = prepare_splits(occurrences, args)
    defense_texts, defense_y, _ = render(splits["defense"], TRAIN_PREFIX)
    attack_texts, attack_y, attack_retained = render(splits["attacker"], TRAIN_PREFIX)
    test_texts, test_y, test_retained = render(splits["test"], TEST_PREFIX)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers = [int(item) for item in args.layers.split(",")]
    state_width = model.config.intermediate_size * model.config.state_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = {
        layer: torch.randperm(state_width, generator=generator)[:args.coordinates_per_layer].to(args.device)
        for layer in layers
    }

    boundary = prefix_features(model, tokenizer, defense_texts, layers, indices, args.batch_size, args.device)
    mean_np = boundary.mean(axis=0, keepdims=True)
    std_np = boundary.std(axis=0, keepdims=True) + 1e-5
    basis_np, singular = sensitive_rows((boundary - mean_np) / std_np, defense_y[:, None])
    rng = np.random.default_rng(args.seed)
    random_matrix = rng.standard_normal((boundary.shape[1], basis_np.shape[0])).astype(np.float32)
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
            layers, indices, args.batch_size, args.device, args.seed,
        )
        x_test, test_nll, test_event = run_streams(
            model, tokenizer, test_texts, method, mean, std, basis, random_basis,
            layers, indices, args.batch_size, args.device, args.seed + 1,
        )
        accuracy, balanced = fit_score(x_attack, attack_y, x_test, test_y)
        retained_accuracy, retained_balanced = fit_score(
            x_attack, attack_retained, x_test, test_retained
        )
        row = {
            "method": method,
            "adaptive_identity_accuracy": accuracy,
            "adaptive_identity_balanced_accuracy": balanced,
            "retained_fact_accuracy": retained_accuracy,
            "retained_fact_balanced_accuracy": retained_balanced,
            "suffix_nll": test_nll,
            "mean_boundary_event_ms": 1000 * (attack_event + test_event) / 2,
            "runtime_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    doc_sets = {name: {row["doc_key"] for row in rows} for name, rows in splits.items()}
    output = {
        "status": "stage_6_tab_live_defense_transfer",
        "dataset": "Text Anonymization Benchmark v1.0 (MIT)",
        "configuration": vars(args),
        "identity_surfaces": surfaces,
        "chance_accuracy": 1.0 / args.classes,
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

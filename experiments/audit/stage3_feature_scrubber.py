"""Adaptive-attacker proof of concept for sensitive-subspace cache scrubbing.

This acts on sampled standardized cache coordinates. It is a mechanism gate,
not yet a deployable intervention on the live Mamba cache.
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

from stage2_unseen_reconstruction import (
    TEST_TEMPLATES,
    TRAIN_TEMPLATES,
    create_records,
    extract,
    linear_reconstruction,
    nonlinear_reconstruction,
    render,
    retained_score,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage3_feature_scrubber.json")
    parser.add_argument("--defense-samples", type=int, default=256)
    parser.add_argument("--attacker-samples", type=int, default=384)
    parser.add_argument("--test-samples", type=int, default=192)
    parser.add_argument("--distance", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--layers", default="12,15,17,20,23")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sensitive_rows(x, y):
    rows = []
    for position in range(y.shape[1]):
        probe = RidgeClassifier(alpha=10.0).fit(x, y[:, position])
        rows.append(probe.coef_)
    stacked = np.concatenate(rows, axis=0).astype(np.float64)
    # SVD gives a stable orthonormal basis for the learned sensitive row space.
    _, singular, vt = np.linalg.svd(stacked, full_matrices=False)
    threshold = singular.max() * 1e-6
    return vt[singular > threshold].astype(np.float32), singular


def remove_subspace(x, basis, strength):
    return x - strength * ((x @ basis.T) @ basis)


def evaluate(name, x_attack, y_attack, c_attack, x_test, y_test, c_test, args):
    linear = linear_reconstruction(x_attack, y_attack, x_test, y_test)
    nonlinear = nonlinear_reconstruction(
        x_attack, y_attack, x_test, y_test,
        hidden=256, epochs=60, seed=args.seed + abs(hash(name)) % 10000, device=args.device,
    )
    return {
        "method": name,
        "linear": linear,
        "nonlinear": nonlinear,
        "retained_city_accuracy": retained_score(x_attack, c_attack, x_test, c_test),
        "mean_squared_distortion": float(np.mean((x_test - ORIGINAL_TEST) ** 2)),
    }


ORIGINAL_TEST: np.ndarray


def main():
    global ORIGINAL_TEST
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers = [int(item) for item in args.layers.split(",")]
    width = model.config.intermediate_size * model.config.state_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = {
        layer: torch.randperm(width, generator=generator)[: args.coordinates_per_layer].to(args.device)
        for layer in layers
    }

    total_train = args.defense_samples + args.attacker_samples
    records, used = create_records(total_train, args.distance, args.seed)
    test_records, _ = create_records(args.test_samples, args.distance, args.seed + 1, used)
    train_text, train_y, train_city = render(records, TRAIN_TEMPLATES, args.distance, False)
    test_text, test_y, test_city = render(test_records, TEST_TEMPLATES, args.distance, False)
    started = time.perf_counter()
    x_all = extract(model, tokenizer, train_text, layers, indices, args.batch_size, args.max_length, args.device)
    x_test_raw = extract(model, tokenizer, test_text, layers, indices, args.batch_size, args.max_length, args.device)
    extraction_seconds = time.perf_counter() - started

    defense_slice = slice(0, args.defense_samples)
    attack_slice = slice(args.defense_samples, total_train)
    mean = x_all[defense_slice].mean(axis=0, keepdims=True)
    std = x_all[defense_slice].std(axis=0, keepdims=True) + 1e-5
    x_all = ((x_all - mean) / std).astype(np.float32)
    x_test = ((x_test_raw - mean) / std).astype(np.float32)
    ORIGINAL_TEST = x_test.copy()

    basis, singular = sensitive_rows(x_all[defense_slice], train_y[defense_slice])
    rng = np.random.default_rng(args.seed)
    random_matrix = rng.standard_normal((x_all.shape[1], basis.shape[0])).astype(np.float32)
    random_basis = np.linalg.qr(random_matrix, mode="reduced")[0].T.astype(np.float32)

    results = []
    results.append(evaluate(
        "none", x_all[attack_slice], train_y[attack_slice], train_city[attack_slice],
        x_test, test_y, test_city, args,
    ))
    for strength in (0.25, 0.5, 0.75, 1.0):
        defended_attack = remove_subspace(x_all[attack_slice], basis, strength)
        defended_test = remove_subspace(x_test, basis, strength)
        results.append(evaluate(
            f"sensitive_subspace_{strength}", defended_attack, train_y[attack_slice], train_city[attack_slice],
            defended_test, test_y, test_city, args,
        ))
    for strength in (0.5, 1.0):
        random_attack = remove_subspace(x_all[attack_slice], random_basis, strength)
        random_test = remove_subspace(x_test, random_basis, strength)
        results.append(evaluate(
            f"random_subspace_{strength}", random_attack, train_y[attack_slice], train_city[attack_slice],
            random_test, test_y, test_city, args,
        ))
    for sigma in (0.1, 0.25, 0.5, 1.0):
        noise_rng = np.random.default_rng(args.seed + int(100 * sigma))
        noisy_attack = x_all[attack_slice] + noise_rng.normal(0, sigma, x_all[attack_slice].shape).astype(np.float32)
        noisy_test = x_test + noise_rng.normal(0, sigma, x_test.shape).astype(np.float32)
        results.append(evaluate(
            f"gaussian_{sigma}", noisy_attack, train_y[attack_slice], train_city[attack_slice],
            noisy_test, test_y, test_city, args,
        ))
    for row in results:
        print(json.dumps(row), flush=True)

    output = {
        "status": "stage_3_feature_space_mechanism_gate",
        "claim_limit": "The intervention acts on sampled standardized features, not yet the live cache.",
        "configuration": vars(args),
        "sensitive_subspace_rank": int(basis.shape[0]),
        "largest_singular_values": singular[:10].tolist(),
        "extraction_seconds": extraction_seconds,
        "results": results,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

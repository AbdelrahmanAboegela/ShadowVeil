"""Apply a learned sensitive-subspace intervention to the live Mamba cache.

The scrubber is learned from boundary states, applied exactly when the secret
expires, and evaluated after the model processes a neutral continuation.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from transformers import AutoTokenizer, MambaForCausalLM

from stage1_leakage_probe import NEUTRAL_SENTENCES, RETAINED_CITIES
from stage2_unseen_reconstruction import create_records, linear_reconstruction, retained_score
from stage3_feature_scrubber import remove_subspace, sensitive_rows


TRAIN_PREFIX = (
    "Account memo. Useful city: {city}. Private recovery phrase:{secret}. "
    "Retention event: the recovery phrase has expired."
)
TEST_PREFIX = (
    "Keep {city} as a harmless travel preference. The protected phrase was:{secret}. "
    "Its storage period is now over."
)
# Stage 2 showed strong unseen-token leakage after two neutral chunks. Keeping
# the live-cache suffix at that validated distance avoids an unnecessary 3x
# slowdown in the token-by-token Windows fallback.
SUFFIX = " " + " ".join(NEUTRAL_SENTENCES[:2]) + " Continue with ordinary non-private context."
CODEWORDS = (
    "apple", "banana", "orange", "river", "forest", "mountain", "ocean", "garden",
    "silver", "golden", "purple", "yellow", "window", "table", "chair", "paper",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage4_live_cache_scrubber.json")
    parser.add_argument("--defense-samples", type=int, default=256)
    parser.add_argument("--attacker-samples", type=int, default=384)
    parser.add_argument("--test-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--layers", default="12,15,17,20,23")
    parser.add_argument("--methods", default="none,sensitive_subspace,random_subspace,gaussian,full_reset")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def render_prefix(records, template):
    texts, secrets, cities = [], [], []
    for record in records:
        secret_text = "".join(" " + CODEWORDS[token] for token in record.secret)
        texts.append(template.format(city=RETAINED_CITIES[record.city], secret=secret_text))
        secrets.append(record.secret)
        cities.append(record.city)
    return texts, np.asarray(secrets), np.asarray(cities)


def selected_features(cache, layers, indices):
    chunks = []
    for layer in layers:
        flat = cache.ssm_states[layer].float().flatten(1)
        chunks.append(flat.index_select(1, indices[layer]))
    return torch.cat(chunks, dim=1)


def overwrite_selected(cache, raw_features, layers, indices):
    cursor = 0
    for layer in layers:
        width = indices[layer].numel()
        flat = cache.ssm_states[layer].flatten(1)
        flat.scatter_(1, indices[layer].unsqueeze(0).expand(flat.shape[0], -1), raw_features[:, cursor:cursor+width].to(flat.dtype))
        cursor += width


def transform_cache(cache, method, mean, std, basis, random_basis, layers, indices, seed):
    if method == "full_reset":
        for state in cache.ssm_states:
            state.zero_()
        for state in cache.conv_states:
            state.zero_()
        return
    if method == "none":
        return
    raw = selected_features(cache, layers, indices)
    standardized = (raw - mean) / std
    if method.startswith("sensitive_subspace"):
        strength = float(method.rsplit("_", 1)[-1]) if method != "sensitive_subspace" else 1.0
        transformed = standardized - strength * ((standardized @ basis.T) @ basis)
    elif method.startswith("random_subspace"):
        strength = float(method.rsplit("_", 1)[-1]) if method != "random_subspace" else 1.0
        transformed = standardized - strength * ((standardized @ random_basis.T) @ random_basis)
    elif method == "gaussian":
        generator = torch.Generator(device=standardized.device).manual_seed(seed)
        transformed = standardized + torch.randn(standardized.shape, generator=generator, device=standardized.device) * 1.0
    else:
        raise ValueError(method)
    overwrite_selected(cache, transformed * std + mean, layers, indices)


def grouped_token_ids(tokenizer, texts):
    groups = defaultdict(list)
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    for index, ids in enumerate(encoded):
        groups[len(ids)].append((index, ids))
    return groups


@torch.inference_mode()
def prefix_features(model, tokenizer, texts, layers, indices, batch_size, device):
    output = np.empty((len(texts), sum(item.numel() for item in indices.values())), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cache = model(**encoded, use_cache=True, return_dict=True).cache_params
        output[start:start+len(batch)] = selected_features(cache, layers, indices).cpu().numpy()
    return output


@torch.inference_mode()
def run_streams(
    model, tokenizer, texts, method, mean, std, basis, random_basis,
    layers, indices, batch_size, device, seed,
):
    feature_width = sum(item.numel() for item in indices.values())
    output = np.empty((len(texts), feature_width), dtype=np.float32)
    suffix_ids_1d = tokenizer.encode(SUFFIX, add_special_tokens=False)
    total_nll = 0.0
    total_tokens = 0
    event_times = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prefix_ids = encoded["input_ids"]
        prefix_output = model(**encoded, use_cache=True, return_dict=True)
        cache = prefix_output.cache_params
        event_start = time.perf_counter()
        transform_cache(
            cache, method, mean, std, basis, random_basis,
            layers, indices, seed + start + len(batch),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        event_times.append(time.perf_counter() - event_start)
        suffix_ids = torch.tensor([suffix_ids_1d] * len(batch), device=device)
        # The Transformers slow Mamba backend only supports one-token
        # cached decoding steps. A multi-token cached call incorrectly
        # collapses the convolution path to length one.
        logits_parts = []
        continued = None
        for offset in range(suffix_ids.shape[1]):
            position = torch.tensor([prefix_ids.shape[1] + offset], device=device)
            continued = model(
                input_ids=suffix_ids[:, offset:offset+1],
                cache_params=cache,
                cache_position=position,
                use_cache=True,
                return_dict=True,
            )
            cache = continued.cache_params
            logits_parts.append(continued.logits)
        logits = torch.cat(logits_parts, dim=1)[:, :-1].float()
        labels = suffix_ids[:, 1:]
        total_nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        ).item()
        total_tokens += labels.numel()
        output[start:start+len(batch)] = selected_features(
            cache, layers, indices
        ).cpu().numpy()
    return output, total_nll / total_tokens, float(np.mean(event_times))


def main():
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
    state_width = model.config.intermediate_size * model.config.state_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = {
        layer: torch.randperm(state_width, generator=generator)[:args.coordinates_per_layer].to(args.device)
        for layer in layers
    }

    total_train = args.defense_samples + args.attacker_samples
    records, used = create_records(total_train, 0, args.seed)
    test_records, _ = create_records(args.test_samples, 0, args.seed + 1, used)
    train_texts, train_secret, train_city = render_prefix(records, TRAIN_PREFIX)
    test_texts, test_secret, test_city = render_prefix(test_records, TEST_PREFIX)

    boundary = prefix_features(
        model, tokenizer, train_texts[:args.defense_samples], layers, indices,
        args.batch_size, args.device,
    )
    boundary_mean_np = boundary.mean(axis=0, keepdims=True)
    boundary_std_np = boundary.std(axis=0, keepdims=True) + 1e-5
    boundary_z = (boundary - boundary_mean_np) / boundary_std_np
    basis_np, singular = sensitive_rows(boundary_z, train_secret[:args.defense_samples])
    rng = np.random.default_rng(args.seed)
    random_matrix = rng.standard_normal((boundary_z.shape[1], basis_np.shape[0])).astype(np.float32)
    random_basis_np = np.linalg.qr(random_matrix, mode="reduced")[0].T.astype(np.float32)
    mean = torch.from_numpy(boundary_mean_np).to(args.device)
    std = torch.from_numpy(boundary_std_np).to(args.device)
    basis = torch.from_numpy(basis_np).to(args.device)
    random_basis = torch.from_numpy(random_basis_np).to(args.device)

    attack_texts = train_texts[args.defense_samples:]
    attack_secret = train_secret[args.defense_samples:]
    attack_city = train_city[args.defense_samples:]
    results = []
    for method in [item.strip() for item in args.methods.split(",") if item.strip()]:
        started = time.perf_counter()
        x_attack, attack_nll, attack_event = run_streams(
            model, tokenizer, attack_texts, method, mean, std, basis, random_basis,
            layers, indices, args.batch_size, args.device, args.seed,
        )
        x_test, test_nll, test_event = run_streams(
            model, tokenizer, test_texts, method, mean, std, basis, random_basis,
            layers, indices, args.batch_size, args.device, args.seed + 1,
        )
        row = {
            "method": method,
            "adaptive_linear": linear_reconstruction(x_attack, attack_secret, x_test, test_secret),
            "retained_city_accuracy": retained_score(x_attack, attack_city, x_test, test_city),
            "suffix_nll": test_nll,
            "mean_boundary_event_ms": 1000 * (attack_event + test_event) / 2,
            "runtime_seconds": time.perf_counter() - started,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    output = {
        "status": "stage_4_live_cache_intervention",
        "configuration": vars(args),
        "sensitive_subspace_rank": int(basis.shape[0]),
        "largest_singular_values": singular[:10].tolist(),
        "suffix_tokens": len(tokenizer.encode(SUFFIX, add_special_tokens=False)),
        "results": results,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

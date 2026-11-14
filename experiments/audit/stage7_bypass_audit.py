"""Audit whether StateVeil can be bypassed through unedited Mamba cache state.

The original live-defense experiment evaluates an adaptive attacker on the
same sampled SSM coordinates that the intervention edits.  This audit exposes
four attacker views after the stream resumes: edited coordinates, disjoint
coordinates from the same layers, coordinates from other layers, and sampled
convolutional state.  The combined view is the strongest inexpensive bypass
attacker that fits on the target laptop.
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

from stage1_leakage_probe import fit_score
from stage3_feature_scrubber import sensitive_rows
from stage4_live_cache_scrubber import SUFFIX, prefix_features, transform_cache
from stage6_tab_live_defense import TRAIN_PREFIX, TEST_PREFIX, prepare_splits, render
from stage5_tab_context_validation import load_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage7_bypass_audit.json")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--defense-per-class", type=int, default=12)
    parser.add_argument("--attacker-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--context-characters", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument(
        "--defense-coordinates-per-layer", type=int, default=None,
        help="Coordinates edited per selected layer; defaults to --coordinates-per-layer.",
    )
    parser.add_argument("--edited-layers", default="12,15,17,20,23")
    parser.add_argument("--other-layers", default="0,3,6,9,18")
    parser.add_argument("--methods", default="none,sensitive_subspace,random_subspace")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def sample_indices(width, layers, count, generator, device, excluded=None):
    output = {}
    for layer in layers:
        order = torch.randperm(width, generator=generator)
        if excluded is not None and layer in excluded:
            mask = torch.ones(width, dtype=torch.bool)
            mask[excluded[layer].cpu()] = False
            order = order[mask[order]]
        output[layer] = order[:count].to(device)
    return output


def ssm_features(cache, layers, indices):
    return torch.cat([
        cache.ssm_states[layer].float().flatten(1).index_select(1, indices[layer])
        for layer in layers
    ], dim=1)


def conv_features(cache, layers, indices):
    return torch.cat([
        cache.conv_states[layer].float().flatten(1).index_select(1, indices[layer])
        for layer in layers
    ], dim=1)


@torch.inference_mode()
def run_views(
    model, tokenizer, texts, method, mean, std, basis, random_basis,
    edited_layers, edited_indices, heldout_indices, other_layers, other_indices,
    conv_indices, batch_size, device, seed,
):
    names = ["edited", "other_layers", "conv_state"]
    if all(index.numel() for index in heldout_indices.values()):
        names.insert(1, "heldout_same_layers")
    chunks = {name: [] for name in names}
    suffix_ids_1d = tokenizer.encode(SUFFIX, add_special_tokens=False)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cache = model(**encoded, use_cache=True, return_dict=True).cache_params
        transform_cache(
            cache, method, mean, std, basis, random_basis,
            edited_layers, edited_indices, seed + start + len(batch),
        )
        suffix_ids = torch.tensor([suffix_ids_1d] * len(batch), device=device)
        for offset in range(suffix_ids.shape[1]):
            position = torch.tensor([encoded["input_ids"].shape[1] + offset], device=device)
            continued = model(
                input_ids=suffix_ids[:, offset:offset + 1], cache_params=cache,
                cache_position=position, use_cache=True, return_dict=True,
            )
            cache = continued.cache_params
        chunks["edited"].append(ssm_features(cache, edited_layers, edited_indices).cpu())
        if "heldout_same_layers" in chunks:
            chunks["heldout_same_layers"].append(
                ssm_features(cache, edited_layers, heldout_indices).cpu()
            )
        chunks["other_layers"].append(ssm_features(cache, other_layers, other_indices).cpu())
        chunks["conv_state"].append(conv_features(cache, edited_layers, conv_indices).cpu())
    arrays = {name: torch.cat(parts).numpy().astype(np.float32, copy=False) for name, parts in chunks.items()}
    arrays["combined"] = np.concatenate([arrays[name] for name in names], axis=1)
    return arrays


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    occurrences = load_candidates(args.context_characters)
    _, splits = prepare_splits(occurrences, args)
    defense_texts, defense_y, _ = render(splits["defense"], TRAIN_PREFIX)
    attack_texts, attack_y, attack_retained = render(splits["attacker"], TRAIN_PREFIX)
    test_texts, test_y, test_retained = render(splits["test"], TEST_PREFIX)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    edited_layers = [int(item) for item in args.edited_layers.split(",")]
    other_layers = [int(item) for item in args.other_layers.split(",")]
    state_width = model.config.intermediate_size * model.config.state_size
    conv_width = model.config.intermediate_size * model.config.conv_kernel
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    defense_count = args.defense_coordinates_per_layer or args.coordinates_per_layer
    if defense_count > state_width:
        raise ValueError(f"defense coordinate count exceeds per-layer state width {state_width}")
    edited_indices = sample_indices(state_width, edited_layers, defense_count, generator, args.device)
    heldout_count = min(args.coordinates_per_layer, state_width - defense_count)
    heldout_indices = sample_indices(
        state_width, edited_layers, heldout_count, generator, args.device,
        excluded=edited_indices,
    )
    other_indices = sample_indices(
        state_width, other_layers, args.coordinates_per_layer, generator, args.device
    )
    conv_indices = sample_indices(
        conv_width, edited_layers, min(args.coordinates_per_layer, conv_width), generator, args.device
    )

    boundary = prefix_features(
        model, tokenizer, defense_texts, edited_layers, edited_indices,
        args.batch_size, args.device,
    )
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
        attack_views = run_views(
            model, tokenizer, attack_texts, method, mean, std, basis, random_basis,
            edited_layers, edited_indices, heldout_indices, other_layers, other_indices,
            conv_indices, args.batch_size, args.device, args.seed,
        )
        test_views = run_views(
            model, tokenizer, test_texts, method, mean, std, basis, random_basis,
            edited_layers, edited_indices, heldout_indices, other_layers, other_indices,
            conv_indices, args.batch_size, args.device, args.seed + 1,
        )
        for view in attack_views:
            identity_accuracy, identity_balanced = fit_score(
                attack_views[view], attack_y, test_views[view], test_y
            )
            retained_accuracy, retained_balanced = fit_score(
                attack_views[view], attack_retained, test_views[view], test_retained
            )
            row = {
                "method": method,
                "attacker_view": view,
                "features": int(attack_views[view].shape[1]),
                "identity_accuracy": identity_accuracy,
                "identity_balanced_accuracy": identity_balanced,
                "retained_accuracy": retained_accuracy,
                "retained_balanced_accuracy": retained_balanced,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        print(f"{method} runtime_seconds={time.perf_counter() - started:.2f}", flush=True)

    output = {
        "status": "stage_7_unedited_state_bypass_audit",
        "configuration": vars(args),
        "chance_accuracy": 1.0 / args.classes,
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

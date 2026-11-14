"""Stage-2 unseen compositional-secret reconstruction for Mamba caches.

Unlike Stage 1, train and test secrets are unique combinations. An attacker
must recover each codeword position rather than memorize one of eight codes.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, MambaForCausalLM

from stage1_leakage_probe import NEUTRAL_SENTENCES, RETAINED_CITIES


CODEWORDS = (
    "apple", "banana", "orange", "river", "forest", "mountain", "ocean", "garden",
    "silver", "golden", "purple", "yellow", "window", "table", "chair", "paper",
)
TRAIN_TEMPLATES = (
    "Account memo. Useful city: {city}. Private recovery phrase:{secret}. "
    "Retention event: the recovery phrase has expired. {filler} Continue with public context.",
    "Session profile names {city} as the retained destination. Confidential words:{secret}. "
    "Erase the confidential words under the retention policy. {filler} Resume ordinary assistance.",
)
TEST_TEMPLATES = (
    "Keep {city} as a harmless travel preference. The protected phrase was:{secret}. "
    "Its storage period is now over. {filler} Proceed using only the remaining preference.",
    "The non-private location is {city}. A sensitive authentication phrase appeared as:{secret}. "
    "That phrase is now expired and should be forgotten. {filler} Carry on with safe context.",
)


@dataclass(frozen=True)
class Record:
    secret: tuple[int, int, int, int]
    city: int
    filler_draws: tuple[int, ...]


class MultiPositionAttacker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, positions: int, vocabulary: int):
        super().__init__()
        self.positions = positions
        self.vocabulary = vocabulary
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, positions * vocabulary),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).view(-1, self.positions, self.vocabulary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage2_unseen_reconstruction.json")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--layers", default="12,15,17,20,23")
    parser.add_argument("--distances", default="0,2,6,12")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--attacker-hidden", type=int, default=256)
    parser.add_argument("--attacker-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def create_records(count: int, max_distance: int, seed: int, forbidden: set[tuple[int, ...]] | None = None):
    rng = random.Random(seed)
    forbidden = set() if forbidden is None else set(forbidden)
    records = []
    used = set(forbidden)
    while len(records) < count:
        secret = tuple(rng.randrange(len(CODEWORDS)) for _ in range(4))
        if secret in used:
            continue
        used.add(secret)
        draws = tuple(rng.randrange(len(NEUTRAL_SENTENCES)) for _ in range(max_distance))
        records.append(Record(secret=secret, city=rng.randrange(len(RETAINED_CITIES)), filler_draws=draws))
    return records, used


def render(records: list[Record], templates: tuple[str, ...], distance: int, redacted: bool):
    texts, secrets, cities = [], [], []
    for index, record in enumerate(records):
        if redacted:
            secret_text = " stone" * len(record.secret)
        else:
            secret_text = "".join(" " + CODEWORDS[token] for token in record.secret)
        filler = " ".join(NEUTRAL_SENTENCES[item] for item in record.filler_draws[:distance])
        texts.append(templates[index % len(templates)].format(
            city=RETAINED_CITIES[record.city], secret=secret_text, filler=filler
        ))
        secrets.append(record.secret)
        cities.append(record.city)
    return texts, np.asarray(secrets, dtype=np.int64), np.asarray(cities, dtype=np.int64)


@torch.inference_mode()
def extract(model, tokenizer, texts, layers, indices, batch_size, max_length, device):
    output_chunks = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cache = model(**encoded, use_cache=True, return_dict=True).cache_params
        features = []
        for layer in layers:
            state = cache.ssm_states[layer].float().flatten(1)
            features.append(state.index_select(1, indices[layer]))
        output_chunks.append(torch.cat(features, dim=1).cpu())
    return torch.cat(output_chunks).numpy().astype(np.float32, copy=False)


def linear_reconstruction(x_train, y_train, x_test, y_test):
    predictions = []
    per_position = []
    for position in range(y_train.shape[1]):
        probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
        probe.fit(x_train, y_train[:, position])
        prediction = probe.predict(x_test)
        predictions.append(prediction)
        per_position.append(float(accuracy_score(y_test[:, position], prediction)))
    matrix = np.stack(predictions, axis=1)
    return {
        "token_accuracy": float((matrix == y_test).mean()),
        "exact_secret_accuracy": float((matrix == y_test).all(axis=1).mean()),
        "per_position_accuracy": per_position,
    }


def nonlinear_reconstruction(x_train, y_train, x_test, y_test, hidden, epochs, seed, device):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))
    split = max(1, int(0.85 * len(order)))
    fit_idx, validation_idx = order[:split], order[split:]
    mean = x_train[fit_idx].mean(axis=0, keepdims=True)
    std = x_train[fit_idx].std(axis=0, keepdims=True) + 1e-5
    normalized_train = (x_train - mean) / std
    normalized_test = (x_test - mean) / std

    model = MultiPositionAttacker(x_train.shape[1], hidden, y_train.shape[1], len(CODEWORDS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    train_dataset = TensorDataset(
        torch.from_numpy(normalized_train[fit_idx]).float(), torch.from_numpy(y_train[fit_idx]).long()
    )
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_x = torch.from_numpy(normalized_train[validation_idx]).float().to(device)
    validation_y = torch.from_numpy(y_train[validation_idx]).long().to(device)
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_logits = model(validation_x)
            validation_loss = nn.functional.cross_entropy(
                validation_logits.flatten(0, 1), validation_y.flatten()
            ).item()
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(normalized_test).float().to(device)).argmax(-1).cpu().numpy()
    return {
        "token_accuracy": float((prediction == y_test).mean()),
        "exact_secret_accuracy": float((prediction == y_test).all(axis=1).mean()),
        "per_position_accuracy": [
            float((prediction[:, position] == y_test[:, position]).mean())
            for position in range(y_test.shape[1])
        ],
        "validation_loss": best_loss,
    }


def retained_score(x_train, y_train, x_test, y_test):
    probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
    probe.fit(x_train, y_train)
    return float(accuracy_score(y_test, probe.predict(x_test)))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for word in CODEWORDS + ("stone",):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Codeword {word!r} is not one token: {ids}")

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers = [int(item) for item in args.layers.split(",")]
    distances = [int(item) for item in args.distances.split(",")]
    state_width = model.config.intermediate_size * model.config.state_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = {
        layer: torch.randperm(state_width, generator=generator)[: args.coordinates_per_layer].to(args.device)
        for layer in layers
    }
    train_records, used = create_records(args.train_samples, max(distances), args.seed)
    test_records, _ = create_records(args.test_samples, max(distances), args.seed + 1, used)

    results = []
    for distance in distances:
        condition_data = {}
        for redacted in (False, True):
            condition = "redacted" if redacted else "secret"
            train_texts, train_secret, train_city = render(train_records, TRAIN_TEMPLATES, distance, redacted)
            test_texts, test_secret, test_city = render(test_records, TEST_TEMPLATES, distance, redacted)
            started = time.perf_counter()
            x_train = extract(model, tokenizer, train_texts, layers, indices, args.batch_size, args.max_length, args.device)
            x_test = extract(model, tokenizer, test_texts, layers, indices, args.batch_size, args.max_length, args.device)
            condition_data[condition] = (x_train, train_secret, train_city, x_test, test_secret, test_city)
            print(f"distance={distance} {condition} extraction={time.perf_counter()-started:.2f}s", flush=True)

        for condition, values in condition_data.items():
            x_train, train_secret, train_city, x_test, test_secret, test_city = values
            row = {
                "distance": distance,
                "condition": condition,
                "linear": linear_reconstruction(x_train, train_secret, x_test, test_secret),
                "nonlinear": nonlinear_reconstruction(
                    x_train, train_secret, x_test, test_secret,
                    args.attacker_hidden, args.attacker_epochs,
                    args.seed + distance + (1000 if condition == "redacted" else 0), args.device,
                ),
                "retained_city_accuracy": retained_score(x_train, train_city, x_test, test_city),
            }
            results.append(row)
            print(json.dumps(row), flush=True)

    output = {
        "status": "stage_2_unseen_compositional_reconstruction",
        "configuration": vars(args),
        "vocabulary_size": len(CODEWORDS),
        "secret_positions": 4,
        "token_chance": 1.0 / len(CODEWORDS),
        "exact_chance": (1.0 / len(CODEWORDS)) ** 4,
        "train_test_secret_overlap": 0,
        "results": results,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

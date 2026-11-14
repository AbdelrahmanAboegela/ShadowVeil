"""Stage-1 falsification probe for StateVeil.

Tests whether sensitive and retained categorical facts can be decoded from the
persistent SSM cache of a frozen Mamba model after a policy-expiry sentence.
This is a premise test, not paper-quality evidence or a privacy guarantee.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from transformers import AutoTokenizer, MambaForCausalLM


SECRET_CODES = (
    "QAV-184", "ZEK-293", "RUP-517", "BIX-640",
    "NOF-731", "TUD-852", "WEM-964", "CYG-305",
)
RETAINED_CITIES = (
    "Lima", "Oslo", "Kyoto", "Accra", "Doha", "Riga", "Perth", "Tunis",
)
NEUTRAL_SENTENCES = (
    "The service checked the ordinary system status.",
    "A routine message was added to the activity log.",
    "The interface continued processing normal requests.",
    "A standard confirmation was recorded for this session.",
    "The assistant reviewed a general help article.",
    "Nothing unusual was reported during this interval.",
)

TRAIN_TEMPLATES = (
    "Customer record. Preferred city: {city}. Private access code: {secret}. "
    "The access code has now expired and must not be retained. {filler} "
    "Continue assisting using only non-expired profile information.",
    "Session facts follow. Keep the travel city {city}. The confidential token is {secret}. "
    "Retention policy event: delete the confidential token now. {filler} "
    "Proceed with the harmless session context.",
)
TEST_TEMPLATES = (
    "For this request, the useful destination is {city}; the sensitive credential was {secret}. "
    "That credential is past its retention deadline. {filler} "
    "Carry on while preserving only the useful destination.",
    "Profile update: remember {city} as the selected city. Treat {secret} as a private identifier. "
    "The identifier is now subject to immediate erasure. {filler} "
    "Resume the session with the remaining public preference.",
)


@dataclass
class ConditionResult:
    condition: str
    train_samples: int
    test_samples: int
    secret_accuracy: float
    secret_balanced_accuracy: float
    retained_accuracy: float
    retained_balanced_accuracy: float
    permuted_secret_accuracy: float
    chance_accuracy: float
    extraction_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage1_results.json")
    parser.add_argument("--train-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--coordinates-per-layer", type=int, default=2048)
    parser.add_argument("--layers", default="0,5,11,17,23")
    parser.add_argument("--delayed-sentences", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def filler_text(count: int, rng: random.Random) -> str:
    return " ".join(rng.choice(NEUTRAL_SENTENCES) for _ in range(count))


def make_split(
    per_class: int,
    templates: tuple[str, ...],
    condition: str,
    delayed_sentences: int,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows: list[tuple[str, int, int]] = []
    retained = [i for i in range(len(RETAINED_CITIES)) for _ in range(per_class)]
    rng.shuffle(retained)
    cursor = 0
    for secret_id, secret in enumerate(SECRET_CODES):
        for repetition in range(per_class):
            city_id = retained[cursor]
            cursor += 1
            displayed_secret = "[REDACTED]" if condition.startswith("no_secret") else secret
            n_filler = delayed_sentences if condition.endswith("delayed") else 0
            template = templates[(secret_id + repetition) % len(templates)]
            text = template.format(
                city=RETAINED_CITIES[city_id],
                secret=displayed_secret,
                filler=filler_text(n_filler, rng),
            )
            rows.append((text, secret_id, city_id))
    rng.shuffle(rows)
    texts, secret_y, retained_y = zip(*rows)
    return list(texts), np.asarray(secret_y), np.asarray(retained_y)


@torch.inference_mode()
def extract_features(
    model: MambaForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layers: list[int],
    coordinate_indices: dict[int, torch.Tensor],
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            # Dynamic left-padding avoids hundreds of artificial EOS updates in
            # short conditions while retaining efficient batched extraction.
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=True, return_dict=True)
        cache = output.cache_params
        selected = []
        for layer in layers:
            flat = cache.ssm_states[layer].float().flatten(1)
            selected.append(flat.index_select(1, coordinate_indices[layer]))
        features = torch.cat(selected, dim=1)
        chunks.append(features.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0)


def fit_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    probe = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=10.0),
    )
    probe.fit(x_train, y_train)
    pred = probe.predict(x_test)
    return float(accuracy_score(y_test, pred)), float(balanced_accuracy_score(y_test, pred))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    layers = [int(item) for item in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    state_width = model.config.intermediate_size * model.config.state_size
    if args.coordinates_per_layer > state_width:
        raise ValueError(f"coordinates-per-layer exceeds flattened state width {state_width}")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    coordinate_indices = {
        layer: torch.randperm(state_width, generator=generator)[: args.coordinates_per_layer].to(args.device)
        for layer in layers
    }

    all_results: list[ConditionResult] = []
    conditions = (
        "immediate",
        "delayed",
        "no_secret_immediate",
        "no_secret_delayed",
    )
    for condition_id, condition in enumerate(conditions):
        # Matched secret/no-secret pairs use identical labels, template choices,
        # and filler draws. Only the sensitive string changes.
        pair_id = 1 if condition.endswith("delayed") else 0
        train_texts, train_secret, train_retained = make_split(
            args.train_per_class,
            TRAIN_TEMPLATES,
            condition,
            args.delayed_sentences,
            args.seed + 100 * pair_id,
        )
        test_texts, test_secret, test_retained = make_split(
            args.test_per_class,
            TEST_TEMPLATES,
            condition,
            args.delayed_sentences,
            args.seed + 100 * pair_id + 1,
        )
        started = time.perf_counter()
        x_train = extract_features(
            model, tokenizer, train_texts, layers, coordinate_indices,
            args.batch_size, args.max_length, args.device,
        )
        x_test = extract_features(
            model, tokenizer, test_texts, layers, coordinate_indices,
            args.batch_size, args.max_length, args.device,
        )
        extraction_seconds = time.perf_counter() - started

        secret_acc, secret_bal = fit_score(x_train, train_secret, x_test, test_secret)
        retained_acc, retained_bal = fit_score(x_train, train_retained, x_test, test_retained)
        permuted = np.random.default_rng(args.seed + condition_id).permutation(train_secret)
        permuted_acc, _ = fit_score(x_train, permuted, x_test, test_secret)
        result = ConditionResult(
            condition=condition,
            train_samples=len(train_texts),
            test_samples=len(test_texts),
            secret_accuracy=secret_acc,
            secret_balanced_accuracy=secret_bal,
            retained_accuracy=retained_acc,
            retained_balanced_accuracy=retained_bal,
            permuted_secret_accuracy=permuted_acc,
            chance_accuracy=1.0 / len(SECRET_CODES),
            extraction_seconds=extraction_seconds,
        )
        all_results.append(result)
        print(json.dumps(asdict(result), indent=2), flush=True)

    output = {
        "status": "exploratory_stage_1",
        "claim_limit": "Linear decodability is evidence of information availability, not proof of practical reconstruction or policy-compliant deletion.",
        "configuration": vars(args),
        "model_state_width_per_layer": state_width,
        "feature_dimensions": len(layers) * args.coordinates_per_layer,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / (2**20), 2) if torch.cuda.is_available() else None,
        "results": [asdict(item) for item in all_results],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, output)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()

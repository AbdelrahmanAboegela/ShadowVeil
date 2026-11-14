"""Prospective shadow-state revocation on human-annotated Enron email text."""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, MambaForCausalLM

from stage1_leakage_probe import RETAINED_CITIES
from stage10_shadow_revocation_tab import (
    fit_predict,
    run_streams,
    sample_indices,
    timed_prefill,
    verify_exact_boundary,
)


DATA_ROOT = Path("data/enron-annotated")
NAME_TAG = re.compile(r"<true_name>\s*(.*?)\s*</true_name>", flags=re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]+>")
TRAIN_PREFIX = (
    "Internal email excerpt: {context}\n"
    "Non-private routing category: {retained}. "
    "Retention event: the identified person in this excerpt expires now."
)
TEST_PREFIX = (
    "Corporate message passage: {context}\n"
    "Public routing category to retain: {retained}. "
    "Privacy deadline: revoke the person in this passage now."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage11_shadow_revocation_enron.json")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--attacker-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--context-characters", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--surrogate-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def valid_name(name):
    return (
        2 <= len(name) <= 60
        and 1 <= len(name.split()) <= 5
        and re.fullmatch(r"[^\W\d_]+(?:[ .,'\-][^\W\d_]+)*", name, flags=re.UNICODE) is not None
    )


def normalize(text):
    return " ".join(text.split())


def load_enron(context_characters):
    records = {"train": [], "test": []}
    surfaces = Counter()
    for path in DATA_ROOT.rglob("*.txt"):
        lower_parts = {part.casefold() for part in path.parts}
        split = "train" if "train" in lower_parts else "test" if "test" in lower_parts else None
        if split is None:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        matches = list(NAME_TAG.finditer(raw))
        for mention_index, match in enumerate(matches):
            surface = normalize(match.group(1))
            if not valid_name(surface):
                continue
            left = max(0, match.start() - context_characters)
            right = min(len(raw), match.end() + context_characters)
            window = raw[left:right]
            local_start = match.start() - left
            local_end = match.end() - left
            marked = window[:local_start] + " __TARGET_PERSON__ " + window[local_end:]
            marked = NAME_TAG.sub(" [OTHER PERSON] ", marked)
            marked = TAG.sub(" ", marked)
            base = normalize(marked)
            if "__TARGET_PERSON__" not in base:
                continue
            records[split].append({
                "doc_key": str(path.relative_to(DATA_ROOT)).replace("\\", "/"),
                "mention_index": mention_index,
                "surface": surface.casefold(),
                "present": base.replace("__TARGET_PERSON__", surface, 1),
                "redacted": base.replace("__TARGET_PERSON__", "[REDACTED PERSON]", 1),
            })
            surfaces[surface.casefold()] += 1
    return records, surfaces


def build_rows(records, surfaces, per_class, seed, neutral_surrogates):
    needed = len(surfaces) * per_class
    by_doc = {}
    for row in records:
        by_doc.setdefault(row["doc_key"], row)
    contexts = list(by_doc.values())
    rng = random.Random(seed)
    rng.shuffle(contexts)
    if len(contexts) < needed:
        raise RuntimeError(f"Need {needed} unique Enron messages, found {len(contexts)}")
    labels = [label for label in range(len(surfaces)) for _ in range(per_class)]
    retained = [index % len(RETAINED_CITIES) for index in range(needed)]
    rng.shuffle(labels)
    rng.shuffle(retained)
    surrogate_ids = [index % len(neutral_surrogates) for index in range(needed)]
    rng.shuffle(surrogate_ids)
    output = []
    for source, label, retained_label, surrogate_id in zip(
        contexts[:needed], labels, retained, surrogate_ids
    ):
        output.append({
            "doc_key": source["doc_key"],
            "present": source["redacted"].replace("[REDACTED PERSON]", surfaces[label], 1),
            "redacted": source["redacted"].replace(
                "[REDACTED PERSON]", neutral_surrogates[surrogate_id], 1
            ),
            "label": label,
            "retained_label": retained_label,
        })
    return output


def render(rows, template, condition):
    return [
        template.format(context=row[condition], retained=RETAINED_CITIES[row["retained_label"]])
        for row in rows
    ]


def labels(rows):
    return (
        np.asarray([row["label"] for row in rows]),
        np.asarray([row["retained_label"] for row in rows]),
    )


def choose_length_matched_surfaces(counts, tokenizer, classes, surrogate_count=16):
    groups = {}
    for name, frequency in counts.items():
        if not valid_name(name):
            continue
        token_count = len(tokenizer.encode(" " + name, add_special_tokens=False))
        groups.setdefault(token_count, []).append((name, frequency))
    candidates = [
        (sum(freq for _, freq in rows), token_count, rows)
        for token_count, rows in groups.items() if len(rows) >= classes + surrogate_count
    ]
    if not candidates:
        raise RuntimeError(f"No tokenizer-length group contains {classes} names")
    _, token_count, rows = max(candidates)
    rows.sort(key=lambda item: item[1], reverse=True)
    surfaces = [name for name, _ in rows[:classes]]
    surrogates = [name for name, _ in rows[classes:classes + surrogate_count]]
    return surfaces, surrogates, token_count


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    records, counts = load_enron(args.context_characters)
    surfaces, neutral_surrogates, identity_token_count = choose_length_matched_surfaces(
        counts, tokenizer, args.classes, args.surrogate_count
    )
    attack_rows = build_rows(
        records["train"], surfaces, args.attacker_per_class, args.seed + 17, neutral_surrogates
    )
    test_rows = build_rows(
        records["test"], surfaces, args.test_per_class, args.seed + 23, neutral_surrogates
    )
    attack_present = render(attack_rows, TRAIN_PREFIX, "present")
    attack_redacted = render(attack_rows, TRAIN_PREFIX, "redacted")
    test_present = render(test_rows, TEST_PREFIX, "present")
    test_redacted = render(test_rows, TEST_PREFIX, "redacted")
    attack_y, attack_retained = labels(attack_rows)
    test_y, test_retained = labels(test_rows)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers, indices = sample_indices(model, args.coordinates_per_layer, args.seed, args.device)
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
        results.append({
            "condition": condition,
            "identity": identity,
            "retained_fact": retained,
            "suffix_nll": nll,
        })
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
        "status": "stage_11_prospective_shadow_revocation_enron",
        "dataset": "CMU Enron-Meetings and Enron-Random personal-name annotations",
        "source": "https://www.cs.cmu.edu/~einat/datasets.html",
        "configuration": vars(args),
        "data_protocol": "real human-annotated emails and person spans with counterfactual name-context assignment",
        "ethical_handling": "raw names and excerpts are excluded from result artifacts",
        "train_mentions": len(records["train"]),
        "test_mentions": len(records["test"]),
        "train_documents": len({row["doc_key"] for row in records["train"]}),
        "test_documents": len({row["doc_key"] for row in records["test"]}),
        "identity_token_count": identity_token_count,
        "surrogate_count": len(neutral_surrogates),
        "surrogate_set_sha256": hashlib.sha256(
            "\n".join(neutral_surrogates).encode("utf-8")
        ).hexdigest(),
        "surface_set_sha256": hashlib.sha256("\n".join(surfaces).encode("utf-8")).hexdigest(),
        "chance_accuracy": 1.0 / args.classes,
        "document_overlap_attacker_test": len(
            {row["doc_key"] for row in attack_rows} & {row["doc_key"] for row in test_rows}
        ),
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

"""Real-context identity recovery using the MIT-licensed TAB/ECHR corpus."""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, MambaForCausalLM

from stage1_leakage_probe import NEUTRAL_SENTENCES
from stage2_unseen_reconstruction import extract


DATA_ROOT = Path("data/text-anonymization-benchmark")
TRAIN_WRAPPER = (
    "Legal document excerpt: {context}\n"
    "Retention event: the identified person in this excerpt has expired and must not be retained. "
    "{suffix} Continue processing only general legal context."
)
TEST_WRAPPER = (
    "Court-record passage: {context}\n"
    "Privacy deadline: forget the person named in this passage now. "
    "{suffix} Proceed with non-identifying legal information."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-130m-hf")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", default="results/stage5_tab_context_validation.json")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--train-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=6)
    parser.add_argument("--context-characters", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coordinates-per-layer", type=int, default=1024)
    parser.add_argument("--layers", default="12,15,17,20,23")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--counterfactual-swap", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_candidates(context_characters):
    occurrences = defaultdict(list)
    for split in ("train", "dev", "test"):
        documents = json.loads((DATA_ROOT / f"echr_{split}.json").read_text(encoding="utf-8"))
        for document in documents:
            annotator = sorted(document["annotations"])[0]
            all_mentions = [
                mention for mention in document["annotations"][annotator]["entity_mentions"]
                if mention["identifier_type"] in {"DIRECT", "QUASI"}
            ]
            seen_surfaces = set()
            for mention in all_mentions:
                if mention["entity_type"] != "PERSON" or mention["identifier_type"] not in {"DIRECT", "QUASI"}:
                    continue
                surface = " ".join(mention["span_text"].split()).casefold()
                if not surface or surface in seen_surfaces:
                    continue
                seen_surfaces.add(surface)
                start, end = mention["start_offset"], mention["end_offset"]
                left = max(0, start - context_characters)
                right = min(len(document["text"]), end + context_characters)
                # Mask every other annotated identifier so the negative control
                # cannot identify the target through co-occurring judges, case
                # numbers, locations, or dates. Keep ordinary legal prose.
                window_mentions = sorted(
                    (
                        other for other in all_mentions
                        if other["start_offset"] >= left and other["end_offset"] <= right
                    ),
                    key=lambda item: (item["start_offset"], item["end_offset"]),
                )
                pieces_present, pieces_redacted = [], []
                cursor = left
                target_id = mention["entity_mention_id"]
                for other in window_mentions:
                    other_start, other_end = other["start_offset"], other["end_offset"]
                    if other_start < cursor:
                        continue
                    ordinary = document["text"][cursor:other_start]
                    pieces_present.append(ordinary)
                    pieces_redacted.append(ordinary)
                    if other["entity_mention_id"] == target_id:
                        pieces_present.append(document["text"][other_start:other_end])
                        pieces_redacted.append("[REDACTED PERSON]")
                    else:
                        placeholder = f"[OTHER {other['entity_type']}]"
                        pieces_present.append(placeholder)
                        pieces_redacted.append(placeholder)
                    cursor = other_end
                pieces_present.append(document["text"][cursor:right])
                pieces_redacted.append(document["text"][cursor:right])
                raw_context = "".join(pieces_present)
                redacted_context = "".join(pieces_redacted)
                occurrences[surface].append({
                    "doc_key": f"{split}:{document['doc_id']}",
                    "context": " ".join(raw_context.split()),
                    "redacted_context": " ".join(redacted_context.split()),
                })
    return occurrences


def choose_dataset(occurrences, classes, train_per_class, test_per_class, seed):
    needed = train_per_class + test_per_class
    eligible = sorted(
        ((surface, rows) for surface, rows in occurrences.items() if len(rows) >= needed),
        key=lambda item: len(item[1]), reverse=True,
    )
    rng = random.Random(seed)
    selected = []
    used_docs = set()
    for surface, rows in eligible:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        available = [row for row in shuffled if row["doc_key"] not in used_docs]
        if len(available) < needed:
            continue
        chosen = available[:needed]
        selected.append((surface, chosen))
        used_docs.update(row["doc_key"] for row in chosen)
        if len(selected) == classes:
            break
    if len(selected) < classes:
        raise RuntimeError(f"Only {len(selected)} document-disjoint person classes satisfy the requested size")
    train, test = [], []
    for label, (surface, rows) in enumerate(selected):
        train.extend((label, surface, row) for row in rows[:train_per_class])
        test.extend((label, surface, row) for row in rows[train_per_class:])
    rng.shuffle(train)
    rng.shuffle(test)
    return selected, train, test


def render(rows, wrapper, redacted):
    suffix = " ".join(NEUTRAL_SENTENCES[:2])
    texts, labels = [], []
    for label, _, row in rows:
        context = row["redacted_context"] if redacted else row["context"]
        texts.append(wrapper.format(context=context, suffix=suffix))
        labels.append(label)
    return texts, np.asarray(labels)


def counterfactual_swap(rows, surfaces, seed):
    """Randomly assign real names to anonymized contexts, balanced by class."""
    rng = random.Random(seed)
    per_class = len(rows) // len(surfaces)
    labels = [label for label in range(len(surfaces)) for _ in range(per_class)]
    rng.shuffle(labels)
    swapped = []
    for (_, _, row), label in zip(rows, labels):
        new_row = dict(row)
        placeholder = "[REDACTED PERSON]"
        if placeholder not in row["redacted_context"]:
            raise RuntimeError("Target placeholder missing from cue-controlled context")
        new_row["context"] = row["redacted_context"].replace(placeholder, surfaces[label], 1)
        new_row["redacted_context"] = row["redacted_context"]
        swapped.append((label, surfaces[label], new_row))
    return swapped


def attack_scores(x_train, y_train, x_test, y_test, seed):
    linear = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))
    linear.fit(x_train, y_train)
    linear_accuracy = float(accuracy_score(y_test, linear.predict(x_test)))
    nonlinear = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128,), alpha=1e-2, batch_size=32,
            learning_rate_init=1e-3, max_iter=120, early_stopping=True,
            random_state=seed,
        ),
    )
    nonlinear.fit(x_train, y_train)
    nonlinear_accuracy = float(accuracy_score(y_test, nonlinear.predict(x_test)))
    return {"linear_accuracy": linear_accuracy, "nonlinear_accuracy": nonlinear_accuracy}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    occurrences = load_candidates(args.context_characters)
    selected, train_rows, test_rows = choose_dataset(
        occurrences, args.classes, args.train_per_class, args.test_per_class, args.seed
    )
    if args.counterfactual_swap:
        surfaces = [surface for surface, _ in selected]
        train_rows = counterfactual_swap(train_rows, surfaces, args.seed + 11)
        test_rows = counterfactual_swap(test_rows, surfaces, args.seed + 29)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    layers = [int(item) for item in args.layers.split(",")]
    width = model.config.intermediate_size * model.config.state_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = {
        layer: torch.randperm(width, generator=generator)[:args.coordinates_per_layer].to(args.device)
        for layer in layers
    }

    results = []
    for redacted in (False, True):
        train_texts, train_labels = render(train_rows, TRAIN_WRAPPER, redacted)
        test_texts, test_labels = render(test_rows, TEST_WRAPPER, redacted)
        x_train = extract(model, tokenizer, train_texts, layers, indices, args.batch_size, args.max_length, args.device)
        x_test = extract(model, tokenizer, test_texts, layers, indices, args.batch_size, args.max_length, args.device)
        condition = "redacted" if redacted else "person_present"
        row = {"condition": condition, **attack_scores(x_train, train_labels, x_test, test_labels, args.seed)}
        results.append(row)
        print(json.dumps(row), flush=True)

    output = {
        "status": "stage_5_tab_real_context_validation",
        "dataset": "Text Anonymization Benchmark v1.0 (MIT)",
        "configuration": vars(args),
        "chance_accuracy": 1.0 / args.classes,
        "document_overlap": 0,
        "counterfactual_identity_context_assignment": args.counterfactual_swap,
        "selected_classes": [
            {"label": index, "surface": surface, "available_documents": len(rows)}
            for index, (surface, rows) in enumerate(selected)
        ],
        "results": results,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, output)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()

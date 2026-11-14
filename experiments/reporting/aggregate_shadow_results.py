"""Aggregate ShadowVeil results, including failed learned-scrubber audits."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from shadowveil.iohelp import results_dir, write_json, write_text_atomic


RESULTS = results_dir()
OUTPUTS = Path("artifacts")


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def exact_mcnemar(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    p = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "untreated_only_correct": left_only,
        "shadow_only_correct": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p,
    }


def aggregate_dataset(files):
    metrics = defaultdict(lambda: defaultdict(list))
    identity_pairs = [[], []]
    retained_pairs = [[], []]
    exact = []
    compute_ratios = []
    swap_ns = []
    retire_ms = []
    for file in files:
        payload = load(file)
        rows = {row["condition"]: row for row in payload["results"]}
        for condition, row in rows.items():
            metrics[condition]["identity"].append(row["identity"]["accuracy"])
            metrics[condition]["retained"].append(row["retained_fact"]["accuracy"])
            metrics[condition]["nll"].append(row["suffix_nll"])
        identity_pairs[0].extend(rows["untreated"]["identity"]["correct"])
        identity_pairs[1].extend(rows["shadow_swap"]["identity"]["correct"])
        retained_pairs[0].extend(rows["untreated"]["retained_fact"]["correct"])
        retained_pairs[1].extend(rows["shadow_swap"]["retained_fact"]["correct"])
        exact.append(payload["exactness"]["bitwise_equal_to_independent_redacted_reference"])
        exact.append(payload["exactness"]["post_suffix_sampled_features_bitwise_equal"])
        compute_ratios.append(payload["prefill_timing"]["dual_to_live_compute_ratio"])
        if "python_pointer_swap_nanoseconds" in payload["exactness"]:
            swap_ns.append(payload["exactness"]["python_pointer_swap_nanoseconds"])
        if "secure_live_cache_retire_ms" in payload["exactness"]:
            retire_ms.append(payload["exactness"]["secure_live_cache_retire_ms"])
    return {
        "conditions": {
            condition: {
                "identity_accuracy": mean_sd(values["identity"]),
                "retained_accuracy": mean_sd(values["retained"]),
                "suffix_nll": mean_sd(values["nll"]),
                "seeds": len(values["identity"]),
            }
            for condition, values in metrics.items()
        },
        "paired_identity_mcnemar": exact_mcnemar(*identity_pairs),
        "paired_retained_mcnemar": exact_mcnemar(*retained_pairs),
        "shadow_identity_vs_chance": {
            "correct": int(sum(identity_pairs[1])),
            "total": len(identity_pairs[1]),
            "chance": 0.125,
            "two_sided_exact_p": float(
                binomtest(sum(identity_pairs[1]), len(identity_pairs[1]), 0.125).pvalue
            ),
        },
        "all_exactness_checks_passed": bool(all(exact)),
        "dual_compute_ratio": mean_sd(compute_ratios),
        "pointer_swap_ns": mean_sd(swap_ns) if swap_ns else None,
        "secure_retire_ms": mean_sd(retire_ms) if retire_ms else None,
    }


def confirm_files(dataset, tag):
    if tag == "130m":
        return [f"confirm_{dataset}_{seed}.json" for seed in (20261001, 20261002, 20261003)]
    return [f"confirm_{dataset}_{tag}_{seed}.json" for seed in (20261001, 20261002, 20261003)]


def main():
    scales = {}
    for tag, label in (("130m", "130M"), ("370m", "370M"), ("1p4b", "1.4B")):
        scales[label] = {
            "TAB": aggregate_dataset(confirm_files("tab", tag)),
            "Enron": aggregate_dataset(confirm_files("enron", tag)),
        }
    tab = scales["130M"]["TAB"]
    enron = scales["130M"]["Enron"]

    bypass = load("stage7_bypass_audit.json")
    generic32 = load("stage9_unseen_identity_scrubber.json")
    generic127 = load("stage9_unseen_identity_rank127_seed_20260928.json")

    def optional(name):
        return load(name) if (RESULTS / name).exists() else None

    per_scale = {}
    for tag, label in (("130m", "130M"), ("370m", "370M"), ("1p4b", "1.4B")):
        sweep = optional(f"fused_sweep_{tag}.json")
        tail = optional(f"batched_tail_{tag}.json")
        snap = optional(f"snapshot_batch_{tag}.json")
        multi = optional(f"multi_item_scaling_{tag}.json")
        per_scale[label] = {
            "fused_sweep": None if sweep is None else {
                "step_ms_by_batch_rows": sweep["step_ms_by_batch_rows"], "chunk_ms_by_tokens": sweep["chunk_ms_by_tokens"],
                "delta_sweep": sweep["delta_sweep"], "exactness_additional_documents": sweep["exactness_additional_documents"],
                "fused_vs_hf_relative_state_difference": sweep["fused_vs_hf_relative_state_difference"],
                "host_load": sweep.get("host_load")},
            "batched_tail": None if tail is None else tail["rows"],
            "snapshot_and_batch_tests": None if snap is None else snap["summary"],
            "multi_item_scaling": None if multi is None else multi["rows"],
        }

    summary = {
        "method": "ShadowVeil: deadline-aware scheduling of exact counterfactual recurrent-state reconstruction",
        "protocol": "docs/prespecified_protocol.md",
        "primary_results_130m": {"TAB": tab, "Enron": enron},
        "scale_replication": scales,
        "measurements_by_scale": per_scale,
        "retirement_latency": optional("retirement_latency.json"),
        "equivalence_statistics": optional("equivalence_stats.json"),
        "mamba2_portability_check": load("stage24_mamba2_portability_check.json"),
        "mamba2_lookahead_sweep": load("stage25_mamba2_lookahead_sweep.json"),
        "subspace_editing_baseline": {
            "same_layer_bypass": [
                row for row in bypass["results"]
                if row["attacker_view"] in {"edited", "heldout_same_layers", "combined"}
            ],
            "unseen_identity_rank_32": generic32["results"],
            "unseen_identity_rank_127": generic127["results"],
            "reading": (
                "Projection suppresses probes on the edited coordinates but leaves signal in disjoint coordinates "
                "and does not generalize to identities absent from defense training."
            ),
        },
        "sanitizer_ablation": {
            "fixed_placeholder": "preserves privacy but shifts the retained probe more on Enron",
            "sixteen_surrogate_pool": "reduces semantic distortion but injects nuisance identity variation",
            "selected": "one disjoint natural pseudonym with tokenizer length matched to all protected names",
        },
        "scope": [
            "Exactness is relative to the declared sanitized input stream, the recorded execution schedule and the batch geometry; it is not a statement about inferences from correlated public context.",
            "Revoking w items whose expiry order is unknown requires 2^w concurrent branches under the stated service model.",
            "The runtime revokes Mamba cache state, not model weights, logs, host-language strings, tool state or generated output.",
            "Attackers read seeded coordinates, the full cache, next-token logits or greedy generations; adaptive fine-tuned attackers are not evaluated.",
            "Measurements use one laptop GPU and an in-repository Triton continuation; windows reach 8,192 tokens.",
            "Mamba-3 is not evaluated.",
        ],
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUTS / "shadowveil-results-summary.json", summary)
    with (OUTPUTS / "shadowveil-main-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "dataset", "condition", "identity_mean", "identity_sd",
            "retained_mean", "retained_sd", "nll_mean", "nll_sd", "seeds",
        ])
        for model_label, datasets in scales.items():
            for dataset, aggregate in datasets.items():
                for condition, values in aggregate["conditions"].items():
                    writer.writerow([
                        model_label, dataset, condition,
                        values["identity_accuracy"]["mean"], values["identity_accuracy"]["sd"],
                        values["retained_accuracy"]["mean"], values["retained_accuracy"]["sd"],
                        values["suffix_nll"]["mean"], values["suffix_nll"]["sd"], values["seeds"],
                    ])
    print(json.dumps(summary["scale_replication"], indent=2))


if __name__ == "__main__":
    main()

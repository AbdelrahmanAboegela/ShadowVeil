# ShadowVeil confirmation protocol freeze

Frozen before the confirmation runs on 2026-09-28.

## Primary question

Does prospective counterfactual shadow-state revocation reduce adaptive
eight-way identity recovery to the sanitized-reference/chance level while
preserving an independently assigned retained fact and suffix language-model
loss?

## Locked implementation

- Model: `state-spaces/mamba-130m-hf`, FP16 on CUDA.
- Attacker: standardized RidgeClassifier with alpha 10, trained separately on
  each post-condition state (adaptive defense-aware attacker).
- Readout: 1,024 seeded SSM coordinates from every Mamba layer after the fixed
  neutral suffix.
- TAB sanitizer: the existing class-independent `[REDACTED PERSON]` reference.
- Enron sanitizer: exactly one disjoint real pseudonym, selected from the same
  tokenizer-length group as all eight protected names.
- Samples per seed: 12 attacker and 6 test documents per identity class.
- Documents: disjoint attacker/test documents; counterfactual identity-context
  assignment; no raw names or excerpts in result artifacts.
- Conditions: untreated, shadow swap, and full reset.
- Seeds: 20261001, 20261002, 20261003.
- No sample exclusions or post-run hyperparameter changes.

## Locked outcomes

Primary privacy outcome: adaptive identity accuracy. Primary utility outcome:
retained-fact accuracy. Secondary outcomes: suffix NLL, bitwise equality to an
independently evaluated sanitized reference, dual/live compute ratio, and
explicit cache-retirement checks.

Aggregation reports mean and sample standard deviation across seeds, pooled
paired exact McNemar tests over per-example correctness, and raw per-seed
values. The paired p-values are descriptive because runs share a checkpoint;
seed count, not example count, controls architecture-level generalization.

## Decision rule

The mechanism passes this gate only if every exactness check passes, mean
shadow identity accuracy is no greater than 18.75% (chance plus one-half of a
class), and retained accuracy is not significantly worse under the paired
exact test at alpha 0.05 on either dataset. No claim of certified deletion,
weight unlearning, arbitrary multi-item revocation, or cross-architecture
generality is permitted.

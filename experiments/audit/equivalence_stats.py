"""Seed-level equivalence statistics for the shadow identity-recovery result.

Replaces 'pooled Wilson upper bound vs an unjustified 18.75% threshold' by tests that respect the data
structure. Unit of analysis = seed (documents are re-drawn from the same pool, so pooled-document
binomial tests overstate power). For each model x corpus:
  * per-seed shadow identity accuracy (prespecified seeds 20261001-3 plus seeds 20261004-8),
  * mean, sample SD, 95% t interval,
  * TOST equivalence to chance with margin delta = half a class (6.25 pp), i.e. both
    H0: mean >= 18.75% and H0: mean <= 6.25% must be rejected (the 18.75% bound is the prespecified rule),
  * the prespecified decision rule (mean <= 18.75%) evaluated on all seeds.
The three prespecified seeds are the primary result; further seeds tighten the interval.
"""

from __future__ import annotations

from shadowveil.iohelp import results_dir, write_json, write_text_atomic

import glob
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = results_dir()
CHANCE, MARGIN = 0.125, 0.0625


def shadow_accuracy(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    row = next(r for r in d["results"] if r["condition"] == "shadow_swap")
    return row["identity"]["accuracy"], len(row["identity"]["correct"])


def collect():
    groups = {}
    for p in sorted(glob.glob(str(RESULTS / "confirm_*.json"))) + sorted(glob.glob(str(RESULTS / "protocol_*.json"))):
        name = Path(p).name
        m = re.match(r"(?:confirm|protocol)_(tab|enron)(?:_(370m|790m|1p4b))?_(\d+)\.json", name)
        if not m:
            continue
        dataset, scale, seed = m.group(1), m.group(2) or "130m", int(m.group(3))
        prespecified = name.startswith("confirm")
        acc, n = shadow_accuracy(p)
        groups.setdefault((scale, dataset), []).append({"seed": seed, "acc": acc, "n": n, "prespecified": prespecified})
    return groups


def tost(acc):
    a = np.asarray(acc)
    n = len(a)
    mean, sd = a.mean(), a.std(ddof=1)
    se = sd / math.sqrt(n)
    t_up = (mean - (CHANCE + MARGIN)) / se          # H0: mean >= chance + margin
    t_lo = (mean - (CHANCE - MARGIN)) / se          # H0: mean <= chance - margin
    p_up = stats.t.cdf(t_up, n - 1)
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    half = stats.t.ppf(0.975, n - 1) * se
    return {"n_seeds": n, "mean": float(mean), "sd": float(sd), "ci95": [float(mean - half), float(mean + half)],
            "p_upper": float(p_up), "p_lower": float(p_lo), "tost_p": float(max(p_up, p_lo)),
            "mean_le_threshold": bool(mean <= CHANCE + MARGIN)}


def main():
    groups = collect()
    out = {"margin_pp": MARGIN * 100, "chance_pp": CHANCE * 100, "rows": []}
    for (scale, dataset), items in sorted(groups.items()):
        prespecified = [i["acc"] for i in items if i["prespecified"]]
        allseeds = [i["acc"] for i in items]
        row = {"scale": scale, "dataset": dataset, "prespecified_only": tost(prespecified) if len(prespecified) > 1 else None,
               "all_seeds": tost(allseeds) if len(allseeds) > len(prespecified) else None,
               "seeds": [{"seed": i["seed"], "acc": i["acc"], "prespecified": i["prespecified"]} for i in items]}
        out["rows"].append(row)
        for tag in ("prespecified_only", "all_seeds"):
            r = row[tag]
            if r:
                print(f"{scale:>5} {dataset:>5} {tag:>14}: n={r['n_seeds']} mean={100*r['mean']:.1f}% sd={100*r['sd']:.1f} "
                      f"CI=[{100*r['ci95'][0]:.1f},{100*r['ci95'][1]:.1f}] TOST p={r['tost_p']:.4f}")
    write_json(RESULTS / "equivalence_stats.json", out)
if __name__ == "__main__":
    main()

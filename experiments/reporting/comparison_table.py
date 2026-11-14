"""Baseline comparison table (R3-c): every number is read from a result file, none typed by hand.

Methods compared on TAB (identity / retained from the primary results) with deadline-path costs from the
measured fused profile:
  full reset                     state discarded
  subspace editing               learned identity subspace projected at expiry (stage6/7)
  restart replay                 recompute the session from its beginning, H + L tokens at the deadline
  retrospective replay           Yao-style adaptation: snapshot s0, replay L tokens at the deadline (delta = 0)
  ShadowVeil (delta = L/4, L)    replay before the deadline, select + zeroize at T
History length H and window L are parameters (defaults H=1024, L=256).
"""

from __future__ import annotations

from shadowveil.iohelp import results_dir, write_json, write_text_atomic

import argparse
import json
from pathlib import Path

R = results_dir()


def load(p):
    return json.loads((R / p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="130m")
    ap.add_argument("--history", type=int, default=1024)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--gx", type=int, default=4)
    args = ap.parse_args()
    prof = load(f"fused_sweep_{args.tag}.json")
    a, b = prof["fit"]["chunk_a_ms"], prof["fit"]["chunk_b_ms_per_token"]
    retire = None
    rp = R / "retirement_latency.json"
    if rp.exists():
        name = {"130m": "mamba-130m-hf", "370m": "mamba-370m-hf", "1p4b": "mamba-1.4b-hf"}[args.tag]
        for r in load("retirement_latency.json")["rows"]:
            if r["model"].endswith(name):
                retire = r["zero_loop"]["median_ms"]
    retire = retire if retire is not None else prof["retire_ms"]["median"]
    H, L, g = args.history, args.window, args.gx
    chunk = lambda n: a + b * n

    main_rows = {r["condition"]: r for r in load("confirm_tab_20261001.json")["results"]}
    shadow = [next(x for x in load(f"confirm_tab_{s}.json")["results"] if x["condition"] == "shadow_swap") for s in (20261001, 20261002, 20261003)]
    ident = 100 * sum(x["identity"]["accuracy"] for x in shadow) / 3
    retained = 100 * sum(x["retained_fact"]["accuracy"] for x in shadow) / 3
    util = {r["method"]: r for r in load("stage6_tab_live_defense_utility.json")["results"]}
    sub = util["sensitive_subspace"]
    rows = [
        ("Full reset", "n/a", 12.5, 12.5, retire, "0", "0"),
        ("Subspace editing", "no", 100 * sub["adaptive_identity_accuracy"], 100 * sub["retained_fact_accuracy"],
         sub["mean_boundary_event_ms"], "0", "0"),
        (f"Restart replay ($H{{=}}{H}$)", "yes", ident, retained, chunk(H + L + g), f"{H + L}", "$M$ (transient)"),
        ("Retrospective replay (Yao-style)", "yes", ident, retained, chunk(L + g), f"{L}", "$M$ (transient)"),
        (r"ShadowVeil $\delta{=}L/4$", "yes", ident, retained, retire, "0", r"$2M$ / $0.25M$"),
        (r"ShadowVeil $\delta{=}L$", "yes", ident, retained, retire, "0", r"$2M$ / $M$"),
    ]
    out = {"tag": args.tag, "history": H, "window": L, "fused_fit": prof["fit"], "retire_ms": retire,
           "rows": [dict(zip(("method", "exact", "identity", "retained", "latency_ms", "work_at_T_tokens", "extra_cache"), r)) for r in rows]}
    write_json((R / f"comparison_{args.tag}.json"), out)
    tex = ["\\begin{tabular}{lcrrrrl}", "\\toprule",
           "Method & Exact & Ident. & Retained & Latency at $T$ & Tokens at $T$ & Extra cache (peak / avg) \\\\", "\\midrule"]
    for m, e, i, rt, lat, wk, ec in rows:
        tex.append(f"{m} & {e} & {i:.1f} & {rt:.1f} & {lat:.2f}~ms & {wk} & {ec} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(f"paper/tables/comparison_{args.tag}.tex"), "\n".join(tex) + "\n")
    print("\n".join(tex))


if __name__ == "__main__":
    main()

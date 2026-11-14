"""Emit paper/numbers.tex: every quantitative claim in the text as a LaTeX macro computed from result files.

Macro names use letters only; scale suffix A = 130M, B = 370M, C = 1.4B. Missing inputs are skipped (the macro
is then undefined and the paper build flags it), so the script can be re-run as results arrive.
"""

from __future__ import annotations

from shadowveil.iohelp import results_dir, write_json, write_text_atomic

import glob
import os
import json
import re
import statistics
from pathlib import Path

from shadowveil.hostload import is_quiet

R = results_dir()
SC = [("130m", "A"), ("370m", "B"), ("1p4b", "C")]
MODEL = {"130m": "mamba-130m-hf", "370m": "mamba-370m-hf", "1p4b": "mamba-1.4b-hf"}
out = {}


skipped = []


def put(name, value, digits=None, unit="", timing_of=None):
    """timing_of: the result dict a timing value came from; it is cited only if its host-load record is quiet."""
    if value is None:
        return
    if timing_of is not None and not is_quiet(timing_of.get("host_load")):
        skipped.append(name)
        return
    out[name] = (f"{value:.{digits}f}" if digits is not None else str(value)) + unit


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fused():
    ret = load("retirement_latency.json")
    for tag, L in SC:
        d = load(f"fused_sweep_{tag}.json")
        if d:
            st, ch = d["step_ms_by_batch_rows"], d["chunk_ms_by_tokens"]
            put(f"nStepOne{L}", st["1"]["median"], 1, timing_of=d)
            put(f"nStepEight{L}", st["8"]["median"], 1, timing_of=d)
            put(f"nChunkSmall{L}", ch["256"]["median"], 0, timing_of=d)
            put(f"nChunkLarge{L}", ch["8192"]["median"] if "8192" in ch else None, 0, timing_of=d)
            ns = sorted(int(k) for k in ch)
            slope = (ch[str(ns[-1])]["median"] - ch[str(ns[-2])]["median"]) / (ns[-1] - ns[-2])
            put(f"nChunkFixed{L}", ch[str(ns[0])]["median"], 1, timing_of=d)
            put(f"nChunkPerTok{L}", slope, 3, timing_of=d)
            put(f"nStepPerChunkTok{L}", st["1"]["median"] / slope, 0, timing_of=d)
            put(f"nCacheMB{L}", d["cache_bytes"] / 2**20, 2)
            put(f"nFusedHFDiff{L}", 100 * d["fused_vs_hf_relative_state_difference"], 2)
            sw = d["delta_sweep"]
            put(f"nSweepExact{L}", sum(r["exact_bitwise"] for r in sw))
            put(f"nSweepN{L}", len(sw))
            ex = d.get("exactness_additional_documents", [])
            put(f"nExtraExact{L}", sum(r["exact_bitwise"] for r in ex))
            put(f"nExtraN{L}", len(ex))
            if ex and "shadow_token_transitions_counted" in ex[0]:
                put(f"nTransOK{L}", int(all(r["shadow_token_transitions_counted"] == r["expected_g_plus_window"] for r in ex)))
            by = {r["delta"]: r for r in sw}
            if 0 in by and 256 in by:
                put(f"nBurstZero{L}", by[0]["burst_ms"]["median"], 1, timing_of=d)
            if 0 in by:
                dmax = max(by)
                put(f"nTotalFull{L}", by[dmax]["total_extra_ms"]["median"], 0, timing_of=d)
        if ret:
            for r in ret["rows"]:
                if r["model"].endswith(MODEL[tag]):
                    put(f"nRetire{L}", r["zero_loop"]["median_ms"], 2, timing_of=ret)
                    put(f"nRetireGrouped{L}", r["zero_grouped"]["median_ms"], 2, timing_of=ret)
        bt = load(f"batched_tail_{tag}.json")
        if bt:
            fr = [r["marginal_fraction_of_live"] for r in bt["rows"]]
            put(f"nBatchMarginLo{L}", 100 * min(fr), 0, timing_of=bt)
            put(f"nBatchMarginHi{L}", 100 * max(fr), 0, timing_of=bt)
            put(f"nBatchExact{L}", sum(r["shadow_row_bitwise_equal_to_reference_row"] for r in bt["rows"]))
            put(f"nBatchN{L}", len(bt["rows"]))


def kill_tests():
    for tag, L in SC:
        d = load(f"snapshot_batch_{tag}.json")
        if not d:
            continue
        s = d["summary"]
        flat = [x for c in d["k1_cases"] for x in c["rows"]]
        put(f"nKOneExact{L}", sum(x["shadow_equals_reference_bitwise"] for x in flat))
        put(f"nKOneN{L}", len(flat))
        put(f"nKOneChunkRec{L}", 100 * s["k1_max_rel_chunk_vs_recurrent"], 2)
        put(f"nKOneMono{L}", 100 * s["k1_max_rel_shadow_vs_monolithic"], 2)
        put(f"nKOneSignal{L}", 100 * s["k1_min_rel_live_vs_sanitized"], 1)
        put(f"nKOneNaiveLo{L}", 100 * min(x["naive_hf_burst_vs_recurrent_rel"] for x in flat), 1)
        put(f"nKOneNaiveHi{L}", 100 * max(x["naive_hf_burst_vs_recurrent_rel"] for x in flat), 1)
        put(f"nKOneTop{L}", 100 * s["k1_top1_agreement_vs_monolithic"], 0)
        g = s.get("k2_geometry") or {}
        put(f"nKTwoGeom{L}", int(all(g.values())) if g else None)


def sim():
    for tag, L in SC:
        d = load(f"scheduler_sim_fused_{tag}.json")
        if not d:
            continue
        rows = [r for r in d["rows"] if r["cost_model"] == "measured_rows" and r["tau_ms"] == (50.0 if L == "C" else d["primary_tau_ms"])
                and r["aligned_fraction"] == 1.0 and r["concurrency"] == 64]
        by = {r["policy"]: r for r in rows}
        for pol, nm in (("retro", "Retro"), ("lastar", "Star"), ("la16", "LaSixteen"), ("la25", "LaQuarter"), ("la50", "LaHalf"), ("full", "Full"), ("capped", "Capped")):
            if pol in by:
                put(f"nSim{nm}PNinetyNine{L}", by[pol]["latency_p99_ms"], 1)
                put(f"nSim{nm}Mem{L}", by[pol]["mean_cache_mb"], 1)
                put(f"nSim{nm}Miss{L}", 100 * by[pol]["slo_miss_rate"], 0)
        put(f"nSimTau{L}", 50.0 if L == "C" else d["primary_tau_ms"], 0)
        if "la16" in by and "retro" in by and "full" in by:
            put(f"nSimLaMemVsRetro{L}", 100 * (by["la16"]["mean_cache_mb"] / by["retro"]["mean_cache_mb"] - 1), 0)
            put(f"nSimLaMemVsFull{L}", 100 * (1 - by["la16"]["mean_cache_mb"] / by["full"]["mean_cache_mb"]), 0)
        typ = {r["policy"]: r for r in d["rows"] if r["cost_model"] == "measured_rows" and r["tau_ms"] == (50.0 if L == "C" else d["primary_tau_ms"])
               and r["aligned_fraction"] == 0.5 and r["concurrency"] == 16}
        for pol, nm in (("retro", "Retro"), ("lastar", "Star"), ("la16", "LaSixteen"), ("la50", "LaHalf"), ("full", "Full")):
            if pol in typ:
                put(f"nTyp{nm}Miss{L}", 100 * typ[pol]["slo_miss_rate"], 0)
                put(f"nTyp{nm}Mem{L}", typ[pol]["mean_cache_mb"], 0)
                put(f"nTyp{nm}PNinetyNine{L}", typ[pol]["latency_p99_ms"], 0)


def audit():
    stats = load("equivalence_stats.json")
    scale_key = {"130m": "A", "370m": "B", "790m": "D", "1p4b": "C"}
    if stats:
        for r in stats["rows"]:
            sc = scale_key[r["scale"]]
            ds = {"tab": "Tab", "enron": "Enron"}[r["dataset"]]
            t = r["all_seeds"] or r["prespecified_only"]
            if t:
                put(f"nEq{ds}{sc}Mean", 100 * t["mean"], 1)
                put(f"nEq{ds}{sc}Sd", 100 * t["sd"], 1)
                put(f"nEq{ds}{sc}Lo", 100 * t["ci95"][0], 1)
                put(f"nEq{ds}{sc}Hi", 100 * t["ci95"][1], 1)
                put(f"nEq{ds}{sc}P", t["tost_p"], 3)
                put(f"nEq{ds}{sc}N", t["n_seeds"])
    cname = {"untreated": "Untreated", "shadow_primary_sanitizer": "Primary", "shadow_leaky_sanitizer": "Leaky"}
    kname = {"ridge": "Ridge", "full": "Full", "logit": "Logit", "logitbest": "LogitBest", "gen": "Gen", "permp": "PermP"}
    for prefix, pattern in (("", "audit_controls_{tag}_*.json"), ("E", "audit_controls_enron_{tag}_*.json")):
        for tag, L in SC:
            files = sorted(glob.glob(str(R / pattern.format(tag=tag))))
            if not files:
                continue
            acc, exact = {}, []
            for f in files:
                d = json.loads(Path(f).read_text(encoding="utf-8"))
                exact.append(d["exactness"]["primary_sanitizer"]["bitwise_equal"] and d["exactness"]["leaky_sanitizer"]["bitwise_equal"])
                for row in d["rows"]:
                    c = row["condition"]
                    acc.setdefault((c, "ridge"), []).append(row["ridge_seeded_coordinates"]["accuracy"])
                    if "ridge_full_cache" in row:
                        acc.setdefault((c, "full"), []).append(row["ridge_full_cache"]["accuracy"])
                    if "logit_candidate_attacker" in row:
                        acc.setdefault((c, "logit"), []).append(row["logit_candidate_attacker"]["calibrated_accuracy"])
                        acc.setdefault((c, "logitbest"), []).append(row["logit_candidate_attacker"]["best_calibrated_over_probes"])
                    if "generation_attacker" in row:
                        acc.setdefault((c, "gen"), []).append(row["generation_attacker"]["true_name_rate"])
                    acc.setdefault((c, "permp"), []).append(row["permutation_null_ridge"]["p_value_ge_observed"])
            put(f"n{prefix}AuditSeeds{L}", len(files))
            put(f"n{prefix}AuditExact{L}", sum(exact))
            for (c, kind), v in acc.items():
                put(f"n{prefix}Aud{cname[c]}{kname[kind]}{L}", 100 * statistics.mean(v) if kind != "permp" else statistics.mean(v), 1 if kind != "permp" else 3)


def table_main():
    key = {"130m": "A", "370m": "B", "790m": "D", "1p4b": "C"}
    groups = {}
    for p in sorted(glob.glob(str(R / "confirm_*.json"))) + sorted(glob.glob(str(R / "protocol_*.json"))):
        name = Path(p).name
        if name.endswith(".host.json"):
            continue
        m = re.match(r"(?:confirm|protocol)_(tab|enron)(?:_(370m|790m|1p4b))?_(\d+)\.json", name)
        if not m:
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = {r["condition"]: r for r in d["results"]}
        g = groups.setdefault((m.group(2) or "130m", m.group(1)), {"ui": [], "si": [], "ur": [], "sr": []})
        g["ui"].append(rows["untreated"]["identity"]["accuracy"]); g["si"].append(rows["shadow_swap"]["identity"]["accuracy"])
        g["ur"].append(rows["untreated"]["retained_fact"]["accuracy"]); g["sr"].append(rows["shadow_swap"]["retained_fact"]["accuracy"])
    for (tag, ds), g in groups.items():
        k, name = key[tag], {"tab": "Tab", "enron": "Enron"}[ds]
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        put(f"nTU{name}{k}", 100 * statistics.mean(g["ui"]), 1); put(f"nTU{name}{k}Sd", 100 * sd(g["ui"]), 1)
        put(f"nTS{name}{k}", 100 * statistics.mean(g["si"]), 1); put(f"nTS{name}{k}Sd", 100 * sd(g["si"]), 1)
        put(f"nTRU{name}{k}", 100 * statistics.mean(g["ur"]), 1); put(f"nTRS{name}{k}", 100 * statistics.mean(g["sr"]), 1)
        put(f"nTN{name}{k}", len(g["ui"]))


def strong():
    """Nonlinear probes (MLP and sparse-autoencoder features) on the full cache, three seeds per scale."""
    for tag, L in SC:
        files = sorted(glob.glob(str(R / f"stage19_strong_{tag}_2026*.json")))
        if not files:
            continue
        acc = {}
        for f in files:
            for r in json.loads(Path(f).read_text(encoding="utf-8"))["results"]:
                for k, nm in (("linear_attacker", "Linear"), ("mlp_attacker", "Mlp"), ("sae_attacker", "Sae")):
                    acc.setdefault((r["condition"], nm), []).append(r[k]["accuracy"])
        put(f"nStrongSeeds{L}", len(files))
        for (cond, nm), v in acc.items():
            if cond in ("untreated", "shadow_swap"):
                put(f"nStrong{'Untreated' if cond == 'untreated' else 'Shadow'}{nm}{L}", 100 * statistics.mean(v), 1)


def cfd():
    """Distance of each treated state to the counterfactual (sanitized-stream) execution, mean over seeds."""
    names = {"untreated": "Untreated", "full_reset": "Reset", "sensitive_subspace": "Subspace", "random_subspace": "Random", "shadow": "Shadow"}
    for tag, L in (("130m", "A"), ("130m_fulllayer", "F"), ("370m", "B"), ("1p4b", "C")):
        files = sorted(glob.glob(str(R / f"counterfactual_distance_{tag}_2026*.json")))
        if not files:
            continue
        acc = {}
        for f in files:
            d = json.loads(Path(f).read_text(encoding="utf-8"))["summary"]
            for k, nm in names.items():
                if k in d:
                    for m, mn in (("state_dist_mean", "Dist"), ("kl_mean", "Kl"), ("generation_identical", "GenId")):
                        acc.setdefault((nm, mn), []).append(d[k][m])
        put(f"nCfdSeeds{L}", len(files))
        for (nm, mn), v in acc.items():
            x = statistics.mean(v)
            put(f"nCfd{nm}{mn}{L}", 100 * x if mn == "GenId" else x, 1 if mn == "GenId" else (3 if mn == "Dist" else 4))


def harness():
    """Agreement between the scheduler simulation and the serving harness (same padded batch, same workloads)."""
    for tag, L in SC:
        d = load(f"sim_vs_harness_{tag}.json")
        if not d:
            continue
        put(f"nSvhConfigs{L}", d["configs"])
        put(f"nSvhSlo{L}", 100 * d["slo_verdict_agreement"], 0)
        put(f"nSvhSpearman{L}", d["mean_spearman_p99"], 2)
        put(f"nSvhMedRatio{L}", 2 ** d["median_abs_log2_latency_ratio"], 1)
        put(f"nSvhMemRatio{L}", 2 ** d["median_abs_log2_memory_ratio"], 2)


def harness_ticks():
    """Co-tenant cost in the serving harness at the typical load (16 sessions, half of the expiries aligned):
    p99 tick duration and mean tick duration relative to nominal, averaged over the harness seeds."""
    for tag, L in SC:
        d = load(f"serving_harness_{tag}.json")
        if not d:
            continue
        for pol, nm in (("retro", "Retro"), ("la16", "LaSixteen"), ("full", "Full")):
            rr = [r for r in d["rows"] if r["concurrency"] == 16 and r["aligned_fraction"] == 0.5 and r["policy"] == pol]
            if not rr:
                continue
            put(f"nHarTick{nm}{L}", statistics.mean(r["tick_p99_ms"] for r in rr), 0, timing_of=d)
            put(f"nHarSlow{nm}{L}", statistics.mean(r["slowdown_mean"] for r in rr), 2, timing_of=d)
            put(f"nHarDropped{nm}{L}", sum(r["dropped_arrivals"] for r in rr))


T95_DF7 = 1.8946      # one-sided 95% quantile of Student t, 7 degrees of freedom (eight seeds)


def retained_ni():
    """Paired seed-level retained-label difference (ShadowVeil minus untreated) per (scale, corpus) cell, with the
    one-sided 95% lower bound, against the 6.25-point margin used for the identity equivalence test."""
    groups = {}
    for p in sorted(glob.glob(str(R / "confirm_*.json"))) + sorted(glob.glob(str(R / "protocol_*.json"))):
        name = Path(p).name
        if name.endswith(".host.json"):
            continue
        m = re.match(r"(?:confirm|protocol)_(tab|enron)(?:_(370m|790m|1p4b))?_(\d+)\.json", name)
        if not m:
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows = {r["condition"]: r for r in d["results"]}
        groups.setdefault((m.group(2) or "130m", m.group(1)), []).append(
            100 * (rows["shadow_swap"]["retained_fact"]["accuracy"] - rows["untreated"]["retained_fact"]["accuracy"]))
    cells = []
    for v in groups.values():
        if len(v) != 8:
            continue
        mu, se = statistics.mean(v), statistics.stdev(v) / len(v) ** 0.5
        cells.append((mu, mu - T95_DF7 * se))
    if not cells:
        return
    put("nRetCells", len(cells))
    put("nRetDiffLo", min(c[0] for c in cells), 1)
    put("nRetDiffHi", max(c[0] for c in cells), 1)
    put("nRetNIOk", sum(c[1] >= -6.25 for c in cells))
    put("nRetNIWorstBound", min(c[1] for c in cells), 1)


def scan():
    d = load("scan_bench.json")
    if not d:
        return
    rows = d["rows"]
    for L, tag in (("A", "130m"), ("B", "370m"), ("C", "1p4b")):
        for n, nm in ((1, "One"), (256, "TwoFiftySix"), (1024, "Kilo")):
            r = [x for x in rows if x["scale"] == tag and x["tokens"] == n]
            if r:
                put(f"nScan{nm}Ours{L}", r[0]["ours_ms"], 3)
                put(f"nScan{nm}Official{L}", r[0]["official_ms"], 3)


def multi():
    for tag, L in SC:
        d = load(f"multi_item_scaling_{tag}.json")
        if not d:
            continue
        rows = d["rows"]
        put(f"nMultiExact{L}", sum(r["all_branches_exact"] and r["exact_on_reveal"] for r in rows))
        put(f"nMultiN{L}", len(rows))
        top = max(rows, key=lambda r: r["branches_required"])
        put(f"nMultiMaxBranches{L}", top["branches_required"])
        put(f"nMultiMaxMB{L}", top["peak_cache_bytes"] / 2**20, 0)


def mamba2():
    sizes = {"A": ("mamba2_audit_2026*.json", "130M"), "B": ("mamba2_audit_370m_2026*.json", "370M"), "D": ("mamba2_audit_780m_2026*.json", "780M")}
    for key, (pattern, _) in sizes.items():
        files = sorted(glob.glob(str(R / pattern)))
        if not files:
            continue
        u, s, rs, ru, ex = [], [], [], [], []
        for f in files:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            ex.append(d["exactness"]["bitwise_equal_to_independent_redacted_reference"])
            for r in d["results"]:
                if r["condition"] == "untreated":
                    u.append(r["identity"]["accuracy"]); ru.append(r["retained_fact"]["accuracy"])
                if r["condition"] == "shadow_swap":
                    s.append(r["identity"]["accuracy"]); rs.append(r["retained_fact"]["accuracy"])
        put(f"nMtwoSeeds{key}", len(files))
        put(f"nMtwoExact{key}", sum(ex))
        put(f"nMtwoUntreated{key}", 100 * statistics.mean(u), 1)
        put(f"nMtwoShadow{key}", 100 * statistics.mean(s), 1)
        put(f"nMtwoShadowSd{key}", 100 * statistics.stdev(s) if len(s) > 1 else 0, 1)
        put(f"nMtwoRetU{key}", 100 * statistics.mean(ru), 1)
        put(f"nMtwoRetS{key}", 100 * statistics.mean(rs), 1)
    for key, suffix in (("A", ""), ("B", "_370m"), ("D", "_780m")):
        f = R / f"stage25_mamba2_lookahead_sweep{suffix}.json"
        g = R / f"stage24_mamba2_portability_check{suffix}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            put(f"nMtwoDeltas{key}", len(d["rows"]))
            put(f"nMtwoDeltasExact{key}", sum(bool(r["bitwise_equal_to_independent_reference"]) for r in d["rows"]))
            put(f"nMtwoWindow{key}", d["window_tokens"])
        if g.exists():
            d = json.loads(g.read_text(encoding="utf-8"))
            put(f"nMtwoBoundaryExact{key}", int(bool(d["boundary_equivalence"]["bitwise_equal_to_independent_reference"])))
            put(f"nMtwoParams{key}", round(d["params"] / 1e6), 0)


def main():
    for fn in (fused, kill_tests, sim, audit, table_main, strong, cfd, harness, harness_ticks, retained_ni, scan, multi, mamba2):
        try:
            fn()
        except Exception as e:  # keep going; report
            print(f"[warn] {fn.__name__}: {type(e).__name__}: {e}")
    lines = ["% generated by experiments/make_paper_numbers.py - do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())]
    write_text_atomic(Path(os.environ.get("SV_OUT", "paper/numbers.tex")), "\n".join(lines) + "\n")
    print(f"wrote paper/numbers.tex with {len(out)} macros")
    if skipped:
        print("timing macros NOT emitted (host load not quiet or no record):", ", ".join(sorted(set(skipped))))


if __name__ == "__main__":
    main()

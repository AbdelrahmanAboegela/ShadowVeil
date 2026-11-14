"""Validate the scheduler simulation against the real serving harness (same workloads, same policies, same padded batch).

For every (policy, concurrency, aligned fraction) present in both files the seed-averaged p99 revocation latency and
mean cache memory are compared. Reported: median and maximum absolute log2 ratio (sim / measured), the fraction of
configurations in which both agree on whether a policy meets the deadline (miss rate <= 5%), and the mean Spearman
rank correlation of the policies' p99 latency within each load level.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(v):
    return sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--tau-ms", type=float, default=20.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    sim = json.loads(Path(args.sim).read_text(encoding="utf-8"))["rows"]
    har = json.loads(Path(args.harness).read_text(encoding="utf-8"))["rows"]
    s = {(r["policy"], r["concurrency"], r["aligned_fraction"]): r for r in sim
         if r["cost_model"] == "measured_rows" and r["tau_ms"] == args.tau_ms}
    h = defaultdict(list)
    for r in har:
        h[(r["policy"], r["concurrency"], r["aligned_fraction"])].append(r)
    rows, lat_ratios, mem_ratios = [], [], []
    for key, runs in sorted(h.items()):
        if key not in s:
            continue
        m_lat, m_mem = mean([r["latency_p99_ms"] for r in runs]), mean([r["mean_cache_mb"] for r in runs])
        m_miss = mean([r["slo_miss_rate"] for r in runs])
        floor = 0.05  # ms; below this both are 'zero' for ratio purposes
        lr = math.log2(max(s[key]["latency_p99_ms"], floor) / max(m_lat, floor))
        mr = math.log2(max(s[key]["mean_cache_mb"], 1e-6) / max(m_mem, 1e-6))
        lat_ratios.append(abs(lr))
        mem_ratios.append(abs(mr))
        rows.append({"policy": key[0], "concurrency": key[1], "aligned": key[2], "sim_p99_ms": s[key]["latency_p99_ms"],
                     "measured_p99_ms": m_lat, "sim_miss": s[key]["slo_miss_rate"], "measured_miss": m_miss, "sim_mean_cache_mb": s[key]["mean_cache_mb"], "measured_mean_cache_mb": m_mem,
                     "log2_latency_ratio": lr, "log2_memory_ratio": mr})
    # ranking agreement per (concurrency, aligned)
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["concurrency"], r["aligned"])][r["policy"]] = r
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    def spearman(x, y):
        rx, ry = ranks(x), ranks(y)
        mx, my = mean(rx), mean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else None

    rhos = []
    for g in groups.values():
        if len(g) < 3:
            continue
        pols = sorted(g)
        rho = spearman([g[q]["sim_p99_ms"] for q in pols], [g[q]["measured_p99_ms"] for q in pols])
        if rho is not None:
            rhos.append(rho)
    slo_agree = mean([1.0 if (r["sim_miss"] <= 0.05) == (r["measured_miss"] <= 0.05) else 0.0 for r in rows]) if rows else None
    tot = len(rhos)
    out = {"configs": len(rows), "median_abs_log2_latency_ratio": sorted(lat_ratios)[len(lat_ratios) // 2] if lat_ratios else None,
           "max_abs_log2_latency_ratio": max(lat_ratios) if lat_ratios else None,
           "median_abs_log2_memory_ratio": sorted(mem_ratios)[len(mem_ratios) // 2] if mem_ratios else None,
           "slo_verdict_agreement": slo_agree, "mean_spearman_p99": mean(rhos) if rhos else None, "ranking_groups": tot, "rows": rows}
    write_json(Path(args.output), out)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()

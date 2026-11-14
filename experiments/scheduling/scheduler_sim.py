"""Scheduler-level simulation of deadline-aware shadow scheduling.

A tick-based discrete simulation of ONE GPU serving many concurrent sessions, each with a protected
span that expires at a deadline T. All costs come from a MEASURED profile (cost_profile.py):
decode-step latency vs batch rows, chunk (burst) latency vs tokens, cache retirement latency, cache size.
This is a simulation driven by measured costs, not a production-scheduler measurement.

Model
  * Every tick (nominal tau ms) each active session needs one decode step; all live rows and all
    shadow-tail rows share one batched step costing c0 + c1*(rows-1) (measured; c1 ~ 0 on this backend).
  * Replay ("burst") work is executed in chunks of <= q tokens with the idle budget left in the tick
    (cost a + b*n per chunk, non-preemptive, earliest-start-first).
  * Policy = lookahead delta: the shadow joins the batched decode for the last delta ticks; before that
    it is built by replaying the sanitized span + allowed tokens observed so far.
      retro  : delta = 0, whole replay starts at the deadline (retrospective replay, Yao-style baseline)
      full   : delta = L, shadow exists from entry (full-window ShadowVeil)
      la25/50: delta = 25% / 50% of the window
      capped : delta = 25% but at most `slots` shadow caches resident at once (memory-capped lookahead)
  * At the deadline, if the shadow is not ready, the remaining replay runs on the critical path and its
    duration is the revocation latency (plus retirement). Expiries in the same tick serialise.
Metrics: revocation latency p50/p99, SLO-miss rate (latency > tau), peak and time-average GPU cache
bytes, tick-duration p99 (decode jitter seen by co-tenants) and mean slowdown vs nominal.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shadowveil.hostload import is_quiet
from shadowveil.iohelp import results_dir, write_json
from shadowveil.slots import slots_for


@dataclass
class Profile:
    c0: float
    c1: float
    a: float
    b: float
    retire: float
    cache_bytes: int
    gx_tokens: int = 4
    ns: tuple = ()
    ms: tuple = ()
    step_rows: tuple = ()          # measured decode-step latency at these row counts (empty: linear model c0 + c1 (rows-1))
    step_ms: tuple = ()

    @staticmethod
    def load(path, burst_scale=1.0, row_scale=None, retire_path=None, retire_kind="zero_grouped", step_rows_path=None):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        f = d["fit"]
        # Retirement must come from the dedicated measurement (profile files built before it include a clone).
        rp = Path(retire_path) if retire_path else results_dir() / "retirement_latency.json"
        if not rp.exists():
            raise FileNotFoundError(f"{rp} missing: run retirement_latency.py first (profile retire_ms includes a clone)")
        rows = json.loads(rp.read_text(encoding="utf-8"))["rows"]
        match = [r for r in rows if r["model"] == d["model"]]
        if not match:
            raise KeyError(f"no retirement measurement for {d['model']}")
        d["retire_ms"] = {"median": match[0][retire_kind]["median_ms"]}
        c1 = f["step_c1_ms_per_extra_row"]
        c0 = f["step_c0_ms"]
        if row_scale is not None:          # pessimistic: a shadow row costs row_scale * (one live step)
            c1 = row_scale * c0
        pts = sorted((int(k), v["median"] * burst_scale) for k, v in d["chunk_ms_by_tokens"].items())
        ns, ms = tuple(x for x, _ in pts), tuple(y for _, y in pts)
        # large-n slope (throughput regime) for extrapolation; small-n floor is the measured fixed cost
        slope = (ms[-1] - ms[-2]) / (ns[-1] - ns[-2])
        sr, sm = (), ()
        if step_rows_path and row_scale is None and Path(step_rows_path).exists():       # measured step curve (the pessimistic scenario keeps its linear model)
            tab = sorted((int(k), v["median"]) for k, v in json.loads(Path(step_rows_path).read_text(encoding="utf-8"))["step_ms_by_rows"].items())
            sr, sm = tuple(x for x, _ in tab), tuple(y for _, y in tab)
        return Profile(c0=c0, c1=max(c1, 0.0), a=ms[0], b=max(slope, 1e-6),
                       retire=d["retire_ms"]["median"], cache_bytes=d["cache_bytes"], ns=ns, ms=ms, step_rows=sr, step_ms=sm)

    def step(self, rows):
        if not rows:
            return 0.0
        if not self.step_rows:
            return self.c0 + self.c1 * max(rows - 1, 0)
        import bisect
        xs, ys = self.step_rows, self.step_ms
        if rows <= xs[0]:
            return ys[0]
        if rows >= xs[-1]:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if len(xs) > 1 else 0.0
            return ys[-1] + slope * (rows - xs[-1])
        i = bisect.bisect_left(xs, rows)
        return ys[i - 1] + (ys[i] - ys[i - 1]) * (rows - xs[i - 1]) / (xs[i] - xs[i - 1])

    def max_tokens_within(self, budget_ms):
        """Largest chunk size whose measured latency fits the budget (0 if none)."""
        lo, hi = 0, 8192
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.chunk(mid) <= budget_ms:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def chunk(self, n):
        """Measured chunk latency: piecewise-linear interpolation over measured token counts."""
        if n <= 0:
            return 0.0
        if not self.ns:
            return self.a + self.b * n
        if n <= self.ns[0]:
            return self.ms[0]
        if n >= self.ns[-1]:
            return self.ms[-1] + self.b * (n - self.ns[-1])
        import bisect
        i = bisect.bisect_left(self.ns, n)
        x0, x1, y0, y1 = self.ns[i - 1], self.ns[i], self.ms[i - 1], self.ms[i]
        return y0 + (y1 - y0) * (n - x0) / (x1 - x0)


@dataclass
class Session:
    t0: int
    T: int
    delta: int
    ts: int = 0                      # tick at which the tail (dual decode) begins
    need: int = 0                    # tokens still to replay by chunk
    released: bool = False
    allocated: bool = False
    tail_active: bool = False
    done_burst: bool = False
    latency: float = 0.0
    release_tick: int = -1

    def __post_init__(self):
        self.ts = self.T - self.delta


def run(profile: Profile, policy: str, n_conc: int, aligned: float, tau: float, q: int = 32, slots: int = 4,
        lead_margin: float = 3.0, q_catch: int = 64, steal: float = 0.5, ticks: int = 3000, warm: int = 600, lo: int = 128, hi: int = 256, seed: int = 0,
        align_period: int = 120, padded: bool = False):
    rng = np.random.default_rng(seed)
    mean_len = (lo + hi) / 2 + (align_period / 2 if aligned > 0 else 0)
    lam = n_conc / mean_len
    active: list[Session] = []
    S = slots_for(n_conc) if padded else 0       # padded: the decode step always runs S rows (2S with shadow rows), as in the serving harness
    lat, tick_ms, mem = [], [], []
    peak_mem = 0
    last_shadow_rows = 0
    for tick in range(ticks):
        for _ in range(rng.poisson(lam)):
            L = int(rng.integers(lo, hi + 1))
            T = tick + L
            if rng.random() < aligned:
                T = int(math.ceil(T / align_period) * align_period)
            L = T - tick
            dstar = max(1, math.ceil(profile.chunk(L) / tau) + 1)      # tokens that arrive while the burst runs, plus one
            delta = {"retro": 0, "full": L, "lastar": min(dstar, L), "la16": min(16, L), "la25": L // 4, "la50": L // 2, "capped": L // 4}[policy]
            s = Session(t0=tick, T=T, delta=delta)
            s.need = profile.gx_tokens                     # sanitized span; observed tokens added at release
            active.append(s)

        # --- chasing: a released shadow observes one new token per tick --------------------------------
        for s_ in active:
            if s_.released and not s_.tail_active and s_.release_tick < tick:
                s_.need += 1
        # --- release bursts just in time (idle-budget estimate) ------------------------------------------
        idle_est = max(tau * (1.0 + steal) - profile.step((2 * S if last_shadow_rows else S) if padded else len(active) + last_shadow_rows), 0.1 * tau)
        # earliest-deadline-first demand: replay work queued ahead of (and including) each session's burst
        ahead, cum = {}, 0.0
        for x in sorted((y for y in active if y.delta > 0 and not y.tail_active), key=lambda y: y.ts):
            if x.released:
                cum += (x.need // q) * profile.chunk(q) + profile.chunk(x.need % q)
            else:
                n_x = profile.gx_tokens + (tick - x.t0)
                cum += (n_x // q) * profile.chunk(q) + profile.chunk(n_x % q)
                ahead[id(x)] = cum
        for s_ in active:
            if s_.released or s_.delta == 0:
                continue
            if s_.delta >= (s_.T - s_.t0):
                ready = True                                   # full window: build at entry
            else:
                ready = (ahead[id(s_)] / idle_est) + lead_margin >= (s_.ts - tick)
            if not ready:
                continue
            if policy == "capped" and sum(1 for x in active if x.allocated) >= slots:
                continue
            s_.released = True
            s_.allocated = True
            s_.need = profile.gx_tokens + (tick - s_.t0)
            s_.release_tick = tick
        mandatory = 0.0
        # --- tail join: final catch-up chunk on the critical path of tick ts ---------------------------
        rows_shadow = 0
        for s_ in active:
            if s_.released and not s_.tail_active and tick >= s_.ts and s_.need <= q_catch:
                mandatory += profile.chunk(s_.need)
                s_.need = 0
                s_.tail_active = True
            if s_.tail_active:
                rows_shadow += 1
        last_shadow_rows = rows_shadow
        # --- expiries ------------------------------------------------------------------------------
        expiring = sorted((s for s in active if s.T == tick), key=lambda s: s.T)
        cum = 0.0
        for s in expiring:
            if s.delta == 0 or not s.tail_active:
                # fallback: complete the remaining replay on the critical path
                if s.delta == 0 or not s.released:
                    remaining = profile.gx_tokens + (s.T - s.t0)
                else:
                    remaining = s.need
                cum += profile.chunk(remaining)
            cum += profile.retire
            if tick >= warm:
                lat.append(cum)
            s.done_burst = True
        mandatory += cum
        # --- batched decode of all live rows + shadow-tail rows --------------------------------------
        rows = (2 * S if rows_shadow else S) if padded else len(active) + rows_shadow
        mandatory += profile.step(rows if active else 0)
        # --- idle budget -> replay chunks (earliest tail-start first) --------------------------------
        budget = max(0.0, tau * (1.0 + steal) - mandatory)
        for s in sorted((x for x in active if x.released and not x.tail_active and x.need > 0), key=lambda x: x.ts):
            while s.need > 0 and budget > profile.chunk(1):
                n = min(q, s.need, profile.max_tokens_within(budget))
                if n <= 0:
                    break
                s.need -= n
                budget -= profile.chunk(n)
                mandatory += profile.chunk(n)
            if budget <= profile.chunk(1):
                break
        dur = max(tau, mandatory)
        # --- memory ---------------------------------------------------------------------------------
        nbytes = profile.cache_bytes * (len(active) + sum(1 for s in active if s.allocated and s.T != tick))
        if policy == "retro":
            nbytes += profile.cache_bytes * len(expiring)      # shadow materialised only at the deadline
        peak_mem = max(peak_mem, nbytes) if tick >= warm else peak_mem
        if tick >= warm:
            tick_ms.append(dur)
            mem.append(nbytes)
        active = [s for s in active if s.T > tick]
    lat = np.array(lat) if lat else np.array([0.0])
    tm = np.array(tick_ms)
    return {
        "policy": policy, "concurrency": n_conc, "aligned_fraction": aligned,
        "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p99_ms": float(np.percentile(lat, 99)),
        "slo_miss_rate": float((lat > tau).mean()), "expiries": int(len(lat)),
        "peak_cache_mb": peak_mem / 2**20, "mean_cache_mb": float(np.mean(mem)) / 2**20,
        "tick_p99_ms": float(np.percentile(tm, 99)), "slowdown_mean": float(tm.mean() / tau),
        "frac_ticks_overrun": float((tm > tau * 1.0001).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tau-ms", default="20", help="comma list: nominal per-session token interval (workload parameter)")
    ap.add_argument("--primary-tau", type=float, default=None, help="tau used for the headline figure (default: first)")
    ap.add_argument("--burst-scale", type=float, default=1.0, help="scale burst cost (what-if)")
    ap.add_argument("--concurrency", default="8,16,32,64,128")
    ap.add_argument("--aligned", default="0.0,0.5,1.0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--padded", action="store_true", help="pad the decode batch to slots_for(N) rows, as the serving harness does")
    ap.add_argument("--retire-kind", default="zero_grouped", choices=["zero_grouped", "zero_loop"])
    args = ap.parse_args()
    step_rows = str(Path(args.profile).with_name(Path(args.profile).name.replace("cost_profile_fused_", "step_rows_")))
    prof = Profile.load(args.profile, burst_scale=args.burst_scale, retire_kind=args.retire_kind, step_rows_path=step_rows)
    prof_pess = Profile.load(args.profile, burst_scale=args.burst_scale, row_scale=1.0, retire_kind=args.retire_kind)
    taus = [float(x) for x in args.tau_ms.split(",")]
    rows = []
    policies = ["retro", "lastar", "la16", "la25", "la50", "full", "capped"]
    for tau in taus:
        for name, prof_use in (("measured_rows", prof), ("pessimistic_rows", prof_pess)):
            for aligned in [float(x) for x in args.aligned.split(",")]:
                for n in [int(x) for x in args.concurrency.split(",")]:
                    for pol in policies:
                        runs = [run(prof_use, pol, n, aligned, tau, seed=sd, padded=args.padded) for sd in range(args.seeds)]
                        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if isinstance(runs[0][k], (int, float))}
                        agg.update({"policy": pol, "cost_model": name, "tau_ms": tau,
                                    "latency_p99_sd": float(np.std([r["latency_p99_ms"] for r in runs])),
                                    "seeds": args.seeds})
                        rows.append(agg)
    out = {"status": "scheduler_simulation", "profile": args.profile, "tau_ms_values": taus,
           "primary_tau_ms": args.primary_tau or taus[0], "burst_scale": args.burst_scale, "retire_kind": args.retire_kind,
           "profile_quiet": bool(is_quiet(json.loads(Path(args.profile).read_text(encoding="utf-8")).get("host_load"))),
           "profile_fit": {"c0": prof.c0, "c1": prof.c1, "a": prof.a, "b": prof.b, "retire": prof.retire,
                           "cache_bytes": prof.cache_bytes},
           "note": "simulation driven by measured costs; not a production-scheduler measurement", "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output), out)
    primary = args.primary_tau or taus[0]
    for r in rows:
        if (r["cost_model"] == "measured_rows" and r["aligned_fraction"] == 1.0 and r["concurrency"] == 32
                and r["tau_ms"] == primary):
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in
                   ("policy", "latency_p99_ms", "slo_miss_rate", "peak_cache_mb", "mean_cache_mb", "tick_p99_ms", "slowdown_mean")})


if __name__ == "__main__":
    main()

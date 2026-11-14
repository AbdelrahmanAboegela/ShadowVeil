"""Real multi-session serving harness on the GPU (validates the scheduler simulation).

Concurrent sessions are executed for real: batched CUDA-graph decode over live rows (and live + shadow-tail rows
when any shadow is in its dual-decode phase), fused burst replay in power-of-two chunks through a scratch cache,
real retirement of the live cache, wall-clock tick pacing. The workload generator replays the simulator's random
calls in the same order, so a (policy, concurrency, aligned, seed) run is the same workload as the simulation.
Revocation latency is wall-clock time from the start of the tick's expiry handling to the completion of that
session's retirement (expiries in one tick serialise), exactly as defined in scheduler_sim.py.

Correctness: for a sample of sessions the selected shadow state is compared numerically with a monolithic fused
replay of the sanitized stream (bitwise exactness is established separately in fused_lookahead_sweep/batched_tail).
Run on a quiet host; the host load is recorded.
"""

from __future__ import annotations

from shadowveil.iohelp import write_json, write_text_atomic

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from transformers.cache_utils import MambaCache

from shadowveil.fused import FusedMamba
from shadowveil.hostload import snapshot
from shadowveil.slots import slots_for
from snapshot_and_batch_tests import build_cases

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


class Views:
    def __init__(self, ssm, conv):
        self.ssm_states, self.conv_states = ssm, conv


def view_rows(cache, n):
    return Views([t[:n] for t in cache.ssm_states], [t[:n] for t in cache.conv_states])


def row_list(cache, j):
    return [t[j] for t in cache.ssm_states] + [t[j] for t in cache.conv_states]


def new_cache(model, rows):
    return MambaCache(model.config, max_batch_size=rows, dtype=next(model.parameters()).dtype, device="cuda")


@torch.inference_mode()
def capture(fm, cache, batch, n):
    ids = torch.zeros((batch, n), dtype=torch.long, device="cuda")
    for _ in range(2):
        fm.forward(cache, ids)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fm.forward(cache, ids)
    return ids, g


def decomp(n, biggest=512):
    out = []
    while n >= biggest:
        out.append(biggest)
        n -= biggest
    for s in reversed(SIZES[:-1]):
        if n >= s:
            out.append(s)
            n -= s
    return out


class Env:
    """Everything shared by runs of one model."""

    def __init__(self, model, tok, args, S, fm=None):
        self.model, self.args = model, args
        self.fm = fm or FusedMamba(model)
        self.S = S
        with torch.inference_mode():
            case = build_cases(tok, 1, 256, args.seed)[0]
            self.live = new_cache(model, 2 * S)                    # rows [0,S) live, [S,2S) dual-decode shadow
            self.stage = new_cache(model, S)                       # burst-building state of released shadows
            self.scratch = new_cache(model, 1)
            self.init_live = new_cache(model, 1)                   # state after P + X (every session starts here)
            self.s0 = new_cache(model, 1)                          # state after P (snapshot)
            self.fm.forward(self.s0, torch.tensor([case["P"]], device="cuda"))
            for a, b in zip(self.init_live.ssm_states + self.init_live.conv_states, self.s0.ssm_states + self.s0.conv_states):
                a.copy_(b)
            self.fm.forward(self.init_live, torch.tensor([case["X"]], device="cuda"))
            self.gX = torch.tensor(case["gX"], dtype=torch.long, device="cuda")
            self.P = case["P"]
            self.g = len(case["gX"])
            self.cache_bytes = sum(t[0].numel() * t.element_size() for t in self.live.ssm_states + self.live.conv_states)
            self.live_v, self.full_v = view_rows(self.live, S), view_rows(self.live, 2 * S)
            self.ids_live, self.g_live = capture(self.fm, self.live_v, S, 1)
            self.ids_full, self.g_full = capture(self.fm, self.full_v, 2 * S, 1)
            self.chunk_graphs = {n: capture(self.fm, self.scratch, 1, n) for n in SIZES}
            self.K = 1024
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            self.tok_live = torch.randint(0, 50000, (self.K, S), generator=gen, device="cuda")
            self.tok_full = torch.cat([self.tok_live, self.tok_live], dim=1)
            self.live_rows = [row_list(self.live, j) for j in range(S)]
            self.stage_rows = [row_list(self.stage, j) for j in range(S)]
            self.dual_rows = [row_list(self.live, S + j) for j in range(S)]
            self.scratch_row = row_list(self.scratch, 0)
            self.init_row = row_list(self.init_live, 0)
            self.s0_row = row_list(self.s0, 0)
        self.cost_chunk, self.cost_live, self.cost_full = self.calibrate()

    def timed(self, fn, reps=15):
        out = []
        for _ in range(3):
            fn()
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append(1000 * (time.perf_counter() - t0))
        return statistics.median(out)

    @torch.inference_mode()
    def calibrate(self):
        c = {n: self.timed(lambda n=n: self.chunk_graphs[n][1].replay()) for n in SIZES}
        return c, self.timed(self.g_live.replay), self.timed(self.g_full.replay)

    def cost_of(self, n):
        return sum(self.cost_chunk[p] for p in decomp(n))

    # -- primitives (async GPU work; callers synchronise when they need wall time) ----------------------------
    def replay_chunk(self, sess, n):
        """Advance session's shadow staging state by n tokens of its sanitized stream."""
        r = sess["replayed"]
        src = self.s0_row if r == 0 else self.stage_rows[sess["j"]]
        torch._foreach_copy_(self.scratch_row, src)
        ids, g = self.chunk_graphs[n]
        ids.copy_(sess["seq"][r:r + n].view(1, -1))
        g.replay()
        torch._foreach_copy_(self.stage_rows[sess["j"]], self.scratch_row)
        sess["replayed"] = r + n

    def zero_live(self, j):
        torch._foreach_zero_(self.live_rows[j])


def run_one(env, policy, N, aligned, seed, args):
    S, g, M = env.S, env.g, env.cache_bytes
    tau, steal, q, q_catch, lead_margin = args.tau_ms, args.steal, 64, 64, 3.0
    lo, hi, align_period = 128, 256, 120
    rng = np.random.default_rng(seed)
    mean_len = (lo + hi) / 2 + (align_period / 2 if aligned > 0 else 0)
    lam = N / mean_len
    free = list(range(S))
    active = {}
    last_dual = 0
    lat, tick_ms, mem, checks = [], [], [], []
    peak_mem = 0
    dropped = 0
    pending_checks = []
    wall0 = time.perf_counter()
    with torch.inference_mode():
        for tick in range(args.ticks):
            t_tick = time.perf_counter()
            for _ in range(rng.poisson(lam)):
                L = int(rng.integers(lo, hi + 1))
                T = tick + L
                if rng.random() < aligned:
                    T = int(math.ceil(T / align_period) * align_period)
                L = T - tick
                if not free:
                    dropped += 1
                    continue
                j = free.pop()
                dstar = max(1, math.ceil(env.cost_of(L) / tau) + 1)
                delta = {"retro": 0, "full": L, "lastar": min(dstar, L), "la16": min(16, L), "la25": L // 4, "la50": L // 2}[policy]
                idx = (tick + torch.arange(L, device="cuda")) % env.K
                seq = torch.cat([env.gX, env.tok_live[idx, j]])
                torch._foreach_copy_(env.live_rows[j], env.init_row)
                active[j] = dict(j=j, t0=tick, T=T, delta=delta, ts=T - delta, need=g, released=False, allocated=False,
                                 tail=False, replayed=0, release_tick=-1, seq=seq)
            sess_list = list(active.values())
            for s in sess_list:
                if s["released"] and not s["tail"] and s["release_tick"] < tick:
                    s["need"] += 1
            dual_rows_now = sum(1 for s in sess_list if s["tail"])
            step_cost = env.cost_full if dual_rows_now else env.cost_live
            idle_est = max(tau * (1 + steal) - step_cost, 0.1 * tau)
            ahead, cum = {}, 0.0
            for x in sorted((y for y in sess_list if y["delta"] > 0 and not y["tail"]), key=lambda y: y["ts"]):
                if x["released"]:
                    cum += env.cost_of(x["need"])
                else:
                    cum += env.cost_of(g + (tick - x["t0"]))
                    ahead[x["j"]] = cum
            for s in sess_list:
                if s["released"] or s["delta"] == 0:
                    continue
                if s["delta"] >= s["T"] - s["t0"]:
                    ready = True
                else:
                    ready = ahead[s["j"]] / idle_est + lead_margin >= (s["ts"] - tick)
                if ready:
                    s["released"], s["allocated"], s["need"], s["release_tick"] = True, True, g + (tick - s["t0"]), tick
            # tail join: catch-up chunks on the critical path of tick ts, then stage -> dual-decode row
            for s in sess_list:
                if s["released"] and not s["tail"] and tick >= s["ts"] and s["need"] <= q_catch:
                    for p in decomp(s["need"]):
                        env.replay_chunk(s, p)
                    s["need"] = 0
                    torch._foreach_copy_(env.dual_rows[s["j"]], env.stage_rows[s["j"]])
                    s["tail"] = True
            # expiries: serialised on the critical path
            expiring = [s for s in sess_list if s["T"] == tick]
            if expiring:
                torch.cuda.synchronize()
                t_exp = time.perf_counter()
                for s in expiring:
                    if s["delta"] == 0 or not s["tail"]:
                        remaining = (g + (s["T"] - s["t0"])) - s["replayed"]
                        for p in decomp(remaining):
                            env.replay_chunk(s, p)
                        torch._foreach_copy_(env.dual_rows[s["j"]], env.stage_rows[s["j"]])
                    if len(checks) + len(pending_checks) < args.check_sessions and tick >= args.warm:
                        pending_checks.append((s["seq"][: g + (s["T"] - s["t0"])].clone(),
                                               [t.clone() for t in env.dual_rows[s["j"]]]))
                    env.zero_live(s["j"])
                    torch.cuda.synchronize()
                    if tick >= args.warm:
                        lat.append(1000 * (time.perf_counter() - t_exp))
                for s in expiring:
                    del active[s["j"]]
                    free.append(s["j"])
            sess_list = list(active.values())
            # batched decode of live rows (+ dual rows when any shadow is in its tail)
            any_tail = any(s["tail"] for s in sess_list)
            if any_tail:
                env.ids_full.copy_(env.tok_full[tick % env.K].view(-1, 1))
                env.g_full.replay()
            elif sess_list:
                env.ids_live.copy_(env.tok_live[tick % env.K].view(-1, 1))
                env.g_live.replay()
            # background replay with the remaining tick budget (earliest tail start first)
            torch.cuda.synchronize()
            for s in sorted((x for x in sess_list if x["released"] and not x["tail"] and x["need"] > 0), key=lambda x: x["ts"]):
                while s["need"] > 0:
                    left = tau * (1 + steal) - 1000 * (time.perf_counter() - t_tick)
                    n = max((p for p in SIZES if p <= min(q, s["need"]) and env.cost_chunk[p] <= left), default=0)
                    if n == 0:
                        break
                    env.replay_chunk(s, n)
                    s["need"] -= n
                    torch.cuda.synchronize()
                else:
                    continue
                break
            torch.cuda.synchronize()
            dur = 1000 * (time.perf_counter() - t_tick)
            if dur < tau:
                time.sleep((tau - dur) / 1000)
                dur = tau
            if tick >= args.warm:
                nbytes = M * (len(active) + sum(1 for x in sess_list if x["allocated"]))
                if policy == "retro":
                    nbytes += M * len(expiring)
                tick_ms.append(dur)
                mem.append(nbytes)
                peak_mem = max(peak_mem, nbytes)
    # numerical correctness of selected shadows against a monolithic replay of the sanitized stream
    with torch.inference_mode():
        for seq, rows in pending_checks:
            ref = new_cache(env.model, 1)
            torch._foreach_copy_(row_list(ref, 0), env.s0_row)
            env.fm.forward(ref, seq.view(1, -1))
            num = max(float((a.float() - b.float()).abs().max()) for a, b in zip(rows, row_list(ref, 0)))
            den = max(float(b.float().abs().max()) for b in row_list(ref, 0))
            checks.append(num / den)
    lat_a = np.array(lat) if lat else np.array([0.0])
    tm = np.array(tick_ms)
    return {"policy": policy, "concurrency": N, "aligned_fraction": aligned, "seed": seed, "tau_ms": tau,
            "latency_p50_ms": float(np.percentile(lat_a, 50)), "latency_p99_ms": float(np.percentile(lat_a, 99)),
            "slo_miss_rate": float((lat_a > tau).mean()), "expiries": int(len(lat)),
            "peak_cache_mb": peak_mem / 2**20, "mean_cache_mb": float(np.mean(mem)) / 2**20,
            "tick_p99_ms": float(np.percentile(tm, 99)), "slowdown_mean": float(tm.mean() / tau),
            "shadow_vs_replay_max_rel_diff": max(checks) if checks else None, "checked_sessions": len(checks),
            "slots": S, "dropped_arrivals": dropped,
            "wall_seconds": time.perf_counter() - wall0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="state-spaces/mamba-130m-hf")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--tag", default="130m")
    ap.add_argument("--concurrency", default="8,16,32,64")
    ap.add_argument("--aligned", default="0.0,1.0")
    ap.add_argument("--policies", default="retro,lastar,la16,la25,la50,full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tau-ms", type=float, default=20.0)
    ap.add_argument("--steal", type=float, default=0.5)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--check-sessions", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    h0 = snapshot()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MambaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    fm = FusedMamba(model)
    env = None
    calib = {}
    path = Path(args.output or f"results/serving_harness_{args.tag}.json")
    partial = path.with_name(path.name + ".partial")
    sig = {k: getattr(args, k) for k in ("model", "tau_ms", "steal", "ticks", "warm", "seed", "check_sessions")}
    sig["batch"] = "padded to slots_for(N) rows per load level"
    rows = []
    if partial.exists():
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("signature") == sig:
            rows = saved["rows"]
            calib = saved.get("calibration_ms", {})
            print(f"resuming: {len(rows)} runs already done", flush=True)
    done = {(r["policy"], r["concurrency"], r["aligned_fraction"], r["seed"]) for r in rows}
    for N in [int(x) for x in args.concurrency.split(",")]:
        todo = [(aligned, pol, sd) for aligned in [float(x) for x in args.aligned.split(",")] for pol in args.policies.split(",")
                for sd in range(args.seeds) if (pol, N, aligned, sd) not in done]
        if not todo:
            continue
        env = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        env = Env(model, tok, args, slots_for(N), fm)           # graphs sized for this load level
        calib[str(N)] = {"slots": env.S, "chunk": env.cost_chunk, "decode_live_rows": env.cost_live, "decode_live_plus_shadow_rows": env.cost_full}
        for aligned, pol, sd in todo:
            r = run_one(env, pol, N, aligned, sd, args)
            rows.append(r)
            write_json(partial, {"signature": sig, "rows": rows, "calibration_ms": calib})
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                              if k in ("policy", "concurrency", "aligned_fraction", "seed", "latency_p99_ms", "slo_miss_rate",
                                       "mean_cache_mb", "slowdown_mean", "shadow_vs_replay_max_rel_diff", "dropped_arrivals")}), flush=True)
    out = {"status": "serving_harness", "model": args.model, "tag": args.tag, "configuration": vars(args),
           "calibration_ms": calib,
           "cache_bytes": env.cache_bytes if env else None, "rows": rows, "host_load": {"before": h0, "after": snapshot(1.0)}}
    write_json(path, out)
    if partial.exists():
        partial.unlink()
    print(f"saved {path}")


if __name__ == "__main__":
    main()

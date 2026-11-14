"""Ordered, resumable, interrupt-safe execution of every experiment in the study.

    python tools/pipeline.py list                  # steps in order with their state
    python tools/pipeline.py status                # compact summary
    python tools/pipeline.py run                   # run everything not yet done (resumes after any interruption)
    python tools/pipeline.py run --only fused      # steps whose name matches a regex
    python tools/pipeline.py run --smoke           # one tiny run of every script into scratch/smoke (no real results touched)

Behaviour
  * A step is done when all of its output files exist (and, for timing steps, the recorded host load was quiet).
    Results are written atomically by the scripts, so an interrupted step leaves no half-written output; long steps
    also keep a `.partial` file and continue from it.
  * Timing steps wait until the host CPU is quiet before starting; every step runs at below-normal priority with
    single-threaded BLAS/OpenMP, so it does not take CPU from other work on the machine.
  * Ctrl-C / termination stops the running step cleanly; running the same command again continues.
  * Output of every step is appended to logs/pipeline/<step>.log.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import site
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(ROOT / "src"))
from shadowveil.hostload import is_quiet, wait_until_quiet  # noqa: E402
from shadowveil.iohelp import load_json  # noqa: E402

EXP = ROOT / "experiments"
SUBDIRS = ["mechanism", "cost", "scheduling", "audit", "reporting"]
MODEL = {"130m": "state-spaces/mamba-130m-hf", "370m": "state-spaces/mamba-370m-hf", "790m": "state-spaces/mamba-790m-hf", "1p4b": "state-spaces/mamba-1.4b-hf"}
MAMBA2 = {"130m": "AntonV/mamba2-130m-hf", "370m": "AntonV/mamba2-370m-hf", "780m": "AntonV/mamba2-780m-hf"}
PY = sys.executable
SMOKE_DIR = ROOT / "scratch" / "smoke"


@dataclass
class Step:
    name: str
    script: str                               # path under experiments/
    args: list
    outputs: list                             # may contain {OUT}
    smoke_args: list | None = None            # None: not part of the smoke run
    timing: bool = False                      # needs a quiet host; quiet record read from quiet_file
    quiet_file: str | None = None             # JSON with "host_load" or a run_timed sidecar with "quiet"
    wrap_timed: str | None = None            # sidecar path: run through run_timed.py
    phase: str = ""
    analysis: bool = False
    always: bool = False                      # cheap reporting step: rerun every time


def S(tag):  # scale helpers
    return MODEL[tag]


def build_steps() -> list[Step]:
    st: list[Step] = []
    A = st.append
    scales = ["130m", "370m", "1p4b"]
    HARNESS_N = {"130m": "8,16,32,64,128", "370m": "8,16,32,64", "1p4b": "8,16,32"}

    # ---- 1. timing core -------------------------------------------------------------------------------------------
    ph = "1-timing-core"
    A(Step("retirement_latency", "cost/retirement_latency.py", ["--output", "{OUT}/retirement_latency.json"],
           ["{OUT}/retirement_latency.json"], ["--models", S("130m"), "--output", "{OUT}/retirement_latency.json"],
           timing=True, quiet_file="{OUT}/retirement_latency.json", phase=ph))
    for t in scales:
        extra = ["--repeats", "5"] if t == "1p4b" else []
        A(Step(f"fused_sweep_{t}", "mechanism/fused_lookahead_sweep.py", ["--model", S(t), "--tag", t, "--out-dir", "{OUT}"] + extra,
               [f"{{OUT}}/fused_sweep_{t}.json", f"{{OUT}}/cost_profile_fused_{t}.json"],
               ["--model", S(t), "--tag", t, "--window", "64", "--deltas", "0,16,64", "--repeats", "2", "--lengths", "8,64,256",
                "--batches", "1,2", "--exact-cases", "2", "--out-dir", "{OUT}"] if t in ("130m", "1p4b") else None,
               timing=True, quiet_file=f"{{OUT}}/fused_sweep_{t}.json", phase=ph))
    for t in scales:
        A(Step(f"step_by_rows_{t}", "cost/step_by_rows.py", ["--model", S(t), "--tag", t, "--out-dir", "{OUT}"],
               [f"{{OUT}}/step_rows_{t}.json"],
               ["--model", S(t), "--tag", t, "--rows", "1,8,32", "--repeats", "5", "--out-dir", "{OUT}"] if t == "130m" else None,
               timing=True, quiet_file=f"{{OUT}}/step_rows_{t}.json", phase=ph))
    for t in ("130m", "370m"):
        tag = f"{t}_w1024"
        A(Step(f"fused_sweep_{tag}", "mechanism/fused_lookahead_sweep.py",
               ["--model", S(t), "--tag", tag, "--window", "1024", "--deltas", "0,128,256,512,1024", "--exact-cases", "3", "--repeats", "5",
                "--out-dir", "{OUT}"],
               [f"{{OUT}}/fused_sweep_{tag}.json"], None, timing=True, quiet_file=f"{{OUT}}/fused_sweep_{tag}.json", phase=ph))
    for t in scales:
        A(Step(f"batched_tail_{t}", "mechanism/batched_tail.py", ["--model", S(t), "--tag", t, "--out-dir", "{OUT}"],
               [f"{{OUT}}/batched_tail_{t}.json"],
               ["--model", S(t), "--tag", t, "--window", "64", "--deltas", "16,64", "--repeats", "2", "--out-dir", "{OUT}"] if t == "130m" else None,
               timing=True, quiet_file=f"{{OUT}}/batched_tail_{t}.json", phase=ph))
    # ---- Mamba-1 790M: fused-path measurements and exactness ------------------------------------------------------------
    A(Step("fused_sweep_790m", "mechanism/fused_lookahead_sweep.py", ["--model", S("790m"), "--tag", "790m", "--out-dir", "{OUT}"],
           ["{OUT}/fused_sweep_790m.json", "{OUT}/cost_profile_fused_790m.json"], None, timing=True, quiet_file="{OUT}/fused_sweep_790m.json", phase=ph))
    A(Step("batched_tail_790m", "mechanism/batched_tail.py", ["--model", S("790m"), "--tag", "790m", "--out-dir", "{OUT}"],
           ["{OUT}/batched_tail_790m.json"], None, timing=True, quiet_file="{OUT}/batched_tail_790m.json", phase=ph))
    for t in scales:
        extra = ["--repeats", "8"] if t == "1p4b" else []
        A(Step(f"cost_profile_hf_{t}", "cost/cost_profile.py", ["--model", S(t), "--output", f"{{OUT}}/cost_profile_hf_{t}.json"] + extra,
               [f"{{OUT}}/cost_profile_hf_{t}.json"],
               ["--model", S(t), "--output", f"{{OUT}}/cost_profile_hf_{t}.json", "--batches", "1,2", "--lengths", "8,32", "--repeats", "3"] if t == "130m" else None,
               timing=True, quiet_file=f"{{OUT}}/cost_profile_hf_{t}.json", phase=ph))
    for t in scales:
        A(Step(f"deadline_latency_{t}", "cost/stage12_deadline_latency.py",
               ["--model", S(t), "--output", f"{{OUT}}/deadline_latency_{t}.json"], [f"{{OUT}}/deadline_latency_{t}.json"],
               ["--model", S(t), "--output", f"{{OUT}}/deadline_latency_{t}.json", "--delay-tokens", "0,16", "--repeats", "2"] if t == "130m" else None,
               timing=True, quiet_file=f"{{OUT}}/deadline_latency_{t}.host.json", wrap_timed=f"{{OUT}}/deadline_latency_{t}.host.json", phase=ph))
    A(Step("online_runtime_validation", "mechanism/stage14_online_runtime_validation.py", ["--output", "{OUT}/online_runtime_validation.json"],
           ["{OUT}/online_runtime_validation.json"], ["--output", "{OUT}/online_runtime_validation.json", "--examples", "2"],
           timing=True, quiet_file="{OUT}/online_runtime_validation.host.json", wrap_timed="{OUT}/online_runtime_validation.host.json", phase=ph))

    # ---- 2. exactness and multi-item scaling ------------------------------------------------------------------------
    ph = "2-exactness"
    for t in scales:
        extra = ["--k2-steps", "48"] if t == "1p4b" else []
        A(Step(f"snapshot_batch_{t}", "mechanism/snapshot_and_batch_tests.py",
               ["--model", S(t), "--cases", "8", "--output", f"{{OUT}}/snapshot_batch_{t}.json"] + extra, [f"{{OUT}}/snapshot_batch_{t}.json"],
               ["--model", S(t), "--cases", "2", "--window", "64", "--deltas", "0,32", "--k2-steps", "8", "--output", f"{{OUT}}/snapshot_batch_{t}.json"] if t in ("130m", "1p4b") else None,
               timing=True, quiet_file=f"{{OUT}}/snapshot_batch_{t}.json", phase=ph))
    for t in scales:
        A(Step(f"multi_item_{t}", "mechanism/multi_item_scaling.py", ["--model", S(t), "--output", f"{{OUT}}/multi_item_scaling_{t}.json"],
               [f"{{OUT}}/multi_item_scaling_{t}.json"],
               ["--model", S(t), "--output", f"{{OUT}}/multi_item_scaling_{t}.json"] if t == "130m" else None, phase=ph))

    A(Step("snapshot_batch_790m", "mechanism/snapshot_and_batch_tests.py", ["--model", S("790m"), "--cases", "8", "--output", "{OUT}/snapshot_batch_790m.json"],
           ["{OUT}/snapshot_batch_790m.json"], None, timing=True, quiet_file="{OUT}/snapshot_batch_790m.json", phase=ph))
    A(Step("multi_item_790m", "mechanism/multi_item_scaling.py", ["--model", S("790m"), "--output", "{OUT}/multi_item_scaling_790m.json"],
           ["{OUT}/multi_item_scaling_790m.json"], None, phase=ph))

    # ---- 3. scheduler simulations (CPU) ---------------------------------------------------------------------------
    ph = "3-simulation"
    for t in scales:
        A(Step(f"sim_fused_{t}", "scheduling/scheduler_sim.py",
               ["--profile", f"{{OUT}}/cost_profile_fused_{t}.json", "--tau-ms", "20,10,50", "--primary-tau", "20", "--output", f"{{OUT}}/scheduler_sim_fused_{t}.json"],
               [f"{{OUT}}/scheduler_sim_fused_{t}.json"],
               ["--profile", f"{{OUT}}/cost_profile_fused_{t}.json", "--tau-ms", "20", "--concurrency", "8", "--aligned", "1.0", "--seeds", "1",
                "--output", f"{{OUT}}/scheduler_sim_fused_{t}.json"] if t == "130m" else None, phase=ph, analysis=True))
    for t in scales:
        tau = "50" if t == "1p4b" else "20"
        A(Step(f"sim_padded_{t}", "scheduling/scheduler_sim.py",
               ["--profile", f"{{OUT}}/cost_profile_fused_{t}.json", "--tau-ms", tau, "--padded", "--seeds", "5", "--concurrency", HARNESS_N[t],
                "--aligned", "0.0,0.5,1.0", "--output", f"{{OUT}}/scheduler_sim_padded_{t}.json"],
               [f"{{OUT}}/scheduler_sim_padded_{t}.json"], None, phase=ph, analysis=True))
    A(Step("sim_fused_130m_retire_loop", "scheduling/scheduler_sim.py",
           ["--profile", "{OUT}/cost_profile_fused_130m.json", "--tau-ms", "20", "--retire-kind", "zero_loop", "--output", "{OUT}/scheduler_sim_fused_130m_retire_loop.json"],
           ["{OUT}/scheduler_sim_fused_130m_retire_loop.json"], None, phase=ph, analysis=True))
    for t, tau in (("130m", "150"), ("370m", "250"), ("1p4b", "500")):
        A(Step(f"sim_hf_{t}", "scheduling/scheduler_sim.py",
               ["--profile", f"{{OUT}}/cost_profile_hf_{t}.json", "--tau-ms", tau, "--output", f"{{OUT}}/scheduler_sim_hf_{t}.json"],
               [f"{{OUT}}/scheduler_sim_hf_{t}.json"], None, phase=ph, analysis=True))

    # ---- 4. further protocol seeds, all scales and corpora --------------------------------------------------------
    ph = "4-protocol-seeds"
    for s in range(20261004, 20261009):
        for t in scales:
            bs = [] if t == "130m" else ["--batch-size", "4"]
            suf = "" if t == "130m" else f"_{t}"
            A(Step(f"protocol_tab_{t}_{s}", "audit/stage10_shadow_revocation_tab.py",
                   ["--model", S(t), "--seed", str(s), "--output", f"{{OUT}}/protocol_tab{suf}_{s}.json"] + bs, [f"{{OUT}}/protocol_tab{suf}_{s}.json"],
                   ["--model", S(t), "--seed", str(s), "--attacker-per-class", "4", "--test-per-class", "2", "--output", f"{{OUT}}/protocol_tab{suf}_{s}.json"] + bs
                   if (t == "130m" and s == 20261004) else None,
                   timing=True, quiet_file=f"{{OUT}}/protocol_tab{suf}_{s}.host.json", wrap_timed=f"{{OUT}}/protocol_tab{suf}_{s}.host.json", phase=ph))
            A(Step(f"protocol_enron_{t}_{s}", "audit/stage11_shadow_revocation_enron.py",
                   ["--model", S(t), "--surrogate-count", "1", "--seed", str(s), "--output", f"{{OUT}}/protocol_enron{suf}_{s}.json"] + bs,
                   [f"{{OUT}}/protocol_enron{suf}_{s}.json"],
                   ["--model", S(t), "--surrogate-count", "1", "--seed", str(s), "--attacker-per-class", "4", "--test-per-class", "2",
                    "--output", f"{{OUT}}/protocol_enron{suf}_{s}.json"] + bs if (t == "130m" and s == 20261004) else None,
                   timing=True, quiet_file=f"{{OUT}}/protocol_enron{suf}_{s}.host.json", wrap_timed=f"{{OUT}}/protocol_enron{suf}_{s}.host.json", phase=ph))
    for s in range(20261001, 20261009):
        A(Step(f"protocol_tab_790m_{s}", "audit/stage10_shadow_revocation_tab.py",
               ["--model", S("790m"), "--batch-size", "4", "--seed", str(s), "--output", f"{{OUT}}/protocol_tab_790m_{s}.json"], [f"{{OUT}}/protocol_tab_790m_{s}.json"], None,
               timing=True, quiet_file=f"{{OUT}}/protocol_tab_790m_{s}.host.json", wrap_timed=f"{{OUT}}/protocol_tab_790m_{s}.host.json", phase=ph))
        A(Step(f"protocol_enron_790m_{s}", "audit/stage11_shadow_revocation_enron.py",
               ["--model", S("790m"), "--batch-size", "4", "--surrogate-count", "1", "--seed", str(s), "--output", f"{{OUT}}/protocol_enron_790m_{s}.json"],
               [f"{{OUT}}/protocol_enron_790m_{s}.json"], None,
               timing=True, quiet_file=f"{{OUT}}/protocol_enron_790m_{s}.host.json", wrap_timed=f"{{OUT}}/protocol_enron_790m_{s}.host.json", phase=ph))
    A(Step("equivalence_stats", "audit/equivalence_stats.py", [], ["{OUT}/equivalence_stats.json"], [], phase=ph, analysis=True, always=True))

    # ---- 5. attackers and controls --------------------------------------------------------------------------------
    ph = "5-attackers"
    seeds8 = range(20261001, 20261009)
    for s in seeds8:
        for t in ("130m", "370m"):
            bs = [] if t == "130m" else ["--batch-size", "4"]
            A(Step(f"audit_controls_{t}_{s}", "audit/audit_controls.py", ["--model", S(t), "--seed", str(s), "--output", f"{{OUT}}/audit_controls_{t}_{s}.json"] + bs,
                   [f"{{OUT}}/audit_controls_{t}_{s}.json"],
                   ["--model", S(t), "--seed", str(s), "--attacker-per-class", "4", "--test-per-class", "2", "--permutations", "5",
                    "--output", f"{{OUT}}/audit_controls_{t}_{s}.json"] if (t == "130m" and s == 20261001) else None, phase=ph))
    for s in seeds8:
        for t in ("130m", "370m"):
            bs = [] if t == "130m" else ["--batch-size", "4"]
            A(Step(f"audit_controls_enron_{t}_{s}", "audit/audit_controls.py",
                   ["--corpus", "enron", "--context-characters", "220", "--model", S(t), "--seed", str(s), "--output", f"{{OUT}}/audit_controls_enron_{t}_{s}.json"] + bs,
                   [f"{{OUT}}/audit_controls_enron_{t}_{s}.json"],
                   ["--corpus", "enron", "--context-characters", "220", "--model", S(t), "--seed", str(s), "--attacker-per-class", "4", "--test-per-class", "2",
                    "--permutations", "5", "--output", f"{{OUT}}/audit_controls_enron_{t}_{s}.json"] if (t == "130m" and s == 20261001) else None, phase=ph))
    for s in range(20261001, 20261004):
        for t in ("130m", "370m"):
            bs = [] if t == "130m" else ["--batch-size", "4"]
            A(Step(f"stage19_strong_{t}_{s}", "audit/stage19_nonlinear_attacker_tab.py",
                   ["--model", S(t), "--seed", str(s), "--attacker-per-class", "48", "--defender-per-class", "24", "--output", f"{{OUT}}/stage19_strong_{t}_{s}.json"] + bs,
                   [f"{{OUT}}/stage19_strong_{t}_{s}.json"],
                   ["--model", S(t), "--seed", str(s), "--attacker-per-class", "4", "--defender-per-class", "4", "--test-per-class", "2",
                    "--output", f"{{OUT}}/stage19_strong_{t}_{s}.json"] if (t == "130m" and s == 20261001) else None, phase=ph))
    for s in seeds8:
        A(Step(f"mamba2_audit_{s}", "audit/mamba2_audit.py", ["--seed", str(s), "--output", f"{{OUT}}/mamba2_audit_{s}.json"], [f"{{OUT}}/mamba2_audit_{s}.json"],
               ["--seed", str(s), "--attacker-per-class", "4", "--test-per-class", "2", "--output", f"{{OUT}}/mamba2_audit_{s}.json"] if s == 20261001 else None, phase=ph))

    for tag in ("370m", "780m"):
        A(Step(f"mamba2_mechanism_{tag}", "mechanism/stage24_mamba2_portability_check.py", ["--model", MAMBA2[tag], "--output", f"{{OUT}}/stage24_mamba2_portability_check_{tag}.json"],
               [f"{{OUT}}/stage24_mamba2_portability_check_{tag}.json"], None, phase=ph))
        A(Step(f"mamba2_lookahead_{tag}", "mechanism/stage25_mamba2_lookahead_sweep.py", ["--model", MAMBA2[tag], "--deltas", "0,32,64,128", "--output", f"{{OUT}}/stage25_mamba2_lookahead_sweep_{tag}.json"],
               [f"{{OUT}}/stage25_mamba2_lookahead_sweep_{tag}.json"], None, phase=ph))
    for tag in ("370m", "780m"):
        for s in seeds8:
            A(Step(f"mamba2_audit_{tag}_{s}", "audit/mamba2_audit.py", ["--model", MAMBA2[tag], "--seed", str(s), "--output", f"{{OUT}}/mamba2_audit_{tag}_{s}.json"],
                   [f"{{OUT}}/mamba2_audit_{tag}_{s}.json"], None, phase=ph))

    ph = "6-attackers-1p4b"
    for s in seeds8:
        A(Step(f"audit_controls_1p4b_{s}", "audit/audit_controls.py", ["--model", S("1p4b"), "--batch-size", "4", "--seed", str(s), "--output", f"{{OUT}}/audit_controls_1p4b_{s}.json"],
               [f"{{OUT}}/audit_controls_1p4b_{s}.json"],
               ["--model", S("1p4b"), "--batch-size", "4", "--seed", str(s), "--attacker-per-class", "4", "--test-per-class", "2", "--permutations", "5",
                "--output", f"{{OUT}}/audit_controls_1p4b_{s}.json"] if s == 20261001 else None, phase=ph))
    for s in seeds8:
        A(Step(f"audit_controls_enron_1p4b_{s}", "audit/audit_controls.py",
               ["--corpus", "enron", "--context-characters", "220", "--model", S("1p4b"), "--batch-size", "4", "--seed", str(s), "--output", f"{{OUT}}/audit_controls_enron_1p4b_{s}.json"],
               [f"{{OUT}}/audit_controls_enron_1p4b_{s}.json"], None, phase=ph))
    for s in range(20261001, 20261004):
        A(Step(f"stage19_strong_1p4b_{s}", "audit/stage19_nonlinear_attacker_tab.py",
               ["--model", S("1p4b"), "--batch-size", "4", "--seed", str(s), "--attacker-per-class", "48", "--defender-per-class", "24", "--output", f"{{OUT}}/stage19_strong_1p4b_{s}.json"],
               [f"{{OUT}}/stage19_strong_1p4b_{s}.json"], None, phase=ph))

    # ---- 7. distance to the counterfactual ------------------------------------------------------------------------
    ph = "7-counterfactual-distance"
    for s in range(20261001, 20261004):
        for t, extra, tagname in (("130m", [], "130m"), ("130m", ["--full-layer"], "130m_fulllayer"), ("370m", [], "370m"), ("1p4b", [], "1p4b")):
            A(Step(f"counterfactual_distance_{tagname}_{s}", "audit/counterfactual_distance.py",
                   ["--model", S(t), "--seed", str(s), "--output", f"{{OUT}}/counterfactual_distance_{tagname}_{s}.json"] + extra,
                   [f"{{OUT}}/counterfactual_distance_{tagname}_{s}.json"],
                   ["--model", S(t), "--seed", str(s), "--defender-per-class", "4", "--test-per-class", "1", "--samples", "2", "--gen-tokens", "4",
                    "--output", f"{{OUT}}/counterfactual_distance_{tagname}_{s}.json"] if (tagname == "130m" and s == 20261001) else None,
                   timing=False, phase=ph))

    # ---- 8. serving harness and its validation ----------------------------------------------------------------------
    ph = "8-serving-harness"
    harness = {t: ["--concurrency", HARNESS_N[t], "--tau-ms", "50" if t == "1p4b" else "20"] for t in scales}
    for t in scales:
        A(Step(f"serving_harness_{t}", "scheduling/serving_harness.py",
               ["--model", S(t), "--tag", t, "--aligned", "0.0,0.5,1.0", "--seeds", "5", "--output", f"{{OUT}}/serving_harness_{t}.json"] + harness[t],
               [f"{{OUT}}/serving_harness_{t}.json"],
               ["--model", S(t), "--tag", t, "--aligned", "1.0", "--seeds", "1", "--ticks", "200", "--warm", "50", "--concurrency", "8",
                "--policies", "retro,full,la25", "--check-sessions", "2", "--output", f"{{OUT}}/serving_harness_{t}.json"] if t == "130m" else None,
               timing=True, quiet_file=f"{{OUT}}/serving_harness_{t}.json", phase=ph))
    for t in scales:
        A(Step(f"sim_vs_harness_{t}", "scheduling/sim_vs_harness.py",
               ["--sim", f"{{OUT}}/scheduler_sim_padded_{t}.json", "--harness", f"{{OUT}}/serving_harness_{t}.json", "--tau-ms", "50" if t == "1p4b" else "20", "--output", f"{{OUT}}/sim_vs_harness_{t}.json"],
               [f"{{OUT}}/sim_vs_harness_{t}.json"],
               ["--sim", f"{{OUT}}/scheduler_sim_padded_{t}.json", "--harness", f"{{OUT}}/serving_harness_{t}.json", "--tau-ms", "50" if t == "1p4b" else "20", "--output", f"{{OUT}}/sim_vs_harness_{t}.json"] if t == "130m" else None,
               phase=ph, analysis=True))

    # selective-scan kernel: ours versus the official mamba_ssm kernel (WSL), same GPU, run one after the other
    A(Step("scan_bench_ours", "cost/scan_bench_ours.py", ["--output", "{OUT}/scan_bench_ours.json"], ["{OUT}/scan_bench_ours.json"],
           ["--lengths", "1,16", "--repeats", "3", "--output", "{OUT}/scan_bench_ours.json"],
           timing=True, quiet_file="{OUT}/scan_bench_ours.json", phase=ph))
    A(Step("scan_bench_official", "cost/scan_bench_official.py", ["--output", "{OUT}/scan_bench_official.json"], ["{OUT}/scan_bench_official.json"],
           ["--lengths", "1,16", "--repeats", "3", "--output", "{OUT}/scan_bench_official.json"],
           timing=True, quiet_file="{OUT}/scan_bench_official.json", phase=ph))
    A(Step("scan_bench_compare", "cost/scan_bench_compare.py", [], ["{OUT}/scan_bench.json"], None, phase=ph, analysis=True))

    # ---- 9. tables and numbers ---------------------------------------------------------------------------------------
    ph = "9-reporting"
    for t in scales:
        A(Step(f"comparison_{t}", "reporting/comparison_table.py", ["--tag", t], [f"{{OUT}}/comparison_{t}.json"], ["--tag", t] if t == "130m" else None, phase=ph, analysis=True, always=True))
    A(Step("paper_numbers", "reporting/make_paper_numbers.py", [], ["paper/numbers.tex"], [], phase=ph, analysis=True, always=True))
    return st


def out_dir(smoke: bool) -> str:
    return str(SMOKE_DIR.relative_to(ROOT)).replace("\\", "/") if smoke else "results"


def sub(items, smoke):
    o = out_dir(smoke)
    return [(o + "/numbers.tex" if (smoke and x == "paper/numbers.tex") else x.replace("{OUT}", o)) for x in items]


def env_for(smoke: bool) -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + [str(EXP / d) for d in SUBDIRS] + ([e["SV_EXTRA_PYTHONPATH"]] if e.get("SV_EXTRA_PYTHONPATH") else []))
    e.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
    e["HF_HUB_OFFLINE"] = "1"
    e["TRANSFORMERS_OFFLINE"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    e["TF_ENABLE_ONEDNN_OPTS"] = "0"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        e[k] = "1"
    tcc = Path(site.getusersitepackages()) / "triton" / "runtime" / "tcc" / "tcc.exe"
    if tcc.exists():
        e.setdefault("CC", str(tcc))
    if smoke:
        e["SV_RESULTS"] = str(SMOKE_DIR)
        e["SV_OUT"] = str(SMOKE_DIR / "numbers.tex")
    return e


def command(step: Step, smoke: bool) -> list:
    args = sub(step.smoke_args if smoke else step.args, smoke)
    base = [PY, str(EXP / step.script)] + args
    if step.wrap_timed:
        sidecar = sub([step.wrap_timed], smoke)[0]
        base = [PY, str(EXP / "cost/run_timed.py"), sidecar, "--"] + base
    return base


def is_done(step: Step, smoke: bool) -> tuple[bool, str]:
    outs = [ROOT / p for p in sub(step.outputs, smoke)]
    if not all(p.exists() for p in outs):
        return False, "pending"
    if step.timing and step.quiet_file:
        qf = ROOT / sub([step.quiet_file], smoke)[0]
        if not qf.exists():
            return False, "pending"
        rec = load_json(qf)
        quiet = rec.get("quiet") if "quiet" in rec else is_quiet(rec.get("host_load"))
        if not quiet:
            return False, "rerun (host was loaded)"
    return True, "done"


_child: subprocess.Popen | None = None


def _stop(signum=None, frame=None):
    raise KeyboardInterrupt


def run_step(step: Step, smoke: bool, wait_quiet: bool) -> int:
    global _child
    if step.timing and wait_quiet and not smoke:
        print(f"  waiting for a quiet host ...", flush=True)
        wait_until_quiet()
    cmd = command(step, smoke)
    log = ROOT / "logs" / "pipeline" / f"{step.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = 0x00004000 if os.name == "nt" else 0           # BELOW_NORMAL_PRIORITY_CLASS
    t0 = time.time()
    with log.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        _child = subprocess.Popen(cmd, cwd=ROOT, env=env_for(smoke), creationflags=flags, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        skip = ("warn", "absl", "onednn", "fast path", "it/s]", "s/it]", "Loading checkpoint")
        try:
            for line in _child.stdout:
                lf.write(line)
                if not any(s in line.lower() for s in skip):
                    print("    " + line.rstrip(), flush=True)
            code = _child.wait()
        except KeyboardInterrupt:
            _child.terminate()
            try:
                _child.wait(15)
            except subprocess.TimeoutExpired:
                _child.kill()
            lf.write("=== interrupted\n")
            raise
    print(f"  finished in {time.time() - t0:.0f}s (exit {code})", flush=True)
    return code


def _keep_awake():
    """Ask Windows not to sleep while experiments run (released automatically when the process exits)."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)      # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["list", "status", "run"])
    ap.add_argument("--only", default=None, help="regex on step names")
    ap.add_argument("--phase", default=None, help="regex on phase names")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--no-quiet-wait", action="store_true")
    ap.add_argument("--redo", action="store_true", help="ignore existing outputs")
    ap.add_argument("--reverse", action="store_true", help="run the steps in reverse order (second machine meets the first in the middle)")
    args = ap.parse_args()
    steps = build_steps()
    if args.smoke:
        steps = [s for s in steps if s.smoke_args is not None]
    if args.only:
        steps = [s for s in steps if re.search(args.only, s.name)]
    if args.phase:
        steps = [s for s in steps if re.search(args.phase, s.phase)]
    if os.environ.get("SV_SKIP_PHASE"):
        steps = [s for s in steps if not re.search(os.environ["SV_SKIP_PHASE"], s.phase)]      # phases assigned to another machine
    if args.reverse:
        steps = steps[::-1]
    if args.action in ("list", "status"):
        counts = {}
        for s in steps:
            ok, why = is_done(s, args.smoke)
            counts[why] = counts.get(why, 0) + 1
            if args.action == "list":
                print(f"{s.phase:26s} {s.name:42s} {why}")
        print({k: v for k, v in counts.items()}, f"total {len(steps)}")
        return
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _stop)
    if args.smoke:
        SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        for p in (ROOT / "results").glob("confirm_*.json"):
            (SMOKE_DIR / p.name).write_bytes(p.read_bytes())
        for name in ("stage6_tab_live_defense_utility.json", "stage17_fixed_state_memory_scaling.json"):
            if (ROOT / "results" / name).exists():
                (SMOKE_DIR / name).write_bytes((ROOT / "results" / name).read_bytes())
    failed = []
    _keep_awake()
    try:
        for i, s in enumerate(steps, 1):
            ok, why = is_done(s, args.smoke)
            if ok and not s.always and not args.redo:
                print(f"[{i}/{len(steps)}] {s.name}: done, skipping", flush=True)
                continue
            print(f"[{i}/{len(steps)}] {s.name}  ({s.phase})", flush=True)
            code = run_step(s, args.smoke, not args.no_quiet_wait)
            ok2 = code == 0 and all((ROOT / p).exists() for p in sub(s.outputs, args.smoke))
            if not ok2:
                failed.append(s.name)
                print(f"  FAILED: {s.name} (exit {code}); see logs/pipeline/{s.name}.log", flush=True)
                if args.stop_on_error:
                    break
        # one automatic retry of every step that failed (transient errors, out-of-memory after another process released memory, ...)
        retry, failed = failed, []
        for name in retry:
            s = next(x for x in steps if x.name == name)
            print(f"retrying {name}", flush=True)
            code = run_step(s, args.smoke, not args.no_quiet_wait)
            if not (code == 0 and all((ROOT / p).exists() for p in sub(s.outputs, args.smoke))):
                failed.append(name)
    except KeyboardInterrupt:
        print("\ninterrupted: rerun the same command to continue", flush=True)
        sys.exit(130)
    print("\nfailed steps:" if failed else "\nall requested steps completed", ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

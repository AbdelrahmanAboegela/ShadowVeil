"""Results figure: measured deadline-path cost vs window (fused path, three scales) and the simulated
deadline-miss rate vs concurrent sessions at each scale. Vector PDF, IEEE sizing.

Palette: blue/orange/aqua/violet (validated colour-vision separation, adjacent dE >= 9) plus distinct
markers and line styles as the secondary encoding (aqua is below 3:1 contrast on white, so lines are
direct-labelled).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
OUT = Path("paper/figures")
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#6b6b6b"

mpl.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2, "lines.markersize": 3.6,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5, "axes.axisbelow": True,
})

SCALES = [("130m", "130M", BLUE, "o"), ("370m", "370M", ORANGE, "s"), ("1p4b", "1.4B", AQUA, "^")]
POLICIES = [("retro", r"retrospective ($\delta{=}0$)", ORANGE, "s", "-"),
            ("lastar", r"lookahead $\delta^\star$ (burst time / token interval)", AQUA, "D", "-"),
            ("la16", r"lookahead $\delta{=}16$", VIOLET, "v", "-"),
            ("la50", r"lookahead $\delta{=}L/2$", VIOLET, "D", "--"),
            ("la25", r"lookahead $\delta{=}L/4$", AQUA, "^", "--"),
            ("full", r"full ($\delta{=}L$)", BLUE, "o", "-")]


def panel_cost(ax):
    retire = {r["model"].split("/")[-1]: r for r in json.loads((R / "retirement_latency.json").read_text())["rows"]}
    rets = []
    for tag, label, color, marker in SCALES:
        d = json.loads((R / f"fused_sweep_{tag}.json").read_text())
        pts = sorted((int(k), v["median"]) for k, v in d["chunk_ms_by_tokens"].items())
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, marker=marker, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(3, 0), textcoords="offset points", color=color, va="center", fontsize=6.5)
        name = {"130m": "mamba-130m-hf", "370m": "mamba-370m-hf", "1p4b": "mamba-1.4b-hf"}[tag]
        if name in retire:
            rets.append(retire[name]["zero_loop"]["median_ms"])
    if rets:
        ax.axhspan(min(rets), max(rets), color=GREY, alpha=0.35, lw=0)
        ax.annotate("ShadowVeil at $T$: select + zeroize", (700, max(rets)), xytext=(0, -2), textcoords="offset points",
                    color="#333333", fontsize=6, va="top", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("window length $L$ (tokens)")
    ax.set_ylabel("revocation work at $T$ (ms)")
    ax.set_title("(a) replay at the deadline grows with $L$", loc="left")
    ax.set_xlim(8, 8192 * 2.4)


def panel_miss(ax, tag, label, tau):
    """Deadline-miss rate vs concurrent sessions: half of the expiries aligned (solid) and all aligned (dashed)."""
    rows = json.loads((R / f"scheduler_sim_fused_{tag}.json").read_text())["rows"]
    rows = [r for r in rows if r["cost_model"] == "measured_rows" and r["tau_ms"] == tau]
    for pol, name, color, marker in (("retro", "retro", ORANGE, "s"), ("la16", "la16", VIOLET, "v"), ("full", "full", BLUE, "o")):
        for al, ls in ((0.5, "-"), (1.0, "--")):
            pts = sorted((r["concurrency"], 100 * r["slo_miss_rate"]) for r in rows if r["policy"] == pol and r["aligned_fraction"] == al)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=color, marker=marker, ls=ls, label=f"{name}" if al == 0.5 else None)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("concurrent sessions")
    ax.set_title(f"({'bcd'[('130m', '370m', '1p4b').index(tag)]}) {label}, tick {int(tau)} ms", loc="left")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    panel_cost(axes[0])
    for ax, (tag, label, tau) in zip(axes[1:], (("130m", "130M", 20.0), ("370m", "370M", 20.0), ("1p4b", "1.4B", 50.0))):
        panel_miss(ax, tag, label, tau)
    axes[1].set_ylabel("deadline misses (%)")
    axes[3].legend(frameon=False, loc="center left", handlelength=1.8, borderaxespad=0.1, title="solid: half aligned\ndashed: all aligned", title_fontsize=5.5)
    fig.savefig(OUT / "fig_cost_sched.pdf")
    fig.savefig(OUT / "fig_cost_sched.png", dpi=220)
    print("saved", OUT / "fig_cost_sched.pdf")


if __name__ == "__main__":
    main()

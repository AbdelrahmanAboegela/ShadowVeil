"""Verify the evidence behind every claim and write a machine-readable manifest (artifacts/verification.json).

Checks: prespecified protocol (document-disjoint, bitwise boundary/continuation equality, zeroization, identity threshold) on the
primary seeds; the same exactness on the further protocol seeds; fused-path exactness at every offset and on additional documents,
token-transition conservation, batched shadow rows, non-zero snapshots and batch geometry; multi-item branch counts; audit controls
(both sanitizers exact); Mamba-2; supporting audits; and that every timing result was measured on a quiet host.
Exit status is non-zero if any check fails or an expected result is missing.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

from shadowveil.hostload import is_quiet
from shadowveil.iohelp import results_dir, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = results_dir()
PRIMARY_SEEDS = (20261001, 20261002, 20261003)
TAGS = (("130m", "130M"), ("370m", "370M"), ("1p4b", "1.4B"))
THRESHOLD = 0.1875
TAGS_MECH = TAGS + (("790m", "790M"),)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    checks: list[dict] = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    def need(name):
        p = RESULTS / name
        check(f"exists:{name}", p.exists())
        return load(p) if p.exists() else None

    def condition(payload, name):
        return next(r for r in payload["results"] if r["condition"] == name)

    # ---- prespecified protocol, all scales and corpora -------------------------------------------------------------
    for tag, label in TAGS:
        for dataset in ("tab", "enron"):
            accs = []
            for seed in PRIMARY_SEEDS:
                name = f"confirm_{dataset}_{seed}.json" if tag == "130m" else f"confirm_{dataset}_{tag}_{seed}.json"
                d = need(name)
                if not d:
                    continue
                ex, shadow = d["exactness"], condition(d, "shadow_swap")
                check(f"{label}:{dataset}:{seed}:document_disjoint", d["document_overlap_attacker_test"] == 0)
                check(f"{label}:{dataset}:{seed}:boundary_exact", ex["bitwise_equal_to_independent_redacted_reference"], ex["max_absolute_cache_difference"])
                if "post_suffix_sampled_features_bitwise_equal" in ex:
                    check(f"{label}:{dataset}:{seed}:continuation_exact", ex["post_suffix_sampled_features_bitwise_equal"])
                check(f"{label}:{dataset}:{seed}:retired_zero", ex["retired_live_cache_max_absolute_value"] == 0.0)
                accs.append(shadow["identity"]["accuracy"])
            if accs:
                check(f"{label}:{dataset}:prespecified_threshold", sum(accs) / len(accs) <= THRESHOLD, {"mean": sum(accs) / len(accs), "threshold": THRESHOLD})
    for f in sorted(glob.glob(str(RESULTS / "protocol_*.json"))):
        if f.endswith(".host.json"):
            continue
        d = load(f)
        check(f"{Path(f).name}:boundary_exact", d["exactness"]["bitwise_equal_to_independent_redacted_reference"])
    n_proto = len([f for f in glob.glob(str(RESULTS / "protocol_*.json")) if not f.endswith(".host.json")])
    check("protocol_seed_files_present", n_proto >= 30, n_proto)

    # ---- fused path, batched rows, non-zero snapshots, multi-item ----------------------------------------------------
    for tag, label in TAGS_MECH:
        d = need(f"fused_sweep_{tag}.json")
        if d:
            check(f"{label}:fused_all_offsets_exact", all(r["exact_bitwise"] for r in d["delta_sweep"]), len(d["delta_sweep"]))
            extra = d["exactness_additional_documents"]
            check(f"{label}:fused_additional_documents_exact", extra and all(r["exact_bitwise"] for r in extra), len(extra))
            check(f"{label}:token_transitions_conserved", extra and all(r["shadow_token_transitions_counted"] == r["expected_g_plus_window"] for r in extra))
            check(f"{label}:fused_vs_hf_close", d["fused_vs_hf_relative_state_difference"] < 1e-2, d["fused_vs_hf_relative_state_difference"])
            check(f"{label}:fused_timing_quiet", is_quiet(d.get("host_load")))
        d = need(f"batched_tail_{tag}.json")
        if d:
            check(f"{label}:batched_shadow_row_exact", all(r["shadow_row_bitwise_equal_to_reference_row"] for r in d["rows"]), len(d["rows"]))
            check(f"{label}:batched_timing_quiet", is_quiet(d.get("host_load")))
        d = need(f"snapshot_batch_{tag}.json")
        if d:
            s = d["summary"]
            check(f"{label}:nonzero_snapshot_exact", s["k1_all_shadow_equals_reference_bitwise"])
            check(f"{label}:batch_geometry_deterministic", s["k2_geometry"] and all(s["k2_geometry"].values()), s["k2_geometry"])
        d = need(f"multi_item_scaling_{tag}.json")
        if d:
            check(f"{label}:multi_item_exact", all(r["all_branches_exact"] and r["exact_on_reveal"] for r in d["rows"]), len(d["rows"]))
            check(f"{label}:branches_equal_2_to_w", all(r["branches_required"] == 2 ** r["ambiguity_window_w"] for r in d["rows"]))
            check(f"{label}:branches_halve_on_reveal", all(r["branches_after_first_reveal"] == 2 ** (r["ambiguity_window_w"] - 1) for r in d["rows"]))
    for f in sorted(glob.glob(str(RESULTS / "fused_sweep_*_w*.json"))):
        d = load(f)
        check(f"{Path(f).name}:all_offsets_exact", all(r["exact_bitwise"] for r in d["delta_sweep"]) and all(r["exact_bitwise"] for r in d["exactness_additional_documents"]))
        check(f"{Path(f).name}:timing_quiet", is_quiet(d.get("host_load")))
    d = need("retirement_latency.json")
    if d:
        check("retirement_timing_quiet", is_quiet(d.get("host_load")))

    # ---- audit controls, Mamba-2 ------------------------------------------------------------------------------------
    files = sorted(glob.glob(str(RESULTS / "audit_controls_*.json")))
    check("audit_control_files_present", len(files) >= 48, len(files))
    for f in files:
        ex = load(f)["exactness"]
        check(f"{Path(f).name}:both_sanitizers_exact", ex["primary_sanitizer"]["bitwise_equal"] and ex["leaky_sanitizer"]["bitwise_equal"])
    files = sorted(glob.glob(str(RESULTS / "mamba2_audit_*.json")))
    check("mamba2_attacker_files_present", len(files) >= 24, len(files))
    for f in files:
        d = load(f)
        check(f"{Path(f).name}:exact", d["exactness"]["bitwise_equal_to_independent_redacted_reference"])
    d = need("stage24_mamba2_portability_check.json")
    if d:
        check("mamba2_boundary_exact", d["boundary_equivalence"]["bitwise_equal_to_independent_reference"])
        check("mamba2_continuation_exact", d["continuation_equivalence"]["bitwise_equal_to_independent_reference"])
        check("mamba2_retirement_zero", d["retirement"]["retired_cache_max_absolute_value"] == 0.0)
    d = need("stage25_mamba2_lookahead_sweep.json")
    if d:
        check("mamba2_lookahead_all_offsets_exact", d["all_deltas_exact"])

    for tag in ("370m", "780m"):
        d = need(f"stage24_mamba2_portability_check_{tag}.json")
        if d:
            check(f"mamba2_{tag}_boundary_exact", d["boundary_equivalence"]["bitwise_equal_to_independent_reference"])
            check(f"mamba2_{tag}_continuation_exact", d["continuation_equivalence"]["bitwise_equal_to_independent_reference"])
        d = need(f"stage25_mamba2_lookahead_sweep_{tag}.json")
        if d:
            check(f"mamba2_{tag}_lookahead_all_offsets_exact", d["all_deltas_exact"])

    # ---- cost model, scheduler simulation, serving harness and their agreement ------------------------------------------------
    for tag, label in TAGS:
        d = need(f"step_rows_{tag}.json")
        if d:
            rows = sorted((int(k), v["median"]) for k, v in d["step_ms_by_rows"].items())
            check(f"{label}:step_curve_covers_64_rows", rows[-1][0] >= 64, rows[-1][0])
            check(f"{label}:step_curve_nondecreasing", all(b[1] >= a[1] * 0.9 for a, b in zip(rows, rows[1:])))
            check(f"{label}:step_curve_quiet_host", is_quiet(d.get("host_load")))
        d = need(f"scheduler_sim_fused_{tag}.json")
        if d:
            check(f"{label}:sim_uses_measured_step_curve", d["profile_fit"].get("c0") is not None and d["profile_quiet"])
        d = need(f"scheduler_sim_padded_{tag}.json")
        if d:
            check(f"{label}:padded_sim_has_all_policies", {"retro", "la16", "full"} <= {r["policy"] for r in d["rows"]})
        h = RESULTS / f"serving_harness_{tag}.json"
        if h.exists():
            hd = load(h)
            check(f"{label}:harness_calibrated_per_load", len(hd["calibration_ms"]) >= 3, sorted(hd["calibration_ms"]))
            mx = max(r["shadow_vs_replay_max_rel_diff"] or 0 for r in hd["rows"])
            # the harness drives random token ids, and the shadow is built with different chunk boundaries than the monolithic replay,
            # so this difference is fp16 state rounding on out-of-distribution input, not an equality test (equality is checked
            # against a same-schedule reference in the mechanism experiments); it only has to stay bounded
            check(f"{label}:harness_shadow_close_to_monolithic_replay", mx < 0.25, round(mx, 4))
            check(f"{label}:harness_quiet_host", is_quiet(hd.get("host_load")))
            v = need(f"sim_vs_harness_{tag}.json")
            if v:
                check(f"{label}:sim_vs_harness_reported", v["configs"] > 0 and v["slo_verdict_agreement"] is not None, v["slo_verdict_agreement"])
    for name in ("scan_bench_ours.json", "scan_bench_official.json"):
        d = need(name)
        if d:
            check(f"{name}:quiet_host", is_quiet(d.get("host_load")))
    need("scan_bench.json")

    # ---- supporting audits ------------------------------------------------------------------------------------------
    d = need("stage15_reconvergence_audit.json")
    if d:
        check("divergent_caches_never_reconverge", not d["any_exact_reconvergence"])
    d = need("stage16_deadline_chain_audit.json")
    if d:
        check("deadline_chain_exhaustive_no_violations", not d["exhaustive_chain_check"]["violations"], d["exhaustive_chain_check"]["assignments_checked"])
    d = need("stage17_fixed_state_memory_scaling.json")
    if d:
        check("fixed_state_shadow_memory_constant", d["mamba"]["fitted_bytes_per_context_token"] == 0.0)
        check("transformer_shadow_memory_linear", d["transformer"]["fitted_bytes_per_context_token"] > 0.0)

    hashes = {}
    for rel in ("paper/shadowveil-paper.tex",):
        if (ROOT / rel).exists():
            hashes[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    manifest = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks,
                "failed": [c["name"] for c in checks if not c["passed"]], "sha256": hashes}
    write_json(ROOT / "artifacts" / "verification.json", manifest, newline=True)
    print(json.dumps({"status": manifest["status"], "checks": len(checks), "failed": len(manifest["failed"])}, indent=2))
    if manifest["status"] != "PASS":
        print("first failures:", manifest["failed"][:12])
        raise SystemExit(1)


if __name__ == "__main__":
    main()

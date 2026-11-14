"""Join the two selective-scan benchmarks: latency ratio ours / official per scale and chunk length."""

from __future__ import annotations

from shadowveil.iohelp import results_dir, write_json

import json


def main():
    R = results_dir()
    ours = json.loads((R / "scan_bench_ours.json").read_text(encoding="utf-8"))
    off = json.loads((R / "scan_bench_official.json").read_text(encoding="utf-8"))
    o = {(r["scale"], r["tokens"]): r for r in off["rows"]}
    rows = []
    for r in ours["rows"]:
        k = (r["scale"], r["tokens"])
        if k in o:
            rows.append({"scale": r["scale"], "tokens": r["tokens"], "ours_ms": r["median_ms"], "official_ms": o[k]["median_ms"],
                         "ratio_ours_over_official": r["median_ms"] / o[k]["median_ms"]})
    write_json(R / "scan_bench.json", {"status": "scan_kernel_comparison", "gpu": ours["gpu"], "official": off["kernel"],
                                       "ours": ours["kernel"], "rows": rows})
    for r in rows:
        print(r["scale"], r["tokens"], "ours %.3f ms official %.3f ms ratio %.2f" % (r["ours_ms"], r["official_ms"], r["ratio_ours_over_official"]))


if __name__ == "__main__":
    main()

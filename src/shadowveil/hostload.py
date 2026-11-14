"""Host-load record for timing measurements.

Latency numbers are only meaningful on a quiet machine: CPU-launch-bound paths (eager PyTorch, per-tensor
zeroing, graph replay loops) slow down when other processes compete for cores. Every timing script stores a
before/after snapshot; paper numbers are generated only from runs whose snapshots are below QUIET_CPU_PERCENT.
"""

from __future__ import annotations

import time

import psutil

QUIET_CPU_PERCENT = 30.0


def snapshot(seconds: float = 2.0) -> dict:
    return {
        "cpu_percent": psutil.cpu_percent(interval=seconds),
        "logical_cores": psutil.cpu_count(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quiet_threshold": QUIET_CPU_PERCENT,
    }


def is_quiet(record: dict | None) -> bool:
    if not record:
        return False
    return record["before"]["cpu_percent"] <= QUIET_CPU_PERCENT and record["after"]["cpu_percent"] <= QUIET_CPU_PERCENT


def wait_until_quiet(poll_s: float = 30.0, samples: int = 3) -> None:
    """Block until `samples` consecutive 5-second CPU readings are at or below the threshold."""
    ok = 0
    while ok < samples:
        ok = ok + 1 if psutil.cpu_percent(interval=5.0) <= QUIET_CPU_PERCENT else 0
        if ok < samples:
            time.sleep(poll_s if ok == 0 else 1.0)

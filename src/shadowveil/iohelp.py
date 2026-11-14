"""Interrupt-safe result writing and path conventions.

Results are written to a temporary file in the destination directory and renamed into place, so an interrupted run
never leaves a truncated file that looks complete. The results folder can be redirected with SV_RESULTS (used by the
smoke tests) without touching real results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_text_atomic(path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def write_json(path, obj, newline: bool = False) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2) + ("\n" if newline else ""))


def results_dir() -> Path:
    return Path(os.environ.get("SV_RESULTS", "results"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

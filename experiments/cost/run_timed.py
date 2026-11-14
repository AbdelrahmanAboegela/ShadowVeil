"""Run a timing-sensitive command and record the host load around it in a sidecar file.

Usage: python experiments/run_timed.py <sidecar.json> -- <command...>
The sidecar holds {"command": [...], "host_load": {"before": ..., "after": ...}, "quiet": bool}; paper numbers taken from the
command's output are cited only when the sidecar is quiet.
"""

from shadowveil.iohelp import write_json, write_text_atomic
import json
import subprocess
import sys
from pathlib import Path

from shadowveil.hostload import is_quiet, snapshot

sep = sys.argv.index("--")
sidecar = Path(sys.argv[1])
cmd = sys.argv[sep + 1:]
before = snapshot()
code = subprocess.call(cmd)
after = snapshot(1.0)
record = {"command": cmd, "host_load": {"before": before, "after": after}}
record["quiet"] = is_quiet(record["host_load"])
sidecar.parent.mkdir(parents=True, exist_ok=True)
write_json(sidecar, record)
sys.exit(code)

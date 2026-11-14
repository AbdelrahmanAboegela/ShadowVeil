"""Run the whole pipeline unattended: restart it if it ends early, stop when every step is done.

    python tools/run_all.py

The pipeline skips finished steps, so each restart continues where the previous attempt stopped. Intended for overnight runs.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPE = [sys.executable, str(ROOT / "tools" / "pipeline.py")]


def pending() -> int:
    """Number of steps not done; -1 if the status could not be read (the caller then simply restarts the pipeline)."""
    for _ in range(5):
        out = subprocess.run(PIPE + ["status"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
        try:
            # last line looks like: {'done': 12, 'pending': 109} total 121
            return int(eval(out[-1].split(" total")[0])["pending"])
        except (IndexError, KeyError, SyntaxError, NameError):
            time.sleep(10)
    return -1


for attempt in range(1, 13):
    print(f"=== attempt {attempt}: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    code = subprocess.call(PIPE + ["run"], cwd=ROOT)
    left = pending()
    print(f"=== attempt {attempt} ended with exit {code}; steps not done: {left}", flush=True)
    if left == 0 or code == 130:
        break
    time.sleep(60)

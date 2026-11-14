"""Batch size a serving system pads its decode step to at a given load level.

A CUDA-graph decode step has a fixed number of rows. The serving harness and the scheduler simulation both pad the
batch to `slots_for(N)` rows (live sessions) and twice that when shadow rows share the step, where N is the mean number
of concurrent sessions, so that the two can be compared under the same cost model.
"""

import math


def slots_for(n_concurrent: int) -> int:
    """Smallest multiple of 8 that is at least 1.5 N + 8 (headroom for Poisson fluctuations of the session count)."""
    return int(8 * math.ceil((1.5 * n_concurrent + 8) / 8))

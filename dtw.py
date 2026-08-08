"""Minimal DTW compatibility layer for the legacy anomaly detector."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DTWResult:
    distance: float
    normalizedDistance: float


def dtw(x, y, keep_internals=False):
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return DTWResult(float("nan"), float("nan"))
    prev = np.full(b.size + 1, np.inf, dtype=float)
    prev[0] = 0.0
    for av in a:
        curr = np.full(b.size + 1, np.inf, dtype=float)
        cost = np.abs(b - av)
        for j in range(1, b.size + 1):
            curr[j] = cost[j - 1] + min(curr[j - 1], prev[j], prev[j - 1])
        prev = curr
    distance = float(prev[-1])
    return DTWResult(distance, distance / float(a.size + b.size))


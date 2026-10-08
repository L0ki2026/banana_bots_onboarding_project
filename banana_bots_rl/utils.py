"""Shared utilities."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    """Compute rolling mean; shorter prefix uses available samples."""
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return arr
    window = max(1, window)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

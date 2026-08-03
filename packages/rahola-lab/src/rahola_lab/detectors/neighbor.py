"""Faithful continuous rematch of Story's 2009 neighbor-loss detector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def neighbor_count_scores(
    features: NDArray[np.floating], *, radius: float, samples_per_period: int
) -> NDArray[np.float64]:
    """Return negative historical-neighbor count for the current phase point.

    Thesis Sec. 3.2.2 uses a warning at fewer than 50 neighbors. Sec. 4.2.2
    searches prior phase-space history and caches previously visited roll
    regions. Rahola resolves the unspecified radius in causally normalized
    (roll, roll-rate) distance and omits the immediately preceding roll period
    so serially adjacent samples cannot satisfy the count by themselves.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("features must have shape (windows, time, 2)")
    if radius <= 0.0 or samples_per_period < 1 or samples_per_period >= values.shape[1]:
        raise ValueError("invalid radius or temporal exclusion")
    current = values[:, -1:, :]
    history = values[:, :-samples_per_period, :]
    distance = np.linalg.norm(history - current, axis=2)
    return -np.sum(distance <= radius, axis=1).astype(np.float64)

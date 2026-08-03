"""Classical critical-slowing-down statistics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau


def classical_ews_scores(
    features: NDArray[np.floating], *, statistic: str, subwindow_fraction: float
) -> NDArray[np.float64]:
    """Kendall trend of rolling variance or lag-1 autocorrelation.

    Dakos et al. (PLoS ONE 7, 2012) use Kendall's tau to summarize
    monotonic change in rolling EWS values. The nested local window is a
    predeclared fraction of Rahola's causal 60-period detector history.
    """
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("features must have shape (windows, time, 2)")
    if statistic not in {"variance", "ac1"} or not 0.0 < subwindow_fraction < 1.0:
        raise ValueError("invalid EWS statistic or subwindow fraction")
    width = max(3, round(values.shape[1] * subwindow_fraction))
    roll = values[:, :, 0]
    endpoints = np.unique(np.linspace(width, len(roll[0]), 12, dtype=np.int64))
    indicators = np.empty((len(roll), len(endpoints)), dtype=np.float64)
    for column, end in enumerate(endpoints):
        local = roll[:, end - width : end]
        if statistic == "variance":
            indicators[:, column] = np.var(local, axis=1, ddof=1)
        else:
            left, right = local[:, :-1], local[:, 1:]
            left = left - left.mean(axis=1, keepdims=True)
            right = right - right.mean(axis=1, keepdims=True)
            denominator = np.sqrt(np.mean(left**2, axis=1) * np.mean(right**2, axis=1))
            indicators[:, column] = np.divide(
                np.mean(left * right, axis=1),
                denominator,
                out=np.zeros(len(local)),
                where=denominator > 1e-12,
            )
    output = np.empty(len(roll), dtype=np.float64)
    for row in range(len(roll)):
        tau = kendalltau(endpoints, indicators[row]).statistic
        output[row] = 0.0 if not np.isfinite(tau) else float(tau)
    return output

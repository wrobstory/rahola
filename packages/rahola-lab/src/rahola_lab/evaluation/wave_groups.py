"""Evaluator-only wave-envelope group identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


@dataclass(frozen=True)
class WaveGroup:
    start_s: float
    end_s: float
    maximum_height_m: float


def identify_wave_groups(
    times_s: NDArray[np.floating],
    elevation_m: NDArray[np.floating],
    *,
    significant_height_m: float,
    peak_period_s: float,
    height_fraction: float,
    minimum_periods: float,
) -> tuple[WaveGroup, ...]:
    """Return sustained runs whose Hilbert-envelope wave height exceeds a threshold.

    Twice the analytic-signal envelope is the local wave-height proxy. A run is
    retained when its inclusive sample duration is at least ``minimum_periods``
    times the spectral peak period.
    """
    times = np.asarray(times_s, dtype=np.float64)
    elevation = np.asarray(elevation_m, dtype=np.float64)
    if times.ndim != 1 or elevation.shape != times.shape or len(times) < 2:
        raise ValueError("times and elevation must be matching vectors with two samples")
    if np.any(np.diff(times) <= 0.0) or not np.all(np.isfinite(elevation)):
        raise ValueError("times must increase and elevation must be finite")
    if min(significant_height_m, peak_period_s, height_fraction, minimum_periods) <= 0.0:
        raise ValueError("wave-group parameters must be positive")
    dt_s = float(np.median(np.diff(times)))
    height_m = 2.0 * np.abs(hilbert(elevation))
    exceeds = height_m >= height_fraction * significant_height_m
    changes = np.diff(np.pad(exceeds.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    minimum_duration_s = minimum_periods * peak_period_s
    groups = []
    for start, stop in zip(starts, stops, strict=True):
        duration_s = (stop - start) * dt_s
        if duration_s >= minimum_duration_s:
            groups.append(
                WaveGroup(
                    start_s=float(times[start]),
                    end_s=float(times[stop - 1]),
                    maximum_height_m=float(np.max(height_m[start:stop])),
                )
            )
    return tuple(groups)


def intervals_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Return whether two closed time intervals overlap."""
    return start_a <= end_b and start_b <= end_a

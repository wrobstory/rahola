"""Autocorrelation-envelope timing and exceedance declustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from rahola_lab.evaluation.episodes import AlarmEpisode


def decorrelation_lag_from_autocorrelation(
    autocorrelation: NDArray[np.floating], *, significance_level: float = 0.05
) -> float:
    """Return the first crossing of the absolute-peak envelope.

    This follows Belenky et al. (Ocean Engineering 292, 2024, Fig. 16):
    connect absolute autocorrelation peaks and take the first crossing of the
    selected significance level.
    """
    values = np.asarray(autocorrelation, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("autocorrelation must be a finite vector with at least two lags")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance level must lie in (0, 1)")
    magnitude = np.abs(values)
    peaks = find_peaks(magnitude)[0]
    nodes = np.unique(np.concatenate(([0], peaks, [len(values) - 1])))
    envelope = np.interp(np.arange(len(values)), nodes, magnitude[nodes])
    below = np.flatnonzero(envelope <= significance_level)
    if not len(below):
        return float(len(values) - 1)
    upper = int(below[0])
    if upper == 0:
        return 0.0
    lower = upper - 1
    y0, y1 = envelope[lower], envelope[upper]
    if y0 == y1:
        return float(upper)
    return float(lower + (y0 - significance_level) / (y0 - y1))


def estimate_decorrelation_time(
    values: NDArray[np.floating], sample_interval_s: float, *, significance_level: float = 0.05
) -> float:
    """Estimate decorrelation time from a de-biased sample autocorrelation."""
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 3 or not np.all(np.isfinite(signal)):
        raise ValueError("values must be a finite vector with at least three samples")
    if sample_interval_s <= 0.0:
        raise ValueError("sample interval must be positive")
    centered = signal - signal.mean()
    variance = float(centered @ centered / len(centered))
    if variance <= 0.0:
        return sample_interval_s
    full = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    overlap = np.arange(len(centered), 0, -1, dtype=np.float64)
    autocorrelation = (full / overlap) / variance
    lag = decorrelation_lag_from_autocorrelation(
        autocorrelation, significance_level=significance_level
    )
    return max(sample_interval_s, lag * sample_interval_s)


def decluster_episodes(
    episodes: tuple[AlarmEpisode, ...], decorrelation_time_s: float
) -> tuple[AlarmEpisode, ...]:
    """Merge alarm episodes whose quiet gap is shorter than decorrelation time."""
    if decorrelation_time_s < 0.0:
        raise ValueError("decorrelation time cannot be negative")
    if not episodes:
        return ()
    clusters: list[AlarmEpisode] = []
    current = episodes[0]
    for episode in episodes[1:]:
        if episode.start_s - current.end_s <= decorrelation_time_s:
            current = AlarmEpisode(
                start_index=current.start_index,
                end_index=episode.end_index,
                start_s=current.start_s,
                end_s=episode.end_s,
                maximum_score=max(current.maximum_score, episode.maximum_score),
            )
        else:
            clusters.append(current)
            current = episode
    clusters.append(current)
    return tuple(clusters)

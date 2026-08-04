"""Causal intermediate-level crossings and Belenky-style declustering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from rahola_lab.evaluation import estimate_decorrelation_time
from rahola_lab.forecast import DangerMarginFit


@dataclass(frozen=True)
class Crossing:
    """One outward crossing detected when ``detection_index`` becomes available."""

    time_s: float
    detection_index: int
    side: int
    outward_rate_rad_s: float
    critical_rate_rad_s: float
    severity_u: float


def detect_crossings(
    time_s: NDArray[np.floating],
    angle_rad: NDArray[np.floating],
    rate_rad_s: NDArray[np.floating],
    fit: DangerMarginFit,
    *,
    critical_rate_scales: Mapping[int, NDArray[np.floating]] | None = None,
) -> tuple[Crossing, ...]:
    """Detect both-side crossings using only each interval's endpoint samples.

    The first non-finite angle or rate ends the stream, matching the absorbing
    post-capsize representation used by Rahola.
    """
    time = np.asarray(time_s, dtype=np.float64)
    angle = np.asarray(angle_rad, dtype=np.float64)
    rate = np.asarray(rate_rad_s, dtype=np.float64)
    if time.ndim != 1 or angle.shape != time.shape or rate.shape != time.shape:
        raise ValueError("time, angle, and rate must be matching vectors")
    if len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be finite and strictly increasing")

    finite = np.isfinite(angle) & np.isfinite(rate)
    invalid = np.flatnonzero(~finite)
    stop = int(invalid[0]) if len(invalid) else len(time)
    if np.any(finite[stop:]):
        raise ValueError("non-finite motion samples must end the stream")

    positive_level = fit.positive.threshold_angle_rad
    negative_level = fit.negative.threshold_angle_rad
    critical_by_side = {
        1: fit.positive.critical_rate_at_threshold(),
        -1: fit.negative.critical_rate_at_threshold(),
    }
    scales = {}
    for side in (1, -1):
        values = np.ones_like(time) if critical_rate_scales is None else np.asarray(
            critical_rate_scales[side], dtype=np.float64
        )
        if values.shape != time.shape or not np.all(np.isfinite(values[:stop])):
            raise ValueError("critical-rate scales must be finite vectors matching time")
        if np.any(values[:stop] <= 0.0):
            raise ValueError("critical-rate scales must be positive")
        scales[side] = values
    crossings: list[Crossing] = []
    for index in range(stop - 1):
        a0 = float(angle[index])
        a1 = float(angle[index + 1])
        side = 0
        level = 0.0
        if a0 < positive_level <= a1:
            side = 1
            level = positive_level
        elif a0 > negative_level >= a1:
            side = -1
            level = negative_level
        if side == 0:
            continue
        fraction = (level - a0) / (a1 - a0)
        crossing_time = float(time[index] + fraction * (time[index + 1] - time[index]))
        interpolated_rate = float(rate[index] + fraction * (rate[index + 1] - rate[index]))
        outward_rate = side * interpolated_rate
        critical_rate = critical_by_side[side] * float(scales[side][index + 1])
        if not np.isfinite(critical_rate) or critical_rate <= 0.0:
            raise ValueError("critical crossing rates must be finite and positive")
        crossings.append(
            Crossing(
                time_s=crossing_time,
                detection_index=index + 1,
                side=side,
                outward_rate_rad_s=outward_rate,
                critical_rate_rad_s=critical_rate,
                severity_u=outward_rate / critical_rate,
            )
        )
    return tuple(crossings)


def roll_decorrelation_time(
    angle_rad: NDArray[np.floating],
    sample_interval_s: float,
    *,
    significance_level: float = 0.05,
) -> float:
    """Estimate the first 0.05 crossing of the absolute-extrema ACF envelope."""
    return estimate_decorrelation_time(
        angle_rad,
        sample_interval_s,
        significance_level=significance_level,
    )


def decluster_crossings(
    crossings: tuple[Crossing, ...] | list[Crossing], decorrelation_time_s: float
) -> tuple[Crossing, ...]:
    """Retain the maximum-severity crossing from each chainwise cluster.

    Section 4.3 of Belenky et al. grows a cluster until no further crossing
    lies within the influence time. Consequently, membership compares each
    crossing with the preceding raw crossing, not with the current maximum.
    """
    if not np.isfinite(decorrelation_time_s) or decorrelation_time_s < 0.0:
        raise ValueError("decorrelation time must be finite and nonnegative")
    events = tuple(crossings)
    if any(right.time_s < left.time_s for left, right in pairwise(events)):
        raise ValueError("crossings must be ordered by time")
    if not events:
        return ()
    retained: list[Crossing] = []
    maximum = events[0]
    previous = events[0]
    for event in events[1:]:
        if event.time_s - previous.time_s <= decorrelation_time_s:
            if event.severity_u > maximum.severity_u:
                maximum = event
        else:
            retained.append(maximum)
            maximum = event
        previous = event
    retained.append(maximum)
    return tuple(retained)

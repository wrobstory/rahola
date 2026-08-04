from __future__ import annotations

import math

import numpy as np
import pytest
from rahola_lab.forecast import fit_piecewise_linear_restoring
from rahola_lab.splittime import Crossing, decluster_crossings, detect_crossings


def _fit():
    return fit_piecewise_linear_restoring(
        {
            "family": "softening",
            "natural_period_s": 4.0,
            "escape_angle_rad": math.radians(35.0),
            "negative_escape_angle_rad": None,
            "damping_ratio": 0.05,
            "bias_moment": 0.0,
            "quintic_coefficient": 0.0,
            "initial_angle_rad": 0.0,
        }
    )


def test_crossings_detect_both_sides_and_interpolate_at_sample_level() -> None:
    fit = _fit()
    time = np.arange(7, dtype=np.float64)
    angle = np.array([0.0, 0.2, 0.5, 0.2, -0.5, -0.2, 0.5])
    rate = np.array([0.2, 0.3, 0.5, -0.2, -0.6, 0.1, 0.7])
    crossings = detect_crossings(time, angle, rate, fit)

    assert [event.side for event in crossings] == [1, -1, 1]
    first_fraction = (fit.positive.threshold_angle_rad - 0.2) / 0.3
    assert crossings[0].time_s == pytest.approx(1.0 + first_fraction)
    assert abs(crossings[0].time_s - 2.0) <= 1.0
    assert crossings[0].detection_index == 2
    assert crossings[1].outward_rate_rad_s > 0.0


def test_crossing_detection_is_unchanged_when_only_future_samples_are_nan() -> None:
    fit = _fit()
    time = np.arange(8, dtype=np.float64)
    angle = np.array([0.0, 0.2, 0.5, 0.1, -0.5, 0.0, 0.5, 0.0])
    rate = np.gradient(angle)
    complete = detect_crossings(time, angle, rate, fit)
    truncated_angle = angle.copy()
    truncated_rate = rate.copy()
    truncated_angle[5:] = np.nan
    truncated_rate[5:] = np.nan
    truncated = detect_crossings(time, truncated_angle, truncated_rate, fit)
    assert truncated == tuple(event for event in complete if event.detection_index < 5)


def test_dynamic_critical_rate_scale_changes_only_normalized_severity() -> None:
    fit = _fit()
    time = np.arange(3, dtype=np.float64)
    angle = np.array([0.0, 0.2, 0.5])
    rate = np.full(3, 0.4)
    fixed = detect_crossings(time, angle, rate, fit)[0]
    adaptive = detect_crossings(
        time,
        angle,
        rate,
        fit,
        critical_rate_scales={1: np.full(3, 2.0), -1: np.ones(3)},
    )[0]
    assert adaptive.time_s == fixed.time_s
    assert adaptive.outward_rate_rad_s == fixed.outward_rate_rad_s
    assert adaptive.critical_rate_rad_s == pytest.approx(2.0 * fixed.critical_rate_rad_s)
    assert adaptive.severity_u == pytest.approx(0.5 * fixed.severity_u)


def _event(time_s: float, severity: float) -> Crossing:
    return Crossing(time_s, int(time_s), 1, severity, 1.0, severity)


def test_declustering_grows_chainwise_and_retains_cluster_maximum() -> None:
    crossings = [_event(0.0, 0.4), _event(3.0, 1.2), _event(7.0, 0.8), _event(20.0, 0.9)]
    assert decluster_crossings(crossings, 5.0) == (crossings[1], crossings[3])


def test_whitened_spaced_crossings_are_unchanged_by_declustering() -> None:
    crossings = [_event(0.0, 0.4), _event(2.0, 1.2), _event(4.0, 0.8)]
    assert decluster_crossings(crossings, 1.0) == tuple(crossings)

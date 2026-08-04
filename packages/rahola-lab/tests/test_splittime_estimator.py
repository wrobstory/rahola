from __future__ import annotations

import math

import numpy as np
import pytest
from rahola_lab.forecast import fit_piecewise_linear_restoring
from rahola_lab.splittime import GammaRatePrior, SplitTimeConfig, estimate_rate_trajectory


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


def _stationary_record(seed: int, duration_s: int = 3_600):
    rng = np.random.default_rng(seed)
    fit = _fit()
    time = np.arange(duration_s + 1, dtype=np.float64)
    angle = rng.uniform(-0.03, 0.03, size=len(time))
    rate = rng.normal(0.0, 0.01, size=len(time))
    candidate_times = rng.choice(np.arange(2, duration_s - 2), size=90, replace=False)
    candidate_times.sort()
    selected: list[int] = []
    for crossing_time in candidate_times:
        if not selected or crossing_time - selected[-1] > 4:
            selected.append(int(crossing_time))
    severities = 0.20 + rng.exponential(1.0 / 4.0, size=len(selected))
    threshold = fit.positive.threshold_angle_rad
    critical = fit.positive.critical_rate_at_threshold()
    for crossing_time, severity in zip(selected, severities, strict=True):
        angle[crossing_time - 1] = threshold - 0.02
        angle[crossing_time] = threshold + 0.02
        angle[crossing_time + 1] = 0.0
        rate[crossing_time - 1 : crossing_time + 1] = severity * critical
    return time, angle, rate, np.asarray(severities)


def test_stationary_known_configuration_rate_is_self_calibrated() -> None:
    fit = _fit()
    predicted = 0.0
    realized = 0
    lower = 0.0
    upper = 0.0
    for seed in range(6):
        time, angle, rate, severities = _stationary_record(seed)
        trajectory = estimate_rate_trajectory(
            time,
            angle,
            rate,
            fit,
            prior=GammaRatePrior.from_mean(4.0, strength=10.0),
            config=SplitTimeConfig(
                tail_quantile=0.50,
                trailing_window_s=None,
                emission_cadence_s=60.0,
                interval_cadence_s=60.0,
            ),
        )
        predicted += trajectory.integrated_count
        lo, hi = trajectory.integrated_interval
        lower += lo
        upper += hi
        realized += int(np.sum(severities >= 1.0))
    assert predicted == pytest.approx(realized, rel=0.35, abs=2.0)
    assert lower <= realized <= upper


def test_future_only_changes_do_not_change_earlier_rate_emissions() -> None:
    fit = _fit()
    time, angle, rate, _ = _stationary_record(77, duration_s=600)
    config = SplitTimeConfig(
        tail_quantile=0.50,
        trailing_window_s=None,
        emission_cadence_s=10.0,
        interval_cadence_s=60.0,
    )
    prior = GammaRatePrior.from_mean(4.0, strength=10.0)
    complete = estimate_rate_trajectory(time, angle, rate, fit, prior=prior, config=config)
    truncated_angle = angle.copy()
    truncated_rate = rate.copy()
    truncated_angle[time > 300.0] = np.nan
    truncated_rate[time > 300.0] = np.nan
    truncated = estimate_rate_trajectory(
        time,
        truncated_angle,
        truncated_rate,
        fit,
        prior=prior,
        config=config,
    )
    expected = tuple(emission for emission in complete.emissions if emission.time_s <= 300.0)
    assert truncated.emissions == expected


def test_prior_from_start_emits_and_intervals_a_zero_crossing_stream() -> None:
    fit = _fit()
    time = np.arange(0.0, 61.0)
    angle = np.zeros_like(time)
    rate = np.zeros_like(time)
    trajectory = estimate_rate_trajectory(
        time,
        angle,
        rate,
        fit,
        prior=GammaRatePrior.from_mean(
            4.0,
            strength=10.0,
            threshold_w=0.75,
            exceedance_probability=0.25,
        ),
        config=SplitTimeConfig(
            tail_quantile=0.75,
            trailing_window_s=None,
            emission_cadence_s=10.0,
            interval_cadence_s=60.0,
            emission_policy="prior_from_start",
        ),
    )
    assert [emission.time_s for emission in trajectory.emissions] == list(
        np.arange(0.0, 61.0, 10.0)
    )
    assert all(emission.rate_per_hour == 0.0 for emission in trajectory.emissions)
    assert all("prior_dominated" in emission.flags for emission in trajectory.emissions)
    assert trajectory.emissions[0].interval_upper_per_hour > 0.0
    assert trajectory.integrated_count == 0.0
    assert np.all(trajectory.integrated_count_draws == 0.0)

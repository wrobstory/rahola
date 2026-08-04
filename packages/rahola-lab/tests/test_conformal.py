from __future__ import annotations

import math

import numpy as np
import pytest
from rahola_lab.conformal import (
    SplitCQRUpper,
    adaptive_conformal_bounds,
    conformal_quantile,
    dynamically_tuned_aci_bounds,
    normalized_alarm_scores,
    sliding_recalibrated_aci_bounds,
)
from rahola_lab.experiments.e3 import _recovery_time
from scipy.stats import binomtest


def test_inflated_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1.0, 10.0)
    assert conformal_quantile(scores, 0.2) == 8.0
    assert math.isinf(conformal_quantile(scores, 0.05))


def test_alarm_score_crosses_one_at_frozen_escape_fraction() -> None:
    scores = normalized_alarm_scores(np.array([0.29, 0.30, 0.31]), 0.5)
    np.testing.assert_allclose(scores, [29 / 30, 1.0, 31 / 30])


def test_alarm_score_maps_exterior_conformal_bounds_to_finite_extremes() -> None:
    scores = normalized_alarm_scores(np.array([-np.inf, np.inf]), 0.5)
    assert np.all(np.isfinite(scores))
    assert scores[0] < 0.0 < scores[1]
    with pytest.raises(ValueError, match="NaN"):
        normalized_alarm_scores(np.array([np.nan]), 0.5)
    with pytest.raises(ValueError, match="finite"):
        normalized_alarm_scores(np.array([0.25]), float("nan"))


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_split_cqr_exchangeable_finite_sample_coverage(alpha: float) -> None:
    """Falsify Theorem 2 of Romano et al. at two miscoverage levels.

    With distinct exchangeable scores, exact coverage is
    ceil((n+1)(1-alpha))/(n+1), bounded above by 1-alpha+1/(n+1).
    The expected value must lie inside a 99.9% exact Clopper-Pearson interval
    over independent exchangeable trials.
    """
    rng = np.random.default_rng(round(alpha * 10_000))
    calibration_size = 99
    trials = 20_000
    residuals = rng.standard_t(df=3, size=(trials, calibration_size + 1)) + 2.0
    calibration = residuals[:, :calibration_size]
    test = residuals[:, -1]
    rank = math.ceil((calibration_size + 1) * (1.0 - alpha))
    correction = np.partition(calibration, rank - 1, axis=1)[:, rank - 1]
    covered = test <= correction
    expected = rank / (calibration_size + 1)
    interval = binomtest(int(covered.sum()), trials).proportion_ci(confidence_level=0.999)
    assert interval.low <= expected <= interval.high
    assert 1.0 - alpha <= expected <= 1.0 - alpha + 1.0 / (calibration_size + 1)


def test_split_cqr_wraps_a_deliberately_bad_forecaster() -> None:
    targets = np.array([2.0, 3.0, 4.0, 5.0])
    calibration = SplitCQRUpper.calibrate(targets, np.full(4, -10.0))
    assert calibration.upper_bound(np.array([-10.0]), alpha=0.5)[0] >= 4.0


def test_recovery_requires_coverage_to_remain_in_tolerance() -> None:
    times = np.array([300.0, 310.0, 320.0, 330.0, 340.0])
    assert _recovery_time(times, np.array([0.9, 0.9, 0.9, 0.8, 0.9])) is None
    assert _recovery_time(times, np.array([0.8, 0.9, 0.9, 0.9, 0.9])) == 10.0


def test_single_expert_dtaci_reduces_to_scalar_aci() -> None:
    rng = np.random.default_rng(19)
    scores = rng.normal(size=100)
    targets = rng.normal(size=200)
    raw = np.zeros_like(targets)
    scalar = adaptive_conformal_bounds(scores, raw, targets, alpha=0.1, gamma=0.01)
    tuned = dynamically_tuned_aci_bounds(scores, raw, targets, alpha=0.1, gamma_experts=(0.01,))
    np.testing.assert_allclose(tuned.upper_bounds, scalar.upper_bounds)
    np.testing.assert_allclose(tuned.working_alpha, scalar.working_alpha)
    np.testing.assert_array_equal(tuned.errors, scalar.errors)


def test_online_adapters_wait_for_forecast_outcomes() -> None:
    scores = np.linspace(-1.0, 1.0, 99)
    raw = np.zeros(12)
    ordinary = np.zeros(12)
    perturbed = ordinary.copy()
    perturbed[0] = 10.0
    scalar_a = adaptive_conformal_bounds(
        scores, raw, ordinary, alpha=0.1, gamma=0.01, feedback_delay_steps=6
    )
    scalar_b = adaptive_conformal_bounds(
        scores, raw, perturbed, alpha=0.1, gamma=0.01, feedback_delay_steps=6
    )
    tuned_a = dynamically_tuned_aci_bounds(
        scores,
        raw,
        ordinary,
        alpha=0.1,
        gamma_experts=(0.01, 0.02),
        feedback_delay_steps=6,
    )
    tuned_b = dynamically_tuned_aci_bounds(
        scores,
        raw,
        perturbed,
        alpha=0.1,
        gamma_experts=(0.01, 0.02),
        feedback_delay_steps=6,
    )
    sliding_a = sliding_recalibrated_aci_bounds(
        scores,
        raw,
        ordinary,
        alpha=0.1,
        gamma=0.01,
        window_size=20,
        feedback_delay_steps=6,
    )
    sliding_b = sliding_recalibrated_aci_bounds(
        scores,
        raw,
        perturbed,
        alpha=0.1,
        gamma=0.01,
        window_size=20,
        feedback_delay_steps=6,
    )
    for first, second in (
        (scalar_a, scalar_b),
        (tuned_a, tuned_b),
        (sliding_a, sliding_b),
    ):
        np.testing.assert_array_equal(first.upper_bounds[:6], second.upper_bounds[:6])
        assert first.upper_bounds[6] != second.upper_bounds[6]


def test_recent_score_recalibration_repairs_a_shifted_score_stream() -> None:
    rng = np.random.default_rng(21)
    scores = rng.normal(size=200)
    raw = np.zeros(2_000)
    targets = rng.normal(loc=2.0, size=len(raw))
    fixed = SplitCQRUpper(scores).upper_bound(raw, 0.1)
    sliding = sliding_recalibrated_aci_bounds(
        scores, raw, targets, alpha=0.1, gamma=0.02, window_size=50
    )
    assert np.mean(targets > fixed) > 0.5
    assert abs(np.mean(sliding.errors[-1_000:]) - 0.1) < 0.03


@pytest.mark.slow
def test_aci_recovers_long_run_coverage_while_fixed_cqr_breaks() -> None:
    rng = np.random.default_rng(904)
    alpha = 0.1
    gamma = 0.01
    calibration_scores = rng.normal(size=500)
    stationary = rng.normal(size=2_000)
    shifted = rng.normal(loc=3.0, size=6_000)
    targets = np.concatenate((stationary, shifted))
    raw = np.zeros_like(targets)

    fixed = SplitCQRUpper(calibration_scores).upper_bound(raw, alpha)
    fixed_errors = targets > fixed
    adaptive = adaptive_conformal_bounds(
        calibration_scores,
        raw,
        targets,
        alpha=alpha,
        gamma=gamma,
    )
    pathwise_tolerance = (max(alpha, 1.0 - alpha) + gamma) / (len(targets) * gamma)
    assert abs(float(adaptive.errors.mean()) - alpha) <= pathwise_tolerance + 1e-12
    assert float(fixed_errors[2_000:].mean()) > 0.8
    assert abs(float(adaptive.errors[2_000:].mean()) - alpha) < 0.03

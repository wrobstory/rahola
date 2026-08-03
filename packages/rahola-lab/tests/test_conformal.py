from __future__ import annotations

import math

import numpy as np
import pytest
from rahola_lab.conformal import (
    SplitCQRUpper,
    adaptive_conformal_bounds,
    conformal_quantile,
    normalized_alarm_scores,
)
from scipy.stats import binomtest


def test_inflated_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1.0, 10.0)
    assert conformal_quantile(scores, 0.2) == 8.0
    assert math.isinf(conformal_quantile(scores, 0.05))


def test_alarm_score_crosses_one_at_frozen_escape_fraction() -> None:
    scores = normalized_alarm_scores(np.array([0.29, 0.30, 0.31]), 0.5)
    np.testing.assert_allclose(scores, [29 / 30, 1.0, 31 / 30])


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

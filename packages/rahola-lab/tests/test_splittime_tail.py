from __future__ import annotations

import numpy as np
import pytest
from rahola_lab.splittime import GammaRatePrior, estimate_exponential_tail


def test_exponential_tail_recovers_known_critical_exceedance_at_large_n() -> None:
    rng = np.random.default_rng(104)
    theta = 4.0
    base = 0.20
    severities = base + rng.exponential(1.0 / theta, size=100_000)
    prior = GammaRatePrior.from_mean(theta, strength=2.0)
    estimate = estimate_exponential_tail(severities, quantile=0.50, prior=prior)
    truth = math_exp = float(np.exp(-theta * (1.0 - base)))
    assert estimate.critical_probability == pytest.approx(truth, abs=0.003)
    assert math_exp > 0.0


def test_three_exceedances_with_strong_prior_shrink_toward_prior_mean() -> None:
    severities = np.array([0.05, 0.10, 0.15, 0.40, 0.55, 0.80])
    prior_mean = 10.0
    estimate = estimate_exponential_tail(
        severities,
        quantile=0.50,
        prior=GammaRatePrior.from_mean(prior_mean, strength=100.0),
    )
    exceedances = severities[severities > estimate.threshold_w] - estimate.threshold_w
    mle = len(exceedances) / exceedances.sum()
    assert estimate.exceedance_count == 3
    assert abs(estimate.posterior_mean_rate - prior_mean) < abs(mle - prior_mean)


def test_tail_threshold_is_clipped_below_normalized_critical_level() -> None:
    estimate = estimate_exponential_tail(
        np.array([0.9, 1.0, 1.1, 1.2]),
        quantile=0.75,
        prior=GammaRatePrior.from_mean(2.0, strength=5.0),
    )
    assert estimate.threshold_clipped
    assert estimate.threshold_w < 1.0


def test_empty_tail_uses_pooled_fixed_threshold_prior() -> None:
    prior = GammaRatePrior.from_mean(
        4.0,
        strength=10.0,
        threshold_w=0.75,
        exceedance_probability=0.25,
    )
    estimate = estimate_exponential_tail(
        np.empty(0),
        quantile=0.75,
        prior=prior,
    )
    assert estimate.crossing_count == 0
    assert estimate.exceedance_count == 0
    assert estimate.posterior_mean_rate == pytest.approx(prior.mean_rate)
    assert estimate.critical_probability > 0.0

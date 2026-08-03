from __future__ import annotations

import math

import numpy as np
import pytest
from rahola_lab.forecast import fit_piecewise_linear_restoring


def _config(**updates):
    values = {
        "family": "softening",
        "natural_period_s": 4.0,
        "escape_angle_rad": math.radians(35.0),
        "negative_escape_angle_rad": None,
        "damping_ratio": 0.05,
        "bias_moment": 0.0,
        "quintic_coefficient": 0.0,
        "initial_angle_rad": 0.0,
    }
    values.update(updates)
    return values


def test_cubic_piecewise_fit_matches_peak_escape_and_equation_13() -> None:
    config = _config()
    fit = fit_piecewise_linear_restoring(config)
    escape = config["escape_angle_rad"]
    threshold = escape / math.sqrt(3.0)
    expected_k1 = threshold / (escape - threshold)
    delta = 0.05 * 2.0 * math.pi / 4.0
    expected_growth = delta + math.sqrt(expected_k1 * (2.0 * math.pi / 4.0) ** 2 + delta**2)
    assert fit.equilibrium_angle_rad == pytest.approx(0.0)
    assert fit.positive.threshold_angle_rad == pytest.approx(threshold)
    assert fit.negative.threshold_angle_rad == pytest.approx(-threshold)
    assert fit.positive.repeller_slope == pytest.approx(expected_k1)
    assert fit.positive.critical_rate_at_threshold() == pytest.approx(
        expected_growth * (escape - threshold)
    )


def test_forced_solution_correction_matches_equation_15() -> None:
    fit = fit_piecewise_linear_restoring(_config())
    unforced = fit.positive.critical_rate_at_threshold()
    corrected = fit.positive.critical_rate_at_threshold(0.02, -0.01)
    assert corrected == pytest.approx(unforced + fit.positive.growth_rate_s * 0.02 - 0.01)


def test_instantaneous_danger_score_is_zero_on_fitted_separatrix() -> None:
    fit = fit_piecewise_linear_restoring(_config())
    angle = np.array([fit.positive.threshold_angle_rad])
    critical = np.array([fit.positive.critical_rate_at_threshold()])
    np.testing.assert_allclose(fit.danger_score(angle, critical), [0.0], atol=1e-12)


def test_biased_fit_translates_equilibrium_and_uses_asymmetric_escapes() -> None:
    fit = fit_piecewise_linear_restoring(
        _config(
            family="biased",
            bias_moment=-0.15,
            initial_angle_rad=-0.09384459849427874,
            negative_escape_angle_rad=math.radians(25.0),
        )
    )
    assert fit.equilibrium_angle_rad == pytest.approx(-0.09384459849427874)
    assert fit.positive.vanishing_angle_rad == pytest.approx(math.radians(35.0))
    assert fit.negative.vanishing_angle_rad == pytest.approx(-math.radians(25.0))
    assert fit.positive.repeller_slope != pytest.approx(fit.negative.repeller_slope)

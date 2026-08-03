from __future__ import annotations

import math

import numpy as np
import pytest

from rahola.config import ForcingConfig, SeaState, SimulationConfig
from rahola.simulate import simulate_batch
from rahola.spectrum import jonswap_spectrum
from rahola.validation import (
    damped_mathieu_threshold,
    find_harmonic_capsize_boundary,
    harmonic_capsize_fraction,
    linear_transfer_function,
    mathieu_growth_rate,
    melnikov_heteroclinic_threshold,
    numerical_melnikov_threshold,
)


@pytest.mark.slow
def test_linear_limit_variance_matches_spectral_response() -> None:
    config = SimulationConfig(
        duration_s=2048.0,
        natural_period_s=8.0,
        escape_angle_rad=10.0,
        damping_ratio=0.08,
        quadratic_damping=0.0,
        output_rate_hz=5.0,
        forcing=ForcingConfig(
            sea_state=SeaState(hs_m=2.0, tp_s=8.0, gamma=1.0),
            effective_wave_slope=0.05,
        ),
        linear_restoring=True,
    )
    dataset = simulate_batch(config, range(8))
    simulated_variance = np.mean(np.var(dataset.angle_rad[:, len(dataset.time_s) // 4 :], axis=1))

    omega = np.linspace(1e-4, math.pi / (0.5 * config.integration_dt_s), 100_000)
    elevation_spectrum = jonswap_spectrum(
        omega, config.forcing.sea_state, config.forcing.gravity_m_s2
    )
    slope_spectrum = (omega**2 / config.forcing.gravity_m_s2) ** 2 * elevation_spectrum
    moment_spectrum = (
        config.omega_n_rad_s**4 * config.forcing.effective_wave_slope**2 * slope_spectrum
    )
    transfer = linear_transfer_function(omega, config.omega_n_rad_s, config.damping_ratio)
    analytic_variance = np.trapezoid(abs(transfer) ** 2 * moment_spectrum, omega)
    assert simulated_variance == pytest.approx(analytic_variance, rel=0.06)


@pytest.mark.slow
def test_mathieu_principal_tongue_boundary() -> None:
    zeta = 0.02
    prediction = damped_mathieu_threshold(zeta)
    low_rate = mathieu_growth_rate(0.94 * prediction, zeta)
    high_rate = mathieu_growth_rate(1.06 * prediction, zeta)
    assert low_rate < 0
    assert high_rate > 0
    grid = np.linspace(0.85 * prediction, 1.15 * prediction, 13)
    rates = np.asarray([mathieu_growth_rate(value, zeta) for value in grid])
    crossing = grid[np.flatnonzero(rates > 0)[0]]
    assert crossing == pytest.approx(prediction, rel=0.10)


def test_melnikov_closed_form_matches_orbit_quadrature() -> None:
    for frequency in (0.5, 1.0, 1.5):
        analytic = melnikov_heteroclinic_threshold(0.02, frequency)
        numerical = numerical_melnikov_threshold(0.02, frequency)
        assert numerical == pytest.approx(analytic, rel=1e-5)


@pytest.mark.slow
def test_no_capsize_below_melnikov_necessary_condition() -> None:
    for damping in (0.015, 0.04):
        for frequency in (0.7, 1.0, 1.3):
            melnikov = melnikov_heteroclinic_threshold(damping, frequency)
            fraction = harmonic_capsize_fraction(0.95 * melnikov, frequency, damping)
            assert fraction == 0.0


@pytest.mark.slow
def test_capsize_boundary_respects_melnikov_lower_bound_and_shape() -> None:
    frequencies = np.array([0.7, 1.0, 1.3])
    damping = 0.015
    predictions = np.array(
        [melnikov_heteroclinic_threshold(damping, frequency) for frequency in frequencies]
    )
    simulated = np.array(
        [find_harmonic_capsize_boundary(frequency, damping) for frequency in frequencies]
    )
    assert np.all(simulated >= predictions)
    assert np.corrcoef(predictions, simulated)[0, 1] > 0.8


@pytest.mark.slow
def test_capsize_boundary_moves_toward_melnikov_bound_as_damping_falls() -> None:
    frequency = 1.0
    high_damping, low_damping = 0.04, 0.01
    high_prediction = melnikov_heteroclinic_threshold(high_damping, frequency)
    low_prediction = melnikov_heteroclinic_threshold(low_damping, frequency)
    high_simulated = find_harmonic_capsize_boundary(frequency, high_damping)
    low_simulated = find_harmonic_capsize_boundary(frequency, low_damping)
    assert low_simulated < high_simulated
    assert low_simulated - low_prediction < high_simulated - high_prediction

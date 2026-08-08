from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from rahola.config import SeaState
from rahola.spectrum import jonswap_spectrum, synthesize_jonswap


def _reference_sigma(scaled_frequency: np.ndarray) -> np.ndarray:
    return np.where(scaled_frequency <= 1.0, 0.07, 0.09)


def _reference_jonswap(omega_rad_s: np.ndarray, sea_state: SeaState) -> np.ndarray:
    """Independent dimensionless-form JONSWAP evaluator for the production oracle."""
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    peak_frequency = 2.0 * np.pi / sea_state.tp_s
    scaled_frequency = np.divide(
        omega,
        peak_frequency,
        out=np.ones_like(omega),
        where=omega > 0.0,
    )
    spread = _reference_sigma(scaled_frequency)
    peak_enhancement = np.exp(-0.5 * ((scaled_frequency - 1.0) / spread) ** 2)
    base = (
        (9.80665 / peak_frequency) ** 2
        * np.exp(-1.25 / scaled_frequency**4)
        * np.exp(np.log(sea_state.gamma) * peak_enhancement)
        / scaled_frequency**5
    )
    shape = np.where(omega > 0.0, base, 0.0)
    return shape * (sea_state.hs_m**2 / 16.0) / np.trapezoid(shape, omega)


def _reference_shape(omega_rad_s: np.ndarray, sea_state: SeaState) -> np.ndarray:
    """Unnormalized reference shape used for the hand-checked constants below."""
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    peak_frequency = 2.0 * np.pi / sea_state.tp_s
    scaled_frequency = omega / peak_frequency
    spread = _reference_sigma(scaled_frequency)
    peak_enhancement = np.exp(-0.5 * ((scaled_frequency - 1.0) / spread) ** 2)
    return (
        scaled_frequency**-5
        * np.exp(-1.25 / scaled_frequency**4)
        * np.exp(np.log(sea_state.gamma) * peak_enhancement)
    )


def test_jonswap_matches_independent_formula_oracle() -> None:
    sea_state = SeaState(hs_m=4.0, tp_s=10.0, gamma=3.3)
    omega_p = 2.0 * np.pi / sea_state.tp_s
    omega = np.unique(
        np.concatenate(
            (
                np.geomspace(0.05 * omega_p, 20.0 * omega_p, 2048),
                omega_p * np.array([0.99, 1.0, 1.01]),
            )
        )
    )
    np.testing.assert_allclose(
        jonswap_spectrum(omega, sea_state),
        _reference_jonswap(omega, sea_state),
        rtol=5e-13,
        atol=1e-14,
    )

    # Hand checks for the independent formula: S(omega_p)/S(2 omega_p),
    # sigma at the inclusive peak break, and the fitted far-tail log slope.
    assert _reference_shape(np.array([omega_p]), sea_state)[0] / _reference_shape(
        np.array([2.0 * omega_p]), sea_state
    )[0] == pytest.approx(32.71335391913758, rel=1e-13)
    assert _reference_sigma(np.array([1.0]))[0] == pytest.approx(0.07)
    assert _reference_sigma(np.array([np.nextafter(1.0, np.inf)]))[0] == pytest.approx(0.09)
    tail = np.geomspace(20.0 * omega_p, 40.0 * omega_p, 100)
    tail_slope = np.polyfit(np.log(tail), np.log(_reference_shape(tail, sea_state)), 1)[0]
    # Fixed log-spaced 20 omega_p to 40 omega_p fit from the reference curve.
    assert tail_slope == pytest.approx(-4.999990544763571, abs=1e-12)


@pytest.mark.slow
def test_jonswap_spectral_fidelity_and_significant_height() -> None:
    sea_state = SeaState(hs_m=4.0, tp_s=10.0, gamma=3.3)
    realization = synthesize_jonswap(sea_state, 4096.0, 0.25, seed=814)
    recovered_hs = 4.0 * np.std(realization.elevation_m)
    assert recovered_hs == pytest.approx(sea_state.hs_m, rel=0.02)

    frequency_hz, estimated_hz = welch(
        realization.elevation_m,
        fs=4.0,
        nperseg=4096,
        noverlap=2048,
        detrend=False,
        window="hann",
    )
    omega = 2.0 * np.pi * frequency_hz
    target_hz = 2.0 * np.pi * jonswap_spectrum(omega, sea_state)
    band = (frequency_hz >= 0.04) & (frequency_hz <= 0.5) & (target_hz > 1e-8)
    log_correlation = np.corrcoef(np.log(estimated_hz[band]), np.log(target_hz[band]))[0, 1]
    estimated_band_energy = np.trapezoid(estimated_hz[band], frequency_hz[band])
    target_band_energy = np.trapezoid(target_hz[band], frequency_hz[band])
    assert log_correlation > 0.95
    assert estimated_band_energy == pytest.approx(target_band_energy, rel=0.12)


def test_same_seed_is_bitwise_reproducible() -> None:
    kwargs = dict(sea_state=SeaState(), duration_s=64.0, dt_s=0.1, seed=42)
    first = synthesize_jonswap(**kwargs)
    second = synthesize_jonswap(**kwargs)
    assert np.array_equal(first.elevation_m, second.elevation_m)
    assert np.array_equal(first.slope_rad, second.slope_rad)


def test_deep_water_slope_fourier_coefficients_have_expected_magnitude_and_sign() -> None:
    gravity = 9.80665
    dt_s = 0.25
    realization = synthesize_jonswap(
        SeaState(),
        duration_s=512.0,
        dt_s=dt_s,
        seed=91,
        min_components=4,
        max_frequency_rad_s=5.0,
    )
    sample_count = len(realization.elevation_m) - 1
    omega = 2.0 * np.pi * np.fft.rfftfreq(sample_count, d=dt_s)
    elevation_coefficients = np.fft.rfft(realization.elevation_m[:-1])
    slope_coefficients = np.fft.rfft(realization.slope_rad[:-1])
    active = (omega > 0.0) & (omega < 5.0 * (1.0 - 1e-12))
    np.testing.assert_allclose(
        realization.frequencies_rad_s, omega[: len(realization.frequencies_rad_s)]
    )
    expected = -1j * elevation_coefficients[active] * omega[active] ** 2 / gravity
    np.testing.assert_allclose(slope_coefficients[active], expected, rtol=1e-11, atol=1e-11)


def test_zero_upcrossing_rate_matches_rice_formula_predictive_interval() -> None:
    """Permanent analytic crossing-rate oracle for the synthesized spectrum path."""
    sea_state = SeaState(hs_m=4.0, tp_s=4.0, gamma=3.3)
    duration_s = 256.0
    counts = []
    expected_rates = []
    for seed in range(128):
        realization = synthesize_jonswap(
            sea_state,
            duration_s=duration_s,
            dt_s=0.05,
            seed=seed,
            max_frequency_rad_s=20.0,
        )
        values = realization.elevation_m[:-1]
        counts.append(np.count_nonzero((values[:-1] < 0.0) & (values[1:] >= 0.0)))
        delta_omega = realization.frequencies_rad_s[1]
        energy = realization.target_spectrum_m2_s * delta_omega
        m0 = np.sum(energy)
        m2 = np.sum(energy * realization.frequencies_rad_s**2)
        expected_rates.append(np.sqrt(m2 / m0) / (2.0 * np.pi))

    expected_count = float(np.mean(expected_rates)) * duration_s
    standard_error = float(np.std(counts, ddof=1)) / np.sqrt(len(counts))
    assert np.mean(counts) == pytest.approx(expected_count, abs=3.5 * standard_error)

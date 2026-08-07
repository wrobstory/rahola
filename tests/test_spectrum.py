from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from rahola.config import SeaState
from rahola.spectrum import jonswap_spectrum, synthesize_jonswap


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

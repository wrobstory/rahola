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

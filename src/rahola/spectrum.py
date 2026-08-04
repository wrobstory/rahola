"""JONSWAP spectra and seeded inverse-FFT sea realizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola.config import SeaState

FloatArray = NDArray[np.float64]


def jonswap_spectrum(
    omega_rad_s: FloatArray,
    sea_state: SeaState,
    gravity_m_s2: float = 9.80665,
) -> FloatArray:
    """Return one-sided JONSWAP S_eta(omega), normalized to m0 = Hs^2 / 16."""
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    omega_p = 2.0 * np.pi / sea_state.tp_s
    safe = np.where(omega > 0, omega, 1.0)
    sigma = np.where(omega <= omega_p, 0.07, 0.09)
    peak = np.exp(-0.5 * ((omega - omega_p) / (sigma * omega_p)) ** 2)
    shape = (
        gravity_m_s2**2 * safe**-5 * np.exp(-1.25 * (omega_p / safe) ** 4) * sea_state.gamma**peak
    )
    shape = np.where(omega > 0, shape, 0.0)
    area = np.trapezoid(shape, omega)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("frequency grid does not resolve a JONSWAP spectrum")
    return shape * (sea_state.hs_m**2 / 16.0) / area


@dataclass(frozen=True)
class SeaRealization:
    time_s: FloatArray
    elevation_m: FloatArray
    slope_rad: FloatArray
    frequencies_rad_s: FloatArray
    target_spectrum_m2_s: FloatArray


def synthesize_jonswap(
    sea_state: SeaState,
    duration_s: float,
    dt_s: float,
    seed: int,
    *,
    min_components: int = 200,
    gravity_m_s2: float = 9.80665,
    max_frequency_rad_s: float | None = None,
) -> SeaRealization:
    """Synthesize eta and deep-water slope with deterministic amplitudes/random phases.

    With ``max_frequency_rad_s``, the spectral grid depends only on duration and
    cutoff, not the evaluation step. The FFT period is an integer multiple of
    the requested duration so step-halved grids evaluate the same random-phase
    trigonometric field. ``None`` retains the v0.1 Nyquist-limited definition
    for the documented invariance comparison.
    """
    requested_n = round(duration_s / dt_s) + 1
    if max_frequency_rad_s is None:
        fft_n = max(requested_n, 2 * (min_components + 1))
        if fft_n % 2:
            fft_n += 1
    else:
        if not np.isfinite(max_frequency_rad_s) or max_frequency_rad_s <= 0.0:
            raise ValueError("max_frequency_rad_s must be positive and finite")
        interval_count = requested_n - 1
        minimum_period_multiplier = int(
            np.ceil(
                2.0
                * np.pi
                * (min_components + 1)
                / (max_frequency_rad_s * duration_s)
            )
        )
        fft_n = interval_count * max(1, minimum_period_multiplier)
    frequencies_hz = np.fft.rfftfreq(fft_n, d=dt_s)
    omega = 2.0 * np.pi * frequencies_hz
    if max_frequency_rad_s is None:
        active = np.ones_like(omega, dtype=bool)
    else:
        # The cutoff bin is open so a cutoff equal to the coarse-grid Nyquist
        # omits that real-only FFT coefficient on every refined grid as well.
        active = omega < max_frequency_rad_s * (1.0 - 1e-12)
        if int(np.sum(active)) - 1 < min_components:
            raise ValueError("fixed spectral grid did not reach min_components")
    active_omega = omega[active]
    spectrum = jonswap_spectrum(active_omega, sea_state, gravity_m_s2)
    delta_omega = 2.0 * np.pi / (fft_n * dt_s)
    amplitudes = np.sqrt(2.0 * spectrum * delta_omega)
    rng = np.random.default_rng(np.uint64(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=active_omega.size)
    phases[0] = 0.0
    amplitudes[0] = 0.0
    coefficients = np.zeros(omega.size, dtype=np.complex128)
    coefficients[active] = 0.5 * fft_n * amplitudes * np.exp(1j * phases)
    coefficients[-1] = 0.0
    periodic_elevation = np.fft.irfft(coefficients, n=fft_n)
    wave_number = omega**2 / gravity_m_s2
    # A progressive component eta=a*cos(k*x-omega*t+theta) has
    # d(eta)/dx=-a*k*sin(k*x-omega*t+theta): a quadrature phase shift.
    slope_coefficients = -1j * coefficients * wave_number
    periodic_slope = np.fft.irfft(slope_coefficients, n=fft_n)
    elevation = np.concatenate((periodic_elevation, periodic_elevation[:1]))
    slope = np.concatenate((periodic_slope, periodic_slope[:1]))
    time_s = np.arange(requested_n, dtype=np.float64) * dt_s
    return SeaRealization(
        time_s=time_s,
        elevation_m=np.asarray(elevation[:requested_n], dtype=np.float64),
        slope_rad=np.asarray(slope[:requested_n], dtype=np.float64),
        frequencies_rad_s=active_omega,
        target_spectrum_m2_s=spectrum,
    )

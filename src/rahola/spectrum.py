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
) -> SeaRealization:
    """Synthesize eta and deep-water slope with deterministic amplitudes/random phases.

    The FFT grid is extended, when needed, so at least ``min_components`` positive
    frequency bins are represented; the requested prefix is returned.
    """
    requested_n = round(duration_s / dt_s) + 1
    fft_n = max(requested_n, 2 * (min_components + 1))
    if fft_n % 2:
        fft_n += 1
    frequencies_hz = np.fft.rfftfreq(fft_n, d=dt_s)
    omega = 2.0 * np.pi * frequencies_hz
    spectrum = jonswap_spectrum(omega, sea_state, gravity_m_s2)
    delta_omega = 2.0 * np.pi / (fft_n * dt_s)
    amplitudes = np.sqrt(2.0 * spectrum * delta_omega)
    rng = np.random.default_rng(np.uint64(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=omega.size)
    phases[0] = 0.0
    amplitudes[0] = 0.0
    amplitudes[-1] = 0.0
    coefficients = 0.5 * fft_n * amplitudes * np.exp(1j * phases)
    elevation = np.fft.irfft(coefficients, n=fft_n)
    wave_number = omega**2 / gravity_m_s2
    # A progressive component eta=a*cos(k*x-omega*t+theta) has
    # d(eta)/dx=-a*k*sin(k*x-omega*t+theta): a quadrature phase shift.
    slope_coefficients = -1j * coefficients * wave_number
    slope = np.fft.irfft(slope_coefficients, n=fft_n)
    time_s = np.arange(requested_n, dtype=np.float64) * dt_s
    return SeaRealization(
        time_s=time_s,
        elevation_m=np.asarray(elevation[:requested_n], dtype=np.float64),
        slope_rad=np.asarray(slope[:requested_n], dtype=np.float64),
        frequencies_rad_s=omega,
        target_spectrum_m2_s=spectrum,
    )

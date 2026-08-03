"""Analytic comparators and small deterministic validation experiments."""

from __future__ import annotations

import math

import jax
import numpy as np
from numpy.typing import NDArray

from rahola.dynamics import integrate_rk4_batch

FloatArray = NDArray[np.float64]


def linear_transfer_function(
    omega_rad_s: FloatArray, omega_n_rad_s: float, damping_ratio: float
) -> NDArray[np.complex128]:
    """Frequency response from angular acceleration moment to roll angle."""
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    denominator = omega_n_rad_s**2 - omega**2 + 2j * damping_ratio * omega_n_rad_s * omega
    return np.asarray(1.0 / denominator, dtype=np.complex128)


def damped_mathieu_threshold(damping_ratio: float) -> float:
    """First-order exact-tuning threshold h_crit = 4*zeta for h*cos(2*tau)."""
    if damping_ratio < 0:
        raise ValueError("damping_ratio must be nonnegative")
    return 4.0 * damping_ratio


def melnikov_heteroclinic_threshold(damping_ratio: float, frequency_ratio: float) -> float:
    """Harmonic amplitude threshold for x''+2*zeta*x'+x-x^3=f*cos(Omega*tau).

    This is the simple-zero condition for the heteroclinic Melnikov function,
    not a sufficient capsize criterion.
    """
    if damping_ratio < 0 or frequency_ratio <= 0:
        raise ValueError("damping must be nonnegative and frequency positive")
    delta = 2.0 * damping_ratio
    return float(
        2.0
        * delta
        * math.sinh(math.pi * frequency_ratio / math.sqrt(2.0))
        / (3.0 * math.pi * frequency_ratio)
    )


def numerical_melnikov_threshold(damping_ratio: float, frequency_ratio: float) -> float:
    """Numerically integrate the two terms along x_h=tanh(tau/sqrt(2))."""
    tau = np.linspace(-20.0, 20.0, 200_001)
    velocity = (1.0 / math.sqrt(2.0)) / np.cosh(tau / math.sqrt(2.0)) ** 2
    damping_integral = np.trapezoid(velocity**2, tau)
    forcing_integral = abs(np.trapezoid(velocity * np.cos(frequency_ratio * tau), tau))
    return float(2.0 * damping_ratio * damping_integral / forcing_integral)


def mathieu_growth_rate(
    h0: float,
    damping_ratio: float,
    *,
    excitation_ratio: float = 2.0,
    periods: float = 80.0,
    steps_per_period: int = 100,
) -> float:
    """Estimate the envelope exponent per nondimensional time for a linear Mathieu run."""
    dt_tau = 2.0 * math.pi / steps_per_period
    n_steps = round(periods * steps_per_period)
    tau_half = np.arange(2 * n_steps + 1, dtype=np.float64) * (0.5 * dt_tau)
    zeros = np.zeros((1, len(tau_half)), dtype=np.float64)
    modulation = h0 * np.cos(excitation_ratio * tau_half)[None, :]
    stiffness = np.ones_like(zeros)
    initial = np.array([[1e-7, 0.0]], dtype=np.float64)
    states, _ = integrate_rk4_batch(
        jax.device_put(zeros),
        jax.device_put(modulation),
        jax.device_put(stiffness),
        dt_tau,
        jax.device_put(initial),
        damping_ratio,
        0.0,
        0.0,
        0.0,
        1e12,
        1e12,
        family_code=1,
        linear_restoring=True,
    )
    values = np.asarray(states)[0]
    energy_amplitude = np.sqrt(values[:, 0] ** 2 + values[:, 1] ** 2)
    early = max(1, n_steps // 4)
    return float(
        (np.log(energy_amplitude[-1]) - np.log(energy_amplitude[early]))
        / ((n_steps - early) * dt_tau)
    )


def harmonic_capsize_fraction(
    amplitude: float,
    frequency_ratio: float,
    damping_ratio: float,
    *,
    phases: int = 32,
    periods: float = 120.0,
    steps_per_period: int = 100,
) -> float:
    """Capsize fraction over uniformly spaced harmonic forcing phases."""
    dt_tau = 2.0 * math.pi / steps_per_period
    n_steps = round(periods * steps_per_period)
    tau_half = np.arange(2 * n_steps + 1, dtype=np.float64) * (0.5 * dt_tau)
    phase_values = np.linspace(0.0, 2.0 * math.pi, phases, endpoint=False)
    forcing = amplitude * np.cos(frequency_ratio * tau_half[None, :] + phase_values[:, None])
    zeros = np.zeros_like(forcing)
    stiffness = np.ones_like(forcing)
    initial = np.zeros((phases, 2), dtype=np.float64)
    _, cap_steps = integrate_rk4_batch(
        jax.device_put(forcing),
        jax.device_put(zeros),
        jax.device_put(stiffness),
        dt_tau,
        jax.device_put(initial),
        damping_ratio,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        family_code=0,
        linear_restoring=False,
    )
    return float(np.mean(np.asarray(cap_steps) >= 0))


def find_harmonic_capsize_boundary(
    frequency_ratio: float,
    damping_ratio: float,
    *,
    phases: int = 24,
    target_fraction: float = 0.5,
    relative_tolerance: float = 0.04,
) -> float:
    """Bisection estimate of the forcing at a target phase-ensemble capsize fraction."""
    lower = melnikov_heteroclinic_threshold(damping_ratio, frequency_ratio)
    upper = max(0.1, 3.0 * lower)
    while (
        harmonic_capsize_fraction(upper, frequency_ratio, damping_ratio, phases=phases)
        < target_fraction
    ):
        upper *= 1.5
        if upper > 5:
            raise RuntimeError("could not bracket capsize boundary")
    while (upper - lower) / max(lower, 1e-12) > relative_tolerance:
        midpoint = 0.5 * (lower + upper)
        fraction = harmonic_capsize_fraction(
            midpoint, frequency_ratio, damping_ratio, phases=phases
        )
        if fraction >= target_fraction:
            upper = midpoint
        else:
            lower = midpoint
    return upper

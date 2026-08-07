"""Analytic comparators and small deterministic validation experiments."""

from __future__ import annotations

import math
from numbers import Integral

import jax
import numpy as np
from numpy.typing import NDArray

from rahola.dynamics import integrate_rk4_batch

FloatArray = NDArray[np.float64]


def _validate_damping_frequency(damping_ratio: float, frequency_ratio: float) -> None:
    if not math.isfinite(damping_ratio) or damping_ratio < 0:
        raise ValueError("damping_ratio must be finite and nonnegative")
    if not math.isfinite(frequency_ratio) or frequency_ratio <= 0:
        raise ValueError("frequency_ratio must be finite and positive")


def linear_transfer_function(
    omega_rad_s: FloatArray, omega_n_rad_s: float, damping_ratio: float
) -> NDArray[np.complex128]:
    """Frequency response from angular acceleration moment to roll angle."""
    omega = np.asarray(omega_rad_s, dtype=np.float64)
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega_rad_s must be finite")
    if not math.isfinite(omega_n_rad_s) or omega_n_rad_s <= 0:
        raise ValueError("omega_n_rad_s must be finite and positive")
    if not math.isfinite(damping_ratio) or damping_ratio < 0:
        raise ValueError("damping_ratio must be finite and nonnegative")
    denominator = omega_n_rad_s**2 - omega**2 + 2j * damping_ratio * omega_n_rad_s * omega
    return np.asarray(1.0 / denominator, dtype=np.complex128)


def damped_mathieu_threshold(damping_ratio: float) -> float:
    """First-order exact-tuning threshold h_crit = 4*zeta for h*cos(2*tau)."""
    if not math.isfinite(damping_ratio) or damping_ratio < 0:
        raise ValueError("damping_ratio must be finite and nonnegative")
    return 4.0 * damping_ratio


def melnikov_heteroclinic_threshold(damping_ratio: float, frequency_ratio: float) -> float:
    """Harmonic amplitude threshold for x''+2*zeta*x'+x-x^3=f*cos(Omega*tau).

    This is the simple-zero condition for the heteroclinic Melnikov function,
    not a sufficient capsize criterion. Finite frequencies whose result is not
    representable as an IEEE-754 float raise ``ValueError``.
    """
    _validate_damping_frequency(damping_ratio, frequency_ratio)
    if damping_ratio == 0:
        return 0.0
    delta = 2.0 * damping_ratio
    try:
        hyperbolic_sine = math.sinh(math.pi * frequency_ratio / math.sqrt(2.0))
    except OverflowError as error:
        raise ValueError(
            "Melnikov threshold is outside the representable floating-point domain"
        ) from error
    threshold = 2.0 * delta * hyperbolic_sine / (3.0 * math.pi * frequency_ratio)
    if not math.isfinite(threshold):
        raise ValueError("Melnikov threshold is outside the representable floating-point domain")
    return float(threshold)


def numerical_melnikov_threshold(damping_ratio: float, frequency_ratio: float) -> float:
    """Numerically integrate the two terms along x_h=tanh(tau/sqrt(2))."""
    _validate_damping_frequency(damping_ratio, frequency_ratio)
    tau = np.linspace(-20.0, 20.0, 200_001)
    velocity = (1.0 / math.sqrt(2.0)) / np.cosh(tau / math.sqrt(2.0)) ** 2
    damping_integral = np.trapezoid(velocity**2, tau)
    forcing_integral = abs(np.trapezoid(velocity * np.cos(frequency_ratio * tau), tau))
    if forcing_integral == 0:
        raise ValueError("numerical Melnikov forcing integral is zero")
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
    if not math.isfinite(h0):
        raise ValueError("h0 must be finite")
    if not math.isfinite(damping_ratio) or damping_ratio < 0:
        raise ValueError("damping_ratio must be finite and nonnegative")
    if not math.isfinite(excitation_ratio) or excitation_ratio <= 0:
        raise ValueError("excitation_ratio must be finite and positive")
    if not math.isfinite(periods) or periods <= 0:
        raise ValueError("periods must be finite and positive")
    if isinstance(steps_per_period, bool) or not isinstance(steps_per_period, Integral):
        raise ValueError("steps_per_period must be a positive integer")
    if steps_per_period < 1:
        raise ValueError("steps_per_period must be a positive integer")
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
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and nonnegative")
    _validate_damping_frequency(damping_ratio, frequency_ratio)
    if isinstance(phases, bool) or not isinstance(phases, Integral) or phases < 1:
        raise ValueError("phases must be a positive integer")
    if not math.isfinite(periods) or periods <= 0:
        raise ValueError("periods must be finite and positive")
    if isinstance(steps_per_period, bool) or not isinstance(steps_per_period, Integral):
        raise ValueError("steps_per_period must be a positive integer")
    if steps_per_period < 1:
        raise ValueError("steps_per_period must be a positive integer")
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
    """Bisection estimate of forcing at a target capsize fraction.

    The bracket and refinement loops are bounded. A valid but unresolvable
    tolerance raises ``RuntimeError`` rather than looping indefinitely.
    """
    _validate_damping_frequency(damping_ratio, frequency_ratio)
    if isinstance(phases, bool) or not isinstance(phases, Integral) or phases < 1:
        raise ValueError("phases must be a positive integer")
    if not math.isfinite(target_fraction) or not 0 < target_fraction < 1:
        raise ValueError("target_fraction must be finite and strictly between zero and one")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be finite and strictly between zero and one")
    melnikov = melnikov_heteroclinic_threshold(damping_ratio, frequency_ratio)
    lower = 0.5 * melnikov
    if (
        harmonic_capsize_fraction(lower, frequency_ratio, damping_ratio, phases=phases)
        >= target_fraction
    ):
        raise RuntimeError("lower bracket already meets the target capsize fraction")
    upper = max(0.1, 3.0 * melnikov)
    for _ in range(32):
        if (
            harmonic_capsize_fraction(upper, frequency_ratio, damping_ratio, phases=phases)
            >= target_fraction
        ):
            break
        upper *= 1.5
        if upper > 5:
            raise RuntimeError("could not bracket capsize boundary")
    else:
        raise RuntimeError("capsize boundary bracket exceeded iteration bound")
    for _ in range(256):
        if (upper - lower) / max(lower, 1e-12) <= relative_tolerance:
            return upper
        midpoint = 0.5 * (lower + upper)
        if midpoint in (lower, upper):
            raise RuntimeError("capsize boundary refinement stagnated")
        fraction = harmonic_capsize_fraction(
            midpoint, frequency_ratio, damping_ratio, phases=phases
        )
        if fraction >= target_fraction:
            upper = midpoint
        else:
            lower = midpoint
    raise RuntimeError("capsize boundary refinement exceeded iteration bound")

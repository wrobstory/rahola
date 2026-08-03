"""Bootstrap filter for stiffness and stiffness drift from causal roll motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rahola.config import Family, SimulationConfig
from rahola_lab.constants import PF_PARTICLES


@dataclass(frozen=True)
class ParticlePosterior:
    """Equally weighted physical-state particles at the end of a causal window."""

    roll_rad: NDArray[np.float64]
    rate_rad_s: NDArray[np.float64]
    stiffness_multiplier: NDArray[np.float64]
    stiffness_rate_per_s: NDArray[np.float64]


def _systematic_resample(
    weights: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.int64]:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.searchsorted(np.cumsum(weights), positions).astype(np.int64)


def bootstrap_particle_filter(
    angle_rad: NDArray[np.floating],
    rate_rad_s: NDArray[np.floating],
    dt_s: float,
    config: SimulationConfig,
    *,
    particle_count: int = PF_PARTICLES,
    seed: int = 0,
    absolute_start_s: float = 0.0,
) -> ParticlePosterior:
    """Infer current stiffness and linear drift from only past roll and roll rate.

    Roll and roll-rate observations are effectively noise-free in the synthetic
    sensor, so they are Rao--Blackwellized rather than needlessly propagated.
    The bootstrap state is current stiffness and drift; unknown wave forcing is
    represented by a robust Gaussian innovation likelihood.
    """
    angle = np.asarray(angle_rad, dtype=np.float64)
    rate = np.asarray(rate_rad_s, dtype=np.float64)
    if (
        angle.ndim != 1
        or rate.shape != angle.shape
        or len(angle) < 8
        or not np.all(np.isfinite(angle))
        or not np.all(np.isfinite(rate))
        or dt_s <= 0.0
        or particle_count < 100
    ):
        raise ValueError("PF inputs must be finite causal records with at least 8 samples")
    rng = np.random.default_rng(seed)
    if config.protocol.ramp_parameter == "stiffness":
        stiffness = rng.uniform(0.05, 1.4, particle_count)
        drift = rng.uniform(-0.003, 0.0005, particle_count)
    else:
        stiffness = np.clip(rng.normal(1.0, 0.12, particle_count), 0.3, 1.5)
        drift = rng.normal(0.0, 0.00015, particle_count)
    weights = np.full(particle_count, 1.0 / particle_count)
    omega = config.omega_n_rad_s
    escape = config.escape_angle_rad
    x = angle / escape
    velocity = rate / (escape * omega)
    acceleration = np.gradient(rate, dt_s) / (escape * omega**2)
    # Two-second assimilation is enough to resolve slow stiffness without
    # pretending the band-limited encounter innovations are independent.
    observation_stride = max(1, round(2.0 / dt_s))
    last_index = 0
    for index in range(observation_stride, len(angle), observation_stride):
        elapsed = (index - last_index) * dt_s
        stiffness += drift * elapsed + rng.normal(0.0, 0.002 * np.sqrt(elapsed), particle_count)
        drift += rng.normal(0.0, 1.5e-5 * np.sqrt(elapsed), particle_count)
        stiffness = np.clip(stiffness, -0.4, 1.6)
        drift = np.clip(drift, -0.004, 0.001)
        shape = (
            x[index]
            if config.linear_restoring
            else (x[index] - x[index] ** 3 + config.quintic_coefficient * x[index] ** 5)
        )
        modulation = 1.0
        if config.family == Family.PARAMETRIC:
            time_s = absolute_start_s + index * dt_s
            modulation += config.parametric.h0 * np.cos(
                config.parametric.excitation_ratio * omega * time_s
            )
        expected = (
            (config.bias_moment if config.family == Family.BIASED else 0.0)
            - 2.0 * config.damping_ratio * velocity[index]
            - config.quadratic_damping * velocity[index] * abs(velocity[index])
            - stiffness * modulation * shape
        )
        innovation = acceleration[index] - expected
        scale = 0.10 + 0.20 * abs(velocity[index])
        log_weight = -0.5 * (innovation / scale) ** 2
        log_weight -= np.max(log_weight)
        weights *= np.exp(log_weight)
        total = float(np.sum(weights))
        weights = (
            weights / total
            if np.isfinite(total) and total > 1e-300
            else np.full(particle_count, 1.0 / particle_count)
        )
        if 1.0 / np.sum(weights**2) < 0.55 * particle_count:
            chosen = _systematic_resample(weights, rng)
            stiffness = stiffness[chosen]
            drift = drift[chosen]
            weights.fill(1.0 / particle_count)
        last_index = index
    chosen = rng.choice(particle_count, size=particle_count, replace=True, p=weights)
    return ParticlePosterior(
        roll_rad=np.full(particle_count, angle[-1]),
        rate_rad_s=np.full(particle_count, rate[-1]),
        stiffness_multiplier=stiffness[chosen],
        stiffness_rate_per_s=drift[chosen],
    )

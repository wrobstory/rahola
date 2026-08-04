"""Public batch simulation orchestration."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import replace
from numbers import Integral

import jax
import numpy as np

from rahola import __version__
from rahola.config import (
    Family,
    ParametricMode,
    ProtocolConfig,
    ProtocolKind,
    SeaState,
    SimulationConfig,
)
from rahola.dataset import SimulationDataset
from rahola.dynamics import integrate_rk4_batch
from rahola.spectrum import synthesize_jonswap

_FAMILY_CODES = {Family.SOFTENING: 0, Family.PARAMETRIC: 1, Family.BIASED: 2}


def _derived_seed(seed: int, segment: int, channel: int) -> int:
    state = np.random.SeedSequence([int(seed), segment, channel]).generate_state(1, dtype=np.uint64)
    return int(state[0])


def _sea_segments(config: SimulationConfig, n_half_steps: int) -> list[tuple[int, int, SeaState]]:
    if config.protocol.kind != ProtocolKind.STEP:
        return [(0, n_half_steps, config.forcing.sea_state)]
    dt_half = 0.5 * config.integration_dt_s
    transitions = [
        (min(n_half_steps, max(0, round(step.time_s / dt_half))), step.sea_state)
        for step in config.protocol.steps
    ]
    segments: list[tuple[int, int, SeaState]] = []
    start = 0
    sea_state = config.forcing.sea_state
    for boundary, next_state in transitions:
        if boundary > start:
            segments.append((start, boundary, sea_state))
        start, sea_state = boundary, next_state
    if start < n_half_steps:
        segments.append((start, n_half_steps, sea_state))
    return segments


def _forcing_for_seed(
    config: SimulationConfig, seed: int, n_half_steps: int, channel: int
) -> tuple[np.ndarray, np.ndarray]:
    dt_half = 0.5 * config.integration_dt_s
    slope = np.zeros(n_half_steps + 1, dtype=np.float64)
    elevation = np.zeros_like(slope)
    for segment_index, (start, end, sea_state) in enumerate(_sea_segments(config, n_half_steps)):
        realization = synthesize_jonswap(
            sea_state,
            duration_s=(end - start) * dt_half,
            dt_s=dt_half,
            seed=_derived_seed(seed, segment_index, channel),
            min_components=config.forcing.min_components,
            gravity_m_s2=config.forcing.gravity_m_s2,
            max_frequency_rad_s=(
                None
                if config.forcing.max_frequency_ratio is None
                else config.forcing.max_frequency_ratio * config.omega_n_rad_s
            ),
        )
        slope[start : end + 1] = realization.slope_rad
        elevation[start : end + 1] = realization.elevation_m
    return slope, elevation


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"


def _trajectory_values(value: float | np.ndarray, size: int, *, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        if not np.isfinite(values):
            raise ValueError(f"{name} must be finite scalar or one value per seed")
        return np.full(size, float(values), dtype=np.float64)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite scalar or one value per seed")
    return values


def _validated_seeds(seeds: Iterable[int]) -> np.ndarray:
    values = list(seeds)
    maximum = np.iinfo(np.uint64).max
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
        or value > maximum
        for value in values
    ):
        raise ValueError("seeds must be a non-empty iterable of uint64 integers")
    return np.asarray(values, dtype=np.uint64)


def _simulate_batch(
    config: SimulationConfig,
    seeds: Iterable[int],
    *,
    initial_angle_rad: float | np.ndarray | None = None,
    initial_rate_rad_s: float | np.ndarray | None = None,
    stiffness_multiplier: float | np.ndarray | None = None,
    stiffness_rate_per_s: float | np.ndarray | None = None,
    time_offset_s: float | np.ndarray | None = None,
) -> SimulationDataset:
    seed_array = _validated_seeds(seeds)
    if seed_array.ndim != 1:
        raise ValueError("seeds must be a non-empty one-dimensional iterable")
    if len(np.unique(seed_array)) != len(seed_array):
        raise ValueError("seeds must be unique within a batch")

    dt_s = config.integration_dt_s
    n_steps = round(config.duration_s / dt_s)
    n_half_steps = 2 * n_steps
    time_half_s = np.arange(n_half_steps + 1, dtype=np.float64) * (0.5 * dt_s)
    time_offsets = _trajectory_values(
        0.0 if time_offset_s is None else time_offset_s,
        seed_array.size,
        name="time_offset_s",
    )
    forcing_rows: list[np.ndarray] = []
    modulation_rows: list[np.ndarray] = []
    for trajectory_index, seed_value in enumerate(seed_array):
        slope, _ = _forcing_for_seed(config, int(seed_value), n_half_steps, channel=0)
        forcing = config.forcing.effective_wave_slope * slope / config.escape_angle_rad
        if config.protocol.kind == ProtocolKind.RAMPED:
            ramp = np.linspace(
                float(config.protocol.ramp_start),
                float(config.protocol.ramp_end),
                n_half_steps + 1,
            )
            if config.protocol.ramp_parameter == "forcing_scale":
                forcing *= ramp
        forcing_rows.append(forcing)

        if config.family == Family.PARAMETRIC:
            if config.parametric.mode == ParametricMode.DETERMINISTIC:
                tau = config.omega_n_rad_s * (time_half_s + time_offsets[trajectory_index])
                modulation = config.parametric.h0 * np.cos(config.parametric.excitation_ratio * tau)
            else:
                _, independent_eta = _forcing_for_seed(
                    config, int(seed_value), n_half_steps, channel=1
                )
                rms = np.sqrt(np.mean(independent_eta**2))
                modulation = config.parametric.stochastic_std * independent_eta / max(rms, 1e-15)
        else:
            modulation = np.zeros(n_half_steps + 1, dtype=np.float64)
        modulation_rows.append(modulation)

    forcing_half = np.stack(forcing_rows)
    modulation_half = np.stack(modulation_rows)
    restarted = stiffness_multiplier is not None or stiffness_rate_per_s is not None
    if restarted:
        starts = _trajectory_values(
            1.0 if stiffness_multiplier is None else stiffness_multiplier,
            seed_array.size,
            name="stiffness_multiplier",
        )
        rates = _trajectory_values(
            0.0 if stiffness_rate_per_s is None else stiffness_rate_per_s,
            seed_array.size,
            name="stiffness_rate_per_s",
        )
        stiffness_half = starts[:, None] + rates[:, None] * time_half_s[None, :]
    else:
        if (
            config.protocol.kind == ProtocolKind.RAMPED
            and config.protocol.ramp_parameter == "stiffness"
        ):
            stiffness_1d = np.linspace(
                float(config.protocol.ramp_start),
                float(config.protocol.ramp_end),
                n_half_steps + 1,
            )
        else:
            stiffness_1d = np.ones(n_half_steps + 1, dtype=np.float64)
        stiffness_half = np.broadcast_to(stiffness_1d, forcing_half.shape).copy()

    initial_angles = _trajectory_values(
        config.initial_angle_rad if initial_angle_rad is None else initial_angle_rad,
        seed_array.size,
        name="initial_angle_rad",
    )
    initial_rates = _trajectory_values(
        config.initial_rate_rad_s if initial_rate_rad_s is None else initial_rate_rad_s,
        seed_array.size,
        name="initial_rate_rad_s",
    )
    initial_state = np.column_stack(
        (
            initial_angles / config.escape_angle_rad,
            initial_rates / (config.escape_angle_rad * config.omega_n_rad_s),
        )
    )
    states, cap_steps = integrate_rk4_batch(
        jax.device_put(forcing_half),
        jax.device_put(modulation_half),
        jax.device_put(stiffness_half),
        config.omega_n_rad_s * dt_s,
        jax.device_put(initial_state),
        config.damping_ratio,
        config.quadratic_damping,
        config.bias_moment,
        config.quintic_coefficient,
        1.0,
        config.negative_escape_rad / config.escape_angle_rad,
        family_code=_FAMILY_CODES[config.family],
        linear_restoring=config.linear_restoring,
    )
    state_array = np.asarray(states)
    cap_step_array = np.asarray(cap_steps, dtype=np.int32)

    output_stride = round((1.0 / config.output_rate_hz) / dt_s)
    output_indices = np.arange(0, n_steps + 1, output_stride, dtype=np.int64)
    output_time = output_indices.astype(np.float64) * dt_s
    angle = state_array[:, output_indices, 0] * config.escape_angle_rad
    rate = state_array[:, output_indices, 1] * config.escape_angle_rad * config.omega_n_rad_s
    capsized = cap_step_array >= 0
    t_capsize = np.where(capsized, cap_step_array * dt_s, np.nan)
    for row, cap_time in enumerate(t_capsize):
        if np.isfinite(cap_time):
            post_event = output_time > cap_time
            angle[row, post_event] = np.nan
            rate[row, post_event] = np.nan

    commit = _git_commit()
    metadata = tuple(
        {
            "family": str(config.family),
            "parameters": config.to_dict(),
            "protocol": str(config.protocol.kind),
            "seed": int(seed),
            "capsized": bool(capsized[index]),
            "t_capsize_s": None if not capsized[index] else float(t_capsize[index]),
            "config_hash": config.config_hash,
            "package_version": __version__,
            "git_commit": commit,
            "restart": None
            if not restarted
            else {
                "initial_angle_rad": float(initial_angles[index]),
                "initial_rate_rad_s": float(initial_rates[index]),
                "stiffness_multiplier": float(stiffness_half[index, 0]),
                "stiffness_rate_per_s": float(
                    (stiffness_half[index, -1] - stiffness_half[index, 0]) / config.duration_s
                ),
            },
        }
        for index, seed in enumerate(seed_array)
    )
    return SimulationDataset(
        time_s=output_time,
        angle_rad=angle,
        rate_rad_s=rate,
        seeds=seed_array,
        capsized=capsized,
        t_capsize_s=t_capsize,
        metadata=metadata,
        config=config.to_dict(),
    )


def simulate_batch(config: SimulationConfig, seeds: Iterable[int]) -> SimulationDataset:
    """Simulate seeded trajectories with FFT forcing and a vmapped JAX RK4 kernel."""
    return _simulate_batch(config, seeds)


def simulate_restarted_batch(
    config: SimulationConfig,
    seeds: Iterable[int],
    *,
    duration_s: float,
    initial_angle_rad: float | np.ndarray,
    initial_rate_rad_s: float | np.ndarray,
    stiffness_multiplier: float | np.ndarray = 1.0,
    stiffness_rate_per_s: float | np.ndarray = 0.0,
    time_offset_s: float | np.ndarray = 0.0,
) -> SimulationDataset:
    """Restart independent futures from arbitrary state and stiffness drift.

    ``seeds`` define fresh forcing realizations. The initial state, current
    stiffness multiplier, and continuing linear stiffness rate are supplied
    explicitly by the restart comparison and may vary by trajectory, which
    keeps heterogeneous restart ensembles batchable. The base configuration
    must be stationary; a time-varying protocol cannot be silently continued
    from an unspecified phase.
    """
    if duration_s <= 0.0:
        raise ValueError("restart duration must be positive")
    if config.protocol.kind != ProtocolKind.STATIONARY:
        raise ValueError(
            "restart simulation requires a stationary base protocol; supply current "
            "stiffness and drift explicitly"
        )
    restart_config = replace(
        config,
        duration_s=duration_s,
        protocol=ProtocolConfig(kind=ProtocolKind.STATIONARY),
        initial_angle_rad=0.0,
        initial_rate_rad_s=0.0,
    )
    return _simulate_batch(
        restart_config,
        seeds,
        initial_angle_rad=initial_angle_rad,
        initial_rate_rad_s=initial_rate_rad_s,
        stiffness_multiplier=stiffness_multiplier,
        stiffness_rate_per_s=stiffness_rate_per_s,
        time_offset_s=time_offset_s,
    )


def with_steps_per_period(config: SimulationConfig, steps: int) -> SimulationConfig:
    """Return a convergence-study variant without mutating the original config."""
    return replace(config, integration_steps_per_period=steps)

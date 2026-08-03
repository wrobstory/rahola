from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rahola.config import (
    Family,
    ForcingConfig,
    ProtocolConfig,
    ProtocolKind,
    SeaState,
    SeaStateStep,
    SimulationConfig,
)
from rahola.simulate import simulate_batch, simulate_restarted_batch


def _small_config(**changes: object) -> SimulationConfig:
    base = SimulationConfig(
        duration_s=32.0,
        natural_period_s=4.0,
        output_rate_hz=2.0,
        forcing=ForcingConfig(sea_state=SeaState(hs_m=1.5, tp_s=5.0), effective_wave_slope=0.03),
    )
    return replace(base, **changes)


def test_batch_determinism_and_seed_separation() -> None:
    config = _small_config()
    first = simulate_batch(config, [10, 11])
    second = simulate_batch(config, [10, 11])
    assert np.array_equal(first.angle_rad, second.angle_rad, equal_nan=True)
    assert not np.array_equal(first.angle_rad[0], first.angle_rad[1])


def test_step_protocol_and_asymmetric_capsize() -> None:
    protocol = ProtocolConfig(
        kind=ProtocolKind.STEP,
        steps=(SeaStateStep(16.0, SeaState(hs_m=3.0, tp_s=4.0)),),
    )
    config = _small_config(
        family=Family.BIASED,
        bias_moment=-0.5,
        negative_escape_angle_rad=0.2,
        protocol=protocol,
    )
    dataset = simulate_batch(config, [3])
    assert dataset.metadata[0]["protocol"] == "step"
    assert dataset.capsized[0]
    cap_index = np.searchsorted(dataset.time_s, dataset.t_capsize_s[0], side="right")
    assert np.all(np.isnan(dataset.angle_rad[0, cap_index:]))


def test_restart_accepts_per_trajectory_state_and_stiffness_drift() -> None:
    config = _small_config()
    angles = np.array([0.01, -0.02])
    rates = np.array([0.003, -0.004])
    restarted = simulate_restarted_batch(
        config,
        [20, 21],
        duration_s=16.0,
        initial_angle_rad=angles,
        initial_rate_rad_s=rates,
        stiffness_multiplier=np.array([0.9, 0.8]),
        stiffness_rate_per_s=np.array([-0.001, -0.002]),
    )
    assert restarted.angle_rad[:, 0] == pytest.approx(angles)
    assert restarted.rate_rad_s[:, 0] == pytest.approx(rates)
    assert restarted.metadata[1]["restart"]["stiffness_multiplier"] == pytest.approx(0.8)
    assert restarted.metadata[1]["restart"]["stiffness_rate_per_s"] == pytest.approx(-0.002)


@pytest.mark.slow
def test_restart_ensemble_matches_full_run_segment_statistics() -> None:
    """Predeclared check: variance within 15%, capsize fraction within 5 points."""
    config = _small_config(duration_s=256.0)
    source = simulate_batch(config, range(1_000, 1_128))
    midpoint = int(np.searchsorted(source.time_s, 128.0))
    restarted = simulate_restarted_batch(
        config,
        range(2_000, 2_128),
        duration_s=128.0,
        initial_angle_rad=source.angle_rad[:, midpoint],
        initial_rate_rad_s=source.rate_rad_s[:, midpoint],
    )
    reference = simulate_batch(config, range(3_000, 3_128))
    restart_start = int(np.searchsorted(restarted.time_s, 32.0))
    reference_start = int(np.searchsorted(reference.time_s, 160.0))
    restart_variance = np.nanmean(np.nanvar(restarted.angle_rad[:, restart_start:], axis=1))
    reference_variance = np.nanmean(np.nanvar(reference.angle_rad[:, reference_start:], axis=1))
    assert restart_variance == pytest.approx(reference_variance, rel=0.15)
    assert abs(np.mean(restarted.capsized) - np.mean(reference.capsized)) <= 0.05


@pytest.mark.slow
def test_step_halving_convergence_statistics() -> None:
    coarse = _small_config(duration_s=256.0, integration_steps_per_period=40, output_rate_hz=1)
    fine = replace(coarse, integration_steps_per_period=80)
    seeds = range(32)
    coarse_data = simulate_batch(coarse, seeds)
    fine_data = simulate_batch(fine, seeds)
    coarse_variance = np.nanmean(np.nanvar(coarse_data.angle_rad, axis=1))
    fine_variance = np.nanmean(np.nanvar(fine_data.angle_rad, axis=1))
    assert coarse_variance == pytest.approx(fine_variance, rel=0.03)
    assert abs(np.mean(coarse_data.capsized) - np.mean(fine_data.capsized)) <= 0.05

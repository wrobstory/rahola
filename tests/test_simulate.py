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
from rahola.simulate import simulate_batch


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

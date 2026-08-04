from __future__ import annotations

import math

import numpy as np
import pytest

from rahola.config import (
    Family,
    ForcingConfig,
    ParametricConfig,
    ProtocolConfig,
    ProtocolKind,
    SeaState,
    SimulationConfig,
)


def test_nondimensional_boundary_conventions() -> None:
    config = SimulationConfig(natural_period_s=8.0, escape_angle_rad=math.radians(30))
    assert config.omega_n_rad_s == pytest.approx(2 * math.pi / 8)
    assert config.integration_dt_s <= config.natural_period_s / 40


def test_config_hash_is_stable_and_sensitive() -> None:
    first = SimulationConfig()
    second = SimulationConfig()
    changed = SimulationConfig(family=Family.BIASED, bias_moment=0.1)
    assert first.config_hash == second.config_hash
    assert first.config_hash != changed.config_hash


def test_ramp_requires_supported_parameter() -> None:
    with pytest.raises(ValueError, match="ramp_parameter"):
        ProtocolConfig(
            kind=ProtocolKind.RAMPED,
            ramp_parameter="future_data",
            ramp_start=0.0,
            ramp_end=1.0,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SeaState(hs_m=float("nan")),
        lambda: ParametricConfig(h0=float("nan")),
        lambda: ForcingConfig(max_frequency_ratio=float("nan")),
        lambda: SimulationConfig(damping_ratio=float("nan")),
    ],
)
def test_config_rejects_non_finite_values(factory) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory()


def test_duration_must_align_with_output_grid() -> None:
    with pytest.raises(ValueError, match="output intervals"):
        SimulationConfig(duration_s=1.1, output_rate_hz=3.0)


def test_forcing_cutoff_ratio_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        ForcingConfig(max_frequency_ratio=0.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ForcingConfig(min_components=float("nan")),
        lambda: ForcingConfig(min_components=200.5),
        lambda: SimulationConfig(integration_steps_per_period=float("nan")),
        lambda: SimulationConfig(integration_steps_per_period=40.5),
    ],
)
def test_integer_controls_reject_non_integral_values(factory) -> None:
    with pytest.raises(ValueError, match="integer"):
        factory()


def test_dormant_protocol_values_must_still_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        ProtocolConfig(kind=ProtocolKind.STATIONARY, ramp_start=float("nan"))


def test_numpy_integer_controls_have_stable_hashes() -> None:
    config = SimulationConfig(integration_steps_per_period=np.int64(40))
    assert config.config_hash == SimulationConfig().config_hash

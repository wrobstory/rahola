from __future__ import annotations

import math

import pytest

from rahola.config import Family, ProtocolConfig, ProtocolKind, SimulationConfig


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
